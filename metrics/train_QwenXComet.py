"""
Training script for QwenXComet, a learned MT-evaluation metric
with a Qwen3-Embedding backbone. Trains on a databin produced by
binarize_data_qwen.py (use the same --model_id for both).

To start from a trained model (e.g. stage 2 starting from stage 1), pass
--init_from <stage-1 output_dir>; the pretrained backbone is then not downloaded.

Example usage (from the metrics/ folder):

python train_QwenXComet.py \
    --model_id Qwen/Qwen3-Embedding-4B \
    --saved_ds_path databins/Qwen_Qwen3-Embedding-4B/stage1 \
    --output_dir models_trained/QwenXComet/stage1 \
    --primary_eval_dataset dev_paraf \
    --batch_size 4 --eval_batch_size 4 \
    --eval_steps 500 --save_steps 500 \
    --num_train_epochs 3 --lr_scheduler_type cosine --learning_rate 1e-5 \
    --warmup_steps 50 --frozen_epochs 0
"""

import argparse
import os
import time
import torch
import datasets
import numpy as np
import sys
import json
import inspect
from pathlib import Path
import shutil
import glob

from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
    EarlyStoppingCallback,
)




# local imports
from modules.QwenXComet import QwenXComet, QwenXCometConfig, QwenXCometCollator
from utils.train_utils import print_trainable_parameters, save_training_metadata

# Captured as early as possible so --time_limit_seconds (the wall-clock budget
# handed to this process, e.g. remaining SLURM time) is measured against
# actual process runtime, not just time spent inside trainer.train().
_PROCESS_START_TIME = time.time()

print("datasets version:", datasets.__version__)

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Backbone freeze / unfreeze callback
# ---------------------------------------------------------------------------

class BackboneFreezeCallback(TrainerCallback):
    """
    Freezes the backbone for epoch 1, unfreezes from epoch 2 onwards.
    Follows the COMET-20 training protocol (Rei et al., 2020).
    """
    def __init__(self, frozen_epochs=1):
        self.frozen_epochs = frozen_epochs

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._set_backbone_grad(model, requires_grad=False)
        print("[INFO] on_train_begin")
        print_trainable_parameters(model)

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        # state.epoch is 0-indexed float; epoch 2 starts at state.epoch >= 1.0
        if state.epoch >= self.frozen_epochs:
            self._set_backbone_grad(model, requires_grad=True)
            print(f"[INFO] Epoch {int(state.epoch) + 1}: backbone unfrozen. Full end-to-end training.")
            print_trainable_parameters(model)



    @staticmethod
    def _set_backbone_grad(model, requires_grad: bool):
        # Use named_parameters() to filter backbone params by name prefix.
        count = 0
        for name, param in model.named_parameters():
            if "backbone." in name:
                param.requires_grad = requires_grad
                count += 1
        print(f"[INFO] _set_backbone_grad: set requires_grad={requires_grad} on {count} backbone parameters.")


class TimeLimitCallback(TrainerCallback):
    """Forces a checkpoint save and a clean stop once wall-clock time
    approaches a deadline (typically the SLURM job's time limit), instead of
    letting whatever training happened since the last save_steps-triggered
    checkpoint be discarded when Slurm SIGTERMs the job on TIMEOUT.

    Polling elapsed wall-clock time from inside the (already synchronous,
    per-rank) training loop is used instead of an OS signal: every rank
    reaches the same `deadline` at the same step deterministically, so no
    cross-node signal delivery/propagation through the torch-elastic launcher
    is needed.
    """
    def __init__(self, deadline):
        self.deadline = deadline
        self.triggered = False

    def on_step_end(self, args, state, control, **kwargs):
        if self.triggered:
            return control
        if time.time() >= self.deadline:
            self.triggered = True
            print(f"[INFO] TimeLimitCallback: wall-clock deadline reached at "
                  f"step {state.global_step} — forcing a checkpoint save and stopping.")
            control.should_save = True
            control.should_training_stop = True
        return control


# ---------------------------------------------------------------------------
# Discriminative learning rate (backbone vs. regression head)
# ---------------------------------------------------------------------------

class DiscriminativeLRTrainer(Trainer):
    """Trainer that gives the Qwen backbone its own (typically lower) learning
    rate, separate from the regression head (ff1/ff2/final, plus
    layerwise_attention when enabled).

    Motivation: when fine-tuning on a small paragraph-level set (stage 2), the
    backbone can lose its sentence-level performance and overfit the paragraph
    data. A lower backbone LR protects the pretrained / stage-1 features while
    the head keeps adapting to paragraph-length inputs.

    Only create_optimizer is overridden: 4 parameter groups (backbone/head x
    decay/no-decay), backbone groups at `backbone_lr`, head groups at
    args.learning_rate. The scheduler is untouched — HF's warmup+cosine scales
    every group proportionally from its own base lr, so both groups warm up and
    anneal together. DeepSpeed wiring is unchanged (HF builds the optimizer,
    then hands it to deepspeed.initialize; a multi-group optimizer is fine).
    backbone_lr == learning_rate reproduces the stock single-group optimizer.
    """

    def __init__(self, *targs, backbone_lr, **tkwargs):
        super().__init__(*targs, **tkwargs)
        self.backbone_lr = float(backbone_lr)

    @staticmethod
    def _is_backbone(name: str) -> bool:
        return name.startswith("backbone.")

    def create_optimizer(self):
        if self.optimizer is None:
            opt_model = self.model
            decay = set(self.get_decay_parameter_names(opt_model))
            head_lr = self.args.learning_rate
            named = [(n, p) for n, p in opt_model.named_parameters() if p.requires_grad]

            groups = [
                {"params": [p for n, p in named if self._is_backbone(n) and n in decay],
                 "lr": self.backbone_lr, "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in named if self._is_backbone(n) and n not in decay],
                 "lr": self.backbone_lr, "weight_decay": 0.0},
                {"params": [p for n, p in named if not self._is_backbone(n) and n in decay],
                 "lr": head_lr, "weight_decay": self.args.weight_decay},
                {"params": [p for n, p in named if not self._is_backbone(n) and n not in decay],
                 "lr": head_lr, "weight_decay": 0.0},
            ]
            n_bb = sum(len(g["params"]) for g in groups[:2])
            n_head = sum(len(g["params"]) for g in groups[2:])
            print(f"[INFO] DiscriminativeLRTrainer: backbone_lr={self.backbone_lr:g} on "
                  f"{n_bb} tensors | head_lr={head_lr:g} on {n_head} tensors")
            groups = [g for g in groups if g["params"]]

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
            optimizer_kwargs.pop("lr", None)   # per-group lr set above
            self.optimizer = optimizer_cls(groups, **optimizer_kwargs)

        return self.optimizer


# ---------------------------------------------------------------------------
# Build model & tokenizer
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    collator = QwenXCometCollator(tokenizer)

    
    config = QwenXCometConfig(
        model_id=str(args.model_id),
        inference_ref_weight=args.inference_ref_weight,
        use_layerwise_attention=args.use_layerwise_attention,
        layer_transformation=args.layer_transformation
    )
    # pretrained backbone weights are only needed when not initialising from a trained checkpoint
    model = QwenXComet(config, load_backbone_weights=not args.init_from)

    return model, tokenizer, collator

# ---------------------------------------------------------------------------
# Train resume
# ---------------------------------------------------------------------------

def get_last_valid_checkpoint(output_dir):
    """Like get_last_checkpoint, but skips (and deletes) checkpoints left
    incomplete by a crash mid-save (e.g. job hit the time limit while writing)."""
    ckpts = sorted(
        (d for d in glob.glob(os.path.join(output_dir, "checkpoint-*"))
         if os.path.isdir(d) and d.rsplit("-", 1)[-1].isdigit()),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
        reverse=True,
    )
    print('Existing ckpts:', ckpts)
    for ckpt in ckpts:
        required = ["trainer_state.json", "config.json"]
        has_required = all(os.path.isfile(os.path.join(ckpt, f)) for f in required)
        # DeepSpeed ZeRO-2: sharded state under global_step*, referenced by "latest"
        latest_file = os.path.join(ckpt, "latest")
        ds_ok = True
        if os.path.isfile(latest_file):
            with open(latest_file) as f:
                tag = f.read().strip()
            ds_ok = os.path.isdir(os.path.join(ckpt, tag))
        has_weights = os.path.isfile(os.path.join(ckpt, "model.safetensors")) or os.path.isfile(latest_file)

        if has_required and has_weights and ds_ok:
            return ckpt
        else:
            print(f"[WARN] Incomplete checkpoint {ckpt}, deleting.")
            shutil.rmtree(ckpt, ignore_errors=True)
    return None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args, train_args):
    print(f"[INFO] Building model and tokenizer for model_id: {args.model_id}")
    model, tokenizer, data_collator = build_model_and_tokenizer(args)
    
    print("--- Model statistics (right after load) ---")
    print_trainable_parameters(model)

    if args.init_from:
        print("[INFO] Initializing model from checkpoint:", args.init_from)
        ckpt = Path(args.init_from)
        sf = ckpt / "model.safetensors"
        sf_index = ckpt / "model.safetensors.index.json"
        bin_ = ckpt / "pytorch_model.bin"

        if sf.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(sf))
        elif sf_index.exists():
            from safetensors.torch import load_file
            with open(sf_index) as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            state_dict = {}
            for shard_file in shard_files:
                state_dict.update(load_file(str(ckpt / shard_file)))
            
        else:
            state_dict = torch.load(str(bin_), map_location="cpu")
    
        import deepspeed
        load_result = {"missing": [], "unexpected": list(state_dict.keys())}
        own_state = dict(model.named_parameters())
        for name, param in own_state.items():
            if name not in state_dict:
                load_result["missing"].append(name)
                continue
            load_result["unexpected"].remove(name)
            with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    param.data.copy_(state_dict[name])
    
        missing, unexpected = load_result["missing"], load_result["unexpected"]
        print(f"[INFO] init_from {ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
        if missing or unexpected:
            print(f"  missing={missing}\n  unexpected={unexpected}")
        assert not missing, f"--init_from checkpoint lacks {len(missing)} parameters (they would stay random)."
    
        print('Parameter check after the if args.init_from block:')
        print('An example parameter (if all 0s - probably randomly generated)')
        with deepspeed.zero.GatheredParameters(dict(model.named_parameters())['backbone.layers.2.self_attn.q_proj.weight'], modifier_rank=0):
            print(dict(model.named_parameters())['backbone.layers.2.self_attn.q_proj.weight'])


        

    last_checkpoint = None
    if os.path.isdir(train_args.output_dir):
        last_checkpoint = get_last_valid_checkpoint(train_args.output_dir)

    if last_checkpoint is None:
        print("[INFO] No last checkpoint found.")
    else:
        print(f"[INFO] Resuming from {last_checkpoint}: Trainer will overwrite backbone "
          f"from the checkpoint via resume_from_checkpoint.")
        if args.init_from:
            print(f"[WARNING] --init_from={args.init_from} was set, but a checkpoint exists in "
              f"{train_args.output_dir} — resume_from_checkpoint will override the init_from weights.")

        # On resume, Trainer replaces self.state wholesale with
        # TrainerState.load_from_json(checkpoint/trainer_state.json) — including
        # its save_steps/eval_steps — and DefaultFlowCallback schedules off
        # state.save_steps/eval_steps, not args.save_steps/eval_steps. So a new
        # --save_steps/--eval_steps passed on this invocation is silently
        # ignored (only the "do not match" warning fires) and the *old*
        # cadence baked into the checkpoint keeps governing. Patch it in place
        # so the intended cadence actually takes effect.
        if int(os.environ.get("RANK", "0")) == 0:
            state_path = os.path.join(last_checkpoint, "trainer_state.json")
            with open(state_path) as f:
                resumed_state = json.load(f)
            if (resumed_state.get("save_steps") != train_args.save_steps
                    or resumed_state.get("eval_steps") != train_args.eval_steps):
                print(f"[INFO] Overriding resumed checkpoint cadence: "
                      f"save_steps {resumed_state.get('save_steps')} -> {train_args.save_steps}, "
                      f"eval_steps {resumed_state.get('eval_steps')} -> {train_args.eval_steps}")
                resumed_state["save_steps"] = train_args.save_steps
                resumed_state["eval_steps"] = train_args.eval_steps
                with open(state_path, "w") as f:
                    json.dump(resumed_state, f, indent=2)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    
    ds = datasets.load_from_disk(args.saved_ds_path)
    eval_datasets = {k: v for k, v in ds.items() if k != "train"}

    if args.mini:
        ds["train"] = ds["train"].shuffle(seed=args.seed).select(range(100))
        if int(os.environ.get("RANK", "0")) == 0:
            eval_datasets = {k: v.shuffle(seed=args.seed).select(range(min(20, len(v)))) for k, v in eval_datasets.items()}
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if int(os.environ.get("RANK", "0")) != 0:
            eval_datasets = {k: v.shuffle(seed=args.seed).select(range(min(20, len(v)))) for k, v in eval_datasets.items()}

            
    print("Dataset loaded.")
    print({"train": len(ds["train"]), **{k: len(v) for k, v in eval_datasets.items()}})
    
    def compute_metrics(eval_pred):
        # The model already returns the BLENDED score as logits
        # (w*score_srcref + (1-w)*score_src for ref rows, score_src otherwise),
        predictions, labels = eval_pred
        predictions = predictions.flatten()


        n_nan_preds = int(np.isnan(predictions).sum())
        valid_mask = ~np.isnan(labels) & ~np.isnan(predictions)
        predictions = predictions[valid_mask]
        labels = labels[valid_mask]
        # Clamp so eval RMSE/MSE line up with the clamped inference script.
        # Correlations are unaffected by clamping.
        predictions = np.clip(predictions, 0.0, 1.0)
        mse = mean_squared_error(labels, predictions)
        rmse = root_mean_squared_error(labels, predictions)

        
        
        pearson, _ = pearsonr(predictions, labels)
        kendall, _ = kendalltau(predictions, labels)   # tau-b, handles ties
        spearman, _ = spearmanr(predictions, labels)

        return {"mse": mse, "rmse": rmse, "pearson": pearson,
            "kendall": kendall, "spearman": spearman, "n_nan_preds": n_nan_preds}

        
        


    # ── Sanity check: inspect one sample to verify the encoding ──
    import random
    idx = random.randint(0, len(ds["train"]) - 1)
    sample = ds["train"][idx]
    print("\n" + "="*60)
    print(f"SAMPLE INSPECTION (train[{idx}])")
    print("="*60)
    for k, v in sample.items():
        print(f"    {k:14}: {v}")
    print(f"    decoded       : {tokenizer.decode(sample['input_ids'], skip_special_tokens=False)}")
    print("="*60 + "\n")




    # ─────────────────────────────────────────────────────────────────────────
    databin_name = Path(args.saved_ds_path).name

    save_training_metadata(args, train_args,
                           files_to_copy=[__file__, inspect.getfile(QwenXComet)],
                           dirs_to_copy=[(Path(args.saved_ds_path) / "metadata", f"databin_metadata_{databin_name}")]
                           )
    
    # BackboneFreezeCallback blanket-freezes the backbone in on_train_begin and
    # only lifts it at the next epoch boundary. With frozen_epochs=0 that is a
    # no-op window before step 1, but it still mutates requires_grad right after
    # the optimizer is built — only attach it when it actually has work to do.
    callbacks = []
    if args.frozen_epochs and args.frozen_epochs > 0:
        callbacks.append(BackboneFreezeCallback(frozen_epochs=args.frozen_epochs))
    if args.early_stopping_patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    if args.time_limit_seconds:
        deadline = _PROCESS_START_TIME + args.time_limit_seconds - args.time_limit_safety_margin
        callbacks.append(TimeLimitCallback(deadline))
        print(f"[INFO] TimeLimitCallback armed: time_limit_seconds={args.time_limit_seconds:.0f}, "
              f"safety_margin={args.time_limit_safety_margin:.0f} -> will force-save+stop after "
              f"{args.time_limit_seconds - args.time_limit_safety_margin:.0f}s of process runtime.")


    use_disc_lr = args.backbone_lr is not None and abs(args.backbone_lr - args.learning_rate) > 1e-12
    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=eval_datasets,  # multiple datasets for eval, anything that isn't "train"
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    if use_disc_lr:
        print(f"[INFO] Discriminative LR: backbone_lr={args.backbone_lr:g}, head lr={args.learning_rate:g}")
        trainer = DiscriminativeLRTrainer(backbone_lr=args.backbone_lr, **trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)
    trainer.can_return_loss = True

    # last_checkpoint was already determined above (it gated backbone seeding).
    if last_checkpoint is not None:
        print(f"[INFO] Resuming training from checkpoint: {last_checkpoint}")
    else:
        print(f"[INFO] No checkpoint in {train_args.output_dir}, starting fresh.")

    print("[INFO] Starting trainer.train")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model()
    print(f"[INFO] Final model saved to {train_args.output_dir}")
    if torch.cuda.is_available():
        print(f"[INFO] PEAK GPU MEM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # defaults
    OUTPUT_DIR = "./metrics/models_trained/QwenXComet"
    MODEL_ID = "Qwen/Qwen3-Embedding-4B"


    parser = argparse.ArgumentParser(description="Train QwenXComet on pre-binarized data.")
    # Data
    parser.add_argument("--mini", action="store_true",
                        help="Use a small subset for quick testing.")
    parser.add_argument("--saved_ds_path", type=str, required=True,
                        help="Path to the binarized dataset (produced by binarize_data_xcomet.py).")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help=f"default: {OUTPUT_DIR}")
    
    # model
    parser.add_argument("--model_id", type=str, default=MODEL_ID,
                        help=f"HuggingFace model ID. Must match the one used by the binarization script. Default: {MODEL_ID}")
    parser.add_argument("--inference_ref_weight", type=float, default=0.5,
                        help="Blend weight w for the SRC+REF score at eval/inference: "
                             "logits = w*score_srcref + (1-w)*score_src for ref rows "
                             "(src-only rows unaffected). Moves the eval metrics. Default: 0.5.")

    # layerwise attention
    parser.add_argument("--use_layerwise_attention", action=argparse.BooleanOptionalAction, default=True) # creates a --no-use_layerwise_attention flag too
    parser.add_argument("--layer_transformation", choices=["softmax", "sparsemax"], default="softmax")
    parser.add_argument("--init_from", type=str, help="Path to a checkpoint from which to initialize the training.")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--backbone_lr", type=float, default=None,
                        help="Separate LR for the Qwen backbone; the regression head keeps "
                             "--learning_rate. Omit (or set equal to --learning_rate) for the "
                             "stock single-LR optimizer. Lower = protect pretrained / stage-1 "
                             "features (see DiscriminativeLRTrainer).")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--frozen_epochs", type=float, default=0, help="Number of epochs to keep the backbone frozen (0 = train full model from the start).")
    parser.add_argument("--lr_scheduler_type", type=str, default='cosine') 
    parser.add_argument("--num_decay_steps", type=int, default=None, help="Required only when --lr_scheduler_type=warmup_stable_decay. Number of steps for the decay phase at the end of training.")
    parser.add_argument("--early_stopping_patience", type=int, default=None, help="Stop if no improvement for N evals. Omit to disable.")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)


    # Eval & checkpoint strategy
    # Rule of thumb: eval_steps = train_size / (batch_size * desired_evals_per_epoch)
    parser.add_argument("--eval_steps", type=int, default=5000,
                        help="Evaluate every N steps. Tune so you get ~5-10 evals per epoch.")
    parser.add_argument("--save_steps", type=int, default=5000,
                        help="Save a checkpoint every N steps. Keep aligned with eval_steps.")
    parser.add_argument("--logging_steps", type=int, default=200,
                    help="Log training metrics every N steps.")
    parser.add_argument("--save_total_limit", type=int, default=20,
                        help="Max checkpoints to keep on disk.")
    parser.add_argument("--primary_eval_dataset", type=str, required=True,
                    choices=["dev", "dev_sent", "dev_paraf"],
                    help="Which eval dataset's drives checkpoint selection.")
    # Logging
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--run_name", type=str, default="QwenXComet_arch-note_date",
                        help="TensorBoard run name. Convention: {model}_{arch_note}_{date}.")
    parser.add_argument("--ds_config", type=str, default=None,
                        help="Path to the DeepSpeed ZeRO-2 JSON config file. Omit to run "
                             "WITHOUT DeepSpeed (single process / single GPU) — e.g. a quick "
                             "smoke test launched as plain `python ...`. DeepSpeed requires a "
                             "torchrun/deepspeed launcher; don't pass this under bare `python`.")

    # Wall-clock budget (e.g. remaining SLURM allocation seconds for this
    # invocation). Omit to disable — no forced save/stop.
    parser.add_argument("--time_limit_seconds", type=float, default=None,
                        help="Wall-clock budget for this process. If set, a checkpoint is "
                             "force-saved and training stopped cleanly "
                             "--time_limit_safety_margin seconds before this elapses, so a "
                             "SLURM TIMEOUT kill never discards progress made since the last "
                             "save_steps-triggered checkpoint.")
    parser.add_argument("--time_limit_safety_margin", type=float, default=300,
                        help="Seconds of buffer before --time_limit_seconds at which to force "
                             "the save (must cover checkpoint-write + clean shutdown time). Default: 300.")

    args = parser.parse_args()

    # Suppress prints on all ranks except 0 (avoids duplicated logs across processes)
    if int(os.environ.get("RANK", "0")) != 0:
        import builtins
        builtins.print = lambda *a, **k: None
    
    
        
    set_seed(args.seed)
    
    lr_scheduler_kwargs = {}
    if args.lr_scheduler_type == "warmup_stable_decay":
        if args.num_decay_steps is None:
            parser.error("--num_decay_steps is required when --lr_scheduler_type=warmup_stable_decay")
        lr_scheduler_kwargs["num_decay_steps"] = args.num_decay_steps
    
    train_args = TrainingArguments(
        args.output_dir,
        report_to=args.report_to,
        run_name=args.run_name,

        # Eval & saving — step-based so frequency doesn't depend on epoch length
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        load_best_model_at_end=True,
        metric_for_best_model=f"{args.primary_eval_dataset}_kendall",
        greater_is_better=True,

        # Hyperparameters
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,

        # Precision & efficiency
        bf16=True,  # use fp16=True if on V100
        
        remove_unused_columns=False,
        group_by_length=True,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        deepspeed=args.ds_config,
        optim="adafactor",

        # Logging
        logging_steps=args.logging_steps,
    )

    train(args, train_args)


