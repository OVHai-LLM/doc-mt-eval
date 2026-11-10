"""
Training script for GemmaItXComet, a learned MT-evaluation metric with a
Gemma-3 instruct backbone (google/gemma-3-1b-it). Trains on a databin produced by
binarize_data_xcomet.py (use the same --model_id for both: the data was tokenized
with that model's chat template).

To start from a trained model (e.g. stage 2 starting from stage 1), pass
--init_from <stage-1 output_dir>; the pretrained backbone is then not downloaded.

The backbone can be kept frozen for the first --frozen_epochs epochs (default 1).
We use --frozen_epochs 0: the separator tokens used for pooling (<unused0>/<unused1>)
have untrained embeddings, so the backbone has to be trained from the start.

Example usage (from the metrics/ folder):

python train_GemmaItXComet.py \
    --saved_ds_path databins/google_gemma-3-1b-it/stage1 \
    --output_dir models_trained/GemmaItXComet/stage1 \
    --primary_eval_dataset dev_paraf \
    --batch_size 8 --eval_batch_size 16 \
    --eval_steps 5000 --save_steps 5000 \
    --num_train_epochs 3 --frozen_epochs 0
"""

import argparse
import os
import torch
import datasets
import numpy as np
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
from modules.GemmaItXComet import GemmaItXCometConfig, GemmaItXComet, GemmaItXCometCollator
from utils.train_utils import print_trainable_parameters, save_training_metadata

print("datasets version:", datasets.__version__)



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


# ---------------------------------------------------------------------------
# Build model & tokenizer
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    collator = GemmaItXCometCollator(tokenizer)

    config = GemmaItXCometConfig(
        model_id=str(args.model_id),
        inference_ref_weight=args.inference_ref_weight,
        sentence_pooling=args.sentence_pooling,
        layer_transformation=args.layer_transformation
    )
    # pretrained backbone weights are only needed when not initialising from a trained checkpoint
    model = GemmaItXComet(config, load_backbone_weights=not args.init_from)

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
    
    print("--- Model statistics (before freeze) ---")
    print_trainable_parameters(model)

    if args.init_from:
        print("[INFO] Initializing model from checkpoint:", args.init_from)
        ckpt = Path(args.init_from)
        sf, bin_ = ckpt / "model.safetensors", ckpt / "pytorch_model.bin"
        if sf.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(sf))
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
    

    last_checkpoint = None
    if os.path.isdir(train_args.output_dir):
        last_checkpoint = get_last_valid_checkpoint(train_args.output_dir)

    if last_checkpoint is None:
        print("[INFO] No last checkpoint found.")
    else:
        print(f"[INFO] Resuming from {last_checkpoint}: Trainer will overwrite backbone "
          f"from the checkpoint via resume_from_checkpoint.")

    
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
    print({split: len(ds[split]) for split in ds})

    def compute_metrics(eval_pred):
        # The model returns the BLENDED score as logits
        # (w*score_srcref + (1-w)*score_src for ref rows, score_src otherwise),
        # so `predictions` are the final per-row predictions
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        n_nan_preds = int(np.isnan(predictions).sum())
        valid_mask = ~np.isnan(labels)  & ~np.isnan(predictions)          # the binarizer rejects NaN labels but to be sure. also non nan predictions (possible if the model outputs NaN for some reason)
        predictions = predictions[valid_mask]
        labels = labels[valid_mask]
        # Optional: the model does not clamp (non-differentiable and would zero 
        # the gradient on out-of-range predictions during training). 
        # Clamp here so eval RMSE/MSE line up with
        # the clamped inference script. Correlations are unaffected by clamping;
        # remove this line if you want raw-range error metrics.
        predictions = np.clip(predictions, 0.0, 1.0)
        mse = mean_squared_error(labels, predictions)
        rmse = root_mean_squared_error(labels, predictions)
        pearson, _ = pearsonr(predictions, labels)
        kendall, _ = kendalltau(predictions, labels)   # tau-b, handles ties
        spearman, _ = spearmanr(predictions, labels)

        return {"mse": mse, "rmse": rmse, "pearson": pearson,
            "kendall": kendall, "spearman": spearman, "n_nan_preds": n_nan_preds}

    databin_name = Path(args.saved_ds_path).name
    
    save_training_metadata(args, train_args, 
                           files_to_copy= [__file__, inspect.getfile(GemmaItXComet)],
                          dirs_to_copy=[(Path(args.saved_ds_path) / "metadata", f"databin_metadata_{databin_name}")]
                          )
    
    callbacks = []
    if args.frozen_epochs and args.frozen_epochs > 0:
        callbacks.append(BackboneFreezeCallback(frozen_epochs=args.frozen_epochs))
    else:
        print("[INFO] frozen_epochs=0 -> BackboneFreezeCallback not attached; "
              "backbone trainable from step 0.")
    if args.early_stopping_patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=eval_datasets, # multiple datasets for eval, anything that isn't "train"
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # defaults
    OUTPUT_DIR = "./metrics/models_trained/GemmaItXComet"
    MODEL_ID = "google/gemma-3-1b-it"


    parser = argparse.ArgumentParser(description="Train GemmaItXComet on pre-binarized data.")
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

    parser.add_argument("--layer_transformation", choices=["softmax", "sparsemax"], default="softmax")
    parser.add_argument("--sentence_pooling", choices=["mean", "last"], default="last")
    parser.add_argument("--init_from", type=str, help="Path to a checkpoint from which to initialize the training.")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--frozen_epochs", type=float, default=1, help="Number of epochs to keep the backbone frozen (0 = train full model from the start).")
    parser.add_argument("--lr_scheduler_type", type=str, default='linear') 
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
    parser.add_argument("--primary_eval_dataset", type=str, default="dev",
                    choices=["dev", "dev_sent", "dev_paraf"],
                    help="Which eval dataset drives checkpoint selection.")
    
    # Logging
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--run_name", type=str, default="GemmaItXComet_arch-note_date",
                        help="TensorBoard run name. Convention: {model}_{arch_note}_{date}.")
    parser.add_argument("--ds_config", type=str, default=None,
                        help="Path to the DeepSpeed ZeRO-2 JSON config file. Omit to run "
                             "WITHOUT DeepSpeed (single process / single GPU) — e.g. a quick "
                             "smoke test launched as plain `python ...`. DeepSpeed requires a "
                             "torchrun/deepspeed launcher; don't pass this under bare `python`.")

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


