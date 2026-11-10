import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import SequenceClassifierOutput, ModelOutput
from dataclasses import dataclass
from typing import Optional
from transformers import PretrainedConfig, PreTrainedModel
from pathlib import Path
import sys

import warnings
warnings.filterwarnings("ignore", message=".*fast tokenizer.*")

# local import
from .layerwise_attention import LayerWiseAttention

"""
GemmaItXComet: an xCOMET-style MT evaluation metric with a Gemma-3 instruct
(causal LLM) backbone.

Differences vs. xCOMET (Guerreiro et al., 2024):
  - Single head (sentence-level regression). No word-level tagger, no MQM
    score inference from spans.
  - Input variants restricted to SRC and SRC+REF (no REF-only mode): the
    source is assumed always available; the reference is optional.
  - Backbone is a causal decoder, so a single forward pass over
        <prompt> <src> <mt> <sep1> <ref> <sep2>
    yields both variants: the representation at sep1 (sep1_pos) has seen only
    src+mt, the one at sep2 (sep2_pos) has seen everything. The sentence
    representation is the layer-mixed hidden state at the separator
    (config.sentence_pooling="last", default) or the masked mean up to it ("mean").

Each variant is mixed (LayerWiseAttention, as in Unbabel COMET) under its own
truncated mask (positions <= sep1_pos for SRC, <= sep2_pos for SRC+REF), so
that the layer normalisation of the SRC variant never sees reference tokens.

Training (xCOMET Sec. 4.2.1, adapted to two variants and optional refs):
    L = (1 - w) * MSE(y_src, labels) + w * MSE(y_srcref, labels[has_ref])
    L = MSE(y_src, labels)                              if no row has a ref

Inference: score = w * y_srcref + (1 - w) * y_src for rows with a reference
(w = config.inference_ref_weight, default 0.5), y_src otherwise.
"""

@dataclass
class GemmaItXCometOutput(ModelOutput):
    loss:         Optional[torch.FloatTensor] = None
    logits:       Optional[torch.FloatTensor] = None  # combined score [B, 1] (canonical)
    score_src:    Optional[torch.FloatTensor] = None  # [B]
    score_srcref: Optional[torch.FloatTensor] = None  # [B], NaN where has_ref == 0
    
class GemmaItXCometConfig(PretrainedConfig):
    model_type = "GemmaItXComet"

    def __init__(self,
                 model_id="google/gemma-3-1b-it",
                 inference_ref_weight=0.5,      # weight of SRC+REF score at inference
                 sentence_pooling="last",       # "last" (default) | "mean"
                 layer_transformation="softmax", # layerwise attention args
                 dropout_p=0.1,
                 layer_dropout_p=0.1, # layerwise attention args
                 layer_norm=True, # layerwise attention args
                 backbone_config=None, # backbone architecture (dict), filled automatically
                 **kwargs):
        super().__init__(**kwargs)

        self.model_id = str(model_id)
        self.backbone_config = backbone_config
        self.dropout_p = dropout_p
        self.inference_ref_weight = inference_ref_weight
        
        # layerwise attn
        self.layer_dropout_p = layer_dropout_p
        self.layer_norm = layer_norm
        assert layer_transformation in ("softmax", "sparsemax")
        self.layer_transformation = layer_transformation

        # sentence pooling
        assert sentence_pooling in ("last", "mean")
        self.sentence_pooling = sentence_pooling


class GemmaItXComet(PreTrainedModel):
    config_class = GemmaItXCometConfig

    def __init__(self, config: GemmaItXCometConfig, load_backbone_weights=False):
        super().__init__(config)

        if config.backbone_config is not None:
            bc = dict(config.backbone_config)
            base_config = AutoConfig.for_model(bc.pop("model_type"), **bc)
        else:
            base_config = AutoConfig.from_pretrained(config.model_id, trust_remote_code=True)
            config.backbone_config = base_config.to_dict()
        self.hf_config = getattr(base_config, "text_config", base_config)

        # Pretrained weights only for training; otherwise they come from the checkpoint
        if load_backbone_weights:
            print('Loading backbone from:', config.model_id)
            full_model = AutoModelForCausalLM.from_pretrained(config.model_id, trust_remote_code=True)
        else:
            full_model = AutoModelForCausalLM.from_config(base_config, trust_remote_code=True)
        self.backbone = full_model.model

        # KV-cache is generation-only and incompatible with gradient
        # checkpointing (HF prints a warning and disables it on the fly
        # otherwise). We never generate with this backbone, so disable it
        # explicitly to keep training logs clean and behaviour deterministic.
        self.backbone.config.use_cache = False

        self.N = self.hf_config.num_hidden_layers + 1 # embedding layer too
        print(f'Number of mixed layers (N, incl. embeddings): {self.N}')

        # ONE LayerWiseAttention shared across both input variants (SRC and
        # SRC+REF)
        self.layerwise_attention = LayerWiseAttention(
            self.N, 
            config.layer_dropout_p, 
            config.layer_norm,
            layer_transformation=config.layer_transformation,
        )

        self.dropout = nn.Dropout(p=config.dropout_p)
        
        # Feed-forward estimator.
        # Estimator over a single sentence embedding (no products/differences:
        # cross-input interactions happen inside the joint encoder).
        # Dropout placement follows Unbabel's FeedForward: Linear -> activation
        # -> Dropout per hidden layer; no dropout on the pooled input.
        self.act = nn.Tanh()

        C = self.hf_config.hidden_size
        
        self.ff1 = nn.Linear(C, 3072, dtype=torch.bfloat16)
        self.ff2 = nn.Linear(3072, 1536, dtype=torch.bfloat16)
        self.final = nn.Linear(1536, 1, dtype=torch.bfloat16)




    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Delegate gradient checkpointing to the backbone — the only submodule
		large enough to benefit from it. LayerWiseAttention and the estimator
		FFN are small and don't need it.

		Must be used with {"use_reentrant": False} (passed in TrainingArguments)"""
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()

    def _pool(self, mixed, attention_mask, sep_pos):
        """
        mixed: [B, T, C] layer-mixed hidden states; attention_mask: [B, T];
        sep_pos: [B] separator positions.
        "last": hidden state at the separator, which has attended to everything
                before it (the causal analogue of [cls]).
        "mean": masked mean pooling over the tokens up to the separator.
        Assumes RIGHT padding (GemmaItXCometCollator pads right).
        """
        if self.config.sentence_pooling == "last":
            batch_idx = torch.arange(mixed.size(0), device=mixed.device)
            out = mixed[batch_idx, sep_pos]                                       # [B, C]
            return out

        elif self.config.sentence_pooling == "mean":
            T = mixed.size(1)
            idx = torch.arange(T, device=mixed.device).unsqueeze(0)   # [1, T]
            mask = (idx <= sep_pos.unsqueeze(1)).float() * attention_mask.float()  # [B, T]  sets tokens after pos to 0
            mask = mask.unsqueeze(-1)
            out = (mixed * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return out

        else:
            raise ValueError(f"Unknown sentence_pooling: {self.config.sentence_pooling}")

    # ---------------------------------------------------------------- forward

    def _ffn(self, sentemb):
        sentemb = sentemb.to(self.ff1.weight.dtype)
        h1 = self.dropout(self.act(self.ff1(sentemb)))
        h2 = self.dropout(self.act(self.ff2(h1)))
        return self.final(h2).view(-1)
    
    def forward(self,
                input_ids, attention_mask,
                sep1_pos, sep2_pos=None,
                has_ref=None,
                labels=None,
                return_variant_scores=False): # whether to output the per variant scores
        """
        input_ids      [B, T]: one sequence per row, shared by both variants.
        sep1_pos       [B]   : position of sep1 (end of src+mt).
        sep2_pos       [B]   : position of sep2 (end of ref); the collator fills
                               it with sep1_pos for rows without a reference.
        has_ref        [B]   : bool mask; False rows contribute only the SRC term.
        Loss:
            L = (1 - w) * MSE(y_src, labels) + w * MSE(y_srcref, labels[has_ref])
        Inference logits:
            w * y_srcref + (1 - w) * y_src   where has_ref, else y_src
        """

        backbone_frozen = not next(self.backbone.parameters()).requires_grad
        if backbone_frozen:
            with torch.no_grad():
                hs = self.backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    ).hidden_states   # keep ALL N+1 (incl. embeddings) → [N, B, T, C]
        else: # no context manager - the default (during eval no_grad)
            hs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    ).hidden_states   # keep ALL N+1 (incl. embeddings) → [N, B, T, C]


        T = attention_mask.size(1)
        pos_idx = torch.arange(T, device=attention_mask.device).unsqueeze(0)  # [1, T]
        
        mask_src = (pos_idx <= sep1_pos.unsqueeze(1)).long() * attention_mask
        mixed_src = self.layerwise_attention(hs, mask_src)
        sentemb_src = self._pool(mixed_src, mask_src, sep1_pos)
        score_src = self._ffn(sentemb_src)

        # --- SRC+REF variant ---
        # When ref info exists this is computed for EVERY row, unconditionally.
        # No-ref rows carry a dummy sep2_pos == sep1_pos from the collator and
        # produce a throwaway score that every downstream use masks out via
        # `has_ref`. It must NOT be guarded by `if has_ref.any()`: under DeepSpeed
        # ZeRO-3 the number of autograd ops -- hence gradient reduce-scatter
        # collectives per step -- would then depend on the local micro-batch
        # composition. group_by_length tends to cluster the (short) no-ref rows,
        # so a single rank can get an all-no-ref micro-batch, skip this sub-graph,
        # post fewer collectives than its peers, and deadlock the whole job.
        score_srcref = None
        if has_ref is not None:
            sep2 = sep2_pos if sep2_pos is not None else sep1_pos
            mask_srcref = (pos_idx <= sep2.unsqueeze(1)).long() * attention_mask
            mixed_srcref = self.layerwise_attention(hs, mask_srcref)
            sentemb_srcref = self._pool(mixed_srcref, mask_srcref, sep2)   # [B, C], all rows
            score_srcref = self._ffn(sentemb_srcref)                      # [B]

        w = self.config.inference_ref_weight
        B = score_src.size(0)

        # --- combined logits: use the srcref score only where has_ref is True ---
        combined = score_src.clone()
        if score_srcref is not None:
            combined[has_ref] = w * score_srcref[has_ref] + (1.0 - w) * score_src[has_ref]

        # --- loss ---
        loss = None
        if labels is not None:
            labels = labels.view(-1).to(torch.float32)
            loss_fct = nn.MSELoss()
            loss = loss_fct(score_src.to(torch.float32), labels)  # over all rows
            if score_srcref is not None:
                has_ref_f = has_ref.to(torch.float32)
                # ref MSE over ref rows only, but kept graph-connected even when
                # the local micro-batch has zero ref rows (constant-0 gradient,
                # so the reduce-scatter schedule stays identical across ranks --
                # see the SRC+REF comment above).
                ref_sq_err = (score_srcref.to(torch.float32) - labels) ** 2 * has_ref_f
                ref_loss = ref_sq_err.sum() / has_ref_f.sum().clamp(min=1.0)
                has_any_ref = (has_ref_f.sum() > 0).to(loss.dtype)   # 0.0 / 1.0, no grad
                # has_any_ref==1 -> (1-w)*loss_src + w*ref_loss ; ==0 -> loss_src
                loss = loss + has_any_ref * w * (ref_loss - loss)

        # --- variant scores for inspection/inference ---
        score_srcref_out = None
        if return_variant_scores and score_srcref is not None:
            score_srcref_out = torch.full((B,), float("nan"), dtype=score_srcref.dtype, device=score_srcref.device)
            score_srcref_out[has_ref] = score_srcref[has_ref]   # NaN at no-ref rows

        return GemmaItXCometOutput(
            loss=loss,
            logits=combined.unsqueeze(-1),
            score_src=score_src if return_variant_scores else None,
            score_srcref=score_srcref_out,
        )

        

class GemmaItXCometCollator:
    """
    Collator for GemmaItXComet. Pads RIGHT.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right" # needs to pad at right side because we saved the ids of the special tokens (left padding would move them)

    def __call__(self, features):
        has_labels = "labels" in features[0]
        labels = [f.pop("labels") for f in features]  if has_labels else None
        
        padded = self.tokenizer.pad([{"input_ids": f["input_ids"], 
                                      "attention_mask": f["attention_mask"]} 
                                     for f in features], padding=True, return_tensors="pt",
                                    )
        input_ids, attention_mask = padded["input_ids"], padded["attention_mask"]
        sep1_pos = torch.tensor([f["sep1_pos"] for f in features], dtype=torch.long)

        has_ref  = torch.tensor([f["sep2_pos"] is not None for f in features], dtype=torch.bool)
    
        sep2_pos = torch.tensor([f["sep2_pos"] if f["sep2_pos"] is not None else f["sep1_pos"]  for f in features], dtype=torch.long)
        
        out= {
            "input_ids":	  input_ids,
            "attention_mask": attention_mask,
            "sep1_pos":        sep1_pos,
            "sep2_pos":        sep2_pos,
            "has_ref": has_ref,
        }
        if labels is not None:
            out["labels"]= torch.tensor(labels, dtype=torch.float32)
        return out







### ======================================================================= ###
# Quick checks (run from the metrics/ folder: python -m modules.GemmaItXComet [trained_checkpoint])
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from binarize_data_xcomet import build_messages, encode_messages

    MAX_LEN = 8192

    print(" ========= 1. Check backbone creation from pretrained LLM ============")
    model_id = "google/gemma-3-1b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = GemmaItXCometConfig(model_id=model_id, layer_transformation="softmax", sentence_pooling="last")
    
    model = GemmaItXComet(config, load_backbone_weights=True)

    print("\n[CHECK] Comparing backbone to the true pretrained weights...")
    _ref = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
    _key = "embed_tokens.weight"
    _a = dict(model.backbone.named_parameters())[_key].detach().float()
    _b = dict(_ref.model.named_parameters())[_key].detach().float()
    del _ref
    print(f"    backbone   first 5: {_a.flatten()[:5].tolist()}")
    print(f"    pretrained first 5: {_b.flatten()[:5].tolist()}")
    print(f"    -> Is the backbone RANDOM? (does NOT match pretrained): "
          f"{not torch.allclose(_a, _b)}")

    


    print(" ========= 2. Check forward run (trained checkpoint if given, else the model above) =========")
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
        model = GemmaItXComet.from_pretrained(ckpt)
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model.eval()

    prompt_path = Path(__file__).resolve().parents[2] / "configs" / "prompts" / "xcomet_user_prompt.txt"
    with open(prompt_path) as f:
        user_prompt_template = f.read()

    sep1, sep2 = "<unused0>", "<unused1>"
    sep1_id = tokenizer.convert_tokens_to_ids(sep1)
    sep2_id = tokenizer.convert_tokens_to_ids(sep2)

    def build_row(src, tgt, ref, src_lang, tgt_lang, label):
        messages = build_messages(user_prompt_template, None, src, tgt, ref, src_lang, tgt_lang, sep1, sep2)
        ids, mask, sep1_pos, sep2_pos = encode_messages(tokenizer, messages, MAX_LEN, sep1_id, sep2_id,  has_ref=(ref is not None))
        
        return {"input_ids": ids, "attention_mask": mask, "sep1_pos": sep1_pos, "sep2_pos": sep2_pos, "labels": label}


    
    feats = [
    build_row(src="The cat sat.", tgt="Le chat.", ref="Le chat s'est assis.", src_lang="English", tgt_lang="French", label=0.3),
    build_row(src="The cat sat.", tgt="Le chat.", ref=None, src_lang="English", tgt_lang="French", label=0.3),
    build_row(src="Hello world.", tgt="Bonjour monde.", ref=None, src_lang="English", tgt_lang="French", label=0.9),  # no-ref row
    ]

    print(' ==== feats ===')
    for f in feats:
        print("f['input_ids']", f['input_ids'])
        print("f['input_ids'] decoded", tokenizer.decode(f['input_ids']))
        print("f['sep1_pos']", f['sep1_pos'])
        print("f['sep2_pos']", f['sep2_pos'])
        print()


    
    batch = GemmaItXCometCollator(tokenizer)(feats)

    with torch.no_grad():
        out = model(**batch, return_variant_scores=True)
    print(f"Scores: {out.logits.view(-1).tolist()}  |  Loss: {out.loss.item():.6f}")

    print(f"score_src: {out.score_src.tolist()}  |  score_srcref: {out.score_srcref.tolist()}")
    