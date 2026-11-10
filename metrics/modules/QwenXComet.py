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
QwenXComet: an xCOMET-style MT evaluation metric with a Qwen3-Embedding
backbone (a causal Qwen3 decoder contrastively trained for sentence
embeddings — architecturally identical to a plain causal LM, so the raw
forward still returns per-token hidden states, but it was trained with
last-token pooling at <|endoftext|>).

Differences vs. xCOMET (Guerreiro et al., 2024):
  - Single head (sentence-level regression). No word-level tagger, no MQM
    score inference from spans.
  - Input variants restricted to SRC and SRC+REF (no REF-only mode): the
    source is assumed always available; the reference is optional.
  - Backbone is a causal decoder, so a single forward pass over
        <src> <mt> <EOS> <ref> <EOS>
    yields both variants: pooling at the FIRST EOS (sep1_pos) sees only
    src+mt, pooling at the FINAL EOS (sep2_pos) sees everything. Rows
    without a reference end at the first EOS and have sep2_pos = None.

Attention masking:
  Each variant is mixed (using LayerWise attention) under its OWN truncated 
  mask (positions <= sep1_pos for SRC, <= sep2_pos for SRC+REF). 
  This matters even though the pooled vector is already causally clean: 
  LayerWiseAttention normalises using mean/variance over all unmasked positions,
  so mixing under the full mask would leak reference statistics 
  into the SRC-only score.

The LayerWiseAttention module follows the Unbabel COMET implementation;
`layer_transformation` defaults to "softmax" here (xCOMET uses sparsemax).
Layer dropout uses the global RNG, exactly as Unbabel does.

Training (xCOMET Sec. 4.2.1, adapted to two variants and optional refs):
the SRC variant is scored for every row; the SRC+REF variant only for the
subset with a reference (has_ref). The two terms are combined as a weighted
average rather than a sum, so batches with and without references produce
comparable gradient magnitudes:

    L = (1 - w) * MSE(y_src, labels) + w * MSE(y_srcref, labels[has_ref])
    L = MSE(y_src, labels)                              if no row has a ref

Note that w (config.inference_ref_weight) weights BOTH the training loss and
the inference blend, so changing it changes the training objective, not just
how scores are combined. Note also that the SRC+REF term is a mean over ref
rows only — batches with few reference rows give those rows proportionally
more influence.

Inference:
    score = w * y_srcref + (1 - w) * y_src    for rows with a reference
    score = y_src                             otherwise
`score_srcref` is returned as a full-length [B] tensor with NaN at rows
without a reference.
"""

@dataclass
class QwenXCometOutput(ModelOutput):
    loss:         Optional[torch.FloatTensor] = None
    logits:       Optional[torch.FloatTensor] = None  # combined score [B, 1] (canonical)
    score_src:    Optional[torch.FloatTensor] = None  # [B]
    score_srcref: Optional[torch.FloatTensor] = None  # [B], NaN where has_ref == 0
    
class QwenXCometConfig(PretrainedConfig):
    model_type = "QwenXComet"

    def __init__(self,
                 model_id="Qwen/Qwen3-Embedding-0.6B",
                 inference_ref_weight=0.5,      # weight of SRC+REF score at inference
                 dropout_p=0.1,
                 layer_transformation="softmax", # layerwise attention args
                 use_layerwise_attention=True, # layerwise attention args
                 layer_dropout_p=0.1, # layerwise attention args
                 layer_norm=True, # layerwise attention args
                 backbone_config=None, # backbone architecture (dict), filled automatically
                 **kwargs):
        super().__init__(**kwargs)

        if not use_layerwise_attention and (layer_transformation or layer_norm or layer_dropout_p):
            warnings.warn("Layerwise args (layer_transformation, layer_norm, layer_dropout_p) ignored when use_layerwise_attention=False")
        
        self.model_id = str(model_id)
        self.backbone_config = backbone_config
        self.dropout_p = dropout_p
        self.inference_ref_weight = inference_ref_weight
        
        # layerwise attn
        self.layer_dropout_p = layer_dropout_p
        self.layer_norm = layer_norm
        assert layer_transformation in ("softmax", "sparsemax")
        self.layer_transformation = layer_transformation
        self.use_layerwise_attention = use_layerwise_attention


class QwenXComet(PreTrainedModel):
    config_class = QwenXCometConfig

    def __init__(self, config: QwenXCometConfig, load_backbone_weights=False):
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

        self.backbone.config.use_cache = False

        self.N = self.hf_config.num_hidden_layers + 1 # embedding layer too
        print(f'Number of mixed layers (N, incl. embeddings): {self.N}')

        # ONE LayerWiseAttention shared across both input variants (SRC and
        # SRC+REF)
        if config.use_layerwise_attention:
            self.layerwise_attention = LayerWiseAttention(
                self.N, 
                config.layer_dropout_p, 
                config.layer_norm,
                layer_transformation=config.layer_transformation,
            )
        else: self.layerwise_attention = None

        self.dropout = nn.Dropout(p=config.dropout_p) # dropout in the ffn
        
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

    def _pool(self, hs, sep_pos):
        """
        hs: [B, T, C] hidden states.
        
        Will extract the hidden state at the given separator position (sep_pos),
        which has attended to everything before it (causal [cls]).
        Note: Qwen3-Embedding was trained to pool at the final EOS
        token; pooling at mid-sequence separators is our deviation.
        Assumes RIGHT padding (QwenXCometCollator pads right).
        """
        batch_idx = torch.arange(hs.size(0), device=hs.device)
        out = hs[batch_idx, sep_pos]                                       # [B, C]
        return out

    # ---------------------------------------------------------------- forward

    def _ffn(self, sentemb):
        sentemb = sentemb.to(self.ff1.weight.dtype)
        h1 = self.dropout(self.act(self.ff1(sentemb)))
        h2 = self.dropout(self.act(self.ff2(h1)))
        return self.final(h2).view(-1)
    
    def forward(self,
                input_ids, attention_mask,
                sep1_pos, sep2_pos, has_ref,
                labels=None,
                return_variant_scores=False): # whether to output the per variant scores
        """
        input_ids      [B, T] : one sequence per row, shared by both variants.
        sep1_pos       [B]    : position of the first EOS (end of src+mt).
        sep2_pos       [B]    : position of the final EOS (end of ref). For
                                rows without a reference the collator fills
                                this with sep1_pos as a dummy; has_ref is what
                                actually decides whether it is used.
        has_ref        [B]    : bool mask; False rows contribute only the SRC
                                term. Required — sep2_pos alone cannot signal
                                this, since the collator fills it with a dummy.

        Loss:
            L = (1 - w) * MSE(y_src, labels) + w * MSE(y_srcref, labels[has_ref])
        Inference logits:
            w * y_srcref + (1 - w) * y_src   where has_ref, else y_src
        """

        # output_hidden_states is only needed by layerwise_attention (which mixes
        # ALL N+1 layers); with it disabled, only last_hidden_state is used, so
        # requesting the full per-layer tuple would just materialize N extra
        # [B, T, C] tensors for nothing.
        need_all_layers = self.config.use_layerwise_attention
        backbone_frozen = not next(self.backbone.parameters()).requires_grad
        if backbone_frozen:
            with torch.no_grad():
                out = self.backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=need_all_layers,
                    )
        else: # no context manager - the default (during eval no_grad)
            out = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=need_all_layers,
                    )

        # Two SEPARATE truncated masks. The pooled vector at sep1 is causally
        # clean, but the layer-norm inside the mixing step computes mean/variance
        # over every unmasked position — with the full mask that includes the
        # reference tokens after sep1, leaking the reference into score_src.
        T = attention_mask.size(1)
        pos_idx = torch.arange(T, device=attention_mask.device).unsqueeze(0)  # [1, T]

        def mix(mask):
            if self.config.use_layerwise_attention:
                # out.hidden_states: ALL N+1 layers (incl. embeddings) → [N, B, T, C]
                return self.layerwise_attention(out.hidden_states, mask)   # [B, T, C]
            mask_f = mask.float()
            # same normalization as the one used inside LayerWiseAttention
            return LayerWiseAttention._layer_norm(out.last_hidden_state,
                                                  mask_f.unsqueeze(-1), mask_f)

        # --- SRC variant: everything up to and including sep1 (all rows) ---
        mask_src = (pos_idx <= sep1_pos.unsqueeze(1)).long() * attention_mask
        sentemb_src = self._pool(mix(mask_src), sep1_pos)          # [B, C]
        score_src = self._ffn(sentemb_src)                          # [B]

        # --- SRC+REF variant ---
        # Computed for EVERY row, unconditionally. No-ref rows carry a dummy
        # sep2_pos == sep1_pos from the collator and produce a throwaway score that
        # every downstream use masks out via `has_ref`.
        #
        # This must NOT be guarded by `if has_ref.any()`: under DeepSpeed ZeRO-3 the
        # number of autograd ops — hence the number of gradient reduce-scatter
        # collectives per step — would then depend on the local batch composition.
        # group_by_length tends to cluster the (short) no-ref rows into the same
        # batch, so a single rank could get an all-no-ref micro-batch, skip this
        # sub-graph, post fewer collectives than its peers, and deadlock the whole
        # job on a reduce-scatter the other ranks are waiting for.
        mask_srcref = (pos_idx <= sep2_pos.unsqueeze(1)).long() * attention_mask
        sentemb_srcref = self._pool(mix(mask_srcref), sep2_pos)     # [B, C]
        score_srcref = self._ffn(sentemb_srcref)                    # [B]

        w = self.config.inference_ref_weight
        B = score_src.size(0)
        has_ref_f = has_ref.to(torch.float32)

        # --- combined logits: use the srcref score only where has_ref is True ---
        combined = score_src.clone()
        combined[has_ref] = w * score_srcref[has_ref] + (1.0 - w) * score_src[has_ref]

        # --- loss ---
        loss = None
        if labels is not None:
            labels = labels.view(-1).to(torch.float32)
            loss_fct = nn.MSELoss()
            loss_src = loss_fct(score_src.to(torch.float32), labels)  # over all rows
            # ref MSE over ref rows only, kept graph-connected even when the local
            # batch has zero ref rows (constant-0 gradient, but the reduce-scatter
            # schedule stays identical across ranks — see comment above).
            ref_sq_err = (score_srcref.to(torch.float32) - labels) ** 2 * has_ref_f
            ref_loss = ref_sq_err.sum() / has_ref_f.sum().clamp(min=1.0)
            has_any_ref = (has_ref_f.sum() > 0).to(loss_src.dtype)   # 0.0 / 1.0, no grad
            # has_any_ref==1 -> (1-w)*loss_src + w*ref_loss ; ==0 -> loss_src
            loss = loss_src + has_any_ref * w * (ref_loss - loss_src)

        # --- variant scores for inspection/inference ---
        # expand-with-NaN kept for API compatibility (score_srcref is already [B]).
        score_srcref_out = None
        if return_variant_scores:
            score_srcref_out = torch.full((B,), float("nan"),
                                          dtype=score_srcref.dtype, device=score_srcref.device)
            score_srcref_out[has_ref] = score_srcref[has_ref]

        return QwenXCometOutput(
            loss=loss,
            logits=combined.unsqueeze(-1),
            score_src=score_src if return_variant_scores else None,
            score_srcref=score_srcref_out,
        )






        
class QwenXCometCollator:
    """
    Collator for QwenXComet. Pads RIGHT.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right" # has to pad right, otherwise the special token position ids won't make sense!

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
        sep2_pos = torch.tensor([f["sep2_pos"] if f["sep2_pos"] is not None else f["sep1_pos"]
                                 for f in features], dtype=torch.long)
        
        out= {
            "input_ids":	   input_ids,
            "attention_mask":  attention_mask,
            "sep1_pos":        sep1_pos,
            "sep2_pos":        sep2_pos,
            "has_ref":         has_ref,
        }
        if labels is not None:
            out["labels"]= torch.tensor(labels, dtype=torch.float32)
        return out


# Quick checks on a databin (run from the metrics/ folder:
#   python -m modules.QwenXComet [databin_path])
if __name__ == "__main__":
    from pathlib import Path
    import json
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    print("\n=== DATA FLOW STRESS CHECK ===")
    DATABIN = sys.argv[1] if len(sys.argv) > 1 else "databins/Qwen_Qwen3-Embedding-0.6B/test_stage1_1.5"
    SPLIT = "train"
    N_REF, N_NOREF = 3, 2          # deliberately mixed batch

    model_id = json.loads((Path(DATABIN) / "metadata" / "tokenizer_info.json").read_text())["model_id"]

    print(f"\nLoading databin: {DATABIN}")
    ds = load_from_disk(DATABIN)
    print(f"Splits: {list(ds.keys())}, sizes: { {s: len(ds[s]) for s in ds} }")
    split_ds = ds[SPLIT]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Read the separator from the databin metadata rather than hardcoding it —
    # this is what the binarizer actually used to build these rows.
    _meta = json.loads((Path(DATABIN) / "metadata" / "tokenizer_info.json").read_text())
    EMBED_EOS    = _meta["sep_token"]
    EMBED_EOS_ID = _meta["sep_token_id"]
    assert tokenizer.convert_tokens_to_ids(EMBED_EOS) == EMBED_EOS_ID, (
        f"databin sep_token {EMBED_EOS!r} maps to {tokenizer.convert_tokens_to_ids(EMBED_EOS)} "
        f"with this tokenizer, but the databin recorded {EMBED_EOS_ID} — tokenizer/databin mismatch.")
    print(f"[meta] sep_token={EMBED_EOS!r} id={EMBED_EOS_ID} (from databin metadata)")

    

    # ── Build a MIXED batch: without this the first N rows are likely all-ref
    #    and the has_ref code path is never exercised. ──
    sep2_col = split_ds["sep2_pos"]
    ref_idx   = [i for i, s2 in enumerate(sep2_col) if s2 is not None][:N_REF]
    noref_idx = [i for i, s2 in enumerate(sep2_col) if s2 is None][:N_NOREF]
    print(f"\nref rows in {SPLIT}: {sum(1 for s in sep2_col if s is not None)} / {len(sep2_col)}")
    if not noref_idx:
        print("[WARN] no no-ref rows in this split — the has_ref path will NOT be tested. "
              "Rebuild the databin with the no-ref-enabled binarizer, or point at a split that has them.")
    idxs = ref_idx + noref_idx
    print(f"sampling indices {idxs} ({len(ref_idx)} ref, {len(noref_idx)} no-ref)")

    config = QwenXCometConfig(model_id=model_id,
                              layer_transformation="softmax",
                              use_layerwise_attention=True)

    model = QwenXComet(config, load_backbone_weights=True)
    model.eval()

    samples = [split_ds[i] for i in idxs]
    collator = QwenXCometCollator(tokenizer)
    batch = collator(samples)

    print(f"\nBatch input_ids shape: {batch['input_ids'].shape}")
    print(f"has_ref: {batch['has_ref'].tolist()}")

    for row in range(len(samples)):
        ids = batch["input_ids"][row].tolist()
        s1 = batch["sep1_pos"][row].item()
        s2 = batch["sep2_pos"][row].item()
        hr = bool(batch["has_ref"][row])
        n_real = batch["attention_mask"][row].sum().item()
        print(f"\n--- row {row} ({'REF' if hr else 'NO-REF'}) ---")
        print(f"sep1_pos={s1} -> token_id={ids[s1]} "
              f"({'OK, is EOS' if ids[s1] == EMBED_EOS_ID else 'MISMATCH'})")
        if hr:
            print(f"sep2_pos={s2} -> token_id={ids[s2]} "
                  f"({'OK, is EOS' if ids[s2] == EMBED_EOS_ID else 'MISMATCH'})")
            print(f"is sep2_pos the last real token? {s2 == n_real - 1}")
            print(f"decoded sep1->sep2: {tokenizer.decode(ids[s1+1:s2+1])!r}")
        else:
            print(f"sep2_pos={s2} (DUMMY = sep1_pos, {'OK' if s2 == s1 else 'BUG: should equal sep1_pos'})")
            print(f"is sep1_pos the last real token? {s1 == n_real - 1}")
        print(f"decoded up to sep1: {tokenizer.decode(ids[:s1+1])!r}")

    with torch.no_grad():
        out = model(**batch, return_variant_scores=True)

    print(f"\nlabels:       {batch['labels'].tolist()}")
    print(f"score_src:    {out.score_src.tolist()}")
    print(f"score_srcref: {None if out.score_srcref is None else out.score_srcref.tolist()}")
    print(f"combined:     {out.logits.view(-1).tolist()}")
    print(f"loss:         {out.loss.item():.6f}")

    # ── no-ref correctness assertions ──
    has_ref  = batch["has_ref"]
    combined = out.logits.view(-1)
    if (~has_ref).any():
        assert torch.equal(combined[~has_ref], out.score_src[~has_ref]), \
            "no-ref rows: combined score should be exactly score_src (no blending)."
        assert torch.isnan(out.score_srcref[~has_ref]).all(), \
            "no-ref rows: score_srcref should be NaN."
        print("OK: no-ref rows keep score_src untouched and report NaN for score_srcref.")
    if has_ref.any():
        assert not torch.isnan(out.score_srcref[has_ref]).any(), \
            "ref rows: score_srcref should not be NaN."
        w = model.config.inference_ref_weight
        expected = w * out.score_srcref[has_ref] + (1.0 - w) * out.score_src[has_ref]
        assert torch.allclose(combined[has_ref], expected, atol=1e-3), \
            "ref rows: combined != w*srcref + (1-w)*src."
        print("OK: ref rows are blended as expected.")

    # ---- training stress check: can the model overfit a tiny batch? ----
    print("\n=== TRAINING STRESS CHECK ===")
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    losses = []
    N_STEPS = 50
    print('N_SAMPLES', len(samples))
    print('N_STEPS', N_STEPS)

    for step in range(N_STEPS):
        optimizer.zero_grad()
        out = model(**batch, return_variant_scores=True)
        out.loss.backward()

        if step == 0:
            # gradient sanity: backbone + layerwise attention + FFN head all getting grads
            if model.layerwise_attention is not None:
                lwa_grad = model.layerwise_attention.alpha.grad
                print(f"  layerwise_attention.alpha.grad: "
                      f"{'None' if lwa_grad is None else f'{lwa_grad.norm().item():.6f}'}")
                assert lwa_grad is not None and lwa_grad.abs().sum() > 0, \
                    "layerwise_attention got no gradient!"
            else:
                print("  layerwise_attention: disabled, skipping grad check")
            backbone_param = next(model.backbone.parameters())
            ff_grad = model.final.weight.grad
            print(f"  backbone first-param.grad: "
                  f"{'None' if backbone_param.grad is None else f'{backbone_param.grad.norm().item():.6f}'}")
            print(f"  final.weight.grad: "
                  f"{'None' if ff_grad is None else f'{ff_grad.norm().item():.6f}'}")
            assert backbone_param.grad is not None and backbone_param.grad.abs().sum() > 0, \
                "backbone got no gradient!"
            assert ff_grad is not None and ff_grad.abs().sum() > 0, "FFN head got no gradient!"

        optimizer.step()
        losses.append(out.loss.item())
        if step % 5 == 0 or step == N_STEPS - 1:
            print(f"   step {step:2d}  loss={out.loss.item():.6f}")

    print(f"\nloss[0]={losses[0]:.6f}  loss[-1]={losses[-1]:.6f}")
    assert losses[-1] < losses[0], "loss did not decrease on a tiny overfit batch — something is broken."
    print("OK: model overfits a tiny batch and gradients flow through backbone + LWA + FFN.")

    # ---- LWA parameter diagnostic ----
    if model.layerwise_attention is not None:
        lwa = model.layerwise_attention
        with torch.no_grad():
            weights = lwa.transform_fn(lwa.alpha.float(), dim=0)
        top = torch.topk(weights, k=min(5, weights.numel()))
        print(f"\nLWA gamma = {lwa.gamma.item():.4f}  (init 1.0)")
        print(f"LWA layer weights: nonzero={int((weights > 1e-6).sum())}/{weights.numel()}, "
              f"max={weights.max().item():.4f}, entropy={-(weights.clamp(min=1e-12) * weights.clamp(min=1e-12).log()).sum().item():.4f}")
        print(f"  top-{top.indices.numel()} layers: "
              + ", ".join(f"L{i}={w:.4f}" for i, w in zip(top.indices.tolist(), top.values.tolist())))
        print(f"LWA alpha (raw, first 8): {[f'{v:.3e}' for v in lwa.alpha.float().tolist()[:8]]}")
        print(f"LWA alpha std={lwa.alpha.float().std().item():.3e}, "
              f"gamma={lwa.gamma.float().item():.8f}, dtype={lwa.alpha.dtype}")

        

    model.eval()
    with torch.no_grad():
        out = model(**batch, return_variant_scores=True)
    print(f"\nafter overfitting, scores vs labels:")
    print(f"labels:   {batch['labels'].tolist()}")
    print(f"combined: {out.logits.view(-1).tolist()}")
    

# ---- save/load roundtrip ----
    print("\n=== SAVE/LOAD CHECK ===")
    import shutil
    model.eval()
    with torch.no_grad():
        ref_out = model(**batch, return_variant_scores=True)

    tmp = "./_saveload_check"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        model.save_pretrained(tmp)
        reloaded = QwenXComet.from_pretrained(tmp)
        reloaded.eval()
        with torch.no_grad():
            new_out = reloaded(**batch, return_variant_scores=True)

        # what the model outputs
        ref, new = ref_out.logits.view(-1), new_out.logits.view(-1)
        print(f"before: {ref.tolist()}")
        print(f"after:  {new.tolist()}")
        assert torch.allclose(ref, new), "save/load changed the scores that the model outputs!"
        print("OK: save/load roundtrip preserves scores that the model outputs.")

        # real model parameters
        ref_sd, new_sd = model.state_dict(), reloaded.state_dict()
        assert ref_sd.keys() == new_sd.keys(), f"key mismatch: {ref_sd.keys() ^ new_sd.keys()}"
        for k in ref_sd:
            assert torch.equal(ref_sd[k].cpu(), new_sd[k].cpu()), f"parameter changed on reload: {k}"
        print(f"OK: all {len(ref_sd)} tensors identical after reload.")

        # model configs
        own = ["model_id", "dropout_p", "inference_ref_weight", "layer_dropout_p",
               "layer_norm", "layer_transformation", "use_layerwise_attention"]
        for k in own:
            assert getattr(reloaded.config, k) == getattr(model.config, k), f"config mismatch: {k}"
        print('OK: the relevant config fields are identical after save/load.')

        # model dtype
        print("dtypes:", next(model.parameters()).dtype, next(reloaded.parameters()).dtype)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[cleanup] removed {tmp}")
