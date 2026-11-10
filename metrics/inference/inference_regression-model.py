"""
Scores machine translations with a trained GemmaItXComet or QwenXComet model.

Input is either a single CSV (--csv, with --src_lang/--tgt_lang) or a CSV
manifest (--csv_manifest, columns: csv,src_lang,tgt_lang) to score several
files without reloading the model. Input CSVs need the columns src, tgt and
optionally ref; rows without a reference are scored from source + translation.

The tokenization settings (prompt, separators, max_length) are read from the
metadata saved with the checkpoint, so inputs are encoded exactly as in training.

Output columns: <score_column>_combined, _src and _srcref (+ _clamped and
_in_range with --clamp).

Example usage (single CSV, from the metrics/ folder):

python inference/inference_regression-model.py \
    --archi GemmaItXComet \
    --ckpt path/to/checkpoint \
    --csv path/to/data.csv \
    --src_lang en --tgt_lang cs \
    --output scores.csv \
    --clamp

Example usage (CSV manifest, contains paths of each CSV to score, plus src_lang and tgt_lang):

python inference/inference_regression-model.py \
    --archi QwenXComet \
    --ckpt path/to/checkpoint \
    --csv_manifest inputs.csv \
    --output_dir results/ \
    --clamp
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import torch
from transformers import AutoTokenizer
import json
from pathlib import Path
import random
import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.lang_names import get_full_names


ARCHIS = ["GemmaItXComet", "QwenXComet"]


def load_training_config(ckpt_path, databin_meta=None):
    """
    Recover binarization params. Prefers the copy stored alongside the
    checkpoint (<ckpt>/../metainfo/databin_metadata_*/), falling back to the
    databin recorded in training_metadata.json.

    databin_meta: optional explicit dir holding binarize_args.json /
    user_prompt.txt / system_prompt.txt. Use it when the databin recorded in
    training_metadata.json was deleted; the params must be identical to the
    checkpoint's actual databin.
    """
    ckpt_path = Path(ckpt_path)
    # metainfo/ can live INSIDE the checkpoint dir (final model saved at the
    # output_dir root) or in its PARENT (checkpoint-N subdir). Prefer the inner one.
    if (ckpt_path / "metainfo" / "training_metadata.json").exists():
        meta_dir = ckpt_path / "metainfo"
    else:
        meta_dir = ckpt_path.parent / "metainfo"

    train_meta = json.loads((meta_dir / "training_metadata.json").read_text())
    sargs = train_meta["script_arguments"]

    if databin_meta is not None:
        meta_dir = Path(databin_meta)
        print(f"[config] using explicit --databin_meta {meta_dir}")
        if not (meta_dir / "binarize_args.json").exists():
            raise RuntimeError(f"--databin_meta {meta_dir} has no binarize_args.json")
    else:
        databin_dirs = list(meta_dir.glob("databin_metadata_*"))
        if len(databin_dirs) > 1:
            raise RuntimeError(f"Multiple databin_metadata_* dirs found in {meta_dir}: {databin_dirs}")
        local_meta_dir = databin_dirs[0] if databin_dirs else meta_dir

        if (local_meta_dir / "binarize_args.json").exists():
            meta_dir = local_meta_dir
        else:
            meta_dir = Path(sargs["saved_ds_path"]) / "metadata"
            print(f"[config] no local binarize_args.json, falling back to {meta_dir}")

    binarize_args = json.loads((meta_dir / "binarize_args.json").read_text())
    cfg = {"max_length": binarize_args["max_length"],
           "overhead": binarize_args.get("overhead")}

    if "sep1" in binarize_args:
        cfg["sep1"] = binarize_args["sep1"]
        cfg["sep2"] = binarize_args["sep2"]
        cfg["overhead"] = binarize_args["overhead"]
        cfg["user_prompt_template"] = (meta_dir / "user_prompt.txt").read_text()
        sys_path = meta_dir / "system_prompt.txt"
        cfg["system_prompt"] = sys_path.read_text() if sys_path.exists() else None
    return cfg


def _prepare(df, max_rows):
    if "ref" not in df.columns:
        df["ref"] = ""
    # has_ref computed from the ORIGINAL ref, before fillna empties it
    df["has_ref"] = df["ref"].notna() & df["ref"].astype(str).str.strip().ne("")
    df = df.reset_index(drop=True)
    df["id"] = range(len(df))             # stable original-order id
    for c in ("src", "tgt", "ref"):
        df[c] = df[c].fillna("")

    df = df.head(max_rows)  # max_rows=None keeps all rows

    n = len(df)
    n_ref = int(df["has_ref"].sum())
    print(f"[data] prepared {n} rows | {n_ref} with ref | {n - n_ref} without ref")
    assert list(df["id"]) == list(range(n)), "id is not contiguous 0..n-1"
    return df


def load_csv(csv_path, src_lang, tgt_lang, max_rows=None):
    df = _prepare(pd.read_csv(csv_path), max_rows)
    if "src_lang" not in df.columns or "tgt_lang" not in df.columns:
        src_lang, tgt_lang = get_full_names(src_lang + '-' + tgt_lang)
        df["src_lang"] = src_lang
        df["tgt_lang"] = tgt_lang
    return df


def load_model(archi, ckpt, device):
    """Load model + tokenizer + collator for the chosen regression model.
    Returns (model, tokenizer, collator). Imports are lazy per branch."""
    if archi == "GemmaItXComet":
        from modules.GemmaItXComet import GemmaItXComet, GemmaItXCometCollator
        model = GemmaItXComet.from_pretrained(ckpt).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        collator = GemmaItXCometCollator(tokenizer)
        return model, tokenizer, collator

    elif archi == "QwenXComet":
        from modules.QwenXComet import QwenXComet, QwenXCometCollator
        # bf16 to match training numerics (backbone + estimator head were both bf16 during training)
        model = QwenXComet.from_pretrained(ckpt, torch_dtype=torch.bfloat16).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        collator = QwenXCometCollator(tokenizer)
        return model, tokenizer, collator

    else:
        raise ValueError(f"Unknown model: {archi}")


def build_inference_ds(archi, df, tokenizer, max_length, cfg):
    ds = Dataset.from_pandas(df, preserve_index=False)

    if archi == "GemmaItXComet":
        from binarize_data_xcomet import build_messages, encode_messages, truncate_to_budget
        sep1, sep2 = cfg["sep1"], cfg["sep2"]
        sep1_id = tokenizer.convert_tokens_to_ids(sep1)
        sep2_id = tokenizer.convert_tokens_to_ids(sep2)

        def preprocess(ex):
            out = {"id": [], "input_ids": [], "attention_mask": [], "sep1_pos": [], "sep2_pos": []}
            for i in range(len(ex["src"])):
                src, tgt, ref = truncate_to_budget(
                    tokenizer, [ex["src"][i], ex["tgt"][i], ex["ref"][i]],
                    max_length - cfg["overhead"]
                )
                has_ref = ex["has_ref"][i]
                msgs = build_messages(cfg["user_prompt_template"], cfg["system_prompt"],
                                      src, tgt, ref,
                                      ex["src_lang"][i], ex["tgt_lang"][i], sep1, sep2)
                ids, mask, s1, s2 = encode_messages(tokenizer, msgs, max_length, sep1_id, sep2_id, has_ref)
                if not has_ref:
                    s2 = s1  # no real ref: both scores will be computed from first separator

                out["id"].append(ex["id"][i]); out["input_ids"].append(ids)
                out["attention_mask"].append(mask); out["sep1_pos"].append(s1); out["sep2_pos"].append(s2)
            return out

        return ds.map(preprocess, batched=True, remove_columns=ds.column_names, load_from_cache_file=False)

    elif archi == "QwenXComet":
        # Mirror binarize_data_qwen.create_xcomet_batch exactly (minus labels)
        # so inference encoding is identical to training. no-ref rows keep
        # sep2_pos = None; QwenXCometCollator derives has_ref from that.
        from datasets import Features, Sequence, Value
        from binarize_data_qwen import build_text, truncate_to_budget, encode_text, EMBED_EOS
        eos_id = tokenizer.convert_tokens_to_ids(EMBED_EOS)
        overhead = cfg["overhead"]

        def preprocess(ex):
            out = {"id": [], "input_ids": [], "attention_mask": [], "sep1_pos": [], "sep2_pos": []}
            for i in range(len(ex["src"])):
                src, tgt, ref = ex["src"][i], ex["tgt"][i], ex["ref"][i]
                ref = ref if isinstance(ref, str) and ref.strip() != "" else None
                src, tgt, ref = truncate_to_budget(tokenizer, [src, tgt, ref], max_length - overhead)
                if ref is not None and ref.strip() == "":
                    ref = None  # ref truncated away entirely -> no-ref row
                text = build_text(src, tgt, ref, ex["src_lang"][i], ex["tgt_lang"][i], EMBED_EOS)
                ids, mask, s1, s2 = encode_text(tokenizer, text, max_length, eos_id,
                                                has_ref=(ref is not None))
                out["id"].append(ex["id"][i])
                out["input_ids"].append(ids)
                out["attention_mask"].append(mask)
                out["sep1_pos"].append(s1)
                out["sep2_pos"].append(s2)
            return out

        feats = Features({
            "id":             Value("int32"),
            "input_ids":      Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
            "sep1_pos":       Value("int32"),
            "sep2_pos":       Value("int32"),   # nullable: None for no-ref rows
        })
        return ds.map(preprocess, batched=True, remove_columns=ds.column_names,
                      load_from_cache_file=False, features=feats)

    else:
        raise ValueError(f"Unknown model: {archi}")


def score_dataframe(df, args, model, tokenizer, collator, cfg, max_length, device):
    df_to_score = df[df["has_ref"]] if args.skip_noref else df
    if args.skip_noref:
        print(f"[score] --skip_noref ON: scoring {len(df_to_score)}/{len(df)} rows")
    ds = build_inference_ds(args.archi, df_to_score, tokenizer, max_length, cfg)

    print('[info] Example of input to the model (detokenized):\n', tokenizer.decode(ds["input_ids"][0], skip_special_tokens=False))

    scores, scores_src, scores_srcref = run_inference(model, ds, collator, device, args.batch_size)
    assert len(scores) == len(df_to_score), f"{len(scores)} scores for {len(df_to_score)} rows — misalignment!"

    df[f"{args.score_column}_combined"] = df["id"].map(dict(zip(ds["id"], scores)))
    df[f"{args.score_column}_src"] = df["id"].map(dict(zip(ds["id"], scores_src)))
    df[f"{args.score_column}_srcref"] = df["id"].map(dict(zip(ds["id"], scores_srcref)))
    df.loc[~df["has_ref"], f"{args.score_column}_srcref"] = float("nan")   # no srcref score without a ref

    if args.clamp:
        for v in ("combined", "src", "srcref"):
            df = clamp_scores(df, score_col=f"{args.score_column}_{v}")

    return df


def run_inference(model, ds, collator, device, batch_size):
    """Score every row, returning lists aligned with dataset (= CSV) order."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)  # shuffle=False keeps order
    scores, scores_src, scores_srcref = [], [], []
    with torch.no_grad():
        for batch in loader:
            print('.', end='')
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, return_variant_scores=True)
            scores.extend(out.logits.view(-1).float().cpu().tolist())
            scores_src.extend(out.score_src.view(-1).float().cpu().tolist())
            scores_srcref.extend(out.score_srcref.view(-1).float().cpu().tolist())
    print()
    return scores, scores_src, scores_srcref


def clamp_scores(df, score_col="score"):
    """Add a clamped column and an in-range flag for a score column."""
    clamped_col = f"{score_col}_clamped"
    df[clamped_col] = df[score_col].clip(0, 1)
    # Suffixed per score_col, so the flags of different variants/checkpoints
    # merged into one CSV don't overwrite each other.
    df[f"{score_col}_in_range"] = (df[score_col] == df[clamped_col]).where(df[score_col].notna())

    n_out = int((df[score_col].notna() & (df[score_col] != df[clamped_col])).sum())
    n_scored = int(df[score_col].notna().sum())
    print(f"[clamp] {n_out} scores out of [0,1] range (clamped) from {n_scored} scored rows ({len(df)} total)")
    return df


def check_trained_weights(model, n=3):
    """Print a few backbone and head weights, to confirm the trained checkpoint was loaded."""
    print('Some backbone parameters:')
    for name, p in random.sample(list(model.backbone.named_parameters()), n):
        print(name, p.detach().to(torch.float32))

    print("Some head parameters:")
    named = list(model.ff1.named_parameters())
    if model.layerwise_attention is not None:
        named += list(model.layerwise_attention.named_parameters())
    for name, p in random.sample(named, min(n, len(named))):
        print(name, p.detach().float().flatten()[:5])


def parse_args():
    p = argparse.ArgumentParser(
        description="Score a CSV (columns: src, tgt, optionally ref) with a trained GemmaItXComet or QwenXComet model.")

    p.add_argument("--archi", required=True, choices=ARCHIS, help="Which model architecture to use.")
    p.add_argument("--ckpt", required=True, help="Path to the trained checkpoint directory.")
    p.add_argument("--output", default="inference_results.csv", help="Output CSV path (with --csv).")
    p.add_argument("--score_column", default=None,
                   help="Prefix of the score columns. Default: derived from --ckpt "
                        "(path under models_trained/, '/' -> '_'), or 'score'.")

    # single CSV input
    p.add_argument("--csv", help="Input CSV with columns: src, tgt, ref (optional).")
    p.add_argument("--src_lang", help="Source language code (required with --csv).")
    p.add_argument("--tgt_lang", help="Target language code (required with --csv).")

    # manifest input: several CSVs
    p.add_argument("--csv_manifest", help="CSV with columns: csv,src_lang,tgt_lang — run inference on each row's file sequentially.")
    p.add_argument("--output_dir", default=".", help="Where to write per-file outputs when using --csv_manifest.")
    p.add_argument("--databin_meta", default=None,
                   help="Explicit dir holding binarize_args.json / user_prompt.txt / "
                        "system_prompt.txt. Use when the databin recorded in the checkpoint's "
                        "training_metadata.json was deleted; params must be identical to "
                        "the checkpoint's actual databin.")
    p.add_argument("--max-rows", type=int, default=None, help="Only score the first N rows of each CSV.")

    # how to handle missing references
    p.add_argument("--skip_noref", action="store_true", help="Do NOT score rows without a reference; their scores are left as NaN. Default: score them from source + translation.")
    p.add_argument('--clamp', action='store_true', help="Also write scores clamped to [0, 1].")

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=None,
                   help="Override the max sequence length. Defaults to the value from training metadata.")
    p.add_argument("--device", default=None, help="cuda / cpu. Auto-detected if not set.")
    p.add_argument("--check-weights", action="store_true",
                   help="Print a few model weights to confirm the checkpoint loaded.")
    return p.parse_args()


def infer_score_column(ckpt):
    """Derive the score-column name from a checkpoint path: everything under
    models_trained/, with '/' -> '_'. E.g.
    .../models_trained/GemmaItXComet/stage2/checkpoint-350 -> GemmaItXComet_stage2_checkpoint-350
    Falls back to "score" if the path has no models_trained/ folder."""
    parts = Path(ckpt).resolve().parts
    if "models_trained" not in parts:
        return "score"
    return "_".join(parts[parts.index("models_trained") + 1:])


def score_columns(args):
    """The exact set of columns score_dataframe() adds for this run --
    used both to check whether a checkpoint's columns are already present
    in an existing scored CSV (skip) and to pull the new columns out for
    merging (see the --csv_manifest loop in main())."""
    variants = ["combined", "src", "srcref"]
    cols = [f"{args.score_column}_{v}" for v in variants]
    if args.clamp:
        cols += [f"{args.score_column}_{v}_clamped" for v in variants]
        cols += [f"{args.score_column}_{v}_in_range" for v in variants]
    return cols


def main():
    args = parse_args()
    if bool(args.csv) == bool(args.csv_manifest):
        raise SystemExit("Provide exactly one of: --csv or --csv_manifest.")
    if args.csv and not (args.src_lang and args.tgt_lang):
        raise SystemExit("--src_lang and --tgt_lang are required with --csv.")

    if args.score_column is None:
        args.score_column = infer_score_column(args.ckpt)
    print(f"[score_column] {args.score_column}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, collator = load_model(args.archi, args.ckpt, device)
    cfg = load_training_config(args.ckpt, args.databin_meta)
    max_length = args.max_length or cfg["max_length"]
    print('Loaded config:', cfg)

    if args.check_weights:
        check_trained_weights(model)

    if args.csv_manifest:
        manifest = pd.read_csv(args.csv_manifest)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        # Each input file gets one <stem>_scored.csv in --output_dir; scoring it
        # with another checkpoint merges the new score columns into that file.
        score_cols = score_columns(args)
        for _, row in manifest.iterrows():
            out_path = Path(args.output_dir) / f"{Path(row['csv']).stem}_scored.csv"
            existing = pd.read_csv(out_path) if out_path.exists() else None
            if existing is not None and all(c in existing.columns for c in score_cols):
                print(f'Skipping {out_path} — {args.score_column} columns already present')
                continue
            df = load_csv(row["csv"], row["src_lang"], row["tgt_lang"], args.max_rows)
            df = score_dataframe(df, args, model, tokenizer, collator, cfg, max_length, device)
            if existing is not None:
                assert len(existing) == len(df), (
                    f"{out_path}: row count mismatch ({len(existing)} vs {len(df)}) -- "
                    f"can't merge, source CSV or --max_rows must differ from the earlier run.")
                for c in score_cols:
                    existing[c] = df[c].values
                existing.to_csv(out_path, index=False)
                print(f"[done] merged {len(score_cols)} column(s) ({args.score_column}) into {out_path}")
            else:
                df.to_csv(out_path, index=False)
                print(f"[done] wrote {len(df)} rows to {out_path}")
    else:
        df = load_csv(args.csv, args.src_lang, args.tgt_lang, args.max_rows)
        df = score_dataframe(df, args, model, tokenizer, collator, cfg, max_length, device)
        df.to_csv(args.output, index=False)
        print(f"[done] wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
