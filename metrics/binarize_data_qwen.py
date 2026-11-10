"""
Binarizes MT quality judgement data for QwenXComet regression training
(Qwen3-Embedding backbone). It takes the CSV files listed in a YAML config
and produces a databin with tokenized data. Padding is handled later by the
training script.
Every stage of training needs its own databin, so there is one YAML file
per stage.
Tokenized data is saved to: ./databins/<model_id>/<save_path>/

NO chat template: Qwen3-Embedding is trained on plain text in the format
    "Instruct: {task}\nQuery: {text}" + EOS,
with the sentence representation pooled at the final EOS. The prompt is
therefore hardcoded below (it is part of the model, not an external choice).

Both separators are the EOS token:
    Instruct: ... Query: <src> <mt> <eos> <ref> <eos>
  sep1_pos = FIRST EOS  (attends only to src+mt  -> src-only score; bit-exact
             with encoding "<src> <mt> <eos>" alone, thanks to causality) 
             Added "manually" via the prompt PROMPT_TEMPLATE.
  sep2_pos = FINAL EOS  (attends to everything   -> src+ref score; this is the
             position Qwen3-Embedding was trained to pool at)
             Added by the tokenizer(text, add_special_tokens=True...

The YAML lists absolute CSV paths per split (environment variables such as
${REPO_ROOT} are expanded):
    train:
        - ${REPO_ROOT}/example_data/...csv
    dev_sent:
        - ...
Accepted splits: train, dev, dev_sent, dev_paraf. A "test" split is ignored.

The resulting databin contains, per row:
    input_ids        input token ids
    attention_mask   attention mask
    sep1_pos         index of the first EOS in input_ids
    sep2_pos         index of the final EOS in input_ids, or None if the row has no reference
    labels           the target score (--label_col)
and a metadata/ folder (arguments, tokenizer info, prompt, copy of the YAML),
which the training and inference scripts read.

Usage example (from the metrics/ folder):

export REPO_ROOT=$(realpath ..)
python binarize_data_qwen.py \
    --save_path stage1 \
    --files_yaml ../configs/data_configs/example_stage1_1.5.yaml \
    --model_id Qwen/Qwen3-Embedding-4B

"""


import argparse
import sys
import random
import pandas as pd
import shutil
from pathlib import Path
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset, Features, Sequence, Value
import json

# local import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.lang_names import get_full_names
from utils.train_utils import load_data_config




# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


# Prompt and EOS special token
PROMPT_TEMPLATE = (
"Instruct: Given a source text in {src_lang} and its machine translation "
"into {tgt_lang}, evaluate the quality of the translation.\n\n"
"Query: {src_lang} source: \n{source}\n\n"
"{tgt_lang} machine translation: \n{translation}{sep1}\n\n"
    "{tgt_lang} reference translation: \n{ref}"
)

EMBED_EOS = "<|endoftext|>"



def build_text(src, tgt, ref, src_lang, tgt_lang, eos):
    """Plain-text prompt (no chat template), defined in this script; separator is EOS."""
    if ref is None:
        noref_template, _, _ = PROMPT_TEMPLATE.partition("{sep1}")
        return noref_template.format(
            source=src, translation=tgt,
            src_lang=src_lang, tgt_lang=tgt_lang,
        )
    return PROMPT_TEMPLATE.format(
        source=src, translation=tgt, ref=ref,
        src_lang=src_lang, tgt_lang=tgt_lang,
        sep1=eos,
    )

def truncate_to_budget(tokenizer, texts, budget):
    """Truncates src/tgt/ref (None allowed) so that their total token count fits in budget.
    The budget is shared evenly; fields shorter than their share give the rest to the others."""
    lengths = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) if t is not None else 0 for t in texts]
    n = len(texts)
    remaining_budget = budget
    remaining_n = n
    caps = [None] * n
    for i in sorted(range(n), key=lambda i: lengths[i]):
        share = remaining_budget // remaining_n
        caps[i] = min(lengths[i], share)
        remaining_budget -= caps[i]
        remaining_n -= 1
    out = []
    for idx, (t, cap) in enumerate(zip(texts, caps)):
        if t is None:
            out.append(None)
            continue
        ids = tokenizer(t, add_special_tokens=False, truncation=True, max_length=cap)["input_ids"]
        if len(ids) < lengths[idx]:  # if it got truncated
            print(f"[truncate_to_budget] lengths={lengths}, caps={caps}, budget={budget}")
        out.append(tokenizer.decode(ids))
    return out

def encode_text(tokenizer, text, max_length, eos_id, has_ref):
    """Plain tokenization (no chat template). Both separators share the EOS id.

    ref rows:    sep1 = FIRST EOS (manual, after MT), sep2 = LAST EOS (appended).
    no-ref rows: exactly one EOS (appended), which is sep1; sep2_pos is None.

    Special tokens act as hard segmentation boundaries, so the token prefix up
    to and including sep1 is bit-identical between the two row types.
    """
    ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]

    n_eos = ids.count(eos_id)
    expected = 2 if has_ref else 1
    assert n_eos == expected, (
        f"expected {expected} EOS tokens, found {n_eos} — does src/tgt/ref contain"
        f"the literal {EMBED_EOS!r} string? sep1_pos/sep2_pos would be wrong.")

    sep1_pos = ids.index(eos_id)

    if not has_ref:
        assert sep1_pos == len(ids) - 1, (
            f"no-ref row: EOS is not the last token (pos {sep1_pos} of {len(ids)}).")
        return ids, [1] * len(ids), sep1_pos, None

    sep2_pos = len(ids) - 1 - ids[::-1].index(eos_id)   # last EOS (after REF)
    # same-position => the second EOS was truncated away
    assert sep2_pos > sep1_pos, (
        f"second EOS truncated out (len={len(ids)}): increase max_length or shorten the input.")
    # trained pooling position: the final EOS must be the very last token
    assert sep2_pos == len(ids) - 1, (
        f"final EOS is not the last token (pos {sep2_pos} of {len(ids)}): check truncation/template.")

    return ids, [1] * len(ids), sep1_pos, sep2_pos


# ---------------------------------------------------------------------------
# Batch creation
# ---------------------------------------------------------------------------

def create_xcomet_batch(tokenizer, examples, label_col, max_length, overhead):
    """
    For each row produces:
      input_ids        input token ids
      attention_mask   attention mask
      sep1_pos         index of the first EOS in input_ids
      sep2_pos         index of the final EOS in input_ids, or None if the row has no reference
      labels           the target score
    No padding — QwenXCometCollator handles dynamic per-batch padding.
    """

    out = {
        "input_ids": [], "attention_mask": [], "sep1_pos": [], "sep2_pos": [], "labels": []
    }

    for i in range(len(examples["src"])):
        src = examples["src"][i]
        tgt = examples["tgt"][i]
        ref = examples["ref"][i]
        src_lang = examples["src_lang"][i]
        tgt_lang = examples["tgt_lang"][i]

        ref = ref if isinstance(ref, str) and ref.strip() != "" else None

        # truncate the samples if necessary - this is BEFORE inserting the special tokens
        src, tgt, ref = truncate_to_budget(tokenizer, [src, tgt, ref], max_length-overhead)
        if ref is not None and ref.strip() == "":
            print("Ref truncated entirely, turning into no-ref row.")
            ref = None   # ref was truncated away entirely -> no-ref row

        text = build_text(src, tgt, ref, src_lang, tgt_lang, EMBED_EOS)

        ids_sr, mask_sr, sep1_pos, sep2_pos = encode_text(tokenizer, text, max_length, EMBED_EOS_ID, has_ref=(ref is not None))
        out["input_ids"].append(ids_sr)
        out["attention_mask"].append(mask_sr)
        out["sep1_pos"].append(sep1_pos)
        out["sep2_pos"].append(sep2_pos)

        lab = examples[label_col][i]
        if lab is None or lab != lab:  # lab != lab catches float nan
            raise ValueError(f"NaN/None label at index {i} in column '{label_col}'. "
        f"Fix the CSV before binarizing.")
        out["labels"].append(float(lab))

    return out


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def save_dataset_info(args, tokenizer):
    save_path = Path(args.save_path)
    info_dir = save_path / "metadata"
    info_dir.mkdir(parents=True, exist_ok=True)

    def make_serializable(obj):
        if isinstance(obj, Path): return str(obj)
        if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [make_serializable(v) for v in obj]
        return obj

    with open(info_dir / "binarize_args.json", "w") as f:
        json.dump(make_serializable(vars(args)), f, indent=4)
        
    tokenizer_info = {
        "model_id": args.model_id,
        "vocab_size": len(tokenizer),
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "chat_template": None,  # Qwen3-Embedding: plain text, no chat template
        "sep_token": EMBED_EOS,
        "sep_token_id": EMBED_EOS_ID,
        "tokenizer_eos_token": tokenizer.eos_token,      # for reference only
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
    }
    
    with open(info_dir / "tokenizer_info.json", "w") as f:
        json.dump(make_serializable(tokenizer_info), f, indent=4)

    # Save the hardcoded prompt so each databin stays self-describing
    with open(info_dir / "user_prompt.txt", "w") as f:
        f.write(PROMPT_TEMPLATE)

    try:
        shutil.copy(str(args.files_yaml), info_dir)
    except Exception as e:
        print(f"[Warning] Could not copy YAML: {e}")

    print(f"[Info] Dataset metadata saved to {info_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        print("[INFO] Tokenizer has no pad_token_id, setting it to eos_token_id")
        tokenizer.pad_token_id = tokenizer.eos_token_id

    global EMBED_EOS_ID
    EMBED_EOS_ID = tokenizer.convert_tokens_to_ids(EMBED_EOS)

    # No chat template needed: Qwen3-Embedding uses plain "Instruct/Query" text.
    
    # checks
    assert tokenizer(EMBED_EOS, add_special_tokens=False)["input_ids"] == [EMBED_EOS_ID], \
        "EMBED_EOS string does not round-trip to a single token id."  
    assert tokenizer("x", add_special_tokens=False)["input_ids"][-1] != EMBED_EOS_ID, \
        "Tokenizer auto-appends EOS even with add_special_tokens=False!"
    _probe = tokenizer("x", add_special_tokens=True)["input_ids"]
    assert _probe[-1] == EMBED_EOS_ID and _probe[0] != EMBED_EOS_ID and len(_probe) == 2, \
        f"add_special_tokens=True did not append exactly one EMBED_EOS and nothing else: {_probe}"
    
    print(f"Tokenizer loaded. pad_token_id={tokenizer.pad_token_id}, EMBED_EOS_ID={EMBED_EOS_ID}")
    if args.max_length is None:
        t_max = getattr(tokenizer, "model_max_length", MAX_LENGTH)
        args.max_length = t_max if t_max < 1_000_000 else MAX_LENGTH
    print(f"Using max_length: {args.max_length}")

    # --- Prompt is hardcoded (PROMPT_TEMPLATE) — no external files. ---
    print("PROMPT_TEMPLATE (hardcoded, Qwen3-Embedding trained format):")
    print(PROMPT_TEMPLATE)

    # Overhead = tokens of the prompt added
    # src/tgt/ref budget = max_length - overhead.
    min_prompt = tokenizer(
        build_text("", "", "", "src_lang", "tgt_lang", EMBED_EOS),
        add_special_tokens=True)["input_ids"]

    args.overhead = len(min_prompt)
    args.overhead += 10 # safety margin

    if args.keep_frac:
        assert 0 < args.keep_frac <= 1, "Fraction must be between 0 and 1."

    
    # --- YAML ---
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    files = load_data_config(args.files_yaml)
    print(f"[Info] Loaded YAML: {args.files_yaml}")
    print(f"[Info] Number of files per split: { {s: len(v) for s, v in files.items()} }")
    
    # --- Load CSVs ---
    dataset_dict = {}
    for split_name, file_list in files.items():
        if split_name == "test":
            continue  # test data is only used after training
        if split_name not in ["train", "dev", "dev_sent", "dev_paraf"]:
            print(f'[Warning] Unknown split "{split_name}", skipping...')
            continue

        dfs = []
        for path in file_list:
            df = pd.read_csv(path)
            total = len(df)
            # empty target is a valid datapoint, just make sure it's a string and not nan
            df["tgt"] = df["tgt"].fillna("")

            # source cannot be empty
            n_bad_src = df["src"].isna().sum() + (~df["src"].apply(lambda x: isinstance(x, str))).sum() 
            if n_bad_src:
                raise ValueError(f"{Path(path).name}: {n_bad_src} bad src (NaN or non-string). Fix the CSV before binarizing.")

            if "ref" not in df.columns: df["ref"] = None
            n_noref = int(df["ref"].isna().sum())
                
            if args.keep_frac < 1.0:
                if "dev" not in split_name:
                    print(f'Sampling {Path(path).stem} to {args.keep_frac*100}% of data.')
                    df = df.sample(frac=args.keep_frac, random_state=args.seed).reset_index(drop=True)

            name = Path(path).stem
            src_full, tgt_full = get_full_names(name) # full language names for the prompt
            df["src_lang"] = src_full
            df["tgt_lang"] = tgt_full
            if len(df) > 0:
                dfs.append(df)

        if not dfs:
            print(f'[Warning] No rows for split "{split_name}", skipping.')
            continue

        dataset_dict[split_name] = Dataset.from_pandas(
            pd.concat(dfs, ignore_index=True)
        )

    dataset = DatasetDict(dataset_dict)
    print(f"Loaded splits: {list(dataset.keys())}, "
          f"sample counts: { {s: len(dataset[s]) for s in dataset} }")

    # --- Tokenize ---
    print(f"Using label column: '{args.label_col}'")
    
    def preprocess(examples):
        return create_xcomet_batch(
            tokenizer, examples,
            args.label_col, args.max_length, args.overhead
        )
        
    
    TOKENIZED_FEATURES = Features({
    "input_ids":         Sequence(Value("int32")),
    "attention_mask":    Sequence(Value("int8")),
    "sep1_pos":          Value("int32"),
    "sep2_pos":          Value("int32"),
    "labels":            Value("float32"),
    })
    
    
    # Remove columns per-split (different CSV sources have different columns;
    # using one split's column list for all would crash on mismatch).
    print(f"[Info] Starting tokenization.")
    tokenized_splits = {}
    for split_name, split_ds in dataset.items():
        tokenized_splits[split_name] = split_ds.map(
            preprocess,
            batched=True,
            batch_size=args.batch_size,
            remove_columns=split_ds.column_names,
            load_from_cache_file=False,
            features=TOKENIZED_FEATURES,
        )
    tokenized_ds = DatasetDict(tokenized_splits)
    # Save
    tokenized_ds.save_to_disk(save_path)
    args.sep1 = args.sep2 = EMBED_EOS
    save_dataset_info(args, tokenizer)
    print(f"[Info] Dataset saved to {save_path}")
    
    # --- Sanity checks ---
    print('Sanity check')

    # --- Metadata round-trip check: does the saved metadata describe the data? ---
    meta_dir = save_path / "metadata"
    saved_tok  = json.loads((meta_dir / "tokenizer_info.json").read_text())
    saved_args = json.loads((meta_dir / "binarize_args.json").read_text())
    saved_prompt = (meta_dir / "user_prompt.txt").read_text()

    assert saved_prompt == PROMPT_TEMPLATE, "user_prompt.txt does not match the PROMPT_TEMPLATE used to build the data."
    assert saved_tok["sep_token_id"] == EMBED_EOS_ID, "tokenizer_info sep_token_id != the id used in encode_text."
    assert tokenizer.convert_tokens_to_ids(saved_tok["sep_token"]) == EMBED_EOS_ID, "sep_token string does not resolve to sep_token_id."
    sep_id = saved_tok["sep_token_id"]

    print(f"[Verify] metadata sep_token={saved_tok['sep_token']!r} id={sep_id}, "
          f"max_length={saved_args['max_length']}, overhead={saved_args['overhead']}")

    for split in tokenized_ds:
        ds_split = tokenized_ds[split]
        n_check = min(200, len(ds_split))
        step = max(1, len(ds_split) // n_check)
        longest = 0
        for j in range(0, len(ds_split), step):
            row = ds_split[j]
            ids, s1, s2 = row["input_ids"], row["sep1_pos"], row["sep2_pos"]
            longest = max(longest, len(ids))
            assert ids[s1] == sep_id, f"{split}[{j}]: token at sep1_pos={s1} is {ids[s1]}, not {sep_id}."
            if s2 is None:
                assert ids.count(sep_id) == 1 and s1 == len(ids) - 1, \
                    f"{split}[{j}]: no-ref row is malformed (sep count={ids.count(sep_id)}, s1={s1}, len={len(ids)})."
            else:
                assert ids[s2] == sep_id, f"{split}[{j}]: token at sep2_pos={s2} is {ids[s2]}, not {sep_id}."
                assert s2 == len(ids) - 1 and s2 > s1, \
                    f"{split}[{j}]: ref row seps out of place (s1={s1}, s2={s2}, len={len(ids)})."
        assert longest <= saved_args["max_length"], \
            f"{split}: longest sequence {longest} exceeds recorded max_length {saved_args['max_length']}."
        print(f"[Verify] '{split}': {n_check} rows sampled, sep positions consistent with metadata, longest={longest}.")

    for split in tokenized_ds:
        labels   = tokenized_ds[split]["labels"]
        sep2_pos = tokenized_ds[split]["sep2_pos"]   # None => no-ref row
        nan_count   = sum(1 for x in labels if x != x)
        range_count = sum(1 for x in labels if x == x and not (0.0 <= x <= 1.0))
        assert nan_count == 0, \
            f"[Error] {nan_count} NaN labels in '{split}' (col '{args.label_col}')."
        assert range_count == 0, \
            f"[Error] {range_count} labels outside [0,1] in '{split}'."
        n_ref = sum(1 for x in sep2_pos if x is not None)
        lo, hi = min(labels), max(labels)
        print(f"[Info] Split '{split}': {len(labels)} samples, "
              f"{n_ref} with ref / {len(labels) - n_ref} without, "
              f"label range [{lo:.4f}, {hi:.4f}]. OK.")

    # --- Print a decoded sample ---
    first_split = next(iter(tokenized_ds))
    i = random.randint(0, len(tokenized_ds[first_split]) - 1)
    sample = tokenized_ds[first_split][i]
    print(f"\nExample, sample {i} {first_split} split")
    print("sample[input_ids]", sample["input_ids"])
    print("detok", tokenizer.decode(sample["input_ids"]))
    print('sep1 position', sample['sep1_pos'])
    print('sep2 position', sample['sep2_pos'])
    print('Label:', sample['labels'])



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Defaults
    LABEL_COL = 'label_01' # label_01 contains the score normalized to a 0-1 scale.
    MODEL_ID = "Qwen/Qwen3-Embedding-4B"
    MAX_LENGTH = 8192
    SEED = 42

    parser = argparse.ArgumentParser(
        description="Binarize MT quality judgement data for QwenXComet (plain-text Instruct/Query encoding, EOS separators)."
    )
    parser.add_argument("--save_path", required=True, help="Databin name, saved to ./databins/<model_id>/<save_path>/")
    parser.add_argument("--files_yaml", required=True, help="YAML with absolute paths to CSV files with data to use.")
    parser.add_argument("--model_id", default=MODEL_ID, help=f"Model to use for the tokenization, default: {MODEL_ID}")
    parser.add_argument("--label_col", default=LABEL_COL, help=f"CSV column with gold labels used as regression target (default: {LABEL_COL}).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help=f"Max tokens per encoded sequence (default: {MAX_LENGTH}).")
    parser.add_argument("--keep_frac", type=float, default=1.0, help="Keep this fraction of each training file (per-file random subsample). Dev splits are kept whole.")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Default: {SEED}")
    
    args = parser.parse_args()
    args.save_path = str(
        Path("./databins") / args.model_id.replace("/", "_") / args.save_path # keeps the name of the model in the path to ensure correct tokenizer usage
    )
    print(f"[Info] Starting binarization script")
    main(args)
    print("[Info] Binarization completed.")
