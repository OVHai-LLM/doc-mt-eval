"""
Binarizes MT quality judgement data for GemmaItXComet regression training.
It takes the CSV files listed in a YAML config and produces a databin with 
tokenized data. Padding is handled later by the training script. 
Every stage of training needs its own databin, so there is one YAML
file per stage. 
Tokenized data is saved to: ./databins/<model_id>/<save_path>/


The prompt template is read from --user_prompt and must contain {source},
{translation}, {ref}, {sep1} and {sep2}. The separators are single special
tokens, <unused0> and <unused1> by default (override with --sep1 / --sep2).
For rows without a reference, the template is cut at {sep1}, so the
reference section is dropped entirely.
A system prompt is optional (--system_prompt). Gemma's chat template has no
native system role, so we omit it here.


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
    sep1_pos         index of sep1 in input_ids
    sep2_pos         index of sep2 in input_ids, or None if the row has no reference
    labels           the target score (--label_col)
and a metadata/ folder (arguments, tokenizer info, prompt, copy of the YAML),
which the training and inference scripts read.

Usage example (from the metrics/ folder):

export REPO_ROOT=$(realpath ..)
python binarize_data_xcomet.py \
      --save_path stage1 \
      --files_yaml ../configs/data_configs/example_stage1_1.5.yaml \
      --user_prompt ../configs/prompts/xcomet_user_prompt.txt

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

def build_messages(user_prompt_template, system_prompt,
                   src, tgt, ref, src_lang, tgt_lang, sep1, sep2):
    """
    Builds the messages list for apply_chat_template.
    system_prompt is optional (None = skip — correct for Gemma).
    """
    if ref is None:
        noref_template, _, _ = user_prompt_template.partition("{sep1}")
        user_content = noref_template.format(
            source=src, 
            translation=tgt, 
            src_lang=src_lang, 
            tgt_lang=tgt_lang
        ) + sep1
    else:
        user_content = user_prompt_template.format(
            source=src,
            translation=tgt,
            ref=ref,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            sep1=sep1,
            sep2=sep2,
        )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages

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

def encode_messages(tokenizer, messages, max_length, sep1_id, sep2_id, has_ref):
    """Applies the chat template and tokenizes.
    Returns (input_ids, attention_mask, sep1_pos, sep2_pos); sep2_pos is None if has_ref is False."""
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=max_length,
        return_dict=False,
    )
    try:
        sep1_pos = ids.index(sep1_id)
    except ValueError:
        print(messages); print(ids)
        raise ValueError("sep1 was truncated out of the sequence...")

    sep2_pos = None
    if has_ref:
        try:
            sep2_pos = ids.index(sep2_id)
        except ValueError:
            print(messages); print(ids)
            raise ValueError("sep2 was truncated out of the sequence...")

    return ids, [1] * len(ids), sep1_pos, sep2_pos
    


# ---------------------------------------------------------------------------
# Batch creation
# ---------------------------------------------------------------------------

def create_xcomet_batch(tokenizer, examples, user_prompt_template, system_prompt, label_col, max_length, sep1, sep2, sep1_id, sep2_id, overhead):
    """
    For each row produces:
        input_ids        input token ids
        attention_mask   attention mask
        sep1_pos         index of sep1 in input_ids
        sep2_pos         index of sep2 in input_ids, or None if the row has no reference
        labels           the target score

    No padding — GemmaItXCometCollator handles dynamic per-batch padding.
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
        
        msgs_sr = build_messages(user_prompt_template, system_prompt,
                                 src, tgt, ref, src_lang, tgt_lang,  sep1, sep2)
        
        ids_sr, mask_sr, sep1_pos, sep2_pos = encode_messages(tokenizer, msgs_sr, max_length, sep1_id, sep2_id, has_ref=(ref is not None))
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

def save_dataset_info(args, tokenizer, user_prompt_template, system_prompt):
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
        "chat_template": tokenizer.chat_template,
    }
    with open(info_dir / "tokenizer_info.json", "w") as f:
        json.dump(make_serializable(tokenizer_info), f, indent=4)

    # Save resolved prompt text so each databin is self-describing
    with open(info_dir / "user_prompt.txt", "w") as f:
        f.write(user_prompt_template)
    if system_prompt:
        with open(info_dir / "system_prompt.txt", "w") as f:
            f.write(system_prompt)

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
    assert tokenizer.chat_template is not None, (
        "Tokenizer has no chat template — use an instruct model (e.g. gemma-3-1b-it)."
    )
    print(f"Tokenizer loaded. pad_token_id={tokenizer.pad_token_id}")

    if args.max_length is None:
        t_max = getattr(tokenizer, "model_max_length", 8192)
        args.max_length = t_max if t_max < 1_000_000 else 8192
    print(f"Using max_length: {args.max_length}")

    # --- Load prompts ---
    with open(args.user_prompt) as f:
        user_prompt_template = f.read()
    print("user_prompt_template used:")
    print(user_prompt_template)

    required = ["{sep1}", "{sep2}", "{source}", "{translation}", "{ref}"]
    assert all(p in user_prompt_template for p in required), f"User prompt must contain {required} placeholders."

    
    system_prompt = None
    if args.system_prompt:
        with open(args.system_prompt) as f:
            system_prompt = f.read()
        print("[Info] System prompt loaded.")
    else:
        print("[Info] No system prompt: correct if using Gemma.")

    min_chat_template = tokenizer.apply_chat_template(
    build_messages(user_prompt_template, system_prompt, "", "", "", "src_lang", "tgt_lang", args.sep1, args.sep2),
    tokenize=True, add_generation_prompt=False, return_dict=False,
    )

    
    args.overhead = len(min_chat_template)
    args.overhead += 10

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
            continue  # test is evaluated post-training on raw CSVs; skip silently
        if split_name not in ["train", "dev", "dev_sent", "dev_paraf"]:
            print(f'[Warning] Unknown split "{split_name}", skipping...')
            continue
        print('Split', split_name)
        dfs = []
        for path in file_list:
            df = pd.read_csv(path)
            # empty target is a valid datapoint, just make sure it's a string and not nan
            df["tgt"] = df["tgt"].fillna("")

            # source cannot be empty
            n_bad_src = df["src"].isna().sum() + (~df["src"].apply(lambda x: isinstance(x, str))).sum() 
            if n_bad_src:
                raise ValueError(f"{Path(path).name}: {n_bad_src} bad src (NaN or non-string). Fix the CSV before binarizing.")

            # Per-file random subsample of training files; dev splits are kept whole.
            if "dev" not in split_name and args.keep_frac < 1.0:
                print(f'Sampling {Path(path).stem} to {args.keep_frac*100:.1f}% of data.')
                df = df.sample(frac=args.keep_frac, random_state=args.seed).reset_index(drop=True)

            name = Path(path).stem
            src_full, tgt_full = get_full_names(name) # full language name for the prompt
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
    
    sep1_id = tokenizer.convert_tokens_to_ids(args.sep1)
    sep2_id = tokenizer.convert_tokens_to_ids(args.sep2)

    
    for name, sep, sid in [("sep1", args.sep1, sep1_id), ("sep2", args.sep2, sep2_id)]:
        assert sid != tokenizer.unk_token_id, f"{name}={sep!r} is not a recognized single token for this tokenizer."
        assert tokenizer.encode(sep, add_special_tokens=False) == [sid], f"{name}={sep!r} is not kept as a single token."
    
    def preprocess(examples):
        return create_xcomet_batch(
            tokenizer, examples,
            user_prompt_template, system_prompt,
            args.label_col, args.max_length, args.sep1, args.sep2, sep1_id, sep2_id, args.overhead
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
    save_dataset_info(args, tokenizer, user_prompt_template, system_prompt)
    print(f"[Info] Dataset saved to {save_path}")
    
    # --- Sanity checks ---
    print('Sanity checks:')
    for split in tokenized_ds:
        labels  = tokenized_ds[split]["labels"]
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
    print("sample[input_ids]:", sample["input_ids"])
    print("detokenized:", tokenizer.decode(sample["input_ids"]))
    print('sep1 position:', sample['sep1_pos'])
    print('sep2 position:', sample['sep2_pos'])
    print('Label:', sample['labels'])



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    LABEL_COL = 'label_01'
    MODEL_ID = "google/gemma-3-1b-it"
    SEED = 42
    SEP1 = "<unused0>"
    SEP2 = "<unused1>"
    parser = argparse.ArgumentParser(
        description="Binarize MT quality judgement data for GemmaItXComet (joint chat-template encoding)."
    )
    parser.add_argument("--save_path", required=True, help="Databin name — saved to ./databins/<model_id>/<save_path>/")
    parser.add_argument("--files_yaml", required=True, help="YAML with absolute paths to CSV files with data to use.")
    parser.add_argument("--user_prompt", required=True, help="Path to user prompt .txt; must contain {source}, {translation}, {ref}, {sep1} and {sep2}.")
    parser.add_argument("--system_prompt", default=None, help="Path to system prompt .txt (optional; skip for Gemma).")
    parser.add_argument("--model_id", default=MODEL_ID, help=f"default: {MODEL_ID}")
    
    parser.add_argument("--label_col", default=LABEL_COL, help=f"CSV column with gold labels used as regression target (default: {LABEL_COL}).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=None, help="Max tokens per encoded sequence. Inferred from tokenizer if unset.")
    parser.add_argument("--keep_frac", type=float, default=1.0, help="Keep this fraction of each training file (per-file random subsample). Dev splits are kept whole.")
    parser.add_argument("--seed", type=int, default=SEED, help=f"Default: {SEED}")
    parser.add_argument("--sep1", default=SEP1, help=f"Separator to insert after MT (default: {SEP1}).")
    parser.add_argument("--sep2", default=SEP2, help=f"Separator to insert after MT+REF (default: {SEP2}).")

    
    args = parser.parse_args()
    args.save_path = str(
        Path("./databins") / args.model_id.replace("/", "_") / args.save_path
    )
    print(f"[Info] Starting binarization script")
    main(args)
    print("[Info] Binarization completed.")

