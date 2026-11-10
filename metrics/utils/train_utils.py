import json, shutil
from pathlib import Path
import yaml
import os


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        num = param.ds_numel if hasattr(param, "ds_numel") else param.numel()
        all_param += num
        if param.requires_grad:
            trainable_params += num
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")



def save_training_metadata(args, train_args, files_to_copy=(), dirs_to_copy=()):
    output_dir = Path(train_args.output_dir)
    metainfo_dir = output_dir / "metainfo"
    metainfo_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "script_arguments": vars(args),
        "training_arguments": train_args.to_dict(),
    }
    with open(metainfo_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, default=str)

    for fpath in files_to_copy:
        shutil.copy(fpath, metainfo_dir / Path(fpath).name)

    for dpath, dest_name in dirs_to_copy:
        shutil.copytree(Path(dpath), metainfo_dir / dest_name, dirs_exist_ok=True)

    print(f"[INFO] Training metadata saved to {metainfo_dir}")


# ---------------------------------------------------------------------------
# Data-config loading (flat YAML: split name -> list of absolute CSV paths)
# ---------------------------------------------------------------------------
def _validate_paths(paths):
    """Warn if a file does not exist. Returns the list unchanged (paths must be absolute)."""
    out=[]
    for p in paths:
        p = os.path.expandvars(p)
        assert os.path.isabs(p), f"Path must be absolute: {p}"
        if not os.path.isfile(p):
            print(f"[Warning] File not found: {p}")
        out.append(p)
    return out

def load_data_config(yaml_path):
    """
    Loads a data-config YAML into {split: [absolute_paths]}.
    Flat format: top-level keys are split names, values are lists of absolute paths.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}

    splits = {}
    for split, raw_paths in cfg.items():
        splits[split] = sorted(_validate_paths(raw_paths or []))

    return splits


