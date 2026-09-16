"""Copy a Qwen p-head checkpoint with a new local backbone path.

Use the same pinned Qwen snapshot that produced the original head. Structural
checks and matching snapshot directory names do not verify model weight bytes.
This creates a new checkpoint and changes only ``vlm_model_name``; it does not
convert precision, refit the head, or rewrite a generator resume checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def snapshot_revision(model_ref: str) -> str | None:
    match = re.search(r"(?:^|[/\\])snapshots[/\\]([0-9a-fA-F]{40})(?:[/\\]|$)", model_ref)
    return match.group(1).lower() if match else None


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def validate_local_qwen(model_path: str | Path) -> Path:
    """Check local configuration, processor assets, and referenced weight files."""
    model_path = Path(model_path).expanduser().resolve(strict=True)
    if not model_path.is_dir():
        raise ValueError(f"Qwen snapshot must be a local directory: {model_path}")
    config = _read_json(model_path / "config.json")
    if config.get("model_type") != "qwen2_5_vl":
        raise ValueError("Expected a Qwen2.5-VL snapshot (model_type=qwen2_5_vl)")
    if not isinstance(config.get("vision_config"), dict):
        raise ValueError("Qwen config is missing vision_config")
    _read_json(model_path / "preprocessor_config.json")
    tokenizer_config = _read_json(model_path / "tokenizer_config.json")
    if not ((model_path / "tokenizer.json").is_file() or
            ((model_path / "vocab.json").is_file() and
             (model_path / "merges.txt").is_file())):
        raise ValueError("Qwen snapshot is missing tokenizer.json or vocab.json + merges.txt")
    if not (tokenizer_config.get("chat_template") or
            (model_path / "chat_template.json").is_file() or
            (model_path / "chat_template.jinja").is_file()):
        raise ValueError("Qwen snapshot is missing its chat template")

    # Prefer safetensors, matching Transformers' normal loading preference.
    for index_name, single_name in (
        ("model.safetensors.index.json", "model.safetensors"),
        ("pytorch_model.bin.index.json", "pytorch_model.bin"),
    ):
        index_path = model_path / index_name
        if index_path.is_file():
            weight_map = _read_json(index_path).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"Empty or invalid weight_map in {index_path}")
            shard_names = set(weight_map.values())
            for shard_name in shard_names:
                if (not isinstance(shard_name, str) or "\\" in shard_name or
                        PurePosixPath(shard_name).is_absolute() or
                        ".." in PurePosixPath(shard_name).parts):
                    raise ValueError(f"Invalid weight shard path: {shard_name!r}")
                shard = model_path / shard_name
                # Hugging Face cache files may be symlinks to its sibling blobs.
                if not shard.is_file() or shard.stat().st_size == 0:
                    raise ValueError(f"Missing or empty Qwen weight shard: {shard}")
            return model_path
        single = model_path / single_name
        if single.is_file() and single.stat().st_size > 0:
            return model_path
    raise ValueError(f"Qwen snapshot has no supported model weights: {model_path}")


def relocate_p_head(input_path: str | Path, output_path: str | Path,
                    qwen_model: str | Path) -> dict:
    import torch
    from vlm_linear_heads import load_p_head_checkpoint, p_head_backend, P_HEAD_BACKEND_QWEN

    source = Path(input_path).expanduser().resolve(strict=True)
    target = Path(output_path).expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    target = target.resolve()
    if source == target:
        raise ValueError("Input and output checkpoints must be different files")
    checkpoint = load_p_head_checkpoint(source)
    if p_head_backend(checkpoint) != P_HEAD_BACKEND_QWEN:
        raise ValueError("Relocation requires a qwen_answer_state p-head checkpoint")
    model_path = validate_local_qwen(qwen_model)
    previous_ref = str(checkpoint["vlm_model_name"])
    previous_revision = snapshot_revision(previous_ref)
    new_revision = snapshot_revision(str(qwen_model)) or snapshot_revision(str(model_path))
    if previous_revision and new_revision and previous_revision != new_revision:
        raise ValueError(
            f"Qwen snapshot revisions differ: {previous_revision} vs {new_revision}; "
            "use the same pinned revision that produced the p head")

    relocated = dict(checkpoint)
    relocated["vlm_model_name"] = str(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents races from overwriting a newly appeared file.
    with target.open("xb") as stream:
        torch.save(relocated, stream)
    return {
        "input": str(source), "output": str(target),
        "old_vlm_model_name": previous_ref, "new_vlm_model_name": str(model_path),
        "source_snapshot_revision": previous_revision,
        "target_snapshot_revision": new_revision,
        "changed_fields": ["vlm_model_name"],
        "p_head_sha256": checkpoint.get("p_head_sha256"),
        "class_ids_sha256": checkpoint.get("class_ids_sha256"),
        "note": "Model weight bytes were not verified. Use the same pinned Qwen revision; "
                "head tensors, normalization, temperature, class order, and metadata were preserved.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Existing copied p_head.pt")
    parser.add_argument("--output", required=True, type=Path, help="New p-head file; must not exist")
    parser.add_argument("--qwen-model", required=True, type=Path,
                        help="Complete local Qwen2.5-VL snapshot of the original pinned revision")
    args = parser.parse_args()
    try:
        result = relocate_p_head(args.input, args.output, args.qwen_model)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
