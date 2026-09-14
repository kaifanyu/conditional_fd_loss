"""Verify the extracted ImageNet tree gives CANONICAL class indices.

Every consumer here (train_vlm_p_head.py, compute_repr_stats.py, the trainer)
calls torchvision ImageFolder, which numbers classes 0..N-1 by the sorted order
of the directories that EXIST. `--class_ids` / `--train_class_ids` are global
ImageNet ids, so that numbering is only correct when all 1000 class directories
are present. A partial extraction silently shifts every id past the first gap.

Usage:
    python scripts/check_imagenet_layout.py --data_path /path/to/imagenet
    python scripts/check_imagenet_layout.py --class_ids $(seq 0 10 990)
"""
import argparse
import sys
from pathlib import Path


def canonical_wnids(label_file: Path) -> list[str]:
    """wnid per global id, from the released data/train.txt."""
    table: dict[int, str] = {}
    with label_file.open() as fh:
        for line in fh:
            path, _, idx = line.rpartition(" ")
            if not path:
                continue
            table[int(idx)] = path.split("/", 1)[0]
    return [table[i] for i in range(len(table))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="/mnt/projects/jg/kaifany/dataset/imagenet")
    ap.add_argument("--labels", default="data/train.txt")
    ap.add_argument("--class_ids", type=int, nargs="+", default=None,
                    help="the subset a run would request (default: stride-10 100)")
    args = ap.parse_args()

    wnids = canonical_wnids(Path(args.labels))
    if len(wnids) != 1000:
        print(f"ERROR: {args.labels} defines {len(wnids)} classes, expected 1000")
        return 2
    class_ids = args.class_ids if args.class_ids is not None else list(range(0, 1000, 10))

    root = Path(args.data_path)
    ok = True
    for split in ("train", "val"):
        d = root / split
        if not d.is_dir():
            print(f"{split:5s}: MISSING {d}")
            ok = False
            continue
        dirs = sorted(p.name for p in d.iterdir() if p.is_dir())
        tars = [p for p in d.iterdir() if p.suffix == ".tar"]
        missing = [w for w in wnids if w not in set(dirs)]
        extra = [w for w in dirs if w not in set(wnids)]
        print(f"{split:5s}: {len(dirs)}/1000 class dirs, {len(tars)} leftover .tar")
        if extra:
            print(f"       {len(extra)} unexpected dirs, e.g. {extra[:3]}")
            ok = False
        if missing:
            print(f"       INCOMPLETE -- {len(missing)} missing, e.g. {missing[:3]}")
            print(f"       ImageFolder would number {len(dirs)} classes 0..{len(dirs)-1};")
            print(f"       global ids are WRONG. Do not train on this split yet.")
            ok = False
        else:
            # all present => sorted order is the canonical order
            assert dirs == wnids, "sorted dirs != canonical order"
            print(f"       complete; ImageFolder indices ARE the global ImageNet ids")
            bad = [c for c in class_ids if not (root / split / wnids[c]).is_dir()]
            n_img = sum(1 for _ in (root / split / wnids[class_ids[0]]).iterdir())
            print(f"       requested subset: {len(class_ids)} classes, all present"
                  if not bad else f"       requested subset MISSING {bad[:5]}")
            print(f"       e.g. id {class_ids[0]} -> {wnids[class_ids[0]]} "
                  f"({n_img} images)")
    print("\nVERDICT:", "OK" if ok else "NOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
