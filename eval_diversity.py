"""Evaluate intra-class diversity for generated ImageNet-style folders.

Expected folder layout:
    root/class_000/000_0000.png
    root/class_000/000_0001.png
    ...

The script reports pairwise distances among samples from the same class. Lower
numbers mean less intra-class diversity. Pixel metrics always run; optional CLIP
and ResNet feature metrics run when their dependencies are installed.

Example:
    python eval_diversity.py \
        --folders kai_results/eval_out/baseline_1step kai_results/eval_out/cond_1step \
        --names baseline cond

    python eval_diversity.py \
        --folders kai_results/eval_out/baseline_1step kai_results/eval_out/cond_1step \
        --names baseline cond \
        --clip --resnet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Intra-class diversity evaluator")
    p.add_argument("--folders", nargs="+", required=True)
    p.add_argument("--names", nargs="+", default=None)
    p.add_argument("--image_size", type=int, default=128,
                   help="Resize used for pixel diversity metrics.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--clip", action="store_true",
                   help="Also compute CLIP image-embedding cosine diversity.")
    p.add_argument("--clip_model", default="ViT-B-32")
    p.add_argument("--clip_pretrained", default="openai")
    p.add_argument("--resnet", action="store_true",
                   help="Also compute timm ResNet-50 feature cosine diversity.")
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def list_items(root: Path) -> list[tuple[Path, int]]:
    items = []
    for class_dir in sorted(root.glob("class_*")):
        if not class_dir.is_dir():
            continue
        try:
            class_id = int(class_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        for path in sorted(class_dir.glob("*.png")):
            items.append((path, class_id))
    if not items:
        raise RuntimeError(f"No class_*/*.png files found under {root}")
    return items


def load_pixel(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(
        (size, size), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def pairwise_from_vectors(vectors: list[np.ndarray], metric: str) -> list[float]:
    vals = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a, b = vectors[i], vectors[j]
            if metric == "rmse":
                d = a - b
                vals.append(float(np.sqrt(np.mean(d * d))))
            elif metric == "mae":
                vals.append(float(np.mean(np.abs(a - b))))
            elif metric == "cosine_distance":
                a = a.reshape(-1)
                b = b.reshape(-1)
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                cos = float(np.dot(a, b) / denom) if denom > 0 else 1.0
                vals.append(1.0 - cos)
            else:
                raise ValueError(metric)
    return vals


def summarize(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def classwise_pairwise(
    grouped: dict[int, list[np.ndarray]],
    metrics: tuple[str, ...],
) -> tuple[dict[str, dict[str, float]], dict[str, list[tuple[int, float, int]]]]:
    all_vals = {m: [] for m in metrics}
    per_class = {m: [] for m in metrics}
    for class_id, vectors in grouped.items():
        if len(vectors) < 2:
            continue
        for metric in metrics:
            vals = pairwise_from_vectors(vectors, metric)
            all_vals[metric].extend(vals)
            per_class[metric].append((class_id, float(np.mean(vals)), len(vectors)))
    summary = {m: summarize(v) for m, v in all_vals.items()}
    lowest = {m: sorted(per_class[m], key=lambda x: x[1])[:10] for m in metrics}
    return summary, lowest


def pixel_diversity(root: Path, image_size: int) -> tuple[dict, dict]:
    grouped: dict[int, list[np.ndarray]] = {}
    for path, class_id in list_items(root):
        grouped.setdefault(class_id, []).append(load_pixel(path, image_size))
    return classwise_pairwise(grouped, ("rmse", "mae", "cosine_distance"))


@torch.inference_mode()
def clip_diversity(root: Path, args: argparse.Namespace) -> tuple[dict, dict]:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained)
    model = model.to(args.device).eval()
    grouped: dict[int, list[np.ndarray]] = {}
    items = list_items(root)
    for lo in range(0, len(items), args.batch_size):
        batch = items[lo:lo + args.batch_size]
        imgs = [preprocess(Image.open(p).convert("RGB")) for p, _ in batch]
        x = torch.stack(imgs).to(args.device)
        feats = model.encode_image(x)
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        feats_np = feats.cpu().numpy()
        for (_, class_id), feat in zip(batch, feats_np):
            grouped.setdefault(class_id, []).append(feat)
    return classwise_pairwise(grouped, ("cosine_distance",))


@torch.inference_mode()
def resnet_diversity(root: Path, args: argparse.Namespace) -> tuple[dict, dict]:
    import timm
    import torchvision.transforms as T

    model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=0)
    model = model.to(args.device).eval()
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    grouped: dict[int, list[np.ndarray]] = {}
    items = list_items(root)
    for lo in range(0, len(items), args.batch_size):
        batch = items[lo:lo + args.batch_size]
        imgs = [transform(Image.open(p).convert("RGB")) for p, _ in batch]
        x = torch.stack(imgs).to(args.device)
        feats = model(x)
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        feats_np = feats.cpu().numpy()
        for (_, class_id), feat in zip(batch, feats_np):
            grouped.setdefault(class_id, []).append(feat)
    return classwise_pairwise(grouped, ("cosine_distance",))


def print_block(name: str, data: dict) -> None:
    print(f"\n{name}")
    print(f"  images={data['n_images']} classes={data['n_classes']}")
    for group, metrics in data["metrics"].items():
        for metric, stats in metrics.items():
            print(
                f"  {group} {metric}: "
                f"mean={stats['mean']:.4f} median={stats['median']:.4f} "
                f"p05={stats['p05']:.4f} p25={stats['p25']:.4f} "
                f"p75={stats['p75']:.4f} p95={stats['p95']:.4f}"
            )
    if "pixel" in data["lowest_diversity"]:
        print("  lowest-diversity classes by pixel RMSE:")
        for class_id, value, n in data["lowest_diversity"]["pixel"]["rmse"]:
            print(f"    class_{class_id:03d}: rmse={value:.4f} n={n}")


def main() -> None:
    args = parse_args()
    names = args.names or [Path(f).name for f in args.folders]
    if len(names) != len(args.folders):
        raise ValueError("--names must match --folders length")

    results = {}
    for name, folder in zip(names, args.folders):
        root = Path(folder)
        items = list_items(root)
        metrics = {}
        lowest = {}

        metrics["pixel"], lowest["pixel"] = pixel_diversity(root, args.image_size)
        if args.clip:
            metrics["clip_image"], lowest["clip_image"] = clip_diversity(root, args)
        if args.resnet:
            metrics["resnet50_feat"], lowest["resnet50_feat"] = resnet_diversity(root, args)

        results[name] = {
            "folder": str(root),
            "n_images": len(items),
            "n_classes": len({c for _, c in items}),
            "metrics": metrics,
            "lowest_diversity": lowest,
        }
        print_block(name, results[name])

    if len(names) == 2:
        a, b = names
        print(f"\n{b} / {a} diversity ratios (lower means {b} is less diverse):")
        for group in results[a]["metrics"]:
            if group not in results[b]["metrics"]:
                continue
            for metric in results[a]["metrics"][group]:
                va = results[a]["metrics"][group][metric]["mean"]
                vb = results[b]["metrics"][group][metric]["mean"]
                print(f"  {group} {metric}: {vb / va:.3f}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
