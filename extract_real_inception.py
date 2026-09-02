"""Extract 2048-d FID(ADM) inception features from a folder of real images.

Uses the same code path as the released `FID(ADM).npy` gen features:
    frechet_distance.repr_models.load_repr_model("inception", ...) → 2048-d
    frechet_distance.evaluator.extract_ref_features → runs it over a folder

Output is a single .npy file of shape (N, 2048) — directly compatible with
the gen `FID(ADM).npy` files in eval_released/.

Usage:
    python extract_real_inception.py \\
        --data_dir data/imagenet/gt-image50000 \\
        --out      work_dirs/real_features/inception_ADM/real.npy \\
        --batch_size 64

If your ImageNet copy is an ImageFolder (class-subdirectory layout),
extract_ref_features will recursively walk it. Output ordering is sorted
by path, so reproducibility is fine. Subsample to N afterwards if desired.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from frechet_distance.repr_models import load_repr_model
from frechet_distance.evaluator import extract_ref_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Folder of real images (PNG/JPG/JPEG/WEBP; flat or recursive)")
    ap.add_argument("--out", required=True,
                    help="Output .npy path for raw features (N, 2048)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=256,
                    help="Target square crop size (default 256 — matches eval pipeline)")
    ap.add_argument("--n_cap", type=int, default=0,
                    help="If > 0, truncate to this many samples after extraction")
    ap.add_argument("--cache_pt", type=str, default=None,
                    help="Optional .pt cache path that extract_ref_features uses internally")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ---- Load the canonical 2048-d inception repr model ----
    print("[model] loading inception via load_repr_model('inception', ...) ...")
    model, feat_dim, has_logits, native_size = load_repr_model(
        "inception", target_size=args.img_size
    )
    model = model.to(device).eval()

    if torch.cuda.device_count() > 1:
        print(f"[model] Using {torch.cuda.device_count()} GPUs via DataParallel!")
        model = torch.nn.DataParallel(model)
    print(f"[model] feat_dim={feat_dim}  has_logits={has_logits}  native_size={native_size}")
    if feat_dim != 2048:
        print(f"[warn] feat_dim={feat_dim} (expected 2048). Continuing, but check repr_models.py")

    # ---- Wrap model into the feat_fn callable expected by extract_ref_features ----
    # Match how FDEvaluator.update() calls the model: with bfloat16 autocast,
    # taking the features (first return value) and dropping the logits.
    @torch.inference_mode()
    def feat_fn(imgs: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            feats, _ = model(imgs)
        return feats.float()

    # ---- Extract ----
    print(f"[extract] reading images from {args.data_dir} ...")
    t0 = time.perf_counter()
    features = extract_ref_features(
        feat_fn=feat_fn,
        ref_dir=args.data_dir,
        cache_path=args.cache_pt,
        batch_size=args.batch_size,
        device=device,
        img_size=args.img_size,
    )
    elapsed = time.perf_counter() - t0
    print(f"[extract] done in {elapsed:.1f}s")

    # extract_ref_features returns torch.Tensor on CPU
    features = features.numpy()
    print(f"[extract] features.shape = {features.shape}  dtype={features.dtype}")

    if args.n_cap and features.shape[0] > args.n_cap:
        features = features[: args.n_cap]
        print(f"[extract] truncated to {features.shape}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, features)
    print(f"[save] → {out}")
    print("\nSanity check vs gen file:")
    print(f"  expected gen shape: (50000, 2048)")
    print(f"  this real shape:    {features.shape}")


if __name__ == "__main__":
    main()