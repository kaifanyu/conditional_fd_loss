"""Precompute & cache the CLIP class-prompt text embeddings.

The conditional FD-loss uses a fixed (num_classes, d) tensor of CLIP text
embeddings, one per ImageNet class. They never change during training, so we
build them once and cache to disk. Training will build the cache lazily on
first run anyway; this script just lets you warm it ahead of time (and also
runs the near-uniform smoke test).

Usage:
    python precompute_text_embeddings.py                       # ViT-L-14 ensemble
    python precompute_text_embeddings.py --clip_single_template # single prompt
    python precompute_text_embeddings.py --clip_model_name ViT-B-32
"""

import argparse
import logging

import torch

from clip_classifier import CLIPClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("FD_loss")


def get_args():
    p = argparse.ArgumentParser("Precompute CLIP text embeddings")
    p.add_argument("--clip_model_name", default="ViT-L-14", type=str)
    p.add_argument("--clip_pretrained", default="openai", type=str)
    p.add_argument("--clip_dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--clip_single_template", action="store_true")
    p.add_argument("--clip_cache_dir", default="data/clip_text_cache", type=str)
    p.add_argument("--clip_classnames_file", default=None, type=str)
    p.add_argument("--num_classes", default=1000, type=int)
    p.add_argument("--device", default="cuda", type=str)
    return p.parse_args()


def main():
    args = get_args()
    clf = CLIPClassifier(
        model_name=args.clip_model_name,
        pretrained=args.clip_pretrained,
        dtype=args.clip_dtype,
        single_template=args.clip_single_template,
        cache_dir=args.clip_cache_dir,
        classnames_file=args.clip_classnames_file,
        num_classes=args.num_classes,
        device=args.device,
    )
    logger.info(f"text_features ready: {tuple(clf.text_features.shape)} "
                f"on {clf.text_features.device}")

    stats = clf.smoke_test_uniform()
    logger.info(f"[smoke] random-noise mean log p(per class) = "
                f"{stats['mean_logp_per_class']:.3f} "
                f"(uniform ref {stats['uniform_ref']:.3f})")
    if abs(stats["mean_logp_per_class"] - stats["uniform_ref"]) > 1.0:
        logger.warning("[smoke] noise log-probs are far from uniform — check "
                       "normalization / model loading.")


if __name__ == "__main__":
    main()
