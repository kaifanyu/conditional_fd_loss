"""Mandatory validation of the frozen real-data VLM ``p`` head.

Answers exactly one question before any GPU-days are spent on the generator:

    Does the frozen VLM representation + linear head actually recognise these
    classes on REAL images?

A weak p head makes ``log q(c|z) - log p(c|z)`` meaningless -- the term would be
teaching the generator to match a classifier that does not know the classes.

It also reports where the posterior *sits*: a head that is right but saturated
(``log p(c|z) ~ 0`` on every real image) has no gradient left to give, which is
the failure the GMM temperature sweep exists to prevent.  The temperature stored
in the checkpoint is applied here, and alternative temperatures are swept for
reference.

Standalone usage::

    CUDA_VISIBLE_DEVICES=1 /home/nvidia/miniconda3/envs/fdloss/bin/python \
        validate_vlm_p_head.py \
        --p_head work_dirs/vlm_p_head_siglip_c100/p_head.pt \
        --data_path /data/dataset/imagenet
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from vlm_linear_heads import (
    build_head_from_checkpoint,
    head_metrics,
    load_p_head_checkpoint,
)

logger = logging.getLogger("vlm_p_head")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@torch.no_grad()
def _subset_report(head, temperature, z, y, num_classes):
    log_probs = head.log_probs(z, temperature)
    idx = y.long().view(-1, 1)
    target_logp = log_probs.gather(1, idx).squeeze(1)
    target_prob = target_logp.exp()
    probs = log_probs.exp()
    base = head_metrics(log_probs, y, "p_real")
    qs = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=z.device)
    quantiles = torch.quantile(target_prob, qs)
    logp_quantiles = torch.quantile(target_logp, qs)
    out = {
        "p_real_top1": base["p_real_top1"],
        "p_real_top5": base["p_real_top5"],
        "p_real_ce": base["p_real_ce"],
        "p_real_target_logp_mean": base["p_real_target_logp"],
        "p_real_target_logp_median": base["p_real_target_logp_median"],
        "p_real_entropy": base["p_real_entropy"],
        "p_real_top1_conf": base["p_real_top1_conf"],
        "p_real_target_rank_mean": base["p_real_target_rank"],
        "p_real_target_rank_median": base["p_real_target_rank_median"],
        "p_real_margin": base["p_real_margin"],
        "p_real_target_prob_mean": base["p_real_target_prob"],
        "p_real_target_prob_p10": float(quantiles[0]),
        "p_real_target_prob_p25": float(quantiles[1]),
        "p_real_target_prob_p50": float(quantiles[2]),
        "p_real_target_prob_p75": float(quantiles[3]),
        "p_real_target_prob_p90": float(quantiles[4]),
        "p_real_target_logp_p10": float(logp_quantiles[0]),
        "p_real_target_logp_p25": float(logp_quantiles[1]),
        "p_real_target_logp_p50": float(logp_quantiles[2]),
        "p_real_target_logp_p75": float(logp_quantiles[3]),
        "p_real_target_logp_p90": float(logp_quantiles[4]),
        # A saturated head has no gradient left to give on realistic samples.
        "p_real_frac_target_prob_above_0.99": float((target_prob > 0.99).float().mean()),
        "p_real_frac_target_prob_below_0.10": float((target_prob < 0.10).float().mean()),
        "p_real_n": int(y.numel()),
        "chance_top1": 1.0 / num_classes,
        "chance_mean_rank": (num_classes + 1) / 2.0,
        "uniform_logp": -math.log(num_classes),
        "max_prob_mean": float(probs.max(dim=-1).values.mean()),
    }
    return out


@torch.no_grad()
def build_report(head, temperature: float, z: torch.Tensor, y: torch.Tensor,
                 *, class_ids, holdout_idx=None, calib_idx=None,
                 temperature_sweep=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)):
    """Full validation report on held-out REAL images."""
    num_classes = head.num_classes
    y = y.long()
    report = {
        "num_classes": num_classes,
        "temperature": float(temperature),
        "feature_dim": int(head.feature_dim),
        "feature_norm": head.feature_norm,
        "all": _subset_report(head, temperature, z, y, num_classes),
    }
    if holdout_idx is not None and holdout_idx.numel() > 0:
        report["holdout"] = _subset_report(head, temperature, z[holdout_idx],
                                           y[holdout_idx], num_classes)
    if calib_idx is not None and calib_idx.numel() > 0:
        report["calibration_split"] = _subset_report(head, temperature, z[calib_idx],
                                                     y[calib_idx], num_classes)

    # -- temperature sweep, for reference only; the checkpoint T is frozen --
    sweep = []
    for temp in sorted(set(list(temperature_sweep) + [float(temperature)])):
        log_probs = head.log_probs(z, temp)
        target_logp = log_probs.gather(1, y.view(-1, 1)).squeeze(1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(-1)
        sweep.append({
            "temperature": float(temp),
            "top1": float((log_probs.argmax(-1) == y).float().mean()),
            "ce": float(-target_logp.mean()),
            "target_logp_median": float(target_logp.median()),
            "target_prob_median": float(target_logp.median().exp()),
            "entropy": float(entropy.mean()),
            "frac_target_prob_above_0.99": float((target_logp.exp() > 0.99).float().mean()),
        })
    report["temperature_sweep"] = sweep

    # -- per-class statistics --
    log_probs = head.log_probs(z, temperature)
    pred = log_probs.argmax(-1)
    target_logp = log_probs.gather(1, y.view(-1, 1)).squeeze(1)
    rank = (log_probs > target_logp.view(-1, 1)).sum(-1) + 1
    per_class = []
    for local in range(num_classes):
        mask = y == local
        n = int(mask.sum())
        if n == 0:
            per_class.append({"local_index": local, "class_id": int(class_ids[local]),
                              "count": 0})
            continue
        top5 = log_probs[mask].topk(min(5, num_classes), dim=-1).indices
        per_class.append({
            "local_index": local,
            "class_id": int(class_ids[local]),
            "count": n,
            "top1": float((pred[mask] == local).float().mean()),
            "top5": float(top5.eq(local).any(-1).float().mean()),
            "target_prob_mean": float(target_logp[mask].exp().mean()),
            "target_logp_mean": float(target_logp[mask].mean()),
            "target_rank_mean": float(rank[mask].float().mean()),
        })
    report["per_class"] = per_class
    worst = sorted((c for c in per_class if c["count"] > 0),
                   key=lambda c: c["top1"])[:10]
    report["worst_classes"] = worst
    report["per_class_top1_min"] = min((c["top1"] for c in per_class if c["count"] > 0),
                                       default=float("nan"))
    report["per_class_count_min"] = min((c["count"] for c in per_class), default=0)
    report["per_class_count_max"] = max((c["count"] for c in per_class), default=0)

    # -- confusion matrix (C x C counts) --
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=z.device)
    confusion.index_put_((y, pred), torch.ones_like(y), accumulate=True)
    report["_confusion"] = confusion.cpu()
    return report


def save_confusion(report, path):
    confusion = report.pop("_confusion", None)
    if confusion is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.save(path, confusion.numpy())
    report["confusion_matrix_path"] = str(path)
    return path


def print_report(report, header="p-head validation") -> None:
    num_classes = report["num_classes"]
    section = report.get("holdout", report["all"])
    bar = "=" * 78
    lines = [
        "", bar, header, bar,
        f"  classes                {num_classes}",
        f"  temperature (frozen)   {report['temperature']:.4f}",
        f"  feature dim / norm     {report['feature_dim']} / {report['feature_norm']}",
        f"  chance top1            {section['chance_top1']:.4f}",
        f"  chance mean rank       {section['chance_mean_rank']:.1f}",
        "",
        "  held-out REAL validation:",
        f"    p_real_top1                  {section['p_real_top1']:.4f}",
        f"    p_real_top5                  {section['p_real_top5']:.4f}",
        f"    p_real_ce                    {section['p_real_ce']:.4f}"
        f"   (uniform {section['uniform_logp']:+.4f} -> ce {-section['uniform_logp']:.4f})",
        f"    p_real_target_logp_mean      {section['p_real_target_logp_mean']:.4f}",
        f"    p_real_target_logp_median    {section['p_real_target_logp_median']:.4f}",
        f"    p_real_entropy               {section['p_real_entropy']:.4f}",
        f"    mean top-1 confidence        {section['p_real_top1_conf']:.4f}",
        f"    target rank mean / median    {section['p_real_target_rank_mean']:.3f}"
        f" / {section['p_real_target_rank_median']:.1f}",
        f"    target margin (logp)         {section['p_real_margin']:.4f}",
        "",
        "  target-class probability quantiles:",
        f"    p10 {section['p_real_target_prob_p10']:.4f}"
        f"   p25 {section['p_real_target_prob_p25']:.4f}"
        f"   p50 {section['p_real_target_prob_p50']:.4f}"
        f"   p75 {section['p_real_target_prob_p75']:.4f}"
        f"   p90 {section['p_real_target_prob_p90']:.4f}",
        f"    frac > 0.99  {section['p_real_frac_target_prob_above_0.99']:.4f}"
        f"     frac < 0.10  {section['p_real_frac_target_prob_below_0.10']:.4f}",
        "",
        f"  per-class counts   min {report['per_class_count_min']} "
        f"max {report['per_class_count_max']}",
        f"  worst per-class top1  {report['per_class_top1_min']:.4f}",
    ]
    worst = report.get("worst_classes", [])[:5]
    if worst:
        lines.append("  hardest classes (top1): " + ", ".join(
            f"{c['class_id']}={c['top1']:.2f}" for c in worst))
    if "confusion_matrix_path" in report:
        lines.append(f"  confusion matrix       {report['confusion_matrix_path']}")
    sweep = report.get("temperature_sweep")
    if sweep:
        lines += ["", "  temperature sweep (reference only; the checkpoint T is frozen):",
                  "      T     top1       ce   median logp   median prob   entropy   >0.99"]
        for row in sweep:
            mark = " *" if abs(row["temperature"] - report["temperature"]) < 1e-9 else "  "
            lines.append(
                f"  {mark}{row['temperature']:>5.2f}  {row['top1']:.4f}  "
                f"{row['ce']:7.4f}   {row['target_logp_median']:>10.4f}   "
                f"{row['target_prob_median']:>11.4f}   {row['entropy']:7.4f}  "
                f"{row['frac_target_prob_above_0.99']:.4f}")
    verdict = _verdict(section, num_classes)
    lines += ["", f"  VERDICT: {verdict}", bar, ""]
    print("\n".join(lines), flush=True)


def _verdict(section, num_classes) -> str:
    """Class-count-independent quality call plus an explicit saturation note.

    ``skill`` is the fraction of the headroom above chance that the head
    recovers, so the thresholds mean the same thing at 5 classes and at 1000 --
    a raw top-1 cut would call a 95%-correct 5-way head "barely above chance".
    """
    top1, chance = section["p_real_top1"], section["chance_top1"]
    skill = (top1 - chance) / max(1e-9, 1.0 - chance)
    saturated = section["p_real_frac_target_prob_above_0.99"]
    note = ""
    if saturated > 0.80:
        # Not fatal on its own -- generated images at a de-conditioned start sit
        # nowhere near real-image confidence -- but it is where the term runs out
        # of gradient at the END of training, so it is worth saying out loud.
        note = (f"  NOTE: {saturated:.1%} of real images sit above p=0.99, so "
                f"log p flattens once samples reach real-data confidence. Raise "
                f"--vlm_head_temperature if the term goes quiet late in the run.")
    if skill < 0.30:
        return ("WEAK -- the head recovers only "
                f"{skill:.1%} of the headroom above chance ({top1:.1%} top-1 vs "
                f"{chance:.1%}). Do NOT start generator training." + note)
    if skill < 0.60:
        return (f"MARGINAL -- {top1:.1%} top-1 on real images ({skill:.1%} of the "
                f"headroom above chance). log p(c|z) will be a noisy teacher." + note)
    return (f"OK -- {top1:.1%} top-1 / {section['p_real_top5']:.1%} top-5 on held-out "
            f"real images ({skill:.1%} of the headroom above chance)." + note)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def get_args_parser():
    p = argparse.ArgumentParser(description="Validate a frozen VLM p head on real images")
    p.add_argument("--p_head", required=True, type=str)
    p.add_argument("--data_path", type=str, default="/data/dataset/imagenet")
    p.add_argument("--split", choices=("val", "train"), default="val")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--temperature", type=float, default=None,
                   help="override the checkpoint temperature for this report only")
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument("--feature_cache", type=str, default=None,
                   help="reuse a feature cache written by train_vlm_p_head.py")
    return p


def main(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ckpt = load_p_head_checkpoint(args.p_head)
    head = build_head_from_checkpoint(ckpt, device="cuda")
    head.eval().requires_grad_(False)
    temperature = float(args.temperature if args.temperature is not None
                        else ckpt["temperature"])
    class_ids = [int(c) for c in ckpt["class_ids"]]

    if args.feature_cache and Path(args.feature_cache).is_file():
        blob = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
        z = blob["feats"].to(device="cuda", dtype=torch.float32)
        y = blob["labels_local"].cuda()
        logger.info("loaded %d cached features from %s", z.shape[0], args.feature_cache)
    else:
        z, y = _extract(args, ckpt, class_ids)

    report = build_report(head, temperature, z, y, class_ids=class_ids)
    out_dir = Path(args.output_json).parent if args.output_json else Path(args.p_head).parent
    save_confusion(report, out_dir / "p_head_confusion.npy")
    print_report(report, header=f"p-head validation  ({args.p_head})")
    out_json = args.output_json or str(out_dir / "p_head_validation.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=1)
    logger.info("wrote %s", out_json)
    return 0


@torch.no_grad()
def _extract(args, ckpt, class_ids):
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    from frechet_distance.repr_models import TimmReprModel
    from train_vlm_p_head import _MirrorTransform

    backbone = TimmReprModel(str(ckpt["vlm_model_name"]), device="cuda",
                             target_size=int(ckpt["vlm_target_size"]))
    transform = _MirrorTransform(int(ckpt["vlm_input_size"]), mirror=False)
    dataset = datasets.ImageFolder(os.path.join(args.data_path, args.split),
                                   transform=transform)
    wanted = set(class_ids)
    keep = [i for i, (_, t) in enumerate(dataset.samples) if t in wanted]
    dataset = Subset(dataset, keep)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=True, shuffle=False)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    table = torch.full((1000,), -1, dtype=torch.long)
    for local, global_id in enumerate(class_ids):
        table[global_id] = local
    feats, labels = [], []
    for images, targets in tqdm(loader, desc=f"{args.split} features"):
        images = images.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=amp, enabled=amp != torch.float32):
            primary, secondary = backbone(images)
        z = secondary if str(ckpt["vlm_pool_type"]) == "avg" else primary
        feats.append(z.float().cpu())
        labels.append(table[targets])
    return torch.cat(feats).cuda(), torch.cat(labels).cuda()


if __name__ == "__main__":
    raise SystemExit(main(get_args_parser().parse_args()))
