#!/usr/bin/env python
"""Rank a checkpoint's classes by how well the held-out probe recognises them.

`probe_top1` in `training_metrics.json` is a single batch-level scalar — one
number over ~96 samples spread across every class. It says whether conditioning
exists; it cannot say *which* classes the model actually learned. This generates
`--probe_samples` images for every class, scores them with the same held-out
ResNet-50 (`ProbeClassifier`, IMAGENET1K_V2) that the training loop logs and
`eval_class_accuracy.py` judges with, and ranks the classes by the result.

Outputs, in `--output_dir`:
  class_probe_ranking.csv   every class, sorted best first
  top<N>_classes.png        contact sheet of the best `--top_n` classes, each
                            block captioned with its measured probe score
  class_XXX.png             per-class strips for the classes on that sheet

The noise stream matches `generate_samples.py` exactly (seed = gen_seed +
class_id * 10000), so the first `--num_samples` images of each class are the
same images that script produces — the sheet shows unselected samples, not the
best few, while the score behind it is measured over the full set.

Usage:
    python rank_classes_by_probe.py \\
        --run_dir work_dirs/JiT_uncond_gmm_100class/jitB_uncond_gmm100_armC_v2_self_w0103 \\
        --ckpt   .../checkpoints/latest.pth \\
        --output_dir .../probe_ranked --use_ema --ema_label edm_1000 \\
        --probe_samples 32 --top_n 20
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from classifier_ensemble import ProbeClassifier
from generate_samples import (
    _class_label,
    _imagenet_class_names,
    apply_run_dir,
    get_eval_parser,
    load_checkpoint,
    resolve_classes,
    save_contact_sheet,
)
from utils.builders import create_generation_model, create_tokenizer


def get_rank_parser():
    parser = get_eval_parser()
    parser.add_argument("--probe_samples", type=int, default=32,
                        help="images generated per class for scoring (more = finer "
                             "ranking; top-1 resolution is 1/this)")
    parser.add_argument("--probe_bsz", type=int, default=32,
                        help="generation chunk size, in images")
    parser.add_argument("--top_n", type=int, default=20,
                        help="how many of the best classes to render")
    return parser


@torch.no_grad()
def probe_scores(probe, images01, label):
    """Per-sample probe readouts for one class's batch. Mirrors ProbeClassifier.stats,
    which returns batch means — here every sample is kept so classes can be ranked."""
    x = F.interpolate(images01.float(), size=(224, 224), mode="bicubic",
                      align_corners=False, antialias=True)
    logits = probe.model((x - probe.mean) / probe.std)
    logp = torch.log_softmax(logits, dim=-1)
    labels = torch.full((logits.shape[0],), label, dtype=torch.long, device=logits.device)
    true_logit = logits.gather(1, labels.view(-1, 1))
    return {
        "top1": (logits.argmax(-1) == labels).float().cpu().numpy(),
        "top5": logits.topk(5, dim=-1).indices.eq(labels.view(-1, 1)).any(-1).float().cpu().numpy(),
        "logp": logp.gather(1, labels.view(-1, 1)).squeeze(1).cpu().numpy(),
        "rank": ((logits > true_logit).sum(-1) + 1).float().cpu().numpy(),
        "pred": logits.argmax(-1).cpu().numpy(),
    }


@torch.no_grad()
def main():
    argv = sys.argv[1:]
    args = get_rank_parser().parse_args(argv)
    device = "cuda"

    run_class_ids = None
    if args.run_dir:
        _, run_class_ids = apply_run_dir(args, argv)
    args.classes, class_source = resolve_classes(args, argv, run_class_ids)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    args.world_size = 1
    args.global_bsz = args.batch_size * args.world_size
    args.total_steps = getattr(args, "total_steps", 1)

    tokenizer = create_tokenizer(args)
    model, _ = create_generation_model(args)
    model = model.to(device).eval()
    if tokenizer is not None:
        tokenizer = tokenizer.to(device).eval()
    weights_desc = load_checkpoint(model, args.ckpt, use_ema=args.use_ema,
                                   ema_idx=args.ema_idx, ema_label=args.ema_label)
    probe = ProbeClassifier(device=device)

    input_shape = (model.in_channels, model.input_size, model.input_size)
    sampling_args = {"t_min": args.interval_min, "t_max": args.interval_max,
                     "cfg": args.cfg, "num_steps": args.num_sampling_steps}
    names = _imagenet_class_names()

    print(f"[rank] {len(args.classes)} classes from {class_source} x "
          f"{args.probe_samples} samples = {len(args.classes) * args.probe_samples} images")
    print(f"[rank] weights={weights_desc}  cfg={args.cfg}  steps={args.num_sampling_steps}  "
          f"gen_seed={args.gen_seed}")

    rows, strips = [], {}
    for i, c in enumerate(args.classes, start=1):
        # same stream as generate_samples.py: z[:num_samples] are its exact images
        g = torch.Generator(device=device).manual_seed(args.gen_seed + c * 10000)
        z = torch.randn(args.probe_samples, *input_shape, device=device,
                        generator=g) * args.noise_scale

        chunks = []
        for start in range(0, args.probe_samples, args.probe_bsz):
            zc = z[start:start + args.probe_bsz]
            y = torch.full((zc.shape[0],), c, dtype=torch.long, device=device)
            sampled = model.sample_images_with_grad(zc, y, sampling_args=sampling_args)
            if tokenizer is not None:
                sampled = tokenizer.decode(tokenizer.denormalize_z(sampled))
            chunks.append((sampled * 0.5 + 0.5).clamp(0, 1))
        images01 = torch.cat(chunks)

        s = probe_scores(probe, images01, c)
        preds, counts = np.unique(s["pred"], return_counts=True)
        top_pred = int(preds[counts.argmax()])
        rows.append({
            "class_id": c,
            "class_name": names[c] if names and c < len(names) else "",
            "n": args.probe_samples,
            "probe_top1": float(s["top1"].mean()),
            "probe_top5": float(s["top5"].mean()),
            "probe_logp": float(s["logp"].mean()),
            "probe_rank_median": float(np.median(s["rank"])),
            "most_common_prediction": top_pred,
            "most_common_prediction_name": names[top_pred] if names and top_pred < len(names) else "",
            "most_common_prediction_frac": float(counts.max() / args.probe_samples),
        })
        arr = (images01[:args.num_samples].float().cpu()
               .permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        strips[c] = np.concatenate(list(arr), axis=1)
        print(f"  [{i:3d}/{len(args.classes)}] {_class_label(c, names):<34s} "
              f"top1={rows[-1]['probe_top1']:.3f} top5={rows[-1]['probe_top5']:.3f} "
              f"median_rank={rows[-1]['probe_rank_median']:.0f}")

    rows.sort(key=lambda r: (-r["probe_top1"], -r["probe_logp"]))
    csv_path = out / "class_probe_ranking.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    overall_t1 = float(np.mean([r["probe_top1"] for r in rows]))
    overall_t5 = float(np.mean([r["probe_top5"] for r in rows]))
    n_above = sum(1 for r in rows if r["probe_top1"] >= 0.5)
    n_zero = sum(1 for r in rows if r["probe_top1"] == 0.0)
    print(f"\n[rank] overall probe top-1 {overall_t1:.4f} | top-5 {overall_t5:.4f} "
          f"over {len(rows)} classes x {args.probe_samples} samples")
    print(f"[rank] classes at >=50% top-1: {n_above}/{len(rows)} | "
          f"classes at 0%: {n_zero}/{len(rows)}")
    print(f"[rank] ranking -> {csv_path}")

    top = rows[:args.top_n]
    classes_top = [r["class_id"] for r in top]
    caption_extra = {
        r["class_id"]: f"probe top-1 {r['probe_top1'] * 100:.0f}%  (top-5 {r['probe_top5'] * 100:.0f}%)"
        for r in top
    }
    for r in top:
        Image.fromarray(strips[r["class_id"]]).save(out / f"class_{r['class_id']:03d}.png")

    sheet = out / f"top{len(top)}_classes.png"
    size = save_contact_sheet(
        [strips[c] for c in classes_top], classes_top, args.num_samples, sheet,
        class_names=names,
        title=args.grid_title or f"Best {len(top)} classes by held-out probe accuracy",
        subtitle=(f"ranked over {len(rows)} classes x {args.probe_samples} samples each  |  "
                  f"overall probe top-1 {overall_t1:.3f} / top-5 {overall_t5:.3f}  |  "
                  f"{weights_desc}  |  cfg {args.cfg}  |  {args.num_sampling_steps} step(s)  |  "
                  f"seed {args.gen_seed}  |  {Path(args.ckpt).resolve()}\n"
                  f"images are the first {args.num_samples} samples of each class, not the "
                  f"best-scoring ones — the percentage is measured over all {args.probe_samples}"),
        cols=args.sheet_cols, tile_size=args.tile_size, caption_extra=caption_extra,
    )
    print(f"[rank] top-{len(top)} sheet ({size[0]}x{size[1]}) -> {sheet}")


if __name__ == "__main__":
    main()
