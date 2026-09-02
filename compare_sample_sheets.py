#!/usr/bin/env python
"""Pair two `generate_samples.py` output folders into one side-by-side contact sheet.

Both folders must have been generated with the same `--gen_seed`, `--num_samples`
and class list: the seed scheme is deterministic per (gen_seed, class), so
`class_207.png` in either folder came from the *same* initial noise. Any visible
difference is therefore attributable to the checkpoint alone.

Each class gets one block: its label, then model A's samples and model B's
samples separated by a divider. A model that never trained on a class still
carries the de-conditioned (null) label row for it, so that half is flagged —
it is unconditional by construction, not a failed conditional sample.

Usage:
    python compare_sample_sheets.py \\
        --dirs work_dirs/.../samples_20class_3each work_dirs/.../samples_20diag_3each \\
        --labels "20-class model" "100-class model" \\
        --run_dirs work_dirs/.../jitB_uncond_gmm20_armC_logp_logq \\
                   work_dirs/.../jitB_uncond_gmm100_armC_v2_self_w0103 \\
        --out kai_results/plots_gmm/samples_20c_vs_100c.png
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (18, 20, 25)
CAPTION_BG = (38, 42, 52)
CAPTION_FG = (240, 243, 250)
FLAG_BG = (74, 54, 32)
FLAG_FG = (232, 186, 128)
FRAME = (70, 76, 90)


def _load_font(size):
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _imagenet_class_names():
    try:
        from torchvision.models import ResNet50_Weights
        return ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    except Exception:
        return None


def _class_label(class_id, names):
    if names is not None and 0 <= class_id < len(names):
        return f"{class_id}: {names[class_id]}"
    return f"class {class_id}"


def _centered(draw, xy, text, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]) / 2 - box[0],
               xy[1] - (box[3] - box[1]) / 2 - box[1]), text, font=font, fill=fill)


def _fit(draw, text, size, max_w, min_size=11):
    while size > min_size:
        font = _load_font(size)
        if draw.textlength(text, font=font) <= max_w:
            return font
        size -= 1
    return _load_font(min_size)


def trained_classes(run_dir):
    """The class ids a run actually fine-tuned; None when it trained on all."""
    if not run_dir:
        return None
    files = sorted(Path(run_dir).glob("args_*.json"))
    if not files:
        return None
    cfg = json.load(open(files[-1]))
    ids = cfg.get("train_class_ids")
    return set(ids) if ids else None


def load_strips(sample_dir, classes=None):
    """{class_id: HxWx3 strip} from the class_XXX.png files a run wrote."""
    strips = {}
    for path in sorted(Path(sample_dir).glob("class_*.png")):
        try:
            cid = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if classes is None or cid in classes:
            strips[cid] = np.asarray(Image.open(path).convert("RGB"))
    return strips


def build(dirs, labels, run_dirs, out_path, cols=2, tile_size=0, title=None):
    names = _imagenet_class_names()
    trained = [trained_classes(d) for d in (run_dirs or [None] * len(dirs))]

    per_dir = [load_strips(d) for d in dirs]
    classes = sorted(set.intersection(*(set(p) for p in per_dir)))
    if not classes:
        raise SystemExit("no class_*.png files common to all folders")

    def prep(arr):
        if tile_size and tile_size > 0:
            n = arr.shape[1] // arr.shape[0]
            arr = np.asarray(Image.fromarray(arr).resize(
                (tile_size * n, tile_size), Image.LANCZOS))
        return arr

    per_dir = [{c: prep(p[c]) for c in classes} for p in per_dir]
    cell_h, strip_w = per_dir[0][classes[0]].shape[:2]
    n_samples = max(1, round(strip_w / cell_h))
    cell_w = strip_w // n_samples

    n_models = len(dirs)
    gap = max(8, cell_h // 24)                 # between the two models' strips
    caption_h = max(30, int(cell_h * 0.16))
    flag_h = max(20, int(cell_h * 0.10))
    block_w = n_models * strip_w + (n_models - 1) * gap
    block_h = caption_h + cell_h + flag_h
    gutter = max(14, cell_h // 14)
    margin = gutter * 2

    cols = max(1, min(cols, len(classes)))
    n_rows = math.ceil(len(classes) / cols)

    title_font = _load_font(max(22, int(cell_h * 0.115)))
    head_font = _load_font(max(15, int(cell_h * 0.075)))
    flag_font = _load_font(max(12, int(cell_h * 0.055)))
    caption_size = max(14, int(cell_h * 0.078))

    header_h = int(title_font.size * 2.0) if title else 0
    colhead_h = int(head_font.size * 2.2)

    sheet_w = margin * 2 + cols * block_w + (cols - 1) * gutter
    sheet_h = margin * 2 + header_h + colhead_h + n_rows * block_h + (n_rows - 1) * gutter

    canvas = Image.new("RGB", (sheet_w, sheet_h), BG)
    draw = ImageDraw.Draw(canvas)

    if title:
        _centered(draw, (sheet_w / 2, margin + title_font.size * 0.8),
                  title, title_font, CAPTION_FG)

    # model names, repeated over every block column so no half is ambiguous
    colhead_y = margin + header_h
    for col in range(cols):
        bx = margin + col * (block_w + gutter)
        for m, lab in enumerate(labels):
            x0 = bx + m * (strip_w + gap)
            _centered(draw, (x0 + strip_w / 2, colhead_y + colhead_h / 2),
                      lab, head_font, (190, 197, 210))

    grid_y0 = colhead_y + colhead_h
    for idx, cid in enumerate(classes):
        col, row = idx % cols, idx // cols
        bx = margin + col * (block_w + gutter)
        by = grid_y0 + row * (block_h + gutter)

        draw.rectangle([bx, by, bx + block_w - 1, by + caption_h - 1], fill=CAPTION_BG)
        label = _class_label(cid, names)
        _centered(draw, (bx + block_w / 2, by + caption_h / 2),
                  label, _fit(draw, label, caption_size, block_w - 16), CAPTION_FG)

        for m in range(n_models):
            x0 = bx + m * (strip_w + gap)
            y0 = by + caption_h
            canvas.paste(Image.fromarray(per_dir[m][cid]), (x0, y0))
            for s in range(1, n_samples):
                x = x0 + s * cell_w
                draw.line([(x, y0), (x, y0 + cell_h - 1)], fill=BG, width=2)

            untrained = trained[m] is not None and cid not in trained[m]
            fy = y0 + cell_h
            draw.rectangle([x0, fy, x0 + strip_w - 1, fy + flag_h - 1],
                           fill=FLAG_BG if untrained else CAPTION_BG)
            _centered(draw, (x0 + strip_w / 2, fy + flag_h / 2),
                      "class not in this model's training set — unconditional"
                      if untrained else "trained on this class",
                      flag_font, FLAG_FG if untrained else (150, 158, 172))
            draw.rectangle([x0, y0, x0 + strip_w - 1, fy + flag_h - 1],
                           outline=FLAG_FG if untrained else FRAME, width=1)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[compare] {len(classes)} classes x {n_samples} samples x {n_models} models "
          f"-> {out_path} ({canvas.width}x{canvas.height})")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="two or more generate_samples.py output folders")
    ap.add_argument("--labels", nargs="+", required=True, help="one name per folder")
    ap.add_argument("--run_dirs", nargs="+", default=None,
                    help="each folder's training run dir, to flag classes that "
                         "checkpoint never trained on")
    ap.add_argument("--out", required=True, help="output PNG")
    ap.add_argument("--cols", type=int, default=2, help="class blocks side by side")
    ap.add_argument("--tile_size", type=int, default=0, help="resize tiles (0 = native)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if len(args.labels) != len(args.dirs):
        ap.error("--labels must give one label per --dirs entry")
    if args.run_dirs and len(args.run_dirs) != len(args.dirs):
        ap.error("--run_dirs must give one run dir per --dirs entry")

    build(args.dirs, args.labels, args.run_dirs, args.out,
          cols=args.cols, tile_size=args.tile_size, title=args.title)


if __name__ == "__main__":
    main()
