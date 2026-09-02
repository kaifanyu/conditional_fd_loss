"""
Side-by-side visual comparison of two paired image directories.

Because both folders were generated with the same --gen_seed in
generate_samples.py, file `class_207/207_0042.png` in folder A was produced
from the same (z, y) as the same path in folder B. Any visible difference is
attributable to the loss, not to randomness.

Usage:
    python visual_compare.py \\
        --dir_a eval_out/baseline --label_a "FD only" \\
        --dir_b eval_out/cond     --label_b "FD + cond" \\
        --output_dir visual_compare/ \\
        --classes 207 360 387 974 88 979 417 279 \\
        --samples_per_class 8
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_class_imgs(root, c, n):
    class_dir = Path(root) / f"class_{c:03d}"
    if not class_dir.exists():
        return []
    paths = sorted(class_dir.glob("*.png"))[:n]
    return [np.array(Image.open(p).convert("RGB")) for p in paths]


def make_grid(imgs_a, imgs_b, label_a, label_b, class_id):
    """Two rows (a on top, b on bottom), one image per column,
    label gutter on the left, class title on top."""
    h, w = imgs_a[0].shape[:2]
    n = len(imgs_a)
    gutter = 100
    pad = 4
    title_h = 24

    canvas_h = title_h + 2 * h + pad
    canvas_w = gutter + n * (w + pad) - pad
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    for i in range(n):
        x = gutter + i * (w + pad)
        canvas[title_h:title_h + h, x:x + w] = imgs_a[i]
        canvas[title_h + h + pad:title_h + 2 * h + pad, x:x + w] = imgs_b[i]

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    draw.text((10, 4), f"class {class_id}", fill=(0, 0, 0))
    draw.text((10, title_h + h // 2 - 6), label_a, fill=(0, 0, 0))
    draw.text((10, title_h + h + pad + h // 2 - 6), label_b, fill=(0, 0, 0))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_a", required=True)
    ap.add_argument("--dir_b", required=True)
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--classes", type=int, nargs="+", required=True,
                    help="class IDs to compare (ImageNet indices)")
    ap.add_argument("--samples_per_class", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for c in args.classes:
        imgs_a = load_class_imgs(args.dir_a, c, args.samples_per_class)
        imgs_b = load_class_imgs(args.dir_b, c, args.samples_per_class)
        if not imgs_a or not imgs_b:
            print(f"skipping class {c}: missing images")
            continue
        n = min(len(imgs_a), len(imgs_b))
        grid = make_grid(imgs_a[:n], imgs_b[:n], args.label_a, args.label_b, c)
        grid.save(out / f"compare_class_{c:03d}.png")
        print(f"saved compare_class_{c:03d}.png  ({n} samples)")


if __name__ == "__main__":
    main()
