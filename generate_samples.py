"""
Generate paired sample sets from a trained generation checkpoint.

For each requested class, generate --num_samples images (each from a different
initial noise) and save them as ONE PNG per class (a horizontal strip), plus a
single contact sheet (`all_classes.png`) that contains EVERY generated image
with the class it is supposed to be printed above it.

Both checkpoints (FD-only baseline and FD+conditional) should be run with
IDENTICAL args except --ckpt and --output_dir. The seed scheme is deterministic
per (gen_seed, class), so `class_207.png` in the baseline folder was generated
from the same (z, y) as `class_207.png` in the conditional folder. That pairing
makes visual diffs clean.

Usage:
    # 20-class run: read the arch/sampling config + class list straight out of
    # the run directory, so no arch flags have to be repeated by hand.
    python generate_samples.py \\
        --run_dir work_dirs/JiT_uncond_gmm_20class/jitB_uncond_gmm20_armC_logp_logq \\
        --ckpt   work_dirs/JiT_uncond_gmm_20class/jitB_uncond_gmm20_armC_logp_logq/checkpoints/latest.pth \\
        --output_dir eval_out/jitB_uncond_gmm20 \\
        --num_samples 3 --use_ema --ema_label edm_1000

    # Or without a run dir: the built-in `imagenet20` preset is the same
    # 20 train_class_ids used by the JiT_uncond_gmm_20class experiments.
    python generate_samples.py \\
        --model JiT_B --img_size 256 --num_classes 1000 --rope_2d --learned_pe \\
        --legacy_time_convention --num_sampling_steps 1 --cfg 3.0 \\
        --ckpt .../checkpoints/latest.pth --output_dir eval_out/run \\
        --class_preset imagenet20 --num_samples 3

    # Explicit class list (overrides both --run_dir and --class_preset)
    python generate_samples.py ... --classes 207 360 387 974 88 --num_samples 5
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")  # headless-safe backend (no display needed on GPU nodes)
import matplotlib.pyplot as plt

# Project imports — must be in PYTHONPATH (run this script from your repo root)
from utils.builders import create_generation_model, create_tokenizer
from main_fd import get_args_parser as get_train_args_parser


# The 20 ImageNet classes the JiT_uncond_gmm_20class runs are fine-tuned on
# (`--train_class_ids` in conditional_main_fd_gmm.py). Any checkpoint from that
# project only has meaningful label embeddings for these IDs.
CLASS_PRESETS = {
    "imagenet20": [0, 9, 88, 130, 207, 279, 281, 340, 360, 387,
                   404, 417, 444, 555, 569, 817, 920, 949, 974, 979],
    "imagenet8": [207, 360, 387, 974, 88, 979, 417, 279],
}

# Keys copied out of a training args_*.json. Restricted to model architecture and
# sampling so run-specific bookkeeping (output_dir, ckpt_dir, seed, ...) can never
# leak into an eval run.
ARCH_KEYS = (
    "model", "img_size", "patch_size", "num_classes", "label_drop_prob",
    "attn_dropout", "proj_dropout", "class_tokens", "time_tokens",
    "guidance_tokens", "interval_tokens", "norm_eps", "norm_p", "rope_2d",
    "learned_pe", "disable_v_head", "t_eps", "rf_dropout", "rf_grad_checkpoint",
    "tokenizer", "token_channels", "tokenizer_patch_size", "P_mean", "P_std",
    "legacy_time_convention", "tr_uniform", "ratio_r_neq_t", "cfg_beta",
    "cfg_omega_max", "aux_head_depth", "loss_type", "aux_pred_type",
    "perceptual_threshold", "perceptual_loss_on_aux", "noise_scale",
    "ema_type", "ema_rates", "ema_halflife_kimg",
)
SAMPLING_KEYS = (
    "sampling_method", "num_sampling_steps", "cfg", "interval_min", "interval_max",
)


def get_eval_parser():
    """Reuse the training parser so model-arch args match exactly."""
    parser = get_train_args_parser()
    parser.add_argument("-h", "--help", action="help",
                        help="show this message and exit")
    parser.add_argument("--ckpt", required=True, help="path to checkpoint .pth")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="training run directory (or a path to its args_*.json). "
                             "Its architecture/sampling settings and --train_class_ids "
                             "are used as defaults, so arch flags need not be repeated. "
                             "Anything passed explicitly on the command line still wins.")
    parser.add_argument("--class_preset", type=str, default=None,
                        choices=sorted(CLASS_PRESETS),
                        help=f"named class list to generate; one of {sorted(CLASS_PRESETS)}. "
                             "Used when --classes is not given and --run_dir has no class list.")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="class ids to generate (one PNG per class). Overrides "
                             "--class_preset and the run dir's train_class_ids.")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="images per class (different initial noise), laid out in one PNG")
    parser.add_argument("--use_ema", action="store_true",
                        help="load EMA weights instead of online")
    parser.add_argument("--ema_idx", type=int, default=0,
                        help="which EMA model to load (if EMA stored as list)")
    parser.add_argument("--ema_label", type=str, default=None,
                        help="named EMA copy to load, e.g. edm_500 (overrides --ema_idx)")
    parser.add_argument("--gen_seed", type=int, default=42,
                        help="base seed for paired noise. KEEP THE SAME ACROSS BOTH RUNS.")
    parser.add_argument("--same_noise_across_classes", action="store_true",
                        help="reuse the same noise samples for every class; corresponding "
                             "columns then differ only in their class label")
    parser.add_argument("--label_grid", action="store_true",
                        help="also save all_classes_labeled.png with ImageNet class names "
                             "on rows and fixed-noise sample indices on columns")
    parser.add_argument("--row_plot", action="store_true",
                        help="also save all_classes_rows.png, the tall one-row-per-class "
                             "matplotlib figure")
    parser.add_argument("--sheet_cols", type=int, default=4,
                        help="how many class blocks sit side by side in all_classes.png")
    parser.add_argument("--tile_size", type=int, default=0,
                        help="resize every image to this many pixels in the contact sheet "
                             "(0 = keep native resolution)")
    parser.add_argument("--grid_title", type=str, default=None,
                        help="optional title for all_classes.png / all_classes_labeled.png, "
                             "e.g. a run name")
    return parser


def _cli_overrides(argv):
    """Dest names explicitly passed on the command line (so a json can't clobber them)."""
    given = set()
    for token in argv:
        if not token.startswith("--") or token == "--":
            continue
        given.add(token[2:].split("=", 1)[0].replace("-", "_"))
    return given


def _find_args_json(run_dir):
    """Accept either a run directory or a direct path to its args_*.json."""
    path = Path(run_dir)
    if path.is_file():
        return path
    candidates = sorted(path.glob("args_*.json"))
    if not candidates:
        raise FileNotFoundError(f"no args_*.json found in {path}")
    return candidates[-1]  # newest by name == newest timestamp


def apply_run_dir(args, argv):
    """Overlay a training run's arch/sampling config onto `args`.

    Returns (json_path, train_class_ids or None). Explicit CLI flags win over the
    json; the json wins over the training parser's defaults.
    """
    json_path = _find_args_json(args.run_dir)
    with open(json_path) as f:
        cfg = json.load(f)

    given = _cli_overrides(argv)
    applied = []
    for key in ARCH_KEYS + SAMPLING_KEYS:
        if key not in cfg or key in given:
            continue
        if getattr(args, key, None) != cfg[key]:
            applied.append(f"{key}={cfg[key]}")
        setattr(args, key, cfg[key])

    print(f"[cfg] loaded run config from {json_path}")
    if applied:
        print(f"[cfg] overrides from run config: {', '.join(applied)}")
    else:
        print("[cfg] run config matched the current defaults; nothing changed")

    class_ids = cfg.get("train_class_ids") or cfg.get("class_of_interest")
    return json_path, class_ids


def resolve_classes(args, argv, run_class_ids):
    """Pick the class list: explicit --classes > --class_preset > run dir > preset default."""
    if args.classes:
        return list(args.classes), "--classes"
    if args.class_preset:
        return list(CLASS_PRESETS[args.class_preset]), f"--class_preset {args.class_preset}"
    if run_class_ids:
        return list(run_class_ids), f"run config train_class_ids ({len(run_class_ids)} classes)"
    return list(CLASS_PRESETS["imagenet20"]), "default imagenet20 preset"


def _load_font(size):
    """Use PIL's bundled/common DejaVu font when available, then fall back safely."""
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _imagenet_class_names():
    """Return the standard torchvision ImageNet-1K index-to-name mapping."""
    try:
        from torchvision.models import ResNet50_Weights
        return ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    except Exception as exc:
        print(f"[labels] warning: could not load ImageNet class names: {exc}")
        return None


def _class_label(class_id, class_names):
    """Return a readable label such as '207: golden retriever'."""
    if class_names is not None and 0 <= class_id < len(class_names):
        return f"{class_id}: {class_names[class_id]}"
    return f"{class_id}: unknown class name"


def _centered_text(draw, xy, text, font, fill, spacing=4):
    """Draw possibly multiline text centered on xy."""
    bbox = draw.multiline_textbbox(
        (0, 0), text, font=font, spacing=spacing, align="center"
    )
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.multiline_text(
        (xy[0] - width / 2 - bbox[0], xy[1] - height / 2 - bbox[1]),
        text,
        font=font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def _fit_text(draw, text, font_size, max_w, min_size=11):
    """Shrink the font until `text` fits in max_w (long class names, narrow tiles)."""
    size = font_size
    while size > min_size:
        font = _load_font(size)
        if draw.textlength(text, font=font) <= max_w:
            return font
        size -= 1
    return _load_font(min_size)


def save_contact_sheet(class_rows, classes, num_samples, out_path, *,
                       class_names=None, title=None, subtitle=None,
                       cols=4, tile_size=0, trained_classes=None, caption_extra=None):
    """One image holding EVERY generated sample, captioned with its intended class.

    Each class becomes a block: a caption bar naming the class it is supposed to
    be ('207: golden retriever') above that class's `num_samples` images. Blocks
    are tiled `cols` across so 20 classes fit on a readable, poster-shaped sheet
    instead of one 5000px-tall column.
    """
    if class_names is None:
        class_names = _imagenet_class_names()

    rows = list(class_rows)
    if tile_size and tile_size > 0:
        rows = [
            np.asarray(Image.fromarray(r).resize(
                (tile_size * num_samples, tile_size), Image.LANCZOS))
            for r in rows
        ]

    cell_h, strip_w = rows[0].shape[:2]
    cell_w = strip_w // num_samples

    n_classes = len(rows)
    cols = max(1, min(cols, n_classes))
    n_block_rows = math.ceil(n_classes / cols)

    caption_h = max(30, int(cell_h * 0.16))
    gutter = max(10, cell_h // 20)
    margin = gutter * 2
    block_w, block_h = strip_w, caption_h + cell_h

    title_font = _load_font(max(20, int(cell_h * 0.11)))
    sub_font = _load_font(max(13, int(cell_h * 0.062)))
    caption_size = max(14, int(cell_h * 0.075))

    header_h = 0
    if title:
        header_h += int(title_font.size * 1.9)
    if subtitle:
        header_h += int(sub_font.size * 1.8)

    sheet_w = margin * 2 + cols * block_w + (cols - 1) * gutter
    sheet_h = margin * 2 + header_h + n_block_rows * block_h + (n_block_rows - 1) * gutter

    canvas = Image.new("RGB", (sheet_w, sheet_h), color=(18, 20, 25))
    draw = ImageDraw.Draw(canvas)

    y = margin
    if title:
        _centered_text(draw, (sheet_w / 2, y + title_font.size * 0.75),
                       title, title_font, fill=(255, 255, 255))
        y += int(title_font.size * 1.9)
    if subtitle:
        _centered_text(draw, (sheet_w / 2, y + sub_font.size * 0.7),
                       subtitle, sub_font, fill=(165, 172, 185))
        y += int(sub_font.size * 1.8)

    grid_y0 = margin + header_h
    for idx, (class_id, row) in enumerate(zip(classes, rows)):
        col, block_row = idx % cols, idx // cols
        x0 = margin + col * (block_w + gutter)
        y0 = grid_y0 + block_row * (block_h + gutter)

        # caption bar: the class this block is *supposed* to be. A class the
        # checkpoint never trained on still has its de-conditioned (null) label
        # row, so the block below is effectively unconditional — say so on it
        # rather than letting it read as a failed conditional sample.
        untrained = trained_classes is not None and class_id not in trained_classes
        draw.rectangle([x0, y0, x0 + block_w - 1, y0 + caption_h - 1],
                       fill=(58, 44, 30) if untrained else (38, 42, 52))
        label = _class_label(class_id, class_names)
        if caption_extra and class_id in caption_extra:
            label += f"   —   {caption_extra[class_id]}"
        if untrained:
            label += "   [not in this run's class set]"
        font = _fit_text(draw, label, caption_size, block_w - 16)
        _centered_text(draw, (x0 + block_w / 2, y0 + caption_h / 2),
                       label, font,
                       fill=(232, 186, 128) if untrained else (240, 243, 250))

        img_y0 = y0 + caption_h
        canvas.paste(Image.fromarray(row), (x0, img_y0))

        # hairlines between the samples of one class + a frame round the block
        for s in range(1, num_samples):
            x = x0 + s * cell_w
            draw.line([(x, img_y0), (x, img_y0 + cell_h - 1)],
                      fill=(18, 20, 25), width=2)
        draw.rectangle([x0, y0, x0 + block_w - 1, y0 + block_h - 1],
                       outline=(70, 76, 90), width=1)

    canvas.save(out_path)
    return canvas.size


def save_combined_plot(class_rows, classes, num_samples, out_path, *,
                       class_names=None, title=None):
    """Render every class's sample strip as one labeled matplotlib figure.

    Each class occupies a single row; its readable label (e.g.
    '207: golden retriever') is drawn to the left of that row's images, so it is
    clear from the plot alone which images belong to which class.
    """
    if class_names is None:
        class_names = _imagenet_class_names()

    n_classes = len(class_rows)
    cell_h, strip_w = class_rows[0].shape[:2]

    # Size the figure so each image row is close to its native pixel size,
    # reserving an extra column of width on the left for the text labels.
    dpi = 100
    label_col_in = 3.4
    fig_w = label_col_in + strip_w / dpi
    fig_h = (cell_h / dpi) * n_classes + (0.5 if title else 0.0)

    fig, axes = plt.subplots(
        n_classes, 1,
        figsize=(fig_w, fig_h),
        dpi=dpi,
        squeeze=False,
        constrained_layout=True,
    )

    for ax, class_id, row in zip(axes[:, 0], classes, class_rows):
        ax.imshow(row, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel(
            _class_label(class_id, class_names),
            rotation=0,
            ha="right",
            va="center",
            fontsize=12,
        )

    if title:
        fig.suptitle(title, fontsize=15)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_labeled_grid(class_rows, classes, num_samples, out_path, *,
                      gen_seed, same_noise_across_classes, title=None):
    """Add expected-class and fixed-noise labels around a generated sample grid."""
    names = _imagenet_class_names()
    cell_h = class_rows[0].shape[0]
    cell_w = class_rows[0].shape[1] // num_samples
    label_w = max(260, cell_w)
    title_h = 54 if title else 0
    header_h = 72
    grid_w = num_samples * cell_w
    canvas = Image.new(
        "RGB",
        (label_w + grid_w, title_h + header_h + len(classes) * cell_h),
        color=(20, 23, 28),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(28)
    header_font = _load_font(20)
    class_font = _load_font(24)
    small_font = _load_font(15)

    if title:
        _centered_text(
            draw,
            ((label_w + grid_w) / 2, title_h / 2),
            title,
            title_font,
            fill=(255, 255, 255),
        )

    header_y = title_h
    _centered_text(
        draw,
        (label_w / 2, header_y + header_h / 2),
        "Expected ImageNet class",
        header_font,
        fill=(235, 235, 235),
    )
    for sample_idx in range(num_samples):
        x0 = label_w + sample_idx * cell_w
        label = f"fixed noise z[{sample_idx}]"
        if same_noise_across_classes:
            label += f"\nstream seed {gen_seed}"
        _centered_text(
            draw,
            (x0 + cell_w / 2, header_y + header_h / 2),
            label,
            header_font,
            fill=(235, 235, 235),
        )

    image_y0 = title_h + header_h
    for row_idx, (class_id, row) in enumerate(zip(classes, class_rows)):
        y0 = image_y0 + row_idx * cell_h
        canvas.paste(Image.fromarray(row), (label_w, y0))

        class_label = _class_label(class_id, names)
        _centered_text(
            draw,
            (label_w / 2, y0 + cell_h / 2 - 18),
            class_label,
            class_font,
            fill=(255, 255, 255),
        )
        if not same_noise_across_classes:
            row_seed = gen_seed + class_id * 10000
            _centered_text(
                draw,
                (label_w / 2, y0 + cell_h / 2 + 22),
                f"noise stream seed {row_seed}",
                small_font,
                fill=(180, 185, 195),
            )

        draw.line(
            [(0, y0), (label_w + grid_w, y0)],
            fill=(95, 100, 110),
            width=1,
        )
        for sample_idx in range(num_samples + 1):
            x = label_w + sample_idx * cell_w
            draw.line(
                [(x, y0), (x, y0 + cell_h)],
                fill=(235, 235, 235),
                width=1,
            )

    canvas.save(out_path)


def load_checkpoint(model, ckpt_path, use_ema=False, ema_idx=0, ema_label=None):
    """Robust ckpt loader — handles a few common save conventions."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if use_ema:
        ema_state = None
        for k in ("model_ema", "ema_model", "ema_state_dict", "ema"):
            if k in ckpt:
                ema_state = ckpt[k]
                break
        if ema_state is None:
            raise ValueError(f"--use_ema set but no EMA state in {ckpt_path}")

        selected_ema = ema_label
        # Current FD-Loss format:
        # {"model_ema": {"shadows": {"edm_250": state_dict, ...}, ...}}
        if isinstance(ema_state, dict) and "shadows" in ema_state:
            shadows = ema_state["shadows"]
            labels = list(shadows)
            if not labels:
                raise ValueError(f"EMA 'shadows' is empty in {ckpt_path}")
            if selected_ema is None:
                if not 0 <= ema_idx < len(labels):
                    raise IndexError(
                        f"--ema_idx {ema_idx} is out of range; available labels: {labels}"
                    )
                selected_ema = labels[ema_idx]
            if selected_ema not in shadows:
                raise ValueError(
                    f"EMA label '{selected_ema}' not found; available labels: {labels}"
                )
            ema_state = shadows[selected_ema]
        # Also accept a directly label-keyed legacy mapping.
        elif (
            selected_ema is not None
            and isinstance(ema_state, dict)
            and selected_ema in ema_state
            and isinstance(ema_state[selected_ema], dict)
        ):
            ema_state = ema_state[selected_ema]

        # EMA may be stored as list of dicts (one per EMA rate)
        if isinstance(ema_state, (list, tuple)):
            if not 0 <= ema_idx < len(ema_state):
                raise IndexError(
                    f"--ema_idx {ema_idx} is out of range for {len(ema_state)} EMA copies"
                )
            ema_state = ema_state[ema_idx]
            selected_ema = selected_ema or str(ema_idx)
        # Or nested as {state_dict: ...}
        if isinstance(ema_state, dict) and "state_dict" in ema_state:
            ema_state = ema_state["state_dict"]
        missing, unexpected = model.load_state_dict(ema_state, strict=False)
        selected_ema = selected_ema or str(ema_idx)
        print(f"[ckpt] loaded EMA[{selected_ema}]  "
              f"missing={len(missing)}  unexpected={len(unexpected)}")
        return f"EMA {selected_ema}"

    state = None
    for k in ("model", "model_state_dict", "state_dict"):
        if k in ckpt:
            state = ckpt[k]
            break
    if state is None:
        state = ckpt  # raw state_dict
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[ckpt] loaded online  "
          f"missing={len(missing)}  unexpected={len(unexpected)}")
    return "online"


@torch.no_grad()
def main():
    argv = sys.argv[1:]
    args = get_eval_parser().parse_args(argv)
    device = "cuda"

    run_class_ids = None
    if args.run_dir:
        _, run_class_ids = apply_run_dir(args, argv)
    args.classes, class_source = resolve_classes(args, argv, run_class_ids)
    trained = set(run_class_ids) if run_class_ids else None
    untrained = [c for c in args.classes if trained is not None and c not in trained]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    args.world_size = 1
    args.global_bsz = args.batch_size * args.world_size
    # total_steps is read by EMA/scheduler init in some configs; harmless here
    args.total_steps = getattr(args, "total_steps", 1)
    # Build model + tokenizer exactly like training does
    tokenizer = create_tokenizer(args)
    model, _ema_model = create_generation_model(args)
    model = model.to(device).eval()
    if tokenizer is not None:
        tokenizer = tokenizer.to(device).eval()

    weights_desc = load_checkpoint(
        model,
        args.ckpt,
        use_ema=args.use_ema,
        ema_idx=args.ema_idx,
        ema_label=args.ema_label,
    )

    # Model knows its own input shape
    in_ch = model.in_channels
    in_size = model.input_size
    input_shape = (in_ch, in_size, in_size)

    sampling_args = {
        "t_min": args.interval_min,
        "t_max": args.interval_max,
        "cfg": args.cfg,
        "num_steps": args.num_sampling_steps,
    }

    class_names = _imagenet_class_names()
    total_classes = len(args.classes)
    total_images = total_classes * args.num_samples

    print(
        f"[gen] total classes={total_classes} | "
        f"images/class={args.num_samples} | total images={total_images}"
    )
    print(f"[gen] class list from {class_source}")
    if untrained:
        print(f"[gen] WARNING: {len(untrained)}/{total_classes} requested classes are NOT in "
              f"this checkpoint's training set: {untrained}")
        print("[gen]          their label rows are still the de-conditioned null row, so those "
              "blocks are effectively unconditional. They are marked on the contact sheet.")
    print(f"[gen] requested class IDs={args.classes}")
    print(f"[gen] cfg={args.cfg}  steps={args.num_sampling_steps}  "
          f"gen_seed={args.gen_seed}  weights={weights_desc}")
    print(f"[gen] same_noise_across_classes={args.same_noise_across_classes}")
    print("[gen] expected labels:")
    for class_idx, class_id in enumerate(args.classes, start=1):
        print(
            f"  [{class_idx}/{total_classes}] {_class_label(class_id, class_names)} "
            f"| images for this class={args.num_samples}"
        )

    class_rows = []
    for class_idx, c in enumerate(args.classes, start=1):
        class_label = _class_label(c, class_names)
        print(
            f"\n[gen] generating class {class_idx}/{total_classes}: "
            f"{class_label} | {args.num_samples} images"
        )
        # PAIRED SEED: same args.gen_seed across two runs => same z, same y
        # => differences in output are attributable to the model only.
        # num_samples images of class c, each from a different initial noise.
        noise_seed = (
            args.gen_seed
            if args.same_noise_across_classes
            else args.gen_seed + c * 10000
        )
        g = torch.Generator(device=device).manual_seed(noise_seed)
        z = torch.randn(args.num_samples, *input_shape, device=device,
                        generator=g) * args.noise_scale
        y = torch.full((args.num_samples,), c, dtype=torch.long, device=device)

        sampled = model.sample_images_with_grad(z, y, sampling_args=sampling_args)
        if tokenizer is not None:
            sampled = tokenizer.decode(tokenizer.denormalize_z(sampled))
        sampled = (sampled * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]

        # Lay the num_samples images out in a single horizontal strip.
        arr = (sampled.float().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        row = np.concatenate(list(arr), axis=1)  # (H, num_samples*W, C)
        class_rows.append(row)
        class_output = out / f"class_{c:03d}.png"
        Image.fromarray(row).save(class_output)

        # Print the expected class for every generated image in this row.
        for sample_idx in range(args.num_samples):
            global_image_idx = (class_idx - 1) * args.num_samples + sample_idx + 1
            print(
                f"  [image {sample_idx + 1}/{args.num_samples}; "
                f"overall {global_image_idx}/{total_images}] "
                f"expected class = {class_label}"
            )

        print(
            f"[gen] saved class {class_idx}/{total_classes}: {class_label} "
            f"-> {class_output}"
        )

    # The headline artefact: every generated image on one sheet, each block
    # captioned with the class it was conditioned on.
    sheet_path = out / "all_classes.png"
    subtitle = (
        f"{total_classes} classes x {args.num_samples} samples = {total_images} images  |  "
        f"{weights_desc}  |  cfg {args.cfg}  |  {args.num_sampling_steps} step(s)  |  "
        f"seed {args.gen_seed}  |  {Path(args.ckpt).resolve()}"
    )
    if untrained:
        subtitle += (f"\n{len(untrained)} of {total_classes} classes (amber) are outside this "
                     f"checkpoint's training set — those blocks are unconditional by construction")
    sheet_size = save_contact_sheet(
        class_rows,
        args.classes,
        args.num_samples,
        sheet_path,
        class_names=class_names,
        title=args.grid_title or "Generated samples — label above each block is the intended class",
        subtitle=subtitle,
        cols=args.sheet_cols,
        tile_size=args.tile_size,
        trained_classes=trained,
    )
    print(f"\n[gen] contact sheet ({sheet_size[0]}x{sheet_size[1]}, all {total_images} "
          f"images, class label above each block) -> {sheet_path}")

    if args.row_plot:
        combined_path = out / "all_classes_rows.png"
        save_combined_plot(
            class_rows,
            args.classes,
            args.num_samples,
            combined_path,
            class_names=class_names,
            title=args.grid_title,
        )
        print(f"[gen] combined row plot -> {combined_path} "
              f"(one row per class, class label drawn beside each row)")
    if args.label_grid:
        labeled_path = out / "all_classes_labeled.png"
        save_labeled_grid(
            class_rows,
            args.classes,
            args.num_samples,
            labeled_path,
            gen_seed=args.gen_seed,
            same_noise_across_classes=args.same_noise_across_classes,
            title=args.grid_title,
        )
        print(f"[gen] labeled grid -> {labeled_path}")
    print(
        f"[gen] done. {total_classes} class PNGs / {total_images} total images "
        f"saved to {out}"
    )


if __name__ == "__main__":
    main()
