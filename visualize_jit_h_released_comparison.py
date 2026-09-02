"""Generate matched JiT-H samples for base, FD-Inception, and FD-SIM.

The script loads one checkpoint at a time, samples the same five class labels
with the same latent noise, and writes one combined comparison grid plus one
per-model grid.

Example:
    python visualize_jit_h_released_comparison.py

    python visualize_jit_h_released_comparison.py \
        --classes 207 360 387 974 88 \
        --output kai_results/generated_sample_images/jit_h_three_way.png
"""

from __future__ import annotations

import argparse
import gc
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

JIT_H_DEFAULTS = {
    "model": "JiT_H",
    "img_size": 256,
    "cfg": 2.2,
    "interval_min": 0.1,
    "interval_max": 1.0,
    "num_sampling_steps": 1,
    "rope_2d": True,
    "learned_pe": True,
    "legacy_time_convention": True,
    "ema_type": "edm",
    "noise_scale": 1.0,
}

DEFAULT_CLASSES = [207, 360, 387, 974, 88]


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Matched JiT-H released-checkpoint sample comparison",
    )
    parser.add_argument("--model", default=JIT_H_DEFAULTS["model"])
    parser.add_argument("--img_size", type=int, default=JIT_H_DEFAULTS["img_size"])
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Only used for EMA/global-batch setup.")
    parser.add_argument("--cfg", type=float, default=JIT_H_DEFAULTS["cfg"])
    parser.add_argument("--interval_min", type=float, default=JIT_H_DEFAULTS["interval_min"])
    parser.add_argument("--interval_max", type=float, default=JIT_H_DEFAULTS["interval_max"])
    parser.add_argument("--num_sampling_steps", type=int,
                        default=JIT_H_DEFAULTS["num_sampling_steps"])
    parser.add_argument("--sampling_method", choices=["euler", "heun"], default="heun")
    parser.add_argument("--noise_scale", type=float, default=JIT_H_DEFAULTS["noise_scale"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--tokenizer", default=None)

    # JiT-H architecture defaults from scripts/evaluate_released_ckpt.sh.
    parser.add_argument("--rope_2d", action="store_true",
                        default=JIT_H_DEFAULTS["rope_2d"])
    parser.add_argument("--learned_pe", action="store_true",
                        default=JIT_H_DEFAULTS["learned_pe"])
    parser.add_argument("--legacy_time_convention", action="store_true",
                        default=JIT_H_DEFAULTS["legacy_time_convention"])
    parser.add_argument("--ema_type", choices=["const", "edm"],
                        default=JIT_H_DEFAULTS["ema_type"])
    parser.add_argument("--ema_rates", type=float, nargs="+", default=[0.9999, 0.9996])
    parser.add_argument("--ema_halflife_kimg", type=float, nargs="+",
                        default=[250, 500, 1000, 2000])

    # Constructor defaults consumed by utils.builders.create_generation_model.
    parser.add_argument("--label_drop_prob", type=float, default=0.1)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--P_mean", type=float, default=0.8)
    parser.add_argument("--P_std", type=float, default=0.8)
    parser.add_argument("--t_eps", type=float, default=0.05)

    parser.add_argument("--base_ckpt", default="checkpoints/base/JiT-H.pth")
    parser.add_argument("--fd_inception_ckpt",
                        default="checkpoints/post-trained/JiT-H_FD-Inception.pth")
    parser.add_argument("--fd_sim_ckpt",
                        default="checkpoints/post-trained/JiT-H_FD-SIM.pth")
    parser.add_argument("--classes", type=int, nargs="+", default=DEFAULT_CLASSES,
                        help="ImageNet class IDs to sample, one column per label.")
    parser.add_argument("--output",
                        default="kai_results/generated_sample_images/"
                                "JiT_H_base_fd_inception_fd_sim.png",
                        help="Combined comparison grid path.")
    parser.add_argument("--individual_dir",
                        default="kai_results/generated_sample_images/"
                                "JiT_H_base_fd_inception_fd_sim",
                        help="Directory for the three per-model grids.")
    parser.add_argument("--no_individual", action="store_true",
                        help="Only write the combined comparison grid.")
    return parser


def prepare_single_process_args(args: argparse.Namespace) -> None:
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    args.distributed = False
    args.global_bsz = args.batch_size
    args.enable_amp = args.dtype != "fp32"
    args.amp_dtype = DTYPE_MAP[args.dtype]

    if args.tokenizer is None:
        args.token_channels = 3
        args.tokenizer_patch_size = 1
    input_size = args.img_size // args.tokenizer_patch_size
    args.input_size = (args.token_channels, input_size, input_size)
    args.total_steps = getattr(args, "total_steps", 1)


def strip_wrappers(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    for prefix in ("module.", "_orig_mod.", "ema_model.", "ema.", "model."):
        if state and all(k.startswith(prefix) for k in state.keys()):
            state = {k[len(prefix):]: v for k, v in state.items()}
            break
    return {re.sub(r"\._flax_([^.]+)\.", r".\1.", k): v for k, v in state.items()}


def load_online_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in checkpoint:
                state = checkpoint[key]
                break
        else:
            state = checkpoint
    else:
        state = checkpoint

    state = strip_wrappers(state)
    model_state = model.state_dict()

    for key, value in list(state.items()):
        if key not in model_state:
            continue
        expected = model_state[key].shape
        if value.shape == expected:
            continue
        if value.dim() == len(expected) + 1 and value.shape[0] == 1 and value.shape[1:] == expected:
            state[key] = value.squeeze(0)
        elif value.dim() + 1 == len(expected) and expected[0] == 1 and expected[1:] == value.shape:
            state[key] = value.unsqueeze(0)

    msg = model.load_state_dict(state, strict=False)
    print(f"[ckpt] {ckpt_path} missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if msg.missing_keys:
        raise RuntimeError(f"Checkpoint did not fully load: {msg.missing_keys[:10]}")


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    arr = img.float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def make_grid(rows: list[tuple[str, list[Image.Image]]], classes: list[int],
              title: str | None = None) -> Image.Image:
    font = ImageFont.load_default()
    cell_w, cell_h = rows[0][1][0].size
    label_w = 132
    header_h = 30
    pad = 6
    title_h = 28 if title else 0

    width = label_w + len(classes) * cell_w + (len(classes) - 1) * pad
    height = title_h + header_h + len(rows) * cell_h + (len(rows) - 1) * pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((6, 6), title, fill=(0, 0, 0), font=font)

    y0 = title_h
    for col, class_id in enumerate(classes):
        x = label_w + col * (cell_w + pad)
        draw.text((x + 4, y0 + 8), f"class {class_id}", fill=(0, 0, 0), font=font)

    for row_idx, (label, imgs) in enumerate(rows):
        y = title_h + header_h + row_idx * (cell_h + pad)
        draw.text((8, y + cell_h // 2 - 6), label, fill=(0, 0, 0), font=font)
        for col, img in enumerate(imgs):
            x = label_w + col * (cell_w + pad)
            canvas.paste(img, (x, y))

    return canvas


@torch.inference_mode()
def sample_checkpoint(args: argparse.Namespace, label: str, ckpt_path: str,
                      z_seed: int) -> list[Image.Image]:
    from utils.builders import create_generation_model, create_tokenizer
    from utils.sampling_util import generate_images

    print(f"\n[{label}] loading {ckpt_path}")
    tokenizer = create_tokenizer(args)
    if tokenizer is not None:
        tokenizer = tokenizer.to("cuda").eval()
    model, _ = create_generation_model(args)
    model = model.to("cuda").eval()
    load_online_weights(model, ckpt_path)

    labels = torch.tensor(args.classes, device="cuda", dtype=torch.long)
    g = torch.Generator(device="cuda").manual_seed(z_seed)
    z = args.noise_scale * torch.randn(
        len(args.classes), 3, args.img_size, args.img_size,
        device="cuda", generator=g,
    )
    imgs = generate_images(args, model, labels=labels, cfg=args.cfg,
                           tokenizer=tokenizer, z_t=z)
    pil_imgs = [tensor_to_pil(img) for img in imgs]

    del imgs, z, labels, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return pil_imgs


def main() -> None:
    args = get_parser().parse_args()
    prepare_single_process_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("This script needs CUDA for JiT-H sampling.")

    print("[setup] JiT_H comparison")
    print(f"[setup] classes={args.classes}")
    print(f"[setup] cfg={args.cfg} steps={args.num_sampling_steps} "
          f"interval=[{args.interval_min}, {args.interval_max}] seed={args.seed}")

    specs = [
        ("JiT-H", args.base_ckpt),
        ("JiT-H FD-Inception", args.fd_inception_ckpt),
        ("JiT-H FD-SIM", args.fd_sim_ckpt),
    ]

    rows = [(label, sample_checkpoint(args, label, ckpt, args.seed))
            for label, ckpt in specs]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined = make_grid(rows, args.classes, title="JiT-H matched samples")
    combined.save(output)
    print(f"[save] combined -> {output}")

    if not args.no_individual:
        out_dir = Path(args.individual_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for label, imgs in rows:
            safe = label.lower().replace(" ", "_").replace("-", "_")
            grid = make_grid([(label, imgs)], args.classes)
            path = out_dir / f"{safe}.png"
            grid.save(path)
            print(f"[save] {label} -> {path}")


if __name__ == "__main__":
    main()
