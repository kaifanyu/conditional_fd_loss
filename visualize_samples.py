"""
Generate 5 samples for any released preset.
Usage: python visualize_samples.py --preset pMF_H_256 --ckpt checkpoints/base/pMF-H_256.pth
"""
from pathlib import Path
import os, re
import matplotlib.pyplot as plt
import torch

from utils.builders import create_generation_model, create_tokenizer
from main_fd import get_args_parser


PRESETS = {
    "pMF_B_256": dict(model="pMF_B", cfg=8.5, interval_min=0.1,  interval_max=0.7,
                      rope_2d=True, learned_pe=True, disable_v_head=True, noise_scale=1.0),
    "pMF_L_256": dict(model="pMF_L", cfg=7.0, interval_min=0.2,  interval_max=0.7,
                      rope_2d=True, learned_pe=True, disable_v_head=True, noise_scale=1.0),
    "pMF_H_256": dict(model="pMF_H", cfg=7.0, interval_min=0.2,  interval_max=0.6,
                      rope_2d=True, learned_pe=True, disable_v_head=True, noise_scale=2.0),
    "pMF_B_512": dict(model="pMF_B", cfg=6.5, interval_min=0.1,  interval_max=0.7,
                      rope_2d=True, learned_pe=True, disable_v_head=True, noise_scale=2.0,
                      img_size=512, patch_size=32),
    "pMF_L_512": dict(model="pMF_L", cfg=7.5, interval_min=0.2,  interval_max=0.6,
                      rope_2d=True, learned_pe=True, disable_v_head=True, noise_scale=4.0,
                      img_size=512, patch_size=32),
    "pMF_H_512": dict(model="pMF_H", cfg=5.5, interval_min=0.1,  interval_max=0.6,
                      rope_2d=True, learned_pe=True, disable_v_head=True, noise_scale=4.0,
                      img_size=512, patch_size=32),
    "iMF_B":  dict(model="iMF_B",  cfg=8.0,  interval_min=0.4,  interval_max=0.65,
                   tokenizer="sdvae", tokenizer_patch_size=8, patch_size=2, disable_v_head=True, noise_scale=1.0),
    "iMF_L":  dict(model="iMF_L",  cfg=10.5, interval_min=0.4,  interval_max=0.6,
                   tokenizer="sdvae", tokenizer_patch_size=8, patch_size=2, disable_v_head=True, noise_scale=1.0),
    "iMF_XL": dict(model="iMF_XL", cfg=8.0,  interval_min=0.42, interval_max=0.62,
                   tokenizer="sdvae", tokenizer_patch_size=8, patch_size=2, disable_v_head=True, noise_scale=1.0),
    "JiT_B":  dict(model="JiT_B", cfg=3.0, interval_min=0.1, interval_max=1.0,
                   rope_2d=True, learned_pe=True, legacy_time_convention=True, ema_type="edm", noise_scale=1.0),
    "JiT_L":  dict(model="JiT_L", cfg=2.4, interval_min=0.1, interval_max=1.0,
                   rope_2d=True, learned_pe=True, legacy_time_convention=True, ema_type="edm", noise_scale=1.0),
    "JiT_H":  dict(model="JiT_H", cfg=2.2, interval_min=0.1, interval_max=1.0,
                   rope_2d=True, learned_pe=True, legacy_time_convention=True, ema_type="edm", noise_scale=1.0),
}


def main():
    parser = get_args_parser()
    parser.add_argument("--preset",  required=True, choices=list(PRESETS))
    parser.add_argument("--ckpt",    required=True)
    parser.add_argument("--output",  default=None)
    parser.add_argument("--classes", type=int, nargs="+",
                        default=[207, 360, 387, 974, 88])  # golden, otter, red panda, geyser, macaw
    # in the argparse block, add:
    parser.add_argument("--steps", type=int, nargs="+", default=[1],
                        help="One or more sampling step counts; one row per value.")
    
    args = parser.parse_args()

    # Apply preset — overrides any CLI values for these fields
    for k, v in PRESETS[args.preset].items():
        setattr(args, k, v)
    if args.output is None:
        args.output = f"./samples_{args.preset}.png"
    print(f"[preset] {args.preset}: cfg={args.cfg} interval=[{args.interval_min},{args.interval_max}] "
          f"noise_scale={args.noise_scale}")

    # Distributed mocks (we are single-process)
    args.world_size = 1; args.rank = 0; args.local_rank = 0
    args.global_bsz = args.batch_size; args.distributed = False

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    print(f"[gpu] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}")
    print(f"[gpu] {torch.cuda.get_device_name(0)}")

    tokenizer = create_tokenizer(args)
    if tokenizer is not None:
        tokenizer = tokenizer.to(device).eval()
    model, _ = create_generation_model(args)
    model = model.to(device).eval()

    # ---- load checkpoint with flax-name remap ----
    ckpt  = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model_ema", ckpt.get("model", ckpt))

    for prefix in ("module.", "_orig_mod.", "ema_model.", "ema.", "model."):
        if all(k.startswith(prefix) for k in state.keys()):
            state = {k[len(prefix):]: v for k, v in state.items()}
            break
    state = {re.sub(r"\._flax_([^.]+)\.", r".\1.", k): v for k, v in state.items()}

    model_state = model.state_dict()
    for k, v in list(state.items()):
        if k not in model_state:
            continue
        exp = model_state[k].shape
        if v.shape == exp: continue
        if v.dim() == len(exp) + 1 and v.shape[0] == 1 and v.shape[1:] == exp:
            state[k] = v.squeeze(0)
        elif v.dim() + 1 == len(exp) and exp[0] == 1 and exp[1:] == v.shape:
            state[k] = v.unsqueeze(0)

    msg = model.load_state_dict(state, strict=False)
    print(f"[ckpt] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    if len(msg.missing_keys) > 0:
        print("       missing:", msg.missing_keys)
        raise RuntimeError("Model not fully loaded.")
    
    # replace the sampling + plotting blocks:
    n = len(args.classes)
    z = args.noise_scale * torch.randn(
        n, model.in_channels, model.input_size, model.input_size, device=device)
    y = torch.tensor(args.classes, device=device, dtype=torch.long)

    @torch.no_grad()
    def sample(num_steps):
        sa = {"t_min": args.interval_min, "t_max": args.interval_max,
            "cfg":   args.cfg,          "num_steps": num_steps}
        x = model.sample_images_with_grad(z, y, sampling_args=sa)
        if tokenizer is not None:
            x = tokenizer.decode(tokenizer.denormalize_z(x))
        return (x * 0.5 + 0.5).clamp(0, 1).cpu()

    rows = [(s, sample(s)) for s in args.steps]
    nr   = len(rows)

    fig, axes = plt.subplots(nr, n, figsize=(2.2 * n, 2.5 * nr), squeeze=False)
    for r, (s, imgs) in enumerate(rows):
        for j in range(n):
            axes[r, j].imshow(imgs[j].permute(1, 2, 0).numpy())
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            if r == 0:
                axes[r, j].set_title(f"class {args.classes[j]}", fontsize=9)
        fig.text(0.01, (nr - r - 0.5) / nr, f"{s}-step",
                rotation=90, fontsize=11, va="center", weight="bold")

    fig.suptitle(args.preset, fontsize=11)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.03, 0, 1, 0.97])
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()