"""Visualize the RF/NCSN++ CIFAR-10 models: initial noise -> generated image.

Two models are tested:

  1. BASE (unconditional)  : checkpoints/base/cifar_10_base.pth
     gnobitab 1-rectified-flow NCSN++. No class input. We show the initial
     Gaussian noise z0 and the final image it integrates to.

  2. POST-TRAINED (conditional log p_c FD distillation) :
     work_dirs/RF_cifar_cond3e-4/checkpoints/latest.pth
     Same backbone + a trained class-embedding. For every class y we draw a few
     *different* noises so you can see both the conditioning and the diversity.

Sampling is the rectified-flow forward Euler ODE (noise t=0 -> data t=1,
x += v*dt, model fed t*999), exactly the convention RFDenoiser was built with.

Usage (conda env `fdloss`):
    python scripts/visualize_rf_base_vs_cond.py \
        --base_ckpt checkpoints/base/cifar_10_base.pth \
        --cond_ckpt work_dirs/RF_cifar_cond3e-4/checkpoints/latest.pth \
        --out_dir   work_dirs/RF_cifar_cond3e-4/samples_viz \
        --steps 100 --n_base 8 --n_per_class 6 --ema edm_500 --seed 0
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# import the wrapper from the project (run from repo root)
from models.denoiser_rf import RFDenoiser, convert_rf_checkpoint

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

# rectified-flow integration endpoints (must match RFDenoiser)
_RF_EPS, _RF_T, _RF_TIME = 1e-3, 1.0, 999.0


def build_model(num_classes, device, dropout=0.0):
    model = RFDenoiser(img_size=32, in_channels=3, num_classes=num_classes,
                       dropout=dropout, grad_checkpoint=False)
    return model.to(device).eval()


def load_base(model, ckpt_path):
    """Unconditional base: keys are `module.*` -> remap to `net.*`. label_emb is
    absent and stays zero-initialized (strict=False) => purely unconditional."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = convert_rf_checkpoint(ckpt["model"])
    missing, unexpected = model.load_state_dict(sd, strict=False)
    miss = [m for m in missing if "label_emb" not in m]  # label_emb is expected-missing
    print(f"[base] loaded {ckpt_path}  (missing(non-label)={miss}, unexpected={unexpected})")
    return model


def load_cond(model, ckpt_path, ema="edm_500"):
    """Post-trained conditional model. Use an EMA shadow (cleaner samples) if
    available, otherwise the online weights."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = None
    if ema and ema != "online":
        shadows = ckpt.get("model_ema", {}).get("shadows", {})
        if ema in shadows:
            src = shadows[ema]
            print(f"[cond] using EMA shadow '{ema}' from {ckpt_path}")
    if src is None:
        src = ckpt["model"]
        print(f"[cond] using online weights from {ckpt_path}")
    missing, unexpected = model.load_state_dict(src, strict=False)
    # `net.sigmas` is a buffer recomputed from config; absent in EMA shadows -> fine
    missing = [m for m in missing if m != "net.sigmas"]
    print(f"[cond] step={ckpt.get('current_step')}  label_emb_norm="
          f"{src['net.label_emb.weight'].float().norm().item():.3f}  "
          f"(missing={missing}, unexpected={unexpected})")
    return model


@torch.inference_mode()
def sample(model, z, y, steps, device):
    """Forward Euler ODE from noise z (B,3,32,32) to image in [-1,1].
    y is a LongTensor of class labels, or None for unconditional."""
    x = z.to(device)
    y = None if y is None else y.to(device)
    dt = 1.0 / steps
    for i in range(steps):
        t = i / steps * (_RF_T - _RF_EPS) + _RF_EPS
        t_vec = torch.full((x.shape[0],), t * _RF_TIME, device=device, dtype=torch.float32)
        x = x + model.net(x, t_vec, y) * dt
    return x


def to_img(x):
    """[-1,1] tensor -> HWC uint8-friendly float in [0,1]."""
    x = (x * 0.5 + 0.5).clamp(0, 1)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def noise_to_img(z):
    """Per-image min-max normalize the raw Gaussian noise for display only."""
    arr = z.permute(0, 2, 3, 1).cpu().numpy()
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        a = arr[i]
        out[i] = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return out


def save_pairs(noise, imgs, titles, path, suptitle):
    """Two rows: top = initial noise, bottom = generated image."""
    n = imgs.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(1.5 * n, 3.4))
    if n == 1:
        axes = axes.reshape(2, 1)
    for j in range(n):
        axes[0, j].imshow(noise[j]); axes[0, j].axis("off")
        axes[1, j].imshow(imgs[j]);  axes[1, j].axis("off")
        if titles is not None:
            axes[0, j].set_title(titles[j], fontsize=8)
    axes[0, 0].set_ylabel("init noise z0", fontsize=9)
    axes[1, 0].set_ylabel("output", fontsize=9)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def save_class_grid(imgs, n_per_class, path, suptitle):
    """Grid: one row per class, n_per_class generated images per row (diversity)."""
    ncls = len(CIFAR10_CLASSES)
    fig, axes = plt.subplots(ncls, n_per_class,
                             figsize=(1.4 * n_per_class, 1.4 * ncls))
    for c in range(ncls):
        for j in range(n_per_class):
            ax = axes[c, j] if n_per_class > 1 else axes[c]
            ax.imshow(imgs[c * n_per_class + j]); ax.axis("off")
        lab = axes[c, 0] if n_per_class > 1 else axes[c]
        lab.set_ylabel(CIFAR10_CLASSES[c], rotation=0, ha="right", va="center",
                       fontsize=9)
        lab.axis("on"); lab.set_xticks([]); lab.set_yticks([])
        for sp in lab.spines.values():
            sp.set_visible(False)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", default="checkpoints/base/cifar_10_base.pth")
    p.add_argument("--cond_ckpt", default="work_dirs/RF_cifar_cond3e-4/checkpoints/latest.pth")
    p.add_argument("--out_dir",   default="work_dirs/RF_cifar_cond3e-4/samples_viz")
    p.add_argument("--base_steps", type=int, default=100,
                   help="Euler ODE steps for the BASE model (original multi-step rectified flow)")
    p.add_argument("--cond_steps", type=int, default=1,
                   help="Euler steps for the POST-TRAINED model. The FD distillation "
                        "was trained/evaluated at num_sampling_steps=1 -> it is a 1-step "
                        "generator; >1 step degrades it.")
    p.add_argument("--n_base", type=int, default=8, help="# unconditional samples")
    p.add_argument("--n_per_class", type=int, default=6, help="# samples per class")
    p.add_argument("--ema", default="online",
                   help="Weights for the conditional model: 'online' (best at 1-step, "
                        "FID~15) or an EMA shadow edm_250/500/1000/2000")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"device={device}  base_steps={args.base_steps}  cond_steps={args.cond_steps}  seed={args.seed}")

    # ---- 1. BASE (unconditional) ----------------------------------------
    base = load_base(build_model(num_classes=10, device=device), args.base_ckpt)
    z = torch.randn(args.n_base, 3, 32, 32)
    out = sample(base, z, y=None, steps=args.base_steps, device=device)
    save_pairs(noise_to_img(z), to_img(out), titles=None,
               path=os.path.join(args.out_dir, "base_unconditional.png"),
               suptitle=f"BASE (unconditional RF, {args.base_steps}-step Euler)  —  "
                        f"initial noise (top) -> output (bottom)")
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- 2. POST-TRAINED (conditional) ----------------------------------
    cond = load_cond(build_model(num_classes=10, device=device), args.cond_ckpt, ema=args.ema)
    ncls, k = len(CIFAR10_CLASSES), args.n_per_class

    # label the figures with the actual run + checkpoint (derived from the path),
    # so different ckpts/runs aren't mislabeled.
    run_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.cond_ckpt))))
    ckpt_stem = os.path.splitext(os.path.basename(os.path.realpath(args.cond_ckpt)))[0]
    tag = f"{run_name}/{ckpt_stem}, {args.ema}, {args.cond_steps}-step"

    # one fresh noise per (class, sample) -> diversity comes from the noise
    zc = torch.randn(ncls * k, 3, 32, 32)
    yc = torch.arange(ncls).repeat_interleave(k)
    outs = []
    for i in range(0, zc.shape[0], 64):  # small batches to be gentle on memory
        outs.append(sample(cond, zc[i:i + 64], yc[i:i + 64], args.cond_steps, device).cpu())
    outs = torch.cat(outs, 0)

    save_class_grid(
        to_img(outs), k,
        path=os.path.join(args.out_dir, "cond_class_grid.png"),
        suptitle=f"CONDITIONAL ({tag})  —  rows=class y, cols={k} different noises")

    # also an explicit noise->output panel for the first sample of each class
    first = torch.arange(ncls) * k  # index of sample 0 per class
    save_pairs(noise_to_img(zc[first]), to_img(outs[first]),
               titles=CIFAR10_CLASSES,
               path=os.path.join(args.out_dir, "cond_noise_to_output.png"),
               suptitle=f"CONDITIONAL ({tag})  —  one noise z0 (top) + class label y -> output (bottom)")

    print("\nDone. Figures in", args.out_dir)


if __name__ == "__main__":
    main()
