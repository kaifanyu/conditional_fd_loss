"""
Extract features for distribution-visualization experiment.
Three sources × three judges; features saved as .npy per (judge, source).
"""
import argparse, os, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from utils.builders import create_generation_model, create_tokenizer
from frechet_distance.repr_models import load_repr_model, model_short_name
from frechet_distance.judges import extract_judge_features

JUDGE_SPECS = [
    ("inception",                                "default", "avg"),
    ("vit_so400m_patch16_siglip_256.v2_webli",   256,       "cls"),
    ("vit_large_patch16_224.mae",                224,       "cls"),
]


# ---------------------------------------------------------------------------
# Robust checkpoint loader: tries common structures + prefix strips, reports.
# ---------------------------------------------------------------------------
def smart_load_ckpt(path, model):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"[ckpt] file: {path}")
    print(f"[ckpt] top type: {type(ckpt).__name__}")
    if isinstance(ckpt, dict):
        print(f"[ckpt] top keys: {list(ckpt.keys())}")

    model_keys = set(model.state_dict().keys())

    # Enumerate every plausible state_dict inside the ckpt
    candidates = []
    if isinstance(ckpt, dict):
        # Common single-level locations
        for k in ["model_ema", "ema_model", "ema", "model",
                  "state_dict", "params", "module", "weights",
                  "ema_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                v = ckpt[k]
                if any(torch.is_tensor(x) for x in v.values()):
                    candidates.append((k, v))
                else:
                    # Nested: e.g. ckpt["model_ema"]["0.9999"] = state_dict
                    for kk, vv in v.items():
                        if isinstance(vv, dict) and any(torch.is_tensor(x) for x in vv.values()):
                            candidates.append((f"{k}.{kk}", vv))
        # Whole-ckpt-is-state-dict case
        if any(torch.is_tensor(v) for v in ckpt.values()):
            candidates.append(("<root>", ckpt))
    else:
        candidates.append(("<root>", ckpt))

    def canonicalize(sd):
            return {k.replace("._flax_linear.",    ".linear.")
                    .replace("._flax_embedding.", ".embedding."): v
                    for k, v in sd.items()}
    candidates = [(name, canonicalize(sd)) for name, sd in candidates]

    # Score each candidate × prefix-strip combo by how many keys overlap
    PREFIXES = ["", "module.", "_orig_mod.", "module._orig_mod.",
                "ema.module.", "ema."]
    best = None  # (name, strip, sd, score)
    for name, sd in candidates:
        for strip in PREFIXES:
            stripped = {(k[len(strip):] if k.startswith(strip) else k): v
                        for k, v in sd.items()}
            score = len(model_keys & set(stripped.keys()))
            if best is None or score > best[3]:
                best = (name, strip, stripped, score)

    if best is None or best[3] == 0:
        raise RuntimeError(f"No usable state_dict found in {path}")

    name, strip, sd, score = best
    # Reconcile leading-singleton shape drift (checkpoint [1,N,D] vs model [N,D]).
    # These are conditional embedding tensors — broadcasting makes the shapes
    # semantically equivalent, only convention differs.
    target_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    fixed = []
    for k in list(sd.keys()):
        if k not in target_shapes:
            continue
        src, tgt = tuple(sd[k].shape), target_shapes[k]
        if src == tgt:
            continue
        if len(src) == len(tgt) + 1 and src[0] == 1 and src[1:] == tgt:
            sd[k] = sd[k].squeeze(0)            # [1,N,D] -> [N,D]
            fixed.append(k)
        elif len(tgt) == len(src) + 1 and tgt[0] == 1 and tgt[1:] == src:
            sd[k] = sd[k].unsqueeze(0)          # [N,D] -> [1,N,D]
            fixed.append(k)
    if fixed:
        print(f"[ckpt] squeezed leading singleton on {len(fixed)} tensor(s): {fixed}")

    msg = model.load_state_dict(sd, strict=False)
    coverage = 100 * (len(model_keys) - len(msg.missing_keys)) / max(len(model_keys), 1)
    print(f"[ckpt] picked: key={name!r}  strip={strip!r}  matched={score}/{len(model_keys)}")
    print(f"[ckpt] coverage: {coverage:.1f}%  missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print(f"[ckpt] first missing:    {msg.missing_keys[:3]}")
    if msg.unexpected_keys:
        print(f"[ckpt] first unexpected: {msg.unexpected_keys[:3]}")
    if coverage < 95.0:
        raise RuntimeError(
            f"[ckpt] FATAL: only {coverage:.1f}% of model weights loaded. "
            f"The checkpoint structure isn't what the loader expects. "
            f"Run the diagnostic snippet and share output."
        )
    return msg


def build_judge(name, target_size, pool_type, device):
    ts = None if target_size == "default" else target_size
    m, feat_dim, _, _ = load_repr_model(name, target_size=ts)
    m = m.to(device).eval()
    return {"name": model_short_name(name), "model": m,
            "feat_dim": feat_dim, "pool_type": pool_type}


@torch.no_grad()
def real_features(judge, data_path, n, batch_size, device, seed=42):
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder(data_path, transform=tfm)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    loader = DataLoader(Subset(ds, idx), batch_size=batch_size,
                        num_workers=8, shuffle=False, pin_memory=True)
    out = []
    pbar = tqdm(loader, desc=f"[real    ] {judge['name']:24s}", unit="batch")
    for imgs, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        f = extract_judge_features(judge, imgs)
        out.append(f.detach().cpu().float().numpy())
    return np.concatenate(out, 0)[:n]


@torch.no_grad()
def generated_features(model, tokenizer, judge, n, batch_size,
                       num_classes, sampling_args, device, noise_scale=1.0, seed=0, tag="gen"):
    torch.manual_seed(seed)
    in_c, in_s = model.in_channels, model.input_size
    out, done = [], 0
    n_steps = sampling_args["num_steps"]
    pbar = tqdm(total=n, desc=f"[{tag:>8s}] {judge['name']:24s}",
                unit="img", smoothing=0.1)
    t_first = None
    while done < n:
        bs = min(batch_size, n - done)
        t0 = time.perf_counter()
        z = torch.randn(bs, in_c, in_s, in_s, device=device) * noise_scale   # NEW
        y = torch.randint(0, num_classes, (bs,), device=device)
        x = model.sample_images_with_grad(z, y, sampling_args=sampling_args)
        if tokenizer is not None:
            x = tokenizer.decode(tokenizer.denormalize_z(x))
        x = (x * 0.5 + 0.5).clamp(0, 1)
        f = extract_judge_features(judge, x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if t_first is None:
            t_first = dt
            eta_min = (n / bs) * dt / 60
            print(f"  [timing] first batch: {dt:.2f}s ({dt/n_steps*1000:.1f}ms/step), "
                  f"projected total: {eta_min:.1f} min")
        out.append(f.detach().cpu().float().numpy())
        done += bs
        pbar.update(bs)
        pbar.set_postfix(s_per_batch=f"{dt:.2f}")
    pbar.close()
    return np.concatenate(out, 0)[:n]


def main():
    from main_fd import get_args_parser
    parser = get_args_parser()                     # already has --data_path, --output_dir,
                                                   # --cfg, --interval_min/max, --seed,
                                                   # --num_classes, --model, --img_size, ...
    # Only add what's NEW:
    # parser.add_argument("--ckpt", type=str, required=True,
    #                     help="Path to pMF-H base checkpoint")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size_gen",  type=int, default=32)
    parser.add_argument("--batch_size_feat", type=int, default=64)
    parser.add_argument("--judges", nargs="+", default=None,
                        help="Subset of timm names to run (default: all 3)")
    parser.add_argument("--gen_label", default="base_50step",
                        help="Filename label (sans .npy) for the 50-step samples")
    parser.add_argument("--sources", nargs="+",
                        default=["real", "base_50step", "base_1step"])
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    print(f"[gpu] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}")
    print(f"[gpu] using: {torch.cuda.get_device_name(0)}")

    # setup(args) normally populates these; we're not calling setup() so mock them:
    args.world_size = 1
    args.rank        = 0
    args.local_rank  = 0
    args.global_bsz  = args.batch_size
    args.distributed = False

    needs_model = any(s != "real" for s in args.sources)
    if needs_model:
        tokenizer = create_tokenizer(args)
        if tokenizer is not None:
            tokenizer = tokenizer.to(device).eval()
        model, _ = create_generation_model(args)
        model = model.to(device).eval()
        smart_load_ckpt(args.ckpt, model)
    else:
        tokenizer, model = None, None
        print("[info] real-only run — skipping diffusion model load")

    samp_50 = {"t_min": args.interval_min, "t_max": args.interval_max,
               "cfg": args.cfg, "num_steps": 50}
    samp_1  = {"t_min": args.interval_min, "t_max": args.interval_max,
               "cfg": args.cfg, "num_steps": 1}

    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)
    specs = JUDGE_SPECS if args.judges is None else \
            [s for s in JUDGE_SPECS if s[0] in args.judges]
    print(f"[plan] judges: {[s[0] for s in specs]}")
    print(f"[plan] sources: {args.sources}")

    for name, ts, pool in tqdm(specs, desc="judges", position=0):
        judge = build_judge(name, ts, pool, device)
        jdir = out_root / judge["name"]; jdir.mkdir(exist_ok=True)
        print(f"\n=== Judge: {judge['name']}  ({name})  feat_dim={judge['feat_dim']} ===")

        for src, fn, kwargs in [
            ("real",        real_features,
             dict(data_path=args.data_path, n=args.num_samples,
                  batch_size=args.batch_size_feat, device=device, seed=args.seed)),
            (args.gen_label, generated_features,
             dict(model=model, tokenizer=tokenizer, n=args.num_samples,
                  batch_size=args.batch_size_gen, num_classes=args.num_classes,
                  sampling_args=samp_50, device=device, noise_scale=args.noise_scale, seed=args.seed+1, tag=args.gen_label)),
            ("base_1step",  generated_features,
             dict(model=model, tokenizer=tokenizer, n=args.num_samples,
                  batch_size=args.batch_size_gen, num_classes=args.num_classes,
                  sampling_args=samp_1, device=device, noise_scale=args.noise_scale, seed=args.seed+2, tag="1-step")),
        ]:
            if src not in args.sources:
                continue
            out_path = jdir / f"{src}.npy"
            if args.skip_existing and out_path.exists():
                print(f"  [skip] {out_path} exists ({np.load(out_path,mmap_mode='r').shape})")
                continue
            feats = fn(judge=judge, **kwargs)
            np.save(out_path, feats)
            print(f"  [save] {out_path}  shape={feats.shape}")

        del judge; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()