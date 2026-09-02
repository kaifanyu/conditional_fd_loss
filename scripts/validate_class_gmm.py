"""Sanity-check a fitted class GMM before spending GPU-days on it.

Reports top-1/top-5 accuracy and the mean ``-log p(c|x)`` of the reference
posterior on held-out real images.  Two things matter here:

* If accuracy is at chance, the whitened space does not carry class structure
  and the loss term is worthless.
* The mean ``-log p(c|x)`` is the scale of the ``l_cls`` term at initialisation,
  which is what ``--fd_gmm_weight`` has to be set against.

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/validate_class_gmm.py \
        --stats data/fid_stats/inception_in256_t256_classgmm_k128.npz \
        --data_path /data/dataset/imagenet --split val
"""

import argparse
import os
import sys

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frechet_distance.gmm import ClassGMMReference, OnlineClassStats
from frechet_distance.repr_models import load_repr_model
from utils.data_util import center_crop_arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats", required=True)
    p.add_argument("--model", default="inception")
    p.add_argument("--data_path", default="/data/dataset/imagenet")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--pca_dim", type=int, default=None)
    p.add_argument("--shrinkage", type=float, nargs="+", default=[0.25],
                   help="sweep p-side shrinkage; 1.0 makes p tied (LDA), matching "
                        "q's functional form")
    p.add_argument("--q_shrinkage", type=float, default=0.25)
    p.add_argument("--class_ids", type=int, nargs="+", default=None,
                   help="restrict the held-out set to these classes; required when "
                        "--stats was fitted with compute_class_stats.py --class_ids, "
                        "since the posterior is only defined over the subset")
    p.add_argument("--num_images", type=int, default=25000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--temperature", type=float, nargs="+", default=None,
                   help="sweep --fd_gmm_temp over real images and report where the "
                        "posterior sits relative to --fd_gmm_cls_cap. The operating "
                        "point is the T whose median log p lands just above the cap: "
                        "below it the term has no gradient on already-real-looking "
                        "images, above it the term keeps pushing past real-data "
                        "confidence, which is the adversarial regime")
    p.add_argument("--temp_shrinkage", type=float, default=0.75,
                   help="p-side shrinkage the temperature table is built at")
    p.add_argument("--cls_cap", type=float, default=-0.69,
                   help="--fd_gmm_cls_cap the temperature table reports against")
    args = p.parse_args()

    model, feat_dim, _, _ = load_repr_model(args.model, device="cuda")

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(os.path.join(args.data_path, args.split),
                                   transform=transform)
    # Held-out images must not be a single-class prefix.
    generator = torch.Generator().manual_seed(0)
    if args.class_ids is not None:
        wanted = set(int(c) for c in args.class_ids)
        pool = torch.tensor([i for i, (_, t) in enumerate(dataset.samples) if t in wanted])
        idx = pool[torch.randperm(pool.numel(), generator=generator)][:args.num_images]
    else:
        idx = torch.randperm(len(dataset), generator=generator)[:args.num_images]
    loader = DataLoader(torch.utils.data.Subset(dataset, idx.tolist()),
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=True)

    # Extract once; the shrinkage sweep only changes how the stats are read
    # back, not the features.
    feats, ys = [], []
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc=f"extracting {args.split}", leave=False):
            images = images.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                primary, _ = model(images)
            feats.append(primary.float().half().cpu())
            ys.append(targets.clone())
    feats = torch.cat(feats).cuda().float()
    y_global = torch.cat(ys).cuda()
    seen = y_global.numel()
    _probe = ClassGMMReference.from_npz(args.stats, pca_dim=args.pca_dim)
    if _probe.is_subset and args.class_ids is None:
        raise SystemExit(
            f"{args.stats} was fitted on the {_probe.num_classes}-class subset "
            f"{_probe.class_ids.tolist()}; pass --class_ids with the same list so the "
            f"held-out set matches the posterior's support."
        )
    y_all = _probe.to_local(y_global)
    uniform = float(torch.tensor(float(_probe.num_classes)).log())
    del _probe

    print(f"\n{seen} held-out {args.split} images | uniform -log p = {uniform:.3f}\n")
    header = (f"{'p_shrink':>9} {'top1%':>7} {'top5%':>7} {'-log p':>8} "
              f"{'log q':>8} {'postKL':>8} {'within':>8} {'spread':>8}")
    print(header)
    print("-" * len(header))

    for shrink in args.shrinkage:
        ref = ClassGMMReference.from_npz(args.stats, pca_dim=args.pca_dim,
                                         shrinkage=shrink)
        with torch.inference_mode():
            z = ref.project(feats)
            logp = torch.log_softmax(ref.logits(z), dim=-1)
            order = logp.topk(5, dim=-1).indices
            top1 = float((order[:, 0] == y_all).float().mean()) * 100
            top5 = float((order == y_all.unsqueeze(1)).any(-1).float().mean()) * 100
            nll = float(-logp.gather(1, y_all.unsqueeze(1)).mean())

            # An "ideal generator": q fitted on the very features being scored.
            # Whatever posterior KL remains here is a floor the training signal
            # can never reach -- it measures the mismatch between p's and q's
            # *model classes*, not any deficiency of the generator.
            online = OnlineClassStats(ref.num_classes, ref.k, ema_beta=0.99999,
                                      shrinkage=args.q_shrinkage).cuda()
            for start in range(0, z.shape[0], 2048):
                online.update(z[start:start + 2048], y_all[start:start + 2048])
            online.refresh_cache(ref)

            logq = torch.log_softmax(online.logits(z), dim=-1)
            mean_q = float(logq.gather(1, y_all.unsqueeze(1)).mean())
            diag = online.diagnostics(ref)

        print(f"{shrink:>9.2f} {top1:>7.2f} {top5:>7.2f} {nll:>8.3f} "
              f"{mean_q:>8.3f} {mean_q + nll:>8.3f} "
              f"{diag['gmm_within_trace_ratio']:>8.3f} "
              f"{diag['gmm_class_mean_spread']:>8.3f}")

    print("\npostKL is the residual left by a *perfect* generator. It only reaches 0 when p "
          "\nand q share a functional form (p_shrink=1.0 makes p tied, like q).")

    if args.temperature:
        # Temperature changes no ranking, so top-1/top-5 are identical at every
        # T and are not repeated here; only the confidence scale moves. The
        # density-ratio term reads log_likelihood rather than the softmax, so it
        # is unaffected by any of this -- this table calibrates l_cls alone.
        ref = ClassGMMReference.from_npz(args.stats, pca_dim=args.pca_dim,
                                         shrinkage=args.temp_shrinkage)
        with torch.inference_mode():
            z = ref.project(feats)
            print(f"\n\nTemperature sweep at p_shrink={args.temp_shrinkage:.2f}, "
                  f"cap={args.cls_cap:g} | uniform -log p = {uniform:.3f}")
            header = (f"{'T':>7} {'mean':>9} {'median':>9} {'saturated':>10} "
                      f"{'>cap':>8} {'>-2.30':>8}")
            print(header)
            print("-" * len(header))
            for temp in args.temperature:
                logp = torch.log_softmax(ref.logits(z, temperature=temp), dim=-1)
                target = logp.gather(1, y_all.unsqueeze(1)).squeeze(1)
                # log_softmax is <= 0; "saturated" means it has collapsed onto 0
                # to float precision, where l_cls has no gradient at all.
                sat = float((target > -1e-4).float().mean()) * 100
                above_cap = float((target > args.cls_cap).float().mean()) * 100
                above_10pct = float((target > -2.30).float().mean()) * 100
                print(f"{temp:>7g} {float(target.mean()):>9.3f} "
                      f"{float(target.median()):>9.3f} {sat:>9.1f}% "
                      f"{above_cap:>7.1f}% {above_10pct:>7.1f}%")
        print("\nPick the T whose median lands just above the cap and whose saturated "
              "\nfraction is ~0: at 20 classes that was T=100, at 1000 it was T=10.")


if __name__ == "__main__":
    main()
