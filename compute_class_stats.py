"""Fit the real-data class-conditional GMM used by the log p / log q loss.

Produces the frozen "p" side of :mod:`frechet_distance.gmm`: a whitening PCA
projection plus per-class Gaussians, all in one pass over the dataset.

Why a whitened PCA space
------------------------
A full ``2048 x 2048`` covariance per class is both unstorable
(1000 * 2048^2 * 8B = 33 TB) and unestimable (~1300 images per ImageNet class
against 2048 dimensions -> singular).  Projecting onto the leading ``k``
whitened principal directions of the *real* features fixes both: storage drops
to ``C * k^2`` and ``1300 >> k`` makes the per-class covariances well posed.
Whitening (rather than plain PCA) makes the real pooled covariance the identity
in ``z``-space, which gives both sides a common, meaningful shrinkage target.

The projection is fitted once on real data and frozen; the generated side must
use the identical map or the p/q comparison is meaningless.

Procedure
---------
1. One pass over the dataset extracting features; each rank caches its own
   features in CPU fp16 (~1.8 GB/rank for ImageNet on 3 ranks) and accumulates
   the global sufficient statistics.
2. Rank 0 eigendecomposes the pooled covariance, builds ``P = V_k S_k^{-1/2}``
   and broadcasts it.
3. Each rank projects its cached features and accumulates per-class sums and
   outer products in ``k`` dimensions; these are reduced to rank 0 and turned
   into per-class means/covariances plus the pooled within-class covariance.

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6 torchrun --nproc_per_node=3 compute_class_stats.py \
        --model inception --data_path /data/dataset/imagenet --img_size 256 --pca_dim 128
"""

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

logger = logging.getLogger("FD_loss")

from frechet_distance.repr_models import load_repr_model
from utils.data_util import center_crop_arr
from utils.distributed_util import enable_distributed, get_global_rank, get_world_size


def parse_args():
    p = argparse.ArgumentParser(description="Fit real-data class-conditional GMM stats")
    p.add_argument("--model", type=str, default="inception",
                   help="'inception' or a timm model name; must match the training judge")
    p.add_argument("--data_path", type=str, default="data/imagenet",
                   help="ImageNet root with a 'train/' subfolder")
    p.add_argument("--img_size", type=int, default=256, help="center-crop resolution")
    p.add_argument("--target_size", type=int, default=None,
                   help="override the model's native preprocessing resolution")
    p.add_argument("--pca_dim", type=int, default=128,
                   help="number of whitened principal directions to keep")
    p.add_argument("--pool_type", type=str, default="cls", choices=["cls", "avg"],
                   help="which head of a dual-output repr model to use; must match training")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--max_per_class", type=int, default=None,
                   help="cap images per class (default: all)")
    p.add_argument("--class_ids", type=int, nargs="+", default=None,
                   help="fit on this subset of ImageNet class indices only. The "
                        "whitening PCA, the pooled within-class covariance and the "
                        "posterior denominator are then all defined over the subset, "
                        "which is what a --train_class_ids run needs. Stored in the "
                        "npz as 'class_ids' so the loss can map global labels onto "
                        "the compacted component axis. Default: all 1000 classes.")
    p.add_argument("--output_dir", type=str, default="data/fid_stats")
    p.add_argument("--output_name", type=str, default=None)
    return p.parse_args()


def build_dataloader(data_path, img_size, batch_size, num_workers, rank, world_size,
                     max_per_class=None, class_ids=None):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)

    # Both filters are applied against the raw ImageFolder index in one pass.
    # ``class_ids`` must drop its images *before* any statistic is touched, so
    # that the pooled mean/covariance the whitening is built from describes the
    # subset's marginal rather than ImageNet's.
    wanted = None if class_ids is None else set(int(c) for c in class_ids)
    if wanted is not None or max_per_class is not None:
        # ImageFolder samples are grouped by class and sorted, so a per-class
        # prefix is a contiguous run.
        per_class, keep = {}, []
        for idx, (_, target) in enumerate(dataset.samples):
            if wanted is not None and target not in wanted:
                continue
            seen = per_class.get(target, 0)
            if max_per_class is None or seen < max_per_class:
                keep.append(idx)
                per_class[target] = seen + 1
        if not keep:
            raise ValueError(f"no images left after filtering (class_ids={sorted(wanted or [])})")
        if wanted is not None and len(per_class) != len(wanted):
            missing = sorted(wanted - set(per_class))
            raise ValueError(f"no images found for class_ids {missing}")
        dataset = torch.utils.data.Subset(dataset, keep)

    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=False, drop_last=False)
               if world_size > 1 else None)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=num_workers, pin_memory=True,
                        shuffle=False, drop_last=False)
    return loader, len(dataset)


@torch.inference_mode()
def extract_features(model, loader, feat_dim, rank, world_size, pool_type, has_logits):
    """One pass: cache per-rank features (CPU fp16) and pool global statistics."""
    device = torch.device("cuda")
    feat_sum = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    feat_outer = torch.zeros(feat_dim, feat_dim, dtype=torch.float64, device=device)
    count = 0

    chunks, labels = [], []
    desc = f"[rank {rank}] features" if world_size > 1 else "features"
    pbar = tqdm(loader, desc=desc, position=rank)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            primary, secondary = model(images)
        if pool_type == "avg":
            if has_logits or secondary is None:
                raise ValueError(f"pool_type='avg' is not available for this model")
            feats = secondary
        else:
            feats = primary

        f64 = feats.double()
        feat_sum.add_(f64.sum(0))
        feat_outer.addmm_(f64.T, f64)
        count += feats.shape[0]

        chunks.append(feats.half().cpu())
        labels.append(targets.clone())
        pbar.set_postfix({"images": count})

    feats_local = torch.cat(chunks) if chunks else torch.zeros(0, feat_dim, dtype=torch.half)
    labels_local = torch.cat(labels) if labels else torch.zeros(0, dtype=torch.long)

    if world_size > 1:
        dist.all_reduce(feat_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(feat_outer, op=dist.ReduceOp.SUM)
        count_t = torch.tensor([count], dtype=torch.long, device=device)
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
        count = int(count_t.item())

    return feat_sum, feat_outer, count, feats_local, labels_local


def build_whitening(feat_sum, feat_outer, count, pca_dim):
    """Return (mean (d,), P (d, k)) with ``P = V_k S_k^{-1/2}``.

    In the resulting coordinates the real pooled covariance is the identity, so
    per-class covariances are O(1) and directly interpretable as the fraction of
    the global spread that each class occupies.
    """
    mean = feat_sum / count
    cov = (feat_outer - count * torch.outer(mean, mean)) / (count - 1)
    cov = 0.5 * (cov + cov.T)

    evals, evecs = torch.linalg.eigh(cov)          # ascending
    evals = torch.flip(evals, dims=[0])
    evecs = torch.flip(evecs, dims=[1])

    top_vals = evals[:pca_dim]
    if (top_vals <= 0).any():
        raise RuntimeError(
            f"Non-positive eigenvalue within the leading {pca_dim} directions "
            f"(min={float(top_vals.min()):.3e}); lower --pca_dim"
        )
    explained = float(top_vals.sum() / evals.clamp(min=0).sum())
    proj = evecs[:, :pca_dim] / top_vals.sqrt().unsqueeze(0)
    return mean, proj, explained


@torch.inference_mode()
def accumulate_class_stats(feats_local, labels_local, mean, proj, num_classes,
                           world_size, batch=8192):
    """Per-class sums and outer products in whitened space, reduced across ranks."""
    device = torch.device("cuda")
    k = proj.shape[1]
    cls_count = torch.zeros(num_classes, dtype=torch.float64, device=device)
    cls_sum = torch.zeros(num_classes, k, dtype=torch.float64, device=device)
    cls_outer = torch.zeros(num_classes, k, k, dtype=torch.float64, device=device)

    for start in range(0, feats_local.shape[0], batch):
        f = feats_local[start:start + batch].to(device).double()
        y = labels_local[start:start + batch].to(device)
        z = (f - mean) @ proj                                     # (b, k)
        cls_count.index_add_(0, y, torch.ones_like(y, dtype=torch.float64))
        cls_sum.index_add_(0, y, z)
        # (b, k, k) outer products scattered per class
        cls_outer.index_add_(0, y, z.unsqueeze(2) * z.unsqueeze(1))

    if world_size > 1:
        dist.all_reduce(cls_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(cls_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(cls_outer, op=dist.ReduceOp.SUM)

    return cls_count, cls_sum, cls_outer


def finalize(cls_count, cls_sum, cls_outer):
    """Turn per-class sufficient statistics into means, covariances, and the
    pooled within-class covariance."""
    empty = cls_count == 0
    if empty.any():
        raise RuntimeError(f"{int(empty.sum())} class(es) received no images")

    n = cls_count.unsqueeze(1)
    class_mu = cls_sum / n
    # Sigma_c = E[zz^T] - mu mu^T, with the unbiased (n-1) denominator.
    scatter = cls_outer - n.unsqueeze(2) * (class_mu.unsqueeze(2) * class_mu.unsqueeze(1))
    denom = (cls_count - 1).clamp(min=1).view(-1, 1, 1)
    class_cov = scatter / denom
    class_cov = 0.5 * (class_cov + class_cov.transpose(1, 2))

    # Pooled within-class covariance: the shrinkage target for both sides and
    # the denominator of the intra-class collapse meter.
    within_cov = scatter.sum(0) / (cls_count.sum() - cls_count.shape[0])
    within_cov = 0.5 * (within_cov + within_cov.T)
    return class_mu, class_cov, within_cov


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("httpx", "timm", "huggingface_hub", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)

    args = parse_args()
    enable_distributed()
    rank, world_size = get_global_rank(), get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    if rank != 0:
        logger.setLevel(logging.WARNING)

    logger.info(f"[class-stats] model={args.model} img_size={args.img_size} "
                f"pca_dim={args.pca_dim} gpus={world_size}")

    model, feat_dim, has_logits, target_size = load_repr_model(
        args.model, device="cuda", target_size=args.target_size,
    )
    class_ids = None if args.class_ids is None else sorted(set(int(c) for c in args.class_ids))
    if class_ids is not None:
        logger.info(f"[class-stats] class subset ({len(class_ids)}): {class_ids}")

    loader, total = build_dataloader(
        args.data_path, args.img_size, args.batch_size, args.num_workers,
        rank, world_size, max_per_class=args.max_per_class, class_ids=class_ids,
    )
    logger.info(f"[class-stats] dataset: {total} images, feat_dim={feat_dim}")

    t0 = time.perf_counter()
    feat_sum, feat_outer, count, feats_local, labels_local = extract_features(
        model, loader, feat_dim, rank, world_size, args.pool_type, has_logits,
    )
    logger.info(f"[class-stats] extracted {count} images in {time.perf_counter() - t0:.1f}s "
                f"({count / (time.perf_counter() - t0):.0f} img/s)")

    if args.pca_dim > feat_dim:
        raise ValueError(f"--pca_dim={args.pca_dim} exceeds feat_dim={feat_dim}")

    # Every rank computes the same eigendecomposition from identical all-reduced
    # inputs, which avoids a broadcast and any rank-0 serialisation.
    mean, proj, explained = build_whitening(feat_sum, feat_outer, count, args.pca_dim)
    logger.info(f"[class-stats] whitening PCA: k={args.pca_dim}, "
                f"explained variance={explained:.3f}")

    if class_ids is None:
        num_classes = 1000
    else:
        # Compact the global labels onto [0, len(class_ids)) so the per-class
        # tensors carry no empty rows; the original ids travel in the npz.
        num_classes = len(class_ids)
        remap = torch.full((max(class_ids) + 1,), -1, dtype=torch.long)
        remap[torch.tensor(class_ids)] = torch.arange(num_classes)
        labels_local = remap[labels_local.long()]
        if int(labels_local.min()) < 0:
            raise RuntimeError("label outside the requested subset survived filtering")

    cls_count, cls_sum, cls_outer = accumulate_class_stats(
        feats_local, labels_local, mean, proj, num_classes, world_size,
    )
    class_mu, class_cov, within_cov = finalize(cls_count, cls_sum, cls_outer)

    if rank == 0:
        k = args.pca_dim
        within_trace = float(within_cov.diagonal().sum())
        between_trace = float(class_mu.var(dim=0, unbiased=False).sum())
        logger.info(
            f"[class-stats] tr(within)/k={within_trace / k:.4f}  "
            f"tr(between)/k={between_trace / k:.4f}  "
            f"(the two sum to ~1.0 by construction of the whitening)"
        )

        os.makedirs(args.output_dir, exist_ok=True)
        if args.output_name:
            fname = args.output_name
        else:
            safe = args.model.replace("/", "_").replace(".", "_")
            if safe == "inception":
                target_size = 256
            suffix = "" if class_ids is None else f"_c{num_classes}"
            fname = f"{safe}_in{args.img_size}_t{target_size}_classgmm_k{k}{suffix}.npz"
        out_path = os.path.join(args.output_dir, fname)

        np.savez(
            out_path,
            feat_mean=mean.cpu().numpy(),
            pca_basis=proj.cpu().numpy(),
            class_mu=class_mu.cpu().numpy(),
            class_cov=class_cov.cpu().numpy(),
            class_count=cls_count.cpu().numpy(),
            within_cov=within_cov.cpu().numpy(),
            class_ids=np.arange(num_classes) if class_ids is None
                      else np.asarray(class_ids, dtype=np.int64),
            meta=np.array([args.model, str(args.img_size), str(target_size),
                           str(k), args.pool_type, str(count)]),
        )
        size_mb = os.path.getsize(out_path) / 1e6
        logger.info(f"[class-stats] saved {out_path} ({size_mb:.1f} MB, n={count})")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
