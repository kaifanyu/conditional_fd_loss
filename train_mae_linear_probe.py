"""Train a supervised ImageNet linear head on a frozen pretrained MAE.

The dataset transforms intentionally return images in ``[0, 1]``.  Resize and
ImageNet normalization are owned by ``TimmReprModel``, exactly as in the FD
loss path, so normalization is never applied twice.

Example (two GPUs)::

    torchrun --standalone --nproc_per_node=2 train_mae_linear_probe.py \
        --data_path /data/dataset/imagenet \
        --output_dir work_dirs/mae_probe_vitl_cls \
        --pool_type cls --batch_size 128 --epochs 90
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from frechet_distance.repr_models import TimmReprModel
from mae_linear_probe import (
    CHECKPOINT_FORMAT_VERSION,
    DEFAULT_MAE_MODEL,
    build_linear_probe_head,
    load_probe_checkpoint,
    select_pooled_features,
)
from utils.distributed_util import (
    enable_distributed,
    get_global_rank,
    get_local_rank,
    get_world_size,
)


logger = logging.getLogger("mae_linear_probe")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an ImageNet linear classifier on a frozen MAE backbone"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/dataset/imagenet",
        help="ImageNet root containing train/ and val/ ImageFolder directories",
    )
    parser.add_argument("--output_dir", type=str, default="work_dirs/mae_probe_vitl_cls")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MAE_MODEL)
    parser.add_argument("--pool_type", choices=("cls", "avg"), default="cls")
    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument(
        "--head_norm",
        choices=("bn", "none"),
        default="bn",
        help="official MAE-style non-affine BN before the linear layer, or none",
    )

    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128, help="per-GPU batch size")
    parser.add_argument("--base_lr", type=float, default=0.1,
                        help="learning rate at global batch 256")
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=("lars", "sgd"), default="lars")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16",
                        help="frozen-backbone autocast dtype; head remains fp32")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=0,
                        help="debug only: stop each train epoch after N batches")
    parser.add_argument("--max_val_batches", type=int, default=0,
                        help="debug only: stop validation after N batches")
    return parser


def _mapping_hash(class_to_idx: dict[str, int]) -> str:
    payload = json.dumps(class_to_idx, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _build_datasets(data_path: str, target_size: int):
    root = Path(data_path)
    train_dir, val_dir = root / "train", root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"expected ImageNet directories {train_dir} and {val_dir}"
        )

    interpolation = transforms.InterpolationMode.BICUBIC
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(target_size, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    # 256 -> 224 is the standard ImageNet validation crop for this MAE.
    resize_size = int(round(target_size / 0.875))
    val_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])
    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    val_set = datasets.ImageFolder(val_dir, transform=val_transform)
    if train_set.class_to_idx != val_set.class_to_idx:
        raise ValueError("ImageNet train/val class_to_idx mappings differ")
    return train_set, val_set


def _build_loaders(args, train_set, val_set):
    rank, world_size = get_global_rank(), get_world_size()
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_set,
        sampler=train_sampler,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        sampler=val_sampler,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, train_sampler


def _build_optimizer(args, parameters, actual_lr: float):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=actual_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    try:
        from timm.optim.lars import Lars
    except ImportError:
        from timm.optim import Lars
    return Lars(
        parameters,
        lr=actual_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def _lr_multiplier(epoch: int, args) -> float:
    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        return float(epoch + 1) / args.warmup_epochs
    span = max(1, args.epochs - args.warmup_epochs)
    progress = min(1.0, max(0.0, (epoch - args.warmup_epochs) / span))
    min_ratio = args.min_lr / max(args.actual_lr, 1e-12)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _reduce_metrics(loss_sum, correct1, correct5, count, device):
    values = torch.tensor(
        [loss_sum, correct1, correct5, count],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    loss_sum, correct1, correct5, count = values.tolist()
    count = max(count, 1.0)
    return {
        "loss": loss_sum / count,
        "top1": correct1 / count,
        "top5": correct5 / count,
        "n": int(count),
    }


def _run_epoch(
    backbone,
    head,
    loader,
    pool_type,
    device,
    amp_dtype,
    *,
    optimizer=None,
    max_batches=0,
    print_freq=0,
    epoch=0,
):
    training = optimizer is not None
    backbone.eval()
    head.train(training)
    loss_sum = correct1 = correct5 = count = 0.0
    start = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Probe training freezes MAE completely.  The generator-training path
        # deliberately does not use no_grad because it needs d(logp)/d(image).
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_dtype != torch.float32,
            ):
                features = select_pooled_features(backbone(images), pool_type)
        logits = head(features.float())
        loss = F.cross_entropy(logits, labels)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = labels.shape[0]
        topk = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
        loss_sum += float(loss.detach()) * batch_size
        correct1 += float((topk[:, 0] == labels).sum())
        correct5 += float(topk.eq(labels[:, None]).any(dim=-1).sum())
        count += batch_size

        if print_freq > 0 and batch_idx % print_freq == 0 and get_global_rank() == 0:
            logger.info(
                "epoch=%d batch=%d/%d loss=%.4f top1=%.3f elapsed=%.1fs",
                epoch,
                batch_idx,
                len(loader),
                loss_sum / max(count, 1),
                correct1 / max(count, 1),
                time.perf_counter() - start,
            )
    return _reduce_metrics(loss_sum, correct1, correct5, count, device)


def _unwrap_head(head):
    return head.module if isinstance(head, DistributedDataParallel) else head


def _cpu_state_dict(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _save_checkpoint(
    path: Path,
    args,
    head,
    optimizer,
    scheduler,
    class_to_idx,
    epoch,
    best_val_top1,
    best_val_top5,
    val_metrics,
):
    import timm

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name": args.model_name,
        "pool_type": args.pool_type,
        "target_size": args.target_size,
        "feature_dim": args.feature_dim,
        "num_classes": args.num_classes,
        "head_norm": args.head_norm,
        "temperature": 1.0,
        "head_state_dict": _cpu_state_dict(_unwrap_head(head)),
        "class_to_idx": class_to_idx,
        "class_mapping_sha256": _mapping_hash(class_to_idx),
        "epoch": epoch,
        "best_val_top1": best_val_top1,
        "best_val_top5": best_val_top5,
        "val_metrics": val_metrics,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
        "torch_version": str(torch.__version__),
        "timm_version": timm.__version__,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def main(args) -> int:
    enable_distributed()
    rank, local_rank, world_size = get_global_rank(), get_local_rank(), get_world_size()
    device = torch.device("cuda", local_rank)
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    seed = args.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_set, val_set = _build_datasets(args.data_path, args.target_size)
    if len(train_set.classes) != args.num_classes:
        raise ValueError(
            f"ImageNet train has {len(train_set.classes)} classes; expected {args.num_classes}"
        )
    expected_indices = list(range(args.num_classes))
    if sorted(train_set.class_to_idx.values()) != expected_indices:
        raise ValueError("ImageNet class_to_idx is not contiguous 0..num_classes-1")
    train_loader, val_loader, train_sampler = _build_loaders(args, train_set, val_set)

    backbone = TimmReprModel(
        args.model_name,
        device=device,
        target_size=args.target_size,
    )
    args.feature_dim = int(backbone.feat_dim)
    head = build_linear_probe_head(
        args.feature_dim,
        args.num_classes,
        args.head_norm,
    ).to(device=device, dtype=torch.float32)
    if world_size > 1 and args.head_norm == "bn":
        head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(head)
    head = DistributedDataParallel(
        head,
        device_ids=[local_rank],
        broadcast_buffers=True,
    )

    args.actual_lr = args.base_lr * (args.batch_size * world_size) / 256.0
    optimizer = _build_optimizer(args, head.parameters(), args.actual_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _lr_multiplier(epoch, args),
    )
    amp_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    start_epoch, best_top1, best_top5 = 0, -1.0, -1.0
    if args.resume:
        checkpoint = load_probe_checkpoint(args.resume)
        for key in (
            "model_name",
            "pool_type",
            "target_size",
            "feature_dim",
            "num_classes",
            "head_norm",
        ):
            current = getattr(args, key)
            if checkpoint[key] != current:
                raise ValueError(
                    f"resume checkpoint {key}={checkpoint[key]!r} does not match {current!r}"
                )
        if dict(checkpoint["class_to_idx"]) != train_set.class_to_idx:
            raise ValueError("resume checkpoint uses a different ImageNet class mapping")
        _unwrap_head(head).load_state_dict(checkpoint["head_state_dict"], strict=True)
        if not args.eval_only:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
        best_top1 = float(checkpoint.get("best_val_top1", -1.0))
        best_top5 = float(checkpoint.get("best_val_top5", -1.0))

    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "dataset train=%d val=%d classes=%d mapping_sha256=%s",
            len(train_set),
            len(val_set),
            len(train_set.classes),
            _mapping_hash(train_set.class_to_idx),
        )
        logger.info(
            "frozen backbone=%s pool=%s dim=%d target=%d; head=%s; global_batch=%d lr=%.6g",
            args.model_name,
            args.pool_type,
            args.feature_dim,
            args.target_size,
            args.head_norm,
            args.batch_size * world_size,
            args.actual_lr,
        )

    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval_only requires --resume /path/to/probe.pt")
        metrics = _run_epoch(
            backbone,
            head,
            val_loader,
            args.pool_type,
            device,
            amp_dtype,
            max_batches=args.max_val_batches,
        )
        if rank == 0:
            logger.info("validation %s", json.dumps(metrics, sort_keys=True))
        dist.barrier()
        dist.destroy_process_group()
        return 0

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_metrics = _run_epoch(
            backbone,
            head,
            train_loader,
            args.pool_type,
            device,
            amp_dtype,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
            print_freq=args.print_freq,
            epoch=epoch,
        )
        val_metrics = _run_epoch(
            backbone,
            head,
            val_loader,
            args.pool_type,
            device,
            amp_dtype,
            max_batches=args.max_val_batches,
            epoch=epoch,
        )
        scheduler.step()

        improved = val_metrics["top1"] > best_top1
        if improved:
            best_top1 = val_metrics["top1"]
            best_top5 = val_metrics["top5"]
        if rank == 0:
            record = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
                "best_val_top1": best_top1,
                "best_val_top5": best_top5,
            }
            logger.info("metrics %s", json.dumps(record, sort_keys=True))
            _save_checkpoint(
                Path(args.output_dir) / "last.pt",
                args,
                head,
                optimizer,
                scheduler,
                train_set.class_to_idx,
                epoch,
                best_top1,
                best_top5,
                val_metrics,
            )
            if improved:
                _save_checkpoint(
                    Path(args.output_dir) / "best.pt",
                    args,
                    head,
                    optimizer,
                    scheduler,
                    train_set.class_to_idx,
                    epoch,
                    best_top1,
                    best_top5,
                    val_metrics,
                )
        dist.barrier()

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(get_args_parser().parse_args()))
