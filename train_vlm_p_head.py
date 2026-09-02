"""Phase 1 of the VLM-delta trial: fit and freeze the real-data ``p`` head.

    z = E_VLM(x_real)          # frozen VLM, whichever backend
    L_p = -log p_phi(c | z)    # only the linear head trains

The VLM is frozen throughout; the head is a single ``Linear`` on top of a
**frozen** affine feature normalisation fitted on the same real training
features (see ``vlm_linear_heads.VLMLinearHead`` for why this is not a
BatchNorm).

Two feature backends
--------------------
``--vlm_backend timm`` (default)
    ``z`` is a frozen image encoder's pooled token -- SigLIP-SO400M's CLS by
    default, which is *already* an FD judge in the generator run, so the
    conditional term there costs one extra matmul and nothing else.

``--vlm_backend qwen_answer_state``
    ``z`` is Qwen2.5-VL-7B's hidden state at the position it would answer

        [<image>] "What is the ImageNet class of this image? Answer:"

    from -- the vector the LM head multiplies to emit the first token of the
    class name (see ``qwen_answer_state.py``).  The feature is label-aligned by
    construction rather than generically contrastive, at the price of a 7B
    forward per scored image in the generator run.

Both write the same checkpoint format and are judged by the same criterion, so
"is the answer state a better ``p``?" is answered by one number: held-out real
validation top-1.

Which layer, and why it is measured rather than chosen
-----------------------------------------------------
The last layer of a decoder is next-token-specialised; a late-middle layer is
often the better linear probe.  Every requested layer comes out of the *same*
VLM forward, so ``--vlm_probe_layers`` fits one head per candidate for the price
of one extraction pass and the winner is picked on held-out top-1.  Selection
uses the calibration half of the real validation split; the other half is
reported untouched.

Features are extracted once and cached, so the heads themselves train in seconds
and the (slow) VLM forward is paid exactly once per view.  The cached view is
the *center-cropped 256px full frame* -- deliberately identical in geometry to
the generated images the head will later have to score -- optionally plus its
horizontal mirror.

The checkpoint written here is the contract the generator run validates itself
against: VLM identity (backend, model, layer, prompt hash), feature dim, class
subset, normalisation, weights, and the fitted temperature.
``conditional_main_fd_vlm_delta.py`` refuses to launch if any of it disagrees
with the run.

Usage (SigLIP, 4 GPUs, the 100-class stride-10 subset)::

    CUDA_VISIBLE_DEVICES=1,2,3,4 \
    /home/nvidia/miniconda3/envs/fdloss/bin/python -m torch.distributed.run \
        --standalone --nproc_per_node=4 --master_port=29591 \
        train_vlm_p_head.py \
        --data_path /data/dataset/imagenet \
        --output_dir work_dirs/vlm_p_head_siglip_c100 \
        --class_ids $(seq 0 10 990 | tr '\n' ' ')

Usage (Qwen answer state)::

    bash scripts/train_vlm_p_head_qwen.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from frechet_distance.repr_models import TimmReprModel
from utils.data_util import center_crop_arr
from utils.distributed_util import (
    enable_distributed,
    get_global_rank,
    get_local_rank,
    get_world_size,
)
from vlm_linear_heads import (
    DEFAULT_VLM_INPUT_SIZE,
    DEFAULT_VLM_MODEL,
    DEFAULT_VLM_POOL,
    DEFAULT_VLM_TARGET_SIZE,
    P_HEAD_BACKEND_QWEN,
    P_HEAD_BACKEND_TIMM,
    P_HEAD_FORMAT_VERSION,
    VLMLinearHead,
    class_ids_hash,
    p_head_checksum,
)

logger = logging.getLogger("vlm_p_head")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the frozen real-data VLM linear p head for the "
                    "log q - log p conditional trial")
    p.add_argument("--data_path", type=str, default="/data/dataset/imagenet")
    p.add_argument("--output_dir", type=str, default="work_dirs/vlm_p_head_siglip_c100")

    # -- representation: must match the generator run's feature source exactly --
    p.add_argument("--vlm_backend",
                   choices=(P_HEAD_BACKEND_TIMM, P_HEAD_BACKEND_QWEN),
                   default=P_HEAD_BACKEND_TIMM,
                   help="timm = a pooled frozen image encoder (also an FD judge); "
                        "qwen_answer_state = a prompted VLM's answer hidden state")
    p.add_argument("--vlm_model_name", type=str, default=None,
                   help="timm model name, or the local Qwen2.5-VL directory "
                        "(default: the backend's own default)")
    p.add_argument("--vlm_pool_type", choices=("cls", "avg"), default=DEFAULT_VLM_POOL,
                   help="timm only; the answer-state backend returns the same "
                        "feature for either value")
    p.add_argument("--vlm_target_size", type=int, default=DEFAULT_VLM_TARGET_SIZE,
                   help="timm only: resolution the backbone resizes to internally; "
                        "must match --fd_target_sizes for that judge in the "
                        "generator run. The answer-state backend measures its own")
    p.add_argument("--vlm_input_size", type=int, default=DEFAULT_VLM_INPUT_SIZE,
                   help="resolution real images are center-cropped to before the "
                        "backbone sees them; matches the generated image size")

    # -- answer-state backend --
    p.add_argument("--vlm_prompt", type=str, default=None,
                   help="the question put to the VLM (default: the module's). It "
                        "must not name the class: z has to be the model's answer, "
                        "not a re-encoding of a label handed to it")
    p.add_argument("--vlm_probe_layers", type=int, nargs="+", default=None,
                   help="hidden_states indices to fit a head for, all from the "
                        "same forward. The winner on held-out top-1 is the one "
                        "saved (default: qwen_answer_state.DEFAULT_PROBE_LAYERS)")
    p.add_argument("--vlm_layer", type=int, default=None,
                   help="pin the saved layer instead of picking the best probe")
    p.add_argument("--vlm_attn_implementation", type=str, default="sdpa")
    p.add_argument("--head_init", choices=("random", "lm_head"), default="random",
                   help="lm_head seeds W from the LM-head row of each class "
                        "name's first token -- a zero-shot classifier on day "
                        "zero -- and reports its top-1 before any fitting")
    p.add_argument("--classnames_file", type=str, default=None,
                   help="newline-separated 1000 ImageNet class names for "
                        "--head_init lm_head (default: torchvision's table)")

    # -- class subset --
    p.add_argument("--class_ids", type=int, nargs="+", default=None,
                   help="global ImageNet ids to train on (default: all 1000). The "
                        "head is a len(class_ids)-way classifier and the generator "
                        "run must draw exactly this set")
    p.add_argument("--num_global_classes", type=int, default=1000)

    # -- feature cache --
    p.add_argument("--feature_cache_dir", type=str, default=None,
                   help="where to cache extracted features (default: <output_dir>/cache)")
    p.add_argument("--train_views", type=int, default=2, choices=(1, 2),
                   help="1 = center crop only; 2 = center crop and its mirror")
    p.add_argument("--max_per_class", type=int, default=0,
                   help="cap real training images per class (0 = all)")
    p.add_argument("--extract_batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--recompute_features", action="store_true")

    # -- head optimisation (on cached features; cheap) --
    p.add_argument("--feature_norm", choices=("standardize", "none"),
                   default="standardize")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--head_batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=0.0)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)

    # -- temperature --
    p.add_argument("--fit_temperature", action="store_true", default=True,
                   help="fit a single scalar temperature on a held-out half of the "
                        "real validation split (temperature scaling) and freeze it")
    p.add_argument("--no_fit_temperature", action="store_false", dest="fit_temperature")
    p.add_argument("--temperature", type=float, default=None,
                   help="override: use this temperature instead of fitting one")

    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--eval_only", action="store_true",
                   help="skip head training; load <output_dir>/p_head.pt and revalidate")
    return p


def resolve_backend_defaults(args) -> None:
    """Fill in the per-backend defaults the parser cannot express."""
    if args.vlm_backend == P_HEAD_BACKEND_QWEN:
        from qwen_answer_state import (
            DEFAULT_ANSWER_PROMPT,
            DEFAULT_PROBE_LAYERS,
            DEFAULT_QWEN_MODEL,
        )
        if args.vlm_model_name is None:
            args.vlm_model_name = DEFAULT_QWEN_MODEL
        if args.vlm_prompt is None:
            args.vlm_prompt = DEFAULT_ANSWER_PROMPT
        if args.vlm_probe_layers is None:
            args.vlm_probe_layers = (list(DEFAULT_PROBE_LAYERS)
                                     if args.vlm_layer is None else [args.vlm_layer])
        if args.vlm_layer is not None and args.vlm_layer not in args.vlm_probe_layers:
            args.vlm_probe_layers = sorted(set(args.vlm_probe_layers + [args.vlm_layer]))
    else:
        if args.vlm_model_name is None:
            args.vlm_model_name = DEFAULT_VLM_MODEL
        if args.vlm_probe_layers is not None or args.vlm_layer is not None:
            raise ValueError(
                "--vlm_layer/--vlm_probe_layers only apply to "
                f"--vlm_backend {P_HEAD_BACKEND_QWEN}")
        if args.head_init != "random":
            raise ValueError(
                f"--head_init {args.head_init} needs an LM head; it only applies "
                f"to --vlm_backend {P_HEAD_BACKEND_QWEN}")


# ---------------------------------------------------------------------------
# Feature sources
# ---------------------------------------------------------------------------

class _TimmFeatureSource:
    """A pooled frozen image encoder, presented as a one-entry ``{label: z}``.

    Wrapping the single feature in the same dict the multi-layer backend returns
    is what lets extraction, caching, head fitting and reporting stay one code
    path for both backends.
    """

    def __init__(self, args, device):
        self.backbone = TimmReprModel(args.vlm_model_name, device=device,
                                      target_size=args.vlm_target_size)
        self.pool_type = args.vlm_pool_type
        self.feature_dim = int(self.backbone.feat_dim)
        self.target_size = int(self.backbone.target_size)
        self.labels = (args.vlm_pool_type,)
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                          "fp32": torch.float32}[args.dtype]

    def __call__(self, images):
        with torch.autocast("cuda", dtype=self.amp_dtype,
                            enabled=self.amp_dtype != torch.float32):
            primary, secondary = self.backbone(images)
        return {self.labels[0]: (secondary if self.pool_type == "avg" else primary)}

    def release(self):
        del self.backbone


class _AnswerStateFeatureSource:
    """Qwen2.5-VL's answer hidden state at several candidate layers at once."""

    def __init__(self, args, device):
        from qwen_answer_state import QwenAnswerStateExtractor

        self.extractor = QwenAnswerStateExtractor(
            args.vlm_model_name,
            prompt=args.vlm_prompt,
            layer=args.vlm_probe_layers[-1],
            image_size=args.vlm_input_size,
            dtype=args.dtype,
            device=device,
            attn_implementation=args.vlm_attn_implementation,
        )
        self.layer_indices = [self.extractor.normalize_layer(k)
                              for k in args.vlm_probe_layers]
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("--vlm_probe_layers contains duplicates")
        self.labels = tuple(layer_label(k) for k in self.layer_indices)
        self.feature_dim = int(self.extractor.feat_dim)
        self.target_size = int(self.extractor.target_size)

    def __call__(self, images):
        states = self.extractor.answer_states(images, layers=self.layer_indices)
        return {layer_label(k): states[k] for k in self.layer_indices}

    def release(self):
        del self.extractor


def layer_label(index: int) -> str:
    return f"L{int(index)}"


def label_layer(label: str) -> int:
    return int(str(label)[1:])


def build_feature_source(args, device):
    if args.vlm_backend == P_HEAD_BACKEND_QWEN:
        return _AnswerStateFeatureSource(args, device)
    return _TimmFeatureSource(args, device)


# ---------------------------------------------------------------------------
# Dataset / extraction
# ---------------------------------------------------------------------------

class _MirrorTransform:
    """Center-crop to ``size``, to tensor in [0, 1], optionally mirrored.

    Defined at module scope (not a lambda/closure) so DataLoader workers can
    pickle it.
    """

    def __init__(self, size: int, mirror: bool) -> None:
        self.size = int(size)
        self.mirror = bool(mirror)

    def __call__(self, img):
        img = center_crop_arr(img, self.size)
        tensor = transforms.functional.to_tensor(img)
        if self.mirror:
            tensor = torch.flip(tensor, dims=(2,))
        return tensor


def _build_subset(data_path: str, split: str, class_ids, max_per_class: int,
                  transform):
    root = Path(data_path) / split
    if not root.is_dir():
        raise FileNotFoundError(f"expected ImageNet directory {root}")
    dataset = datasets.ImageFolder(root, transform=transform)
    wanted = None if class_ids is None else set(int(c) for c in class_ids)
    if wanted is None and max_per_class <= 0:
        return dataset, dataset.class_to_idx
    per_class, keep = {}, []
    for idx, (_, target) in enumerate(dataset.samples):
        if wanted is not None and target not in wanted:
            continue
        seen = per_class.get(target, 0)
        if max_per_class <= 0 or seen < max_per_class:
            keep.append(idx)
            per_class[target] = seen + 1
    if not keep:
        raise ValueError(f"no {split} images left after filtering")
    if wanted is not None and len(per_class) != len(wanted):
        missing = sorted(wanted - set(per_class))
        raise ValueError(f"no {split} images found for class ids {missing}")
    return Subset(dataset, keep), dataset.class_to_idx


def _all_gather_variable(tensor: torch.Tensor, sizes, max_n: int):
    """Gather rank-local rows of a possibly ragged tensor onto every rank."""
    pad = torch.zeros(max_n, *tensor.shape[1:], dtype=tensor.dtype, device="cuda")
    pad[: tensor.shape[0]] = tensor.cuda()
    gathered = [torch.zeros_like(pad) for _ in range(len(sizes))]
    dist.all_gather(gathered, pad)
    out = torch.cat([g[:n].cpu() for g, n in zip(gathered, sizes)])
    del gathered, pad
    return out


@torch.inference_mode()
def _extract_split(source, dataset, args, desc: str):
    """Distributed one-pass feature extraction, all-gathered onto every rank.

    Returns ``({label: fp16 features}, labels)``.  Every label comes from the
    same forward, so probing N layers costs N fp16 arrays and no extra VLM work.
    """
    rank, world_size = get_global_rank(), get_world_size()
    sampler = (DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=False, drop_last=False)
               if world_size > 1 else None)
    loader = DataLoader(dataset, batch_size=args.extract_batch_size, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True,
                        shuffle=False, drop_last=False)
    chunks = {label: [] for label in source.labels}
    labels = []
    iterator = tqdm(loader, desc=f"[rank {rank}] {desc}", disable=rank != 0)
    for images, targets in iterator:
        images = images.cuda(non_blocking=True)
        feats = source(images)
        for label in source.labels:
            chunks[label].append(feats[label].float().half().cpu())
        labels.append(targets.clone())

    out = {label: (torch.cat(parts) if parts else
                   torch.zeros(0, source.feature_dim, dtype=torch.float16))
           for label, parts in chunks.items()}
    labels = torch.cat(labels) if labels else torch.zeros(0, dtype=torch.long)

    if world_size > 1:
        # DistributedSampler pads with repeats; gather variable counts safely,
        # one label at a time so peak device memory stays at a single array.
        counts = [torch.zeros(1, dtype=torch.long, device="cuda")
                  for _ in range(world_size)]
        dist.all_gather(counts, torch.tensor([labels.shape[0]], device="cuda"))
        sizes = [int(c.item()) for c in counts]
        max_n = max(sizes)
        for label in source.labels:
            out[label] = _all_gather_variable(out[label], sizes, max_n)
        labels = _all_gather_variable(labels, sizes, max_n)
    return out, labels


def _cache_path(args, split: str) -> Path:
    cache_dir = Path(args.feature_cache_dir or (Path(args.output_dir) / "cache"))
    tag = args.vlm_model_name.replace("/", "_").replace(".", "_")
    n_cls = "all" if args.class_ids is None else str(len(args.class_ids))
    if args.vlm_backend == P_HEAD_BACKEND_QWEN:
        # The prompt is part of the feature's definition, so it is part of the
        # cache key: a reworded question must never silently reuse old features.
        tag = f"qwen_{Path(args.vlm_model_name).name[:16]}"
        return cache_dir / (
            f"{tag}_answer_{args.prompt_sha256[:12]}_in{args.vlm_input_size}"
            f"_c{n_cls}_{split}.pt"
        )
    return cache_dir / (
        f"{tag}_{args.vlm_pool_type}_in{args.vlm_input_size}_t{args.vlm_target_size}"
        f"_c{n_cls}_{split}.pt"
    )


def _cache_is_valid(blob, args, split, wanted_labels) -> bool:
    if blob.get("class_ids_sha256") != args.class_ids_sha256:
        return False
    if blob.get("vlm_model_name") != args.vlm_model_name:
        return False
    if blob.get("vlm_backend", P_HEAD_BACKEND_TIMM) != args.vlm_backend:
        return False
    if int(blob.get("train_views", 1)) != (args.train_views if split == "train" else 1):
        return False
    if args.vlm_backend == P_HEAD_BACKEND_QWEN:
        if blob.get("prompt_sha256") != args.prompt_sha256:
            return False
    return set(wanted_labels).issubset(blob.get("feats", {}))


def build_or_load_features(args, source):
    """Cached ``({label: feats}, labels)`` per split, in local class ids."""
    rank = get_global_rank()
    class_ids = args.class_ids
    out = {}
    for split in ("train", "val"):
        path = _cache_path(args, split)
        if path.is_file() and not args.recompute_features:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            if _cache_is_valid(blob, args, split, source.labels):
                if rank == 0:
                    logger.info("reusing cached %s features: %s (%d samples, %s)",
                                split, path, blob["labels_local"].shape[0],
                                sorted(blob["feats"]))
                out[split] = ({k: blob["feats"][k] for k in source.labels},
                              blob["labels_local"])
                continue
            if rank == 0:
                logger.info("cache %s does not match this configuration; recomputing",
                            path)

        views = args.train_views if split == "train" else 1
        max_per_class = args.max_per_class if split == "train" else 0
        per_view_feats, per_view_labels = [], []
        for view in range(views):
            transform = _MirrorTransform(args.vlm_input_size, mirror=(view == 1))
            dataset, _ = _build_subset(args.data_path, split, class_ids,
                                       max_per_class, transform)
            f, l = _extract_split(source, dataset, args,
                                  f"{split} features (view {view + 1}/{views})")
            per_view_feats.append(f)
            per_view_labels.append(l)
        feats = {label: torch.cat([f[label] for f in per_view_feats])
                 for label in source.labels}
        labels_global = torch.cat(per_view_labels)

        if class_ids is None:
            labels_local = labels_global
        else:
            table = torch.full((args.num_global_classes,), -1, dtype=torch.long)
            for local, global_id in enumerate(class_ids):
                table[int(global_id)] = local
            labels_local = table[labels_global]
            if int(labels_local.min()) < 0:
                raise RuntimeError("extracted a label outside the requested subset")

        if rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            torch.save({
                "feats": feats,
                "labels_local": labels_local,
                "class_ids": class_ids,
                "class_ids_sha256": args.class_ids_sha256,
                "vlm_backend": args.vlm_backend,
                "vlm_model_name": args.vlm_model_name,
                "vlm_pool_type": args.vlm_pool_type,
                "vlm_target_size": args.vlm_target_size,
                "vlm_input_size": args.vlm_input_size,
                "prompt_sha256": getattr(args, "prompt_sha256", None),
                "prompt": args.vlm_prompt,
                "train_views": views,
            }, tmp)
            os.replace(tmp, path)
            logger.info("cached %s features -> %s (%s x %s)", split, path,
                        sorted(feats), tuple(next(iter(feats.values())).shape))
        out[split] = (feats, labels_local)
        if get_world_size() > 1:
            dist.barrier()
    return out["train"], out["val"]


# ---------------------------------------------------------------------------
# Head training (rank 0 only; the heads are tiny and the features are cached)
# ---------------------------------------------------------------------------

def train_head(args, z_train, y_train, z_val, y_val, num_classes: int,
               *, select_idx=None, init_linear=None, quiet=False):
    """Fit one linear head on one cached feature set.

    ``select_idx`` restricts the epoch-selection metric to a subset of the
    validation split (the calibration half), leaving the other half untouched by
    anything that was fitted -- including the choice of epoch and, across calls,
    the choice of layer.
    """
    device = z_train.device
    torch.manual_seed(args.seed)

    head = VLMLinearHead(z_train.shape[1], num_classes,
                         feature_norm=args.feature_norm).to(device)
    # The normalisation is fitted on the real TRAIN features and frozen forever.
    head.set_feature_norm_stats(z_train.mean(0), z_train.std(0))
    if init_linear is not None:
        with torch.no_grad():
            head.linear.weight.copy_(init_linear[0])
            head.linear.bias.copy_(init_linear[1])

    z_select = z_val if select_idx is None else z_val[select_idx]
    y_select = y_val if select_idx is None else y_val[select_idx]

    optimizer = torch.optim.AdamW(head.linear.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    n = z_train.shape[0]
    steps_per_epoch = max(1, n // args.head_batch_size)
    best = {"top1": -1.0}
    best_state = None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            lr = args.lr * (epoch + 1) / max(1, args.warmup_epochs)
        else:
            span = max(1, args.epochs - args.warmup_epochs)
            progress = min(1.0, (epoch - args.warmup_epochs) / span)
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr

        head.train()
        perm = torch.randperm(n, generator=generator).to(device)
        running, seen = 0.0, 0
        for step in range(steps_per_epoch):
            idx = perm[step * args.head_batch_size:(step + 1) * args.head_batch_size]
            logits = head.logits(z_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * idx.numel()
            seen += idx.numel()

        head.eval()
        with torch.no_grad():
            select_logits = head.logits(z_select)
            select_top1 = float((select_logits.argmax(-1) == y_select).float().mean())
            select_ce = float(F.cross_entropy(select_logits, y_select))
        if select_top1 > best["top1"]:
            best = {"top1": select_top1, "ce": select_ce, "epoch": epoch}
            best_state = {k: v.detach().clone()
                          for k, v in head.linear.state_dict().items()}
        if not quiet and (epoch % 10 == 0 or epoch == args.epochs - 1):
            logger.info("epoch %3d lr=%.5f train_ce=%.4f select_ce=%.4f "
                        "select_top1=%.4f", epoch, lr, running / max(seen, 1),
                        select_ce, select_top1)

    if best_state is not None:
        head.linear.load_state_dict(best_state)
    with torch.no_grad():
        full_logits = head.logits(z_val)
        best["val_top1_full"] = float((full_logits.argmax(-1) == y_val).float().mean())
        best["val_ce_full"] = float(F.cross_entropy(full_logits, y_val))
    if not quiet:
        logger.info("best head: epoch %d, selection top1 %.4f, full-val top1 %.4f",
                    best.get("epoch", -1), best["top1"], best["val_top1_full"])
    return head, best


@torch.no_grad()
def _standardised_linear(rows: torch.Tensor, mean: torch.Tensor,
                         std: torch.Tensor, eps: float = 1e-5):
    """Express the raw logit ``u . z`` in the head's standardised coordinates.

    The head computes ``W z_hat + b`` on ``z_hat = (z - mu) / sigma``.  Setting
    ``W = u * sigma`` and ``b = u . mu`` makes that identically ``u . z``, so the
    seeded head *is* the LM head restricted to the class-name tokens -- an
    honest zero-shot classifier, not an approximation of one.
    """
    std = std.clamp_min(eps)
    weight = rows.to(mean) * std
    bias = rows.to(mean) @ mean
    return weight, bias


@torch.no_grad()
def _zero_shot_top1(init_linear, z_train, z_val, y_val, num_classes, args,
                    device) -> float:
    head = VLMLinearHead(z_train.shape[1], num_classes,
                         feature_norm=args.feature_norm).to(device)
    head.set_feature_norm_stats(z_train.mean(0), z_train.std(0))
    head.linear.weight.copy_(init_linear[0])
    head.linear.bias.copy_(init_linear[1])
    return float((head.logits(z_val).argmax(-1) == y_val).float().mean())


def fit_all_candidates(args, train_feats, train_labels, val_feats, val_labels,
                       num_classes, calib_idx, holdout_idx, device,
                       init_weights=None):
    """One head per cached feature label; the table is the layer measurement."""
    y_train = train_labels.to(device)
    y_val = val_labels.to(device)
    results, heads = [], {}
    for label in sorted(train_feats, key=_label_sort_key):
        z_train = train_feats[label].to(device=device, dtype=torch.float32)
        z_val = val_feats[label].to(device=device, dtype=torch.float32)
        init_linear, zero_shot = None, None
        if init_weights is not None:
            init_linear = _standardised_linear(init_weights[label], z_train.mean(0),
                                               z_train.std(0))
            zero_shot = _zero_shot_top1(init_linear, z_train, z_val, y_val,
                                        num_classes, args, device)
        logger.info("--- fitting head on %s (d=%d) ---", label, z_train.shape[1])
        head, best = train_head(args, z_train, y_train, z_val, y_val, num_classes,
                                select_idx=calib_idx, init_linear=init_linear,
                                quiet=len(train_feats) > 1)
        with torch.no_grad():
            logits = head.logits(z_val)
            pred = logits.argmax(-1)
            row = {
                "label": label,
                "epoch": best.get("epoch", -1),
                "calib_top1": best["top1"],
                "holdout_top1": float((pred[holdout_idx] == y_val[holdout_idx])
                                      .float().mean()),
                "val_top1_full": best["val_top1_full"],
                "val_ce_full": best["val_ce_full"],
            }
        if zero_shot is not None:
            row["zero_shot_top1"] = zero_shot
        results.append(row)
        heads[label] = (head, best)
        logger.info("%s: calib %.4f | holdout %.4f | full-val %.4f (epoch %d)",
                    label, row["calib_top1"], row["holdout_top1"],
                    row["val_top1_full"], row["epoch"])
        del z_train, z_val
        torch.cuda.empty_cache()
    return results, heads


def _label_sort_key(label: str):
    try:
        return (0, label_layer(label))
    except ValueError:
        return (1, label)


def print_candidate_table(results, chosen_label: str) -> None:
    bar = "=" * 78
    has_zs = any("zero_shot_top1" in r for r in results)
    lines = ["", bar, "candidate feature comparison (one head each, same forward)",
             bar,
             "  feature   epoch   calib top1   holdout top1   full-val top1   full-val ce"
             + ("   zero-shot" if has_zs else "")]
    for row in results:
        mark = "*" if row["label"] == chosen_label else " "
        line = (f" {mark}{row['label']:>8}  {row['epoch']:5d}   {row['calib_top1']:10.4f}"
                f"   {row['holdout_top1']:12.4f}   {row['val_top1_full']:13.4f}"
                f"   {row['val_ce_full']:11.4f}")
        if has_zs:
            line += f"   {row.get('zero_shot_top1', float('nan')):9.4f}"
        lines.append(line)
    lines += ["", "  * = saved. Selection is on the calibration half of real val;",
              "    holdout top1 is untouched by epoch or layer selection.", bar, ""]
    print("\n".join(lines), flush=True)


def fit_temperature(head: VLMLinearHead, z: torch.Tensor, y: torch.Tensor) -> float:
    """Standard temperature scaling: one scalar, fitted by NLL on held-out real data."""
    with torch.no_grad():
        logits = head.logits(z)
    log_t = torch.zeros(1, device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args) -> int:
    enable_distributed()
    rank, local_rank, world_size = get_global_rank(), get_local_rank(), get_world_size()
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    resolve_backend_defaults(args)
    if args.class_ids is not None:
        args.class_ids = sorted(int(c) for c in args.class_ids)
        if len(set(args.class_ids)) != len(args.class_ids):
            raise ValueError("--class_ids contains duplicates")
    num_classes = (args.num_global_classes if args.class_ids is None
                   else len(args.class_ids))
    args.class_ids_sha256 = class_ids_hash(
        args.class_ids if args.class_ids is not None
        else list(range(args.num_global_classes)))

    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    source = build_feature_source(args, device)
    feature_dim = int(source.feature_dim)
    args.vlm_target_size = int(source.target_size)
    if args.vlm_backend == P_HEAD_BACKEND_QWEN:
        args.prompt_sha256 = source.extractor.prompt_sha256
        args.rendered_prompt = source.extractor.rendered_prompt
        # Resolve any negative layer index while the extractor -- the only thing
        # that knows how many hidden states there are -- is still alive.
        args.pinned_layer = (None if args.vlm_layer is None
                             else source.extractor.normalize_layer(args.vlm_layer))
    else:
        args.prompt_sha256 = None
        args.rendered_prompt = None
        args.pinned_layer = None

    if rank == 0:
        logger.info("backend=%s VLM=%s input=%d target=%d d=%d | %d classes | "
                    "candidates %s", args.vlm_backend, args.vlm_model_name,
                    args.vlm_input_size, args.vlm_target_size, feature_dim,
                    num_classes, list(source.labels))

    # -- optional zero-shot LM-head initialisation, before the VLM is released --
    init_weights = None
    if args.head_init == "lm_head":
        init_weights = _lm_head_init_weights(args, source, num_classes)

    (train_feats, train_labels), (val_feats, val_labels) = build_or_load_features(
        args, source)
    source.release()
    del source
    torch.cuda.empty_cache()

    if rank != 0:
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return 0

    logger.info("features: train %s, val %s per candidate (feature_dim=%d)",
                tuple(next(iter(train_feats.values())).shape),
                tuple(next(iter(val_feats.values())).shape), feature_dim)

    # Split the real validation set: the first half selects (epoch, layer) and
    # calibrates the temperature; the second half is untouched by anything fitted.
    generator = torch.Generator().manual_seed(12345)
    perm = torch.randperm(val_labels.shape[0], generator=generator)
    half = val_labels.shape[0] // 2
    calib_idx, holdout_idx = perm[:half].cuda(), perm[half:].cuda()

    ckpt_path = Path(args.output_dir) / "p_head.pt"
    candidates = []
    if args.eval_only:
        from vlm_linear_heads import build_head_from_checkpoint, load_p_head_checkpoint
        ckpt = load_p_head_checkpoint(ckpt_path)
        head = build_head_from_checkpoint(ckpt, device="cuda")
        temperature = float(ckpt["temperature"])
        best = ckpt.get("train_metadata", {}).get("best", {})
        chosen_label = ckpt.get("train_metadata", {}).get("feature_label",
                                                          list(val_feats)[0])
    else:
        candidates, heads = fit_all_candidates(
            args, train_feats, train_labels, val_feats, val_labels, num_classes,
            calib_idx, holdout_idx, torch.device("cuda"), init_weights=init_weights)
        if args.pinned_layer is not None:
            chosen_label = layer_label(args.pinned_layer)
            if chosen_label not in heads:
                raise ValueError(
                    f"--vlm_layer {args.vlm_layer} resolved to {chosen_label}, "
                    f"which is not among the fitted candidates {sorted(heads)}")
            logger.info("layer pinned by --vlm_layer: %s", chosen_label)
        else:
            chosen_label = max(candidates, key=lambda r: r["calib_top1"])["label"]
        head, best = heads[chosen_label]
        print_candidate_table(candidates, chosen_label)

        z_val_chosen = val_feats[chosen_label].to(device="cuda", dtype=torch.float32)
        if args.temperature is not None:
            temperature = float(args.temperature)
            logger.info("using the supplied temperature T=%.4f (no fit)", temperature)
        elif args.fit_temperature:
            temperature = fit_temperature(head, z_val_chosen[calib_idx],
                                          val_labels.cuda()[calib_idx])
            logger.info("fitted temperature on %d held-out real images: T=%.4f",
                        calib_idx.numel(), temperature)
        else:
            temperature = 1.0

    z_val_chosen = val_feats[chosen_label].to(device="cuda", dtype=torch.float32)
    y_val_cuda = val_labels.cuda()

    # -- save --
    class_ids = args.class_ids if args.class_ids is not None else list(
        range(args.num_global_classes))
    payload = {
        "format_version": P_HEAD_FORMAT_VERSION,
        "vlm_backend": args.vlm_backend,
        "vlm_model_name": args.vlm_model_name,
        "vlm_pool_type": args.vlm_pool_type,
        "vlm_target_size": int(args.vlm_target_size),
        "vlm_input_size": int(args.vlm_input_size),
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "class_ids": [int(c) for c in class_ids],
        "class_ids_sha256": class_ids_hash(class_ids),
        "feature_norm": args.feature_norm,
        "feature_mean": head.feature_mean.detach().cpu(),
        "feature_std": head.feature_std.detach().cpu(),
        "head_state_dict": {k: v.detach().cpu()
                            for k, v in head.linear.state_dict().items()},
        "temperature": float(temperature),
        "train_metadata": {
            "best": best,
            "feature_label": chosen_label,
            "candidates": candidates,
            "num_train_features": int(train_labels.shape[0]),
            "num_val_features": int(val_labels.shape[0]),
            "train_views": int(args.train_views),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "head_batch_size": int(args.head_batch_size),
            "head_init": args.head_init,
            "seed": int(args.seed),
            "torch_version": str(torch.__version__),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "args": {k: v for k, v in vars(args).items()},
    }
    if args.vlm_backend == P_HEAD_BACKEND_QWEN:
        payload["vlm_layer"] = label_layer(chosen_label)
        payload["vlm_prompt"] = args.vlm_prompt
        payload["vlm_prompt_sha256"] = args.prompt_sha256
        payload["vlm_rendered_prompt"] = args.rendered_prompt
    payload["p_head_sha256"] = p_head_checksum(payload)
    if not args.eval_only:
        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, ckpt_path)
        logger.info("wrote %s (sha256 %s)", ckpt_path, payload["p_head_sha256"][:16])

    # -- mandatory validation report --
    from validate_vlm_p_head import build_report, print_report, save_confusion
    report = build_report(
        head, temperature, z_val_chosen, y_val_cuda,
        class_ids=[int(c) for c in class_ids],
        holdout_idx=holdout_idx, calib_idx=calib_idx,
    )
    report["checkpoint"] = str(ckpt_path)
    report["p_head_sha256"] = payload["p_head_sha256"]
    report["vlm_backend"] = args.vlm_backend
    report["feature_label"] = chosen_label
    report["candidates"] = candidates
    if args.vlm_backend == P_HEAD_BACKEND_QWEN:
        report["vlm_layer"] = label_layer(chosen_label)
        report["vlm_prompt"] = args.vlm_prompt
        report["vlm_prompt_sha256"] = args.prompt_sha256
    save_confusion(report, Path(args.output_dir) / "p_head_confusion.npy")
    report_path = Path(args.output_dir) / "p_head_validation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=1)
    print_report(report, header=f"p-head validation  ({ckpt_path})")
    logger.info("wrote %s", report_path)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def _lm_head_init_weights(args, source, num_classes):
    """``{label: W0}`` seeding each head from the LM-head rows of the class names.

    The answer state is the vector the LM head multiplies to emit the first token
    of the answer, so the LM-head row for a class name's first token already *is*
    a direction meaning "this class".  Because the head sees the standardised
    ``z_hat = (z - mu) / sigma``, ``W = u * sigma`` and ``b = u . mu`` make it
    compute ``u . z`` exactly (see :func:`_standardised_linear`), so the seeded
    head *is* the frozen LM head restricted to the class-name tokens.
    """
    from vlm_judge import _load_class_names

    if args.vlm_backend != P_HEAD_BACKEND_QWEN:
        raise ValueError("--head_init lm_head requires the answer-state backend")
    names = _load_class_names(args.classnames_file, args.num_global_classes)
    class_ids = (args.class_ids if args.class_ids is not None
                 else list(range(args.num_global_classes)))
    subset = [names[int(c)] for c in class_ids]
    token_ids = source.extractor.class_first_token_ids(subset)
    unique = int(torch.unique(token_ids).numel())
    if unique < len(subset):
        logger.warning(
            "[head_init] only %d/%d class names have a distinct first token; the "
            "LM-head seed cannot separate the collisions on its own",
            unique, len(subset))
    rows = source.extractor.lm_head_rows(token_ids)
    if rows.shape != (num_classes, source.feature_dim):
        raise RuntimeError(
            f"LM-head rows have shape {tuple(rows.shape)}, expected "
            f"{(num_classes, source.feature_dim)}")
    logger.info("[head_init] seeded from %d LM-head rows (%d distinct tokens)",
                rows.shape[0], unique)
    # Each candidate layer has its own (mu, sigma), so the raw LM-head rows are
    # returned here and _standardised_linear rescales them per layer.
    return {label: rows.clone() for label in source.labels}


if __name__ == "__main__":
    raise SystemExit(main(get_args_parser().parse_args()))
