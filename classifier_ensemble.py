"""Multi-classifier ensemble for the conditional correction term log p(c|x).

Post-mortem of the single-CLIP runs (cond-v2 eval): the generator drove the
frozen training CLIP to p(c|x) ~ 0.96 while a held-out CLIP scored the same
samples at 22.5% top-1 and the external supervised ResNet-50 at 5.6% — the
class embeddings learned CLIP-specific adversarial directions, not class
content. The one-sided +grad log p(c|x) term is a white-box attack on whatever
frozen classifier supplies it, and a single model's adversarial subspace is
huge.

This module makes the term harder to cheat by requiring *consensus* across
classifiers that differ in architecture (ViT vs modern CNN) AND training
paradigm (contrastive image-text vs supervised cross-entropy). A perturbation
that simultaneously fools all members — under EOT augmentation, with
per-member confidence caps removing the incentive to over-optimize any single
member — is far closer to genuine class evidence than one that fools CLIP
alone.

Spec grammar (comma-separated members)::

    kind:name[:k=v]*
      kind = clip | timm
      name = open_clip arch (clip) or timm model name (timm)
      k=v  = w=<weight>          member weight in the 'mean' combiner
             cap=<logp cap, <0>  per-member confidence cap (default: global
                                 --cond_target_logp)
             T=<temperature>     divide member logits by T (>1 flattens)
             scale=<clip scale>  clip only: logit-scale override (0 = native)
             pre=<pretrained>    clip only: open_clip pretrained tag

Example::

    clip:ViT-L-14:cap=-0.69,timm:convnext_base.fb_in22k_ft_in1k:cap=-0.69,timm:deit3_base_patch16_224.fb_in22k_ft_in1k:cap=-0.69

Combination modes (both are AND-semantics — every member must be satisfied;
never average probabilities, which is OR-semantics and lets the generator
satisfy the easiest member only):

    mean: sum_i w_i * min(logp_i, cap_i) / sum_i w_i — log of a weighted
          geometric mean (product of experts). A member that reaches its cap
          stops contributing gradient while unsatisfied members keep pushing.
    min:  min_i min(logp_i, cap_i) — all gradient goes to the currently
          weakest member (weights unused). Saturation-proof by construction.

NEVER put the eval classifier (torchvision resnet50 IMAGENET1K_V2, used by
eval_class_accuracy.py and ProbeClassifier below) in the training spec — it is
the held-out judge that keeps the accuracy numbers honest.
"""

import logging
import re

import torch
import torch.nn.functional as F

logger = logging.getLogger("FD_loss")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def augment_view(images, noise_std, crop_min):
    """One differentiable EOT view of ``images`` ([0,1], NCHW): random
    horizontal flip, random-resized-crop, additive Gaussian noise. Same
    transform family as CLIPClassifier._augment_view; each member draws its
    own views so the ensemble members are decorrelated per step."""
    x = images
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), torch.flip(x, dims=[3]), x)
    if crop_min < 1.0:
        B, C, H, W = x.shape
        scale = float(torch.empty(1).uniform_(crop_min, 1.0))
        ch, cw = max(1, int(round(H * scale))), max(1, int(round(W * scale)))
        top = int(torch.randint(0, H - ch + 1, (1,)))
        left = int(torch.randint(0, W - cw + 1, (1,)))
        x = x[:, :, top:top + ch, left:left + cw]
        x = F.interpolate(x, size=(H, W), mode="bicubic",
                          align_corners=False, antialias=True)
    if noise_std > 0:
        x = (x + noise_std * torch.randn_like(x)).clamp(0, 1)
    return x


def _parse_spec(spec, default_cap):
    """'kind:name[:k=v]*, ...' -> list of (kind, name, opts) with defaults."""
    members = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        fields = raw.split(":")
        if len(fields) < 2:
            raise ValueError(f"member '{raw}' must be 'kind:name[:k=v]*'")
        kind, name = fields[0].strip().lower(), fields[1].strip()
        if kind not in ("clip", "timm"):
            raise ValueError(f"unknown member kind '{kind}' in '{raw}' "
                             f"(expected clip|timm)")
        opts = {"w": 1.0, "cap": float(default_cap), "T": 1.0,
                "scale": 0.0, "pre": "openai"}
        for f in fields[2:]:
            if "=" not in f:
                raise ValueError(f"bad option '{f}' in '{raw}' (need k=v)")
            k, v = (s.strip() for s in f.split("=", 1))
            if k not in opts:
                raise ValueError(f"unknown option '{k}' in '{raw}' "
                                 f"(expected {sorted(opts)})")
            opts[k] = v if k == "pre" else float(v)
        members.append((kind, name, opts))
    if not members:
        raise ValueError("empty --cond_classifiers spec")
    return members


def _mk_tag(kind, name, existing):
    """Short unique log key: 'timm_convnext_base', 'clip_ViT-L-14', ..."""
    base = name.split(".")[0] if kind == "timm" else name
    tag = f"{kind}_" + re.sub(r"[^A-Za-z0-9]+", "-", base).strip("-")
    t, j = tag, 2
    while t in existing:
        t, j = f"{tag}{j}", j + 1
    return t


class _ClipMember(torch.nn.Module):
    """Open_clip zero-shot member — wraps CLIPClassifier for its prompt
    ensemble / text-feature cache, exposing differentiable full log-probs."""

    def __init__(self, name, opts, clip_kwargs):
        super().__init__()
        from clip_classifier import CLIPClassifier
        self.clip = CLIPClassifier(model_name=name, pretrained=opts["pre"],
                                   logit_scale_override=opts["scale"],
                                   **clip_kwargs)
        self.temperature = opts["T"]
        self.image_size = self.clip.image_size

    def log_probs(self, images01):
        c = self.clip
        x = c._preprocess(images01).to(c.dtype)
        feats = F.normalize(c.model.encode_image(x), dim=-1)
        logits = c.logit_scale * (feats @ c.text_features.t())
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return torch.log_softmax(logits.float(), dim=-1)


class _TimmMember(torch.nn.Module):
    """Frozen supervised timm classifier (direct 1000-way head, no prompts).
    Preprocessing (input size, mean/std) comes from the model's own data
    config; label order is the standard ImageNet-1k index order, i.e. the same
    sorted-WNID order the ImageFolder training labels use."""

    def __init__(self, name, opts, num_classes, device, dtype):
        super().__init__()
        import timm
        from timm.data import resolve_model_data_config
        model = timm.create_model(name, pretrained=True)
        got = getattr(model, "num_classes", None)
        if got != num_classes:
            raise ValueError(f"timm model '{name}' has {got} classes, "
                             f"expected {num_classes}")
        cfg = resolve_model_data_config(model)
        self.image_size = int(cfg["input_size"][-1])
        self.dtype = dtype
        self.model = model.eval().requires_grad_(False).to(device=device,
                                                           dtype=dtype)
        self.register_buffer("mean", torch.tensor(cfg["mean"],
                             device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"],
                             device=device).view(1, 3, 1, 1))
        self.temperature = opts["T"]

    def log_probs(self, images01):
        x = images01
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                              mode="bicubic", align_corners=False,
                              antialias=True)
        x = ((x - self.mean) / self.std).to(self.dtype)
        logits = self.model(x).float()
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return torch.log_softmax(logits, dim=-1)


class ClassifierEnsemble(torch.nn.Module):
    """Drop-in replacement for CLIPClassifier.log_p_c_given_x backed by N
    frozen classifiers. Per-member mean log p is published in ``last_stats``
    after every call (keys 'logp_<tag>') so the training loop can log the
    tell-tale cheating signature: one member saturating while others lag.
    """

    def __init__(self, spec, num_classes=1000, device="cuda", dtype="bf16",
                 default_cap=0.0, combine="mean", members_per_step=0,
                 clip_kwargs=None):
        super().__init__()
        if combine not in ("mean", "min"):
            raise ValueError(f"combine must be mean|min, got '{combine}'")
        self.combine = combine
        self.num_classes = num_classes
        self.members_per_step = int(members_per_step)
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                       "fp32": torch.float32}[dtype]
        clip_kwargs = dict(clip_kwargs or {})
        clip_kwargs.update(num_classes=num_classes, device=device, dtype=dtype)

        self.members = torch.nn.ModuleList()
        self.tags, self.weights, self.caps = [], [], []
        for kind, name, opts in _parse_spec(spec, default_cap):
            if kind == "clip":
                m = _ClipMember(name, opts, clip_kwargs)
            else:
                m = _TimmMember(name, opts, num_classes, device, torch_dtype)
            tag = _mk_tag(kind, name, self.tags)
            self.members.append(m)
            self.tags.append(tag)
            self.weights.append(opts["w"])
            self.caps.append(opts["cap"])
            n_par = sum(p.numel() for p in m.parameters()) / 1e6
            logger.info(f"[Ensemble] member '{tag}' ({name}): {n_par:.0f}M "
                        f"params, input {m.image_size}px, w={opts['w']}, "
                        f"cap={opts['cap']}, T={opts['T']}")
        logger.info(f"[Ensemble] {len(self.members)} members, combine="
                    f"{combine}, members_per_step="
                    f"{self.members_per_step or 'all'}")
        # ponly's fd_train_step must not re-apply the global cap on the
        # combined output — member caps from the spec may differ from it.
        self.handles_cap = True
        self._rr = 0          # round-robin cursor; same call count on every
        self.last_stats = {}  # DDP rank keeps member selection in sync

    def _select(self):
        n, k = len(self.members), self.members_per_step
        if k <= 0 or k >= n:
            return list(range(n))
        start = self._rr % n
        self._rr += k
        return [(start + j) % n for j in range(k)]

    @torch._dynamo.disable  # run eager even if fd_train_step is compiled
    def log_p_c_given_x(self, images, labels, eot_views=1, noise_std=0.0,
                        crop_min=1.0):
        idx = torch.arange(labels.shape[0], device=images.device)
        views = max(1, int(eot_views))
        use_aug = (views > 1) or (noise_std > 0) or (crop_min < 1.0)

        stats, capped = {}, []
        for i in self._select():
            m = self.members[i]
            acc = None
            for _ in range(views):
                x = augment_view(images, noise_std, crop_min) if use_aug else images
                lp = m.log_probs(x)[idx, labels]
                acc = lp if acc is None else acc + lp
            lp = acc / views
            stats[f"logp_{self.tags[i]}"] = float(lp.mean().detach())
            cap = self.caps[i]
            capped.append((self.weights[i],
                           torch.clamp(lp, max=cap) if cap < 0 else lp))
        self.last_stats = stats

        if self.combine == "min":
            return torch.stack([lp for _, lp in capped], 0).amin(0)
        wsum = sum(w for w, _ in capped)
        out = capped[0][0] * capped[0][1]
        for w, lp in capped[1:]:
            out = out + w * lp
        return out / wsum

    @torch.no_grad()
    def smoke_test_uniform(self, batch_size=16, image_size=256):
        """On noise images every member's per-class log-prob should sit near
        -log(num_classes); a member far off has broken preprocessing/labels."""
        x = torch.rand(batch_size, 3, image_size, image_size,
                       device=next(self.parameters()).device)
        per_member, mx = {}, -float("inf")
        for tag, m in zip(self.tags, self.members):
            lp = m.log_probs(x)
            per_member[tag] = float(lp.mean())
            mx = max(mx, float(lp.max()))
            logger.info(f"[Ensemble] smoke '{tag}': mean log p/class="
                        f"{per_member[tag]:.3f}")
        return {
            "mean_logp_per_class": sum(per_member.values()) / len(per_member),
            "uniform_ref": float(-torch.log(torch.tensor(
                float(self.num_classes)))),
            "max_logp": mx,
            "per_member": per_member,
        }


class ProbeClassifier(torch.nn.Module):
    """Held-out canary: torchvision ResNet-50 IMAGENET1K_V2 — the exact
    weights eval_class_accuracy.py judges with. Logged at diag steps, never in
    the loss. If the ensemble members' logp rise while probe_top1 stays at
    chance (0.001), the ensemble is being gamed and the run is already dead —
    kill it early instead of discovering it after 28h at eval time."""

    def __init__(self, device="cuda"):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model = m.eval().requires_grad_(False).to(device)  # fp32
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN,
                             device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD,
                             device=device).view(1, 3, 1, 1))

    @torch._dynamo.disable
    @torch.no_grad()
    def stats(self, images01, labels):
        x = F.interpolate(images01.float(), size=(224, 224), mode="bicubic",
                          align_corners=False, antialias=True)
        logits = self.model((x - self.mean) / self.std)
        logp = torch.log_softmax(logits, dim=-1)
        idx = torch.arange(labels.shape[0], device=labels.device)
        top5 = logits.topk(5, dim=-1).indices.eq(labels.view(-1, 1)).any(-1)
        # Rank of the true class (1 = best, num_classes = worst). top1/top5 are
        # quantised to 0 for a long time on a 1000-way problem with a small
        # batch, so they cannot show partial progress; the mean rank moves
        # continuously and is the usable early conditioning signal. Chance is
        # (num_classes + 1) / 2 = 500.5 for ImageNet-1k.
        true_logit = logits.gather(1, labels.view(-1, 1))
        rank = (logits > true_logit).sum(-1) + 1
        return {
            "probe_top1": float((logits.argmax(-1) == labels).float().mean()),
            "probe_top5": float(top5.float().mean()),
            "probe_logp": float(logp[idx, labels].mean()),
            "probe_rank": float(rank.float().mean()),
        }
