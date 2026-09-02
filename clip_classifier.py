"""CLIP zero-shot classifier for the conditional correction term in FD-loss.

This implements the classifier-guidance-style correction
    +[ ∇_x log p(c|x) - ∇_x log q(c|x) ]
of which we keep only the *one-sided* p(c|x) term (the q(c|x) side is dropped,
as in classifier guidance / NP-Edit). p(c|x) is approximated by a frozen,
pretrained CLIP used as a zero-shot classifier:

    log p(c_k | x) ≈ log softmax_k( s * <φ_img(x), φ_text(c_k)> )

where ``s`` is CLIP's own learned logit scale (= 1/τ ≈ 100), φ_text(c_k) is a
class-prompt text embedding (cached once for the 1000 ImageNet classes), and
the gradient flows back through CLIP's *image* encoder into ``x`` (and hence the
generator). CLIP's weights stay frozen — frozen params do not break the graph,
they simply do not accumulate their own gradient.

Why open_clip (not the repo's timm CLIP): the timm CLIP wrapper in
``frechet_distance/repr_models.py`` exposes only the image tower. Zero-shot
classification needs the text tower + class prompts too, which open_clip
bundles. Install with ``pip install open_clip_torch``.
"""

import logging
import os

import torch
import torch.nn.functional as F

logger = logging.getLogger("FD_loss")

# OpenAI CLIP normalization (NOT ImageNet's). Used as a fallback when the
# constants cannot be read off the model's own preprocess transform.
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_SINGLE_TEMPLATE = [lambda c: f"a photo of a {c}."]


def _load_imagenet_metadata():
    """Resolve (classnames, templates) from open_clip across versions."""
    classnames = templates = None
    try:  # newer open_clip re-exports at top level
        from open_clip import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
        classnames, templates = IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
    except Exception:
        try:  # older open_clip keeps them in zero_shot_metadata
            from open_clip.zero_shot_metadata import (
                IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES,
            )
            classnames, templates = IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
        except Exception:
            pass
    return classnames, templates


def _extract_preprocess_stats(preprocess):
    """Pull (image_size, mean, std) off an open_clip preprocess Compose.

    Falls back to OpenAI CLIP constants / 224 if the transform can't be parsed.
    """
    image_size, mean, std = 224, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
    try:
        for t in preprocess.transforms:
            name = type(t).__name__
            if name == "Normalize":
                mean, std = tuple(t.mean), tuple(t.std)
            elif name in ("Resize", "RandomResizedCrop", "CenterCrop"):
                size = t.size if hasattr(t, "size") else None
                if isinstance(size, (tuple, list)):
                    image_size = int(size[0])
                elif isinstance(size, int):
                    image_size = int(size)
    except Exception:
        pass
    return image_size, mean, std


class CLIPClassifier(torch.nn.Module):
    """Frozen CLIP used as a zero-shot p(c|x) estimator.

    Args:
        model_name:  open_clip architecture, e.g. ``ViT-L-14`` (default) or
                     ``ViT-B-32`` (cheaper, lower zero-shot accuracy).
        pretrained:  open_clip pretrained tag, e.g. ``openai``.
        dtype:       compute/param dtype (bf16 default to save memory).
        single_template: use one prompt instead of the 80-prompt ensemble.
        cache_dir:   where to cache the (num_classes, d) text-embedding tensor.
        classnames_file: optional newline-separated class-name override.
        num_classes: number of conditioning classes (1000 for ImageNet).
        device:      device to place the model and cached embeddings on.
    """

    def __init__(
        self,
        model_name="ViT-L-14",
        pretrained="openai",
        dtype="bf16",
        single_template=False,
        cache_dir="data/clip_text_cache",
        classnames_file=None,
        num_classes=1000,
        device="cuda",
        logit_scale_override=0.0,
    ):
        super().__init__()
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "open_clip is required for the conditional correction term. "
                "Install it with `pip install open_clip_torch`."
            ) from e

        self.device = device
        self.num_classes = num_classes
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                      "fp32": torch.float32}[dtype]

        # OpenAI CLIP was trained with QuickGELU; open_clip's plain "ViT-L-14"
        # config uses GELU and only emits a soft warning, silently degrading the
        # zero-shot classifier. Prefer the *-quickgelu variant for the openai tag.
        if pretrained == "openai" and "quickgelu" not in model_name.lower():
            qg = f"{model_name}-quickgelu"
            try:
                if qg in open_clip.list_models():
                    logger.info(f"[CLIPClassifier] openai tag: using {qg} "
                                f"(QuickGELU) instead of {model_name}")
                    model_name = qg
            except Exception:
                pass

        logger.info(f"[CLIPClassifier] loading {model_name} ({pretrained}) in {dtype} ...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        # Freeze hard: no grad on params, eval mode (disables dropout), cast dtype.
        model = model.eval().requires_grad_(False).to(device=device, dtype=self.dtype)
        self.model = model
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # CLIP's learned temperature (1/τ ≈ 100). Detached scalar — frozen.
        # The native scale (~99) makes the 1000-way softmax razor-sharp, which is
        # exactly what lets the generator drive p(c|x)->1 adversarially. Allow an
        # override to soften it (smaller scale = flatter softmax = weaker, less
        # exploitable per-pixel gradient).
        self.native_logit_scale = float(model.logit_scale.exp().detach().cpu())
        if logit_scale_override and logit_scale_override > 0:
            self.logit_scale = float(logit_scale_override)
            logger.info(f"[CLIPClassifier] logit_scale override: "
                        f"{self.native_logit_scale:.2f} -> {self.logit_scale:.2f}")
        else:
            self.logit_scale = self.native_logit_scale

        image_size, mean, std = _extract_preprocess_stats(preprocess)
        self.image_size = image_size
        self.register_buffer("mean", torch.tensor(mean, device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, device=device).view(1, 3, 1, 1))
        logger.info(f"[CLIPClassifier] image_size={image_size}, logit_scale={self.logit_scale:.2f}, "
                    f"mean={tuple(round(m, 4) for m in mean)}")

        self._resolve_classnames(classnames_file)
        self.templates = _SINGLE_TEMPLATE if single_template else self._resolve_templates()
        logger.info(f"[CLIPClassifier] {len(self.classnames)} classes, "
                    f"{len(self.templates)} prompt template(s) "
                    f"({'single' if single_template else 'ensemble'})")

        # (num_classes, d) text embeddings on GPU in self.dtype — built once.
        text_features = self._load_or_build_text_features(
            model_name, pretrained, single_template, cache_dir
        )
        self.register_buffer("text_features", text_features.to(device=device, dtype=self.dtype))
        logger.info(f"[CLIPClassifier] text_features: {tuple(self.text_features.shape)}")

    # -- metadata ----------------------------------------------------------
    def _resolve_classnames(self, classnames_file):
        if classnames_file is not None:
            with open(classnames_file) as f:
                self.classnames = [ln.strip() for ln in f if ln.strip()]
        else:
            classnames, _ = _load_imagenet_metadata()
            if classnames is None:
                raise RuntimeError(
                    "Could not load ImageNet class names from open_clip. "
                    "Upgrade open_clip or pass --clip_classnames_file with one "
                    "class name per line."
                )
            self.classnames = list(classnames)
        if len(self.classnames) != self.num_classes:
            logger.warning(f"[CLIPClassifier] got {len(self.classnames)} class names "
                           f"but num_classes={self.num_classes}; using "
                           f"first {self.num_classes}.")
            self.classnames = self.classnames[: self.num_classes]

    def _resolve_templates(self):
        _, templates = _load_imagenet_metadata()
        if templates is None:
            logger.warning("[CLIPClassifier] OpenAI template ensemble unavailable; "
                           "falling back to single template.")
            return _SINGLE_TEMPLATE
        return list(templates)

    # -- text embedding cache ---------------------------------------------
    def _cache_path(self, cache_dir, model_name, pretrained, single_template):
        tag = "single" if single_template else "ensemble"
        safe = f"{model_name}_{pretrained}".replace("/", "-").replace(" ", "")
        return os.path.join(cache_dir, f"clip_text_{safe}_{tag}_n{self.num_classes}.pt")

    def _load_or_build_text_features(self, model_name, pretrained, single_template, cache_dir):
        from utils.distributed_util import is_main_process

        path = self._cache_path(cache_dir, model_name, pretrained, single_template)
        dist_on = torch.distributed.is_available() and torch.distributed.is_initialized()

        if os.path.exists(path):
            logger.info(f"[CLIPClassifier] loading cached text features from {path}")
            return torch.load(path, map_location="cpu").float()

        # Build once (rank 0 only when distributed), then everyone loads.
        if not dist_on or is_main_process():
            logger.info(f"[CLIPClassifier] building text features -> {path} "
                        f"(this is a one-time cost) ...")
            feats = self._build_text_features()
            os.makedirs(cache_dir, exist_ok=True)
            tmp = path + ".tmp"
            torch.save(feats.cpu().float(), tmp)
            os.replace(tmp, path)  # atomic
        if dist_on:
            torch.distributed.barrier()
        return torch.load(path, map_location="cpu").float()

    @torch.no_grad()
    def _build_text_features(self):
        """Standard open_clip zero-shot classifier weights: encode every
        prompt, L2-normalize, average over templates, L2-normalize again."""
        weights = []
        for i, classname in enumerate(self.classnames):
            prompts = [t(classname) for t in self.templates]
            tokens = self.tokenizer(prompts).to(self.device)
            emb = self.model.encode_text(tokens)            # (T, d)
            emb = F.normalize(emb.float(), dim=-1)
            emb = F.normalize(emb.mean(0), dim=-1)          # ensemble -> (d,)
            weights.append(emb)
            if (i + 1) % 200 == 0:
                logger.info(f"[CLIPClassifier]   encoded {i + 1}/{len(self.classnames)} classes")
        return torch.stack(weights, dim=0)                  # (num_classes, d)

    # -- preprocessing -----------------------------------------------------
    def _preprocess(self, images):
        """images in [0,1], NCHW -> resize to CLIP input -> CLIP-normalize.

        Keeps the input dtype so the cast to self.dtype (and its backward) is
        explicit and the gradient returns in the generator's dtype.
        """
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            images = F.interpolate(images, size=(self.image_size, self.image_size),
                                   mode="bicubic", align_corners=False, antialias=True)
        return (images - self.mean) / self.std

    # -- EOT augmentation (anti-adversarial) -------------------------------
    def _augment_view(self, images, noise_std, crop_min):
        """One differentiable augmented view of ``images`` ([0,1], NCHW).

        Expectation-over-transformations: a perturbation that fools CLIP on one
        fixed view rarely survives random flip / crop / noise, so averaging
        log p(c|x) over augmented views forces genuinely class-discriminative
        content rather than a single adversarial texture. All ops are
        differentiable so the gradient still reaches the generator.
        """
        x = images
        # per-sample random horizontal flip
        flip = torch.rand(x.shape[0], device=x.device) < 0.5
        x = torch.where(flip.view(-1, 1, 1, 1), torch.flip(x, dims=[3]), x)
        # random-resized-crop (one random window per view), differentiable
        if crop_min < 1.0:
            B, C, H, W = x.shape
            scale = float(torch.empty(1).uniform_(crop_min, 1.0))
            ch, cw = max(1, int(round(H * scale))), max(1, int(round(W * scale)))
            top = int(torch.randint(0, H - ch + 1, (1,)))
            left = int(torch.randint(0, W - cw + 1, (1,)))
            x = x[:, :, top:top + ch, left:left + cw]
            x = F.interpolate(x, size=(H, W), mode="bicubic",
                              align_corners=False, antialias=True)
        # additive Gaussian noise (randomized smoothing)
        if noise_std > 0:
            x = (x + noise_std * torch.randn_like(x)).clamp(0, 1)
        return x

    # -- the conditional term ---------------------------------------------
    @torch._dynamo.disable  # run eager even when fd_train_step is torch.compile'd
    def log_p_c_given_x(self, images, labels, eot_views=1, noise_std=0.0,crop_min=1.0):

        idx = torch.arange(labels.shape[0], device=images.device)
        use_aug = (eot_views > 1) or (noise_std > 0) or (crop_min < 1.0)

        def _one(x):
            # CLIP resize with mean/std
            x = self._preprocess(x).to(self.dtype)
            # Run CLIP ViT image embeddings with L2 normalize
            feats = F.normalize(self.model.encode_image(x), dim=-1)
            # cosine similarity with precomputed class embeddings x temperature, outputs logits (sample, class) pair
            logits = self.logit_scale * (feats @ self.text_features.t())
            # softmax across classes
            return torch.log_softmax(logits.float(), dim=-1)[idx, labels]

        views = max(1, int(eot_views))
        acc = None

        for _ in range(views):
            x_view = self._augment_view(images, noise_std, crop_min) if use_aug else images
            lp = _one(x_view)
            acc = lp if acc is None else acc + lp

        return acc / views

    # -- feature access for the q_phi density-ratio head -------------------
    @property
    def feature_dim(self):
        """Embedding dim d of the shared CLIP image/text space (768 for ViT-L-14)."""
        return self.text_features.shape[-1]

    @torch._dynamo.disable  # run eager even inside a torch.compile'd fd_train_step
    def image_features(self, images):
        """L2-normalized CLIP image embedding (B, d) of clean [0,1] NCHW images,
        kept differentiable w.r.t. ``images`` so the gradient reaches the
        generator. Same feature space as log_p_c_given_x; consumed by the learned
        q_phi head. No EOT augmentation here — q_phi is its own classifier, and
        the EOT views are an anti-adversarial guard specific to the p_clip
        *maximization* term, not to this density-ratio side."""
        x = self._preprocess(images).to(self.dtype)
        return F.normalize(self.model.encode_image(x), dim=-1)

    @torch.no_grad()
    def full_log_probs(self, images):
        """(B, C) log-probs over all classes — for eval / smoke tests."""
        x = self._preprocess(images).to(self.dtype)
        feats = F.normalize(self.model.encode_image(x), dim=-1)
        logits = self.logit_scale * (feats @ self.text_features.t())
        return torch.log_softmax(logits.float(), dim=-1)

    @torch.no_grad()
    def smoke_test_uniform(self, batch_size=16, image_size=256):
        """On random-noise images the per-class log-prob should sit near the
        uniform value -log(num_classes) (= -6.91 for 1000 classes)."""
        x = torch.rand(batch_size, 3, image_size, image_size, device=self.device)
        log_probs = self.full_log_probs(x)
        uniform = -torch.log(torch.tensor(float(self.num_classes)))
        return {
            "mean_logp_per_class": float(log_probs.mean()),
            "uniform_ref": float(uniform),
            "max_logp": float(log_probs.max()),
        }


@torch.enable_grad()
def assert_clip_grad_flows(clip_classifier, model_wo_ddp, args,
                           sampling_args, tokenizer=None):
    """Sanity check: a backward through L_cond *only* must put a finite, nonzero
    gradient on the generator's trainable params. Catches a detached CLIP graph
    or a frozen-by-accident generator. Leaves grads zeroed afterwards.
    """
    model_wo_ddp.zero_grad(set_to_none=True)
    z = torch.randn(2, args.input_channels, args.input_size, args.input_size,
                    device="cuda") * args.noise_scale
    y = torch.randint(0, args.num_classes, (2,), device="cuda")
    sampled = model_wo_ddp.sample_images_with_grad(z, y, sampling_args=sampling_args)
    if tokenizer is not None:
        sampled = tokenizer.decode(tokenizer.denormalize_z(sampled))
    sampled = (sampled * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]

    l_cond = -clip_classifier.log_p_c_given_x(sampled, y).mean()
    l_cond.backward()

    n_with_grad = sum(
        1 for p in model_wo_ddp.parameters()
        if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
        and p.grad.abs().sum() > 0
    )
    n_trainable = sum(1 for p in model_wo_ddp.parameters() if p.requires_grad)
    model_wo_ddp.zero_grad(set_to_none=True)
    assert n_with_grad > 0, (
        "L_cond produced NO gradient on the generator — the CLIP graph is "
        "detached from the generator output (check preprocessing / dtype casts)."
    )
    logger.info(f"[CLIPClassifier] grad-flow OK: L_cond={float(l_cond):.4f}, "
                f"{n_with_grad}/{n_trainable} generator tensors received gradient.")