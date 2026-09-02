"""Frozen MAE linear classifier used as a differentiable ``log p(y | x)``.

The MAE backbone is already pretrained.  The checkpoint loaded here contains
only a supervised ImageNet head (and metadata); both the backbone and head are
frozen while JiT is optimized.  Frozen parameters still permit gradients with
respect to the input image.

The conditional FD entrypoint can attach this module to the MAE representation
model that is already loaded as an FD judge.  In that case
``log_p_c_given_features`` reuses the judge's local features and avoids a
second ViT-L forward pass.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


CHECKPOINT_FORMAT_VERSION = 1
DEFAULT_MAE_MODEL = "vit_large_patch16_224.mae"


def build_linear_probe_head(
    feature_dim: int,
    num_classes: int,
    head_norm: str = "bn",
) -> torch.nn.Module:
    """Build the head used by both offline training and frozen inference."""
    linear = torch.nn.Linear(feature_dim, num_classes)
    torch.nn.init.trunc_normal_(linear.weight, std=0.01)
    torch.nn.init.zeros_(linear.bias)
    if head_norm == "none":
        return linear
    if head_norm == "bn":
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(feature_dim, affine=False, eps=1e-6),
            linear,
        )
    raise ValueError(f"head_norm must be 'bn' or 'none', got {head_norm!r}")


def select_pooled_features(
    outputs: tuple[torch.Tensor, torch.Tensor | None],
    pool_type: str,
) -> torch.Tensor:
    """Select the same CLS/average feature convention used by FD judges."""
    primary, secondary = outputs
    if pool_type == "cls":
        return primary
    if pool_type == "avg":
        if secondary is None:
            raise ValueError("pool_type='avg' requested but the backbone has no patch-token average")
        return secondary
    raise ValueError(f"pool_type must be 'cls' or 'avg', got {pool_type!r}")


def _torch_load(path: str | Path) -> dict[str, Any]:
    """Load a metadata checkpoint on old and new PyTorch versions."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"MAE probe checkpoint must be a dict, got {type(obj).__name__}")
    return obj


def load_probe_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate the non-tensor metadata needed to reconstruct a head."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MAE linear-probe checkpoint not found: {path}")
    ckpt = _torch_load(path)

    required = {
        "format_version",
        "model_name",
        "pool_type",
        "target_size",
        "feature_dim",
        "num_classes",
        "head_norm",
        "head_state_dict",
        "class_to_idx",
    }
    missing = sorted(required.difference(ckpt))
    if missing:
        raise ValueError(f"MAE probe checkpoint {path} is missing keys: {missing}")
    if int(ckpt["format_version"]) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported MAE probe checkpoint format {ckpt['format_version']}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )
    if ckpt["pool_type"] not in ("cls", "avg"):
        raise ValueError(f"invalid checkpoint pool_type={ckpt['pool_type']!r}")
    if ckpt["head_norm"] not in ("bn", "none"):
        raise ValueError(f"invalid checkpoint head_norm={ckpt['head_norm']!r}")

    class_to_idx = ckpt["class_to_idx"]
    if not isinstance(class_to_idx, Mapping):
        raise ValueError("checkpoint class_to_idx must be a mapping")
    expected_indices = list(range(int(ckpt["num_classes"])))
    if sorted(int(v) for v in class_to_idx.values()) != expected_indices:
        raise ValueError("checkpoint class_to_idx is not a contiguous 0..C-1 mapping")
    return ckpt


def _augment_view(
    images: torch.Tensor,
    noise_std: float,
    crop_min: float,
) -> torch.Tensor:
    """Differentiable EOT augmentation matching the existing classifiers."""
    x = images
    flip = torch.rand(x.shape[0], device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], torch.flip(x, dims=(3,)), x)
    if crop_min < 1.0:
        _, _, height, width = x.shape
        scale = float(torch.empty((), device=x.device).uniform_(crop_min, 1.0))
        crop_h = max(1, int(round(height * scale)))
        crop_w = max(1, int(round(width * scale)))
        top = int(torch.randint(0, height - crop_h + 1, (), device=x.device))
        left = int(torch.randint(0, width - crop_w + 1, (), device=x.device))
        x = x[:, :, top : top + crop_h, left : left + crop_w]
        x = F.interpolate(
            x,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    if noise_std > 0:
        x = (x + noise_std * torch.randn_like(x)).clamp(0, 1)
    return x


class MAELinearProbe(torch.nn.Module):
    """Frozen ImageNet linear probe attached to a frozen MAE feature model.

    ``backbone`` must follow ``TimmReprModel``'s contract and return
    ``(cls_token, mean_patch_token)``.  It may be the exact model object already
    used by an FD judge.
    """

    handles_cap = False
    is_vlm_judge = False
    display_name = "MAELinearProbe"

    def __init__(
        self,
        backbone: torch.nn.Module,
        checkpoint: Mapping[str, Any],
        *,
        device: str | torch.device = "cuda",
        temperature: float | None = None,
        feature_judge_index: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.model_name = str(checkpoint["model_name"])
        self.pool_type = str(checkpoint["pool_type"])
        self.target_size = int(checkpoint["target_size"])
        self.feature_dim = int(checkpoint["feature_dim"])
        self.num_classes = int(checkpoint["num_classes"])
        self.head_norm = str(checkpoint["head_norm"])
        self.feature_judge_index = feature_judge_index
        saved_temperature = float(checkpoint.get("temperature", 1.0))
        self.temperature = saved_temperature if temperature is None else float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError(f"temperature must be finite and > 0, got {self.temperature}")

        backbone_dim = getattr(backbone, "feat_dim", self.feature_dim)
        if int(backbone_dim) != self.feature_dim:
            raise ValueError(
                f"MAE backbone feature_dim={backbone_dim} does not match "
                f"probe checkpoint feature_dim={self.feature_dim}"
            )
        backbone_size = getattr(backbone, "target_size", self.target_size)
        if int(backbone_size) != self.target_size:
            raise ValueError(
                f"MAE backbone target_size={backbone_size} does not match "
                f"probe checkpoint target_size={self.target_size}"
            )

        self.head = build_linear_probe_head(
            self.feature_dim,
            self.num_classes,
            self.head_norm,
        ).to(device=device, dtype=torch.float32)
        self.head.load_state_dict(checkpoint["head_state_dict"], strict=True)

        # These metrics are descriptive metadata, not runtime acceptance tests.
        self.best_val_top1 = float(checkpoint.get("best_val_top1", float("nan")))
        self.best_val_top5 = float(checkpoint.get("best_val_top5", float("nan")))
        self.class_to_idx = dict(checkpoint["class_to_idx"])
        self.last_stats: dict[str, float] = {}

        self.eval().requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        backbone: torch.nn.Module,
        *,
        device: str | torch.device = "cuda",
        temperature: float | None = None,
        feature_judge_index: int | None = None,
        expected_model_name: str | None = None,
        expected_pool_type: str | None = None,
        expected_target_size: int | None = None,
        expected_num_classes: int | None = None,
    ) -> "MAELinearProbe":
        ckpt = load_probe_checkpoint(checkpoint_path)
        expectations = {
            "model_name": expected_model_name,
            "pool_type": expected_pool_type,
            "target_size": expected_target_size,
            "num_classes": expected_num_classes,
        }
        for key, expected in expectations.items():
            if expected is not None and ckpt[key] != expected:
                raise ValueError(
                    f"MAE probe checkpoint {key}={ckpt[key]!r} does not match "
                    f"required value {expected!r}"
                )
        return cls(
            backbone,
            ckpt,
            device=device,
            temperature=temperature,
            feature_judge_index=feature_judge_index,
        )

    def train(self, mode: bool = True) -> "MAELinearProbe":
        """Keep the frozen backbone and BN head in inference mode permanently."""
        super().train(False)
        return self

    def features_from_images(self, images: torch.Tensor) -> torch.Tensor:
        return select_pooled_features(self.backbone(images), self.pool_type)

    def logits_from_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected features shaped (B, {self.feature_dim}), got "
                f"{tuple(features.shape)}"
            )
        return self.head(features.float()) / self.temperature

    def log_probs_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.logits_from_features(features), dim=-1)

    def log_p_c_given_features(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = self.log_probs_from_features(features)
        if labels.ndim != 1 or labels.shape[0] != log_probs.shape[0]:
            raise ValueError(
                f"labels must have shape ({log_probs.shape[0]},), got {tuple(labels.shape)}"
            )
        selected = log_probs.gather(1, labels.long()[:, None]).squeeze(1)
        self.last_stats = {
            "logp_mae_probe": float(selected.mean().detach()),
            "mae_probe_top1": float(
                (log_probs.argmax(dim=-1) == labels).float().mean().detach()
            ),
        }
        return selected

    @torch._dynamo.disable
    def log_p_c_given_x(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        eot_views: int = 1,
        noise_std: float = 0.0,
        crop_min: float = 1.0,
    ) -> torch.Tensor:
        """Image-space fallback used for EOT and startup gradient checks."""
        views = max(1, int(eot_views))
        use_aug = views > 1 or noise_std > 0 or crop_min < 1.0
        total = None
        for _ in range(views):
            view = _augment_view(images, noise_std, crop_min) if use_aug else images
            logp = self.log_p_c_given_features(self.features_from_images(view), labels)
            total = logp if total is None else total + logp
        return total / views

    @torch.no_grad()
    def full_log_probs(self, images: torch.Tensor) -> torch.Tensor:
        return self.log_probs_from_features(self.features_from_images(images))

    @torch.no_grad()
    def smoke_test_uniform(self, batch_size: int = 4, image_size: int = 256) -> dict[str, float]:
        """Check finite, normalized output; noise need not produce uniform logits."""
        device = next(self.head.parameters()).device
        images = torch.rand(batch_size, 3, image_size, image_size, device=device)
        log_probs = self.full_log_probs(images)
        if log_probs.shape != (batch_size, self.num_classes):
            raise AssertionError(
                f"probe returned {tuple(log_probs.shape)}, expected "
                f"({batch_size}, {self.num_classes})"
            )
        if not torch.isfinite(log_probs).all():
            raise AssertionError("MAE probe produced NaN/Inf log-probabilities")
        probs = log_probs.exp()
        if not torch.allclose(
            probs.sum(dim=-1),
            torch.ones(batch_size, device=device),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise AssertionError("MAE probe probabilities do not sum to one")
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return {
            "mean_logp_per_class": float(log_probs.mean()),
            "uniform_ref": -math.log(self.num_classes),
            "max_logp": float(log_probs.max()),
            "entropy": float(entropy),
            "checkpoint_val_top1": self.best_val_top1,
            "checkpoint_val_top5": self.best_val_top5,
        }
