"""Frozen-VLM linear heads for the ``log q(c|z) - log p(c|z)`` conditional term.

This module owns everything that is *not* the training loop for the VLM-delta
experiment:

* :class:`VLMLinearHead` -- one linear classifier over a frozen VLM feature
  ``z = E_VLM(x)``, with a **frozen** affine feature normalisation baked in as
  buffers rather than a trainable/BatchNorm layer.  That choice is what makes
  ``W_q - W_p`` a meaningful drift measurement: p and q differ in exactly
  ``(weight, bias)`` and in nothing else.
* :class:`VLMDeltaHeads` -- the p / q_student / q_teacher triple, the EMA
  update, the drift meters, and checkpoint save/restore.
* :class:`ClassBalancedFeatureBuffer` -- the replay buffer q_student is trained
  from, with a per-class ring so no class can dominate q's training data.
* metric helpers shared by the offline validation script, the initial
  "is the VLM clueless?" diagnostic, and the per-step training logs.

The objective this serves is

    L = L_FD + w(s) * E[ log q_teacher(c|z) - log p(c|z) ],    z = E_VLM(G(eps, c))

with ``c`` the *sampled* conditioning label.  It is deliberately the sampled
target-class scalar, NOT the class-summed posterior KL: see ``docs/vlm_delta.md``
and ``docs/gmm_posterior_loss.md`` §3(a) for why that distinction is the whole
point of this trial, and what went wrong the last time a sampled-label scalar
was tried (with a *fitted generative* q, which this is not).

Nothing here trains the VLM.  ``p_head`` is fitted offline on real images by
``train_vlm_p_head.py`` and then frozen forever; ``q_student`` is initialised
from ``p_head`` and only ever sees detached generated features.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


# Bumped whenever the p-head checkpoint payload changes shape.
P_HEAD_FORMAT_VERSION = 1

# Which extractor produced z.  ``timm`` is a TimmReprModel that doubles as an FD
# judge (SigLIP CLS, MAE, ...); ``qwen_answer_state`` is a prompted VLM's hidden
# state at the position it would answer from (see ``qwen_answer_state.py``).  The
# tag is absent from every checkpoint written before the second backend existed
# and defaults to ``timm`` on load, so old checkpoints stay valid unchanged.
P_HEAD_BACKEND_TIMM = "timm"
P_HEAD_BACKEND_QWEN = "qwen_answer_state"
P_HEAD_BACKENDS = (P_HEAD_BACKEND_TIMM, P_HEAD_BACKEND_QWEN)

DEFAULT_VLM_MODEL = "vit_so400m_patch16_siglip_256.v2_webli"
DEFAULT_VLM_POOL = "cls"
DEFAULT_VLM_TARGET_SIZE = 224
DEFAULT_VLM_INPUT_SIZE = 256

FEATURE_NORM_MODES = ("standardize", "none")


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

class VLMLinearHead(torch.nn.Module):
    """``softmax((W z_hat + b) / T)`` over a frozen VLM feature.

    ``z_hat = (z - feature_mean) / feature_std`` with both statistics stored as
    **buffers fitted once on real training features and never updated**.  A
    BatchNorm here would have been the conventional linear-probe choice, but its
    running statistics would drift differently for p and q and would make the
    p/q comparison depend on something other than the two weight matrices.
    """

    def __init__(self, feature_dim: int, num_classes: int,
                 feature_norm: str = "standardize") -> None:
        super().__init__()
        if feature_norm not in FEATURE_NORM_MODES:
            raise ValueError(
                f"feature_norm must be one of {FEATURE_NORM_MODES}, got {feature_norm!r}"
            )
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.feature_norm = feature_norm
        self.linear = torch.nn.Linear(self.feature_dim, self.num_classes)
        torch.nn.init.trunc_normal_(self.linear.weight, std=0.01)
        torch.nn.init.zeros_(self.linear.bias)
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_std", torch.ones(self.feature_dim))

    # -- feature normalisation ------------------------------------------------

    def set_feature_norm_stats(self, mean: torch.Tensor, std: torch.Tensor,
                               eps: float = 1e-5) -> None:
        """Install the frozen normalisation fitted on real training features."""
        mean = mean.detach().to(self.feature_mean)
        std = std.detach().to(self.feature_std).clamp_min(eps)
        if mean.shape != self.feature_mean.shape or std.shape != self.feature_std.shape:
            raise ValueError(
                f"feature norm stats must be ({self.feature_dim},), got "
                f"{tuple(mean.shape)} / {tuple(std.shape)}"
            )
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float()
        if self.feature_norm == "none":
            return z
        return (z - self.feature_mean) / self.feature_std

    # -- forward --------------------------------------------------------------

    def logits(self, z: torch.Tensor, temperature: float = 1.0, *,
               detach_parameters: bool = False) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected features shaped (B, {self.feature_dim}), got {tuple(z.shape)}"
            )
        with torch.autocast(device_type=z.device.type, enabled=False):
            if detach_parameters:
                return F.linear(self.normalize(z), self.weight.detach(),
                                self.bias.detach()) / float(temperature)
            return self.linear(self.normalize(z)) / float(temperature)

    def log_probs(self, z: torch.Tensor, temperature: float = 1.0, *,
                  detach_parameters: bool = False) -> torch.Tensor:
        return torch.log_softmax(self.logits(z, temperature,
                                 detach_parameters=detach_parameters), dim=-1)

    def forward(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        return self.logits(z, temperature)

    # -- misc -----------------------------------------------------------------

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias

    def clone_frozen(self) -> "VLMLinearHead":
        other = copy.deepcopy(self)
        other.eval().requires_grad_(False)
        return other

    def parameter_checksum(self) -> str:
        payload = b"".join(
            t.detach().float().cpu().contiguous().numpy().tobytes()
            for t in (self.linear.weight, self.linear.bias,
                      self.feature_mean, self.feature_std)
        )
        return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# p-head checkpoint I/O
# ---------------------------------------------------------------------------

def class_ids_hash(class_ids: Sequence[int]) -> str:
    payload = json.dumps([int(c) for c in class_ids], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _torch_load(path: str | Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # older torch, or a payload with non-tensor metadata
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"p-head checkpoint must be a dict, got {type(obj).__name__}")
    return obj


P_HEAD_REQUIRED_KEYS = (
    "format_version",
    "vlm_model_name",
    "vlm_pool_type",
    "vlm_target_size",
    "vlm_input_size",
    "feature_dim",
    "num_classes",
    "class_ids",
    "feature_norm",
    "feature_mean",
    "feature_std",
    "head_state_dict",
    "temperature",
)


def load_p_head_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a p-head checkpoint."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"VLM p-head checkpoint not found: {path}")
    ckpt = _torch_load(path)
    missing = sorted(set(P_HEAD_REQUIRED_KEYS).difference(ckpt))
    if missing:
        raise ValueError(f"p-head checkpoint {path} is missing keys: {missing}")
    if int(ckpt["format_version"]) != P_HEAD_FORMAT_VERSION:
        raise ValueError(
            f"unsupported p-head checkpoint format {ckpt['format_version']}; "
            f"expected {P_HEAD_FORMAT_VERSION}"
        )
    if ckpt["feature_norm"] not in FEATURE_NORM_MODES:
        raise ValueError(f"invalid checkpoint feature_norm={ckpt['feature_norm']!r}")
    class_ids = [int(c) for c in ckpt["class_ids"]]
    if len(class_ids) != int(ckpt["num_classes"]):
        raise ValueError(
            f"checkpoint has {len(class_ids)} class_ids but num_classes="
            f"{ckpt['num_classes']}"
        )
    if sorted(class_ids) != class_ids or len(set(class_ids)) != len(class_ids):
        raise ValueError("checkpoint class_ids must be sorted and unique")
    stored_hash = ckpt.get("class_ids_sha256")
    if stored_hash is not None and stored_hash != class_ids_hash(class_ids):
        raise ValueError(f"p-head checkpoint {path}: class_ids hash mismatch")
    backend = p_head_backend(ckpt)
    if backend not in P_HEAD_BACKENDS:
        raise ValueError(
            f"p-head checkpoint {path} has unknown vlm_backend={backend!r}; "
            f"expected one of {P_HEAD_BACKENDS}"
        )
    if backend == P_HEAD_BACKEND_QWEN:
        # z is the answer state for one exact question read off one exact layer.
        # Without both recorded, nothing downstream could tell two heads apart.
        missing = [k for k in ("vlm_layer", "vlm_prompt_sha256") if k not in ckpt]
        if missing:
            raise ValueError(
                f"p-head checkpoint {path} declares backend {backend!r} but is "
                f"missing {missing}"
            )
    return ckpt


def build_head_from_checkpoint(ckpt: Mapping[str, Any], device="cuda") -> VLMLinearHead:
    head = VLMLinearHead(
        int(ckpt["feature_dim"]),
        int(ckpt["num_classes"]),
        feature_norm=str(ckpt["feature_norm"]),
    ).to(device=device, dtype=torch.float32)
    head.linear.load_state_dict(ckpt["head_state_dict"], strict=True)
    head.set_feature_norm_stats(
        torch.as_tensor(ckpt["feature_mean"], dtype=torch.float32),
        torch.as_tensor(ckpt["feature_std"], dtype=torch.float32),
    )
    return head


def p_head_backend(ckpt: Mapping[str, Any]) -> str:
    """Which feature extractor produced this head's ``z``.

    ``timm`` is the implicit value for every checkpoint written before the
    prompted-VLM backend existed, so those checkpoints keep exactly the identity
    they had and a run resuming against one is unaffected.
    """
    return str(ckpt.get("vlm_backend", P_HEAD_BACKEND_TIMM))


def p_head_identity(ckpt: Mapping[str, Any]) -> dict[str, Any]:
    """The identity a generator run must match to be allowed to use this head."""
    identity = {
        "vlm_model_name": str(ckpt["vlm_model_name"]),
        "vlm_pool_type": str(ckpt["vlm_pool_type"]),
        "vlm_target_size": int(ckpt["vlm_target_size"]),
        "vlm_input_size": int(ckpt["vlm_input_size"]),
        "feature_dim": int(ckpt["feature_dim"]),
        "num_classes": int(ckpt["num_classes"]),
        "feature_norm": str(ckpt["feature_norm"]),
        "class_ids_sha256": class_ids_hash([int(c) for c in ckpt["class_ids"]]),
        "p_head_sha256": p_head_checksum(ckpt),
    }
    backend = p_head_backend(ckpt)
    if backend != P_HEAD_BACKEND_TIMM:
        # Only non-default backends add keys, so a timm checkpoint's identity --
        # and therefore the resume check of any run already using one -- is
        # byte-for-byte what it was.
        identity["vlm_backend"] = backend
        identity["vlm_layer"] = int(ckpt["vlm_layer"])
        identity["vlm_prompt_sha256"] = str(ckpt["vlm_prompt_sha256"])
    return identity


def p_head_checksum(ckpt: Mapping[str, Any]) -> str:
    """Content hash of the frozen p head (weights + bias + normalisation)."""
    state = ckpt["head_state_dict"]
    parts = [
        torch.as_tensor(state["weight"], dtype=torch.float32),
        torch.as_tensor(state["bias"], dtype=torch.float32),
        torch.as_tensor(ckpt["feature_mean"], dtype=torch.float32),
        torch.as_tensor(ckpt["feature_std"], dtype=torch.float32),
    ]
    payload = b"".join(t.cpu().contiguous().numpy().tobytes() for t in parts)
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# p / q_student / q_teacher
# ---------------------------------------------------------------------------

class VLMDeltaHeads(torch.nn.Module):
    """Frozen real-data ``p``, trainable ``q_student``, slow EMA ``q_teacher``.

    ``q_student`` and ``q_teacher`` are both initialised from ``p``, so at step
    0 ``log q(c|z) - log p(c|z) == 0`` exactly and the conditional term supplies
    exactly zero gradient.  It only becomes a force as q tracks the generator.
    """

    def __init__(self, p_head: VLMLinearHead, *, temperature: float = 1.0,
                 ema_beta: float = 0.999, use_ema: bool = True) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"temperature must be finite and > 0, got {temperature}")
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError(f"ema_beta must be in [0, 1), got {ema_beta}")
        self.temperature = float(temperature)
        self.ema_beta = float(ema_beta)
        # Keep the library default compatible with older analysis utilities;
        # the training entry point explicitly defaults use_ema to False.
        self.use_ema = bool(use_ema)

        self.p_head = p_head
        self.p_head.eval().requires_grad_(False)

        self.q_student = p_head.clone_frozen()
        self.q_student.requires_grad_(True)
        self.q_teacher = p_head.clone_frozen()

        # Immutable reference copies of p's parameters for the drift meters.
        self.register_buffer("p_weight_ref", p_head.weight.detach().clone())
        self.register_buffer("p_bias_ref", p_head.bias.detach().clone())
        self.register_buffer("q_train_steps", torch.zeros((), dtype=torch.long))
        # Norm of the last student update and the last EMA teacher update.
        self._last_student_update_norm = 0.0
        self._last_teacher_update_norm = 0.0

    # -- properties -----------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return self.p_head.num_classes

    @property
    def feature_dim(self) -> int:
        return self.p_head.feature_dim

    def q_parameters(self):
        return self.q_student.parameters()

    # -- forward paths --------------------------------------------------------

    def p_log_probs(self, z: torch.Tensor) -> torch.Tensor:
        return self.p_head.log_probs(z, self.temperature)

    def q_student_log_probs(self, z: torch.Tensor) -> torch.Tensor:
        return self.q_student.log_probs(z, self.temperature)

    def q_teacher_log_probs(self, z: torch.Tensor) -> torch.Tensor:
        return self.q_teacher.log_probs(z, self.temperature)

    @property
    def generator_head(self) -> VLMLinearHead:
        return self.q_teacher if self.use_ema else self.q_student

    def q_generator_log_probs(self, z: torch.Tensor) -> torch.Tensor:
        """Read the selected q with fixed weights, retaining the image/feature VJP."""
        return self.generator_head.log_probs(z, self.temperature, detach_parameters=True)

    def delta_log_qp(self, z: torch.Tensor, labels: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample ``log q_generator(c|z) - log p(c|z)`` and its two halves.

        Differentiable in ``z``; both heads' parameters are frozen here.
        """
        logp_all = self.p_log_probs(z)
        logq_all = self.q_generator_log_probs(z)
        idx = labels.long().view(-1, 1)
        logp_c = logp_all.gather(1, idx).squeeze(1)
        logq_c = logq_all.gather(1, idx).squeeze(1)
        return logq_c - logp_c, logq_c, logp_c

    # -- EMA ------------------------------------------------------------------

    @torch.no_grad()
    def ema_update(self) -> None:
        """``psi_teacher <- beta * psi_teacher + (1 - beta) * psi_student``."""
        if not self.use_ema:
            return
        beta = self.ema_beta
        total_sq = 0.0
        for teacher_p, student_p in zip(self.q_teacher.linear.parameters(),
                                        self.q_student.linear.parameters()):
            delta = (1.0 - beta) * (student_p.detach() - teacher_p)
            teacher_p.add_(delta)
            total_sq += float(delta.pow(2).sum())
        self._last_teacher_update_norm = math.sqrt(total_sq)

    @torch.no_grad()
    def record_student_update(self, before: Sequence[torch.Tensor]) -> None:
        total_sq = 0.0
        for prev, cur in zip(before, self.q_student.linear.parameters()):
            total_sq += float((cur.detach() - prev).pow(2).sum())
        self._last_student_update_norm = math.sqrt(total_sq)

    def student_snapshot(self) -> list[torch.Tensor]:
        return [p.detach().clone() for p in self.q_student.linear.parameters()]

    # -- diagnostics ----------------------------------------------------------

    @torch.no_grad()
    def drift_metrics(self) -> dict[str, float]:
        """How far q has moved from p, and how fast."""
        out: dict[str, float] = {}
        p_w, p_b = self.p_weight_ref, self.p_bias_ref
        p_w_norm = max(float(p_w.norm()), 1e-12)
        p_b_norm = max(float(p_b.norm()), 1e-12)
        for name, head in (("q_student", self.q_student), ("q_teacher", self.q_teacher)):
            dw = head.weight.detach() - p_w
            db = head.bias.detach() - p_b
            out[f"{name}_weight_delta_l2"] = float(dw.norm())
            out[f"{name}_bias_delta_l2"] = float(db.norm())
            out[f"{name}_weight_delta_rel"] = float(dw.norm()) / p_w_norm
            out[f"{name}_bias_delta_rel"] = float(db.norm()) / p_b_norm
            out[f"{name}_weight_cos_to_p"] = float(
                F.cosine_similarity(head.weight.detach().reshape(1, -1),
                                    p_w.reshape(1, -1), dim=1)
            )
            out[f"{name}_weight_norm"] = float(head.weight.detach().norm())
            out[f"{name}_bias_norm"] = float(head.bias.detach().norm())
        # Per-class drift of the teacher, normalised by the per-class norm of p.
        per_class = (self.q_teacher.weight.detach() - p_w).norm(dim=1)
        p_per_class = p_w.norm(dim=1).clamp_min(1e-12)
        rel = per_class / p_per_class
        out["q_teacher_class_weight_delta_mean"] = float(rel.mean())
        out["q_teacher_class_weight_delta_max"] = float(rel.max())
        out["q_teacher_class_weight_delta_min"] = float(rel.min())
        out["q_teacher_class_weight_delta_std"] = float(rel.std(unbiased=False))
        out["q_student_update_norm"] = self._last_student_update_norm
        out["q_teacher_ema_update_norm"] = self._last_teacher_update_norm
        out["q_train_steps"] = float(self.q_train_steps.item())
        return out if self.use_ema else {k: v for k, v in out.items()
                                        if not k.startswith("q_teacher_")}

    @torch.no_grad()
    def teacher_student_agreement(self, z: torch.Tensor) -> dict[str, float]:
        """How far ahead of the teacher the student has raced on this batch."""
        if not self.use_ema:
            return {}
        s_logits = self.q_student.logits(z, self.temperature)
        t_logits = self.q_teacher.logits(z, self.temperature)
        s_logp = torch.log_softmax(s_logits, dim=-1)
        t_logp = torch.log_softmax(t_logits, dim=-1)
        return {
            "q_teacher_student_logit_mse": float((s_logits - t_logits).pow(2).mean()),
            # KL(student || teacher): how much information the EMA is holding back
            "q_teacher_student_kl": float(
                (s_logp.exp() * (s_logp - t_logp)).sum(-1).mean()
            ),
        }

    @torch.no_grad()
    def init_equality_check(self, z: torch.Tensor) -> dict[str, float]:
        """Max |log q - log p| over a batch; ~0 immediately after construction."""
        logp = self.p_log_probs(z)
        out = {}
        for name, logq in (("q_student", self.q_student_log_probs(z)),
                           ("q_teacher", self.q_teacher_log_probs(z))):
            out[f"init_max_abs_logdiff_{name}_vs_p"] = float((logq - logp).abs().max())
        out["init_max_abs_weight_diff_q_student_vs_p"] = float(
            (self.q_student.weight - self.p_weight_ref).abs().max()
        )
        out["init_max_abs_weight_diff_q_teacher_vs_p"] = float(
            (self.q_teacher.weight - self.p_weight_ref).abs().max()
        )
        return out

    # -- checkpointing --------------------------------------------------------

    def q_state_dict(self) -> dict[str, Any]:
        return {
            "q_student": {k: v.detach().cpu()
                          for k, v in self.q_student.state_dict().items()},
            "q_teacher": {k: v.detach().cpu()
                          for k, v in self.q_teacher.state_dict().items()},
            "q_train_steps": int(self.q_train_steps.item()),
            "ema_beta": self.ema_beta,
            "use_ema": self.use_ema,
            "temperature": self.temperature,
        }

    def load_q_state_dict(self, state: Mapping[str, Any], *, strict_config=True) -> None:
        if strict_config and bool(state.get("use_ema", True)) != self.use_ema:
            raise ValueError("checkpoint q EMA mode differs from this run; use --load_from "
                             "for a new experiment or --vlm_q_use_ema for a legacy EMA run")
        self.q_student.load_state_dict(state["q_student"], strict=True)
        self.q_teacher.load_state_dict(state["q_teacher"], strict=True)
        self.q_train_steps.fill_(int(state.get("q_train_steps", 0)))
        saved_beta = float(state.get("ema_beta", self.ema_beta))
        saved_temp = float(state.get("temperature", self.temperature))
        if strict_config:
            if abs(saved_temp - self.temperature) > 1e-9:
                raise ValueError(
                    f"checkpoint q temperature {saved_temp} != configured "
                    f"{self.temperature}; the p/q comparison is only valid at a "
                    f"single shared temperature"
                )
            if self.use_ema and abs(saved_beta - self.ema_beta) > 1e-12:
                raise ValueError(
                    f"checkpoint q ema_beta {saved_beta} != configured {self.ema_beta}"
                )
        self.q_student.requires_grad_(True)
        self.q_teacher.requires_grad_(False)
        self.p_head.requires_grad_(False)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ClassBalancedFeatureBuffer:
    """Per-class ring buffer of detached generated VLM features.

    A single shared FIFO would let a run of one class dominate q's training
    data and make ``q_train_accuracy`` meaningless.  A per-class ring makes the
    class distribution balanced by construction once every class has been seen
    ``per_class_capacity`` times, and makes the coverage meters exact.

    Every rank pushes the *same* all-gathered batch and samples with a shared
    seeded generator, so the buffer -- and therefore q -- stays bit-identical
    across ranks without any collective.
    """

    def __init__(self, num_classes: int, feature_dim: int, per_class_capacity: int,
                 device="cuda", dtype=torch.float16, seed: int = 1234) -> None:
        if per_class_capacity <= 0:
            raise ValueError("per_class_capacity must be positive")
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.per_class_capacity = int(per_class_capacity)
        self.device = device
        self.dtype = dtype
        self.feats = torch.zeros(self.num_classes, self.per_class_capacity,
                                 self.feature_dim, dtype=dtype, device=device)
        self.fill = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        self.ptr = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        self.total_pushed = 0
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    @torch.no_grad()
    def push(self, z: torch.Tensor, labels: torch.Tensor) -> None:
        z = z.detach().to(self.dtype)
        labels = labels.long()
        # Vectorised scatter: slot = (ptr[c] + rank_within_class) % capacity.
        order = torch.argsort(labels, stable=True)
        sorted_labels = labels[order]
        counts = torch.bincount(sorted_labels, minlength=self.num_classes)
        # position of each element within its own class run
        starts = torch.cumsum(counts, 0) - counts
        within = torch.arange(sorted_labels.numel(), device=labels.device) - starts[sorted_labels]
        slots = (self.ptr[sorted_labels] + within) % self.per_class_capacity
        self.feats[sorted_labels, slots] = z[order]
        self.ptr = (self.ptr + counts) % self.per_class_capacity
        self.fill = torch.clamp(self.fill + counts, max=self.per_class_capacity)
        self.total_pushed += int(labels.numel())

    @torch.no_grad()
    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Draw a class-balanced batch: uniform over classes, then over slots."""
        present = torch.nonzero(self.fill > 0, as_tuple=False).flatten()
        if present.numel() == 0:
            return None
        n = int(batch_size)
        pick = torch.randint(0, present.numel(), (n,), generator=self.generator)
        classes = present[pick.to(present.device)]
        fills = self.fill[classes]
        frac = torch.rand(n, generator=self.generator).to(fills.device)
        slots = torch.minimum((frac * fills.float()).long(), fills - 1)
        return self.feats[classes, slots].float(), classes

    @property
    def size(self) -> int:
        return int(self.fill.sum().item())

    def stats(self) -> dict[str, float]:
        fill = self.fill
        return {
            "q_buffer_size": float(fill.sum().item()),
            "q_buffer_class_coverage": float((fill > 0).float().mean().item()),
            "q_buffer_min_samples_per_class": float(fill.min().item()),
            "q_buffer_max_samples_per_class": float(fill.max().item()),
            "q_buffer_total_pushed": float(self.total_pushed),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "feats": self.feats.detach().cpu(),
            "fill": self.fill.detach().cpu(),
            "ptr": self.ptr.detach().cpu(),
            "total_pushed": self.total_pushed,
            "per_class_capacity": self.per_class_capacity,
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for key in ("num_classes", "feature_dim", "per_class_capacity"):
            if int(state[key]) != int(getattr(self, key)):
                raise ValueError(
                    f"q buffer checkpoint {key}={state[key]} != configured "
                    f"{getattr(self, key)}"
                )
        self.feats.copy_(state["feats"].to(self.feats.device, self.dtype))
        self.fill.copy_(state["fill"].to(self.fill.device))
        self.ptr.copy_(state["ptr"].to(self.ptr.device))
        self.total_pushed = int(state["total_pushed"])
        gen_state = state.get("generator_state")
        if gen_state is not None:
            self.generator.set_state(gen_state.cpu() if torch.is_tensor(gen_state) else gen_state)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def head_metrics(log_probs: torch.Tensor, labels: torch.Tensor, prefix: str,
                 ) -> dict[str, float]:
    """Everything one classifier head has to say about a labelled batch.

    ``target_rank`` (1 = best) is the meter that matters before conditioning
    takes off: top-1 is quantised at ``1/batch`` and reads exactly 0 for tens of
    thousands of steps, while the mean/median rank moves continuously.  Chance
    is ``(C + 1) / 2``.
    """
    labels = labels.long()
    idx = labels.view(-1, 1)
    num_classes = log_probs.shape[-1]
    target_logp = log_probs.gather(1, idx).squeeze(1)
    probs = log_probs.exp()
    target_prob = probs.gather(1, idx).squeeze(1)
    top1 = log_probs.argmax(-1)
    k5 = min(5, num_classes)
    top5_hit = log_probs.topk(k5, dim=-1).indices.eq(idx).any(-1)
    rank = (log_probs > target_logp.view(-1, 1)).sum(-1) + 1
    # best log-prob among the non-target classes
    masked = log_probs.scatter(1, idx, float("-inf"))
    best_other = masked.max(dim=-1).values
    entropy = -(probs * log_probs).sum(-1)
    return {
        f"{prefix}_ce": float(-target_logp.mean()),
        f"{prefix}_target_logp": float(target_logp.mean()),
        f"{prefix}_target_logp_median": float(target_logp.median()),
        f"{prefix}_target_prob": float(target_prob.mean()),
        f"{prefix}_top1": float((top1 == labels).float().mean()),
        f"{prefix}_top5": float(top5_hit.float().mean()),
        f"{prefix}_entropy": float(entropy.mean()),
        f"{prefix}_top1_conf": float(probs.max(dim=-1).values.mean()),
        f"{prefix}_target_rank": float(rank.float().mean()),
        f"{prefix}_target_rank_median": float(rank.float().median()),
        f"{prefix}_margin": float((target_logp - best_other).mean()),
    }


@torch.no_grad()
def pq_agreement_metrics(logp_all: torch.Tensor, logq_all: torch.Tensor,
                         labels: torch.Tensor, prefix: str = "vlm_pq",
                         ) -> dict[str, float]:
    """Distribution-level p-vs-q diagnostics.  **Never** the optimised objective.

    These exist to separate "q has learned something about the generator" from
    "p and q are still making identical decisions".
    """
    labels = labels.long().view(-1, 1)
    p = logp_all.exp()
    q = logq_all.exp()
    kl_qp = (q * (logq_all - logp_all)).sum(-1)
    kl_pq = (p * (logp_all - logq_all)).sum(-1)
    m = 0.5 * (p + q)
    log_m = m.clamp_min(1e-30).log()
    js = 0.5 * (q * (logq_all - log_m)).sum(-1) + 0.5 * (p * (logp_all - log_m)).sum(-1)
    p_top1 = logp_all.argmax(-1)
    q_top1 = logq_all.argmax(-1)
    p_c = p.gather(1, labels).squeeze(1)
    q_c = q.gather(1, labels).squeeze(1)
    return {
        f"{prefix}_full_kl_qp": float(kl_qp.mean()),
        f"{prefix}_full_kl_pq": float(kl_pq.mean()),
        f"{prefix}_js": float(js.mean()),
        f"{prefix}_top1_agreement": float((p_top1 == q_top1).float().mean()),
        f"{prefix}_argmax_disagreement": float((p_top1 != q_top1).float().mean()),
        "vlm_target_prob_gap": float((p_c - q_c).mean()),
    }


@torch.no_grad()
def delta_distribution_metrics(delta: torch.Tensor, prefix: str = "vlm_delta_logqp",
                               ) -> dict[str, float]:
    """Tail meters for ``log q(c|z) - log p(c|z)``.

    The fitted-GMM sampled-label scalar became strongly negative and unstable;
    these are the meters that would show the same pathology here early.
    """
    d = delta.detach().float().flatten()
    qs = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=d.device)
    quant = torch.quantile(d, qs)
    absd = d.abs()
    aq = torch.quantile(absd, torch.tensor([0.95, 0.99], device=d.device))
    return {
        f"{prefix}_mean": float(d.mean()),
        f"{prefix}_std": float(d.std(unbiased=False)),
        f"{prefix}_min": float(d.min()),
        f"{prefix}_max": float(d.max()),
        f"{prefix}_p10": float(quant[0]),
        f"{prefix}_p25": float(quant[1]),
        f"{prefix}_p50": float(quant[2]),
        f"{prefix}_p75": float(quant[3]),
        f"{prefix}_p90": float(quant[4]),
        "vlm_delta_abs_p95": float(aq[0]),
        "vlm_delta_abs_p99": float(aq[1]),
    }


class PerClassAccumulator:
    """Accumulate per-class conditioning statistics over a window of steps."""

    def __init__(self, num_classes: int, class_ids: Sequence[int], device="cuda") -> None:
        self.num_classes = int(num_classes)
        self.class_ids = [int(c) for c in class_ids]
        self.device = device
        self._fields = (
            "count", "p_top1", "q_top1", "probe_top1", "p_target_prob",
            "q_target_prob", "p_target_rank", "q_target_rank", "delta_logqp",
        )
        self.reset()

    def reset(self) -> None:
        self.acc = {f: torch.zeros(self.num_classes, dtype=torch.float64,
                                   device=self.device) for f in self._fields}
        self.steps = 0
        self.has_probe = False

    @torch.no_grad()
    def update(self, labels: torch.Tensor, *, logp_all: torch.Tensor,
               logq_all: torch.Tensor, probe_correct: torch.Tensor | None = None,
               ) -> None:
        labels = labels.long()
        idx = labels.view(-1, 1)
        ones = torch.ones_like(labels, dtype=torch.float64)
        p_logp_c = logp_all.gather(1, idx).squeeze(1)
        q_logp_c = logq_all.gather(1, idx).squeeze(1)
        p_rank = (logp_all > p_logp_c.view(-1, 1)).sum(-1).double() + 1
        q_rank = (logq_all > q_logp_c.view(-1, 1)).sum(-1).double() + 1
        contributions = {
            "count": ones,
            "p_top1": (logp_all.argmax(-1) == labels).double(),
            "q_top1": (logq_all.argmax(-1) == labels).double(),
            "p_target_prob": p_logp_c.exp().double(),
            "q_target_prob": q_logp_c.exp().double(),
            "p_target_rank": p_rank,
            "q_target_rank": q_rank,
            "delta_logqp": (q_logp_c - p_logp_c).double(),
        }
        if probe_correct is not None:
            contributions["probe_top1"] = probe_correct.double()
            self.has_probe = True
        for key, value in contributions.items():
            self.acc[key].index_add_(0, labels, value)
        self.steps += 1

    def to_records(self) -> list[dict[str, float]]:
        count = self.acc["count"].clamp_min(1.0)
        out = []
        for i in range(self.num_classes):
            n = float(self.acc["count"][i])
            row = {"local_index": i, "class_id": self.class_ids[i], "count": n}
            for field in self._fields:
                if field == "count":
                    continue
                # A never-supplied probe must read NaN, not 0.0 -- 0.0 would be
                # indistinguishable from "the probe got every sample wrong".
                unmeasured = (field == "probe_top1" and not self.has_probe)
                row[field] = (float("nan") if (n == 0 or unmeasured)
                              else float(self.acc[field][i] / count[i]))
            out.append(row)
        return out

    def dump(self, path: str | Path, *, step: int, extra: Mapping[str, Any] | None = None,
             ) -> None:
        payload = {
            "step": int(step),
            "window_steps": self.steps,
            "num_classes": self.num_classes,
            "per_class": self.to_records(),
        }
        if extra:
            payload.update(dict(extra))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        tmp.replace(path)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

class LocalClassMap:
    """Global ImageNet label <-> local head index for a class subset."""

    def __init__(self, class_ids: Sequence[int], num_global: int = 1000,
                 device="cuda") -> None:
        self.class_ids = [int(c) for c in class_ids]
        self.num_global = int(num_global)
        table = torch.full((self.num_global,), -1, dtype=torch.long, device=device)
        for local, global_id in enumerate(self.class_ids):
            if not 0 <= global_id < self.num_global:
                raise ValueError(f"class id {global_id} outside [0, {self.num_global})")
            table[global_id] = local
        self.table = table
        self.ids_tensor = torch.tensor(self.class_ids, dtype=torch.long, device=device)

    def to_local(self, labels: torch.Tensor) -> torch.Tensor:
        return self.table.index_select(0, labels.long())

    def to_global(self, local: torch.Tensor) -> torch.Tensor:
        return self.ids_tensor.index_select(0, local.long())

    def validate(self, drawable: Sequence[int]) -> None:
        missing = [int(c) for c in drawable if int(self.table[int(c)]) < 0]
        if missing:
            raise ValueError(
                f"the p head has no output unit for generator label(s) {missing}; it "
                f"was trained on {len(self.class_ids)} class(es) "
                f"{self.class_ids[:8]}{'...' if len(self.class_ids) > 8 else ''}. "
                f"Retrain with train_vlm_p_head.py --class_ids matching the run."
            )
        if len(set(int(c) for c in drawable)) != len(self.class_ids):
            raise ValueError(
                f"the generator draws {len(set(int(c) for c in drawable))} label(s) but "
                f"the p head is a {len(self.class_ids)}-way classifier. The softmax "
                f"denominator must be exactly the set of drawable classes."
            )
