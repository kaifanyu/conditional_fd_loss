"""FD post-training with the sampled-label VLM posterior-delta conditional term.

    L = sum_j FD_j / (FD_j.detach() + eps)                        # marginal
      + w(s) * E[ log q(c | z) - log p(c | z) ],  z = E_VLM(G(eps, c))

By default, one frozen VLM and two linear heads on top of it. With
``--vlm_q_lora``, Qwen q additionally learns LoRA adapters; p always uses the
unadapted backbone. See ``docs/vlm_lora_q.md`` for the image-based update path.
Which VLM is decided by the
p-head checkpoint, not by a flag here -- the head *is* the definition of ``z``:

    ``vlm_backend=timm``               SigLIP-SO400M's CLS token.  It is already
                                       an FD judge, so ``z`` is computed exactly
                                       once per step and the term is nearly free.
    ``vlm_backend=qwen_answer_state``  Qwen2.5-VL-7B's hidden state at the
                                       position it would answer "what class is
                                       this?" from (``qwen_answer_state.py``).
                                       Label-aligned by construction, and paid
                                       for with a 7B forward+backward per scored
                                       image, injected as a first-order VJP so
                                       the 7B graph never coexists with the FD
                                       judges'.

The two linear heads are:

    p_phi   trained offline on REAL (image, label) pairs, then frozen forever
            (``train_vlm_p_head.py``)
    q_psi   initialised from p, then trained online by cross entropy on
            DETACHED generated (image, sampled label) pairs

The generator loss uses the current q student with detached parameters. An EMA
teacher is available only with ``--vlm_q_use_ema`` for comparisons. At
initialisation q == p exactly, so the term is exactly zero and supplies exactly
zero gradient; it only becomes a force as q learns what the *current* generator
actually looks like.

Scientific question
-------------------
Does ``grad_z [ log q(c|z) - log p(c|z) ]`` teach conditioning when p is a real-
data VLM linear probe and q is a slowly moving probe on the generator?

This is deliberately the **sampled-label target-class scalar**, not the class-
summed posterior KL and not a second ``-log p(c|z)`` driver.  The class-summed
form is ``--fd_gmm_mode posterior`` in the GMM entry point and is label-blind
(it died at its takeoff gate on 2026-08-27); the sampled-label form was tried
once before with a *fitted generative* q and diverged (``docs/gmm_posterior_loss``
§3(a)).  The bet here is that a *discriminative, normalised, p-initialised,
EMA-slowed* q does not have that pathology -- and §16's tail meters exist
specifically to catch it early if it does.

Nothing in the GMM implementation is touched by this file.

Usage: see ``scripts/run_jit_uncond_vlm_delta_100class.sh`` and
``docs/vlm_delta.md``.
"""

import argparse
import datetime
import json
import logging
import math
import os
import sys
import time
from collections import deque
from functools import partial
from pathlib import Path

import torch
import torch.distributed
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from utils.builders import create_generation_model, create_tokenizer
from utils.checkpoint_util import AsyncCheckpointSaver, ckpt_resume, save_checkpoint
from utils.distributed_util import (all_reduce_mean, broadcast_bool, preempt_requested,
                                    register_preempt_handler)
from utils.eval_util import evaluate_all_emas
from utils.grad_util import get_grad_norm
from utils.logging_util import MetricLogger, SmoothedValue
from utils.optimizer_util import create_optimizer
from frechet_distance.evaluator import FDEvaluator
from frechet_distance.queue import FeatureQueue
from frechet_distance.losses import (
    all_gather_plain,
    compute_frechet_distance_loss,
    diff_all_gather,
    load_mu_and_sigma_reference, precompute_sigma_ref_sqrt,
)
from frechet_distance.repr_models import load_repr_model, model_short_name
from frechet_distance.judges import (
    extract_judge_features,
    resolve_per_model_args, save_fd_queue_states, load_fd_queue_states,
    fill_all_queues, run_sanity_check,
)
from utils.rng_util import RNGStateManager
from utils.schedule_util import adjust_learning_rate
from utils.setup_util import setup
from utils.vis_util import visualize

from classifier_ensemble import ProbeClassifier  # held-out canary, never in the loss
from vlm_lora_q import (LoRAQ, build_lora_optimizer, load_q_optimizer_state,
                        q_lora_update_step, validate_loss_scale)
from vlm_linear_heads import (
    P_HEAD_BACKEND_QWEN,
    P_HEAD_BACKEND_TIMM,
    ClassBalancedFeatureBuffer,
    LocalClassMap,
    PerClassAccumulator,
    VLMDeltaHeads,
    build_head_from_checkpoint,
    delta_distribution_metrics,
    head_metrics,
    load_p_head_checkpoint,
    p_head_backend,
    p_head_identity,
    pq_agreement_metrics,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.optimize_ddp = False

logger = logging.getLogger("FD_loss")

# Metrics logged with a window of 1, i.e. the value measured at that exact step
# rather than a trailing median (see the comment where they are registered).
INSTANT_METERS = (
    "grad_x_fd", "grad_x_vlm_delta", "grad_ratio_vlm_fd", "cos_update_fd_vlm",
    # The five principals of §8 must come from the SAME step, or
    # vlm_delta_logqp != vlm_logq_c - vlm_logp_c and the table does not reconcile
    # (a difference of medians is not the median of differences).
    "vlm_delta_logqp", "vlm_delta_logqp_mean", "vlm_logq_c", "vlm_logp_c",
    "vlm_probq_c", "vlm_probp_c", "vlm_delta_loss_weighted",
    "vlm_delta_logqp_min", "vlm_delta_logqp_max", "vlm_delta_abs_p95",
    "vlm_delta_abs_p99", "vlm_delta_clamp_frac", "generator_grad_norm",
    "q_student_weight_delta_l2", "q_teacher_weight_delta_l2",
    "q_student_bias_delta_l2", "q_teacher_bias_delta_l2",
    "q_student_weight_cos_to_p", "q_teacher_weight_cos_to_p",
    "q_student_weight_delta_rel", "q_teacher_weight_delta_rel",
    "q_student_update_norm", "q_teacher_ema_update_norm", "q_train_steps",
    "q_buffer_size", "q_buffer_class_coverage",
    "q_buffer_min_samples_per_class", "q_buffer_max_samples_per_class",
    "samples_per_class", "vlm_scale_eff",
)

# Backend-specific meters. These MUST stay out of INSTANT_METERS: a meter that is
# registered but never updated keeps an empty deque, and MetricLogger.__str__
# calls max() on it at the first print -- which crashes the run at step 0 (or at
# the first print after a resume). Register them only when the backend that
# produces them is active.
BACKEND_INSTANT_METERS = {
    P_HEAD_BACKEND_QWEN: (
        "vlm_answer_state_samples", "vlm_answer_state_microbatch",
        "vlm_answer_state_norm",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_train_class_ids(train_class_ids, num_classes):
    """Validate and normalize an optional subset of generator label IDs."""
    if train_class_ids is None:
        return None
    class_ids = [int(class_id) for class_id in train_class_ids]
    if not class_ids:
        raise ValueError("--train_class_ids must contain at least one class ID")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("--train_class_ids must not contain duplicate class IDs")
    invalid = [c for c in class_ids if c < 0 or c >= num_classes]
    if invalid:
        raise ValueError(
            f"--train_class_ids contains IDs outside [0, {num_classes}): {invalid}")
    return class_ids


def vlm_scale_schedule(step, args):
    """Effective conditional weight at ``step``: 0 during warmup, then a ramp.

    Largely a formality here: because q is initialised from p the term ramps
    itself in from exactly zero as q drifts.  The explicit ramp is kept so a
    resumed run with an already-drifted q does not take a step change.
    """
    weight = args.vlm_delta_weight
    warmup = args.vlm_delta_warmup_steps
    ramp = args.vlm_delta_ramp_steps
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return weight
    return weight * min(1.0, (step - warmup) / float(ramp))


def evaluate_vlm_takeoff_gate(metrics, cond_delta_min=0.10, rank_ratio_max=0.90,
                              logic="any", check_cond_delta=True, check_rank=True):
    """Opt-in early takeoff check, evaluated from already-reduced metrics.

    Deliberately keyed on generator-side quantities that cannot be gamed by q:
    ``cond_delta`` (does the class token change the image at all?) and the
    frozen real-data p head's view of the *sampled* label's rank, expressed as a
    fraction of chance.  Missing / non-finite values fail closed.
    """
    if logic not in ("any", "all"):
        raise ValueError(f"takeoff gate logic must be 'any' or 'all', got {logic!r}")
    if not check_cond_delta and not check_rank:
        raise ValueError("takeoff gate must enable at least one criterion")
    failures = {}
    if check_cond_delta:
        value = metrics.get("cond_delta")
        failures["cond_delta"] = (
            value is None or not math.isfinite(float(value))
            or float(value) < float(cond_delta_min))
    if check_rank:
        value = metrics.get("vlm_p_target_rank_ratio_to_chance")
        failures["p_rank"] = (
            value is None or not math.isfinite(float(value))
            or float(value) > float(rank_ratio_max))
    reducer = any if logic == "any" else all
    return reducer(failures.values()), failures


def validate_vlm_takeoff_gate_args(args):
    step = args.vlm_takeoff_gate_step
    if step < 0:
        return
    if args.vlm_takeoff_cond_delta_min <= 0.0 and args.vlm_takeoff_no_rank:
        raise ValueError("takeoff gate has no enabled criteria")
    total_steps = args.epochs * args.steps_per_epoch
    if step >= total_steps:
        raise ValueError(
            f"--vlm_takeoff_gate_step={step} is outside the {total_steps}-step run")
    if step <= args.vlm_delta_warmup_steps + args.vlm_delta_ramp_steps:
        raise ValueError(
            f"--vlm_takeoff_gate_step={step} is inside the warmup+ramp "
            f"({args.vlm_delta_warmup_steps}+{args.vlm_delta_ramp_steps})")


@torch.no_grad()
def probe_per_sample_correct(probe, images01, labels):
    """Per-sample held-out ResNet-50 correctness (for the per-class table).

    ``ProbeClassifier.stats`` only returns batch aggregates; the per-class
    accumulator needs the per-sample vector.  Preprocessing is copied from
    ``ProbeClassifier.stats`` so the two always agree.
    """
    x = F.interpolate(images01.float(), size=(224, 224), mode="bicubic",
                      align_corners=False, antialias=True)
    logits = probe.model((x - probe.mean) / probe.std)
    return (logits.argmax(-1) == labels).float()


# ---------------------------------------------------------------------------
# VLM-delta setup
# ---------------------------------------------------------------------------

@torch.no_grad()
def vlm_features_no_grad(vlm_judge, images, args):
    """``z`` for a batch of images, with no graph and no memory surprises.

    The FD-judge backend is cheap enough to take any batch in one go.  The
    answer-state backend is a 7B forward, so the batch is split into microbatches
    of ``--vlm_microbatch``; a queue-fill batch of 256 images in one call would
    otherwise be a several-tens-of-GB activation spike for no benefit.
    """
    if vlm_judge.get("vlm_backend") != P_HEAD_BACKEND_QWEN:
        return extract_judge_features(vlm_judge, images)
    step = max(1, int(args.vlm_microbatch))
    chunks = [extract_judge_features(vlm_judge, images[i:i + step])
              for i in range(0, images.shape[0], step)]
    return torch.cat(chunks, dim=0)


def _attach_fd_judge_extractor(judges, judge_model_names, identity, args):
    """z comes from one of the FD judges -- free, because it is already computed.

    Refuses to launch unless that judge's representation is bit-for-bit the one
    the frozen p head was fitted on.
    """
    names = [judge["name"] for judge in judges]
    if args.vlm_judge is None:
        raise ValueError("--vlm_judge is required (the short judge name carrying the VLM)")
    if args.vlm_judge not in names:
        raise ValueError(
            f"--vlm_judge='{args.vlm_judge}' not among the configured judges {names}")
    index = names.index(args.vlm_judge)
    judge = judges[index]

    problems = []
    if identity["vlm_model_name"] != judge_model_names[index]:
        problems.append(
            f"p head was trained on VLM '{identity['vlm_model_name']}' but judge "
            f"'{args.vlm_judge}' is '{judge_model_names[index]}'")
    if identity["vlm_pool_type"] != judge["pool_type"]:
        problems.append(
            f"p head pool_type={identity['vlm_pool_type']} != judge "
            f"pool_type={judge['pool_type']}")
    judge_target = int(getattr(judge["model"], "target_size", -1))
    if judge_target > 0 and identity["vlm_target_size"] != judge_target:
        problems.append(
            f"p head vlm_target_size={identity['vlm_target_size']} != judge "
            f"target_size={judge_target}")
    if identity["feature_dim"] != int(judge["feat_dim"]):
        problems.append(
            f"p head feature_dim={identity['feature_dim']} != judge "
            f"feat_dim={judge['feat_dim']}")
    if identity["vlm_input_size"] != int(args.img_size):
        problems.append(
            f"p head was trained on {identity['vlm_input_size']}px real crops but "
            f"the generator produces {args.img_size}px images")
    if problems:
        raise ValueError(
            "the VLM representation does not match the frozen p head:\n  - "
            + "\n  - ".join(problems)
            + "\nRetrain the head with train_vlm_p_head.py against this judge.")
    judge["vlm_backend"] = P_HEAD_BACKEND_TIMM
    judge["vlm_extractor"] = None
    return judge, index


ANSWER_STATE_JUDGE_NAME = "qwen_answer"


def _attach_answer_state_extractor(ckpt, identity, args, fd_judge_names=()):
    """z comes from a prompted Qwen2.5-VL answer state -- a second, separate VLM.

    This is NOT an FD judge: it owns no reference statistics, no feature queue
    and no Frechet term, and it is not in ``judges``.  It is only the
    representation the p/q heads sit on, and the price of that is a 7B
    forward+backward per scored image every step.
    """
    from qwen_answer_state import QwenAnswerStateExtractor

    extractor = QwenAnswerStateExtractor(
        str(ckpt["vlm_model_name"]),
        prompt=str(ckpt["vlm_prompt"]),
        layer=int(ckpt["vlm_layer"]),
        image_size=int(ckpt["vlm_input_size"]),
        dtype=args.vlm_dtype,
        attn_implementation=args.vlm_attn_implementation,
        microbatch_size=args.vlm_microbatch,
        max_samples_per_step=args.vlm_samples_per_step,
    )
    live = extractor.identity()
    problems = [
        f"{key}: p head has {identity[key]!r}, this extractor produces {live[key]!r}"
        for key in ("vlm_model_name", "vlm_layer", "vlm_prompt_sha256",
                    "feature_dim", "vlm_input_size")
        if identity.get(key) != live.get(key)
    ]
    if identity["vlm_input_size"] != int(args.img_size):
        problems.append(
            f"p head was trained on {identity['vlm_input_size']}px real crops but "
            f"the generator produces {args.img_size}px images")
    if problems:
        raise ValueError(
            "the VLM representation does not match the frozen p head:\n  - "
            + "\n  - ".join(problems)
            + "\nRefit the head with train_vlm_p_head.py --vlm_backend "
              "qwen_answer_state against this exact prompt and layer.")

    # The name MUST NOT collide with any FD judge: fill_all_queues' feature
    # collector routes buffer pushes by name, so a shared name would push a
    # 1152-d SigLIP feature into this 3584-d buffer.
    if ANSWER_STATE_JUDGE_NAME in fd_judge_names:
        raise ValueError(
            f"an FD judge is already called {ANSWER_STATE_JUDGE_NAME!r}; the "
            "answer-state extractor needs a distinct name to keep the q replay "
            "buffer from being fed the wrong features")
    if args.vlm_judge and args.vlm_judge != ANSWER_STATE_JUDGE_NAME:
        logger.warning(
            "[VLM-delta] --vlm_judge=%r is ignored for the answer-state backend: "
            "z comes from a separately loaded VLM, not from an FD judge. Using "
            "%r.", args.vlm_judge, ANSWER_STATE_JUDGE_NAME)
    judge = {
        "name": ANSWER_STATE_JUDGE_NAME,
        "model": extractor,
        "feat_dim": int(extractor.feat_dim),
        # forward() returns (z, z), so either pool selects the same feature.
        "pool_type": str(identity["vlm_pool_type"]),
        "vlm_backend": P_HEAD_BACKEND_QWEN,
        "vlm_extractor": extractor,
    }
    logger.info(
        "[VLM-delta] answer-state extractor: layer %d, prompt sha %s, %d visual "
        "tokens, microbatch %d, %s images/rank/step",
        extractor.layer, extractor.prompt_sha256[:16], extractor.num_visual_tokens,
        args.vlm_microbatch,
        "all" if args.vlm_samples_per_step == 0 else args.vlm_samples_per_step)
    logger.warning(
        "[VLM-delta] this backend is NOT an FD judge: it adds a 7B forward and "
        "backward per scored image every step (measured ~20 img/s/GPU at "
        "microbatch 24). Measure the step time before committing to a long run.")
    return judge, None


def setup_vlm_delta(judges, judge_model_names, args):
    """Attach the p / q_student / q_teacher heads to a feature source.

    Which source -- an existing FD judge or a separately loaded prompted VLM --
    is read off the p-head checkpoint, because the head *is* the definition of
    z.  Refuses to launch unless the run's class set and the live representation
    match the ones the frozen p head was trained on.
    """
    if not args.vlm_p_head:
        raise ValueError("--vlm_p_head is required (see train_vlm_p_head.py)")
    ckpt = load_p_head_checkpoint(args.vlm_p_head)
    identity = p_head_identity(ckpt)
    backend = p_head_backend(ckpt)
    use_lora = getattr(args, "vlm_q_lora", False)
    validate_loss_scale(getattr(args, "vlm_vjp_loss_scale", 1.0))
    validate_loss_scale(getattr(args, "vlm_q_loss_scale", 1.0))
    if use_lora and backend != P_HEAD_BACKEND_QWEN:
        raise ValueError("--vlm_q_lora currently requires a qwen_answer_state p checkpoint")
    if use_lora and args.vlm_q_bootstrap_updates:
        raise ValueError("LoRA q starts equal to p: set --vlm_q_bootstrap_updates 0")

    # The p head IS the definition of z, so it -- not a flag here -- decides
    # which VLM this run has to instantiate.
    if backend == P_HEAD_BACKEND_QWEN:
        judge, index = _attach_answer_state_extractor(
            ckpt, identity, args, fd_judge_names=[j["name"] for j in judges])
    else:
        judge, index = _attach_fd_judge_extractor(judges, judge_model_names,
                                                  identity, args)

    drawable = (args.train_class_ids if args.train_class_ids is not None
                else list(range(args.num_classes)))
    class_map = LocalClassMap([int(c) for c in ckpt["class_ids"]],
                              num_global=args.num_classes)
    class_map.validate(drawable)

    p_head = build_head_from_checkpoint(ckpt, device="cuda")
    temperature = float(args.vlm_head_temperature if args.vlm_head_temperature is not None
                        else ckpt["temperature"])
    heads = VLMDeltaHeads(p_head, temperature=temperature,
                          ema_beta=args.vlm_q_ema_beta,
                          use_ema=args.vlm_q_use_ema).cuda()

    # How many times q trains on any one generated sample.  A sample lives in the
    # buffer for buffer_size/global_batch steps and each update draws q_batch of
    # buffer_size, so
    #     reuse = q_batch * updates_per_step / global_batch
    # -- independent of the buffer size, which controls staleness, not reuse.
    # Measured offline on this run's own replay buffer: at reuse 5.3 the head
    # reaches 40% top-1 on buffer samples while staying at chance (1%) on fresh
    # ones, i.e. it memorises and the field it injects on the samples the
    # generator is actually evaluated at is noise.  Reuse ~1 makes that
    # impossible.
    global_batch = args.batch_size * max(1, getattr(args, "world_size", 1))
    if args.vlm_q_batch_size <= 0:
        args.vlm_q_batch_size = global_batch
    args.vlm_q_reuse = (args.vlm_q_batch_size * max(1, args.vlm_q_updates_per_step)
                        / max(1, global_batch))

    per_class_capacity = max(1, args.vlm_q_buffer_size // heads.num_classes)
    buffer = None if use_lora else ClassBalancedFeatureBuffer(
        heads.num_classes, heads.feature_dim, per_class_capacity,
        device="cuda", seed=args.seed + 7717)

    lora = None
    if use_lora:
        lora = LoRAQ(judge["vlm_extractor"], heads, rank=args.vlm_q_lora_rank,
                     alpha=args.vlm_q_lora_alpha, scope=args.vlm_q_lora_scope,
                     targets=args.vlm_q_lora_targets)
        logger.info("[LoRA q] %d adapted modules, %d student adapter parameters; "
                    "training on fresh scored images (feature replay/bootstrap and "
                    "--vlm_q_batch_size do not apply). Config: %s",
                    len(lora.layers), sum(p.numel() for p in lora.adapter_parameters()),
                    lora.config)
    q_optimizer = (_build_q_optimizer(heads, args) if lora is None
                   else build_lora_optimizer(lora, args))

    judge["vlm_heads"] = heads
    judge["vlm_lora"] = lora
    judge["vlm_buffer"] = buffer
    judge["vlm_q_optimizer"] = q_optimizer
    judge["vlm_class_map"] = class_map
    judge["vlm_feat_index"] = index
    judge["vlm_p_identity"] = identity
    judge["vlm_p_ckpt_meta"] = {
        "path": os.path.abspath(args.vlm_p_head),
        "identity": identity,
        "val_top1": (ckpt.get("train_metadata", {}) or {}).get("best", {}).get("top1"),
    }

    logger.info(
        "[VLM-delta] Attached to '%s' (%s): C=%d, d=%d, T=%.4f, "
        "weight=%.6g, q_lr=%.3g, q_opt=%s, q_updates/step=%d, ema_beta=%.5f, "
        "buffer=%d (%d/class), q_batch=%d",
        judge["name"], identity["vlm_model_name"], heads.num_classes,
        heads.feature_dim, temperature, args.vlm_delta_weight, args.vlm_q_lr,
        args.vlm_q_optimizer, args.vlm_q_updates_per_step, args.vlm_q_ema_beta,
        0 if lora is not None else per_class_capacity * heads.num_classes,
        0 if lora is not None else per_class_capacity,
        (min(args.batch_size, args.vlm_samples_per_step or args.batch_size)
         * max(1, args.world_size)) if lora is not None else args.vlm_q_batch_size)
    logger.info("[VLM-delta] p head: %s (sha256 %s..., real val top-1 %s)",
                args.vlm_p_head, identity["p_head_sha256"][:16],
                judge["vlm_p_ckpt_meta"]["val_top1"])
    logger.info("[VLM-delta] generator q source: %s; q AdamW betas=(%g, %g)",
                "EMA teacher" if heads.use_ema else "current student (no EMA)",
                args.vlm_q_beta1, args.vlm_q_beta2)
    if lora is None:
        logger.info(
            "[VLM-delta] q sample reuse = q_batch(%d) * updates(%d) / global_batch(%d) "
            "= %.2fx; q weight_decay=%.4g. Watch q_generalization_gap "
            "(fresh-batch CE minus buffer CE): growing = q is memorising its replay "
            "buffer and the injected field is noise on fresh samples.",
            args.vlm_q_batch_size, args.vlm_q_updates_per_step, global_batch,
            args.vlm_q_reuse, args.vlm_q_weight_decay)
    if lora is None and args.vlm_q_reuse > 2.0:
        logger.warning(
            "[VLM-delta] q sample reuse is %.1fx. Above ~2x the q head memorises "
            "its replay buffer instead of estimating q(c|z); set "
            "--vlm_q_batch_size to the global batch (%d) unless you mean to.",
            args.vlm_q_reuse, global_batch)
    if lora is None and args.vlm_q_batch_size < heads.num_classes / 2:
        # Reuse and batch size trade off: reuse 1.0 pins the q batch to the
        # global batch, and if that is small relative to the number of classes
        # each CE step is dominated by a handful of samples, so q overfits them
        # and its CE on everything else RISES.  Measured at global batch 8 with
        # C=100: q_ce climbed 5.3 -> 7.5 in twelve steps.
        logger.warning(
            "[VLM-delta] the q batch is %d for a %d-way problem (reuse %.2fx). "
            "Each CE step sees fewer than half a sample per class, so q takes "
            "very noisy steps and its CE can rise. Raise the global batch, or "
            "accept reuse >1 by setting --vlm_q_batch_size explicitly.",
            args.vlm_q_batch_size, heads.num_classes, args.vlm_q_reuse)
    logger.info(
        "[VLM-delta] objective is the SAMPLED-LABEL scalar "
        "E[log q(c|z) - log p(c|z)] -- NOT the class-summed posterior KL "
        "and NOT a second -log p(c|z) driver. Calibrate on grad_ratio_vlm_fd "
        "(target sustained 0.22-0.30); the loss value is not comparable to "
        "anything.")
    if args.vlm_delta_clamp > 0:
        logger.warning(
            "[VLM-delta] per-sample clamp ACTIVE at +-%.3f: this CHANGES the "
            "objective. Watch vlm_delta_clamp_frac.", args.vlm_delta_clamp)
    return judge


def _build_q_optimizer(heads, args):
    params = list(heads.q_student.linear.parameters())
    if args.vlm_q_optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.vlm_q_lr,
                               momentum=args.vlm_q_momentum,
                               weight_decay=args.vlm_q_weight_decay)
    return torch.optim.AdamW(params, lr=args.vlm_q_lr,
                             betas=(args.vlm_q_beta1, args.vlm_q_beta2),
                             weight_decay=args.vlm_q_weight_decay)


# ---------------------------------------------------------------------------
# q update
# ---------------------------------------------------------------------------

def q_update_step(vlm_judge, z_detached, y_local, args, collect_metrics=False):
    """Train ``q_student`` on detached features; update EMA only when enabled.

    The generator receives no gradient (``z_detached``), the VLM receives no
    parameter gradient (it is frozen and outside this graph entirely) and the p
    head is never touched.  Every rank runs this on the *same* globally gathered
    batch with the same seeded buffer RNG, so q stays identical across ranks
    without a data-parallel wrapper; the gradient all-reduce below is exactness
    insurance against float non-determinism accumulating over 50k steps.
    """
    heads = vlm_judge["vlm_heads"]
    buffer = vlm_judge["vlm_buffer"]
    optimizer = vlm_judge["vlm_q_optimizer"]
    out = {}

    if collect_metrics:
        with torch.no_grad():
            pre = head_metrics(heads.q_student_log_probs(z_detached), y_local,
                               "vlm_q_student")
        out.update({f"{k}_pre": v for k, v in pre.items()})

    buffer.push(z_detached, y_local)

    before = heads.student_snapshot()
    total_ce, total_acc, total_gn, n_updates = 0.0, 0.0, 0.0, 0
    for _ in range(max(0, args.vlm_q_updates_per_step)):
        batch = buffer.sample(args.vlm_q_batch_size)
        if batch is None:
            break
        z_batch, y_batch = batch
        logits = heads.q_student.logits(z_batch, heads.temperature)
        loss = F.cross_entropy(logits, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if torch.distributed.is_initialized():
            for param in heads.q_student.linear.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(
                        param.grad, op=torch.distributed.ReduceOp.AVG)
        if args.vlm_q_grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                heads.q_student.linear.parameters(), args.vlm_q_grad_clip)
        else:
            grad_norm = get_grad_norm(heads.q_student.linear.parameters())
        if not torch.isfinite(grad_norm):
            raise RuntimeError(
                "[VLM-delta] non-finite gradient in the q-head update; aborting")
        optimizer.step()
        total_ce += float(loss.detach())
        total_acc += float((logits.argmax(-1) == y_batch).float().mean())
        total_gn += float(grad_norm)
        n_updates += 1

    if n_updates:
        heads.record_student_update(before)
        heads.q_train_steps.add_(n_updates)
        heads.ema_update()
        out.update({
            "q_ce": total_ce / n_updates,
            "q_train_accuracy": total_acc / n_updates,
            "q_grad_norm": total_gn / n_updates,
            "q_lr": float(optimizer.param_groups[0]["lr"]),
            "q_updates_applied": float(n_updates),
        })

    if collect_metrics:
        with torch.no_grad():
            post = head_metrics(heads.q_student_log_probs(z_detached), y_local,
                                "vlm_q_student")
        out.update({f"{k}_post": v for k, v in post.items()})
        out.update(post)  # unsuffixed names = the post-update student view
        # The honest generalisation meter: CE on a batch q has NOT trained on
        # (it enters the buffer only after the _pre read) minus CE on the buffer
        # samples it just trained on.  Growing = memorisation.
        if "q_ce" in out and "vlm_q_student_ce_pre" in out:
            out["q_generalization_gap"] = out["vlm_q_student_ce_pre"] - out["q_ce"]
        out.update(heads.drift_metrics())
        out.update(heads.teacher_student_agreement(z_detached))
        out.update(buffer.stats())
        out["q_weight_norm"] = float(heads.q_student.weight.detach().norm())
        out["q_bias_norm"] = float(heads.q_student.bias.detach().norm())
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            w = heads.q_student.weight.detach()
            w_min, w_max = w.clone(), w.clone()
            torch.distributed.all_reduce(w_min, op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(w_max, op=torch.distributed.ReduceOp.MAX)
            out["q_param_rank_max_dev"] = float((w_max - w_min).abs().max())
    return out


# ---------------------------------------------------------------------------
# FD + VLM-delta train step
# ---------------------------------------------------------------------------

def _diag_images(sampled, selected):
    """The images the VLM actually scored, so probe and heads line up row-wise."""
    images = sampled if selected is None else sampled.index_select(0, selected)
    return images.detach()


def training_judge_features(judge, images, *, microbatch=0, checkpoint_features=False):
    """Bound FD activation memory while retaining gradients of the full batch.

    Frozen judges are in eval mode, so images can be evaluated independently.
    Non-reentrant checkpointing also supports the image-gradient diagnostics.
    Bind the judge now: backward may recompute after the caller's judge loop.
    """
    if microbatch < 0:
        raise ValueError("fd_feature_microbatch must be non-negative")
    forward = partial(extract_judge_features, judge)
    chunks = images.split(microbatch or images.shape[0])
    features = []
    for chunk in chunks:
        if checkpoint_features and torch.is_grad_enabled() and chunk.requires_grad:
            features.append(checkpoint(forward, chunk, use_reentrant=False))
        else:
            features.append(forward(chunk))
    return torch.cat(features, dim=0)


def training_generated_images(model, noise, labels, sampling_args, *,
                              microbatch=0, checkpoint_generator=False):
    """Checkpoint each generator microbatch to bound backward recomputation.

    Concatenate images before computing any batch statistics or loss. This
    keeps the full-batch objective and one generator optimizer step. Noise and
    labels are sampled before splitting, and checkpointing preserves RNG state.
    """
    if microbatch < 0:
        raise ValueError("generator_microbatch must be non-negative")
    if noise.shape[0] != labels.shape[0]:
        raise ValueError("generator noise and labels must have the same batch size")
    size = microbatch or noise.shape[0]
    images = []
    for z, y in zip(noise.split(size), labels.split(size)):
        if checkpoint_generator and torch.is_grad_enabled():
            images.append(checkpoint(model.sample_images_with_grad, z, y,
                                     sampling_args=sampling_args, use_reentrant=False))
        else:
            images.append(model.sample_images_with_grad(z, y, sampling_args=sampling_args))
    return torch.cat(images, dim=0)


def get_fd_train_step(model_wo_ddp, judges, sampling_args, args, tokenizer=None,
                      vlm_judge=None, probe=None):
    fid_norm_eps = args.fd_fid_norm_eps
    batch_size = args.batch_size
    num_classes = args.num_classes
    input_shape = (args.input_channels, args.input_size, args.input_size)
    delta_clamp = args.vlm_delta_clamp
    vlm_answer_state = (vlm_judge is not None
                        and vlm_judge.get("vlm_backend") == P_HEAD_BACKEND_QWEN)
    lora = None if vlm_judge is None else vlm_judge.get("vlm_lora")

    train_class_ids = getattr(args, "train_class_ids", None)
    train_class_ids_tensor = (
        None if train_class_ids is None
        else torch.tensor(train_class_ids, dtype=torch.long, device="cuda"))

    def _sample_labels():
        if train_class_ids_tensor is None:
            return torch.randint(0, num_classes, (batch_size,), device="cuda")
        picks = torch.randint(0, train_class_ids_tensor.numel(), (batch_size,),
                              device="cuda")
        return train_class_ids_tensor[picks]

    def fd_train_step(vlm_scale=None, diag=False):
        if vlm_scale is None:
            vlm_scale = args.vlm_delta_weight

        z_noise = torch.randn(batch_size, *input_shape, device="cuda") * args.noise_scale
        y = _sample_labels()
        sampled = training_generated_images(
            model_wo_ddp, z_noise, y, sampling_args,
            microbatch=getattr(args, "generator_microbatch", 0),
            checkpoint_generator=getattr(args, "grad_checkpointing", False))
        if tokenizer is not None:
            sampled = tokenizer.decode(tokenizer.denormalize_z(sampled))
        sampled = (sampled * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]

        loss = torch.tensor(0.0, device="cuda")
        loss_dict = {}

        all_new_feats = [diff_all_gather(training_judge_features(
                            judge, sampled,
                            microbatch=getattr(args, "fd_feature_microbatch", 0),
                            checkpoint_features=getattr(args, "fd_feature_checkpoint", False)))
                         for judge in judges]

        fd_term = torch.zeros((), device="cuda")
        fd_raw_sum = 0.0
        for i, judge in enumerate(judges):
            new_feats = all_new_feats[i]
            ns_kwargs = dict(sigma_ref_sqrt=judge.get("sigma_ref_sqrt"))
            if judge["queue"].online_accum or judge["queue"].ema_stats:
                mu, sigma = judge["queue"].build_feats_stats(new_feats)
                fid = compute_frechet_distance_loss(judge["mu_ref"], judge["sigma_ref"],
                                                    mu=mu, sigma=sigma, **ns_kwargs)
            else:
                all_feats = judge["queue"].build_feats_snapshot(new_feats)
                fid = compute_frechet_distance_loss(judge["mu_ref"], judge["sigma_ref"],
                                                    all_feats=all_feats, **ns_kwargs)
            fid_loss = fid / (fid.detach() + fid_norm_eps)
            fd_term = fd_term + judge["weight"] * fid_loss
            fd_raw_sum += float(fid.detach())
            loss_dict[f"fid_{judge['name']}"] = float(fid.detach())
        loss = loss + fd_term
        loss_dict["fd_loss_raw"] = fd_raw_sum
        loss_dict["fd_loss_norm"] = float(fd_term.detach())

        # -- the conditional term: E[ log q_generator(c|z) - log p(c|z) ] --
        vlm_term_loss = None
        z_vlm_detached, y_all, y_local, vlm_selected = None, None, None, None
        if vlm_judge is not None:
            heads = vlm_judge["vlm_heads"]
            class_map = vlm_judge["vlm_class_map"]
            y_all = all_gather_plain(y)
            y_local = class_map.to_local(y_all)

            if vlm_answer_state:
                # z is not among the FD features: it needs its own 7B forward,
                # and its graph must not survive to loss.backward().  The
                # surrogate carries this rank's share of the global mean as its
                # value and the term's exact image gradient as its derivative.
                extractor = vlm_judge["vlm_extractor"]
                if lora is not None:
                    scored_global = (min(sampled.shape[0], extractor.max_samples_per_step
                                         or sampled.shape[0])
                                     * max(1, getattr(args, "world_size", 1)))
                    vlm_surrogate, z_local, logq_local, vlm_selected = lora.generator_surrogate(
                        sampled, class_map.to_local(y), denominator=scored_global,
                        clamp=delta_clamp, loss_scale=args.vlm_vjp_loss_scale,
                        need_grad=bool(vlm_scale != 0.0))
                elif vlm_scale == 0.0:
                    # Warmup: the term contributes no gradient, so pay for the
                    # forward that feeds q and skip the 7B backward entirely.
                    with torch.no_grad():
                        vlm_selected = extractor.select_indices(
                            sampled.shape[0], sampled.device)
                        z_local = vlm_features_no_grad(
                            vlm_judge, sampled.index_select(0, vlm_selected).detach(),
                            args)
                    vlm_surrogate = None
                    # Report the same meters as the VJP path: they are
                    # pre-registered in INSTANT_METERS, and a meter that is
                    # never updated has count 0 and blows up the logger.
                    extractor.last_stats = {
                        "vlm_answer_state_samples": float(vlm_selected.numel()),
                        "vlm_answer_state_microbatch": float(args.vlm_microbatch),
                        "vlm_answer_state_norm": float(
                            z_local.float().norm(dim=1).mean()),
                    }
                else:
                    y_local_rank = class_map.to_local(y)
                    # Divided by the GLOBAL scored count, so summing this rank's
                    # share over ranks gives exactly the mean the FD-judge path
                    # computes -- the two backends stay on one weight scale.
                    scored_global = (min(sampled.shape[0],
                                         extractor.max_samples_per_step
                                         or sampled.shape[0])
                                     * max(1, getattr(args, "world_size", 1)))
                    delta_scale = 1.0 / scored_global

                    def _delta_term(z_live, mb_labels):
                        d, _, _ = heads.delta_log_qp(z_live, mb_labels)
                        if delta_clamp > 0:
                            d = d.clamp(-delta_clamp, delta_clamp)
                        return d.sum() * delta_scale

                    vlm_surrogate, z_local, vlm_selected = extractor.vjp_surrogate(
                        sampled, y_local_rank, _delta_term,
                        loss_scale=getattr(args, "vlm_vjp_loss_scale", 1.0))
                loss_dict.update(extractor.last_stats)
                # Every rank pushes the same all-gathered batch, so q stays
                # bit-identical across ranks without a collective of its own.
                z = all_gather_plain(z_local)
                y_all = all_gather_plain(y.index_select(0, vlm_selected))
                y_local = class_map.to_local(y_all)
                logp_all = heads.p_log_probs(z)
                logq_all = (heads.q_generator_log_probs(z) if lora is None
                            else all_gather_plain(logq_local))
            else:
                z = all_new_feats[vlm_judge["vlm_feat_index"]]
                logp_all = heads.p_log_probs(z)
                logq_all = heads.q_generator_log_probs(z)

            # Preserve the actual scoring distribution for per-class diagnostics
            # after the q optimizer changes the student later in this step.
            vlm_judge["vlm_scored_logq"] = logq_all.detach()

            idx = y_local.view(-1, 1)
            logp_c = logp_all.gather(1, idx).squeeze(1)
            logq_c = logq_all.gather(1, idx).squeeze(1)
            delta = logq_c - logp_c

            if delta_clamp > 0:
                clamped = delta.clamp(-delta_clamp, delta_clamp)
                loss_dict["vlm_delta_clamp_frac"] = float(
                    (delta.detach() != clamped.detach()).float().mean())
                delta = clamped
            else:
                loss_dict["vlm_delta_clamp_frac"] = 0.0

            vlm_delta_loss = delta.mean()
            if not torch.isfinite(vlm_delta_loss):
                raise RuntimeError(
                    "[VLM-delta] non-finite log q - log p; aborting rather than "
                    "silently clamping the objective")
            if vlm_answer_state:
                # Same value, same image gradient, no 7B activations retained.
                vlm_term_loss = (
                    vlm_delta_loss.detach() * vlm_scale if vlm_surrogate is None
                    else vlm_scale * (vlm_delta_loss.detach() + vlm_surrogate
                                      - vlm_surrogate.detach()))
            else:
                vlm_term_loss = vlm_scale * vlm_delta_loss
            loss = loss + vlm_term_loss

            # -- always-on scalars: the pathology watch (§16) --
            with torch.no_grad():
                d = delta.detach()
                loss_dict["vlm_delta_logqp"] = float(d.mean())
                loss_dict["vlm_logq_c"] = float(logq_c.detach().mean())
                loss_dict["vlm_logp_c"] = float(logp_c.detach().mean())
                loss_dict["vlm_probq_c"] = float(logq_c.detach().exp().mean())
                loss_dict["vlm_probp_c"] = float(logp_c.detach().exp().mean())
                loss_dict.update(delta_distribution_metrics(d))
                loss_dict["vlm_delta_loss_weighted"] = float(vlm_term_loss.detach())

            if diag:
                with torch.no_grad():
                    lp, lq = logp_all.detach(), logq_all.detach()
                    p_stats = head_metrics(lp, y_local, "vlm_p")
                    p_stats["vlm_p_margin_target_vs_best_other"] = p_stats["vlm_p_margin"]
                    n_cls = heads.num_classes
                    p_stats["vlm_p_target_rank_ratio_to_chance"] = (
                        p_stats["vlm_p_target_rank"] / ((n_cls + 1) / 2.0))
                    loss_dict.update(p_stats)
                    loss_dict.update(head_metrics(lq, y_local, "vlm_q_generator"))
                    if heads.use_ema:
                        loss_dict.update(head_metrics(lq, y_local, "vlm_q_teacher"))
                    loss_dict.update(pq_agreement_metrics(lp, lq, y_local))
                    loss_dict["vlm_chance_top1"] = 1.0 / heads.num_classes
                    loss_dict["vlm_chance_mean_rank"] = (heads.num_classes + 1) / 2.0

            if vlm_answer_state:
                # The same forward that produced the gradient already produced
                # z; a second 7B pass to re-derive it would be pure waste.
                z_vlm_detached = z.detach()
            elif args.vlm_q_recompute_features:
                # The literal formulation from the spec: a second frozen-VLM
                # forward on detached images.  Numerically identical to
                # detaching the graph features (same frozen function, same
                # input, eval mode) and strictly more expensive; available so
                # the exact code path can be exercised and compared.
                with torch.no_grad():
                    z_vlm_detached = diff_all_gather(
                        extract_judge_features(vlm_judge, sampled.detach())).detach()
            else:
                z_vlm_detached = z.detach()

        # -- per-term image-space gradient diagnostics --
        if diag:
            def _grad_x(term):
                # A term with no grad_fn is not an error: at vlm_scale == 0 the
                # answer-state path deliberately skips the 7B backward and
                # returns a detached constant, whose image gradient IS zero.
                if term is None or not term.requires_grad:
                    return None
                g = torch.autograd.grad(term, sampled, retain_graph=True,
                                        allow_unused=True)[0]
                return None if g is None else g.detach().reshape(-1)
            g_fd, g_vlm = _grad_x(fd_term), _grad_x(vlm_term_loss)

            def _norm(g):
                return float(g.norm()) if g is not None else 0.0

            def _cos(a, b):
                if a is None or b is None:
                    return 0.0
                d = float(a.norm() * b.norm())
                return float(a @ b) / d if d > 0 else 0.0

            loss_dict["grad_x_fd"] = _norm(g_fd)
            loss_dict["grad_x_vlm_delta"] = _norm(g_vlm)
            loss_dict["cos_update_fd_vlm"] = _cos(g_fd, g_vlm)
            loss_dict["grad_ratio_vlm_fd"] = (
                loss_dict["grad_x_vlm_delta"] / loss_dict["grad_x_fd"]
                if loss_dict["grad_x_fd"] > 0 else 0.0)

            # -- label sensitivity: same noise, rolled labels --
            with torch.no_grad():
                y_alt = torch.roll(y, 1, dims=0)
                alt = training_generated_images(
                    model_wo_ddp, z_noise, y_alt, sampling_args,
                    microbatch=getattr(args, "generator_microbatch", 0))
                if tokenizer is not None:
                    alt = tokenizer.decode(tokenizer.denormalize_z(alt))
                alt = (alt * 0.5 + 0.5).clamp(0, 1)
                ref = sampled.detach()
                delta_px = (ref - alt).flatten(1).norm(dim=1)
                scale = ref.flatten(1).norm(dim=1).clamp_min(1e-8)
                loss_dict["cond_delta"] = float((delta_px / scale).mean())
                loss_dict["cond_delta_valid"] = float((y_alt != y).float().mean())

                if vlm_judge is not None:
                    # ... and the same question in VLM feature space, which can
                    # move before pixels do.
                    f_ref = vlm_features_no_grad(vlm_judge, ref, args).float()
                    f_alt = vlm_features_no_grad(vlm_judge, alt, args).float()
                    fd_norm = (f_ref - f_alt).norm(dim=1)
                    fscale = f_ref.norm(dim=1).clamp_min(1e-8)
                    loss_dict["vlm_cond_feature_delta"] = float((fd_norm / fscale).mean())
                    loss_dict["vlm_cond_feature_cos"] = float(
                        F.cosine_similarity(f_ref, f_alt, dim=1).mean())

                if probe is not None:
                    loss_dict.update(probe.stats(sampled, y))

        loss.backward(create_graph=False)

        if torch.distributed.is_initialized():
            for param in model_wo_ddp.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(
                        param.grad, op=torch.distributed.ReduceOp.AVG)

        return (loss, loss_dict,
                tuple(f.detach() for f in all_new_feats),
                z_vlm_detached,
                None if y_local is None else y_local.detach(),
                _diag_images(sampled, vlm_selected) if diag or lora is not None else None,
                None if y_all is None else y_all.detach())

    return fd_train_step


# ---------------------------------------------------------------------------
# Initial "is the VLM clueless?" diagnostic
# ---------------------------------------------------------------------------

@torch.no_grad()
def initial_generated_diagnostics(vlm_judge, model, args, tokenizer=None, probe=None):
    """Characterise the de-conditioned starting point before any update.

    The question is NOT "does p recognise something in these images" but "does p
    recognise the SAMPLED REQUESTED class".  On a genuinely de-conditioned
    generator every meter below should read chance.
    """
    heads = vlm_judge["vlm_heads"]
    class_map = vlm_judge["vlm_class_map"]
    n_target = max(1, args.vlm_init_diag_samples)
    ids = torch.tensor(args.train_class_ids if args.train_class_ids is not None
                       else list(range(args.num_classes)),
                       dtype=torch.long, device="cuda")
    model.eval()
    feats, labels, probe_hits, n_probe = [], [], 0.0, 0
    lora = vlm_judge.get("vlm_lora")
    student_logps, teacher_logps = [], []
    collected = 0
    while collected < n_target:
        bsz = min(args.fd_queue_fill_bsz, n_target - collected)
        y = ids[torch.randint(0, ids.numel(), (bsz,), device="cuda")]
        imgs = model.generate(bsz, y, cfg=args.cfg, args=args, verbose=False)
        imgs = tokenizer.detokenize(imgs) if tokenizer is not None else imgs * 0.5 + 0.5
        z = diff_all_gather(vlm_features_no_grad(vlm_judge, imgs, args)).detach()
        y_all = all_gather_plain(y)
        feats.append(z.float())
        if lora is not None:
            student_logps.append(all_gather_plain(lora.log_probs(imgs, "student", args.vlm_microbatch)))
            if heads.use_ema:
                teacher_logps.append(all_gather_plain(lora.log_probs(imgs, "teacher", args.vlm_microbatch)))
        labels.append(class_map.to_local(y_all))
        if probe is not None:
            probe_hits += float(probe_per_sample_correct(probe, imgs, y).sum())
            n_probe += int(y.numel())
        collected += bsz
    z = torch.cat(feats)
    y_local = torch.cat(labels)

    logp = heads.p_log_probs(z)
    logq_s = heads.q_student_log_probs(z) if lora is None else torch.cat(student_logps)
    logq_t = ((heads.q_teacher_log_probs(z) if lora is None else torch.cat(teacher_logps))
              if heads.use_ema else None)
    logq = logq_t if heads.use_ema else logq_s
    idx = y_local.view(-1, 1)
    delta = (logq.gather(1, idx) - logp.gather(1, idx)).squeeze(1)

    report = {
        "num_samples": int(y_local.numel()),
        "num_classes": heads.num_classes,
        "chance_top1": 1.0 / heads.num_classes,
        "chance_mean_rank": (heads.num_classes + 1) / 2.0,
        "temperature": heads.temperature,
        "p": head_metrics(logp, y_local, "p"),
        "q_student": head_metrics(logq_s, y_local, "q_student"),
        "q_generator_source": "teacher" if heads.use_ema else "student",
        "q_generator": head_metrics(logq, y_local, "q_generator"),
        "pq": pq_agreement_metrics(logp, logq, y_local, prefix="pq"),
        "delta": delta_distribution_metrics(delta, prefix="delta_logqp"),
        "init_equality": heads.init_equality_check(z),
    }
    if heads.use_ema:
        report["q_teacher"] = head_metrics(logq_t, y_local, "q_teacher")
    if lora is not None:
        report["init_equality"].update({
            "init_max_abs_logdiff_q_student_vs_p": float((logq_s - logp).abs().max()),
        })
        if heads.use_ema:
            report["init_equality"]["init_max_abs_logdiff_q_teacher_vs_p"] = float((logq_t - logp).abs().max())
    if probe is not None and n_probe:
        report["probe_top1"] = probe_hits / n_probe
    return report


def print_initial_diagnostics(report) -> None:
    n_cls = report["num_classes"]
    lines = [
        "", "=" * 78,
        "INITIAL GENERATED SAMPLE DIAGNOSTICS",
        "-" * 78,
        f"number of classes:   {n_cls}",
        f"samples:             {report['num_samples']}",
        f"chance top1:         {report['chance_top1']:.4f}",
        f"chance mean rank:    {report['chance_mean_rank']:.1f}",
        f"head temperature:    {report['temperature']:.4f}",
        "",
    ]
    for key, label in (("p", "p head"), ("q_student", "q student"),
                       ("q_teacher", "q teacher")):
        if key not in report:
            continue
        m = report[key]
        prefix = key
        lines += [
            f"{label}:",
            f"  target top1        {m[f'{prefix}_top1']:.4f}"
            f"   (chance {report['chance_top1']:.4f})",
            f"  target top5        {m[f'{prefix}_top5']:.4f}",
            f"  target rank mean   {m[f'{prefix}_target_rank']:.2f}"
            f"   median {m[f'{prefix}_target_rank_median']:.1f}"
            f"   (chance {report['chance_mean_rank']:.1f})",
            f"  target logp        {m[f'{prefix}_target_logp']:.4f}"
            f"   (uniform {-math.log(n_cls):.4f})",
            f"  target prob        {m[f'{prefix}_target_prob']:.4f}",
            f"  entropy            {m[f'{prefix}_entropy']:.4f}"
            f"   (uniform {math.log(n_cls):.4f})",
            "",
        ]
    pq, dl, eq = report["pq"], report["delta"], report["init_equality"]
    lines += [
        "p/q:",
        f"  mean logq-logp     {dl['delta_logqp_mean']:+.3e}"
        f"   (std {dl['delta_logqp_std']:.3e}, "
        f"min {dl['delta_logqp_min']:+.3e}, max {dl['delta_logqp_max']:+.3e})",
        f"  top1 agreement     {pq['pq_top1_agreement']:.4f}",
        f"  full KL(q||p)      {pq['pq_full_kl_qp']:.3e}",
        f"  full KL(p||q)      {pq['pq_full_kl_pq']:.3e}",
        "",
        "q == p check (must be ~0 at initialisation):",
        f"  max |log q_student - log p|   {eq['init_max_abs_logdiff_q_student_vs_p']:.3e}",
        f"  max |W_q_student - W_p|       {eq['init_max_abs_weight_diff_q_student_vs_p']:.3e}",
    ]
    if "q_teacher" in report:
        lines += [
            f"  max |log q_teacher - log p|   {eq['init_max_abs_logdiff_q_teacher_vs_p']:.3e}",
            f"  max |W_q_teacher - W_p|       {eq['init_max_abs_weight_diff_q_teacher_vs_p']:.3e}",
        ]
    if "probe_top1" in report:
        lines += ["",
                  f"held-out probe top1  {report['probe_top1']:.4f}"
                  f"   (chance {report['chance_top1']:.4f})"]
    p_rank = report["p"]["p_target_rank"]
    chance_rank = report["chance_mean_rank"]
    verdict = ("p is essentially CLUELESS about the sampled label "
               "(as expected from a de-conditioned generator)"
               if p_rank > 0.9 * chance_rank else
               "p already carries sampled-label signal -- the starting point is "
               "NOT fully de-conditioned; interpret later gains against this baseline")
    lines += ["", f"VERDICT: {verdict}", "=" * 78, ""]
    print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_and_evaluate(args):
    if args.vlm_disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    args.train_class_ids = validate_train_class_ids(
        getattr(args, "train_class_ids", None), args.num_classes)
    validate_vlm_takeoff_gate_args(args)
    wandb_logger = setup(args)
    # SIGTERM here, not just SIGUSR1: on shared GPUs an eviction arrives as
    # SIGTERM, and both earlier trials of this experiment died to one mid-step.
    # Best-effort -- the supervisor in scripts/supervise_vlm_delta.sh is what
    # actually closes the gap.
    register_preempt_handler(handle_sigterm=True)

    if args.train_class_ids is not None:
        logger.info("[ClassSubset] Sampling %d of %d labels during training: %s",
                    len(args.train_class_ids), args.num_classes, args.train_class_ids)

    tokenizer = create_tokenizer(args)
    model, ema_model = create_generation_model(args)
    optimizer = create_optimizer(args, model, print_trainable_params=True)
    model_wo_ddp = model

    extra = ckpt_resume(
        args, model_wo_ddp, optimizer, ema_model,
        extra_keys=["fd_queue_states", "vlm_delta_state",
                    "vlm_takeoff_gate_status"])
    restored_gate_status = None if extra is None else extra.get("vlm_takeoff_gate_status")
    if (args.vlm_takeoff_gate_step >= 0 and restored_gate_status is not None
            and restored_gate_status.get("aborted", False)):
        logger.error("[VLM takeoff gate] Refusing to resume a checkpoint already "
                     "marked as gated: %s", restored_gate_status)
        return 4

    rng = RNGStateManager()
    rng.save()
    if (not args.disable_vis) or args.vis_only:
        visualize(args, model_wo_ddp, ema_model, args.current_step, rng=rng,
                  tokenizer=tokenizer)
        if args.vis_only:
            return 0

    repr_model_eval, feat_dim_eval, _, _ = load_repr_model("inception")
    fid_evaluator = FDEvaluator(repr_model_eval, feat_dim_eval, args.fid_stats_path)

    resolve_per_model_args(args)
    judges, judge_model_names = [], []
    for name, stats_path, weight, pool_type, ts in zip(
        args.fd_repr_models, args.fd_repr_stats_paths,
        args.fd_repr_weights, args.fd_repr_pool_types, args.fd_target_sizes,
    ):
        repr_model, feat_dim, _, _ = load_repr_model(name, target_size=ts)
        mu_ref, sigma_ref = load_mu_and_sigma_reference(stats_path, pool_type=pool_type)
        queue = FeatureQueue(size=args.queue_size, feat_dim=feat_dim,
                             online_accum=args.fd_online_accum,
                             ema_beta=args.fd_ema_beta).cuda()
        short = model_short_name(name)
        sigma_ref_sqrt = precompute_sigma_ref_sqrt(sigma_ref) if args.fd_eigvalsh else None
        judges.append({
            "name": short, "model": repr_model, "feat_dim": feat_dim,
            "pool_type": pool_type, "mu_ref": mu_ref, "sigma_ref": sigma_ref,
            "sigma_ref_sqrt": sigma_ref_sqrt, "queue": queue, "weight": weight,
        })
        judge_model_names.append(name)
        logger.info("[FD] Repr '%s' (%s): feat_dim=%d, weight=%s, pool=%s, stats=%s",
                    short, name, feat_dim, weight, pool_type, stats_path)

    vlm_judge = setup_vlm_delta(judges, judge_model_names, args)
    heads = vlm_judge["vlm_heads"]
    buffer = vlm_judge["vlm_buffer"]

    lora = vlm_judge.get("vlm_lora")

    fd_restored = (extra is not None and "fd_queue_states" in extra
                   and load_fd_queue_states(judges, extra["fd_queue_states"]))
    vlm_restored = False
    if extra is not None and extra.get("vlm_delta_state") is not None:
        state = extra["vlm_delta_state"]
        saved_identity = state.get("p_identity", {})
        if saved_identity and saved_identity != vlm_judge["vlm_p_identity"]:
            diffs = {k: (saved_identity.get(k), vlm_judge["vlm_p_identity"].get(k))
                     for k in vlm_judge["vlm_p_identity"]
                     if saved_identity.get(k) != vlm_judge["vlm_p_identity"].get(k)}
            raise ValueError(
                "refusing to resume: the checkpoint was written with a different "
                f"p head / VLM configuration. Differences (saved, current): {diffs}")
        if (state.get("q_lora") is not None) != (lora is not None):
            raise ValueError("Cannot resume across linear-only and LoRA q modes; start a new run with --load_from")
        if lora is not None:
            lora.load_state_dict(state["q_lora"])
        heads.load_q_state_dict(state["q"])
        heads.cuda()
        vlm_judge["vlm_q_optimizer"] = (_build_q_optimizer(heads, args) if lora is None
                                        else build_lora_optimizer(lora, args))
        if state.get("q_optimizer") is not None:
            load_q_optimizer_state(vlm_judge["vlm_q_optimizer"], state["q_optimizer"])
        if buffer is not None and state.get("q_buffer") is not None:
            buffer.load_state_dict(state["q_buffer"])
        vlm_restored = True
        logger.info("[VLM-delta] Restored q student/teacher (%d q steps), optimizer "
                    "and replay buffer (%d entries) from checkpoint",
                    int(heads.q_train_steps.item()), 0 if buffer is None else buffer.size)

    def _collect(judge_name, feats, labels):
        if buffer is None or judge_name != vlm_judge["name"]:
            return
        buffer.push(feats.float(), vlm_judge["vlm_class_map"].to_local(labels))

    if fd_restored:
        logger.info("[FD] Restored all queue states from checkpoint — skipping queue fill")
        run_sanity_check(judges, args.queue_size, args=args)
        if buffer is not None and not vlm_restored and args.vlm_q_bootstrap > 0:
            logger.info("[VLM-delta] No saved q state — running a standalone "
                        "buffer bootstrap pass")
            bootstrap_vlm_buffer(vlm_judge, model_wo_ddp, args, tokenizer=tokenizer)
    else:
        logger.info("[FD] Filling %d feature queue(s) (%d entries each) ...",
                    len(judges), args.queue_size)
        fill_all_queues(judges, model_wo_ddp, args, tokenizer=tokenizer,
                        feature_collector=None if vlm_restored else _collect)
        run_sanity_check(judges, args.queue_size, args=args)
    if (buffer is not None and not vlm_restored and buffer.size == 0 and args.vlm_q_bootstrap > 0
            and vlm_judge.get("vlm_backend") == P_HEAD_BACKEND_QWEN):
        # The answer-state extractor is not one of the FD judges, so the queue
        # fill produced none of its features and the replay buffer is still
        # empty. Seed it explicitly -- q must not start on an empty buffer.
        logger.info("[VLM-delta] answer-state backend: seeding the q buffer with "
                    "a standalone pass over %d generated images per rank "
                    "(~%.0f min at 46 img/s)", args.vlm_q_bootstrap,
                    args.vlm_q_bootstrap / 46.0 / 60.0)
        bootstrap_vlm_buffer(vlm_judge, model_wo_ddp, args, tokenizer=tokenizer)
    if not vlm_restored and buffer is not None:
        logger.info("[VLM-delta] q replay buffer seeded: %s",
                    ", ".join(f"{k}={v:g}" for k, v in buffer.stats().items()))

    del extra
    torch.distributed.barrier()

    model.train()
    args.input_channels = model_wo_ddp.in_channels
    args.input_size = model_wo_ddp.input_size

    probe = None
    if args.cond_probe:
        probe = ProbeClassifier(device="cuda")
        logger.info("[Probe] held-out resnet50 IMAGENET1K_V2 logged as probe_top1/"
                    "top5/logp/rank at diag steps — never in the loss. It is the "
                    "honest arbiter: if vlm_p_top1 rises while probe_top1 stays at "
                    "chance, that is VLM exploitation, not conditioning.")

    # -- optional q warm-up on the bootstrap buffer --
    q_bootstrap_report = None
    if not vlm_restored and args.vlm_q_bootstrap_updates > 0:
        logger.warning(
            "[VLM-delta] Pre-training q for %d updates on the bootstrap buffer: "
            "q will NOT equal p at step 0 and the term starts with a non-zero "
            "gradient. This changes the experiment; default is 0.",
            args.vlm_q_bootstrap_updates)
        q_bootstrap_report = _q_bootstrap_train(vlm_judge, args)

    # -- initial diagnostic, before any generator update --
    if not args.vlm_skip_init_diag and args.current_step == 0:
        report = initial_generated_diagnostics(vlm_judge, model_wo_ddp, args,
                                               tokenizer=tokenizer, probe=probe)
        report["step"] = int(args.current_step)
        report["p_head"] = vlm_judge["vlm_p_ckpt_meta"]
        if q_bootstrap_report is not None:
            report["q_bootstrap"] = q_bootstrap_report
        if torch.distributed.get_rank() == 0:
            path = os.path.join(args.log_dir, "vlm_init_diagnostics.json")
            with open(path, "w") as f:
                json.dump(report, f, indent=1)
            print_initial_diagnostics(report)
            logger.info("[VLM-delta] wrote %s", path)
        eq = report["init_equality"]
        worst = eq["init_max_abs_logdiff_q_student_vs_p"]
        if heads.use_ema:
            worst = max(worst, eq["init_max_abs_logdiff_q_teacher_vs_p"])
        if not vlm_restored and args.vlm_q_bootstrap_updates == 0 and worst > 1e-4:
            raise RuntimeError(
                f"q was expected to equal p at initialisation but max |log q - log p| "
                f"= {worst:.3e}")
        model_wo_ddp.train()

    sampling_args = {"t_min": args.interval_min, "t_max": args.interval_max,
                     "cfg": args.cfg, "num_steps": args.num_sampling_steps}
    fd_train_step = get_fd_train_step(model_wo_ddp, judges, sampling_args, args,
                                      tokenizer=tokenizer, vlm_judge=vlm_judge,
                                      probe=probe)

    logger.info("training from step %s -> %s (%s -> %s epochs)",
                f"{args.current_step:,}", f"{args.total_steps:,}",
                args.start_epoch, args.epochs)
    # The generator gradient is all-reduced with AVG while every rank computes
    # the loss over the *globally gathered* batch, so the applied gradient is
    # 1/world_size of the true full-batch gradient and the EFFECTIVE learning
    # rate scales as lr/world_size.  Verified numerically at W=1,2,3.  Print it
    # so a run resumed at a different world size cannot silently change step
    # size mid-trajectory.
    logger.info(
        "[LR] lr=%.3g over world_size=%d -> effective lr=%.4g per optimizer step "
        "(global batch %d = %d/rank). A resume at a different world size MUST "
        "rescale --lr to keep this constant.",
        args.lr, args.world_size, args.lr / max(1, args.world_size),
        args.batch_size * args.world_size, args.batch_size)

    global_bsz = args.batch_size * args.world_size
    ckpt_saver = AsyncCheckpointSaver()
    session_start = time.time()
    step_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ckpt_target_minutes = args.ckpt_target_minutes
    ckpt_measure_interval = 1000
    ckpt_timer_start = time.perf_counter()
    ckpt_timer_step = args.current_step
    last_ckpt_step = args.current_step
    last_ckpt_time = time.perf_counter()

    metric_file = os.path.join(args.log_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metric_file, prefetch=True)
    takeoff_gate_status = restored_gate_status
    per_class = PerClassAccumulator(heads.num_classes,
                                    vlm_judge["vlm_class_map"].class_ids)
    nonfinite_streak = 0
    for name, window, fmt in [
        ("lr", 1, "{value:.6f}"),
        ("samples/s/device", args.print_freq, "{avg:.2f}"),
        ("samples/s", args.print_freq, "{avg:.2f}"),
        ("samples_seen(M)", args.print_freq, "{value:.2f}"),
        ("device_mem(GB)", args.print_freq, "{value:.2f}"),
    ]:
        metric_logger.add_meter(name, SmoothedValue(window, fmt))
    # MetricLogger writes the *median of the meter's window*, and these are only
    # updated once per diagnostic step, so the default 20-wide window would turn
    # them into a ~20*print_freq-step trailing median. The calibration quartet
    # must be read at the step it was measured, the tail meters must show a spike
    # rather than hide it in a median, and the drift meters are monotone.
    instant_meters = list(INSTANT_METERS) + list(
        BACKEND_INSTANT_METERS.get(vlm_judge.get("vlm_backend"), ()))
    if not heads.use_ema:
        instant_meters = [name for name in instant_meters if not name.startswith("q_teacher_")]
    for name in instant_meters:
        metric_logger.add_meter(name, SmoothedValue(1, "{value:.6f}"))
    first_step_of_process = args.current_step

    def _infinite():
        while True:
            yield None

    for step, _ in metric_logger.log_every(
        _infinite(), args.print_freq, header="Train:",
        start_iteration=args.current_step, n_iterations=args.total_steps,
    ):
        model.train()
        adjust_learning_rate(optimizer, step, args)

        scale_eff = vlm_scale_schedule(step, args)
        scale_t = torch.as_tensor(scale_eff, device="cuda", dtype=torch.float32)
        gate_due = step == args.vlm_takeoff_gate_step
        diag = (step % args.print_freq == 0) or gate_due
        (loss, loss_dict, new_feats, z_vlm, y_local,
         sampled_detached, y_global) = fd_train_step(vlm_scale=scale_t, diag=diag)
        loss_dict["vlm_scale_eff"] = scale_eff

        grad_norm = (torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                     if args.grad_clip > 0.0 else get_grad_norm(model.parameters()))
        loss_dict["generator_grad_norm"] = float(grad_norm)

        if torch.isfinite(grad_norm):
            nonfinite_streak = 0
            optimizer.step()
            ema_model.step(model)
        else:
            nonfinite_streak += 1
            logger.error("[step %d] NaN/Inf generator grad_norm — skipping optimizer "
                         "& EMA update (streak %d/%d)", step, nonfinite_streak,
                         args.vlm_max_nonfinite_steps)
            if nonfinite_streak >= args.vlm_max_nonfinite_steps:
                raise RuntimeError(
                    f"[VLM-delta] {nonfinite_streak} consecutive non-finite generator "
                    f"gradients at step {step}; aborting loudly rather than "
                    f"continuing a corrupted run")
        optimizer.zero_grad(set_to_none=True)

        for i, judge in enumerate(judges):
            judge["queue"].enqueue(new_feats[i])

        # -- q student update; optional EMA for the legacy comparison --
        if lora is None:
            loss_dict.update(q_update_step(vlm_judge, z_vlm, y_local, args,
                                           collect_metrics=diag))
        else:
            n_local = sampled_detached.shape[0]
            local_labels = y_local[args.rank * n_local:(args.rank + 1) * n_local]
            loss_dict.update(q_lora_update_step(
                lora, vlm_judge["vlm_q_optimizer"], sampled_detached, local_labels,
                args, collect_metrics=diag))

        torch.cuda.synchronize()

        args.current_step = step + 1
        args.samples_seen += global_bsz
        step_time = time.perf_counter() - step_start
        step_start = time.perf_counter()

        loss_value = all_reduce_mean(loss.item())
        loss_dict = {k: all_reduce_mean(v) for k, v in loss_dict.items()}
        sps = args.batch_size / step_time if step_time > 0 else 0.0
        mem_gb = (torch.cuda.max_memory_reserved() / (1024 ** 3)
                  if torch.cuda.is_available() else 0.0)
        loss_dict["samples_per_class"] = args.samples_seen / max(1, heads.num_classes)

        metric_logger.update(
            loss=loss_value, grad_norm=grad_norm,
            lr=optimizer.param_groups[0]["lr"],
            **{"samples/s/device": sps, "samples/s": sps * args.world_size,
               "samples_seen(M)": args.samples_seen / 1e6, "device_mem(GB)": mem_gb},
            **loss_dict)

        if step == first_step_of_process:
            # A pre-registered meter that the step never produces keeps an empty
            # deque, and MetricLogger.__str__ calls max() on it at the very first
            # print -- killing the run at step 0, or immediately after a resume.
            # Drop any such meter instead of crashing; warn, because it means a
            # metric someone expected is missing.
            stale = [n for n in instant_meters
                     if not metric_logger.meters[n].deque]
            for name in stale:
                del metric_logger.meters[name]
            if stale:
                logger.warning(
                    "[metrics] %d registered meter(s) were not produced by the "
                    "training step and have been dropped so logging cannot crash "
                    "on an empty window: %s", len(stale), ", ".join(stale))

        # -- per-class accumulation and periodic dump --
        if diag and sampled_detached is not None:
            with torch.no_grad():
                probe_correct = None
                if probe is not None:
                    # local-batch correctness, gathered so it lines up with the
                    # globally gathered y_local / z_vlm the heads are read on
                    # Per-rank scored count, which is the batch size unless
                    # --vlm_samples_per_step scored only a subset.
                    per_rank = y_global.numel() // max(1, args.world_size)
                    local_y = y_global[args.rank * per_rank:
                                       (args.rank + 1) * per_rank]
                    probe_correct = all_gather_plain(
                        probe_per_sample_correct(probe, sampled_detached, local_y))
                per_class.update(
                    y_local,
                    logp_all=heads.p_log_probs(z_vlm),
                    logq_all=vlm_judge["vlm_scored_logq"],
                    probe_correct=probe_correct)
        if (args.vlm_per_class_every > 0 and step > 0
                and step % args.vlm_per_class_every == 0
                and per_class.steps > 0 and torch.distributed.get_rank() == 0):
            per_class.dump(
                os.path.join(args.log_dir, f"vlm_per_class_step_{step:05d}.json"),
                step=step,
                extra={"samples_per_class": loss_dict.get("samples_per_class"),
                       "class_ids": vlm_judge["vlm_class_map"].class_ids})
            per_class.reset()
        elif (args.vlm_per_class_every > 0 and step > 0
              and step % args.vlm_per_class_every == 0):
            per_class.reset()

        gate_abort = False
        if gate_due:
            gate_abort, failures = evaluate_vlm_takeoff_gate(
                loss_dict,
                cond_delta_min=args.vlm_takeoff_cond_delta_min,
                rank_ratio_max=args.vlm_takeoff_rank_ratio_max,
                logic=args.vlm_takeoff_logic,
                check_cond_delta=args.vlm_takeoff_cond_delta_min > 0.0,
                check_rank=not args.vlm_takeoff_no_rank)
            takeoff_gate_status = {
                "evaluated": True, "aborted": bool(gate_abort), "step": int(step),
                "logic": args.vlm_takeoff_logic, "failures": failures,
                "cond_delta": loss_dict.get("cond_delta"),
                "cond_delta_required": args.vlm_takeoff_cond_delta_min,
                "p_rank_ratio": loss_dict.get("vlm_p_target_rank_ratio_to_chance"),
                "p_rank_ratio_required_below": args.vlm_takeoff_rank_ratio_max,
            }
            metric_logger.update(vlm_takeoff_gate_abort=int(gate_abort))
            logger.info("[VLM takeoff gate] %s at step %d: %s",
                        "ABORT" if gate_abort else "PASS", step,
                        json.dumps(takeoff_gate_status, default=str))

        if step % args.print_freq == 0 and wandb_logger:
            elapsed = time.time() - session_start + args.last_elapsed_time
            remaining = args.total_steps - args.current_step
            eta = elapsed / args.current_step * remaining if args.current_step > 0 else 0.0
            wandb_logger.update({
                "train/loss": loss_value,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": grad_norm,
                "train/samples_seen_M": args.samples_seen / 1e6,
                "perf/samples_per_sec": sps * args.world_size,
                "perf/eta_real_hours": eta / 3600,
                **{f"train/{k}": v for k, v in loss_dict.items()},
            }, step=args.current_step)

        steps_since_timer = args.current_step - ckpt_timer_step
        if steps_since_timer >= ckpt_measure_interval:
            elapsed_minutes = (time.perf_counter() - ckpt_timer_start) / 60.0
            minutes_per_step = elapsed_minutes / steps_since_timer
            new_save_every = max(100, round(ckpt_target_minutes / minutes_per_step / 100) * 100)
            if new_save_every != args.save_every:
                logger.info("adjusting save_every: %d -> %d (%.1f min/1k steps)",
                            args.save_every, new_save_every, minutes_per_step * 1000)
                args.save_every = new_save_every
            ckpt_timer_start = time.perf_counter()
            ckpt_timer_step = args.current_step

        def _save(saver=ckpt_saver):
            elapsed = time.time() - session_start + args.last_elapsed_time
            fd_extra = {"fd_queue_states": save_fd_queue_states(judges)} if judges else {}
            fd_extra["vlm_delta_state"] = {
                "q": heads.q_state_dict(),
                "q_lora": None if lora is None else lora.state_dict(),
                "q_optimizer": vlm_judge["vlm_q_optimizer"].state_dict(),
                "q_buffer": (buffer.state_dict() if buffer is not None and args.vlm_q_checkpoint_buffer
                             else None),
                "p_identity": vlm_judge["vlm_p_identity"],
                "p_head_path": vlm_judge["vlm_p_ckpt_meta"]["path"],
                "weight_schedule": {
                    "weight": args.vlm_delta_weight,
                    "warmup_steps": args.vlm_delta_warmup_steps,
                    "ramp_steps": args.vlm_delta_ramp_steps,
                    "scale_eff": scale_eff,
                },
            }
            if args.vlm_takeoff_gate_step >= 0:
                fd_extra["vlm_takeoff_gate_status"] = takeoff_gate_status
            save_checkpoint(args, step, model_wo_ddp, optimizer, ema_model, elapsed,
                            saver=saver, extra=fd_extra)
            torch.distributed.barrier()

        if gate_abort:
            metric_logger.dump_in_output_file(step, step_time, 0.0)
            logger.error("[VLM takeoff gate] Saving gated checkpoint at step %d and "
                         "exiting with status 4", step)
            ckpt_saver.wait()
            _save(saver=None)
            return 4

        # A slow FP32 VLM run can take hours to reach the first 1,000-step
        # cadence estimate. Honor the wall-clock target from the first step.
        # Rank zero decides so all ranks enter checkpoint collectives together.
        save_due = broadcast_bool(
            (ckpt_target_minutes > 0
             and time.perf_counter() - last_ckpt_time >= ckpt_target_minutes * 60)
            or args.current_step - last_ckpt_step >= args.save_every
            or args.current_step == args.total_steps)
        if save_due:
            _save()
            last_ckpt_step = args.current_step
            last_ckpt_time = time.perf_counter()
        if args.milestone_every > 0 and step > 0 and step % args.milestone_every == 0:
            _save()

        if preempt_requested():
            logger.info("Preemption at step %d: saving checkpoint ...", args.current_step)
            ckpt_saver.wait()
            _save(saver=None)
            return 0

        if args.vis_every > 0 and args.current_step % args.vis_every == 0:
            visualize(args, model_wo_ddp, ema_model, args.current_step, rng=rng,
                      tokenizer=tokenizer)
            model_wo_ddp.train()

        if args.eval_every > 0 and args.online_eval and args.current_step % args.eval_every == 0:
            torch.cuda.empty_cache()
            evaluate_all_emas(args, model_wo_ddp, ema_model, fid_evaluator, tokenizer,
                              step=args.current_step, wandb_logger=wandb_logger,
                              cfg=args.cfg, num_images=args.num_images_for_eval_and_search)
            model_wo_ddp.train()

    ckpt_saver.wait()
    total = time.time() - session_start + args.last_elapsed_time
    metric_logger.synchronize_between_processes()
    logger.info("averaged stats: %s", metric_logger)
    logger.info("Training complete. Total time: %s on %d devices",
                datetime.timedelta(seconds=int(total)), args.world_size)
    torch.cuda.empty_cache()
    return 0


@torch.no_grad()
def bootstrap_vlm_buffer(vlm_judge, model, args, tokenizer=None):
    """Seed the q replay buffer from a standalone generation pass.

    Only needed when the FD queues were restored from a checkpoint that predates
    the VLM-delta state; the normal path folds this into the queue fill.
    """
    buffer = vlm_judge["vlm_buffer"]
    class_map = vlm_judge["vlm_class_map"]
    ids = torch.tensor(args.train_class_ids if args.train_class_ids is not None
                       else list(range(args.num_classes)),
                       dtype=torch.long, device="cuda")
    model.eval()
    filled = 0
    while filled < args.vlm_q_bootstrap:
        bsz = min(args.fd_queue_fill_bsz, args.vlm_q_bootstrap - filled)
        y = ids[torch.randint(0, ids.numel(), (bsz,), device="cuda")]
        imgs = model.generate(bsz, y, cfg=args.cfg, args=args, verbose=False)
        imgs = tokenizer.detokenize(imgs) if tokenizer is not None else imgs * 0.5 + 0.5
        z = diff_all_gather(vlm_features_no_grad(vlm_judge, imgs, args)).detach()
        buffer.push(z.float(), class_map.to_local(all_gather_plain(y)))
        filled += bsz
    logger.info("[VLM-delta] Bootstrap done: %s",
                ", ".join(f"{k}={v:g}" for k, v in buffer.stats().items()))


def _q_bootstrap_train(vlm_judge, args):
    """Fit q to the CURRENT generator's samples instead of leaving it a copy of p.

    The default design clones q from p so that ``log q - log p == 0`` at step 0
    and the term starts with exactly zero gradient.  That is a convenient
    initialisation, not a correct one: at step 0 the generator IS the base
    model, so the true ``q(c|z)`` is whatever a probe fitted on the base model's
    own samples says -- which for a de-conditioned generator is near-uniform,
    not p's confident real-image posterior.  Running this makes the term start
    at its true non-zero value.

    Two consequences, both deliberate:

    * ``log q - log p`` is NOT zero at step 0, so the initial-equality assertion
      is skipped and ``grad_ratio_vlm_fd`` is readable from the first step
      instead of having to grow with q's drift.
    * With --vlm_q_use_ema, the teacher is not synced to the student at the end.
      Historical EMA-run measurements on the replay buffer (20,000 base-model generations,
      lr 1e-3, wd 3.0, held-out 20%):

          updates   student CE / entropy   EMA teacher CE / entropy
             1000     5.97 / 3.60             5.70 / 3.62
             2000     6.03 / 3.64             4.81 / 4.39
             5000     6.52 / 3.43             4.70 / 4.51
            20000     6.50 / 3.54             4.70 / 4.52
                                              (uniform = 4.6052 / 4.6052)

      The student NEVER converges: with no linearly-decodable label signal in a
      de-conditioned generator's samples its CE gradient is unbiased noise, so it
      stays permanently overconfident-and-wrong at CE ~6.0-6.5.  The EMA is what
      averages that noise away, and it is the only reason q_teacher ends up at
      the near-uniform posterior that IS the correct q here.  Syncing the teacher
      to the student throws that away and hands the generator a single noisy
      draw -- which is exactly what it looked like when it was tried: q_teacher
      read CE 6.81 / entropy 3.37 on fresh generations instead of ~4.70 / ~4.52.
    """
    if vlm_judge["vlm_heads"].use_ema and 0 < args.vlm_q_bootstrap_updates < 2000:
        logger.warning(
            "[VLM-delta] --vlm_q_bootstrap_updates=%d is below the ~2000 the EMA "
            "teacher needs to average out the student's noise (beta=%.4f). The "
            "teacher will still be part-way between p and q, and q_teacher's CE "
            "on fresh samples will read well above uniform. Use >=5000.",
            args.vlm_q_bootstrap_updates, args.vlm_q_ema_beta)
    heads = vlm_judge["vlm_heads"]
    buffer = vlm_judge["vlm_buffer"]
    optimizer = vlm_judge["vlm_q_optimizer"]
    trajectory = []
    active_q = "q_teacher" if heads.use_ema else "q_student"
    done = 0
    for i in range(args.vlm_q_bootstrap_updates):
        batch = buffer.sample(args.vlm_q_batch_size)
        if batch is None:
            logger.warning("[VLM-delta] q bootstrap stopped at update %d: the "
                           "replay buffer is empty", i)
            break
        z_batch, y_batch = batch
        loss = F.cross_entropy(heads.q_student.logits(z_batch, heads.temperature), y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.vlm_q_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(heads.q_student.linear.parameters(),
                                           args.vlm_q_grad_clip)
        if torch.distributed.is_initialized():
            for param in heads.q_student.linear.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(
                        param.grad, op=torch.distributed.ReduceOp.AVG)
        optimizer.step()
        heads.q_train_steps.add_(1)
        heads.ema_update()
        done = i + 1
        if i % 500 == 0 or i == args.vlm_q_bootstrap_updates - 1:
            drift = heads.drift_metrics()
            with torch.no_grad():
                t_lp = heads.q_generator_log_probs(z_batch)
                t_ent = float(-(t_lp.exp() * t_lp).sum(-1).mean())
                t_ce = float(F.nll_loss(t_lp, y_batch))
            trajectory.append({
                "update": i, "student_batch_ce": float(loss.detach()),
                "generator_q_batch_ce": t_ce, "generator_q_entropy": t_ent,
                "q_student_drift_from_p": drift["q_student_weight_delta_rel"],
                "q_generator_drift_from_p": drift[f"{active_q}_weight_delta_rel"]})
            logger.info(
                "[VLM-delta] q bootstrap %5d/%d  student_ce=%.4f  generator_q_ce=%.4f  "
                "generator_q_entropy=%.4f (uniform %.4f)  drift student=%.3f active=%.3f",
                i, args.vlm_q_bootstrap_updates, float(loss.detach()), t_ce, t_ent,
                math.log(heads.num_classes), drift["q_student_weight_delta_rel"],
                drift[f"{active_q}_weight_delta_rel"])

    after = heads.drift_metrics()
    logger.info(
        "[VLM-delta] q bootstrap done: %d updates. Generator reads %s. "
        "Student drift %.4f, active q drift %.4f (relative ||W||). "
        "Uniform CE for %d classes is %.4f.",
        done, active_q, after["q_student_weight_delta_rel"], after[f"{active_q}_weight_delta_rel"],
        heads.num_classes, math.log(heads.num_classes))
    return {"updates": done, "trajectory": trajectory,
            "uniform_ce": math.log(heads.num_classes),
            "q_student_drift_from_p": after["q_student_weight_delta_rel"],
            "q_generator_drift_from_p": after[f"{active_q}_weight_delta_rel"]}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser(
        "FD fine-tuning with the VLM sampled-label log q - log p term", add_help=False)

    # training
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--steps_per_epoch", default=1250, type=int)
    parser.add_argument("--batch_size", default=32, type=int, help="batch size per GPU")
    parser.add_argument("--generator_microbatch", default=0, type=int,
                        help="images per generator forward/recomputation; 0 uses the full "
                             "local batch; combine with --grad_checkpointing to bound VRAM")
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--same_noise", action="store_true")

    # model architecture
    parser.add_argument("--model", default="JiT_B", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--label_drop_prob", default=0.1, type=float)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--class_tokens", type=int, default=8)
    parser.add_argument("--time_tokens", type=int, default=4)
    parser.add_argument("--guidance_tokens", type=int, default=4)
    parser.add_argument("--interval_tokens", type=int, default=2)
    parser.add_argument("--norm_eps", type=float, default=0.01)
    parser.add_argument("--norm_p", type=float, default=1.0)
    parser.add_argument("--rope_2d", action="store_true")
    parser.add_argument("--learned_pe", action="store_true")
    parser.add_argument("--disable_v_head", action="store_true")
    parser.add_argument("--t_eps", type=float, default=5e-2)

    parser.add_argument("--rf_dropout", type=float, default=0.0)
    parser.add_argument("--rf_grad_checkpoint", action="store_true", default=True)
    parser.add_argument("--no_rf_grad_checkpoint", action="store_false",
                        dest="rf_grad_checkpoint")

    # tokenizer
    parser.add_argument("--tokenizer", default=None, type=str)
    parser.add_argument("--token_channels", default=3, type=int)
    parser.add_argument("--tokenizer_patch_size", default=1, type=int)

    # optimization
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_sched", type=str, default="constant",
                        choices=["constant", "cosine"])
    parser.add_argument("--warmup_rate", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=-1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--use_muon", action="store_true")
    parser.add_argument("--muon_lr", type=float, default=1e-3)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_type", default="edm", type=str, choices=["const", "edm"])
    parser.add_argument("--ema_rates", default=[0.9999, 0.9996], type=float, nargs="+")
    parser.add_argument("--ema_halflife_kimg", default=[250, 500, 1000, 2000],
                        type=float, nargs="+")
    parser.add_argument("--eval_ema_labels", default=None, type=str, nargs="+")
    parser.add_argument("--grad_checkpointing", action="store_true")

    # diffusion / flow-matching
    parser.add_argument("--P_mean", type=float, default=0.8)
    parser.add_argument("--P_std", type=float, default=0.8)
    parser.add_argument("--legacy_time_convention", action="store_true")
    parser.add_argument("--tr_uniform", action="store_true")
    parser.add_argument("--ratio_r_neq_t", type=float, default=0.5)
    parser.add_argument("--cfg_beta", type=float, default=1.0)
    parser.add_argument("--cfg_omega_max", type=float, default=7.0)
    parser.add_argument("--aux_head_depth", type=int, default=8)
    parser.add_argument("--loss_type", type=str, default="v", choices=["v", "x"])
    parser.add_argument("--aux_pred_type", type=str, default="v", choices=["v", "x"])
    parser.add_argument("--perceptual_threshold", type=float, default=0.8)
    parser.add_argument("--perceptual_loss_on_aux", action="store_true")

    # sampling & generation
    parser.add_argument("--sampling_method", type=str, default="heun",
                        choices=["euler", "heun"])
    parser.add_argument("--num_sampling_steps", type=int, default=1)
    parser.add_argument("--cfg", default=3.0, type=float)
    parser.add_argument("--cfg_list", type=float, nargs="+",
                        default=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 10.0])
    parser.add_argument("--interval_min", type=float, default=0.1)
    parser.add_argument("--interval_max", type=float, default=1.0)
    parser.add_argument("--vis_steps", default=[1], type=int, nargs="+")

    # data
    parser.add_argument("--data_path", default="./data/imagenet", type=str)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--train_class_ids", default=None, type=int, nargs="+",
                        help="label subset sampled during training; must match the "
                             "class set the p head was trained on")
    parser.add_argument("--class_of_interest",
                        default=[207, 360, 387, 974, 88, 979, 417, 279],
                        type=int, nargs="+")
    parser.add_argument("--force_class_of_interest", action="store_true")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # checkpointing
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--keep_n_ckpts", default=3, type=int)
    parser.add_argument("--ckpt_target_minutes", type=float, default=10.0,
                        help="target wall-clock minutes between checkpoints; the "
                             "save cadence is re-derived from measured step time. "
                             "Lower it on preemptible GPUs -- it bounds how much "
                             "work an eviction can destroy")
    parser.add_argument("--milestone_interval", default=20, type=int)

    # evaluation
    parser.add_argument("--online_eval", action="store_true")
    parser.add_argument("--num_images_for_eval_and_search", default=10000, type=int)
    parser.add_argument("--num_images", default=50000, type=int)
    parser.add_argument("--eval_bsz", type=int, default=64)
    parser.add_argument("--fid_stats_path", type=str,
                        default="data/fid_stats/guided_diffusion_stats.npz")
    parser.add_argument("--keep_eval_folder", action="store_true")
    parser.add_argument("--save_eval_images", action="store_true")
    parser.add_argument("--cfg_min", default=1.0, type=float)
    parser.add_argument("--cfg_max", default=25.0, type=float)
    parser.add_argument("--overwrite_cache", action="store_true")

    # FD fine-tuning
    parser.add_argument("--queue_size", type=int, default=50000)
    parser.add_argument("--fd_fid_norm_eps", type=float, default=0.01)
    parser.add_argument("--fd_queue_fill_bsz", type=int, default=256)
    parser.add_argument("--fd_feature_microbatch", type=int, default=0,
                        help="FD feature images per forward; 0 uses the full local batch")
    parser.add_argument("--fd_feature_checkpoint", action="store_true",
                        help="recompute FD features in backward to save activation memory")
    parser.add_argument("--fd_repr_models", type=str, nargs="+", default=["inception"])
    parser.add_argument("--fd_repr_stats_paths", type=str, nargs="+", default=None)
    parser.add_argument("--fd_repr_weights", type=float, nargs="+", default=None)
    parser.add_argument("--fd_repr_pool_types", type=str, nargs="+", default=None)
    parser.add_argument("--fd_target_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--fd_online_accum", action="store_true")
    parser.add_argument("--fd_eigvalsh", action="store_true")
    parser.add_argument("--fd_ema_beta", type=float, default=0.0, metavar="BETA")

    # -- VLM posterior-delta term --
    group = parser.add_argument_group("VLM posterior-delta conditional term")
    group.add_argument("--vlm_p_head", type=str, default=None,
                       help="frozen real-data p-head checkpoint from train_vlm_p_head.py")
    group.add_argument("--vlm_judge", type=str, default=None,
                       help="short name of the FD judge carrying the VLM (e.g. 'siglip'); "
                            "its features are reused so the term costs no extra forward")
    group.add_argument("--vlm_delta_weight", type=float, default=0.0,
                       help="lambda on E[log q(c|z) - log p(c|z)]. Calibrate on "
                            "grad_ratio_vlm_fd (target sustained 0.22-0.30), never on "
                            "the loss value. NOTE the term is exactly 0 at init because "
                            "q == p, so the calibration window must be long enough for "
                            "q to have drifted")
    group.add_argument("--vlm_head_temperature", type=float, default=None,
                       help="shared logit temperature for BOTH heads (default: the one "
                            "fitted on real validation data and stored in the p-head "
                            "checkpoint). p and q are never tuned separately")
    group.add_argument("--vlm_delta_warmup_steps", type=int, default=0)
    group.add_argument("--vlm_delta_ramp_steps", type=int, default=500)
    group.add_argument("--vlm_delta_clamp", type=float, default=0.0,
                       help="OPT-IN emergency per-sample clamp on log q - log p. "
                            "0 disables it (the default: do not silently change the "
                            "objective). When active, vlm_delta_clamp_frac says how "
                            "much of the experiment it changed")
    group.add_argument("--vlm_max_nonfinite_steps", type=int, default=5,
                       help="abort after this many consecutive non-finite generator "
                            "gradients")

    group.add_argument("--vlm_q_lr", type=float, default=1e-4)
    group.add_argument("--vlm_q_lora", action="store_true",
                       help="Qwen only: train student LoRA plus q head on fresh detached images")
    group.add_argument("--vlm_q_lora_rank", type=int, default=8)
    group.add_argument("--vlm_q_lora_alpha", type=float, default=16.0)
    group.add_argument("--vlm_q_lora_scope", choices=("language", "vision", "both"), default="both")
    group.add_argument("--vlm_q_lora_targets", nargs="+",
                       default=["q_proj", "k_proj", "v_proj", "o_proj", "qkv", "proj"],
                       help="Linear-module leaf names; only vision / pre-answer decoder layers are eligible")
    group.add_argument("--vlm_q_lora_lr", type=float, default=1e-5)
    group.add_argument("--vlm_q_lora_weight_decay", type=float, default=0.01)
    group.add_argument("--vlm_vjp_loss_scale", type=float, default=1.0,
                       help="Qwen image-VJP backward scale; unscaled in FP32 before injection; leaves lambda unchanged")
    group.add_argument("--vlm_q_loss_scale", type=float, default=1.0,
                       help="LoRA q CE backward scale; unscaled before gradient averaging/clipping/optimizer")
    group.add_argument("--vlm_disable_tf32", action="store_true",
                       help="disable CUDA matmul and cuDNN TF32 process-wide for precision comparisons")
    group.add_argument("--vlm_q_optimizer", choices=("adamw", "sgd"), default="adamw")
    group.add_argument("--vlm_q_beta1", type=float, default=0.0,
                       help="q AdamW beta1, shared by head and LoRA parameter groups")
    group.add_argument("--vlm_q_beta2", type=float, default=0.999,
                       help="q AdamW beta2, shared by head and LoRA parameter groups")
    group.add_argument("--vlm_q_momentum", type=float, default=0.9)
    group.add_argument("--vlm_q_weight_decay", type=float, default=3.0,
                       help="AdamW decoupled weight decay on the q head. NOT "
                            "optional here: an unregularised online head on a "
                            "signal-free stream random-walks, so ||W_q - W_p|| -- "
                            "and with it the injected field and "
                            "grad_ratio_vlm_fd -- grows without bound and no "
                            "weight stays calibrated. Decay pulls W toward 0 (a "
                            "uniform posterior), which is the correct q at a "
                            "de-conditioned start, and makes the drift stationary")
    group.add_argument("--vlm_q_grad_clip", type=float, default=1.0)
    group.add_argument("--vlm_q_updates_per_step", type=int, default=1,
                       help="q_student CE updates per generator step")
    group.add_argument("--vlm_q_batch_size", type=int, default=0,
                       help="samples drawn from the replay buffer per q update. "
                            "0 (default) means the global batch, which makes the "
                            "per-sample reuse factor 1.0 -- above ~2x the head "
                            "memorises the buffer and q(c|z) becomes noise on the "
                            "fresh samples the generator loss evaluates it at")
    group.add_argument("--vlm_q_use_ema", action="store_true",
                       help="opt in to the old EMA q teacher; default uses current student directly")
    group.add_argument("--vlm_q_ema_beta", type=float, default=0.999,
                       help="q teacher EMA decay, only used with --vlm_q_use_ema")
    group.add_argument("--vlm_q_buffer_size", type=int, default=20000,
                       help="total replay-buffer capacity; split evenly per class")
    group.add_argument("--vlm_q_bootstrap", type=int, default=50000,
                       help="samples for a standalone buffer bootstrap when FD queues "
                            "are restored but VLM state is not")
    group.add_argument("--vlm_q_bootstrap_updates", type=int, default=0,
                       help="OPT-IN q pre-training on the bootstrap buffer before step "
                            "0. Non-zero breaks the q == p initialisation")
    group.add_argument("--vlm_q_checkpoint_buffer", action="store_true", default=True)
    group.add_argument("--vlm_q_no_checkpoint_buffer", action="store_false",
                       dest="vlm_q_checkpoint_buffer")
    group.add_argument("--vlm_q_recompute_features", action="store_true",
                       help="run a second frozen-VLM forward on detached images for the "
                            "q update instead of detaching the graph features "
                            "(numerically identical, strictly more expensive)")

    group.add_argument("--vlm_microbatch", type=int, default=12,
                       help="answer-state backend only: images per 7B forward. "
                            "Measured on A100-80GB: 12 -> 29 GB peak / ~18 img/s, "
                            "24 -> 41 GB / ~20 img/s, both alongside the FD judges")
    group.add_argument("--vlm_samples_per_step", type=int, default=0,
                       help="answer-state backend only: score a round-robin "
                            "subset of K images per rank per step instead of the "
                            "whole batch (0 = all). Unbiased, because generated "
                            "batch elements are exchangeable -- it trades "
                            "gradient variance for step time")
    group.add_argument("--vlm_dtype", type=str, default="bf16",
                       choices=["bf16", "fp16", "fp32"],
                       help="answer-state backend only: the frozen VLM's compute "
                            "dtype")
    group.add_argument("--vlm_attn_implementation", type=str, default="sdpa",
                       help="answer-state backend only")
    group.add_argument("--vlm_init_diag_samples", type=int, default=2048)
    group.add_argument("--vlm_skip_init_diag", action="store_true")
    group.add_argument("--vlm_per_class_every", type=int, default=2500,
                       help="dump vlm_per_class_step_XXXXX.json every N steps; 0 off")

    group.add_argument("--vlm_takeoff_gate_step", type=int, default=-1,
                       help="OPT-IN early takeoff check at this exact step; -1 disables")
    group.add_argument("--vlm_takeoff_cond_delta_min", type=float, default=0.10)
    group.add_argument("--vlm_takeoff_rank_ratio_max", type=float, default=0.90)
    group.add_argument("--vlm_takeoff_no_rank", action="store_true")
    group.add_argument("--vlm_takeoff_logic", choices=["any", "all"], default="any")

    parser.add_argument("--cond_probe", action="store_true",
                        help="log held-out torchvision resnet50 V2 top-1/top-5/logp/rank "
                             "at diag steps; never in the loss")

    # logging & tracking
    parser.add_argument("--output_dir", default="./work_dirs")
    parser.add_argument("--local_eval_dir", type=str, default=None)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--vis_freq", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--vis_only", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--last_elapsed_time", type=float, default=0.0)
    parser.add_argument("--current_step", type=int, default=0)
    parser.add_argument("--samples_seen", type=int, default=0)
    parser.add_argument("--project", default="JiT_uncond_vlm_delta", type=str)
    parser.add_argument("--entity", default=None, type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--disable_wandb", action="store_false", dest="enable_wandb")

    # system
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--dtype", default="bf16", type=str,
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compile", action="store_true",
                        help="unsupported here: the per-term image-gradient "
                             "diagnostics this experiment is calibrated on need "
                             "eager autograd")
    return parser


if __name__ == "__main__":
    parsed = get_args_parser().parse_args()
    if parsed.compile:
        raise SystemExit(
            "--compile is not supported by conditional_main_fd_vlm_delta.py: the "
            "grad_ratio_vlm_fd calibration needs per-term autograd through the "
            "frozen judge at every diagnostic step.")
    sys.exit(train_and_evaluate(parsed))
