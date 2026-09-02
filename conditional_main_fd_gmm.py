"""FD post-training that teaches classification with class-conditional GMMs.

This is the ``conditional_main_fd_ponly.py`` recipe with the source of
``log p(c|x)`` swapped: instead of a frozen CLIP / timm classifier ensemble
consuming its own forward+backward pass, the class posterior is read off a
Gaussian mixture fitted in the *judge's own representation space* -- the space
the FD term already computes features in.  That makes the conditional signal
nearly free, and it makes the second half of the Bayes decomposition available
for the first time:

    L = sum_j FD_j / (FD_j.detach() + eps)                     # marginal, unchanged
      + w * [ lambda_cls * E[-log p(c|x)]                      # class fidelity
            + lambda_ent * E[ log q(x|c) - log p(x|c) ] ]      # anti-collapse

``p`` is fitted offline on the dataset (``compute_class_stats.py``); ``q`` is an
EMA of the generator's own per-class statistics, maintained online.  See
``docs/gmm_posterior_loss.md`` for the derivation and the failure modes that
shaped the estimator.

Why the ``q`` term matters *here* specifically.  On an already-conditional model
the measured control showed FD post-training raising within-class diversity on
its own, so the term had nothing to fix.  Starting from a de-conditioned model
the situation reverses: ``lambda_cls`` is the *driver* that has to carve class
structure out of nothing, and its per-sample minimiser is a point.  The previous
classifier-ensemble run on this exact checkpoint reached 63.5% held-out top-1
and paid for it with FID 10.7 -> 21.5.  ``lambda_ent`` is the counterweight.

Two instruments come free with the ``q`` side and are the reason to run this even
at ``--fd_gmm_weight 0``:

* ``gmm_class_mean_spread`` -- between-class scatter of q over that of p.  On a
  de-conditioned model every class produces the same distribution, so it starts
  near **0** and targets **1.0**.  It moves continuously, unlike ``probe_top1``,
  which is quantised at zero for tens of thousands of steps.
* ``gmm_within_trace_ratio`` -- within-class scatter of q over that of p.  Note
  the interpretation is *inverted* relative to the conditional-model setting:
  with coincident class means the generator's "within-class" scatter is its full
  marginal scatter, so this starts near ``1 / (tr(Sigma_within^p)/k)`` (~2.3 for
  inception on ImageNet-20) and should *fall* to 1.0 as classes separate.  Below
  1.0 is collapse.

Usage: see ``scripts/run_jit_uncond_gmm_20class.sh``.
"""

import argparse
import datetime
import logging
import math
import os
import sys
import time
from collections import deque

import torch
import torch.distributed

from utils.builders import create_generation_model, create_tokenizer
from utils.checkpoint_util import AsyncCheckpointSaver, ckpt_resume, save_checkpoint
from utils.distributed_util import all_reduce_mean, preempt_requested, register_preempt_handler
from utils.eval_util import evaluate_all_emas
from utils.grad_util import get_grad_norm
from utils.logging_util import MetricLogger, SmoothedValue
from utils.optimizer_util import create_optimizer
from frechet_distance.evaluator import FDEvaluator
from frechet_distance.queue import FeatureQueue
from frechet_distance.gmm import ClassGMMReference, OnlineClassStats, gmm_posterior_loss
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


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.optimize_ddp = False

logger = logging.getLogger("FD_loss")


class _StoreExplicit(argparse.Action):
    """Store an option and remember that it was supplied on the command line.

    Argparse's mutually-exclusive-group bookkeeping checks whether the parsed
    value differs from the action default.  ``--option self`` is nevertheless
    explicit even when ``self`` is also the default, so use a private sentinel
    as the action default and resolve it back to ``self`` after parsing.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"_{self.dest}_explicit", True)


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

    invalid = [class_id for class_id in class_ids
               if class_id < 0 or class_id >= num_classes]
    if invalid:
        raise ValueError(
            f"--train_class_ids contains IDs outside [0, {num_classes}): {invalid}"
        )
    return class_ids


def gmm_scale_schedule(step, args):
    """Effective GMM weight at ``step``: 0 during warmup, then a linear ramp.

    Gating the conditional term until the FD queue is warm and the generator is
    already producing real-looking images is what stops the early collapse the
    classifier-ensemble runs hit -- a conditional gradient applied to
    unrecognisable samples steers toward whatever the classifier happens to like
    off-manifold.
    """
    weight = args.fd_gmm_weight
    warmup = args.fd_gmm_warmup_steps
    ramp = args.fd_gmm_ramp_steps
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return weight
    return weight * min(1.0, (step - warmup) / float(ramp))


def evaluate_gmm_takeoff_gate(metrics, spread_multiplier=2.0,
                              cosine_threshold=0.0, logic="any",
                              check_spread=True, check_cosine=True):
    """Evaluate the opt-in early takeoff gate from already-reduced metrics.

    ``logic="any"`` implements the conservative scientific gate: abort when
    *either* the class-mean spread has not cleared its estimator noise floor or
    the conditioning gradient has not become an independent (negative-cosine)
    force. ``logic="all"`` aborts only when every enabled criterion fails.

    Missing and non-finite values fail closed.  Boundaries follow the prose
    exactly: spread equal to the requested multiple passes, while cosine equal
    to the threshold fails (with the default threshold 0, it must be negative).
    """
    if logic not in ("any", "all"):
        raise ValueError(f"takeoff gate logic must be 'any' or 'all', got {logic!r}")
    if not check_spread and not check_cosine:
        raise ValueError("takeoff gate must enable at least one criterion")

    failures = {}
    if check_spread:
        value = metrics.get("gmm_class_mean_spread_to_noise")
        failures["spread"] = (
            value is None or not math.isfinite(float(value))
            or float(value) < float(spread_multiplier)
        )
    if check_cosine:
        value = metrics.get("cos_update_fd_p")
        failures["cosine"] = (
            value is None or not math.isfinite(float(value))
            or float(value) >= float(cosine_threshold)
        )

    reducer = any if logic == "any" else all
    return reducer(failures.values()), failures


def validate_gmm_takeoff_gate_args(args):
    """Reject configurations in which the requested gate cannot be evaluated."""
    step = args.fd_gmm_takeoff_gate_step
    if step < 0:
        return
    if not args.fd_gmm:
        raise ValueError("--fd_gmm_takeoff_gate_step requires --fd_gmm")
    if args.fd_gmm_takeoff_spread_mult < 0.0:
        raise ValueError("--fd_gmm_takeoff_spread_mult must be >= 0")
    if args.fd_gmm_takeoff_spread_mult == 0.0 and args.fd_gmm_takeoff_no_cos:
        raise ValueError("takeoff gate has no enabled criteria")
    if args.fd_gmm_takeoff_cos_window <= 0:
        raise ValueError("--fd_gmm_takeoff_cos_window must be positive")
    if args.compile and not args.fd_gmm_takeoff_no_cos:
        raise ValueError(
            "the cosine takeoff criterion needs image-gradient diagnostics, "
            "which are unavailable with --compile; use --fd_gmm_takeoff_no_cos "
            "or disable compilation"
        )
    total_steps = args.epochs * args.steps_per_epoch
    if step >= total_steps:
        raise ValueError(
            f"--fd_gmm_takeoff_gate_step={step} is outside the {total_steps}-step run"
        )
    if not args.fd_gmm_takeoff_no_cos:
        regular = step // max(1, args.print_freq) + 1
        available = regular + int(step % max(1, args.print_freq) != 0)
        if args.fd_gmm_takeoff_cos_window > available:
            raise ValueError(
                f"--fd_gmm_takeoff_cos_window={args.fd_gmm_takeoff_cos_window} "
                f"needs more than the {available} diagnostic samples available "
                f"by gate step {step}"
            )


def resolve_gmm_cls_normalization(args):
    """Resolve the explicit class scaling mode while preserving legacy flags."""
    mode = getattr(args, "fd_gmm_cls_normalization", None) or "self"
    if mode not in {"self", "log_classes", "none"}:
        raise ValueError(f"invalid --fd_gmm_cls_normalization={mode!r}")

    explicit = getattr(args, "_fd_gmm_cls_normalization_explicit", False)
    if getattr(args, "fd_gmm_no_normalize_cls", False):
        if explicit:
            raise ValueError(
                "--fd_gmm_no_normalize_cls cannot be combined with the explicit "
                "--fd_gmm_cls_normalization option"
            )
        return "none"

    # Historically --fd_gmm_no_normalize disabled both objectives. Preserve
    # that when no class-specific mode was explicitly supplied; an explicit
    # class mode now acts as the documented override.
    if getattr(args, "fd_gmm_no_normalize", False) and not explicit:
        return "none"
    return mode


# ---------------------------------------------------------------------------
# FD train step
# ---------------------------------------------------------------------------

def get_fd_train_step(model_wo_ddp, judges, sampling_args, args, tokenizer=None,
                      gmm_judge=None, probe=None):
    fid_norm_eps = args.fd_fid_norm_eps
    batch_size = args.batch_size
    num_classes = args.num_classes
    input_shape = (args.input_channels, args.input_size, args.input_size)

    gmm_lambda_ent = args.fd_gmm_lambda_ent
    gmm_lambda_cls = args.fd_gmm_lambda_cls
    gmm_use_q = not args.fd_gmm_no_q
    gmm_mode = args.fd_gmm_mode
    gmm_clamp = args.fd_gmm_clamp
    gmm_normalize = not args.fd_gmm_no_normalize
    gmm_cls_normalization = resolve_gmm_cls_normalization(args)
    gmm_cls_cap = args.fd_gmm_cls_cap
    gmm_temp = args.fd_gmm_temp
    gmm_cfg_normalization = args.fd_gmm_cfg_normalization

    train_class_ids = getattr(args, "train_class_ids", None)
    train_class_ids_tensor = (
        None if train_class_ids is None
        else torch.tensor(train_class_ids, dtype=torch.long, device="cuda")
    )

    def _sample_labels():
        if train_class_ids_tensor is None:
            return torch.randint(0, num_classes, (batch_size,), device="cuda")
        picks = torch.randint(0, train_class_ids_tensor.numel(), (batch_size,), device="cuda")
        return train_class_ids_tensor[picks]

    def fd_train_step(gmm_scale=None, diag=False):
        # ``gmm_scale`` arrives as a 0-dim cuda tensor so the warm-up ramp is
        # data rather than a guard: changing it cannot invalidate a compiled
        # graph. Fall back to the python constant when called bare.
        if gmm_scale is None:
            gmm_scale = args.fd_gmm_weight

        z = torch.randn(batch_size, *input_shape, device="cuda") * args.noise_scale
        y = _sample_labels()
        sampled = model_wo_ddp.sample_images_with_grad(z, y, sampling_args=sampling_args)

        if tokenizer is not None:
            sampled = tokenizer.decode(tokenizer.denormalize_z(sampled))
        sampled = (sampled * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]

        loss = torch.tensor(0.0, device="cuda")
        loss_dict = {}

        all_new_feats = []
        for judge in judges:
            feats = extract_judge_features(judge, sampled)
            all_new_feats.append(diff_all_gather(feats))

        fd_term = torch.zeros((), device="cuda")
        fd_raw_sum = 0.0
        for i, judge in enumerate(judges):
            new_feats = all_new_feats[i]
            _ns_kwargs = dict(sigma_ref_sqrt=judge.get("sigma_ref_sqrt"))
            if judge["queue"].online_accum or judge["queue"].ema_stats:
                mu, sigma = judge["queue"].build_feats_stats(new_feats)
                fid = compute_frechet_distance_loss(judge["mu_ref"], judge["sigma_ref"],
                                                    mu=mu, sigma=sigma, **_ns_kwargs)
            else:
                all_feats = judge["queue"].build_feats_snapshot(new_feats)
                fid = compute_frechet_distance_loss(judge["mu_ref"], judge["sigma_ref"],
                                                    all_feats=all_feats, **_ns_kwargs)
            fid_loss = fid / (fid.detach() + fid_norm_eps)
            fd_term = fd_term + judge["weight"] * fid_loss
            fd_raw_sum += float(fid.detach())
            loss_dict[f"fid_{judge['name']}"] = float(fid.detach())
        loss = loss + fd_term
        loss_dict["fd_loss_raw"] = fd_raw_sum
        loss_dict["fd_loss_norm"] = float(fd_term.detach())

        # -- class-conditional GMM term (log p / log q) --
        # The FD term above constrains only the pooled (mu, sigma); this one
        # constrains the class-conditional structure inside it, which is what a
        # de-conditioned generator is missing entirely and what collapse
        # destroys while leaving the pooled moments intact.
        p_term_loss = None
        gmm_z, y_all = None, None
        if gmm_judge is not None:
            y_all = all_gather_plain(y)
            gmm_loss, gmm_parts, gmm_z = gmm_posterior_loss(
                all_new_feats[gmm_judge["gmm_feat_index"]], y_all,
                gmm_judge["gmm_ref"], gmm_judge["gmm_online"],
                lambda_ent=gmm_lambda_ent, lambda_cls=gmm_lambda_cls,
                use_q=gmm_use_q, mode=gmm_mode, clamp=gmm_clamp,
                normalize=gmm_normalize, cls_cap=gmm_cls_cap,
                cls_normalization=gmm_cls_normalization, temperature=gmm_temp,
                cfg_delta_normalization=gmm_cfg_normalization,
            )
            p_term_loss = gmm_scale * gmm_loss
            loss = loss + p_term_loss
            loss_dict.update({k: float(v) for k, v in gmm_parts.items()})
            loss_dict["p_loss_weighted"] = float(p_term_loss.detach())

        # -- per-term gradient diagnostics on the generated pixels (log-steps).
        # Loss *values* are not comparable across terms (both are
        # self-normalized, at different scales), so who actually drives the
        # update is read from grad-w.r.t.-image norms; the cosine exposes
        # conflict between the conditional pull and the realism pull.
        if diag:
            def _grad_x(term):
                if term is None:
                    return None
                g = torch.autograd.grad(term, sampled, retain_graph=True,
                                        allow_unused=True)[0]
                return None if g is None else g.detach().reshape(-1)
            g_fd, g_p = _grad_x(fd_term), _grad_x(p_term_loss)

            def _norm(g):
                return float(g.norm()) if g is not None else 0.0

            def _cos(a, b):
                if a is None or b is None:
                    return 0.0
                d = float(a.norm() * b.norm())
                return float(a @ b) / d if d > 0 else 0.0
            loss_dict["grad_x_fd"] = _norm(g_fd)
            loss_dict["grad_x_p"] = _norm(g_p)
            loss_dict["cos_update_fd_p"] = _cos(g_fd, g_p)
            # The single number that says whether the weight is in a useful
            # range: how much of the image-space pull comes from conditioning
            # vs realism. << 1 means the conditional term is a rounding error.
            loss_dict["grad_ratio_p_fd"] = (
                loss_dict["grad_x_p"] / loss_dict["grad_x_fd"]
                if loss_dict["grad_x_fd"] > 0 else 0.0
            )

            # -- label sensitivity: does the generator use y at all? --
            # Re-sample the *same* noise under rolled labels. cond_delta is the
            # relative pixel change caused purely by swapping the class token,
            # and is exactly 0 on the de-conditioned checkpoint at step 0. roll
            # is used instead of randperm so the global RNG stream is untouched.
            with torch.no_grad():
                y_alt = torch.roll(y, 1, dims=0)
                alt = model_wo_ddp.sample_images_with_grad(
                    z, y_alt, sampling_args=sampling_args)
                if tokenizer is not None:
                    alt = tokenizer.decode(tokenizer.denormalize_z(alt))
                alt = (alt * 0.5 + 0.5).clamp(0, 1)
                ref = sampled.detach()
                delta = (ref - alt).flatten(1).norm(dim=1)
                scale = ref.flatten(1).norm(dim=1).clamp_min(1e-8)
                loss_dict["cond_delta"] = float((delta / scale).mean())
                loss_dict["cond_delta_valid"] = float((y_alt != y).float().mean())

            if probe is not None:
                # Held-out eval judge on this training batch — the honest
                # conditioning meter; the in-loss GMM's own top-1 is not.
                loss_dict.update(probe.stats(sampled, y))

        loss.backward(create_graph=False)

        if torch.distributed.is_initialized():
            for p in model_wo_ddp.parameters():
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

        # Queue and GMM state mutation stay outside the compiled region: inductor
        # can otherwise stale-specialize a circular buffer pointer or reorder
        # chained in-place state updates across graph breaks.
        return (loss, loss_dict,
                tuple(f.detach() for f in all_new_feats),
                None if gmm_z is None else gmm_z.detach(),
                None if y_all is None else y_all.detach())

    if args.compile:
        from utils.runtime_util import _warmup
        logger.info("[Compilation] Compiling fd_train_step ...")
        t0 = time.perf_counter()
        fd_train_step = torch.compile(fd_train_step)
        _zero = torch.zeros((), device="cuda")
        _warmup(lambda: fd_train_step(_zero), n=2)
        for p in model_wo_ddp.parameters():
            p.grad = None
        logger.info(f"[Compilation] fd_train_step compiled in {time.perf_counter() - t0:.2f}s")

    return fd_train_step


# ---------------------------------------------------------------------------
# Class-conditional GMM setup
# ---------------------------------------------------------------------------

def setup_gmm_judge(judges, args):
    """Attach the log p / log q state to one judge; return it (or None if off).

    The term is deliberately restricted to a single judge: it is a per-sample
    signal whose cost is dominated by the (C, k, k) reference, and running it on
    several representation spaces at once buys redundancy rather than coverage.
    """
    if not args.fd_gmm:
        return None

    if args.fd_gmm_judge is None:
        index = 0
    else:
        names = [judge["name"] for judge in judges]
        if args.fd_gmm_judge not in names:
            raise ValueError(
                f"--fd_gmm_judge='{args.fd_gmm_judge}' not among the configured "
                f"judges {names}"
            )
        index = names.index(args.fd_gmm_judge)

    judge = judges[index]
    if args.fd_gmm_stats_path is None:
        raise ValueError("--fd_gmm requires --fd_gmm_stats_path (see compute_class_stats.py)")

    reference = ClassGMMReference.from_npz(
        args.fd_gmm_stats_path,
        pca_dim=args.fd_gmm_pca_dim,
        shrinkage=args.fd_gmm_p_shrinkage,
    )
    if reference.proj.shape[0] != judge["feat_dim"]:
        raise ValueError(
            f"GMM stats were fitted on {reference.proj.shape[0]}-d features but judge "
            f"'{judge['name']}' produces {judge['feat_dim']}-d. The projection must be "
            f"fitted on the same representation space."
        )
    # Every label the generator can draw must have a component, and the softmax
    # denominator must not include classes it never draws.
    drawable = (args.train_class_ids if args.train_class_ids is not None
                else list(range(args.num_classes)))
    reference.validate_labels(drawable)
    if reference.num_classes != len(drawable):
        raise ValueError(
            f"GMM has {reference.num_classes} components but the generator draws "
            f"{len(drawable)} label(s). -log p(c|x) would be a "
            f"{reference.num_classes}-way problem over classes that are never "
            f"generated. Refit with compute_class_stats.py --class_ids."
        )

    online = OnlineClassStats(
        num_classes=reference.num_classes,
        k=reference.k,
        ema_beta=args.fd_gmm_ema_beta,
        shrinkage=args.fd_gmm_q_shrinkage,
        cov_ema_beta=args.fd_gmm_cov_ema_beta,
    ).cuda()

    judge["gmm_ref"] = reference
    judge["gmm_online"] = online
    judge["gmm_feat_index"] = index
    # Initialise the cache from the empty state so ``logits`` is never read
    # against zeroed buffers (e.g. when queue_size=0 skips the bootstrap).
    online.refresh_cache(reference)

    logger.info(
        f"[GMM] Attached to judge '{judge['name']}': C={reference.num_classes}, "
        f"k={reference.k}, weight={args.fd_gmm_weight}, "
        f"lambda_cls={args.fd_gmm_lambda_cls}, lambda_ent={args.fd_gmm_lambda_ent}, "
        f"mode={args.fd_gmm_mode}, "
        f"cls_normalization={resolve_gmm_cls_normalization(args)}, "
        f"cls_cap={args.fd_gmm_cls_cap}, "
        f"temp={args.fd_gmm_temp}, "
        f"q_mean_ema_beta={online.ema_beta}, q_cov_ema_beta={online.cov_ema_beta}, "
        f"p_shrink={args.fd_gmm_p_shrinkage}, "
        f"q_shrink={args.fd_gmm_q_shrinkage}, warmup={args.fd_gmm_warmup_steps}, "
        f"ramp={args.fd_gmm_ramp_steps}"
        + (" [ABLATION: q term disabled]" if args.fd_gmm_no_q else "")
        + (" [CONTROL: weight=0, meters only]" if args.fd_gmm_weight == 0 else "")
    )
    if args.fd_gmm_mode == "cfg_delta":
        # Which of the two experiments this is, said out loud: with a separate
        # -log p(c|x) driver the run is NOT the pure chain-rule objective --
        # the teacher CFG delta already contains -grad log p(c|z), so class
        # fidelity is being emphasised twice. They are different objectives and
        # must never be reported as the same one.
        variant = ("[PURE: no explicit class driver]" if args.fd_gmm_lambda_cls == 0
                   else "[PRACTICAL: explicit -log p(c|z) retained]")
        logger.info(
            f"[GMM] mode=cfg_delta {variant} "
            f"temp={args.fd_gmm_temp}, "
            f"cfg_vector_normalization={args.fd_gmm_cfg_normalization}, "
            f"lambda_cls={args.fd_gmm_lambda_cls}, "
            f"lambda_ent={args.fd_gmm_lambda_ent}, weight={args.fd_gmm_weight}"
        )
        logger.info(
            "[GMM] cfg_delta injects a feature-space vector field, not a KL: "
            "'gmm_cfg_surrogate' is origin-dependent and is not comparable "
            "across runs. Calibrate on grad_ratio_p_fd; a density-mode weight "
            "does not transfer."
        )
    if args.fd_gmm_takeoff_gate_step >= 0:
        spread_rule = (
            "disabled" if args.fd_gmm_takeoff_spread_mult == 0.0
            else f"spread/noise >= {args.fd_gmm_takeoff_spread_mult:g}"
        )
        cosine_rule = (
            "disabled" if args.fd_gmm_takeoff_no_cos
            else f"cosine < {args.fd_gmm_takeoff_cos_threshold:g}"
        )
        logger.info(
            f"[GMM takeoff gate] step={args.fd_gmm_takeoff_gate_step}, "
            f"required=({spread_rule}, {cosine_rule}), "
            f"abort_logic={args.fd_gmm_takeoff_logic}"
        )
    # tr(Sigma_within^p)/k fixes where within_trace_ratio starts on a
    # de-conditioned model (all class means coincide -> q's within-class scatter
    # is its full marginal scatter, ~1.0 in whitened units).
    within_frac = float(reference.within_trace) / reference.k
    logger.info(
        f"[GMM] tr(Sigma_within^p)/k={within_frac:.4f} -> expect "
        f"gmm_within_trace_ratio ~{1.0 / max(within_frac, 1e-6):.2f} at init, "
        f"falling to 1.0 as classes separate (below 1.0 = collapse); "
        f"gmm_class_mean_spread starts ~0.0 and targets 1.0"
    )
    return judge


@torch.no_grad()
def bootstrap_gmm_stats(gmm_judge, model, args, tokenizer=None):
    """Seed the generated-side statistics before the loss switches on.

    Only needed when the FD queues were restored from a checkpoint that predates
    the GMM state; the normal path folds the bootstrap into the queue fill.
    """
    if gmm_judge is None or args.fd_gmm_bootstrap <= 0:
        return

    online = gmm_judge["gmm_online"]
    reference = gmm_judge["gmm_ref"]
    train_class_ids = getattr(args, "train_class_ids", None)
    ids_tensor = (None if train_class_ids is None
                  else torch.tensor(train_class_ids, dtype=torch.long, device="cuda"))
    model.eval()
    filled = 0
    while filled < args.fd_gmm_bootstrap:
        bsz = min(args.fd_queue_fill_bsz, args.fd_gmm_bootstrap - filled)
        if ids_tensor is None:
            y = torch.randint(0, args.num_classes, (bsz,), device="cuda")
        else:
            y = ids_tensor[torch.randint(0, ids_tensor.numel(), (bsz,), device="cuda")]
        imgs = model.generate(bsz, y, cfg=args.cfg, args=args, verbose=False)
        imgs = tokenizer.detokenize(imgs) if tokenizer is not None else imgs * 0.5 + 0.5

        feats = extract_judge_features(gmm_judge, imgs)
        feats_all = diff_all_gather(feats).detach()
        y_all = all_gather_plain(y)
        online.update(reference.project(feats_all), reference.to_local(y_all))

        filled += bsz
        if filled % (args.fd_queue_fill_bsz * 20) == 0 or filled >= args.fd_gmm_bootstrap:
            logger.info(f"[GMM] Bootstrap: {filled}/{args.fd_gmm_bootstrap} "
                        f"(class coverage {online.coverage * 100:.1f}%)")

    online.refresh_cache(reference)
    stats = online.diagnostics(reference)
    logger.info(f"[GMM] Bootstrap done ({int(online.total_seen.item())} samples): "
                + ", ".join(f"{k}={v:.4f}" for k, v in stats.items()))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_and_evaluate(args):
    args.train_class_ids = validate_train_class_ids(
        getattr(args, "train_class_ids", None), args.num_classes,
    )
    validate_gmm_takeoff_gate_args(args)
    wandb_logger = setup(args)
    register_preempt_handler()

    if args.train_class_ids is not None:
        logger.info(
            "[ClassSubset] Sampling %d of %d labels during training: %s",
            len(args.train_class_ids), args.num_classes, args.train_class_ids,
        )

    # -- models, optimizer, checkpoint --
    tokenizer = create_tokenizer(args)
    model, ema_model = create_generation_model(args)
    optimizer = create_optimizer(args, model, print_trainable_params=True)
    model_wo_ddp = model

    extra = ckpt_resume(
        args, model_wo_ddp, optimizer, ema_model,
        extra_keys=[
            "fd_queue_states", "fd_gmm_state",
            "fd_gmm_takeoff_cos_history", "fd_gmm_takeoff_gate_status",
        ],
    )
    takeoff_cos_history = deque(maxlen=args.fd_gmm_takeoff_cos_window)
    if extra is not None and extra.get("fd_gmm_takeoff_cos_history") is not None:
        takeoff_cos_history.extend(
            float(value) for value in extra["fd_gmm_takeoff_cos_history"]
            if math.isfinite(float(value))
        )
    restored_gate_status = (
        None if extra is None else extra.get("fd_gmm_takeoff_gate_status")
    )
    if (args.fd_gmm_takeoff_gate_step >= 0
            and restored_gate_status is not None
            and restored_gate_status.get("aborted", False)):
        logger.error(
            "[GMM takeoff gate] Refusing to resume a checkpoint already marked "
            "as gated: %s. Use a new experiment or disable the gate explicitly.",
            restored_gate_status,
        )
        return 4

    rng = RNGStateManager()
    rng.save()
    if (not args.disable_vis) or args.vis_only:
        visualize(args, model_wo_ddp, ema_model, args.current_step, rng=rng, tokenizer=tokenizer)
        if args.vis_only:
            return 0

    # -- frechet distance evaluator --
    repr_model_eval, feat_dim_eval, _, _ = load_repr_model("inception")
    fid_evaluator = FDEvaluator(repr_model_eval, feat_dim_eval, args.fid_stats_path)

    # -- frechet distance system: repr models, queues --
    resolve_per_model_args(args)

    judges = []
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
            "name": short, "model": repr_model,
            "feat_dim": feat_dim,
            "pool_type": pool_type,
            "mu_ref": mu_ref, "sigma_ref": sigma_ref,
            "sigma_ref_sqrt": sigma_ref_sqrt,
            "queue": queue, "weight": weight,
        })
        eig_mode = "eigvalsh" if args.fd_eigvalsh else "eigvals"
        stats_mode = (f"ema(beta={args.fd_ema_beta})" if args.fd_ema_beta > 0
                      else ("online_accum" if args.fd_online_accum else "snapshot"))
        logger.info(f"[FD] Repr '{short}' ({name}): feat_dim={feat_dim}, "
                    f"weight={weight}, pool={pool_type}, stats={stats_path}, "
                    f"eig_mode={eig_mode}, stats_mode={stats_mode}")

    # -- class-conditional GMM term (log p / log q) --
    gmm_judge = setup_gmm_judge(judges, args)

    fd_restored = (extra is not None
                   and "fd_queue_states" in extra
                   and load_fd_queue_states(judges, extra["fd_queue_states"]))
    gmm_restored = False
    if gmm_judge is not None and extra is not None and extra.get("fd_gmm_state") is not None:
        gmm_judge["gmm_online"].load_state_dict(extra["fd_gmm_state"])
        gmm_judge["gmm_online"].cuda()
        gmm_judge["gmm_online"].refresh_cache(gmm_judge["gmm_ref"])
        gmm_restored = True
        logger.info("[GMM] Restored generated-side class statistics from checkpoint")

    if fd_restored:
        logger.info("[FD] Restored all queue states from checkpoint — skipping queue fill")
        run_sanity_check(judges, args.queue_size, args=args)
        if gmm_judge is not None and not gmm_restored:
            # Queues came from a run that predates the GMM term; q has to be
            # seeded from scratch or the first steps would train against noise.
            logger.info("[GMM] No saved statistics — running a standalone bootstrap pass")
            bootstrap_gmm_stats(gmm_judge, model_wo_ddp, args, tokenizer=tokenizer)
    else:
        logger.info(f"[FD] Filling {len(judges)} feature queue(s) "
                    f"({args.queue_size} entries each) ...")
        fill_all_queues(judges, model_wo_ddp, args, tokenizer=tokenizer,
                        gmm_judge=None if gmm_restored else gmm_judge)
        run_sanity_check(judges, args.queue_size, args=args)

    del extra
    torch.distributed.barrier()

    model.train()
    args.input_channels = model_wo_ddp.in_channels
    args.input_size = model_wo_ddp.input_size

    probe = None
    if args.cond_probe:
        probe = ProbeClassifier(device="cuda")
        logger.info("[Probe] held-out resnet50 IMAGENET1K_V2 (the "
                    "eval_class_accuracy.py judge) logged as probe_top1/top5/"
                    "logp/rank at diag steps — never in the loss")

    # -- FD train step closure --
    sampling_args = {
        "t_min": args.interval_min,
        "t_max": args.interval_max,
        "cfg": args.cfg,
        "num_steps": args.num_sampling_steps,
    }
    fd_train_step = get_fd_train_step(
        model_wo_ddp, judges, sampling_args, args, tokenizer=tokenizer,
        gmm_judge=gmm_judge, probe=probe,
    )

    # -- training loop --
    logger.info(f"training from step {args.current_step:,} -> {args.total_steps:,} "
                f"({args.start_epoch} -> {args.epochs} epochs)")

    global_bsz = args.batch_size * args.world_size
    ckpt_saver = AsyncCheckpointSaver()
    session_start = time.time()
    step_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # dynamic checkpoint frequency: target ~10 min between saves
    ckpt_target_minutes = 10.0
    ckpt_measure_interval = 1000
    ckpt_timer_start = time.perf_counter()
    ckpt_timer_step = args.current_step
    last_ckpt_step = args.current_step

    metric_file = os.path.join(args.log_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metric_file, prefetch=True)
    takeoff_gate_status = restored_gate_status
    for name, window, fmt in [
        ("lr",               1,               "{value:.6f}"),
        ("samples/s/device", args.print_freq, "{avg:.2f}"),
        ("samples/s",        args.print_freq, "{avg:.2f}"),
        ("samples_seen(M)",  args.print_freq, "{value:.2f}"),
        ("device_mem(GB)",   args.print_freq, "{value:.2f}"),
    ]:
        metric_logger.add_meter(name, SmoothedValue(window, fmt))

    def _infinite():
        while True:
            yield None

    for step, _ in metric_logger.log_every(
        _infinite(), args.print_freq, header="Train:",
        start_iteration=args.current_step, n_iterations=args.total_steps,
    ):
        model.train()
        adjust_learning_rate(optimizer, step, args)

        scale_eff = gmm_scale_schedule(step, args) if gmm_judge is not None else 0.0
        # pass as a 0-dim tensor so a compiled step does not recompile as the
        # warmup/ramp changes its value
        scale_t = torch.as_tensor(scale_eff, device="cuda", dtype=torch.float32)
        # Per-term gradient diagnostics cost extra partial backwards through
        # the frozen judges.  The gate forces one at its exact step even when
        # that is not a normal log-step; cosine gating is rejected with compile.
        gate_due = step == args.fd_gmm_takeoff_gate_step
        diag = ((not args.compile)
                and (step % args.print_freq == 0 or gate_due))
        loss, loss_dict, new_feats, gmm_z, y_all = fd_train_step(gmm_scale=scale_t, diag=diag)

        if gmm_judge is not None:
            loss_dict["gmm_scale_eff"] = scale_eff

        grad_norm = (torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                     if args.grad_clip > 0.0 else get_grad_norm(model.parameters()))

        if torch.isfinite(grad_norm):
            optimizer.step()
            ema_model.step(model)
        else:
            logger.warning(f"[step {step}] NaN/Inf grad_norm — skipping optimizer & EMA update")
        optimizer.zero_grad(set_to_none=True)

        # -- state updates, outside the compiled region --
        for i, judge in enumerate(judges):
            judge["queue"].enqueue(new_feats[i])
        if gmm_judge is not None:
            reference = gmm_judge["gmm_ref"]
            online = gmm_judge["gmm_online"]
            online.update(gmm_z, reference.to_local(y_all))
            online.refresh_cache(reference)
            if diag or gate_due:
                loss_dict.update(online.diagnostics(reference))

        torch.cuda.synchronize()

        args.current_step = step + 1
        args.samples_seen += global_bsz

        step_time = time.perf_counter() - step_start
        step_start = time.perf_counter()

        loss_value = all_reduce_mean(loss.item())
        loss_dict = {k: all_reduce_mean(v) for k, v in loss_dict.items()}
        if "cos_update_fd_p" in loss_dict:
            takeoff_cos_history.append(float(loss_dict["cos_update_fd_p"]))
        sps = args.batch_size / step_time if step_time > 0 else 0.0
        mem_gb = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        metric_logger.update(
            loss=loss_value, grad_norm=grad_norm,
            lr=optimizer.param_groups[0]["lr"],
            **{"samples/s/device": sps, "samples/s": sps * args.world_size,
               "samples_seen(M)": args.samples_seen / 1e6, "device_mem(GB)": mem_gb},
            **loss_dict,
        )

        gate_abort = False
        if gate_due:
            cosine_ready = (
                args.fd_gmm_takeoff_no_cos
                or len(takeoff_cos_history) >= args.fd_gmm_takeoff_cos_window
            )
            cosine_mean = (
                sum(takeoff_cos_history) / len(takeoff_cos_history)
                if takeoff_cos_history else None
            )
            gate_metrics = {
                "gmm_class_mean_spread_to_noise": loss_dict.get(
                    "gmm_class_mean_spread_to_noise"
                ),
                "cos_update_fd_p": (
                    cosine_mean if cosine_ready and not args.fd_gmm_takeoff_no_cos
                    else None
                ),
            }
            gate_abort, failures = evaluate_gmm_takeoff_gate(
                gate_metrics,
                spread_multiplier=args.fd_gmm_takeoff_spread_mult,
                cosine_threshold=args.fd_gmm_takeoff_cos_threshold,
                logic=args.fd_gmm_takeoff_logic,
                check_spread=args.fd_gmm_takeoff_spread_mult > 0.0,
                check_cosine=not args.fd_gmm_takeoff_no_cos,
            )
            takeoff_gate_status = {
                "evaluated": True,
                "aborted": bool(gate_abort),
                "step": int(step),
                "logic": args.fd_gmm_takeoff_logic,
                "failures": failures,
                "spread_to_noise": gate_metrics["gmm_class_mean_spread_to_noise"],
                "spread_required": args.fd_gmm_takeoff_spread_mult,
                "cosine_mean": cosine_mean,
                "cosine_required_below": args.fd_gmm_takeoff_cos_threshold,
                "cosine_samples": len(takeoff_cos_history),
                "cosine_window": args.fd_gmm_takeoff_cos_window,
            }
            gate_log = {
                "gmm_takeoff_gate_abort": int(gate_abort),
                "gmm_takeoff_spread_failed": int(failures.get("spread", False)),
                "gmm_takeoff_cosine_failed": int(failures.get("cosine", False)),
                "gmm_takeoff_cosine_samples": len(takeoff_cos_history),
            }
            if cosine_mean is not None:
                gate_log["gmm_takeoff_cosine_mean"] = cosine_mean
            loss_dict.update(gate_log)
            metric_logger.update(**gate_log)
            status_word = "ABORT" if gate_abort else "PASS"
            if args.fd_gmm_takeoff_no_cos:
                cosine_summary = "cosine=disabled"
            elif cosine_mean is None:
                cosine_summary = (
                    f"cosine=missing (0/{args.fd_gmm_takeoff_cos_window} samples)"
                )
            else:
                cosine_summary = (
                    f"cosine_mean={cosine_mean:.6f} over "
                    f"{len(takeoff_cos_history)}/{args.fd_gmm_takeoff_cos_window} "
                    f"samples, required<{args.fd_gmm_takeoff_cos_threshold:g}"
                )
            logger.info(
                f"[GMM takeoff gate] {status_word} at step {step}: "
                f"spread/noise={gate_metrics['gmm_class_mean_spread_to_noise']}, "
                f"required>={args.fd_gmm_takeoff_spread_mult:g}; "
                f"{cosine_summary}; "
                f"failures={failures}, logic={args.fd_gmm_takeoff_logic}"
            )

        if step % args.print_freq == 0 and wandb_logger:
            elapsed = time.time() - session_start + args.last_elapsed_time
            remaining = args.total_steps - args.current_step
            eta = elapsed / args.current_step * remaining if args.current_step > 0 else 0.0
            elapsed_h = elapsed / 3600
            wandb_logger.update({
                "train/loss": loss_value,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": grad_norm,
                "train/samples_seen_M": args.samples_seen / 1e6,
                "perf/samples_per_sec_per_device": sps,
                "perf/samples_per_sec": sps * args.world_size,
                "perf/max_reserved_mem_gb": mem_gb,
                "perf/elapsed_real_hours": elapsed_h,
                "perf/elapsed_device_hours": elapsed_h * args.world_size,
                "perf/eta_real_hours": eta / 3600,
                "perf/eta_device_hours": eta / 3600 * args.world_size,
                **{f"train/{k}": v for k, v in loss_dict.items()},
            }, step=args.current_step)

        # dynamic checkpoint frequency
        steps_since_timer = args.current_step - ckpt_timer_step
        if steps_since_timer >= ckpt_measure_interval:
            elapsed_minutes = (time.perf_counter() - ckpt_timer_start) / 60.0
            minutes_per_step = elapsed_minutes / steps_since_timer
            new_save_every = max(100, round(ckpt_target_minutes / minutes_per_step / 100) * 100)
            if new_save_every != args.save_every:
                logger.info(f"adjusting save_every: {args.save_every} -> {new_save_every} "
                            f"({minutes_per_step * 1000:.1f} min/1k steps)")
                args.save_every = new_save_every
            ckpt_timer_start = time.perf_counter()
            ckpt_timer_step = args.current_step

        def _save(saver=ckpt_saver):
            elapsed = time.time() - session_start + args.last_elapsed_time
            fd_extra = {"fd_queue_states": save_fd_queue_states(judges)} if judges else {}
            if gmm_judge is not None:
                fd_extra["fd_gmm_state"] = gmm_judge["gmm_online"].state_dict()
            if args.fd_gmm_takeoff_gate_step >= 0:
                fd_extra["fd_gmm_takeoff_cos_history"] = list(takeoff_cos_history)
                fd_extra["fd_gmm_takeoff_gate_status"] = takeoff_gate_status
            save_checkpoint(args, step, model_wo_ddp, optimizer, ema_model, elapsed,
                            saver=saver, extra=fd_extra)
            torch.distributed.barrier()

        if gate_abort:
            # Returning from the yielded training loop would otherwise skip the
            # MetricLogger's post-yield JSON write.  Persist the decisive row,
            # then make a synchronous checkpoint explicitly marked as gated so
            # auto-resume cannot silently continue the failed experiment.
            metric_logger.dump_in_output_file(step, step_time, 0.0)
            logger.error(
                f"[GMM takeoff gate] Saving gated checkpoint at step {step} "
                "and exiting with status 4"
            )
            ckpt_saver.wait()
            _save(saver=None)
            logger.error("[GMM takeoff gate] Gated checkpoint saved; training aborted")
            return 4

        if (args.current_step - last_ckpt_step >= args.save_every
                or args.current_step == args.total_steps):
            _save()
            last_ckpt_step = args.current_step

        if args.milestone_every > 0 and step > 0 and step % args.milestone_every == 0:
            _save()

        if preempt_requested():
            logger.info(f"Preemption at step {args.current_step}: saving checkpoint ...")
            ckpt_saver.wait()
            _save(saver=None)
            logger.info(f"Preemption checkpoint saved at step {args.current_step}. Exiting.")
            return 0

        if args.vis_every > 0 and args.current_step % args.vis_every == 0:
            visualize(args, model_wo_ddp, ema_model, args.current_step, rng=rng, tokenizer=tokenizer)
            model_wo_ddp.train()

        if args.eval_every > 0 and args.online_eval and args.current_step % args.eval_every == 0:
            torch.cuda.empty_cache()
            evaluate_all_emas(
                args, model_wo_ddp, ema_model, fid_evaluator, tokenizer,
                step=args.current_step, wandb_logger=wandb_logger,
                cfg=args.cfg, num_images=args.num_images_for_eval_and_search,
            )
            model_wo_ddp.train()

    ckpt_saver.wait()
    total = time.time() - session_start + args.last_elapsed_time
    metric_logger.synchronize_between_processes()
    logger.info(f"averaged stats: {metric_logger}")
    logger.info(f"Training complete. Total time: {datetime.timedelta(seconds=int(total))} "
                f"on {args.world_size} devices")
    torch.cuda.empty_cache()

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser("FD loss fine-tuning with a class-conditional GMM",
                                     add_help=False)

    # training
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--steps_per_epoch", default=1250, type=int)
    parser.add_argument("--batch_size", default=32, type=int, help="batch size per GPU")
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

    # rectified-flow (NCSN++) base model
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
    parser.add_argument("--lr_sched", type=str, default="constant", choices=["constant", "cosine"])
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
    parser.add_argument("--ema_halflife_kimg", default=[250, 500, 1000, 2000], type=float, nargs="+")
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
    parser.add_argument("--sampling_method", type=str, default="heun", choices=["euler", "heun"])
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
                        help="optional label subset sampled during generator training "
                             "(and during the FD queue fill). The GMM stats file must "
                             "have been fitted on exactly this subset.")
    parser.add_argument("--class_of_interest", default=[207, 360, 387, 974, 88, 979, 417, 279],
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
    parser.add_argument("--fd_repr_models", type=str, nargs="+", default=["inception"])
    parser.add_argument("--fd_repr_stats_paths", type=str, nargs="+", default=None)
    parser.add_argument("--fd_repr_weights", type=float, nargs="+", default=None)
    parser.add_argument("--fd_repr_pool_types", type=str, nargs="+", default=None)
    parser.add_argument("--fd_target_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--fd_online_accum", action="store_true")
    parser.add_argument("--fd_eigvalsh", action="store_true")
    parser.add_argument("--fd_ema_beta", type=float, default=0.0, metavar="BETA",
                        help="EMA decay for FD stats (0=disabled, use queue). 0.99 is a "
                             "~100-step window; 0.999 was shown unable to see mode "
                             "collapse (a drifting single mode still accumulates a "
                             "broad covariance over 1000 steps)")

    # -- class-conditional GMM term (log p / log q) --
    parser.add_argument("--fd_gmm", action="store_true",
                        help="enable the class-conditional log p / log q term. With "
                             "--fd_gmm_weight 0 this is the matched control: identical "
                             "code path, all meters logged, exactly zero gradient")
    parser.add_argument("--fd_gmm_stats_path", type=str, default=None,
                        help="real-data class GMM stats (.npz) from compute_class_stats.py")
    parser.add_argument("--fd_gmm_judge", type=str, default=None,
                        help="short name of the judge to attach the term to (default: first)")
    parser.add_argument("--fd_gmm_pca_dim", type=int, default=None,
                        help="truncate the whitened PCA space to this many dims")
    parser.add_argument("--fd_gmm_weight", type=float, default=0.002,
                        help="overall weight of the GMM term. Calibrate with "
                             "grad_ratio_p_fd, not with the loss value: the "
                             "classifier-ensemble run that reached 63.5%% held-out "
                             "top-1 on this checkpoint sat at ~0.18")
    parser.add_argument("--fd_gmm_lambda_ent", type=float, default=1.0,
                        help="weight of the anti-collapse term. 0 leaves pure "
                             "classifier guidance, which is collapse-promoting")
    parser.add_argument("--fd_gmm_lambda_cls", type=float, default=1.0,
                        help="weight of the class-fidelity term E[-log p(c|x)]. Unlike "
                             "the conditional-model recipe (0.1, a regulariser) this is "
                             "the driver here: it is the only term that carves class "
                             "structure out of a de-conditioned model")
    parser.add_argument("--fd_gmm_cls_cap", type=float, default=0.0,
                        help="cap per-sample log p(c|x) at this value (<0); once a "
                             "sample is this confident it stops contributing gradient, "
                             "preventing the race to adversarial certainty. "
                             "e.g. -0.69 ~= cap at 50%%. 0 disables")
    parser.add_argument("--fd_gmm_temp", type=float, default=1.0,
                        help="softmax temperature for the class posterior. In a "
                             "k-dim whitened space the per-class Mahalanobis gaps run "
                             "to tens of nats, so at T=1 the posterior saturates and "
                             "-log p(c|x) has NO gradient on realistic samples "
                             "(measured on the 20-class inception fit: 98.6%% of real "
                             "held-out images sit at exactly log p = 0). T>1 flattens "
                             "it without changing top-1. Pick T so real images land "
                             "just above --fd_gmm_cls_cap; T=100 gives median -0.31 "
                             "there. Does not affect the density-ratio term.")
    parser.add_argument("--fd_gmm_mode", type=str, default="density",
                        choices=["density", "posterior", "cfg_delta"])
    parser.add_argument("--fd_gmm_cfg_normalization", choices=["none", "rms"],
                        default="none",
                        help="normalization for the explicit cfg_delta feature "
                             "vector. 'none' preserves the raw fitted-GMM field; "
                             "'rms' divides by its detached batch RMS and changes "
                             "the objective. Separate from the scalar GMM loss "
                             "self-normalisation, which never applies here")
    parser.add_argument("--fd_gmm_clamp", type=float, default=3.0,
                        help="per-sample bound on the density log-ratio in batch "
                             "standard deviations about the batch mean")
    parser.add_argument("--fd_gmm_no_normalize", action="store_true",
                        help="disable the fid-style self-normalisation of each GMM term")
    cls_norm_group = parser.add_mutually_exclusive_group()
    parser.set_defaults(_fd_gmm_cls_normalization_explicit=False)
    cls_norm_group.add_argument(
        "--fd_gmm_cls_normalization", choices=["self", "log_classes", "none"],
        default=None, action=_StoreExplicit,
        help="class-fidelity scaling: 'self' is the legacy adaptive "
             "l_cls/(|l_cls|+0.01); 'log_classes' divides by fixed ln(C); "
             "'none' uses raw l_cls (default: self)",
    )
    cls_norm_group.add_argument(
        "--fd_gmm_no_normalize_cls", action="store_true",
        help="legacy alias for --fd_gmm_cls_normalization none",
    )
    parser.add_argument("--fd_gmm_no_q", action="store_true",
                        help="ABLATION: drop the q term entirely (-log p only)")
    parser.add_argument("--fd_gmm_ema_beta", type=float, default=0.999,
                        help="per-class mean EMA decay for the generated-side statistics")
    parser.add_argument("--fd_gmm_cov_ema_beta", type=float, default=None,
                        help="EMA decay for the generated tied covariance; defaults "
                             "to --fd_gmm_ema_beta for legacy shared-beta behaviour")
    parser.add_argument("--fd_gmm_p_shrinkage", type=float, default=0.75,
                        help="shrink real per-class covariances toward the pooled "
                             "within-class covariance. 0.75 minimised held-out "
                             "-log p(c|x) on the 1000-class inception fit")
    parser.add_argument("--fd_gmm_q_shrinkage", type=float, default=0.25,
                        help="shrink the generated tied covariance toward the real "
                             "pooled within-class covariance; also floors it")
    parser.add_argument("--fd_gmm_warmup_steps", type=int, default=0,
                        help="steps at weight 0 before the term turns on")
    parser.add_argument("--fd_gmm_ramp_steps", type=int, default=500,
                        help="linear ramp length (steps) after warmup")
    parser.add_argument("--fd_gmm_bootstrap", type=int, default=50000,
                        help="samples for a standalone q bootstrap when queue states "
                             "are restored but GMM state is not; 0 disables")
    parser.add_argument("--fd_gmm_takeoff_gate_step", type=int, default=-1,
                        help="opt-in early takeoff check at this exact zero-based "
                             "training step; -1 disables it")
    parser.add_argument("--fd_gmm_takeoff_spread_mult", type=float, default=2.0,
                        help="require gmm_class_mean_spread to clear this multiple "
                             "of its finite-EMA noise floor at the takeoff gate; "
                             "0 disables the spread criterion")
    parser.add_argument("--fd_gmm_takeoff_cos_threshold", type=float, default=0.0,
                        help="require the windowed mean cos_update_fd_p to be below "
                             "this value at the takeoff gate (default 0 = negative)")
    parser.add_argument("--fd_gmm_takeoff_cos_window", type=int, default=50,
                        help="number of raw diagnostic cosine samples averaged by "
                             "the gate (50 spans ~1000 steps at print_freq=20); "
                             "history is saved in checkpoints")
    parser.add_argument("--fd_gmm_takeoff_logic", choices=["any", "all"],
                        default="any",
                        help="'any' aborts if either enabled criterion fails; 'all' "
                             "aborts only if every enabled criterion fails")
    parser.add_argument("--fd_gmm_takeoff_no_cos", action="store_true",
                        help="disable the cosine criterion (allows a spread-only "
                             "takeoff gate with --compile)")

    parser.add_argument("--cond_probe", action="store_true",
                        help="log held-out torchvision resnet50 V2 top-1/top-5/logp/rank "
                             "on the training batch at diag steps (the same weights "
                             "eval_class_accuracy.py judges with; never in the loss)")

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
    parser.add_argument("--project", default="JiT_uncond_gmm", type=str)
    parser.add_argument("--entity", default=None, type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--disable_wandb", action="store_false", dest="enable_wandb")

    # system
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--dtype", default="bf16", type=str, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compile", action="store_true")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    sys.exit(train_and_evaluate(args))
