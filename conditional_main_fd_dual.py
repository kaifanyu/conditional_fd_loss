"""Two independent semantic judges inside one conditional log-p correction.

`conditional_main_fd_mae.py` uses a frozen MAE linear probe; `..._ponly.py`
uses either a classifier ensemble or one frozen VLM, and refuses to combine
them. This entrypoint runs both the MAE probe and the VLM at once:

    L = L_FD  +  lambda_mae(s) * L_mae  +  lambda_vlm(s) * L_vlm

    L_mae = -mean_B  min(log p_mae(y | x), cap_mae)
    L_vlm = -mean_2K min(log p_vlm(correct answer | x, question), cap_vlm)

Adding the two capped negative log probabilities is a weighted product of
experts, i.e. the AND-semantics that `classifier_ensemble.py` argues for at
length: a judge that reaches its own cap stops contributing gradient while the
unsatisfied one keeps pushing. Averaging *probabilities* would be OR-semantics
and would let the generator satisfy whichever judge is easiest to fool. The two
terms are not comparable term-by-term -- the MAE probe is a 1000-way softmax
(chance -log 1000 = -6.91) and the VLM is a binary Yes/No margin (chance
-log 2 = -0.69) -- which is exactly why each gets its own lambda, its own cap
and its own schedule rather than a shared normalized weight.

Why both, given r5 and r9:

* r9 (MAE probe alone, 20 classes) worked: held-out ResNet rank 505 -> 171,
  probe top-1 0 -> 0.071. But its own training judge reached top-1 0.26 over
  the same samples, and that 0.26-vs-0.071 gap is the signature of partial
  white-box gaming of the probe's decision boundary.
* r5 (Qwen VLM alone, 1000 classes) failed: vlm_p_yes_target went 0.0014 ->
  0.0003 over 100k steps while its gradient dominated FD (ratio 3.8 -> 1.2).
  A binary judge staring at unrecognizable early samples has no usable
  direction to give.

So the VLM is not a replacement for the MAE probe, it is a second opinion from
a different architecture and a different training paradigm -- the one thing
that makes the probe harder to game -- and it only becomes useful once the
probe has pulled the generator into a regime where the images are recognizable
at all. Hence `--vlm_warmup_steps`: the MAE term runs from step 0 with r9's
proven settings, and the VLM phases in later.

The held-out ResNet-50 (`--cond_probe`) stays out of the loss and remains the
only honest conditioning meter.
"""

import argparse
import datetime
import logging
import math
import os
import sys
import time

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
from frechet_distance.losses import (
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


from clip_classifier import assert_clip_grad_flows
from classifier_ensemble import ProbeClassifier
from mae_linear_probe import MAELinearProbe, load_probe_checkpoint
from vlm_judge import BinaryVLMJudge



torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.optimize_ddp = False

logger = logging.getLogger("FD_loss")


# ---------------------------------------------------------------------------
# FD train step
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

def _ramped_lambda(step, peak, warmup, ramp):
    """0 during warmup, then a linear ramp to ``peak`` over ``ramp`` steps."""
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return peak
    return peak * min(1.0, (step - warmup) / float(ramp))


def cond_lambda_schedule(step, args):
    """Effective lambda_cond at ``step``: 0 during warmup, then a linear ramp
    to args.lambda_cond over cond_ramp_steps. Gating the CLIP term until the FD
    queue is warm (and the generator already makes real images) is what stops
    the early CLIP-dominated adversarial collapse."""
    return _ramped_lambda(
        step,
        args.lambda_cond,
        getattr(args, "cond_warmup_steps", 0),
        getattr(args, "cond_ramp_steps", 0),
    )


def vlm_lambda_schedule(step, args):
    """Effective lambda_vlm at ``step``, on its own warmup/ramp.

    Kept separate from the MAE schedule on purpose. The VLM's binary question
    is unanswerable while samples are still unrecognizable -- r5 spent 100k
    steps with p(Yes | target class) below 0.005 -- so it is held at exactly
    zero (and, in fd_train_step, not even evaluated) until the MAE term has
    moved the generator into a regime where a Yes/No judgement means
    something."""
    return _ramped_lambda(
        step,
        getattr(args, "lambda_vlm", 0.0),
        getattr(args, "vlm_warmup_steps", 0),
        getattr(args, "vlm_ramp_steps", 0),
    )


def build_vlm_judge(args):
    """Frozen VLM second opinion, sharing each rank's GPU with the generator."""
    distractor_ids = None
    if args.vlm_distractor_pool == "train_classes":
        if not args.train_class_ids:
            raise ValueError(
                "--vlm_distractor_pool train_classes requires --train_class_ids"
            )
        distractor_ids = list(args.train_class_ids)

    judge = BinaryVLMJudge(
        args.cond_vlm_model,
        num_classes=args.num_classes,
        classnames_file=args.clip_classnames_file,
        device="cuda",
        dtype=args.vlm_dtype,
        question_mode=args.vlm_question_mode,
        max_samples_per_step=args.vlm_samples_per_step,
        microbatch_size=args.vlm_microbatch,
        logp_cap=args.vlm_target_logp,
        question_template=args.vlm_prompt_template,
        distractor_class_ids=distractor_ids,
        attn_implementation=args.vlm_attn_implementation,
    )
    logger.info(
        "[VLMJudge] '%s' | lambda=%g warmup=%d ramp=%d cap=%g | mode=%s "
        "samples/step=%s microbatch=%d dtype=%s | distractors=%s",
        args.cond_vlm_model,
        args.lambda_vlm,
        args.vlm_warmup_steps,
        args.vlm_ramp_steps,
        args.vlm_target_logp,
        args.vlm_question_mode,
        args.vlm_samples_per_step or "all",
        args.vlm_microbatch,
        args.vlm_dtype,
        "all 1000 classes" if distractor_ids is None
        else f"{len(distractor_ids)} training classes",
    )
    return judge


def build_mae_probe(args, judges):
    """Attach the trained head to the exact MAE model already used by FD."""
    if not args.mae_probe_checkpoint:
        raise ValueError(
            "--mae_probe_checkpoint is required when --lambda_cond > 0. "
            "Train the head first with train_mae_linear_probe.py."
        )
    checkpoint = load_probe_checkpoint(args.mae_probe_checkpoint)
    val_top1 = float(checkpoint.get("best_val_top1", float("nan")))
    if not math.isfinite(val_top1):
        raise ValueError(
            "MAE probe checkpoint has no finite best_val_top1; validate it on "
            "real ImageNet before generator training."
        )
    if val_top1 < args.mae_probe_min_val_top1:
        raise ValueError(
            f"MAE probe val top-1 {val_top1:.4f} is below "
            f"--mae_probe_min_val_top1={args.mae_probe_min_val_top1:.4f}"
        )

    matches = []
    for index, judge in enumerate(judges):
        if (
            judge.get("model_name") == checkpoint["model_name"]
            and judge.get("pool_type") == checkpoint["pool_type"]
            and int(judge.get("target_size")) == int(checkpoint["target_size"])
            and int(judge.get("feat_dim")) == int(checkpoint["feature_dim"])
        ):
            matches.append((index, judge))
    if len(matches) != 1:
        available = [
            {
                "model_name": judge.get("model_name"),
                "pool_type": judge.get("pool_type"),
                "target_size": judge.get("target_size"),
                "feature_dim": judge.get("feat_dim"),
            }
            for judge in judges
        ]
        raise ValueError(
            "MAE probe checkpoint must match exactly one FD judge; found "
            f"{len(matches)} matches. checkpoint="
            f"{checkpoint['model_name']}/{checkpoint['pool_type']}/"
            f"{checkpoint['target_size']}px/{checkpoint['feature_dim']}d, "
            f"available={available}"
        )

    index, judge = matches[0]
    temperature = (
        None if args.mae_probe_temperature <= 0
        else args.mae_probe_temperature
    )
    mae_probe = MAELinearProbe.from_checkpoint(
        args.mae_probe_checkpoint,
        judge["model"],
        device="cuda",
        temperature=temperature,
        feature_judge_index=index,
        expected_model_name=judge["model_name"],
        expected_pool_type=judge["pool_type"],
        expected_target_size=judge["target_size"],
        expected_num_classes=args.num_classes,
    )
    logger.info(
        "[MAELinearProbe] checkpoint=%s val_top1=%.4f val_top5=%.4f "
        "judge_index=%d model=%s pool=%s target=%d dim=%d temperature=%.4g",
        args.mae_probe_checkpoint,
        mae_probe.best_val_top1,
        mae_probe.best_val_top5,
        index,
        mae_probe.model_name,
        mae_probe.pool_type,
        mae_probe.target_size,
        mae_probe.feature_dim,
        mae_probe.temperature,
    )
    return mae_probe


def get_fd_train_step(model_wo_ddp, judges, sampling_args, args, tokenizer=None,
                      clip_classifier=None, vlm_judge=None, probe=None):
    fid_norm_eps = args.fd_fid_norm_eps
    batch_size = args.batch_size
    num_classes = args.num_classes
    lambda_cond = args.lambda_cond
    lambda_vlm = getattr(args, "lambda_vlm", 0.0)
    eot_views = getattr(args, "mae_eot_views", 1)
    eot_noise = getattr(args, "mae_eot_noise_std", 0.0)
    eot_crop_min = getattr(args, "mae_eot_crop_min", 1.0)
    vlm_eot_views = getattr(args, "vlm_eot_views", 1)
    vlm_eot_noise = getattr(args, "vlm_eot_noise_std", 0.0)
    vlm_eot_crop_min = getattr(args, "vlm_eot_crop_min", 1.0)
    target_logp = getattr(args, "cond_target_logp", 0.0)  # <0 caps confidence
    if getattr(clip_classifier, "handles_cap", False):
        target_logp = 0.0  # ensemble caps per member; no outer re-clamp
    input_shape = (args.input_channels, args.input_size, args.input_size)
    train_class_ids = validate_train_class_ids(
        getattr(args, "train_class_ids", None), num_classes,
    )
    train_class_ids_tensor = (
        None if train_class_ids is None
        else torch.tensor(train_class_ids, dtype=torch.long, device="cuda")
    )

    def fd_train_step(lambda_cond_eff=None, lambda_vlm_eff=None, diag=False):
        # scalars come in as 0-dim cuda tensors from the loop (constant value =>
        # no torch.compile recompiles across the warmup/ramp); fall back to the
        # python constant if called bare (e.g. the compile warmup before the loop).
        if lambda_cond_eff is None:
            lambda_cond_eff = lambda_cond
        if lambda_vlm_eff is None:
            lambda_vlm_eff = lambda_vlm
        z = torch.randn(batch_size, *input_shape, device="cuda") * args.noise_scale
        # if clip_classifier is not None:
        #     y = torch.randint(0, num_classes, (batch_size,), device="cuda")
        # else:
            # y = None

        # Uniformly sample either all ImageNet labels or an explicit diagnostic
        # subset. The MAE probe remains a frozen 1000-way classifier; only the
        # labels presented to the generator are restricted.
        if train_class_ids_tensor is None:
            y = torch.randint(0, num_classes, (batch_size,), device="cuda")
        else:
            subset_indices = torch.randint(
                0, train_class_ids_tensor.numel(), (batch_size,), device="cuda",
            )
            y = train_class_ids_tensor[subset_indices]
        sampled = model_wo_ddp.sample_images_with_grad(z, y, sampling_args=sampling_args)

        if tokenizer is not None:
            sampled = tokenizer.decode(tokenizer.denormalize_z(sampled))
        sampled = (sampled * 0.5 + 0.5).clamp(0,1)  # [-1,1] -> [0,1]

        loss = torch.tensor(0.0, device="cuda")
        loss_dict = {}

        # The VLM is evaluated before the FD judge graphs are built. It takes
        # image-space VJPs in microbatches and injects an exact first-order
        # surrogate, so its activations are already freed when the three FD
        # feature extractors run and the two never coexist on the GPU.
        mae_term_loss = None
        vlm_term_loss = None
        if vlm_judge is not None:
            lambda_value = (float(lambda_vlm_eff.detach())
                            if torch.is_tensor(lambda_vlm_eff)
                            else float(lambda_vlm_eff))
            if lambda_value > 0.0:  # skip the expensive VLM during its warmup
                l_vlm, log_p_vlm = vlm_judge.conditional_loss(
                    sampled, y, eot_views=vlm_eot_views,
                    noise_std=vlm_eot_noise, crop_min=vlm_eot_crop_min)
                vlm_term_loss = lambda_vlm_eff * l_vlm
                loss = loss + vlm_term_loss
                loss_dict["l_vlm"] = float(l_vlm.detach())
                loss_dict["vlm_loss_weighted"] = float(vlm_term_loss.detach())
                loss_dict.update(getattr(vlm_judge, "last_stats", {}))

        # Keep local features for the MAE head. FD still uses the gathered
        # features, while classification must pair local features with local y.
        local_new_feats = []
        all_new_feats = []
        for judge in judges:
            feats = extract_judge_features(judge, sampled)
            local_new_feats.append(feats)
            new_feats = diff_all_gather(feats)
            all_new_feats.append(new_feats)

        fd_term = torch.zeros((), device="cuda")
        fd_raw_sum = 0.0
        for i, judge in enumerate(judges):
            new_feats = all_new_feats[i]

            _ns_kwargs = dict(sigma_ref_sqrt=judge.get("sigma_ref_sqrt"))
            if judge["queue"].online_accum or judge["queue"].ema_stats:
                mu, sigma = judge["queue"].build_feats_stats(new_feats)
                fid = compute_frechet_distance_loss(judge["mu_ref"], judge["sigma_ref"],
                                                    mu=mu, sigma=sigma,
                                                    **_ns_kwargs)
            else:
                all_feats = judge["queue"].build_feats_snapshot(new_feats)
                fid = compute_frechet_distance_loss(judge["mu_ref"], judge["sigma_ref"],
                                                    all_feats=all_feats,
                                                    **_ns_kwargs)
            fid_loss = fid / (fid.detach() + fid_norm_eps)
            fd_term = fd_term + judge["weight"] * fid_loss
            fd_raw_sum += float(fid.detach())
            loss_dict[f"fid_{judge['name']}"] = float(fid.detach())
        loss = loss + fd_term
        loss_dict["fd_loss_raw"] = fd_raw_sum                  # raw Fréchet distance(s), summed
        loss_dict["fd_loss_norm"] = float(fd_term.detach())   # self-normalized FD term in the loss

        # -- frozen MAE linear-probe correction: -E[log p(y|x)] --
        if clip_classifier is not None:
            reuse_idx = getattr(clip_classifier, "feature_judge_index", None)
            can_reuse = (
                reuse_idx is not None
                and eot_views == 1
                and eot_noise == 0.0
                and eot_crop_min == 1.0
            )
            if can_reuse:
                if not 0 <= int(reuse_idx) < len(local_new_feats):
                    raise IndexError(
                        f"MAE feature_judge_index={reuse_idx} is invalid for "
                        f"{len(local_new_feats)} FD judges"
                    )
                log_p_c = clip_classifier.log_p_c_given_features(
                    local_new_feats[int(reuse_idx)], y
                )
                loss_dict["cond_feature_reuse"] = 1.0
            else:
                # EOT views differ from the clean FD input and require their own
                # MAE forwards through the same shared, frozen backbone.
                log_p_c = clip_classifier.log_p_c_given_x(
                    sampled, y, eot_views=eot_views,
                    noise_std=eot_noise, crop_min=eot_crop_min,
                )
                loss_dict["cond_feature_reuse"] = 0.0
            # confidence cap: once a sample is this confident it stops pushing
            p_term = torch.clamp(log_p_c, max=target_logp) if target_logp < 0 else log_p_c
            l_cond = -p_term.mean()  # flip sign since log-probs are <= 0
            mae_term_loss = lambda_cond_eff * l_cond
            loss = loss + mae_term_loss  # mult with lambda cond eff and add to loss
            loss_dict["l_cond"] = float(l_cond.detach())
            loss_dict["logp_c"] = float(log_p_c.mean().detach())
            loss_dict["p_loss_weighted"] = float(mae_term_loss.detach())
            # per-member logp (ensemble only) — divergence between members is
            # the signature of one classifier being adversarially satisfied
            loss_dict.update(getattr(clip_classifier, "last_stats", {}))

        # The combined semantic pull, for the historical grad_x_p / cos /
        # grad_ratio_p_fd keys that the plotting scripts already read.
        if mae_term_loss is None:
            p_term_loss = vlm_term_loss
        elif vlm_term_loss is None:
            p_term_loss = mae_term_loss
        else:
            p_term_loss = mae_term_loss + vlm_term_loss

        # -- per-term gradient diagnostics on the generated pixels (log-steps only).
        # Loss *values* are not comparable across terms (FD is self-normalized, p
        # is not), so we read who actually drives the update from grad-w.r.t.-image
        # norms; the cosine exposes conflict between the CLIP pull and the FD pull.
        if diag:
            def _grad_x(term):
                if term is None:
                    return None
                g = torch.autograd.grad(term, sampled, retain_graph=True,
                                        allow_unused=True)[0]
                return None if g is None else g.detach().reshape(-1)
            g_fd, g_p = _grad_x(fd_term), _grad_x(p_term_loss)
            g_mae, g_vlm = _grad_x(mae_term_loss), _grad_x(vlm_term_loss)
            def _norm(g):
                return float(g.norm()) if g is not None else 0.0
            def _cos(a, b):
                if a is None or b is None:
                    return 0.0
                d = float(a.norm() * b.norm())
                return float(a @ b) / d if d > 0 else 0.0
            loss_dict["grad_x_fd"] = _norm(g_fd)
            loss_dict["grad_x_p"] = _norm(g_p)
            # Each judge separately: the two lambdas are on different scales
            # (1000-way softmax vs binary margin, all B samples vs K), so the
            # pooled number cannot say which one is actually steering.
            loss_dict["grad_x_mae"] = _norm(g_mae)
            loss_dict["grad_x_vlm"] = _norm(g_vlm)
            loss_dict["cos_update_fd_p"] = _cos(g_fd, g_p)
            # Do the two judges agree about where to move the pixels? Sustained
            # values near or below zero mean they are asking for different
            # things, and one of them is being satisfied adversarially.
            loss_dict["cos_update_mae_vlm"] = _cos(g_mae, g_vlm)
            loss_dict["cos_update_fd_mae"] = _cos(g_fd, g_mae)
            loss_dict["cos_update_fd_vlm"] = _cos(g_fd, g_vlm)
            # The single number that says whether lambda_cond is in a useful
            # range: how much of the image-space pull comes from conditioning
            # vs realism. << 1 means the conditional term is a rounding error.
            fd_norm = loss_dict["grad_x_fd"]
            loss_dict["grad_ratio_p_fd"] = (
                loss_dict["grad_x_p"] / fd_norm if fd_norm > 0 else 0.0
            )
            # Per-judge budget: these are what the two lambdas are tuned on.
            loss_dict["grad_ratio_mae_fd"] = (
                loss_dict["grad_x_mae"] / fd_norm if fd_norm > 0 else 0.0
            )
            loss_dict["grad_ratio_vlm_fd"] = (
                loss_dict["grad_x_vlm"] / fd_norm if fd_norm > 0 else 0.0
            )

            # -- label sensitivity: does the generator use y at all? --
            # Re-sample the *same* noise under rolled labels. cond_delta is the
            # relative pixel change caused purely by swapping the class token.
            # ~0 means the model is still effectively unconditional, in which
            # case no conditional-loss tuning can be measured downstream. roll
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
                # Fraction of label pairs that actually differ, so cond_delta is
                # not read as 0 when a rolled batch happened to repeat a label.
                loss_dict["cond_delta_valid"] = float((y_alt != y).float().mean())

            # -- noise sensitivity: intra-class diversity, the collapse meter --
            # The mirror image of cond_delta: hold the label fixed and swap the
            # noise. This is the number r11 needed and did not have. r11 drove
            # held-out probe top-1 to 0.43 while every noise draw for a class
            # produced the same picture; measured off the saved grids, cross-
            # noise spread fell 0.80 -> 0.10 and nothing in the training log
            # showed it. The in-loss FD cannot: with fd_ema_beta 0.999 its
            # covariance is accumulated over ~1000 steps, so a collapsed mode
            # that drifts still looks diverse (in-loss inception FD 12.5 while
            # the frozen-checkpoint eval read 120.7).
            #
            # z is rolled rather than freshly drawn: rows of z are already iid,
            # so the roll gives each label an independent noise vector without
            # perturbing the global RNG stream -- same reason cond_delta rolls
            # labels instead of calling randperm.
            with torch.no_grad():
                z_alt = torch.roll(z, 1, dims=0)
                alt_noise = model_wo_ddp.sample_images_with_grad(
                    z_alt, y, sampling_args=sampling_args)
                if tokenizer is not None:
                    alt_noise = tokenizer.decode(
                        tokenizer.denormalize_z(alt_noise))
                alt_noise = (alt_noise * 0.5 + 0.5).clamp(0, 1)
                ref = sampled.detach()
                delta = (ref - alt_noise).flatten(1).norm(dim=1)
                scale = ref.flatten(1).norm(dim=1).clamp_min(1e-8)
                loss_dict["noise_delta"] = float((delta / scale).mean())

            if probe is not None:
                # held-out eval judge on this training batch — the honest
                # conditioning meter; the in-loss classifiers' logp is not
                loss_dict.update(probe.stats(sampled, y))

        loss.backward(create_graph=False)

        if torch.distributed.is_initialized():
            for p in model_wo_ddp.parameters():
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

        for i, judge in enumerate(judges):
            judge["queue"].enqueue(all_new_feats[i].detach())

        return loss, loss_dict

    if args.compile:
        from utils.runtime_util import _warmup
        logger.info("[Compilation] Compiling fd_train_step ...")
        t0 = time.perf_counter()
        fd_train_step = torch.compile(fd_train_step)
        # warm up with a 0-dim tensor scalar so the compiled graph matches the
        # training loop's call signature (the value varies at runtime, no recompile)
        _zero = torch.zeros((), device="cuda")
        _warmup(lambda: fd_train_step(_zero, _zero), n=2)
        logger.info(f"[Compilation] fd_train_step compiled in {time.perf_counter() - t0:.2f}s")

    return fd_train_step


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_and_evaluate(args):
    args.train_class_ids = validate_train_class_ids(
        getattr(args, "train_class_ids", None), args.num_classes,
    )
    if args.lambda_vlm > 0 and args.compile:
        raise ValueError(
            "--lambda_vlm is incompatible with --compile: the VLM judge makes "
            "eager processor calls and takes explicit image VJPs"
        )
    if args.lambda_vlm <= 0 and args.lambda_cond <= 0:
        logger.warning(
            "[Dual] both lambda_cond and lambda_vlm are 0 -- this is a pure "
            "FD run with no conditional correction at all"
        )
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

    extra = ckpt_resume(args, model_wo_ddp, optimizer, ema_model,
                        extra_keys=["fd_queue_states"])

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
        sigma_ref_sqrt = None
        if args.fd_eigvalsh:
            sigma_ref_sqrt = precompute_sigma_ref_sqrt(sigma_ref)
        judges.append({
            "name": short, "model": repr_model,
            "model_name": name,
            "target_size": getattr(repr_model, "target_size", ts),
            "feat_dim": feat_dim,
            "pool_type": pool_type,
            "mu_ref": mu_ref, "sigma_ref": sigma_ref,
            "sigma_ref_sqrt": sigma_ref_sqrt,
            "queue": queue, "weight": weight,
        })
        eig_mode = "eigvalsh" if args.fd_eigvalsh else "eigvals"
        stats_mode = f"ema(beta={args.fd_ema_beta})" if args.fd_ema_beta > 0 else ("online_accum" if args.fd_online_accum else "snapshot")
        logger.info(f"[FD] Repr '{short}' ({name}): feat_dim={feat_dim}, "
                     f"weight={weight}, pool={pool_type}, stats={stats_path}, "
                     f"eig_mode={eig_mode}, stats_mode={stats_mode}")

    fd_restored = (extra is not None
                   and "fd_queue_states" in extra
                   and load_fd_queue_states(judges, extra["fd_queue_states"]))
    if fd_restored:
        logger.info("[FD] Restored all queue states from checkpoint — skipping queue fill")
        run_sanity_check(judges, args.queue_size, args=args)
    else:
        logger.info(f"[FD] Filling {len(judges)} feature queue(s) "
                    f"({args.queue_size} entries each) ...")
        fill_all_queues(judges, model_wo_ddp, args, tokenizer=tokenizer)
        run_sanity_check(judges, args.queue_size, args=args)

    del extra
    torch.distributed.barrier()

    model.train()
    args.input_channels = model_wo_ddp.in_channels
    args.input_size = model_wo_ddp.input_size

    # -- supervised MAE linear-probe correction --
    clip_classifier = None
    vlm_judge = None
    probe = None
    if args.lambda_cond > 0:
        clip_classifier = build_mae_probe(args, judges)
        smoke = clip_classifier.smoke_test_uniform(image_size=args.input_size)
        logger.info(
            "[MAELinearProbe] smoke: mean log p/class=%.3f "
            "(uniform reference %.3f), entropy=%.3f",
            smoke["mean_logp_per_class"],
            smoke["uniform_ref"],
            smoke["entropy"],
        )
        logger.info(
            "[MAELinearProbe] L_cond enabled: lambda=%g warmup=%d ramp=%d "
            "cap=%g eot=%d noise=%g crop_min=%g",
            args.lambda_cond,
            args.cond_warmup_steps,
            args.cond_ramp_steps,
            args.cond_target_logp,
            args.mae_eot_views,
            args.mae_eot_noise_std,
            args.mae_eot_crop_min,
        )
    else:
        logger.info("[MAELinearProbe] lambda_cond=0 -> conditional correction disabled")

    # -- frozen VLM second opinion --
    if args.lambda_vlm > 0:
        if not args.cond_vlm_model:
            raise ValueError(
                "--lambda_vlm > 0 requires --cond_vlm_model (a local VLM path)"
            )
        vlm_judge = build_vlm_judge(args)
        if args.vlm_skip_smoke:
            logger.info("[VLMJudge] startup image-gradient smoke test skipped")
        else:
            smoke = vlm_judge.smoke_test_uniform(image_size=args.input_size)
            logger.info(
                "[VLMJudge] smoke: mean binary log p(correct)=%.3f "
                "(uniform %.3f), p(correct)=%.3f",
                smoke["mean_logp_per_class"],
                smoke["uniform_ref"],
                smoke["p_correct"],
            )
        if clip_classifier is None:
            logger.warning(
                "[Dual] the VLM is running without the MAE probe. r5 showed a "
                "binary judge alone does not move held-out accuracy; this is "
                "only meaningful as a deliberate ablation."
            )
    else:
        logger.info("[VLMJudge] lambda_vlm=0 -> VLM second opinion disabled")

    if args.cond_probe and (args.lambda_cond > 0 or args.lambda_vlm > 0):
        probe = ProbeClassifier(device="cuda")
        logger.info(
            "[Probe] held-out ResNet-50 logs top-1/top-5/logp/rank; "
            "it is never included in the loss"
        )

    # -- FD train step closure --
    sampling_args = {
        "t_min": args.interval_min,
        "t_max": args.interval_max,
        "cfg": args.cfg,
        "num_steps": args.num_sampling_steps,
    }

    if (clip_classifier is not None and args.mae_probe_grad_check
            and not getattr(clip_classifier, "is_vlm_judge", False)):
        assert_clip_grad_flows(clip_classifier, model_wo_ddp, args,
                               sampling_args, tokenizer=tokenizer)

    fd_train_step = get_fd_train_step(
        model_wo_ddp, judges, sampling_args, args, tokenizer=tokenizer,
        clip_classifier=clip_classifier, vlm_judge=vlm_judge, probe=probe,
    )

    # -- training loop --
    logger.info(f"training from step {args.current_step:,} -> {args.total_steps:,} "
                f"({args.start_epoch} -> {args.epochs} epochs)")

    global_bsz = args.batch_size * args.world_size
    collapse_strikes = 0
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

    # metric logger
    metric_file = os.path.join(args.log_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metric_file, prefetch=True)
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

        lambda_eff = cond_lambda_schedule(step, args) if clip_classifier is not None else 0.0
        lambda_vlm_eff = vlm_lambda_schedule(step, args) if vlm_judge is not None else 0.0
        # pass scalar as a 0-dim tensor so the compiled step does not recompile
        # as the warmup/ramp changes its value
        lambda_t = torch.as_tensor(lambda_eff, device="cuda", dtype=torch.float32)
        lambda_vlm_t = torch.as_tensor(lambda_vlm_eff, device="cuda",
                                       dtype=torch.float32)
        # per-term grad diagnostics cost extra partial backwards through the
        # frozen judges — only on log-steps, and not under torch.compile.
        diag = (not args.compile) and (step % args.print_freq == 0)
        loss, loss_dict = fd_train_step(lambda_cond_eff=lambda_t,
                                        lambda_vlm_eff=lambda_vlm_t, diag=diag)

        if clip_classifier is not None:
            loss_dict["lambda_eff"] = lambda_eff
        if vlm_judge is not None:
            loss_dict["lambda_vlm_eff"] = lambda_vlm_eff

        # Mode-collapse alarm. r11 lost ~88% of its intra-class diversity over
        # 30k steps and every number in this log kept improving while it
        # happened, so the failure has to announce itself or it gets found in
        # the eval a day later. The base checkpoint sits at ~0.75.
        nd = loss_dict.get("noise_delta")
        if nd is not None and nd < args.noise_delta_warn:
            collapse_strikes += 1
            if collapse_strikes % 10 == 1:
                logger.warning(
                    "[step %d] MODE COLLAPSE: noise_delta=%.3f < %.2f "
                    "(base checkpoint ~0.75) — the generator is ignoring z. "
                    "%d consecutive diag steps below threshold. Lower "
                    "--lambda_cond/--lambda_vlm or --fd_ema_beta.",
                    step, nd, args.noise_delta_warn, collapse_strikes,
                )
        elif nd is not None:
            collapse_strikes = 0

        grad_norm = (torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                     if args.grad_clip > 0.0 else get_grad_norm(model.parameters()))

        if torch.isfinite(grad_norm):
            optimizer.step()
            ema_model.step(model)
        else:
            logger.warning(f"[step {step}] NaN/Inf grad_norm — skipping optimizer & EMA update")
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        args.current_step = step + 1
        args.samples_seen += global_bsz

        # timing & metrics
        step_time = time.perf_counter() - step_start
        step_start = time.perf_counter()

        loss_value = all_reduce_mean(loss.item())
        loss_dict = {k: all_reduce_mean(v) for k, v in loss_dict.items()}
        sps = args.batch_size / step_time if step_time > 0 else 0.0
        mem_gb = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        metric_logger.update(
            loss=loss_value, grad_norm=grad_norm,
            lr=optimizer.param_groups[0]["lr"],
            **{"samples/s/device": sps, "samples/s": sps * args.world_size,
               "samples_seen(M)": args.samples_seen / 1e6, "device_mem(GB)": mem_gb},
            **loss_dict,
        )

        # wandb
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

        # checkpoint
        def _save(saver=ckpt_saver):
            elapsed = time.time() - session_start + args.last_elapsed_time
            fd_extra = {"fd_queue_states": save_fd_queue_states(judges)} if judges else {}
            save_checkpoint(args, step, model_wo_ddp, optimizer, ema_model, elapsed,
                            saver=saver, extra=fd_extra)
            torch.distributed.barrier()

        if (args.current_step - last_ckpt_step >= args.save_every
                or args.current_step == args.total_steps):
            _save()
            last_ckpt_step = args.current_step

        if args.milestone_every > 0 and step > 0 and step % args.milestone_every == 0:
            _save()

        # slurm preemption
        if preempt_requested():
            logger.info(f"Preemption at step {args.current_step}: saving checkpoint ...")
            ckpt_saver.wait()
            _save(saver=None)
            logger.info(f"Preemption checkpoint saved at step {args.current_step}. Exiting.")
            return 0

        # visualization
        if args.vis_every > 0 and args.current_step % args.vis_every == 0:
            visualize(args, model_wo_ddp, ema_model, args.current_step, rng=rng, tokenizer=tokenizer)
            model_wo_ddp.train()

        # online evaluation
        if args.eval_every > 0 and args.online_eval and args.current_step % args.eval_every == 0:
            torch.cuda.empty_cache()
            evaluate_all_emas(
                args, model_wo_ddp, ema_model, fid_evaluator, tokenizer,
                step=args.current_step, wandb_logger=wandb_logger,
                cfg=args.cfg, num_images=args.num_images_for_eval_and_search,
            )
            model_wo_ddp.train()

    # -- final --
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
    parser = argparse.ArgumentParser("FD loss fine-tuning with a frozen MAE linear probe")

    # training
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--steps_per_epoch", default=1250, type=int)
    parser.add_argument("--batch_size", default=32, type=int, help="batch size per GPU")
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--same_noise", action="store_true")

    # model architecture
    parser.add_argument("--model", default="pMF_B", type=str)
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
    parser.add_argument("--rf_dropout", type=float, default=0.0,
                        help="dropout in the NCSN++ backbone during FD sampling "
                             "(0.0 = deterministic forward; the pretrained weights "
                             "are dropout-rate agnostic)")
    parser.add_argument("--rf_grad_checkpoint", action="store_true", default=True,
                        help="gradient-checkpoint each Euler ODE step so backprop "
                             "memory is ~O(1) in num_sampling_steps")
    parser.add_argument("--no_rf_grad_checkpoint", action="store_false",
                        dest="rf_grad_checkpoint")

    # tokenizer
    parser.add_argument("--tokenizer", default=None, type=str)
    parser.add_argument("--token_channels", default=3, type=int)
    parser.add_argument("--tokenizer_patch_size", default=1, type=int)

    # optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_sched", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--warmup_rate", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=-1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=0.0, help="gradient clip, 0.0 means no clip")
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
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--cfg", default=4.0, type=float)
    parser.add_argument("--cfg_list", type=float, nargs="+",
                        default=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    parser.add_argument("--interval_min", type=float, default=0.1)
    parser.add_argument("--interval_max", type=float, default=1.0)
    parser.add_argument("--vis_steps", default=[1], type=int, nargs="+")

    # data
    parser.add_argument("--data_path", default="./data/imagenet/train", type=str)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument(
        "--train_class_ids", default=None, type=int, nargs="+",
        help="optional label subset sampled during generator training; the "
             "frozen MAE probe remains num_classes-way",
    )
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
    parser.add_argument("--fid_stats_path", type=str, default="data/fid_stats/guided_diffusion_stats.npz")
    parser.add_argument("--keep_eval_folder", action="store_true")

    parser.add_argument("--save_eval_images", action="store_true")
    parser.add_argument("--cfg_min", default=1.0, type=float)
    parser.add_argument("--cfg_max", default=25.0, type=float)
    parser.add_argument("--overwrite_cache", action="store_true")

    # FD fine-tuning
    parser.add_argument("--queue_size", type=int, default=50000)
    parser.add_argument("--fd_fid_norm_eps", type=float, default=0.01)
    parser.add_argument("--fd_queue_fill_bsz", type=int, default=256)
    parser.add_argument("--fd_repr_models", type=str, nargs="+", default=["inception"],
                        help="feature extractors: 'inception' or timm model names")
    parser.add_argument("--fd_repr_stats_paths", type=str, nargs="+", default=None,
                        help="reference stats (.npz) per repr model; auto-inferred if omitted")
    parser.add_argument("--fd_repr_weights", type=float, nargs="+", default=None,
                        help="per-model FID loss weight (default 1.0 each)")
    parser.add_argument("--fd_repr_pool_types", type=str, nargs="+", default=None,
                        help="pool type per repr model: 'cls' or 'avg' (default 'cls')")
    parser.add_argument("--fd_target_sizes", type=int, nargs="+", default=None,
                        help="per-model target resolution override (default: model's native size)")
    parser.add_argument("--fd_online_accum", action="store_true",
                        help="use online accumulators for FD (avoids cloning 50k queue each step)")
    parser.add_argument("--fd_eigvalsh", action="store_true",
                        help="use eigvalsh on symmetric product instead of eigvals (~8x faster, exact)")
    parser.add_argument("--fd_ema_beta", type=float, default=0.0, metavar="BETA",
                        help="EMA decay for FD stats (0=disabled, use queue). "
                             "Implies online_accum. E.g. 0.999 → ~1000-batch window")

    # frozen supervised MAE linear probe -- additive to marginal FD losses
    parser.add_argument("--lambda_cond", type=float, default=0.0,
                        help="weight on L_cond=-E[log p_MAE(y|x)]; 0 disables it")
    parser.add_argument("--mae_probe_checkpoint", type=str, default=None,
                        help="checkpoint from train_mae_linear_probe.py; required "
                             "when lambda_cond > 0")
    parser.add_argument("--mae_probe_temperature", type=float, default=0.0,
                        help="logit temperature override (>0); 0 uses checkpoint")
    parser.add_argument("--mae_probe_min_val_top1", type=float, default=0.0,
                        help="refuse a checkpoint below this real ImageNet val "
                             "top-1 fraction")
    parser.add_argument("--mae_probe_grad_check", action="store_true",
                        help="at startup, assert MAE log-p gradients reach JiT")
    parser.add_argument("--cond_warmup_steps", type=int, default=0,
                        help="steps with lambda_cond=0 before MAE log-p turns on")
    parser.add_argument("--cond_ramp_steps", type=int, default=0,
                        help="linear ramp steps after the conditional warmup")
    parser.add_argument("--cond_target_logp", type=float, default=0.0,
                        help="per-sample confidence cap (<0); 0 disables it")
    parser.add_argument("--mae_eot_views", type=int, default=1,
                        help="number of augmented MAE views; 1 reuses FD features")
    parser.add_argument("--mae_eot_noise_std", type=float, default=0.0,
                        help="EOT Gaussian noise std in [0,1] image space")
    parser.add_argument("--mae_eot_crop_min", type=float, default=1.0,
                        help="EOT minimum random crop scale; 1 disables crops")
    parser.add_argument("--cond_probe", action="store_true",
                        help="log held-out ResNet-50 metrics; never in the loss")
    parser.add_argument("--noise_delta_warn", type=float, default=0.35,
                        help="warn when intra-class diversity (noise_delta) "
                             "falls below this; the base checkpoint is ~0.75 "
                             "and r11 collapsed to 0.10")

    # frozen VLM second opinion -- a separate additive term, never blended
    # into the MAE log-p. Its own lambda/cap/schedule because a binary Yes/No
    # margin and a 1000-way softmax are not on a common scale.
    parser.add_argument("--lambda_vlm", type=float, default=0.0,
                        help="weight on L_vlm=-E[log p_VLM(correct|x,q)]; "
                             "0 disables the VLM entirely")
    parser.add_argument("--cond_vlm_model", type=str, default="",
                        help="local path to the frozen VLM; required when "
                             "lambda_vlm > 0")
    parser.add_argument("--vlm_target_logp", type=float, default=-0.69,
                        help="per-question VLM confidence cap (<0); separate "
                             "from --cond_target_logp because chance is "
                             "-log(2), not -log(num_classes)")
    parser.add_argument("--vlm_warmup_steps", type=int, default=0,
                        help="steps with lambda_vlm=0 (the VLM is not even "
                             "run) before the second opinion turns on")
    parser.add_argument("--vlm_ramp_steps", type=int, default=0,
                        help="linear ramp steps after the VLM warmup")
    parser.add_argument("--vlm_distractor_pool", type=str, default="all",
                        choices=["all", "train_classes"],
                        help="'train_classes' draws the pairwise distractor "
                             "from --train_class_ids, so the negative question "
                             "is a real discrimination instead of a free 'No'")
    parser.add_argument("--vlm_dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--vlm_microbatch", type=int, default=1,
                        help="images per VLM forward; 1 keeps one activation "
                             "graph resident at a time")
    parser.add_argument("--vlm_samples_per_step", type=int, default=2,
                        help="K generated images scored per rank per step; "
                             "0 means the whole local batch")
    parser.add_argument("--vlm_question_mode", type=str, default="pairwise",
                        choices=["target", "target_only", "pairwise",
                                 "pairwise_balanced"])
    parser.add_argument("--vlm_prompt_template", type=str, default=None,
                        help="question template containing {class_name}")
    parser.add_argument("--vlm_attn_implementation", type=str, default="sdpa")
    parser.add_argument("--vlm_skip_smoke", action="store_true",
                        help="skip the startup image-gradient smoke check")
    parser.add_argument("--vlm_eot_views", type=int, default=1,
                        help="augmented VLM views per question")
    parser.add_argument("--vlm_eot_noise_std", type=float, default=0.0)
    parser.add_argument("--vlm_eot_crop_min", type=float, default=1.0)
    parser.add_argument("--clip_classnames_file", type=str, default=None,
                        help="newline-separated class names for the VLM "
                             "question; defaults to torchvision's ImageNet-1k "
                             "order, which matches the ImageFolder labels")

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
    parser.add_argument("--project", default="One3", type=str)
    parser.add_argument("--entity", default=None, type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--disable_wandb", action="store_false", dest="enable_wandb")


    # new added plotting
    parser.add_argument("--dist_vis_every", type=int, default=500,
                        help="steps between distribution analysis plots (0 = disabled)")
    parser.add_argument("--dist_vis_n", type=int, default=5000,
                        help="features to subsample per judge for vis (default 5000)")
    parser.add_argument("--real_features_paths", type=str, nargs="+", default=None,
                            help="One or more cached real .npy paths, auto-matched to "
                                "judges by feat_dim. Use this when training with multiple "
                                "judges (FD-SIM). Example: "
                                "--real_features_paths "
                                "work_dirs/real_features/inception_ADM/real.npy "
                                "work_dirs/real_features/mae/real.npy "
                                "work_dirs/real_features/siglip/real.npy")


    # system
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--dtype", default="bf16", type=str, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compile", action="store_true")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    sys.exit(train_and_evaluate(args))
