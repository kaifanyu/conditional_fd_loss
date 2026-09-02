import argparse
import datetime
import logging
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


from clip_classifier import CLIPClassifier, assert_clip_grad_flows  # conditional correction
from qphi import QPhiTrainer  # q_phi density-ratio head (diversity correction)
from lora_qphi import LoRAQPhi, assert_lora_qphi_grad_flows  # LoRA-CLIP backbone



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

def cond_lambda_schedule(step, args):
    """Effective lambda_cond at ``step``: 0 during warmup, then a linear ramp
    to args.lambda_cond over cond_ramp_steps. Gating the CLIP term until the FD
    queue is warm (and the generator already makes real images) is what stops
    the early CLIP-dominated adversarial collapse."""
    w = getattr(args, "cond_warmup_steps", 0)
    r = getattr(args, "cond_ramp_steps", 0)
    if step < w:
        return 0.0
    if r <= 0:
        return args.lambda_cond
    return args.lambda_cond * min(1.0, (step - w) / float(r))


def qphi_alpha_schedule(step, args):
    """Effective q_phi repulsion strength at ``step``: 0 until the head is warm
    (qphi_warmup_steps), then a linear ramp to args.qphi_alpha over
    qphi_ramp_steps. Lets q_phi fit the current generator — and lets the p_clip
    pull establish class identity first — before the diversity repulsion turns on."""
    w = getattr(args, "qphi_warmup_steps", 0)
    r = getattr(args, "qphi_ramp_steps", 0)
    if step < w:
        return 0.0
    if r <= 0:
        return args.qphi_alpha
    return args.qphi_alpha * min(1.0, (step - w) / float(r))


def get_fd_train_step(model_wo_ddp, judges, sampling_args, args, tokenizer=None,
                      clip_classifier=None, qphi_trainer=None):
    fid_norm_eps = args.fd_fid_norm_eps
    batch_size = args.batch_size
    num_classes = args.num_classes
    lambda_cond = args.lambda_cond
    eot_views = getattr(args, "clip_eot_views", 1)
    eot_noise = getattr(args, "clip_eot_noise_std", 0.0)
    eot_crop_min = getattr(args, "clip_eot_crop_min", 1.0)
    target_logp = getattr(args, "cond_target_logp", 0.0)  # <0 caps confidence
    qphi_alpha = getattr(args, "qphi_alpha", 1.0)          # density-ratio strength
    input_shape = (args.input_channels, args.input_size, args.input_size)

    def fd_train_step(lambda_cond_eff=None, alpha_eff=None):
        # scalars come in as 0-dim cuda tensors from the loop (constant value =>
        # no torch.compile recompiles across the warmup/ramp); fall back to the
        # python constants if called bare (e.g. the compile warmup before the loop).
        if lambda_cond_eff is None:
            lambda_cond_eff = lambda_cond
        if alpha_eff is None:
            alpha_eff = qphi_alpha
        z = torch.randn(batch_size, *input_shape, device="cuda") * args.noise_scale
        y = torch.randint(0, num_classes, (batch_size,), device="cuda")
        sampled = model_wo_ddp.sample_images_with_grad(z, y, sampling_args=sampling_args)

        if tokenizer is not None:
            sampled = tokenizer.decode(tokenizer.denormalize_z(sampled))
        sampled = (sampled * 0.5 + 0.5).clamp(0,1)  # [-1,1] -> [0,1]

        loss = torch.tensor(0.0, device="cuda")
        loss_dict = {}

        all_new_feats = []
        for judge in judges:
            feats = extract_judge_features(judge, sampled)
            new_feats = diff_all_gather(feats)
            all_new_feats.append(new_feats)

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
            loss = loss + judge["weight"] * fid_loss
            loss_dict[f"fid_{judge['name']}"] = float(fid.detach())

        # -- conditional correction: -E[ log p_clip(c|x) - alpha * log q_phi(c|x) ]
        clip_feats = None
        if clip_classifier is not None:
            # log p_clip(c|x):
            log_p_c = clip_classifier.log_p_c_given_x(sampled, y, eot_views=eot_views, noise_std=eot_noise, crop_min=eot_crop_min)  # (B,)
            # confidence cap: once a sample is this confident it stops pushing
            p_term = torch.clamp(log_p_c, max=target_logp) if target_logp < 0 else log_p_c
            l_cond = -p_term.mean()  # flip sign since log-probs are <= 0
            loss = loss + lambda_cond_eff * l_cond  # mult with lamda cond eff and add to loss
            loss_dict["l_cond"] = float(l_cond.detach())
            loss_dict["logp_c"] = float(log_p_c.mean().detach())

            if qphi_trainer is not None:
                if getattr(qphi_trainer, "takes_images", False):
                    log_q_c, q_diag = qphi_trainer.repulsion_with_diag(sampled, y)
                else:
                    # legacy linear head on frozen features
                    clip_feats = clip_classifier.image_features(sampled)
                    log_q_c, q_diag = qphi_trainer.repulsion_with_diag(clip_feats, y)
                loss = loss + lambda_cond_eff * alpha_eff * log_q_c.mean()  # mult with lamda_cond eff and alpha_eff for tuning
                loss_dict["logq_c"] = float(log_q_c.mean().detach())  # post-clamp (feeds loss)
                loss_dict.update(q_diag)                              # raw q_phi posterior reads
                
        loss.backward(create_graph=False)

        if torch.distributed.is_initialized():
            for p in model_wo_ddp.parameters():
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

        for i, judge in enumerate(judges):
            judge["queue"].enqueue(all_new_feats[i].detach())

        if clip_feats is not None:
            clip_feats = clip_feats.detach()  # feed the q_phi replay buffer (no graph)
        # LoRA mode: the replay buffer wants IMAGES (adapter updates invalidate
        # any cached features), so hand back the detached pixels instead. The
        # training loop's observe()/train() calls then work unchanged.
        if qphi_trainer is not None and getattr(qphi_trainer, "takes_images", False):
            return loss, loss_dict, sampled.detach(), y
        return loss, loss_dict, clip_feats, y

    if args.compile:
        from utils.runtime_util import _warmup
        logger.info("[Compilation] Compiling fd_train_step ...")
        t0 = time.perf_counter()
        fd_train_step = torch.compile(fd_train_step)
        # warm up with 0-dim tensor scalars so the compiled graph matches the
        # training loop's call signature (the values vary at runtime, no recompile)
        _zero = torch.zeros((), device="cuda")
        _warmup(lambda: fd_train_step(_zero, _zero), n=2)
        logger.info(f"[Compilation] fd_train_step compiled in {time.perf_counter() - t0:.2f}s")

    return fd_train_step


@torch.enable_grad()
def assert_qphi_grad_flows(clip_classifier, qphi_trainer, model_wo_ddp, args,
                           sampling_args, tokenizer=None):
    """Sanity check for the q_phi repulsion path. A backward through
    +alpha*log q_phi *only* must (a) land a finite, nonzero gradient on the
    generator (grad flows through the image), and (b) leave the q_phi head with
    NO gradient — its params are detached in the generator term, so the generator
    optimizer can never corrupt them. Mirrors assert_clip_grad_flows."""
    model_wo_ddp.zero_grad(set_to_none=True)
    z = torch.randn(2, args.input_channels, args.input_size, args.input_size,
                    device="cuda") * args.noise_scale
    y = torch.randint(0, args.num_classes, (2,), device="cuda")
    sampled = model_wo_ddp.sample_images_with_grad(z, y, sampling_args=sampling_args)
    if tokenizer is not None:
        sampled = tokenizer.decode(tokenizer.denormalize_z(sampled))
    sampled = (sampled * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]      # clamp the sampled

    feats = clip_classifier.image_features(sampled)
    l_rep = qphi_trainer.log_q_repulsion(feats, y).mean()
    l_rep.backward()

    n_gen = sum(
        1 for p in model_wo_ddp.parameters()
        if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
        and p.grad.abs().sum() > 0
    )
    n_head = sum(1 for p in qphi_trainer.head.parameters()
                 if p.grad is not None and p.grad.abs().sum() > 0)
    model_wo_ddp.zero_grad(set_to_none=True)
    qphi_trainer.head.zero_grad(set_to_none=True)
    assert n_gen > 0, (
        "q_phi repulsion produced NO gradient on the generator — image_features "
        "is detached from the generator output (check preprocessing / dtype casts)."
    )
    assert n_head == 0, (
        "q_phi head received gradient from the repulsion term — detach_params "
        "failed; the generator optimizer would corrupt the head."
    )
    logger.info(f"[q_phi] grad-flow OK: L_rep={float(l_rep):.4f}, "
                f"{n_gen} generator tensors got gradient, head correctly frozen.")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_and_evaluate(args):
    wandb_logger = setup(args)
    register_preempt_handler()

    # -- models, optimizer, checkpoint --
    tokenizer = create_tokenizer(args)
    model, ema_model = create_generation_model(args)
    optimizer = create_optimizer(args, model, print_trainable_params=True)
    model_wo_ddp = model

    extra = ckpt_resume(args, model_wo_ddp, optimizer, ema_model,
                        extra_keys=["fd_queue_states", "qphi_state"])

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

    qphi_state = extra.get("qphi_state") if extra is not None else None  # for q_phi resume
    del extra
    torch.distributed.barrier()

    model.train()
    args.input_channels = model_wo_ddp.in_channels
    args.input_size = model_wo_ddp.input_size

    # -- conditional correction (CLIP zero-shot p(c|x)) --
    clip_classifier = None
    if args.lambda_cond > 0:
        clip_classifier = CLIPClassifier(
            model_name=args.clip_model_name,
            pretrained=args.clip_pretrained,
            dtype=args.clip_dtype,
            single_template=args.clip_single_template,
            cache_dir=args.clip_cache_dir,
            classnames_file=args.clip_classnames_file,
            num_classes=args.num_classes,
            device="cuda",
            logit_scale_override=args.clip_logit_scale,
        )
        smoke = clip_classifier.smoke_test_uniform(image_size=args.input_size)
        logger.info(f"[CLIPClassifier] smoke (noise images): mean log p/class="
                    f"{smoke['mean_logp_per_class']:.3f} "
                    f"(uniform ref {smoke['uniform_ref']:.3f})")
        logger.info(f"[CLIPClassifier] L_cond enabled, lambda_cond={args.lambda_cond} "
                    f"| warmup={args.cond_warmup_steps} ramp={args.cond_ramp_steps} "
                    f"| eot_views={args.clip_eot_views} noise={args.clip_eot_noise_std} "
                    f"crop_min={args.clip_eot_crop_min} "
                    f"| target_logp={args.cond_target_logp} "
                    f"logit_scale={args.clip_logit_scale or 'native'}")
    else:
        logger.info("[CLIPClassifier] lambda_cond=0 -> conditional correction disabled")

    # -- q_phi density-ratio head (diversity correction) --
    qphi_trainer = None
    if clip_classifier is not None and args.qphi_lora_enabled:
        assert not args.qphi_enabled, \
            "pick ONE q_phi backbone: --qphi_enabled (linear head) or --qphi_lora_enabled (LoRA-CLIP)"
        qphi_trainer = LoRAQPhi(
            clip_classifier,
            r=args.qphi_lora_r,
            lora_alpha=args.qphi_lora_alpha,
            last_k=args.qphi_lora_last_k,
            dropout=args.qphi_lora_dropout,
            attn=args.qphi_lora_attn,
            output_adapter=not args.qphi_lora_no_output_adapter,
            lr=args.qphi_lora_lr,
            weight_decay=args.qphi_lora_weight_decay,
            label_smoothing=args.qphi_label_smoothing,
            buffer_size=args.qphi_lora_buffer,
            train_batch=args.qphi_lora_batch,
        )
        if qphi_state is not None:
            qphi_trainer.load_state_dict(qphi_state)
        logger.info(f"[q_phi/LoRA] r={args.qphi_lora_r} last_k={args.qphi_lora_last_k} "
                    f"attn={args.qphi_lora_attn} alpha={args.qphi_alpha} "
                    f"buffer={args.qphi_lora_buffer} batch={args.qphi_lora_batch} "
                    f"warmup={args.qphi_warmup_steps} ramp={args.qphi_ramp_steps}")
    elif clip_classifier is not None and args.qphi_enabled:
        qphi_trainer = QPhiTrainer(
            clip_dim=clip_classifier.feat_dim,
            num_classes=args.num_classes,
            hidden_dim=args.qphi_hidden_dim,
            lr=args.qphi_lr,
            weight_decay=args.qphi_weight_decay,
            label_smoothing=args.qphi_label_smoothing,
            buffer_size=args.qphi_buffer_size,
            train_batch=args.qphi_batch,
        )
        if qphi_state is not None:
            qphi_trainer.load_state_dict(qphi_state)
        logger.info(f"[q_phi] enabled, alpha={args.qphi_alpha} hidden={args.qphi_hidden_dim} "
                    f"buffer={args.qphi_buffer_size} batch={args.qphi_batch} "
                    f"steps/iter={args.qphi_steps_per_iter} "
                    f"warmup={args.qphi_warmup_steps} ramp={args.qphi_ramp_steps}")

    # -- FD train step closure --
    sampling_args = {
        "t_min": args.interval_min,
        "t_max": args.interval_max,
        "cfg": args.cfg,
        "num_steps": args.num_sampling_steps,
    }

    if clip_classifier is not None and args.clip_grad_check:
        assert_clip_grad_flows(clip_classifier, model_wo_ddp, args,
                               sampling_args, tokenizer=tokenizer)
    if qphi_trainer is not None and args.clip_grad_check:
        if getattr(qphi_trainer, "takes_images", False):
            assert_lora_qphi_grad_flows(clip_classifier, qphi_trainer, model_wo_ddp,
                                        args, sampling_args, tokenizer=tokenizer)
        else:
            assert_qphi_grad_flows(clip_classifier, qphi_trainer, model_wo_ddp,
                                   args, sampling_args, tokenizer=tokenizer)

    fd_train_step = get_fd_train_step(
        model_wo_ddp, judges, sampling_args, args, tokenizer=tokenizer,
        clip_classifier=clip_classifier, qphi_trainer=qphi_trainer,
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
        alpha_eff = qphi_alpha_schedule(step, args) if qphi_trainer is not None else 0.0
        # pass scalars as 0-dim tensors so the compiled step does not recompile
        # as the warmup/ramp changes their values
        lambda_t = torch.as_tensor(lambda_eff, device="cuda", dtype=torch.float32)
        alpha_t = torch.as_tensor(alpha_eff, device="cuda", dtype=torch.float32)
        loss, loss_dict, clip_feats, y_gen = fd_train_step(
            lambda_cond_eff=lambda_t, alpha_eff=alpha_t)

        # update q_phi on the just-generated features (independent of the gen step;
        # head params got no gradient from the frozen-params repulsion above)
        if qphi_trainer is not None:
            qphi_trainer.observe(clip_feats, y_gen)
            q_metrics = qphi_trainer.train(args.qphi_steps_per_iter)
            loss_dict["qphi_ce"] = q_metrics["qphi/ce"]
            loss_dict["qphi_acc"] = q_metrics["qphi/acc"]
            loss_dict["q_prob_y_buf"] = q_metrics["qphi/q_prob_y_buf"]  # buffer q(y|x) vs fresh
            loss_dict["alpha_eff"] = alpha_eff
        if clip_classifier is not None:
            loss_dict["lambda_eff"] = lambda_eff

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
            if qphi_trainer is not None:
                fd_extra["qphi_state"] = qphi_trainer.state_dict()
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
    parser = argparse.ArgumentParser("FD loss fine-tuning for generation models", add_help=False)

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

    # conditional correction (CLIP zero-shot p(c|x)) — additive to FD-loss
    parser.add_argument("--lambda_cond", type=float, default=0.1,
                        help="weight on the conditional correction L_cond = "
                             "-E[log p(c|x)]. 0 disables it. DEFAULT 0.1 NEEDS TUNING.")
    parser.add_argument("--clip_model_name", type=str, default="ViT-L-14",
                        help="open_clip arch for the zero-shot classifier "
                             "(ViT-L-14 best signal; ViT-B-32 cheaper)")
    parser.add_argument("--clip_pretrained", type=str, default="openai",
                        help="open_clip pretrained tag")
    parser.add_argument("--clip_dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="CLIP compute/param dtype (bf16 to save memory)")
    parser.add_argument("--clip_single_template", action="store_true",
                        help="use 'a photo of a {c}.' instead of the 80-prompt "
                             "OpenAI ensemble (faster startup, ~1.5%% lower acc)")
    parser.add_argument("--clip_cache_dir", type=str, default="data/clip_text_cache",
                        help="dir for the cached (num_classes, d) text embeddings")
    parser.add_argument("--clip_classnames_file", type=str, default=None,
                        help="optional newline-separated class-name override")
    parser.add_argument("--clip_grad_check", action="store_true",
                        help="at startup, assert L_cond gradients reach the generator")

    # -- anti-adversarial guards for the one-sided L_cond (q(c|x) still dropped) --
    parser.add_argument("--cond_warmup_steps", type=int, default=0,
                        help="steps with lambda_cond=0 before the CLIP term turns "
                             "on (let FD sharpen real images first). 0 = on from step 0")
    parser.add_argument("--cond_ramp_steps", type=int, default=0,
                        help="linear ramp length (steps) for lambda_cond after "
                             "warmup. 0 = step on at full strength")
    parser.add_argument("--cond_target_logp", type=float, default=0.0,
                        help="cap per-sample log p(c|x) at this value (<0); once a "
                             "sample is this confident it stops contributing grad, "
                             "preventing the race to ~100%% adversarial confidence. "
                             "0 disables the cap. e.g. -0.69 ~= cap at 50%%")
    parser.add_argument("--clip_logit_scale", type=float, default=0.0,
                        help="override CLIP's logit scale (temperature). 0 = native "
                             "(~99, razor-sharp). Smaller (e.g. 30) flattens the "
                             "softmax -> weaker, less exploitable gradient")
    parser.add_argument("--clip_eot_views", type=int, default=1,
                        help="EOT: average log p(c|x) over this many augmented "
                             "views. 1 = no averaging. >1 multiplies CLIP cost")
    parser.add_argument("--clip_eot_noise_std", type=float, default=0.0,
                        help="EOT: additive Gaussian noise std per view ([0,1] space)")
    parser.add_argument("--clip_eot_crop_min", type=float, default=1.0,
                        help="EOT: min random-resized-crop scale per view (1.0 = off)")

    # q_phi conditional density-ratio correction (diversity fix; needs lambda_cond>0)
    parser.add_argument("--qphi_enabled", action="store_true",
                        help="turn the one-sided CLIP push into a density-ratio "
                             "correction -E[log p_clip - alpha*log q_phi], where "
                             "q_phi is a head trained online on generated samples. "
                             "The +alpha*log q_phi term repels collapsed modes -> "
                             "restores within-class diversity")
    parser.add_argument("--qphi_alpha", type=float, default=1.0,
                        help="density-ratio strength. 1.0 = the exact ratio; start "
                             "lower (~0.5) for a gentler, safer correction")
    parser.add_argument("--qphi_hidden_dim", type=int, default=0,
                        help="q_phi head width. 0 = linear probe on CLIP features "
                             "(recommended, cleanest ratio); >0 adds one GELU layer")
    parser.add_argument("--qphi_lr", type=float, default=1e-3)
    parser.add_argument("--qphi_weight_decay", type=float, default=1e-2,
                        help="regularizes q_phi so it does not get razor-sharp "
                             "(which would make the repulsion gradient too spiky)")
    parser.add_argument("--qphi_label_smoothing", type=float, default=0.1)
    parser.add_argument("--qphi_buffer_size", type=int, default=50000,
                        help="replay buffer of recent (feature,label); the ring "
                             "buffer forgets stale feats as the generator drifts")
    parser.add_argument("--qphi_batch", type=int, default=256,
                        help="minibatch size for each q_phi SGD step")
    parser.add_argument("--qphi_steps_per_iter", type=int, default=1,
                        help="q_phi SGD steps per generator step. >1 if q_phi lags; "
                             "fewer keeps it lagging (more stable, less GAN-like)")
    parser.add_argument("--qphi_warmup_steps", type=int, default=0,
                        help="train q_phi but hold the repulsion (alpha) at 0 for "
                             "this many steps so the head is fit before it pushes. "
                             "RECOMMENDED > 0 (e.g. a few hundred steps)")
    parser.add_argument("--qphi_ramp_steps", type=int, default=0,
                        help="linear ramp length for alpha after qphi_warmup_steps")

    parser.add_argument("--qphi_lora_enabled", action="store_true",
                        help="q_phi = the CLIP tower itself + LoRA adapters, "
                             "head = the SAME frozen text prototypes/temperature "
                             "as p_clip (q==p exactly at init). Mutually "
                             "exclusive with --qphi_enabled.")
    parser.add_argument("--qphi_lora_r", type=int, default=8)
    parser.add_argument("--qphi_lora_alpha", type=float, default=16.0)
    parser.add_argument("--qphi_lora_last_k", type=int, default=6,
                        help="adapt only the last K transformer blocks (24 in "
                             "ViT-L); shrinks adapter-train backward cost and "
                             "keeps early features shared with p")
    parser.add_argument("--qphi_lora_attn", action="store_true",
                        help="also LoRA the packed q/v attention projections "
                             "(functional re-dispatch; MLP-only is the default)")
    parser.add_argument("--qphi_lora_no_output_adapter", action="store_true")
    parser.add_argument("--qphi_lora_dropout", type=float, default=0.0)
    parser.add_argument("--qphi_lora_lr", type=float, default=1e-4,
                        help="adapters sit inside a 300M tower — 1e-3 (the old "
                             "linear-head lr) is too hot here")
    parser.add_argument("--qphi_lora_weight_decay", type=float, default=0.0)
    parser.add_argument("--qphi_lora_buffer", type=int, default=20000,
                        help="IMAGE replay buffer, uint8@224 on CPU "
                             "(~2.8 GB host RAM per rank at 20k)")
    parser.add_argument("--qphi_lora_batch", type=int, default=64,
                        help="adapter-train batch; the main extra-cost knob "
                             "(full fwd + last-K bwd through ViT-L per step)")

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