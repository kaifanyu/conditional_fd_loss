#!/usr/bin/env bash
# CIFAR-10 conditional FD-Loss distillation (p-only) onto the UNCONDITIONAL
# 1-rectified-flow checkpoint (gnobitab/RectifiedFlow, score_sde NCSN++).
#
# Idea: the base model has no class input. denoiser_rf.RFDenoiser adds a
# zero-init class embedding into the NCSN++ time embedding, so at step 0 the
# model == the pretrained unconditional model. We then fine-tune the WHOLE model
# with FD-Loss (keeps images on the CIFAR manifold) + a small one-sided CLIP
# correction L_cond = -E[log p(c|x)] that carves out per-class behavior.
#
# Recipe mirrors the first clean conditional success (JiT_H cond3e-4):
#   * small lambda_cond (3e-4), constant (no warmup/ramp needed when small)
#   * NO qphi repulsion (this is the *_ponly entrypoint — there is none)
#   * cond_target_logp = 0.0 (cap at the natural ceiling)
#   * keep FD self-normalization
# All knobs are env-overridable; defaults are a starting point — tune.
#
# PREREQUISITE — CIFAR-10 Inception reference stats (NOT shipped):
#   The FD loss + evaluator need mu/sigma computed on CIFAR-10 with InceptionV3.
#   Point FID_STATS / FD_STATS at a precomputed .npz, e.g. generate via
#   compute_repr_stats.py over CIFAR-10 laid out as an ImageFolder:
#     torchrun --nproc_per_node=1 compute_repr_stats.py \
#       --model inception --data_path /path/to/cifar10_imagefolder \
#       --img_size 32 --target_size 299 --output_name cifar10_inception_stats.npz
#   (or drop in standard CIFAR-10 FID stats renamed to the path below.)

set -euo pipefail

: "${CKPT_ROOT:=./checkpoints/base}"
: "${LOAD:=${CKPT_ROOT}/cifar_10_base.pth}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=1}"
: "${GLOBAL_BSZ:=128}"
: "${ENABLE_WANDB:=0}"
: "${OUTPUT_DIR:=work_dirs}"
: "${RUN_NAME:=CIFAR_FD_ONLY}"
: "${AUTO_RESUME:=0}"

# CIFAR-10 Inception reference stats (see PREREQUISITE above).
: "${FID_STATS:=data/fid_stats/cifar10_inception_stats.npz}"
: "${FD_STATS:=${FID_STATS}}"
: "${CLASSNAMES:=data/cifar10_classnames.txt}"

# ── sampling: 1-rectified-flow needs a real multi-step Euler ODE ───────────
# Backprop runs through every step; each step is gradient-checkpointed
# (--rf_grad_checkpoint, on by default) so memory stays ~flat in step count.
: "${NUM_SAMPLING_STEPS:=1}"

# ── conditional-correction (one-sided CLIP) — JiT_H success recipe ─────────
: "${LAMBDA_COND:=0}"
: "${CLIP_MODEL:=ViT-L-14}"      # openai tag -> auto QuickGELU; ~95% zero-shot on CIFAR-10
: "${CLIP_PRETRAINED:=openai}"
: "${CLIP_DTYPE:=bf16}"
: "${COND_WARMUP_STEPS:=0}"
: "${COND_RAMP_STEPS:=0}"
: "${COND_TARGET_LOGP:=0.0}"     # cap at natural ceiling; <0 to cap confidence
: "${CLIP_LOGIT_SCALE:=0}"       # 0=native(~99). Try 30 if 32px samples get exploited
: "${CLIP_EOT_VIEWS:=1}"
: "${CLIP_EOT_NOISE_STD:=0.0}"
: "${CLIP_EOT_CROP_MIN:=1.0}"
: "${GRAD_CLIP:=0.0}"            # JiT_H success ran with no clip
: "${CLIP_GRAD_CHECK:=1}"

# ── training schedule ──────────────────────────────────────────────────────
: "${EPOCHS:=80}"
: "${STEPS_PER_EPOCH:=1250}"     # total_steps = EPOCHS * STEPS_PER_EPOCH
: "${LR:=1e-5}"
: "${VIS_FREQ:=1}"               # epochs (vis_every = STEPS_PER_EPOCH * VIS_FREQ)
: "${EVAL_FREQ:=8}"
: "${EVAL_BSZ:=128}"
: "${QUEUE_SIZE:=50000}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
WANDB_FLAG=--disable_wandb
[ "$ENABLE_WANDB" = "1" ] && WANDB_FLAG=--enable_wandb
RESUME_FLAG=
[ "$AUTO_RESUME" = "1" ] && RESUME_FLAG=--auto_resume

EXTRA=()
[ "$CLIP_GRAD_CHECK" = "1" ] && EXTRA+=(--clip_grad_check)

torchrun \
    --nnodes="$NNODES" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPUS_PER_NODE" \
    conditional_main_fd_ponly.py \
    --output_dir "$OUTPUT_DIR" \
    --project "" \
    --exp_name "$RUN_NAME" \
    --model RF_cifar --img_size 32 --num_classes 10 \
    --load_from "$LOAD" \
    --batch_size "$BATCH_SIZE" \
    --num_sampling_steps "$NUM_SAMPLING_STEPS" \
    --noise_scale 1.0 \
    --class_of_interest 0 1 2 3 4 5 6 7 8 9 --force_class_of_interest \
    --ema_type edm \
    --fid_stats_path "$FID_STATS" \
    --fd_repr_models inception --fd_repr_stats_paths "$FD_STATS" \
    --fd_eigvalsh --fd_ema_beta 0.999 \
    --queue_size "$QUEUE_SIZE" \
    --online_eval --eval_freq "$EVAL_FREQ" --eval_bsz "$EVAL_BSZ" \
    --num_images_for_eval_and_search 50000 \
    --vis_freq "$VIS_FREQ" \
    --print_freq 20 --save_freq 5 --milestone_interval 10 \
    --epochs "$EPOCHS" --steps_per_epoch "$STEPS_PER_EPOCH" --warmup_epochs 0 \
    --lr "$LR" --lr_sched cosine --min_lr 0.0 \
    --lambda_cond "$LAMBDA_COND" \
    --clip_model_name "$CLIP_MODEL" --clip_pretrained "$CLIP_PRETRAINED" --clip_dtype "$CLIP_DTYPE" \
    --clip_classnames_file "$CLASSNAMES" \
    --cond_warmup_steps "$COND_WARMUP_STEPS" --cond_ramp_steps "$COND_RAMP_STEPS" \
    --cond_target_logp "$COND_TARGET_LOGP" \
    --clip_logit_scale "$CLIP_LOGIT_SCALE" \
    --clip_eot_views "$CLIP_EOT_VIEWS" --clip_eot_noise_std "$CLIP_EOT_NOISE_STD" --clip_eot_crop_min "$CLIP_EOT_CROP_MIN" \
    --grad_clip "$GRAD_CLIP" \
    $RESUME_FLAG "$WANDB_FLAG" \
    "${EXTRA[@]}"
