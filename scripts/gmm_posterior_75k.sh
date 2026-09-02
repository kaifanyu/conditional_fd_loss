#!/usr/bin/env bash
# 75k-step post-training of pMF-B/256 with the class-conditional log p / log q
# term on top of FD-inception.
#
# Prerequisite (one pass over ImageNet train, ~4 min on 3 A100s):
#
#   CUDA_VISIBLE_DEVICES=4,5,6 torchrun --nproc_per_node=3 compute_class_stats.py \
#       --model inception --data_path "$DATA_ROOT" --img_size 256 --pca_dim 128
#
# Calibration behind the numbers below (all measured, not guessed):
#   * FD-only grad_norm on this config is ~0.0125. --fd_gmm_weight 0.002 takes
#     the total to ~0.021, so the new term has real influence without swamping
#     the Frechet term. Recalibrate if the model, judge, or batch size changes.
#   * --fd_gmm_lambda_cls 0.1: at parity the class-fidelity term contributes
#     ~92% of the GMM gradient, and it is the contractive one. It is a
#     regulariser here, not the driver.
#   * --fd_gmm_p_shrinkage 0.75: minimises held-out -log p(c|x) (6.89 vs 6.91
#     for a uniform predictor) at 76.7% top-1, and minimises the residual an
#     ideal generator would leave. See scripts/validate_class_gmm.py.
#
# The metric to watch is gmm_within_trace_ratio -- generated within-class
# feature variance over real. It reads 0.89 at initialisation. Falling means
# collapse; rising toward 1.0 means the term is doing its job.

set -euo pipefail

: "${DATA_ROOT:=/data/dataset/imagenet}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${GPUS:=4,5,6}"
: "${MASTER_PORT:=29500}"
: "${BATCH_SIZE:=96}"          # per GPU; 40 GB/GPU compiled, 53 GB uncompiled
: "${EPOCHS:=60}"              # x STEPS_PER_EPOCH = total steps
: "${STEPS_PER_EPOCH:=1250}"
: "${EXP_NAME:=pMF_B_256-fd_inception-gmm}"
: "${ENABLE_WANDB:=0}"
: "${CLASS_STATS:=data/fid_stats/inception_in256_t256_classgmm_k128.npz}"

NUM_GPUS=$(awk -F, '{print NF}' <<< "$GPUS")
WANDB_FLAG=--disable_wandb
[ "$ENABLE_WANDB" = "1" ] && WANDB_FLAG=--enable_wandb

[ -f "$CLASS_STATS" ] || { echo "[ERR] missing $CLASS_STATS; run compute_class_stats.py first"; exit 1; }

echo "[run] $EXP_NAME | GPUs $GPUS ($NUM_GPUS) | global batch $((BATCH_SIZE * NUM_GPUS))"
echo "[run] $((EPOCHS * STEPS_PER_EPOCH)) steps"

CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
    --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    main_fd.py \
    --project fd_gmm --exp_name "$EXP_NAME" \
    --batch_size "$BATCH_SIZE" --data_path "$DATA_ROOT" \
    --load_from "${CKPT_ROOT}/pMF-B_256.pth" \
    --model pMF_B --rope_2d --learned_pe --disable_v_head \
    --cfg 8.5 --interval_min 0.1 --interval_max 0.7 --num_sampling_steps 1 \
    --epochs "$EPOCHS" --steps_per_epoch "$STEPS_PER_EPOCH" --warmup_epochs 5 \
    --lr 1e-6 --lr_sched cosine --min_lr 0.0 \
    --queue_size 50000 --fd_queue_fill_bsz 256 \
    --fd_repr_models inception --fd_eigvalsh --fd_ema_beta 0.999 \
    --fd_gmm --fd_gmm_stats_path "$CLASS_STATS" \
    --fd_gmm_pca_dim 128 --fd_gmm_mode density \
    --fd_gmm_weight 0.002 --fd_gmm_lambda_ent 1.0 --fd_gmm_lambda_cls 0.1 \
    --fd_gmm_p_shrinkage 0.75 --fd_gmm_q_shrinkage 0.25 \
    --fd_gmm_ema_beta 0.999 --fd_gmm_warmup_steps 2000 \
    --online_eval --eval_freq 10 --eval_bsz 128 \
    --num_images_for_eval_and_search 10000 \
    --vis_freq 5 --print_freq 25 --save_freq 5 --milestone_interval 20 \
    --compile --auto_resume "$WANDB_FLAG" \
    "$@"
