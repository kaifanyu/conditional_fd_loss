#!/usr/bin/env bash
# Table 3 (JiT) + CONDITIONAL correction term.
#
# Same setup as table_3_JiT.sh, with two differences:
#   1. runs conditional_main_fd.py instead of main_fd.py
#   2. adds the CLIP conditional-correction term  L_total = L_FD + LAMBDA_COND * L_cond
#
# Everything else (model, base checkpoint, FD judges, EMA, LR, online eval, and
# the periodic visualizations via --vis_freq) is unchanged from table_3_JiT.sh.
#
# Set MODEL_SIZE in {B,L,H}. Conditional knobs are env-overridable below.
#
# NOTE on memory: the conditional term backprops through a frozen CLIP image
# encoder on every step. ViT-L-14 is the best signal but the heaviest; if you
# OOM, set CLIP_MODEL=ViT-B-32 and/or lower GLOBAL_BSZ.

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the ImageNet root with train/ and val/ subdirectories}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=512}"
: "${ENABLE_WANDB:=0}"
: "${MODEL_SIZE:=B}"

# ── Conditional-correction settings ────────────────────────────
: "${LAMBDA_COND:=0}"          # weight on L_cond (NEEDS TUNING; 0 disables)
: "${CLIP_MODEL:=ViT-L-14}"      # ViT-L-14 (best) | ViT-B-32 (cheaper)
: "${CLIP_PRETRAINED:=openai}"
: "${CLIP_DTYPE:=bf16}"
: "${CLIP_SINGLE_TEMPLATE:=0}"   # 1 -> single prompt instead of 80-prompt ensemble
: "${CLIP_GRAD_CHECK:=1}"        # 1 -> assert L_cond grads reach the generator at startup
: "${COMPILE:=0}"                # off by default: conditional term + compile is new
# DistributionAnalyzer runs rank-0-only with NO matching collective; if its
# per-trigger work (eigendecomp + matplotlib + PNG/wandb upload) outlasts the
# 30-min NCCL timeout, the other ranks hang at the next all_gather. Off here.
# ───────────────────────────────────────────────────────────────

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
WANDB_FLAG=--disable_wandb
if [ "$ENABLE_WANDB" = "1" ]; then
    WANDB_FLAG=--enable_wandb
fi

# Assemble the conditional flags.
COND_FLAGS=(
    --lambda_cond "$LAMBDA_COND"
    --clip_model_name "$CLIP_MODEL"
    --clip_pretrained "$CLIP_PRETRAINED"
    --clip_dtype "$CLIP_DTYPE"
)
[ "$CLIP_SINGLE_TEMPLATE" = "1" ] && COND_FLAGS+=(--clip_single_template)
[ "$CLIP_GRAD_CHECK" = "1" ] && COND_FLAGS+=(--clip_grad_check)
[ "$COMPILE" = "1" ] && COND_FLAGS+=(--compile)
echo "[cond] flags: ${COND_FLAGS[*]}"

MAE="vit_large_patch16_224.mae"
SIGLIP="vit_so400m_patch16_siglip_256.v2_webli"

case "${MODEL_SIZE}" in
    B)
        MODEL=JiT_B; CFG=3.0; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-B-uncond.pth" ;;
    L)
        MODEL=JiT_L; CFG=2.4; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-L.pth" ;;
    H)
        MODEL=JiT_H; CFG=2.2; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-H.pth" ;;
    *) echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE}"; exit 1 ;;
esac

run_one() {
    local exp_name="$1"
    shift
    torchrun \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --nproc_per_node="$GPUS_PER_NODE" \
        conditional_main_fd_ponly.py \
        --project JiT_uncond_baseline \
        --exp_name "$exp_name" \
        --batch_size "$BATCH_SIZE" \
        --data_path "$DATA_ROOT" \
        --load_from "$LOAD" \
        --model "$MODEL" --rope_2d --learned_pe --legacy_time_convention \
        --cfg "$CFG" --interval_min "$INTERVAL_MIN" --interval_max "$INTERVAL_MAX" \
        --ema_type edm \
        --num_sampling_steps 1 \
        --eval_bsz 256 --num_images_for_eval_and_search 50000 \
        --vis_freq 2 --online_eval --eval_freq 10 \
        --print_freq 20 --milestone_interval 10 --save_freq 5 \
        --epochs 40 --steps_per_epoch 1250 --warmup_epochs 5 \
        --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
        --fd_eigvalsh --fd_ema_beta 0.99 \
        --auto_resume "$WANDB_FLAG" \
        "${COND_FLAGS[@]}" \
        "$@"
}

run_one "${MODEL}-fd-inception-0.99EMA-uncond${LAMBDA_COND}" --fd_repr_models inception
