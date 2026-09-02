#!/usr/bin/env bash
# Unconditional JiT baseline, FD loss ONLY, with 3 judges (FD-SIM) instead of
# inception alone.
#
# Same setup as table_3_JiT_conditional.sh with LAMBDA_COND=0, except the FD
# term is a self-normalized sum over three frozen judges:
#     L = FD_siglip + FD_mae + FD_inception
# No CLIP / conditional-correction term is built at all (lambda_cond=0 leaves
# clip_classifier=None), so this is a pure FD-loss run.
#
# Set MODEL_SIZE in {B,L,H}.
#
# NOTE on memory: every step backprops the generated batch through all three
# frozen judges (SigLIP SO400M + MAE ViT-L + InceptionV3), not just inception.
# GLOBAL_BSZ is therefore lowered from 1024 to 512; raise it back if you have
# the headroom.

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the ImageNet root with train/ and val/ subdirectories}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=128}"
: "${ENABLE_WANDB:=0}"
: "${MODEL_SIZE:=B}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
WANDB_FLAG=--disable_wandb
if [ "$ENABLE_WANDB" = "1" ]; then
    WANDB_FLAG=--enable_wandb
fi

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
        --project JiT_uncond_baseline_logp \
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
        --epochs 80 --steps_per_epoch 1250 --warmup_epochs 5 \
        --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
        --fd_eigvalsh --fd_ema_beta 0.999 \
        --lambda_cond 3e-4 \
        --auto_resume "$WANDB_FLAG" \
        "$@"
}

# Reference stats are auto-inferred from (model name, target size) and all three
# already exist under data/fid_stats/:
#   siglip    -> vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz
#   mae       -> vit_large_patch16_224_mae_in256_t224_stats.npz
#   inception -> --fid_stats_path (guided_diffusion_stats.npz)
# The inception target size (256) is only used for path inference; the inception
# extractor always runs at its native 299.
run_one "${MODEL}-fd-sim-uncond" \
    --fd_repr_models "$SIGLIP" "$MAE" inception \
    --fd_repr_pool_types cls cls cls \
    --fd_target_sizes 224 224 256
