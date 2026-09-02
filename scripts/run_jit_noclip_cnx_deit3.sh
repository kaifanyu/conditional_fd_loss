#!/usr/bin/env bash
# Ablation: drop CLIP from the 3-classifier ensemble, keep ConvNeXt + DeiT3.
#
# Identical to table_3_JiT_uncond_fdsim_logp.sh (the SIM + 3-classifier run that
# produced held-out probe 63.5% / FID 21.49) EXCEPT that the CLIP member is
# removed from --cond_classifiers. Everything else is byte-for-byte the same:
#   judges        FD-SIM (siglip + mae + inception), fd_ema_beta 0.999
#   lambda_cond   3e-4, no cond warmup / no ramp
#   caps          -0.69 (50% confidence) per member, combine=mean
#   EOT           views=2, noise=0.05, crop_min=0.8
#   schedule      80 ep x 1250 = 100k steps, lr 1e-5 cosine, LR warmup 5 ep
#   batch         21/GPU x 3 GPUs = 63 global  (matches logp_multiple exactly)
#
# Question being tested: at 98k in the 3-classifier run, ConvNeXt (-0.44) and
# DeiT3 (-0.57) were both ABOVE the -0.69 cap (clamped, zero gradient) while
# CLIP (-0.77) was the only member still pushing. So CLIP was supplying nearly
# all the late-training conditioning signal. Two things to watch:
#   1. does the "object pinned to the left edge" artifact persist without CLIP
#      (-> ensemble-wide) or disappear (-> CLIP-specific)?
#   2. does conditioning stall once both timm members cap out?
# The held-out ResNet-50 probe (--cond_probe) is the honest meter for (2).

set -euo pipefail

: "${DATA_ROOT:=/data/dataset/imagenet}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29700}"
: "${GPUS_PER_NODE:=3}"
: "${GLOBAL_BSZ:=63}"
: "${ENABLE_WANDB:=0}"
: "${MODEL_SIZE:=B}"
# The training deps (diffusers, timm, transformers>=5.5) live in the `fdloss`
# conda env, not base. Pin it so a bare `torchrun` on PATH cannot pick up the
# wrong interpreter.
: "${CONDA_ENV_BIN:=/home/nvidia/miniconda3/envs/fdloss/bin}"
export PATH="${CONDA_ENV_BIN}:${PATH}"

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

"${CONDA_ENV_BIN}/torchrun" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPUS_PER_NODE" \
    conditional_main_fd_ponly.py \
    --project JiT_cond_sweep \
    --exp_name r14_noclip_cnx_deit3 \
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
    --cond_classifiers "timm:convnext_base.fb_in22k_ft_in1k:cap=-0.69,timm:deit3_base_patch16_224.fb_in22k_ft_in1k:cap=-0.69" \
    --cond_combine mean \
    --cond_probe \
    --clip_eot_views 2 --clip_eot_noise_std 0.05 --clip_eot_crop_min 0.8 \
    --fd_repr_models "$SIGLIP" "$MAE" inception \
    --fd_repr_pool_types cls cls cls \
    --fd_target_sizes 224 224 256 \
    --auto_resume "$WANDB_FLAG"
