#!/usr/bin/env bash
# Table 3 (JiT) + CONDITIONAL correction — v2 (anti-adversarial).
#
# Still ONE-SIDED: we keep only +∇_x log p(c|x) and DROP ∇_x log q(c|x), exactly
# like v1. The goal here is to find out whether the one-sided term can work at
# all once the practical failure modes that made v1 collapse are removed:
#
#   v1 collapse mechanism (diagnosed): with native CLIP (logit_scale≈99, GELU
#   bug) applied from step 0 on a cold FD queue, the un-normalized CLIP gradient
#   dominated and the generator drove log p(c|x)->0 ADVERSARIALLY (texture that
#   fools CLIP), wrecking real FID (27 vs 1.5 for the plain run).
#
# v2 guards (all keep the term one-sided):
#   1. QuickGELU CLIP            - auto-selected for openai weights (v1 used the
#                                  wrong GELU activation -> miscalibrated judge)
#   2. warmup + ramp lambda_cond - FD sharpens real images first, then ease in
#   3. EOT views (noise/flip/crop) - adversarial perturbations don't survive aug
#   4. target_logp confidence cap  - stop pushing past a sane confidence
#   5. softened logit_scale        - flatter softmax = weaker per-pixel gradient
#   6. grad clipping               - cap any residual spike
#
# All knobs are env-overridable. Defaults below are a STARTING POINT — tune.

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the ImageNet root with train/ and val/ subdirectories}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=512}"          # smaller than v1 (1024) to leave room for EOT
: "${ENABLE_WANDB:=0}"
: "${MODEL_SIZE:=B}"
: "${OUTPUT_DIR:=work_dirs}"        # base output root
: "${RUN_NAME:=JiT_eot_small_alpha}" # all files land DIRECTLY in $OUTPUT_DIR/$RUN_NAME
: "${AUTO_RESUME:=0}"               # 0 = fresh start (load base ckpt only); 1 = resume latest

# ── Conditional-correction settings (v2) ───────────────────────
: "${LAMBDA_COND:=3e-4}"        # target weight on L_cond (eased in via ramp)
: "${CLIP_MODEL:=ViT-L-14}"     # openai tag -> auto QuickGELU
: "${CLIP_PRETRAINED:=openai}"
: "${CLIP_DTYPE:=bf16}"
# anti-adversarial guards
: "${COND_WARMUP_STEPS:=500}"   # lambda_cond=0 until here (FD warms up first); hedged-short for 20k run
: "${COND_RAMP_STEPS:=500}"     # then linear ramp 0 -> LAMBDA_COND (full strength by step 1000)
: "${COND_TARGET_LOGP:=-0.69}"  # cap per-sample log p(c|x) (~50%); 0 disables
: "${CLIP_LOGIT_SCALE:=30}"     # 0=native(~99). 50 = flatter, less exploitable
: "${CLIP_EOT_VIEWS:=2}"        # EOT averaging views (>1 multiplies CLIP cost)
: "${CLIP_EOT_NOISE_STD:=0.05}" # additive noise per view
: "${CLIP_EOT_CROP_MIN:=0.85}"  # min random-resized-crop scale per view
: "${GRAD_CLIP:=1.0}"           # 0 disables; v1 ran with no clip
: "${CLIP_GRAD_CHECK:=1}"
: "${COMPILE:=0}"
: "${DIST_VIS_EVERY:=0}"
# q_phi diversity-repulsion schedule. With the floor removed, this warmup is the
# ONLY guard keeping a not-yet-trained (clueless) q_phi from injecting a garbage
# repulsion gradient: hold alpha=0 until q_phi tracks the generator, then ramp in.
: "${QPHI_ALPHA:=0.5}"          # repulsion strength (weighed below CLIP attraction)
: "${QPHI_PRETRAIN_STEPS:=0}"   # DISABLED: base-image pretrain goes stale within ~40 gen steps
                                # (fresh acc 0.22->0.02), so skip it and learn q_phi online instead
: "${QPHI_STEPS_PER_ITER:=8}"   # q_phi SGD steps per generator step (raised 4->8 to track better)
: "${QPHI_WARMUP_STEPS:=500}"   # alpha=0 here (q_phi fits + p_clip sets identity first)
: "${QPHI_RAMP_STEPS:=1500}"    # then linear ramp alpha 0 -> qphi_alpha (full by ~step 2000)
# NOTE: vis_freq/eval_freq are in EPOCHS (vis_every = steps_per_epoch*vis_freq).
# v1 used 100/10000 -> never fired in 125k steps. Use small values so the run
# self-reports the verdict: viz every epoch (1250 steps), FID-50k at step 10000.
: "${VIS_FREQ:=1}"              # viz every 1 epoch = 1250 steps
: "${EVAL_FREQ:=8}"            # FID-50k every 8 epochs = 10000 steps
: "${EVAL_BSZ:=128}"          # smaller than v1 (256) for shared-GPU eval safety
# ───────────────────────────────────────────────────────────────

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
WANDB_FLAG=--disable_wandb
[ "$ENABLE_WANDB" = "1" ] && WANDB_FLAG=--enable_wandb
RESUME_FLAG=
[ "$AUTO_RESUME" = "1" ] && RESUME_FLAG=--auto_resume

COND_FLAGS=(
    --lambda_cond "$LAMBDA_COND"
    --clip_model_name "$CLIP_MODEL"
    --clip_pretrained "$CLIP_PRETRAINED"
    --clip_dtype "$CLIP_DTYPE"
    --clip_logit_scale "$CLIP_LOGIT_SCALE"
    --clip_eot_views "$CLIP_EOT_VIEWS"
    --clip_eot_noise_std "$CLIP_EOT_NOISE_STD"
    --clip_eot_crop_min "$CLIP_EOT_CROP_MIN"
    --grad_clip "$GRAD_CLIP"
)
[ "$CLIP_GRAD_CHECK" = "1" ] && COND_FLAGS+=(--clip_grad_check)
[ "$COMPILE" = "1" ] && COND_FLAGS+=(--compile)
echo "[cond-v2] flags: ${COND_FLAGS[*]}"

case "${MODEL_SIZE}" in
    B) MODEL=JiT_B; CFG=3.0; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0; LOAD="${CKPT_ROOT}/JiT-B.pth" ;;
    L) MODEL=JiT_L; CFG=2.4; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0; LOAD="${CKPT_ROOT}/JiT-L.pth" ;;
    H) MODEL=JiT_H; CFG=2.2; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0; LOAD="${CKPT_ROOT}/JiT-H.pth" ;;
    *) echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE}"; exit 1 ;;
esac

run_one() {
    local exp_name="$1"; shift
    torchrun \
        --nnodes="$NNODES" --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        --nproc_per_node="$GPUS_PER_NODE" \
        conditional_main_fd.py \
        --output_dir "$OUTPUT_DIR" \
        --project "" \
        --exp_name "$exp_name" \
        --batch_size "$BATCH_SIZE" \
        --data_path "$DATA_ROOT" \
        --load_from "$LOAD" \
        --model "$MODEL" --rope_2d --learned_pe --legacy_time_convention \
        --cfg "$CFG" --interval_min "$INTERVAL_MIN" --interval_max "$INTERVAL_MAX" \
        --ema_type edm \
        --num_sampling_steps 1 \
        --eval_bsz "$EVAL_BSZ" --num_images_for_eval_and_search 50000 \
        --vis_freq "$VIS_FREQ" --online_eval --eval_freq "$EVAL_FREQ" \
        --print_freq 20 --milestone_interval 10 --save_freq 5 \
        --epochs 20 --steps_per_epoch 1250 --warmup_epochs 0 \
        --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
        --fd_eigvalsh --fd_ema_beta 0.999 \
        --dist_vis_every "$DIST_VIS_EVERY" \
        --qphi_lora_enabled \
        --qphi_alpha "$QPHI_ALPHA" --qphi_warmup_steps "$QPHI_WARMUP_STEPS" --qphi_ramp_steps "$QPHI_RAMP_STEPS" \
        --qphi_pretrain_steps "$QPHI_PRETRAIN_STEPS" \
        --qphi_steps_per_iter "$QPHI_STEPS_PER_ITER" \
        --qphi_lora_r 8 --qphi_lora_last_k 6 --qphi_lora_batch 64 \
        --clip_grad_check \
        $RESUME_FLAG "$WANDB_FLAG" \
        "${COND_FLAGS[@]}" \
        "$@"
}

run_one "$RUN_NAME" --fd_repr_models inception
