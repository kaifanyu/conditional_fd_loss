#!/usr/bin/env bash
# Dual semantic judge: frozen MAE linear probe + frozen Qwen2.5-VL, both
# feeding one additive conditional log-p correction, on the 20-class
# diagnostic that r9 showed real held-out transfer on.
#
#   L = L_FD + lambda_mae(s) * L_mae + lambda_vlm(s) * L_vlm
#
# Schedule is deliberately staggered. The MAE term uses r9's loss settings
# (lambda 5e-5, ramp 500, cap 10%) from step 0, and the VLM only phases in from
# step 10k once samples are recognizable enough for a Yes/No question to mean
# anything -- r5 ran a VLM from step 0 against unrecognizable samples for 100k
# steps and p(Yes|target) fell from 0.0014 to 0.0003 while its gradient
# dominated FD.
#
# NOTE: the first 10k steps are NOT a replication of r9. Same loss, different
# LR: cosine 1e-5->0 is spread over 75k steps here versus 10k in r9, so this
# run sits near 1e-5 where r9 had already decayed. Expect faster early
# movement than r9's trace for that reason alone.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

: "${CUDA_GPUS:=4,5,6,7}"
: "${MASTER_PORT:=29531}"
: "${PY_BIN:=/home/nvidia/miniconda3/envs/fdloss/bin/python}"
: "${DATA_PATH:=/data/dataset/imagenet}"
: "${START_CKPT:=work_dirs/JiT_cond_sweep/r8_mae_probe_inception_lam1e5/checkpoints/step_0071799.pth}"
: "${MAE_PROBE_CKPT:=work_dirs/mae_probe_vitl_cls/best.pt}"
: "${VLM_MODEL:=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
: "${STATS_DIR:=data/fid_stats/imagenet20_v1}"
: "${OUTPUT_DIR:=work_dirs}"
: "${PROJECT:=JiT_cond_sweep}"
: "${EXP_NAME:=r11_dual_mae_vlm_20class}"
: "${LOG_DIR:=sweep_logs}"

# -- MAE probe term --
# NOT r9's 5e-5. At FD_EMA_BETA=0.99 the FD gradient is ~10x larger, so 5e-5
# now buys a ratio of only 0.067 (measured), which is r8 territory -- r8 sat at
# 0.03 and never learned semantics. 2.2e-4 targets ~0.30: below r9's 0.34,
# because that is the regime that collapsed, but well clear of r8's.
: "${LAMBDA_MAE:=2.2e-4}"
: "${COND_WARMUP:=0}"
: "${COND_RAMP:=500}"
: "${TARGET_LOGP:=-2.302585}"   # stop pushing a sample at p(target)=10%
: "${MIN_PROBE_VAL_TOP1:=0.50}"

# -- VLM term: small lambda, late start --
# Calibrated directly, not extrapolated from r5. A 12-step run of this exact
# config off this exact start checkpoint measured, at steady state:
#     lambda_mae 5e-5 -> grad_ratio_mae_fd 0.51
#     lambda_vlm 5e-6 -> grad_ratio_vlm_fd 0.62
# i.e. the VLM pulls ~12x harder per unit lambda than the MAE probe, because
# its whole gradient lands on the K=2 scored images instead of spreading over
# all 21. 8e-7 targets ~0.10, subordinate to the MAE term by design: the VLM
# is here to veto adversarial solutions, not to steer.
#
# RECALIBRATE at step ~15000 (first fully-ramped reading). If
# grad_ratio_vlm_fd is outside 0.05-0.20, kill and relaunch with
# LAMBDA_VLM = 1.0e-5 * 0.08 / <observed ratio>.
#
# Rescaled with LAMBDA_MAE for FD_EMA_BETA=0.99: 8e-7 measured a ratio of
# 0.0064 under the new FD scale. 1.0e-5 targets ~0.08.
: "${LAMBDA_VLM:=1.0e-5}"
: "${VLM_WARMUP:=10000}"
: "${VLM_RAMP:=5000}"
: "${VLM_TARGET_LOGP:=-0.69}"   # binary chance is -log(2); cap at p=50%
: "${VLM_SAMPLES:=2}"
: "${VLM_MICROBATCH:=1}"
: "${VLM_QUESTION_MODE:=pairwise}"
: "${VLM_DISTRACTOR_POOL:=train_classes}"

# -- FD statistics window --
# 0.999 (~1000-step EMA) is what r11 ran and it could not see mode collapse:
# a collapsed generator whose single mode drifts across 1000 steps still
# accumulates a broad covariance, so in-loss inception FD read 12.5 while the
# frozen-checkpoint eval read 120.7 on the same model. 0.99 is a ~100-step
# window: at 84 gathered samples/step that is ~8.4k effective samples, still
# enough to estimate a 2048-dim covariance, but 10x more current.
#
# Side effect that matters: the FD gradient on the live batch is scaled by
# (1 - beta), so this makes grad_x_fd ~10x larger and the conditional ratios
# ~10x smaller at fixed lambda. LAMBDA_MAE/LAMBDA_VLM below are recalibrated
# for it -- do not carry r11's values over.
: "${FD_EMA_BETA:=0.99}"

: "${EPOCHS:=60}"
: "${STEPS_PER_EPOCH:=1250}"    # 60 x 1250 = 75,000 iterations
: "${QUEUE_SIZE:=50000}"
: "${NUM_EVAL_IMAGES:=20000}"
# Online eval is 20k images a time; --cond_probe already reports held-out
# ResNet rank/top-1/top-5 every print_freq steps, so sparse FID checks suffice.
: "${EVAL_FREQ:=10}"
: "${SAVE_FREQ:=5}"
: "${PRINT_FREQ:=20}"
: "${AUTO_RESUME:=1}"
: "${ONLINE_EVAL:=1}"     # 0 for smoke tests: skips periodic 20k-image eval
: "${RUN_FOREGROUND:=0}"

CLASS_IDS=(0 9 88 130 207 279 281 340 360 387 404 417 444 555 569 817 920 949 974 979)
FD_MODELS=(vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception)
FD_POOLS=(cls cls cls)
FD_SIZES=(224 224 256)
FD_STATS=(
    "${STATS_DIR}/siglip_cls.npz"
    "${STATS_DIR}/mae_cls.npz"
    "${STATS_DIR}/inception.npz"
)

IFS=',' read -r -a GPU_IDS <<< "${CUDA_GPUS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"

if [[ ! -x "${PY_BIN}" ]]; then
    echo "ERROR: Python executable not found: ${PY_BIN}" >&2
    exit 2
fi
if [[ ! -f "${START_CKPT}" ]]; then
    echo "ERROR: starting JiT checkpoint not found: ${START_CKPT}" >&2
    exit 2
fi
if [[ ! -f "${MAE_PROBE_CKPT}" ]]; then
    echo "ERROR: trained MAE probe not found: ${MAE_PROBE_CKPT}" >&2
    exit 2
fi
if [[ ! -d "${VLM_MODEL}" ]]; then
    echo "ERROR: local VLM checkpoint not found: ${VLM_MODEL}" >&2
    exit 2
fi
for stats_path in "${FD_STATS[@]}"; do
    if [[ ! -f "${stats_path}" ]]; then
        echo "ERROR: missing 20-class reference stats: ${stats_path}" >&2
        echo "Run first: bash scripts/compute_imagenet20_fd_stats.sh" >&2
        exit 2
    fi
done

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${EXP_NAME}.out"

CMD=(
    "${PY_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_port="${MASTER_PORT}"
    conditional_main_fd_dual.py
    --data_path "${DATA_PATH}"
    --load_from "${START_CKPT}"
    --output_dir "${OUTPUT_DIR}"
    --project "${PROJECT}"
    --exp_name "${EXP_NAME}"
    --batch_size 21
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0
    --ema_type edm --num_sampling_steps 1
    --eval_bsz 256 --num_images_for_eval_and_search "${NUM_EVAL_IMAGES}"
    --vis_freq 2 --eval_freq "${EVAL_FREQ}"
    --print_freq "${PRINT_FREQ}" --milestone_interval 10 --save_freq "${SAVE_FREQ}"
    --epochs "${EPOCHS}" --steps_per_epoch "${STEPS_PER_EPOCH}"
    --warmup_epochs 0
    --lr 1e-5 --lr_sched cosine --min_lr 0.0
    --fd_eigvalsh --fd_ema_beta "${FD_EMA_BETA}"
    --queue_size "${QUEUE_SIZE}"
    --fd_repr_models "${FD_MODELS[@]}"
    --fd_repr_pool_types "${FD_POOLS[@]}"
    --fd_target_sizes "${FD_SIZES[@]}"
    --fd_repr_stats_paths "${FD_STATS[@]}"
    --fid_stats_path "${FD_STATS[2]}"
    --train_class_ids "${CLASS_IDS[@]}"
    --class_of_interest "${CLASS_IDS[@]}"
    --force_class_of_interest
    --lambda_cond "${LAMBDA_MAE}"
    --mae_probe_checkpoint "${MAE_PROBE_CKPT}"
    --mae_probe_min_val_top1 "${MIN_PROBE_VAL_TOP1}"
    --mae_probe_grad_check
    --cond_warmup_steps "${COND_WARMUP}"
    --cond_ramp_steps "${COND_RAMP}"
    --cond_target_logp "${TARGET_LOGP}"
    --mae_eot_views 1
    --mae_eot_noise_std 0.0
    --mae_eot_crop_min 1.0
    --lambda_vlm "${LAMBDA_VLM}"
    --cond_vlm_model "${VLM_MODEL}"
    --vlm_warmup_steps "${VLM_WARMUP}"
    --vlm_ramp_steps "${VLM_RAMP}"
    --vlm_target_logp "${VLM_TARGET_LOGP}"
    --vlm_samples_per_step "${VLM_SAMPLES}"
    --vlm_microbatch "${VLM_MICROBATCH}"
    --vlm_question_mode "${VLM_QUESTION_MODE}"
    --vlm_distractor_pool "${VLM_DISTRACTOR_POOL}"
    --vlm_dtype bf16
    --cond_probe
    --disable_wandb
)

if [[ "${AUTO_RESUME}" == "1" ]]; then
    CMD+=(--auto_resume)
fi
if [[ "${ONLINE_EVAL}" == "1" ]]; then
    CMD+=(--online_eval)
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    CMD+=("${EXTRA[@]}")
fi

echo "Experiment:      ${PROJECT}/${EXP_NAME}"
echo "GPUs:            ${CUDA_GPUS} (${NPROC_PER_NODE} ranks)"
echo "Start ckpt:      ${START_CKPT}"
echo "Classes (${#CLASS_IDS[@]}):   ${CLASS_IDS[*]}"
echo "Iterations:      $(( EPOCHS * STEPS_PER_EPOCH ))"
echo "MAE term:        lambda=${LAMBDA_MAE} warmup=${COND_WARMUP} ramp=${COND_RAMP} cap=${TARGET_LOGP}"
echo "VLM term:        lambda=${LAMBDA_VLM} warmup=${VLM_WARMUP} ramp=${VLM_RAMP} cap=${VLM_TARGET_LOGP}"
echo "VLM distractors: ${VLM_DISTRACTOR_POOL}"
echo "Log:             ${LOG_PATH}"

if [[ "${RUN_FOREGROUND}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_GPUS}" "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
else
    CUDA_VISIBLE_DEVICES="${CUDA_GPUS}" setsid "${CMD[@]}" \
        >"${LOG_PATH}" 2>&1 < /dev/null &
    echo "Started PID $!. Follow with: tail -f ${LOG_PATH}"
fi
