#!/usr/bin/env bash
# Fresh full-BF16 neural training: generator, FD extractors, Qwen, heads, LoRA.
# See docs/vlm_bf16_calibration.md. Run inside a GPU allocation or use --dry-run.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"
die() { echo "ERROR: $*" >&2; exit 2; }
if [[ "${1:-}" == "--dry-run" ]]; then
    export DRY_RUN=1
    shift
fi
(( $# == 0 )) || die "Usage: bash ${BASH_SOURCE[0]} [--dry-run]"

: "${DRY_RUN:=0}"
: "${PY_BIN:=${REPO_DIR}/.venv/bin/python}"
: "${DATA_PATH:=${REPO_DIR}/data/imagenet}"
: "${START_CKPT:=${REPO_DIR}/checkpoints/base/JiT-B-uncond.pth}"
: "${STATS_DIR:=${REPO_DIR}/data/fid_stats/imagenet100_v1}"
: "${P_HEAD:=${REPO_DIR}/work_dirs/vlm_p_head_qwen_answer_c100/p_head.pt}"
: "${OUTPUT_DIR:=${REPO_DIR}/work_dirs}"
: "${LOG_DIR:=${REPO_DIR}/sweep_logs}"
: "${PROJECT:=JiT_uncond_vlm_delta}"
: "${EXP_NAME:=qwen_lora_fullbf16_scaled_cal_${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}}"
: "${NPROC_PER_NODE:=4}"
: "${GLOBAL_BATCH:=96}"
: "${CALIBRATION:=1}"
: "${CAL_STEPS:=3000}"
# Used when CALIBRATION=0, e.g. the Blackwell training preset.
: "${EPOCHS:=40}"
: "${STEPS_PER_EPOCH:=1250}"
: "${VIS_FREQ:=2}"
: "${DISABLE_VIS:=1}"
: "${ONLINE_EVAL:=0}"
# Previous FP32 late ratio .072 at 1e-5 suggests ~3.5e-5 for .25.
# BF16 changes the gradient field: this is a seed to MEASURE, not a calibrated weight.
: "${WEIGHT:=3.5e-5}"
: "${VLM_VJP_LOSS_SCALE:=1024}"
: "${VLM_Q_LOSS_SCALE:=1024}"
: "${VLM_MICROBATCH:=2}"
: "${GENERATOR_MICROBATCH:=2}"
: "${FD_FEATURE_MICROBATCH:=2}"
: "${FD_QUEUE_FILL_BSZ:=8}"
: "${QUEUE_SIZE:=50000}"
: "${Q_LR:=1e-4}"
: "${Q_LORA_LR:=1e-5}"
: "${Q_WEIGHT_DECAY:=3.0}"
: "${Q_LORA_WEIGHT_DECAY:=0.01}"
: "${Q_GRAD_CLIP:=1.0}"
: "${DELTA_RAMP:=500}"
: "${PRINT_FREQ:=10}"
: "${INIT_DIAG_SAMPLES:=32}"
: "${PER_CLASS_EVERY:=500}"
: "${CKPT_TARGET_MINUTES:=10}"
: "${NUM_WORKERS:=4}"
: "${RESUME:=0}"
: "${MASTER_PORT:=29572}"

[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] || die "DRY_RUN must be 0 or 1"
[[ "${RESUME}" == 0 || "${RESUME}" == 1 ]] || die "RESUME must be 0 or 1"
[[ "${CALIBRATION}" == 0 || "${CALIBRATION}" == 1 ]] || die "CALIBRATION must be 0 or 1"
for name in PROJECT EXP_NAME; do
    [[ "${!name}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || die "${name} must be a plain directory name"
done
for name in NPROC_PER_NODE GLOBAL_BATCH CAL_STEPS EPOCHS STEPS_PER_EPOCH VLM_MICROBATCH GENERATOR_MICROBATCH FD_FEATURE_MICROBATCH FD_QUEUE_FILL_BSZ PRINT_FREQ; do
    [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer"
done
(( NPROC_PER_NODE <= GLOBAL_BATCH && GLOBAL_BATCH % NPROC_PER_NODE == 0 )) \
    || die "NPROC_PER_NODE must divide GLOBAL_BATCH=${GLOBAL_BATCH}"
for name in WEIGHT VLM_VJP_LOSS_SCALE VLM_Q_LOSS_SCALE; do
    [[ "${!name}" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] \
        && awk -v n="${!name}" 'BEGIN {exit !(n > 0 && n <= 3.402823466e38)}' \
        || die "${name} must be positive and finite in FP32"
done

# Derive the local batch from the requested global batch. Retain the historical
# rank-based LR default; the Blackwell preset explicitly keeps LR=1e-5.
export BATCH_SIZE=$(( GLOBAL_BATCH / NPROC_PER_NODE ))
: "${LR:=$(awk -v n="${NPROC_PER_NODE}" 'BEGIN {printf "%.10g", 1e-5*n/4}')}"
# Logical rank IDs only. Never replace Slurm's CUDA_VISIBLE_DEVICES mapping.
GPUS=0
for ((i=1; i<NPROC_PER_NODE; i++)); do GPUS+=",${i}"; done
export GPUS PRESERVE_CUDA_VISIBLE_DEVICES=1 RUN_FOREGROUND=1
export CALIBRATION Q_LORA=1 Q_USE_EMA=0 Q_OPTIMIZER=adamw
export GLOBAL_BATCH EPOCHS STEPS_PER_EPOCH VIS_FREQ DISABLE_VIS ONLINE_EVAL
export Q_BETA1=0.0 Q_BETA2=0.999 Q_UPDATES_PER_STEP=1
export Q_BOOTSTRAP=0 Q_BOOTSTRAP_UPDATES=0 VLM_DTYPE=bf16 VLM_SAMPLES_PER_STEP=0
export DELTA_WARMUP=0 DELTA_CLAMP=0 AUTO_RESUME=0
export PY_BIN DATA_PATH START_CKPT STATS_DIR P_HEAD OUTPUT_DIR LOG_DIR PROJECT EXP_NAME
export CAL_STEPS WEIGHT VLM_MICROBATCH QUEUE_SIZE Q_LR Q_WEIGHT_DECAY Q_GRAD_CLIP
export DELTA_RAMP PRINT_FREQ INIT_DIAG_SAMPLES PER_CLASS_EVERY CKPT_TARGET_MINUTES LR
export RESUME MASTER_PORT DRY_RUN

# BF16 neural compute and parameter updates, without FP32 master weights.
# FD moments/eigensolve remain FP64; log-probability and VJP reductions use FP32.
export EXTRA_ARGS="--vlm_q_lora_rank 8 --vlm_q_lora_alpha 16 --vlm_q_lora_scope both --vlm_q_lora_lr ${Q_LORA_LR} --vlm_q_lora_weight_decay ${Q_LORA_WEIGHT_DECAY} --vlm_attn_implementation eager --vlm_vjp_loss_scale ${VLM_VJP_LOSS_SCALE} --vlm_q_loss_scale ${VLM_Q_LOSS_SCALE} --dtype bf16 --parameter_dtype bf16 --grad_checkpointing --generator_microbatch ${GENERATOR_MICROBATCH} --fd_feature_microbatch ${FD_FEATURE_MICROBATCH} --fd_feature_checkpoint --fd_queue_fill_bsz ${FD_QUEUE_FILL_BSZ} --num_workers ${NUM_WORKERS}"

if [[ "${CALIBRATION}" == 1 ]]; then
    RUN_MODE=calibration
    TOTAL_STEPS="${CAL_STEPS}"
else
    RUN_MODE=training
    TOTAL_STEPS=$(( EPOCHS * STEPS_PER_EPOCH ))
fi
echo "Full BF16 ${RUN_MODE}: ${TOTAL_STEPS} steps, weight=${WEIGHT}, VJP scale=${VLM_VJP_LOSS_SCALE}, q CE scale=${VLM_Q_LOSS_SCALE}"
echo "Neural compute/parameters/AdamW moments: BF16; FD statistics/eigensolve: FP64; log-probability/VJP reductions: FP32."
echo "Visibility inherited: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}; ${NPROC_PER_NODE} ranks, global batch ${GLOBAL_BATCH}"
echo "Fixed loss scales are removed before updates; they do not increase the conditional weight."
if [[ "${DRY_RUN}" == 1 ]]; then
    exec bash scripts/run_jit_uncond_vlm_delta_100class.sh
fi

[[ -x "${PY_BIN}" ]] || die "Python executable not found: ${PY_BIN}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${LOG_DIR}"
RUN_DIR="${OUTPUT_DIR}/${PROJECT}/${EXP_NAME}"
if (( ${SLURM_RESTART_COUNT:-0} > 0 )) && [[ "${RESUME}" == 0 ]]; then
    if compgen -G "${RUN_DIR}/checkpoints/step_*.pth" > /dev/null; then
        export RESUME=1
    else
        # Requeue during model loading/queue fill: preserve the incomplete attempt.
        suffix="preempted-${SLURM_RESTART_COUNT}-$(date -u +%Y%m%dT%H%M%SZ)"
        if [[ -d "${RUN_DIR}" ]]; then mv -- "${RUN_DIR}" "${RUN_DIR}.${suffix}"; fi
        if [[ -f "${LOG_DIR}/${EXP_NAME}.out" ]]; then
            mv -- "${LOG_DIR}/${EXP_NAME}.out" "${LOG_DIR}/${EXP_NAME}.out.${suffix}"
        fi
    fi
fi

"${PY_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
    scripts/check_vlm_lora_allocation.py --dtype bf16 --expected-gpus "${NPROC_PER_NODE}" \
    --loss-scale "${VLM_VJP_LOSS_SCALE}" --output "${LOG_DIR}/${EXP_NAME}.allocation.json"
bash scripts/run_jit_uncond_vlm_delta_100class.sh
ANALYSIS_ARGS=(--weight "${WEIGHT}")
if [[ "${CALIBRATION}" == 1 ]]; then ANALYSIS_ARGS+=(--calibration); fi
REPORT_PATH="${LOG_DIR}/${EXP_NAME}.${RUN_MODE}.txt"
"${PY_BIN}" scripts/analyze_vlm_delta_run.py "${ANALYSIS_ARGS[@]}" "${RUN_DIR}" \
    | tee "${REPORT_PATH}"
echo "Metrics: ${RUN_DIR}/training_metrics.json"
echo "Report: ${REPORT_PATH}"
