#!/usr/bin/env bash
# Train JiT-B with the existing three FD realism judges and one frozen
# Qwen2.5-VL binary semantic judge.
#
# By default this launches the full 80 x 1,250-step experiment on GPUs 4 and 5
# in the background. Every commonly changed run setting is environment
# overridable; see VLM_CONDITIONING.md for a short pilot command.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

: "${CUDA_GPUS:=4,5}"
: "${MASTER_PORT:=29504}"
: "${EXP_NAME:=r4_vlm_qwen}"
: "${PROJECT:=JiT_cond_sweep}"
: "${EPOCHS:=80}"
: "${STEPS_PER_EPOCH:=1250}"
: "${WARMUP_EPOCHS:=5}"
: "${VLM_SAMPLES:=2}"
: "${LAMBDA_VLM:=5e-5}"
: "${COND_WARMUP:=15000}"
: "${COND_RAMP:=20000}"
: "${QUEUE_SIZE:=50000}"
: "${ONLINE_EVAL:=1}"
: "${AUTO_RESUME:=1}"
: "${RUN_FOREGROUND:=0}"
# Also gates the per-term gradient diagnostics: diag fires on log steps only.
# Set to 1 for a short lambda-calibration run, where every step should report.
: "${PRINT_FREQ:=20}"

# FD realism judges. Space-separated and word-split into the command below, so
# the three lists must stay the same length. Dropping inception here leaves the
# online-eval inception FID as a genuinely held-out realism metric, since the
# evaluator always loads inception independently of this set.
: "${FD_MODELS:=vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception}"
: "${FD_POOLS:=cls cls cls}"
: "${FD_SIZES:=224 224 256}"

: "${PY_BIN:=/home/nvidia/miniconda3/envs/fdloss/bin/python}"
: "${DATA_PATH:=/data/dataset/imagenet}"
: "${BASE_CKPT:=./checkpoints/base/JiT-B-uncond.pth}"
: "${OUTPUT_DIR:=./work_dirs}"
: "${LOG_DIR:=./sweep_logs}"
: "${VLM_MODEL:=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_GPUS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"
if (( NPROC_PER_NODE < 1 )); then
    echo "ERROR: CUDA_GPUS must contain at least one GPU id." >&2
    exit 2
fi
if [[ ! -x "${PY_BIN}" ]]; then
    echo "ERROR: Python executable not found: ${PY_BIN}" >&2
    exit 2
fi
if [[ ! -d "${VLM_MODEL}" ]]; then
    echo "ERROR: local VLM checkpoint not found: ${VLM_MODEL}" >&2
    exit 2
fi
if [[ ! -f "${BASE_CKPT}" ]]; then
    echo "ERROR: base JiT checkpoint not found: ${BASE_CKPT}" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${EXP_NAME}.out"

RESUME_ARGS=()
if [[ "${AUTO_RESUME}" == "1" ]]; then
    RESUME_ARGS+=(--auto_resume)
fi

ONLINE_EVAL_ARGS=()
if [[ "${ONLINE_EVAL}" == "1" ]]; then
    ONLINE_EVAL_ARGS+=(--online_eval)
fi

CMD=(
    "${PY_BIN}" -m torch.distributed.run
    --nnodes=1
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_port="${MASTER_PORT}"
    conditional_main_fd_ponly.py
    --data_path "${DATA_PATH}"
    --load_from "${BASE_CKPT}"
    --output_dir "${OUTPUT_DIR}"
    --project "${PROJECT}"
    --exp_name "${EXP_NAME}"
    --batch_size 21
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0
    --ema_type edm --num_sampling_steps 1
    --eval_bsz 256 --num_images_for_eval_and_search "${QUEUE_SIZE}"
    --vis_freq 2 --eval_freq 10
    --print_freq "${PRINT_FREQ}" --milestone_interval 10 --save_freq 5
    --epochs "${EPOCHS}" --steps_per_epoch "${STEPS_PER_EPOCH}"
    --warmup_epochs "${WARMUP_EPOCHS}"
    --lr 1e-5 --lr_sched cosine --min_lr 0.0
    --fd_eigvalsh --fd_ema_beta 0.999
    --queue_size "${QUEUE_SIZE}"
    --fd_repr_models ${FD_MODELS}
    --fd_repr_pool_types ${FD_POOLS}
    --fd_target_sizes ${FD_SIZES}
    --lambda_cond "${LAMBDA_VLM}"
    --cond_warmup_steps "${COND_WARMUP}"
    --cond_ramp_steps "${COND_RAMP}"
    --cond_target_logp -0.69
    --cond_vlm_model "${VLM_MODEL}"
    --vlm_dtype bf16
    --vlm_question_mode pairwise
    --vlm_samples_per_step "${VLM_SAMPLES}"
    --vlm_microbatch 1
    --vlm_attn_implementation sdpa
    --clip_eot_views 1
    --clip_eot_noise_std 0.0
    --clip_eot_crop_min 1.0
    --cond_probe
    --disable_wandb
    "${ONLINE_EVAL_ARGS[@]}"
    "${RESUME_ARGS[@]}"
)

echo "Experiment: ${PROJECT}/${EXP_NAME}"
echo "GPUs: ${CUDA_GPUS} (workers=${NPROC_PER_NODE}, port=${MASTER_PORT})"
echo "Steps: ${EPOCHS} x ${STEPS_PER_EPOCH}; VLM samples/rank/step=${VLM_SAMPLES}"
echo "VLM lambda: ${LAMBDA_VLM}; warmup=${COND_WARMUP}; ramp=${COND_RAMP}"
echo "FD judges: ${FD_MODELS}"
echo "FD queue/eval samples: ${QUEUE_SIZE}; online eval=${ONLINE_EVAL}"
echo "Log: ${LOG_PATH}"

if [[ "${RUN_FOREGROUND}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_GPUS}" "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
else
    CUDA_VISIBLE_DEVICES="${CUDA_GPUS}" setsid "${CMD[@]}" \
        >"${LOG_PATH}" 2>&1 < /dev/null &
    RUN_PID=$!
    echo "Started PID ${RUN_PID}. Follow with: tail -f ${LOG_PATH}"
fi
