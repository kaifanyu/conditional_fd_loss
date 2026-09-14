#!/usr/bin/env bash
# Submit from the repository root. REPO_DIR can select a frozen source snapshot.
# sbatch scripts/sbatch_vlm_lora_q_grasp.sh
#SBATCH --job-name=qwen-lora-q-cal
#SBATCH --account=gu-account
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=l40|l40s|a40|a6000
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=sweep_logs/qwen-lora-q-cal-%j.out
set -euo pipefail

TRAINING_ROOT="${TRAINING_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
REPO_DIR="${REPO_DIR:-${TRAINING_ROOT}}"
cd "${REPO_DIR}"
export PY_BIN="${TRAINING_ROOT}/.venv/bin/python"
export HF_HOME=/mnt/projects/jg/kaifany/.hf
export TORCH_HOME=/mnt/projects/jg/kaifany/.torch
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_KERNEL_CACHE_PATH="${TORCH_HOME}/kernels"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
mkdir -p "${PYTORCH_KERNEL_CACHE_PATH}"

# GRASP's device cgroup exposes allocated GPUs with relative indices (as in
# sbatch_grasp.sh). Check the resulting visibility and NCCL before loading 7B.
export GPUS=0,1,2,3 CUDA_VISIBLE_DEVICES=0,1,2,3 RUN_FOREGROUND=1
export DATA_PATH=/mnt/projects/jg/kaifany/dataset/imagenet
export START_CKPT="${TRAINING_ROOT}/checkpoints/base/JiT-B-uncond.pth"
export STATS_DIR="${TRAINING_ROOT}/data/fid_stats/imagenet100_v1"
export P_HEAD="${TRAINING_ROOT}/work_dirs/vlm_p_head_qwen_answer_c100/p_head.pt"
export OUTPUT_DIR="${TRAINING_ROOT}/work_dirs" LOG_DIR="${TRAINING_ROOT}/sweep_logs"
export PROJECT=JiT_uncond_vlm_delta
export EXP_NAME="${EXP_NAME:-qwen_lora_direct_fp32_cal_${SLURM_JOB_ID:?}}"
# The full-model GPU smoke measured conditional/FD gradient ratios of 59-114
# at weight .01 after 1-2 q updates (no ramp). Use a conservative calibration
# seed; the eventual training weight still requires the sustained late window.
export CALIBRATION=1 CAL_STEPS=3000 WEIGHT=1e-5
export BATCH_SIZE=24 LR=1e-5 QUEUE_SIZE=50000
export Q_LORA=1 Q_USE_EMA=0 Q_OPTIMIZER=adamw Q_LR=1e-4
export Q_BETA1=0.0 Q_BETA2=0.999 Q_WEIGHT_DECAY=3.0 Q_GRAD_CLIP=1.0
export Q_UPDATES_PER_STEP=1 Q_BOOTSTRAP=0 Q_BOOTSTRAP_UPDATES=0
export VLM_DTYPE=fp32 VLM_MICROBATCH=1 VLM_SAMPLES_PER_STEP=0
export DELTA_WARMUP=0 DELTA_RAMP=500 DELTA_CLAMP=0
export INIT_DIAG_SAMPLES=32 PRINT_FREQ=10 CKPT_TARGET_MINUTES=10
export PER_CLASS_EVERY=500
export EXTRA_ARGS="--vlm_q_lora_rank 8 --vlm_q_lora_alpha 16 --vlm_q_lora_lr 1e-5 --vlm_q_lora_weight_decay 0.01 --vlm_q_lora_scope both --vlm_attn_implementation eager --vlm_vjp_loss_scale 1 --vlm_q_loss_scale 1 --dtype fp32 --grad_checkpointing --generator_microbatch 2 --fd_feature_microbatch 2 --fd_feature_checkpoint --fd_queue_fill_bsz 8 --num_workers 4"

RUN_DIR="${OUTPUT_DIR}/${PROJECT}/${EXP_NAME}"
export RESUME=0
if (( ${SLURM_RESTART_COUNT:-0} > 0 )); then
    if compgen -G "${RUN_DIR}/checkpoints/step_*.pth" > /dev/null; then
        export RESUME=1
    else
        # Preemption during loading/queue fill may precede the first checkpoint.
        # Preserve that attempt and restart the same experiment from p and G0.
        SUFFIX="preempted-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT}-$(date -u +%Y%m%dT%H%M%SZ)"
        if [[ -d "${RUN_DIR}" ]]; then mv "${RUN_DIR}" "${RUN_DIR}.${SUFFIX}"; fi
        if [[ -f "${LOG_DIR}/${EXP_NAME}.out" ]]; then
            mv "${LOG_DIR}/${EXP_NAME}.out" "${LOG_DIR}/${EXP_NAME}.${SUFFIX}.out"
        fi
    fi
fi

echo "Job ${SLURM_JOB_ID}: $(hostname), source=${REPO_DIR}, resume=${RESUME}"
echo "Output: ${RUN_DIR}; initial calibration weight ${WEIGHT}, ${CAL_STEPS} steps"
"${PY_BIN}" -m torch.distributed.run --standalone --nproc_per_node=4 \
    scripts/check_vlm_lora_allocation.py \
    --output "${LOG_DIR}/${EXP_NAME}.allocation.json"
bash scripts/run_jit_uncond_vlm_delta_100class.sh
"${PY_BIN}" scripts/analyze_vlm_delta_run.py --calibration --weight "${WEIGHT}" "${RUN_DIR}" \
    > "${LOG_DIR}/${EXP_NAME}.calibration.txt"
