#!/usr/bin/env bash
# Submit from the copied repository root on GRASP.
# Six GPUs, global batch 48, 15,000 steps, sample grids every 1,500 steps.
#SBATCH --job-name=qwen-bf16-blackwell
#SBATCH --partition=gu-compute
#SBATCH --nodelist=jg-b6000-0.grasp.maas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-vlm-blackwell-%j.out
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${TRAINING_ROOT:=${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
[[ "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || { echo "This preset uses one node." >&2; exit 2; }
cd "${TRAINING_ROOT}"
[[ -f scripts/run_vlm_lora_q_bf16_calibration.sh ]] || {
    echo "Training launcher missing; TRAINING_ROOT must point to the updated repository." >&2
    exit 2
}

export PY_BIN="${PY_BIN:-${TRAINING_ROOT}/.venv/bin/python}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
export GLOBAL_BATCH="${GLOBAL_BATCH:-48}"
# Regular training keeps the previous visualization machinery enabled.
# setup() measures vis_freq in epochs: 1 * 1500 = 1500 optimizer steps.
export CALIBRATION=0
export EPOCHS="${EPOCHS:-10}" STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1500}"
export VIS_FREQ="${VIS_FREQ:-1}" DISABLE_VIS=0 ONLINE_EVAL=0
# Preserve the previous AdamW learning rate when changing GPU/batch counts.
export LR="${LR:-1e-5}"
export EXP_NAME="${EXP_NAME:-qwen_lora_fullbf16_blackwell_b48_15k_vis1500_${SLURM_JOB_ID}}"
# Set DATA_PATH, START_CKPT, STATS_DIR, P_HEAD, HF_HOME, and TORCH_HOME before
# sbatch if the assets are outside the launcher's repository-relative defaults.
# Use a CUDA PyTorch build supporting Blackwell (e.g. CUDA 12.8 or newer).
# Preserve Slurm's CUDA_VISIBLE_DEVICES; torchrun starts one process per GPU.
exec bash scripts/run_vlm_lora_q_bf16_calibration.sh
