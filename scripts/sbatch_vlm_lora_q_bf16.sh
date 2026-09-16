#!/usr/bin/env bash
# Submit from the repository root; select account/partition/GPU type via sbatch.
#SBATCH --job-name=qwen-fullbf16-cal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-vlm-bf16-%j.out
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${TRAINING_ROOT:=${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
[[ "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || { echo "This preset uses one node." >&2; exit 2; }
cd "${TRAINING_ROOT}"
export PY_BIN="${PY_BIN:-${TRAINING_ROOT}/.venv/bin/python}"
export EXP_NAME="${EXP_NAME:-qwen_lora_fullbf16_scaled_cal_${SLURM_JOB_ID}}"
# torchrun creates one process per GPU. NPROC_PER_NODE defaults to four in the
# launcher; override it together with --gres when requesting a different count.
exec bash scripts/run_vlm_lora_q_bf16_calibration.sh
