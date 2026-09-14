#!/usr/bin/env bash
# Slurm wrapper for the GRASP cluster. Submits any of this repo's launchers
# without editing them: everything they read is an overridable env var.
#
#   sbatch scripts/sbatch_grasp.sh scripts/run_jit_uncond_vlm_delta_100class.sh
#   WEIGHT=0.0006 CALIBRATION=1 EXP_NAME=cal1 \
#       sbatch scripts/sbatch_grasp.sh scripts/run_jit_uncond_vlm_delta_100class.sh
#
# Any #SBATCH default below is overridden by a flag on the sbatch command line:
#   sbatch -p gu-compute --qos=gu-med --gres=gpu:4 scripts/sbatch_grasp.sh <script>
#
#SBATCH --job-name=fdloss
#SBATCH --partition=batch
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=sweep_logs/slurm-%j.out
set -euo pipefail

# sbatch COPIES this file to /var/spool/slurmd, so ${BASH_SOURCE[0]} does NOT
# point into the repo at runtime. Use the submit directory, and verify it.
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [[ ! -f "${REPO_DIR}/conditional_main_fd_vlm_delta.py" ]]; then
    echo "ERROR: REPO_DIR=${REPO_DIR} is not the FD-Loss repo." >&2
    echo "       Submit from the repo root, or pass REPO_DIR=/path/to/repo." >&2
    exit 2
fi
cd "${REPO_DIR}"

# /home is 100% full on this cluster -- every cache must live under the project.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mnt/projects/jg/kaifany/.uv-cache}"
export HF_HOME="${HF_HOME:-/mnt/projects/jg/kaifany/.hf}"
export TORCH_HOME="${TORCH_HOME:-/mnt/projects/jg/kaifany/.torch}"
# /home being full also breaks torch's CUDA kernel cache, which silently
# disables kernel caching ("Specified kernel cache directory could not be
# created"). Point it at the project too.
export PYTORCH_KERNEL_CACHE_PATH="${PYTORCH_KERNEL_CACHE_PATH:-/mnt/projects/jg/kaifany/.torch/kernels}"
mkdir -p "${PYTORCH_KERNEL_CACHE_PATH}" 2>/dev/null || true

# The launchers default PY_BIN to the original author's conda path.
export PY_BIN="${PY_BIN:-${REPO_DIR}/.venv/bin/python}"
export DATA_PATH="${DATA_PATH:-/mnt/projects/jg/kaifany/dataset/imagenet}"
export DATA_ROOT="${DATA_ROOT:-${DATA_PATH}}"

# Slurm's cgroup remaps the allocated GPUs to RELATIVE indices, so a job always
# sees 0..N-1 regardless of which physical devices it got. The launchers' default
# GPUS=4,5,6,7 would name devices that do not exist inside the allocation.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NGPU="$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
else
    NGPU="$(nvidia-smi -L | wc -l)"
fi
(( NGPU > 0 )) || { echo "ERROR: no GPUs in this allocation" >&2; exit 2; }
export GPUS="$(seq -s, 0 $((NGPU - 1)))"
export GPU="${GPUS%%,*}"          # scripts/smoke_vlm_delta.sh takes a single GPU
export CUDA_GPUS="${GPUS}"        # scripts/compute_imagenet100_fd_stats.sh
export GPUS_PER_NODE="${NGPU}"    # the upstream table_*.sh launchers

# The launchers background themselves with setsid by default. Under sbatch the
# batch script would then exit immediately and Slurm would kill the job.
export RUN_FOREGROUND=1

# PreemptType=preempt/qos with GraceTime=0: a normal-QOS job on `batch` can be
# killed and requeued at any instant. On an actual requeue, continue the run
# instead of refusing to touch its work dir.
if (( ${SLURM_RESTART_COUNT:-0} > 0 )); then
    echo "### requeued (restart #${SLURM_RESTART_COUNT}) -- setting RESUME=1"
    export RESUME=1
fi

echo "### job ${SLURM_JOB_ID:-?} on $(hostname), ${NGPU} GPU(s): ${GPUS}"
echo "### physical: ${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-?}}   python: ${PY_BIN}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "### running: $*"

[[ $# -ge 1 ]] || { echo "ERROR: pass the launcher (or command) to run" >&2; exit 2; }
# A path to a .sh launcher runs under bash; anything else runs as a raw command,
# so a bare `python -m torch.distributed.run ... train_vlm_p_head.py` also works.
if [[ "$1" == *.sh && -f "$1" ]]; then
    exec bash "$@"
else
    exec "$@"
fi
