#!/usr/bin/env bash
# Fit the real-data class-conditional GMM (the "p" side of the log p / log q
# loss) on the 100-class scaling-probe subset.
#
# This is NOT the 1000-class file masked down to 100. The whitening PCA, the
# pooled within-class covariance and the log_softmax denominator are all defined
# over the subset, which is what makes -log p(c|x) a 100-way problem rather than
# a 1000-way one against 900 classes the generator never draws -- those would
# leave their mu_c^q parked on mu_c^p as spurious attractors in the softmax.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# shellcheck source=scripts/imagenet100_class_ids.sh
source "${SCRIPT_DIR}/imagenet100_class_ids.sh"

: "${CUDA_GPUS:=4,5,6,7}"
: "${MASTER_PORT:=29563}"
: "${PY_BIN:=/home/nvidia/miniconda3/envs/fdloss/bin/python}"
: "${DATA_PATH:=/data/dataset/imagenet}"
: "${OUTPUT_DIR:=data/fid_stats}"
: "${PCA_DIM:=128}"
: "${BATCH_SIZE:=128}"
: "${NUM_WORKERS:=8}"
# Judge to fit in. Must match --fd_gmm_judge and the judge's target size.
: "${MODEL:=inception}"
: "${TARGET_SIZE:=}"

CLASS_IDS=("${IMAGENET100_CLASS_IDS[@]}")

IFS=',' read -r -a GPU_IDS <<< "${CUDA_GPUS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"

CMD=(
    "${PY_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_port="${MASTER_PORT}"
    compute_class_stats.py
    --model "${MODEL}"
    --data_path "${DATA_PATH}"
    --img_size 256
    --pca_dim "${PCA_DIM}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --output_dir "${OUTPUT_DIR}"
    --class_ids "${CLASS_IDS[@]}"
)
if [[ -n "${TARGET_SIZE}" ]]; then
    CMD+=(--target_size "${TARGET_SIZE}")
fi

echo "Fitting ${MODEL} class GMM on ${#CLASS_IDS[@]} classes (k=${PCA_DIM}) -> ${OUTPUT_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_GPUS}" "${CMD[@]}"
