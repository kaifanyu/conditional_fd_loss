#!/usr/bin/env bash
# Build real-data SigLIP, MAE, and Inception statistics for the same 20
# ImageNet classes used by the conditional diagnostic.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

: "${CUDA_GPUS:=4,5,6,7}"
: "${MASTER_PORT:=29518}"
: "${PY_BIN:=/home/nvidia/miniconda3/envs/fdloss/bin/python}"
: "${DATA_PATH:=/data/dataset/imagenet}"
: "${STATS_DIR:=data/fid_stats/imagenet20_v1}"
: "${BATCH_SIZE:=64}"
: "${NUM_WORKERS:=8}"
: "${OVERWRITE:=0}"

# Visually distinct classes: tench, ostrich, macaw, flamingo, golden retriever,
# Arctic fox, tabby, zebra, otter, red panda, airliner, balloon, tandem bicycle,
# fire engine, garbage truck, sports car, traffic light, strawberry, geyser,
# and valley.
CLASS_IDS=(0 9 88 130 207 279 281 340 360 387 404 417 444 555 569 817 920 949 974 979)

SIGLIP_STATS="${STATS_DIR}/siglip_cls.npz"
MAE_STATS="${STATS_DIR}/mae_cls.npz"
INCEPTION_STATS="${STATS_DIR}/inception.npz"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_GPUS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"

if [[ ! -x "${PY_BIN}" ]]; then
    echo "ERROR: Python executable not found: ${PY_BIN}" >&2
    exit 2
fi
if [[ ! -d "${DATA_PATH}/train" ]]; then
    echo "ERROR: ImageNet train directory not found: ${DATA_PATH}/train" >&2
    exit 2
fi

mkdir -p "${STATS_DIR}"

compute_one() {
    local model="$1"
    local target_size="$2"
    local output_path="$3"
    local port="$4"

    if [[ -f "${output_path}" && "${OVERWRITE}" != "1" ]]; then
        echo "Using existing stats: ${output_path}"
        return
    fi

    echo "Computing ${model} statistics -> ${output_path}"
    CUDA_VISIBLE_DEVICES="${CUDA_GPUS}" "${PY_BIN}" -m torch.distributed.run \
        --standalone \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port="${port}" \
        compute_repr_stats.py \
        --model "${model}" \
        --data_path "${DATA_PATH}" \
        --img_size 256 \
        --target_size "${target_size}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --output_dir "${STATS_DIR}" \
        --output_name "$(basename -- "${output_path}")" \
        --class_ids "${CLASS_IDS[@]}"
}

compute_one "vit_so400m_patch16_siglip_256.v2_webli" 224 "${SIGLIP_STATS}" "${MASTER_PORT}"
compute_one "vit_large_patch16_224.mae" 224 "${MAE_STATS}" "$((MASTER_PORT + 1))"
compute_one "inception" 256 "${INCEPTION_STATS}" "$((MASTER_PORT + 2))"

echo "ImageNet-20 reference statistics are ready in ${STATS_DIR}"
