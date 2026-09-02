#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

: "${CUDA_GPUS:=5,6,7}"
: "${MASTER_PORT:=29514}"
: "${PY_BIN:=/home/nvidia/miniconda3/envs/fdloss/bin/python}"
: "${DATA_PATH:=/data/dataset/imagenet}"
: "${OUTPUT_DIR:=work_dirs/mae_probe_vitl_cls}"
: "${BATCH_SIZE:=128}"
: "${EPOCHS:=90}"
: "${WARMUP_EPOCHS:=10}"
: "${POOL_TYPE:=cls}"
: "${HEAD_NORM:=bn}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_GPUS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"

if [[ ! -x "${PY_BIN}" ]]; then
    echo "ERROR: Python executable not found: ${PY_BIN}" >&2
    exit 2
fi
if [[ ! -d "${DATA_PATH}/train" || ! -d "${DATA_PATH}/val" ]]; then
    echo "ERROR: expected ${DATA_PATH}/train and ${DATA_PATH}/val" >&2
    exit 2
fi

echo "Training frozen-MAE linear probe"
echo "GPUs: ${CUDA_GPUS}; output: ${OUTPUT_DIR}; pool: ${POOL_TYPE}"

CUDA_VISIBLE_DEVICES="${CUDA_GPUS}" "${PY_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    train_mae_linear_probe.py \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name vit_large_patch16_224.mae \
    --pool_type "${POOL_TYPE}" \
    --target_size 224 \
    --head_norm "${HEAD_NORM}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --base_lr 0.1 \
    --optimizer lars \
    --weight_decay 0 \
    --dtype bf16
