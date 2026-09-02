#!/usr/bin/env bash
# Phase 1 of the *answer-state* VLM-delta trial: fit and freeze the real-data p
# head on Qwen2.5-VL-7B's hidden state at the position it would answer
#
#     [<image>] "What is the ImageNet class of this image? Answer:"
#
# from.  Everything downstream of the feature -- the delta term, the replay
# buffer, the EMA teacher, the calibration -- is unchanged from the SigLIP
# trial; only z changes.
#
# THE ONE NUMBER THIS PRODUCES
#
#   held-out real-val top-1 of the linear head.  SigLIP-SO400M's CLS token gets
#   0.982 (full val) / 0.984 (holdout half) on this exact 100-class subset, so
#   that is the bar.  A worse p head makes log q - log p a worse teacher, and no
#   amount of "but it is a real VLM" makes up for it.
#
# WHY SEVERAL LAYERS AT ONCE
#
#   The last layer of a decoder is next-token-specialised; a late-middle layer
#   is often the better linear probe.  Every layer in PROBE_LAYERS comes out of
#   the SAME VLM forward, so the sweep costs six fp16 arrays and zero extra GPU
#   time, and the layer is measured instead of guessed.  Selection is on the
#   calibration half of real val; the holdout half is reported untouched.
#
# COST: ~46 img/s/GPU forward-only at 256px.  100 classes x ~1300 train images
# x 2 views + 5000 val is ~265k forwards, i.e. ~30-40 min on 3 A100s.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# shellcheck source=scripts/imagenet100_class_ids.sh
source "${SCRIPT_DIR}/imagenet100_class_ids.sh"

: "${GPUS:=1,2,3}"
: "${MASTER_PORT:=29614}"
: "${PY_BIN:=/home/nvidia/miniconda3/envs/fdloss/bin/python}"
: "${DATA_PATH:=/data/dataset/imagenet}"
: "${OUTPUT_DIR:=work_dirs/vlm_p_head_qwen_answer_c100}"
: "${VLM_MODEL:=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"

# The question. It must not name the class: z has to be the model's answer, not
# a re-encoding of a label handed to it. Changing this invalidates the feature
# cache and every head fitted on it (the prompt hash is part of both keys).
: "${PROMPT:=What is the ImageNet class of this image? Answer:}"

# hidden_states indices, HuggingFace convention: 0 = embeddings, k = output of
# decoder layer k-1, 28 = the final post-RMSNorm state the LM head consumes.
: "${PROBE_LAYERS:=14 18 21 24 26 28}"
# Empty = save whichever candidate wins on the calibration half.
: "${PIN_LAYER:=}"

# 256px center crop: the same geometry the generator's samples will have. The
# head must never be fitted on a view the generated images cannot produce.
: "${INPUT_SIZE:=256}"
: "${TRAIN_VIEWS:=2}"
: "${MAX_PER_CLASS:=0}"
: "${EXTRACT_BATCH_SIZE:=64}"
: "${NUM_WORKERS:=10}"

# lm_head seeds W from the LM-head row of each class name's first token and
# reports that head's top-1 BEFORE any fitting -- a zero-shot readout of how
# label-aligned the answer state already is.
: "${HEAD_INIT:=lm_head}"

: "${EPOCHS:=100}"
: "${HEAD_BATCH_SIZE:=1024}"
: "${LR:=1e-3}"
: "${WEIGHT_DECAY:=1e-4}"
: "${SEED:=0}"
: "${DTYPE:=bf16}"
: "${LOG_DIR:=sweep_logs}"
: "${RUN_FOREGROUND:=0}"

die() { echo "ERROR: $*" >&2; exit 2; }

[[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "GPUS must be comma-separated GPU ids (got '${GPUS}')"
[[ -x "${PY_BIN}" ]] || die "Python executable not found: ${PY_BIN}"
[[ -d "${VLM_MODEL}" ]] || die "local Qwen2.5-VL checkpoint not found: ${VLM_MODEL}"
[[ -d "${DATA_PATH}/train" ]] || die "ImageNet train split not found: ${DATA_PATH}/train"
[[ -d "${DATA_PATH}/val" ]] || die "ImageNet val split not found: ${DATA_PATH}/val"
[[ -f train_vlm_p_head.py ]] || die "missing entry point: train_vlm_p_head.py"
[[ "${PROMPT}" != *"{"* ]] || die "PROMPT must be a literal question, not a format string"

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
NPROC_PER_NODE="${#GPU_IDS[@]}"
CLASS_IDS=("${IMAGENET100_CLASS_IDS[@]}")
read -r -a LAYERS <<< "${PROBE_LAYERS}"

mkdir -p "${LOG_DIR}"
RUN_TAG="$(basename "${OUTPUT_DIR}")"
LOG_PATH="${LOG_DIR}/${RUN_TAG}.out"

CMD=(
    "${PY_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_port="${MASTER_PORT}"
    train_vlm_p_head.py
    --vlm_backend qwen_answer_state
    --vlm_model_name "${VLM_MODEL}"
    --vlm_prompt "${PROMPT}"
    --vlm_probe_layers "${LAYERS[@]}"
    --vlm_input_size "${INPUT_SIZE}"
    --head_init "${HEAD_INIT}"
    --data_path "${DATA_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --class_ids "${CLASS_IDS[@]}"
    --train_views "${TRAIN_VIEWS}"
    --max_per_class "${MAX_PER_CLASS}"
    --extract_batch_size "${EXTRACT_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --epochs "${EPOCHS}"
    --head_batch_size "${HEAD_BATCH_SIZE}"
    --lr "${LR}"
    --weight_decay "${WEIGHT_DECAY}"
    --seed "${SEED}"
    --dtype "${DTYPE}"
)
if [[ -n "${PIN_LAYER}" ]]; then
    CMD+=(--vlm_layer "${PIN_LAYER}")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA <<< "${EXTRA_ARGS}"
    CMD+=("${EXTRA[@]}")
fi

echo "Output:        ${OUTPUT_DIR}"
echo "VLM:           ${VLM_MODEL}"
echo "Prompt:        ${PROMPT}"
echo "Probe layers:  ${PROBE_LAYERS} (from one forward each; winner saved)"
echo "Classes:       ${#CLASS_IDS[@]} (${CLASS_IDS[0]} ${CLASS_IDS[1]} ... ${CLASS_IDS[-1]}), stride-10 subset"
echo "Views:         ${TRAIN_VIEWS} at ${INPUT_SIZE}px center crop"
echo "GPUs:          ${GPUS} (${NPROC_PER_NODE} ranks)"
echo "Bar to beat:   SigLIP CLS = 0.982 full-val / 0.984 holdout top-1"
echo "Log:           ${LOG_PATH}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${RUN_FOREGROUND}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${GPUS}" "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
else
    CUDA_VISIBLE_DEVICES="${GPUS}" setsid "${CMD[@]}" \
        >"${LOG_PATH}" 2>&1 < /dev/null &
    echo "Started PID $!. Follow with: tail -f ${LOG_PATH}"
fi
