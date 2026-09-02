#!/usr/bin/env bash
# Fresh full-ImageNet v2 takeoff run for the unconditional JiT-B base.
#
# This keeps the successful 20-class Arm C objective (GMM log-p + log-q), but
# fixes the two full-1000-class failure modes:
#   * q class means use beta=.999, above the measured estimator noise floor;
#   * the class loss is divided by the fixed log(C), rather than by its current
#     magnitude, so a hard example cannot turn its own gradient down.
#
# Four GPUs are used at batch 24/GPU to preserve the prior global batch of 96.
# The takeoff gate aborts at step 10k if either class structure has not cleared
# 2x its finite-EMA noise floor or the last 50 raw diagnostic cosines do not
# average below zero.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

: "${GPUS:=4,5,6,7}"
: "${MASTER_PORT:=29558}"
: "${PY_BIN:=/home/nvidia/miniconda3/envs/fdloss/bin/python}"
: "${DATA_PATH:=/data/dataset/imagenet}"
: "${START_CKPT:=checkpoints/base/JiT-B-uncond.pth}"
: "${SIGLIP_STATS:=data/fid_stats/vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz}"
: "${MAE_STATS:=data/fid_stats/vit_large_patch16_224_mae_in256_t224_stats.npz}"
: "${INCEPTION_STATS:=data/fid_stats/guided_diffusion_stats.npz}"
: "${GMM_STATS:=data/fid_stats/inception_in256_t256_classgmm_k128.npz}"
: "${OUTPUT_DIR:=work_dirs}"
: "${PROJECT:=JiT_uncond_gmm_1000class_v2}"
: "${EXP_NAME:=jitB_uncond_gmm1000_armC_v2_takeoff}"
: "${LOG_DIR:=sweep_logs}"

# GMM p+q objective. q is enabled by intentionally omitting --fd_gmm_no_q.
# Exact production geometry calibration (4 GPUs, batch 24/GPU, global 96):
# w=.0060 measured raw grad_ratio_p_fd mean=.420619, median=.423600 over
# steps 900-1200. Rescale .0060 * .35 / .420619 = .004993 -> .0050,
# predicting .3505 at the sustained target.
: "${WEIGHT:=0.0050}"
: "${LAMBDA_CLS:=1.0}"
: "${LAMBDA_ENT:=1.0}"
: "${CLS_NORMALIZATION:=log_classes}"
: "${CLS_CAP:=-0.69}"
: "${GMM_TEMP:=10}"
: "${GMM_WARMUP:=0}"
: "${GMM_RAMP:=500}"
: "${P_SHRINK:=0.75}"
: "${Q_SHRINK:=0.25}"
: "${GMM_MEAN_EMA_BETA:=0.999}"
: "${GMM_COV_EMA_BETA:=0.999}"
: "${GMM_BOOTSTRAP:=50000}"

# Abort when ANY enabled takeoff criterion fails at this zero-based step.
# Set TAKEOFF_GATE_STEP=-1 for a short calibration run.
: "${TAKEOFF_GATE_STEP:=10000}"
: "${TAKEOFF_SPREAD_MULT:=2.0}"
: "${TAKEOFF_COS_THRESHOLD:=0.0}"
: "${TAKEOFF_COS_WINDOW:=50}"
: "${TAKEOFF_LOGIC:=any}"

: "${FD_EMA_BETA:=0.99}"
: "${EPOCHS:=40}"
: "${STEPS_PER_EPOCH:=1250}"
: "${BATCH_SIZE:=24}"
: "${QUEUE_SIZE:=50000}"
: "${NUM_EVAL_IMAGES:=50000}"
: "${EVAL_FREQ:=4}"       # 4 * 1250 = every 5,000 steps
: "${SAVE_FREQ:=4}"
: "${PRINT_FREQ:=20}"
: "${VIS_FREQ:=2}"
: "${ONLINE_EVAL:=1}"
: "${DISABLE_VIS:=0}"
: "${AUTO_RESUME:=0}"
: "${RUN_FOREGROUND:=0}"

# Visualization only. Training and evaluation still sample all 1000 labels.
VIS_CLASSES=(0 9 88 130 207 279 281 340 360 387 404 417 444 555 569 817 920 949 974 979)
FD_MODELS=(vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception)
FD_POOLS=(cls cls cls)
FD_SIZES=(224 224 256)
FD_STATS=("${SIGLIP_STATS}" "${MAE_STATS}" "${INCEPTION_STATS}")

die() {
    echo "ERROR: $*" >&2
    exit 2
}

is_nonnegative_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_bool() {
    [[ "$1" == "0" || "$1" == "1" ]]
}

[[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] \
    || die "GPUS must be a comma-separated list of numeric GPU IDs (got '${GPUS}')"
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] \
    || die "MASTER_PORT must be an integer (got '${MASTER_PORT}')"
(( MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) \
    || die "MASTER_PORT must be between 1 and 65535 (got '${MASTER_PORT}')"
is_nonnegative_int "${EPOCHS}" && (( EPOCHS > 0 )) \
    || die "EPOCHS must be a positive integer (got '${EPOCHS}')"
is_nonnegative_int "${STEPS_PER_EPOCH}" && (( STEPS_PER_EPOCH > 0 )) \
    || die "STEPS_PER_EPOCH must be a positive integer (got '${STEPS_PER_EPOCH}')"
is_nonnegative_int "${BATCH_SIZE}" && (( BATCH_SIZE > 0 )) \
    || die "BATCH_SIZE must be a positive integer (got '${BATCH_SIZE}')"
is_nonnegative_int "${QUEUE_SIZE}" && (( QUEUE_SIZE > 0 )) \
    || die "QUEUE_SIZE must be a positive integer (got '${QUEUE_SIZE}')"
is_nonnegative_int "${GMM_BOOTSTRAP}" && (( GMM_BOOTSTRAP > 0 )) \
    || die "GMM_BOOTSTRAP must be a positive integer (got '${GMM_BOOTSTRAP}')"
is_nonnegative_int "${NUM_EVAL_IMAGES}" && (( NUM_EVAL_IMAGES > 0 )) \
    || die "NUM_EVAL_IMAGES must be a positive integer (got '${NUM_EVAL_IMAGES}')"
is_nonnegative_int "${EVAL_FREQ}" \
    || die "EVAL_FREQ must be a non-negative integer (got '${EVAL_FREQ}')"
is_nonnegative_int "${SAVE_FREQ}" \
    || die "SAVE_FREQ must be a non-negative integer (got '${SAVE_FREQ}')"
is_nonnegative_int "${PRINT_FREQ}" && (( PRINT_FREQ > 0 )) \
    || die "PRINT_FREQ must be a positive integer (got '${PRINT_FREQ}')"
is_nonnegative_int "${VIS_FREQ}" \
    || die "VIS_FREQ must be a non-negative integer (got '${VIS_FREQ}')"
[[ "${TAKEOFF_GATE_STEP}" =~ ^-1$|^[0-9]+$ ]] \
    || die "TAKEOFF_GATE_STEP must be -1 or a non-negative integer (got '${TAKEOFF_GATE_STEP}')"
is_nonnegative_int "${TAKEOFF_COS_WINDOW}" && (( TAKEOFF_COS_WINDOW > 0 )) \
    || die "TAKEOFF_COS_WINDOW must be a positive integer (got '${TAKEOFF_COS_WINDOW}')"
[[ "${TAKEOFF_LOGIC}" == "any" || "${TAKEOFF_LOGIC}" == "all" ]] \
    || die "TAKEOFF_LOGIC must be any or all (got '${TAKEOFF_LOGIC}')"
is_bool "${ONLINE_EVAL}" || die "ONLINE_EVAL must be 0 or 1 (got '${ONLINE_EVAL}')"
is_bool "${DISABLE_VIS}" || die "DISABLE_VIS must be 0 or 1 (got '${DISABLE_VIS}')"
is_bool "${AUTO_RESUME}" || die "AUTO_RESUME must be 0 or 1 (got '${AUTO_RESUME}')"
is_bool "${RUN_FOREGROUND}" || die "RUN_FOREGROUND must be 0 or 1 (got '${RUN_FOREGROUND}')"
[[ "${AUTO_RESUME}" == "0" ]] \
    || die "this is a fresh-run launcher; AUTO_RESUME must remain 0"
[[ -n "${PROJECT}" && -n "${EXP_NAME}" ]] \
    || die "PROJECT and EXP_NAME must be non-empty"

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
NPROC_PER_NODE="${#GPU_IDS[@]}"
GLOBAL_BATCH=$(( BATCH_SIZE * NPROC_PER_NODE ))
TOTAL_STEPS=$(( EPOCHS * STEPS_PER_EPOCH ))

[[ -x "${PY_BIN}" ]] || die "Python executable not found: ${PY_BIN}"
[[ -f conditional_main_fd_gmm.py ]] \
    || die "missing training entry point: conditional_main_fd_gmm.py"
[[ -d "${DATA_PATH}/train" ]] || die "ImageNet train split not found: ${DATA_PATH}/train"
[[ -d "${DATA_PATH}/val" ]] || die "ImageNet val split not found: ${DATA_PATH}/val"
[[ -f "${START_CKPT}" ]] || die "unconditional JiT-B checkpoint not found: ${START_CKPT}"
[[ -f "${GMM_STATS}" ]] || die "full 1000-class GMM stats not found: ${GMM_STATS}"
for stats_path in "${FD_STATS[@]}"; do
    [[ -f "${stats_path}" ]] || die "full-ImageNet FD stats not found: ${stats_path}"
done

# These are v2-only semantics. Refuse to allocate GPUs against an older entry
# point that would silently fall back to the failed objective.
for required_flag in \
    --fd_gmm_cls_normalization \
    --fd_gmm_takeoff_gate_step \
    --fd_gmm_takeoff_spread_mult \
    --fd_gmm_takeoff_cos_threshold \
    --fd_gmm_takeoff_cos_window \
    --fd_gmm_takeoff_logic; do
    grep -q -- "${required_flag}" conditional_main_fd_gmm.py \
        || die "conditional_main_fd_gmm.py does not support ${required_flag}"
done
grep -q -- 'gmm_class_mean_spread_noise_floor' frechet_distance/gmm.py \
    || die "frechet_distance/gmm.py does not expose the finite-EMA spread noise floor"

RUN_DIR="${OUTPUT_DIR}/${PROJECT}/${EXP_NAME}"
if [[ -e "${RUN_DIR}" ]]; then
    die "fresh work directory already exists: ${RUN_DIR}; choose a unique PROJECT/EXP_NAME"
fi

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${EXP_NAME}.out"
if [[ -e "${LOG_PATH}" ]]; then
    die "fresh log path already exists: ${LOG_PATH}; choose a unique EXP_NAME"
fi

CMD=(
    "${PY_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_port="${MASTER_PORT}"
    conditional_main_fd_gmm.py
    --data_path "${DATA_PATH}"
    --num_classes 1000
    --load_from "${START_CKPT}"
    --output_dir "${OUTPUT_DIR}"
    --project "${PROJECT}"
    --exp_name "${EXP_NAME}"
    --batch_size "${BATCH_SIZE}"
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0
    --ema_type edm --num_sampling_steps 1
    --eval_bsz 256 --num_images_for_eval_and_search "${NUM_EVAL_IMAGES}"
    --vis_freq "${VIS_FREQ}" --eval_freq "${EVAL_FREQ}"
    --print_freq "${PRINT_FREQ}" --milestone_interval 10 --save_freq "${SAVE_FREQ}"
    --epochs "${EPOCHS}" --steps_per_epoch "${STEPS_PER_EPOCH}"
    --warmup_epochs 0
    --lr 1e-5 --lr_sched constant --min_lr 0.0
    --fd_eigvalsh --fd_ema_beta "${FD_EMA_BETA}"
    --queue_size "${QUEUE_SIZE}"
    --fd_repr_models "${FD_MODELS[@]}"
    --fd_repr_pool_types "${FD_POOLS[@]}"
    --fd_target_sizes "${FD_SIZES[@]}"
    --fd_repr_stats_paths "${FD_STATS[@]}"
    --fid_stats_path "${INCEPTION_STATS}"
    --class_of_interest "${VIS_CLASSES[@]}"
    --fd_gmm
    --fd_gmm_stats_path "${GMM_STATS}"
    --fd_gmm_judge inception
    --fd_gmm_pca_dim 128
    --fd_gmm_mode density
    --fd_gmm_weight "${WEIGHT}"
    --fd_gmm_lambda_cls "${LAMBDA_CLS}"
    --fd_gmm_lambda_ent "${LAMBDA_ENT}"
    --fd_gmm_cls_normalization "${CLS_NORMALIZATION}"
    --fd_gmm_cls_cap "${CLS_CAP}"
    --fd_gmm_temp "${GMM_TEMP}"
    --fd_gmm_p_shrinkage "${P_SHRINK}"
    --fd_gmm_q_shrinkage "${Q_SHRINK}"
    --fd_gmm_ema_beta "${GMM_MEAN_EMA_BETA}"
    --fd_gmm_cov_ema_beta "${GMM_COV_EMA_BETA}"
    --fd_gmm_bootstrap "${GMM_BOOTSTRAP}"
    --fd_gmm_warmup_steps "${GMM_WARMUP}"
    --fd_gmm_ramp_steps "${GMM_RAMP}"
    --fd_gmm_takeoff_gate_step "${TAKEOFF_GATE_STEP}"
    --fd_gmm_takeoff_spread_mult "${TAKEOFF_SPREAD_MULT}"
    --fd_gmm_takeoff_cos_threshold "${TAKEOFF_COS_THRESHOLD}"
    --fd_gmm_takeoff_cos_window "${TAKEOFF_COS_WINDOW}"
    --fd_gmm_takeoff_logic "${TAKEOFF_LOGIC}"
    --cond_probe
    --disable_wandb
)

if [[ "${ONLINE_EVAL}" == "1" ]]; then
    CMD+=(--online_eval)
fi
if [[ "${DISABLE_VIS}" == "1" ]]; then
    CMD+=(--disable_vis)
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # Intended for simple additional flags/values in calibration runs.
    read -r -a EXTRA <<< "${EXTRA_ARGS}"
    CMD+=("${EXTRA[@]}")
fi

echo "Experiment:       ${PROJECT}/${EXP_NAME} (fresh full-1000-class Arm C v2)"
echo "Repository:       ${REPO_DIR}"
echo "GPUs:             ${GPUS} (${NPROC_PER_NODE} ranks)"
echo "Batch:            ${BATCH_SIZE}/GPU, global ${GLOBAL_BATCH}"
echo "Master port:      ${MASTER_PORT}"
echo "Start checkpoint: ${START_CKPT}"
echo "Classes:          all 1000 (VIS_CLASSES are visualization-only)"
echo "Iterations:       ${TOTAL_STEPS} (${EPOCHS} x ${STEPS_PER_EPOCH})"
echo "Optimizer:        AdamW, constant lr=1e-5, no warmup"
echo "FD window:        beta=${FD_EMA_BETA}, queue=${QUEUE_SIZE}"
echo "GMM p+q:          weight=${WEIGHT}, cls=${LAMBDA_CLS}, ent=${LAMBDA_ENT}, cls_norm=${CLS_NORMALIZATION}"
echo "GMM calibration:  T=${GMM_TEMP}, cap=${CLS_CAP}, p/q shrink=${P_SHRINK}/${Q_SHRINK}"
echo "GMM estimators:   mean_beta=${GMM_MEAN_EMA_BETA}, cov_beta=${GMM_COV_EMA_BETA}, bootstrap=${GMM_BOOTSTRAP}"
echo "GMM schedule:     warmup=${GMM_WARMUP}, ramp=${GMM_RAMP}"
echo "Takeoff gate:     step=${TAKEOFF_GATE_STEP}, spread/noise>=${TAKEOFF_SPREAD_MULT}, mean(last ${TAKEOFF_COS_WINDOW} raw cos)<${TAKEOFF_COS_THRESHOLD}, abort=${TAKEOFF_LOGIC} failure"
echo "Online eval:      ${ONLINE_EVAL}; ${NUM_EVAL_IMAGES} images every $(( EVAL_FREQ * STEPS_PER_EPOCH )) steps"
echo "Visualization:    disabled=${DISABLE_VIS}, frequency=${VIS_FREQ} epochs"
echo "Fresh run:        auto_resume=${AUTO_RESUME}, workdir=${RUN_DIR}"
echo "Log:              ${LOG_PATH}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${RUN_FOREGROUND}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${GPUS}" "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
else
    CUDA_VISIBLE_DEVICES="${GPUS}" setsid "${CMD[@]}" \
        >"${LOG_PATH}" 2>&1 < /dev/null &
    pid=$!
    echo "Started PID ${pid}. Follow with: tail -f ${LOG_PATH}"
fi
