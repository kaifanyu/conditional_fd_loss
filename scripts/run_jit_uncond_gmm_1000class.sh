#!/usr/bin/env bash
# Teach all 1000 ImageNet classes to the de-conditioned JiT-B with the
# class-conditional GMM p+q objective used by the 20-class Arm C run.
#
#   L = L_FD(siglip, mae, inception)
#     + w(s) * [ E[-log p(c|x)] + E[log q(x|c) - log p(x|c)] ]
#
# This is the full-class configuration:
#   * labels are sampled uniformly from all 1000 classes;
#   * FD and evaluation use the standard full-ImageNet reference statistics;
#   * the GMM uses the full C=1000 inception PCA-128 fit;
#   * q is enabled (density mode), making this the analogue of Arm C.
#
# The posterior calibration does not transfer unchanged from 20 classes.
# T=10 with cap=-0.69 is the measured full-class operating point. The online
# q estimator also uses separate horizons: fast per-class means (beta=.99)
# and a statistically stable tied covariance (beta=.999).
#
# Full-class weight calibration, measured on this exact configuration:
#   calibration experiment: w0.0066
#   steps 600-700 reconstructed raw grad_ratio_p_fd:
#       mean=0.3505, median=0.3411
#   rescale: 0.0066 * 0.20 / 0.3505 = 0.00377 -> WEIGHT=0.0038
# Verify the production run remains in the 0.10-0.40 band around steps 600-700.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

: "${GPUS:=3,4}"
: "${MASTER_PORT:=29557}"
: "${PY_BIN:=/home/nvidia/miniconda3/envs/fdloss/bin/python}"
: "${DATA_PATH:=/data/dataset/imagenet}"
: "${START_CKPT:=checkpoints/base/JiT-B-uncond.pth}"
: "${SIGLIP_STATS:=data/fid_stats/vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz}"
: "${MAE_STATS:=data/fid_stats/vit_large_patch16_224_mae_in256_t224_stats.npz}"
: "${INCEPTION_STATS:=data/fid_stats/guided_diffusion_stats.npz}"
: "${GMM_STATS:=data/fid_stats/inception_in256_t256_classgmm_k128.npz}"
: "${OUTPUT_DIR:=work_dirs}"
: "${PROJECT:=JiT_uncond_gmm_1000class}"
: "${EXP_NAME:=jitB_uncond_gmm1000_armC_logp_logq}"
: "${LOG_DIR:=sweep_logs}"

# GMM p+q term. q remains enabled by intentionally omitting --fd_gmm_no_q.
: "${WEIGHT:=0.0038}"
: "${LAMBDA_CLS:=1.0}"
: "${LAMBDA_ENT:=1.0}"
: "${CLS_CAP:=-0.69}"
: "${GMM_TEMP:=10}"
: "${GMM_WARMUP:=0}"
: "${GMM_RAMP:=500}"
: "${P_SHRINK:=0.75}"
: "${Q_SHRINK:=0.25}"
: "${GMM_MEAN_EMA_BETA:=0.99}"
: "${GMM_COV_EMA_BETA:=0.999}"

: "${FD_EMA_BETA:=0.99}"
: "${EPOCHS:=40}"
: "${STEPS_PER_EPOCH:=1250}"
: "${BATCH_SIZE:=48}"
: "${QUEUE_SIZE:=50000}"
: "${NUM_EVAL_IMAGES:=50000}"
: "${EVAL_FREQ:=5}"
: "${SAVE_FREQ:=5}"
: "${PRINT_FREQ:=20}"
: "${VIS_FREQ:=2}"
: "${AUTO_RESUME:=1}"
: "${ONLINE_EVAL:=1}"
: "${RUN_FOREGROUND:=0}"

# Visualization only. Without --force_class_of_interest, training and online
# evaluation still cover all 1000 classes.
VIS_CLASSES=(0 9 88 130 207 279 281 340 360 387 404 417 444 555 569 817 920 949 974 979)
FD_MODELS=(vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception)
FD_POOLS=(cls cls cls)
FD_SIZES=(224 224 256)
FD_STATS=("${SIGLIP_STATS}" "${MAE_STATS}" "${INCEPTION_STATS}")

die() {
    echo "ERROR: $*" >&2
    exit 2
}

[[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] \
    || die "GPUS must be a comma-separated list of numeric GPU IDs (got '${GPUS}')"
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] \
    || die "MASTER_PORT must be an integer (got '${MASTER_PORT}')"
(( MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) \
    || die "MASTER_PORT must be between 1 and 65535 (got '${MASTER_PORT}')"
[[ "${NUM_EVAL_IMAGES}" =~ ^[1-9][0-9]*$ ]] \
    || die "NUM_EVAL_IMAGES must be a positive integer (got '${NUM_EVAL_IMAGES}')"
[[ "${RUN_FOREGROUND}" == "0" || "${RUN_FOREGROUND}" == "1" ]] \
    || die "RUN_FOREGROUND must be 0 or 1 (got '${RUN_FOREGROUND}')"

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
NPROC_PER_NODE="${#GPU_IDS[@]}"

[[ -x "${PY_BIN}" ]] || die "Python executable not found: ${PY_BIN}"
[[ -f conditional_main_fd_gmm.py ]] || die "missing training entry point: conditional_main_fd_gmm.py"
[[ -d "${DATA_PATH}" ]] || die "ImageNet root not found: ${DATA_PATH}"
[[ -d "${DATA_PATH}/train" ]] || die "ImageNet train split not found: ${DATA_PATH}/train"
[[ -d "${DATA_PATH}/val" ]] || die "ImageNet val split not found: ${DATA_PATH}/val"
[[ -f "${START_CKPT}" ]] || die "de-conditioned JiT-B checkpoint not found: ${START_CKPT}"
[[ -f "${GMM_STATS}" ]] || die "full 1000-class GMM stats not found: ${GMM_STATS}"
for stats_path in "${FD_STATS[@]}"; do
    [[ -f "${stats_path}" ]] || die "full-ImageNet FD stats not found: ${stats_path}"
done

# Fail before allocating GPUs if the split covariance-EMA integration has not
# landed in the training entry point.
if ! grep -q -- '--fd_gmm_cov_ema_beta' conditional_main_fd_gmm.py; then
    die "conditional_main_fd_gmm.py does not support --fd_gmm_cov_ema_beta"
fi

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${EXP_NAME}.out"

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
    --lr 1e-5 --lr_sched cosine --min_lr 0.0
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
    --fd_gmm_cls_cap "${CLS_CAP}"
    --fd_gmm_temp "${GMM_TEMP}"
    --fd_gmm_p_shrinkage "${P_SHRINK}"
    --fd_gmm_q_shrinkage "${Q_SHRINK}"
    --fd_gmm_ema_beta "${GMM_MEAN_EMA_BETA}"
    --fd_gmm_cov_ema_beta "${GMM_COV_EMA_BETA}"
    --fd_gmm_warmup_steps "${GMM_WARMUP}"
    --fd_gmm_ramp_steps "${GMM_RAMP}"
    --cond_probe
    --disable_wandb
)

if [[ "${AUTO_RESUME}" == "1" ]]; then
    CMD+=(--auto_resume)
fi
if [[ "${ONLINE_EVAL}" == "1" ]]; then
    CMD+=(--online_eval)
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # EXTRA_ARGS is intended for simple additional flags/values.
    read -r -a EXTRA <<< "${EXTRA_ARGS}"
    CMD+=("${EXTRA[@]}")
fi

echo "Experiment:       ${PROJECT}/${EXP_NAME}  (full 1000-class Arm C)"
echo "Repository:       ${REPO_DIR}"
echo "GPUs:             ${GPUS} (${NPROC_PER_NODE} ranks, batch ${BATCH_SIZE}/GPU, global $(( BATCH_SIZE * NPROC_PER_NODE )))"
echo "Master port:      ${MASTER_PORT}"
echo "Start checkpoint: ${START_CKPT}"
echo "Classes:          all 1000 (VIS_CLASSES affect visualization only)"
echo "FD stats:         ${FD_STATS[*]}"
echo "Eval FID stats:   ${INCEPTION_STATS} (${NUM_EVAL_IMAGES} images/eval)"
echo "GMM stats:        ${GMM_STATS}"
echo "Iterations:       $(( EPOCHS * STEPS_PER_EPOCH )) (${EPOCHS} x ${STEPS_PER_EPOCH})"
echo "FD window:        beta=${FD_EMA_BETA}, queue=${QUEUE_SIZE}"
echo "GMM p+q:          weight=${WEIGHT}, cls=${LAMBDA_CLS}, ent=${LAMBDA_ENT}, T=${GMM_TEMP}, cap=${CLS_CAP}"
echo "GMM estimators:   mean_beta=${GMM_MEAN_EMA_BETA}, cov_beta=${GMM_COV_EMA_BETA}, p_shrink=${P_SHRINK}, q_shrink=${Q_SHRINK}"
echo "GMM schedule:     warmup=${GMM_WARMUP}, ramp=${GMM_RAMP}"
echo "Auto-resume:      ${AUTO_RESUME}"
echo "Online eval:      ${ONLINE_EVAL} every $(( EVAL_FREQ * STEPS_PER_EPOCH )) steps"
echo "Log:              ${LOG_PATH}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'
echo "CALIBRATION: w0.0066 measured mean/median grad_ratio_p_fd=0.3505/0.3411 at steps 600-700; rescaled default=${WEIGHT}."
echo "VERIFY: inspect current grad_ratio_p_fd around steps 600-700; target 0.10-0.40."

if [[ "${RUN_FOREGROUND}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${GPUS}" "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
else
    CUDA_VISIBLE_DEVICES="${GPUS}" setsid "${CMD[@]}" \
        >"${LOG_PATH}" 2>&1 < /dev/null &
    pid=$!
    echo "Started PID ${pid}. Follow with: tail -f ${LOG_PATH}"
fi
