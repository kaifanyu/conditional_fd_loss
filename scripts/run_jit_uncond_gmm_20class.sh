#!/usr/bin/env bash
# Teach classification to the de-conditioned JiT-B with class-conditional GMMs,
# on the 20-class diagnostic.
#
#   L = L_FD(siglip, mae, inception)
#     + w(s) * [ lambda_cls * E[-log p(c|x)] + lambda_ent * E[log q(x|c) - log p(x|c)] ]
#
# Both posteriors live in the *inception judge's* whitened PCA-128 space, which
# the FD term already computes features in, so the conditional signal costs one
# projection and two matmuls rather than three extra frozen-backbone passes.
#
# Three arms, selected with ARM:
#   A  --fd_gmm_weight 0        matched control: identical code path, all meters
#                               logged, exactly zero gradient contribution
#   B  --fd_gmm_no_q            -log p(c|x) only: the direct, cheap replacement
#                               for the classifier ensemble
#   C  (default)                the proposal: log p and log q together
#
# Reference points from the 1000-class runs off this same checkpoint:
#   FD-only                          FID 10.68 @100k, 4.8 s/iter
#   + 3-classifier ensemble -log p   FID 21.49 @100k, probe_top1 0.635, 14.2 s/iter
# The FID doubling is the collapse the log q term exists to counteract.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

: "${ARM:=C}"
: "${CUDA_GPUS:=3,4}"
: "${MASTER_PORT:=29541}"
: "${PY_BIN:=/home/nvidia/miniconda3/envs/fdloss/bin/python}"
: "${DATA_PATH:=/data/dataset/imagenet}"
: "${START_CKPT:=checkpoints/base/JiT-B-uncond.pth}"
: "${STATS_DIR:=data/fid_stats/imagenet20_v1}"
: "${GMM_STATS:=data/fid_stats/inception_in256_t256_classgmm_k128_c20.npz}"
: "${OUTPUT_DIR:=work_dirs}"
: "${PROJECT:=JiT_uncond_gmm_20class}"
: "${LOG_DIR:=sweep_logs}"

# -- GMM term --
# lambda_cls is the DRIVER here, not a regulariser: it is the only term that can
# carve class structure out of a de-conditioned model. lambda_ent is its
# counterweight. FD_EMA_BETA=0.99 makes the FD gradient ~10x larger than at
# 0.999, so WEIGHT is calibrated against that scale.
#
# CALIBRATED, do not carry a value over from another config. Procedure: launch,
# read grad_ratio_p_fd at the first fully-ramped print (step ~600, after
# GMM_RAMP), and if it is outside 0.10-0.40 relaunch with
#     WEIGHT = <current> * 0.20 / <observed ratio>
# Two reference points bracket the band: the classifier-ensemble run that
# reached 63.5% held-out top-1 sat at 0.18, and r9's 0.34 on this same 20-class
# diagnostic is recorded as the regime that collapsed.
#
# Measured here (sweep_logs/jitB_uncond_gmm20_armC_CALIBRATION_w0.02.out):
#   WEIGHT=0.02 -> grad_ratio_p_fd 0.61 (stable over steps 500-640). Too hot.
#   0.02 * 0.20 / 0.61 = 0.0066 -> targets ~0.20.
: "${WEIGHT:=0.0066}"
: "${LAMBDA_CLS:=1.0}"
: "${LAMBDA_ENT:=1.0}"
: "${CLS_CAP:=-0.69}"        # stop pushing a sample past p(target)=50%
# The 20-way posterior in a whitened 128-d space is fully saturated on real
# data at T=1: 98.6% of held-out real images sit at exactly log p = 0, i.e.
# -log p(c|x) has no gradient on anything that already looks real. Measured
# sweep (top-1 is 0.986 at every T, the ranking is unchanged):
#     T=1   median log p  0.000   98.6% saturated
#     T=50  median       -0.010   49.4% saturated
#     T=100 median       -0.312    0.0% saturated, 80% above the cap
#     T=200 median       -1.258    0.0% saturated,  2% above the cap
# T=100 puts real images just above CLS_CAP, so the term stops pushing exactly
# when a sample is as class-confident as a real image. T=200 would keep pushing
# past real-data confidence, which is the adversarial regime.
: "${GMM_TEMP:=100}"
: "${GMM_WARMUP:=0}"
: "${GMM_RAMP:=500}"
: "${P_SHRINK:=0.75}"
: "${Q_SHRINK:=0.25}"
: "${GMM_EMA_BETA:=0.999}"

# 0.999 (~1000-step window) could not see mode collapse on the 20-class
# diagnostic: a collapsed generator whose single mode drifts across 1000 steps
# still accumulates a broad covariance. 0.99 is a ~100-step window.
: "${FD_EMA_BETA:=0.99}"

: "${EPOCHS:=40}"
: "${STEPS_PER_EPOCH:=1250}"   # 40 x 1250 = 50,000 iterations
: "${BATCH_SIZE:=48}"          # per GPU
: "${QUEUE_SIZE:=50000}"
: "${NUM_EVAL_IMAGES:=20000}"
: "${EVAL_FREQ:=5}"
: "${SAVE_FREQ:=5}"
: "${PRINT_FREQ:=20}"
: "${VIS_FREQ:=2}"
: "${AUTO_RESUME:=1}"
: "${ONLINE_EVAL:=1}"
: "${RUN_FOREGROUND:=0}"

CLASS_IDS=(0 9 88 130 207 279 281 340 360 387 404 417 444 555 569 817 920 949 974 979)
FD_MODELS=(vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception)
FD_POOLS=(cls cls cls)
FD_SIZES=(224 224 256)
FD_STATS=(
    "${STATS_DIR}/siglip_cls.npz"
    "${STATS_DIR}/mae_cls.npz"
    "${STATS_DIR}/inception.npz"
)

case "${ARM}" in
    A) ARM_FLAGS=(--fd_gmm_weight 0); ARM_TAG="armA_control" ;;
    B) ARM_FLAGS=(--fd_gmm_weight "${WEIGHT}" --fd_gmm_no_q); ARM_TAG="armB_logp_only" ;;
    C) ARM_FLAGS=(--fd_gmm_weight "${WEIGHT}"); ARM_TAG="armC_logp_logq" ;;
    *) echo "ERROR: ARM must be A, B or C (got '${ARM}')" >&2; exit 2 ;;
esac
: "${EXP_NAME:=jitB_uncond_gmm20_${ARM_TAG}}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_GPUS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"

if [[ ! -x "${PY_BIN}" ]]; then
    echo "ERROR: Python executable not found: ${PY_BIN}" >&2
    exit 2
fi
if [[ ! -f "${START_CKPT}" ]]; then
    echo "ERROR: de-conditioned JiT checkpoint not found: ${START_CKPT}" >&2
    echo "Build it with: python make_uncond_jit.py --in_ckpt checkpoints/base/JiT-B.pth \\" >&2
    echo "                   --out_ckpt ${START_CKPT} --num_classes 1000" >&2
    exit 2
fi
if [[ ! -f "${GMM_STATS}" ]]; then
    echo "ERROR: 20-class GMM stats not found: ${GMM_STATS}" >&2
    echo "Run first: bash scripts/compute_class_gmm_20class.sh" >&2
    exit 2
fi
for stats_path in "${FD_STATS[@]}"; do
    if [[ ! -f "${stats_path}" ]]; then
        echo "ERROR: missing 20-class reference stats: ${stats_path}" >&2
        echo "Run first: bash scripts/compute_imagenet20_fd_stats.sh" >&2
        exit 2
    fi
done

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${EXP_NAME}.out"

CMD=(
    "${PY_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_port="${MASTER_PORT}"
    conditional_main_fd_gmm.py
    --data_path "${DATA_PATH}"
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
    --fid_stats_path "${FD_STATS[2]}"
    --train_class_ids "${CLASS_IDS[@]}"
    --class_of_interest "${CLASS_IDS[@]}"
    --force_class_of_interest
    --fd_gmm
    --fd_gmm_stats_path "${GMM_STATS}"
    --fd_gmm_judge inception
    --fd_gmm_pca_dim 128
    --fd_gmm_mode density
    --fd_gmm_lambda_cls "${LAMBDA_CLS}"
    --fd_gmm_lambda_ent "${LAMBDA_ENT}"
    --fd_gmm_cls_cap "${CLS_CAP}"
    --fd_gmm_temp "${GMM_TEMP}"
    --fd_gmm_p_shrinkage "${P_SHRINK}"
    --fd_gmm_q_shrinkage "${Q_SHRINK}"
    --fd_gmm_ema_beta "${GMM_EMA_BETA}"
    --fd_gmm_warmup_steps "${GMM_WARMUP}"
    --fd_gmm_ramp_steps "${GMM_RAMP}"
    "${ARM_FLAGS[@]}"
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
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    CMD+=("${EXTRA[@]}")
fi

echo "Experiment:      ${PROJECT}/${EXP_NAME}  (arm ${ARM})"
echo "GPUs:            ${CUDA_GPUS} (${NPROC_PER_NODE} ranks, bsz ${BATCH_SIZE}/GPU)"
echo "Start ckpt:      ${START_CKPT}"
echo "GMM stats:       ${GMM_STATS}"
echo "Classes (${#CLASS_IDS[@]}):   ${CLASS_IDS[*]}"
echo "Iterations:      $(( EPOCHS * STEPS_PER_EPOCH ))"
echo "GMM term:        weight=${WEIGHT} cls=${LAMBDA_CLS} ent=${LAMBDA_ENT} cap=${CLS_CAP} T=${GMM_TEMP} ramp=${GMM_RAMP}"
echo "Log:             ${LOG_PATH}"

if [[ "${RUN_FOREGROUND}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_GPUS}" "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
else
    CUDA_VISIBLE_DEVICES="${CUDA_GPUS}" setsid "${CMD[@]}" \
        >"${LOG_PATH}" 2>&1 < /dev/null &
    echo "Started PID $!. Follow with: tail -f ${LOG_PATH}"
fi
