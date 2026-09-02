#!/usr/bin/env bash
# 100-class scaling probe for the GMM log-p/log-q takeoff.
#
# WHY THIS RUN EXISTS
#
# The 20-class pilot reached probe_top1 0.93. The 1000-class run produced
# nothing in 50,000 steps: spread/noise 1.022 (i.e. exactly its own estimator
# noise floor) and probe_rank 497.8 against a chance level of 500. The two are
# not in conflict once both are put on a samples-per-class clock rather than a
# step clock:
#
#   20-class arm C first passes BOTH takeoff criteria at step 5,740
#   = 27,552 samples/class.  The 1000-class run's gate fired at step 10,000
#   = 960 samples/class, where the 20-class run itself read spread/noise 0.71
#   and cos +0.117 -- i.e. it would have failed the same gate even harder.
#
# At global batch 96 the 1000-class run would need ~292,000 steps to reach the
# per-class budget where the pilot took off. Before committing ~2 GPU-days to
# that, this run tests the clock itself at an intermediate class count.
#
#   samples/class = step * GLOBAL_BATCH / num_classes = step * 0.96 here
#
# THE PREDICTION, if the samples-per-class clock is linear in class count:
#
#   step 30,000 (28,800 samples/class)  ~ 20-class step 6,000:  gate PASSES,
#                                         spread ~0.03, cos crosses below 0
#   step 50,000 (48,000 samples/class)  ~ 20-class step 10,000: spread ~0.37,
#                                         probe_top1 ~0.32, cond_delta ~0.41
#
# Outcomes and what each one means:
#   * takes off near step 30,000        -> clock confirmed, launch 1000 classes
#                                          at ~292,000 steps
#   * takes off much earlier            -> sublinear; 1000 classes is cheaper
#                                          than 292k steps. Refit the target
#                                          from this run before launching
#   * flat at 48,000 samples/class      -> the clock is NOT the explanation and
#                                          something is genuinely different
#                                          beyond 20 classes. Do not launch the
#                                          1000-class run; debug here instead,
#                                          where an experiment costs 7 h
#
# Global batch MUST stay 96 for any of these numbers to mean anything: the
# samples-per-class clock is defined against it, and both reference runs used it.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# shellcheck source=scripts/imagenet100_class_ids.sh
source "${SCRIPT_DIR}/imagenet100_class_ids.sh"

: "${GPUS:=4,5,6,7}"
: "${MASTER_PORT:=29566}"
: "${PY_BIN:=/home/nvidia/miniconda3/envs/fdloss/bin/python}"
: "${DATA_PATH:=/data/dataset/imagenet}"
: "${START_CKPT:=checkpoints/base/JiT-B-uncond.pth}"
: "${STATS_DIR:=data/fid_stats/imagenet100_v1}"
: "${GMM_STATS:=data/fid_stats/inception_in256_t256_classgmm_k128_c100.npz}"
: "${OUTPUT_DIR:=work_dirs}"
: "${PROJECT:=JiT_uncond_gmm_100class}"
: "${LOG_DIR:=sweep_logs}"

# -- calibration mode --
# CALIBRATION=1 runs ~1,500 steps with the gate, eval and visualisation off, so
# you can read grad_ratio_p_fd at the first fully-ramped print and rescale
# WEIGHT. Everything else is identical to the real run.
: "${CALIBRATION:=0}"

# -- the two values that must NOT be carried over from another class count --
#
# GMM_TEMP: the 20-class operating point was T=100; re-measured on the
# 1000-class fit the equivalent point was T=10, and T=100 there was catastrophic
# (median real image at -5.41 against a uniform of -6.91, i.e. the term would
# push every already-correct sample into the adversarial regime). Neither value
# transfers to 100 classes. Measure it:
#
#   CUDA_VISIBLE_DEVICES=4 python scripts/validate_class_gmm.py \
#       --stats data/fid_stats/inception_in256_t256_classgmm_k128_c100.npz \
#       --class_ids $(seq 0 10 990 | tr '\n' ' ') \
#       --shrinkage 0.75 --temperature 1 5 10 20 50 100 200
#
# Pick the T whose median log p lands just above CLS_CAP with ~0% saturated.
#
# WEIGHT: calibrated on grad_ratio_p_fd, never on the loss value. Target
# 0.10-0.40, read at the first fully-ramped print (step ~600, after GMM_RAMP).
# Two reference points bracket the band: the classifier-ensemble run that
# reached 63.5% top-1 sat at 0.18, and r9's 0.34 is recorded as the regime that
# collapsed. Run with CALIBRATION=1 and a seed weight, then relaunch with
#     WEIGHT = <seed> * 0.30 / <observed ratio>
# Note the calibration target is class-count-blind -- it matches the *aggregate*
# GMM/FD gradient, which at 100 classes is spread over 5x more class embeddings
# than at 20. Matching it is necessary, not sufficient.
: "${GMM_TEMP:=}"
: "${WEIGHT:=}"

: "${LAMBDA_CLS:=1.0}"
: "${LAMBDA_ENT:=1.0}"
# GMM_MODE: 'density' is the calibrated default and the anti-collapse baseline;
# do not change it without recalibrating WEIGHT from scratch. 'cfg_delta'
# injects the CFG-delta vector field (see CFG_DELTA_GMM_IMPLEMENTATION_SPEC.md)
# and is an ablation -- its field, and therefore its image-space gradient, has
# nothing in common with density mode's, so a density WEIGHT does NOT transfer.
: "${GMM_MODE:=density}"
# Explicit normalisation of the cfg_delta feature vector. 'none' keeps the raw
# fitted-GMM field (the derived quantity); 'rms' rescales it and is a separate
# ablation. Ignored unless GMM_MODE=cfg_delta.
: "${CFG_NORMALIZATION:=none}"
# log_classes divides l_cls by the fixed ln(C) of the *GMM reference* (100 here,
# not the model's 1000-way label table), so a hard example cannot turn its own
# gradient down. This is the v2 choice and the one being scaled to 1000; the
# 20-class pilot used the legacy 'self' normaliser.
: "${CLS_NORMALIZATION:=log_classes}"
: "${CLS_CAP:=-0.69}"
: "${GMM_WARMUP:=0}"
: "${GMM_RAMP:=500}"
: "${P_SHRINK:=0.75}"
: "${Q_SHRINK:=0.25}"
# 1000 appearances/class at beta=0.999 spans 1000*100/96 ~ 1,040 steps at 100
# classes -- the same order as the 20-class pilot's 208 and nowhere near the
# 10,400 steps that made q stale at 1000 classes. Keep 0.999: it also puts the
# spread noise floor 10x lower than 0.99 would, which is what makes the gate
# metric readable.
: "${GMM_MEAN_EMA_BETA:=0.999}"
: "${GMM_COV_EMA_BETA:=0.999}"
: "${GMM_BOOTSTRAP:=50000}"
: "${FD_EMA_BETA:=0.99}"

: "${EPOCHS:=40}"
: "${STEPS_PER_EPOCH:=1250}"   # 40 x 1250 = 50,000 steps = 48,000 samples/class
: "${BATCH_SIZE:=24}"          # per GPU; 24 x 4 = global 96, do not change
: "${QUEUE_SIZE:=50000}"
: "${NUM_EVAL_IMAGES:=50000}"
: "${EVAL_FREQ:=4}"            # 4 * 1250 = every 5,000 steps
: "${SAVE_FREQ:=4}"
: "${PRINT_FREQ:=20}"
: "${VIS_FREQ:=2}"
: "${ONLINE_EVAL:=1}"
: "${DISABLE_VIS:=0}"
: "${AUTO_RESUME:=0}"
: "${RUN_FOREGROUND:=0}"

# -- takeoff gate, on the samples-per-class clock --
# The criteria are unchanged from the 1000-class run and are already
# class-count-normalised (spread is q-between over p-between; the noise floor
# divides by n_eff). Only the clock was wrong there. The reference run passed at
# 27,552 samples/class, so 35,000 carries a 1.27x margin; the gate step is
# derived, never typed in.
: "${TAKEOFF_SAMPLES_PER_CLASS:=35000}"
: "${TAKEOFF_SPREAD_MULT:=2.0}"
: "${TAKEOFF_COS_THRESHOLD:=0.0}"
: "${TAKEOFF_COS_WINDOW:=50}"
: "${TAKEOFF_LOGIC:=any}"

CLASS_IDS=("${IMAGENET100_CLASS_IDS[@]}")
NUM_TRAIN_CLASSES="${#CLASS_IDS[@]}"
# Visualisation only; a stride-5 slice of the subset so the panel stays 20 wide.
mapfile -t VIS_CLASSES < <(printf '%s\n' "${CLASS_IDS[@]}" | awk 'NR % 5 == 1')

FD_MODELS=(vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception)
FD_POOLS=(cls cls cls)
FD_SIZES=(224 224 256)
SIGLIP_STATS="${STATS_DIR}/siglip_cls.npz"
MAE_STATS="${STATS_DIR}/mae_cls.npz"
INCEPTION_STATS="${STATS_DIR}/inception.npz"
FD_STATS=("${SIGLIP_STATS}" "${MAE_STATS}" "${INCEPTION_STATS}")

die() {
    echo "ERROR: $*" >&2
    exit 2
}

is_nonnegative_int() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_bool() { [[ "$1" == "0" || "$1" == "1" ]]; }

[[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] \
    || die "GPUS must be a comma-separated list of numeric GPU IDs (got '${GPUS}')"
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] && (( MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) \
    || die "MASTER_PORT must be between 1 and 65535 (got '${MASTER_PORT}')"
is_nonnegative_int "${EPOCHS}" && (( EPOCHS > 0 )) \
    || die "EPOCHS must be a positive integer (got '${EPOCHS}')"
is_nonnegative_int "${STEPS_PER_EPOCH}" && (( STEPS_PER_EPOCH > 0 )) \
    || die "STEPS_PER_EPOCH must be a positive integer (got '${STEPS_PER_EPOCH}')"
is_nonnegative_int "${BATCH_SIZE}" && (( BATCH_SIZE > 0 )) \
    || die "BATCH_SIZE must be a positive integer (got '${BATCH_SIZE}')"
is_nonnegative_int "${TAKEOFF_SAMPLES_PER_CLASS}" \
    || die "TAKEOFF_SAMPLES_PER_CLASS must be a non-negative integer"
is_nonnegative_int "${TAKEOFF_COS_WINDOW}" && (( TAKEOFF_COS_WINDOW > 0 )) \
    || die "TAKEOFF_COS_WINDOW must be a positive integer"
[[ "${TAKEOFF_LOGIC}" == "any" || "${TAKEOFF_LOGIC}" == "all" ]] \
    || die "TAKEOFF_LOGIC must be any or all (got '${TAKEOFF_LOGIC}')"
is_bool "${ONLINE_EVAL}" || die "ONLINE_EVAL must be 0 or 1"
is_bool "${DISABLE_VIS}" || die "DISABLE_VIS must be 0 or 1"
is_bool "${AUTO_RESUME}" || die "AUTO_RESUME must be 0 or 1"
is_bool "${RUN_FOREGROUND}" || die "RUN_FOREGROUND must be 0 or 1"
is_bool "${CALIBRATION}" || die "CALIBRATION must be 0 or 1"
[[ "${GMM_MODE}" == "density" || "${GMM_MODE}" == "posterior" || "${GMM_MODE}" == "cfg_delta" ]] \
    || die "GMM_MODE must be density, posterior or cfg_delta (got '${GMM_MODE}')"
[[ "${CFG_NORMALIZATION}" == "none" || "${CFG_NORMALIZATION}" == "rms" ]] \
    || die "CFG_NORMALIZATION must be none or rms (got '${CFG_NORMALIZATION}')"
[[ "${AUTO_RESUME}" == "0" ]] || die "this is a fresh-run launcher; AUTO_RESUME must remain 0"

[[ -n "${GMM_TEMP}" ]] || die "GMM_TEMP is unset. It does NOT transfer across class
counts: T=100 was the 20-class operating point and was catastrophic at 1000,
where T=10 was equivalent. Measure it on the 100-class fit first --
see the GMM_TEMP comment at the top of this script for the exact command."
[[ -n "${WEIGHT}" ]] || die "WEIGHT is unset. Calibrate it on grad_ratio_p_fd
(target 0.10-0.40) with a short run:
  CALIBRATION=1 WEIGHT=<seed> GMM_TEMP=${GMM_TEMP} bash ${BASH_SOURCE[0]}
then relaunch with WEIGHT = <seed> * 0.30 / <observed ratio>."

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
NPROC_PER_NODE="${#GPU_IDS[@]}"
GLOBAL_BATCH=$(( BATCH_SIZE * NPROC_PER_NODE ))

if [[ "${CALIBRATION}" == "1" ]]; then
    EPOCHS=2
    STEPS_PER_EPOCH=750          # 1,500 steps: ~1,000 past the ramp
    ONLINE_EVAL=0
    DISABLE_VIS=1
    TAKEOFF_GATE_STEP=-1
    : "${EXP_NAME:=jitB_uncond_gmm100_armC_CALIBRATION_w${WEIGHT}}"
else
    : "${EXP_NAME:=jitB_uncond_gmm100_armC_probe}"
fi

TOTAL_STEPS=$(( EPOCHS * STEPS_PER_EPOCH ))
SAMPLES_PER_CLASS_AT_END=$(( TOTAL_STEPS * GLOBAL_BATCH / NUM_TRAIN_CLASSES ))
# The whole point of this script: the gate step is derived from the per-class
# budget, not typed in. Guard the division so a bad override cannot silently
# produce a gate at step 0.
if [[ "${CALIBRATION}" != "1" ]]; then
    TAKEOFF_GATE_STEP=$(( TAKEOFF_SAMPLES_PER_CLASS * NUM_TRAIN_CLASSES / GLOBAL_BATCH ))
    (( TAKEOFF_GATE_STEP > GMM_RAMP )) \
        || die "derived gate step ${TAKEOFF_GATE_STEP} is inside the ${GMM_RAMP}-step ramp"
    (( TAKEOFF_GATE_STEP < TOTAL_STEPS )) || die "\
the takeoff gate needs ${TAKEOFF_SAMPLES_PER_CLASS} samples/class = step
${TAKEOFF_GATE_STEP}, but this run is only ${TOTAL_STEPS} steps
(${SAMPLES_PER_CLASS_AT_END} samples/class at global batch ${GLOBAL_BATCH}).
The run is too short to reach a decidable point -- this is exactly the
condition that made the 1000-class gate at step 10,000 meaningless. Either
raise EPOCHS/STEPS_PER_EPOCH or lower TAKEOFF_SAMPLES_PER_CLASS deliberately."
fi

(( GLOBAL_BATCH == 96 )) || echo "WARNING: global batch is ${GLOBAL_BATCH}, not the 96 \
both reference runs used. The samples-per-class clock still holds, but the \
step numbers in this script's header do not."

[[ -x "${PY_BIN}" ]] || die "Python executable not found: ${PY_BIN}"
[[ -f conditional_main_fd_gmm.py ]] || die "missing training entry point: conditional_main_fd_gmm.py"
[[ -d "${DATA_PATH}/train" ]] || die "ImageNet train split not found: ${DATA_PATH}/train"
[[ -f "${START_CKPT}" ]] || die "unconditional JiT-B checkpoint not found: ${START_CKPT}"
[[ -f "${GMM_STATS}" ]] || die "100-class GMM stats not found: ${GMM_STATS}
Run first: bash scripts/compute_class_gmm_100class.sh"
for stats_path in "${FD_STATS[@]}"; do
    [[ -f "${stats_path}" ]] || die "missing 100-class FD reference: ${stats_path}
Run first: bash scripts/compute_imagenet100_fd_stats.sh"
done

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
if [[ "${GMM_MODE}" == "cfg_delta" ]]; then
    grep -q -- '--fd_gmm_cfg_normalization' conditional_main_fd_gmm.py \
        || die "conditional_main_fd_gmm.py does not support --fd_gmm_cfg_normalization"
    grep -q -- 'posterior_score_delta' frechet_distance/gmm.py \
        || die "frechet_distance/gmm.py does not implement posterior_score_delta"
fi

RUN_DIR="${OUTPUT_DIR}/${PROJECT}/${EXP_NAME}"
[[ ! -e "${RUN_DIR}" ]] || die "work directory already exists: ${RUN_DIR}; choose a unique EXP_NAME"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${EXP_NAME}.out"
[[ ! -e "${LOG_PATH}" ]] || die "log path already exists: ${LOG_PATH}; choose a unique EXP_NAME"

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
    --train_class_ids "${CLASS_IDS[@]}"
    --class_of_interest "${VIS_CLASSES[@]}"
    --force_class_of_interest
    --fd_gmm
    --fd_gmm_stats_path "${GMM_STATS}"
    --fd_gmm_judge inception
    --fd_gmm_pca_dim 128
    --fd_gmm_mode "${GMM_MODE}"
    --fd_gmm_cfg_normalization "${CFG_NORMALIZATION}"
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

if [[ "${ONLINE_EVAL}" == "1" ]]; then CMD+=(--online_eval); fi
if [[ "${DISABLE_VIS}" == "1" ]]; then CMD+=(--disable_vis); fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA <<< "${EXTRA_ARGS}"
    CMD+=("${EXTRA[@]}")
fi

MODE_LABEL=$([[ "${CALIBRATION}" == "1" ]] && echo "WEIGHT CALIBRATION" || echo "scaling probe")
echo "Experiment:       ${PROJECT}/${EXP_NAME} (${MODE_LABEL})"
echo "GPUs:             ${GPUS} (${NPROC_PER_NODE} ranks)"
echo "Batch:            ${BATCH_SIZE}/GPU, global ${GLOBAL_BATCH}"
echo "Classes:          ${NUM_TRAIN_CLASSES} (${CLASS_IDS[0]} ${CLASS_IDS[1]} ... ${CLASS_IDS[-1]}), stride-10 subset"
echo "Iterations:       ${TOTAL_STEPS} = ${SAMPLES_PER_CLASS_AT_END} samples/class"
echo "GMM p+q:          mode=${GMM_MODE}, weight=${WEIGHT}, cls=${LAMBDA_CLS}, ent=${LAMBDA_ENT}, cls_norm=${CLS_NORMALIZATION}"
if [[ "${GMM_MODE}" == "cfg_delta" ]]; then
    variant=$([[ "${LAMBDA_CLS}" == "0" || "${LAMBDA_CLS}" == "0.0" ]] \
        && echo "PURE chain-rule (no explicit class driver)" \
        || echo "PRACTICAL (explicit -log p(c|z) retained -- NOT the pure objective)")
    echo "CFG-delta:        ${variant}, vector_norm=${CFG_NORMALIZATION}"
fi
echo "GMM calibration:  T=${GMM_TEMP}, cap=${CLS_CAP}, p/q shrink=${P_SHRINK}/${Q_SHRINK}"
echo "GMM estimators:   mean_beta=${GMM_MEAN_EMA_BETA}, cov_beta=${GMM_COV_EMA_BETA}, bootstrap=${GMM_BOOTSTRAP}"
if [[ "${CALIBRATION}" == "1" ]]; then
    echo "Takeoff gate:     disabled (calibration run)"
    echo "READ:             grad_ratio_p_fd at steps ~600-1500; target 0.10-0.40"
else
    echo "Takeoff gate:     ${TAKEOFF_SAMPLES_PER_CLASS} samples/class -> step ${TAKEOFF_GATE_STEP}"
    echo "                  spread/noise>=${TAKEOFF_SPREAD_MULT}, mean(last ${TAKEOFF_COS_WINDOW} cos)<${TAKEOFF_COS_THRESHOLD}, abort=${TAKEOFF_LOGIC} failure"
    echo "                  (reference: 20-class arm C passed at 27,552 samples/class)"
fi
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
