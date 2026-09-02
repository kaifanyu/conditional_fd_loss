#!/usr/bin/env bash
# Full 1000-class cfg_delta run for the unconditional JiT-B base.
#
# WHY THIS RUN EXISTS
#
# The 100-class cfg_delta PURE arm (jitB_gmm100_cfgdelta_PURE_w00643) worked:
# probe_top1 0.510 and gmm_class_mean_spread 0.456 by 45,600 samples/class,
# from a base that ignores its labels entirely. This scales that arm to the
# full label set. It is NOT the v2 density arm -- see
# run_jit_uncond_gmm_1000class_v2.sh for that -- and shares none of its
# calibration.
#
# THE CLOCK. Takeoff happens on a samples-per-class budget, not a step budget:
#
#   samples/class = step * GLOBAL_BATCH / 1000
#
# The 100-class cfg_delta arm first cleared spread/noise 2.0 at ~19,200
# samples/class, went visibly non-linear at 26,400, and reached probe_top1
# 0.44 at 45,600. At global batch 96 the same per-class budget costs 10x the
# steps here. Do not compare this run to the 100-class one on a step axis.
#
# THREE THINGS THAT DO NOT TRANSFER FROM 100 CLASSES, all re-derived below:
#
#   GMM_TEMP  measured on the 1000-class fit with scripts/validate_class_gmm.py.
#             The transfer criterion is posterior sharpness at the true class:
#             the 100-class operating point T=45 read median log p -0.236 with
#             70.0% above the cap. On the 1000-class fit T=10 reads -0.178 /
#             70.5% (T=11 reads -0.234 / 68.7%); T=45 there reads -3.721 /
#             1.1%, i.e. it would be a dead term.
#
#   WEIGHT    the cfg_delta field is 3x stronger at 1000 classes -- the
#             de-conditioned teacher RMS is 0.1456 at C=1000/T=10 against
#             0.0493 at C=100/T=45 -- so the 100-class weight 0.00643 seeds to
#             0.00218 here. That is a seed, not the answer: calibrate it.
#
#   GMM_STATS the 1000-class fit is a genuinely different whitening, not the
#             100-class file unmasked. Validated 2026-08-25 on 25,000 held-out
#             val images: top-1 76.65%, top-5 92.46%, spread 0.917.
#
# FD REFERENCE. Full-ImageNet stats, and --force_class_of_interest is
# deliberately NOT passed, so eval draws labels from range(1000) and scores
# against the matching reference. (At 100 classes those two disagreed; see the
# _prepare_eval_classes note in utils/eval_util.py.)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

: "${GPUS:=1,2,5,7}"
: "${MASTER_PORT:=29571}"
: "${PY_BIN:=/home/nvidia/miniconda3/envs/fdloss/bin/python}"
: "${DATA_PATH:=/data/dataset/imagenet}"
: "${START_CKPT:=checkpoints/base/JiT-B-uncond.pth}"
: "${SIGLIP_STATS:=data/fid_stats/vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz}"
: "${MAE_STATS:=data/fid_stats/vit_large_patch16_224_mae_in256_t224_stats.npz}"
: "${INCEPTION_STATS:=data/fid_stats/guided_diffusion_stats.npz}"
: "${GMM_STATS:=data/fid_stats/inception_in256_t256_classgmm_k128.npz}"
: "${OUTPUT_DIR:=work_dirs}"
: "${PROJECT:=JiT_uncond_gmm_1000class_cfgdelta}"
: "${LOG_DIR:=sweep_logs}"

# -- calibration mode --
# CALIBRATION=1 runs 1,500 steps with the gate, eval and visualisation off, so
# you can read grad_ratio_p_fd over steps 600-1500 and rescale WEIGHT.
# Everything else is identical to the real run.
: "${CALIBRATION:=0}"

# See the header. Both are refused if unset -- neither transfers.
: "${GMM_TEMP:=}"
: "${WEIGHT:=}"

# -- objective --
# PURE chain-rule: the cfg_delta field is the only conditional driver.
# LAMBDA_CLS=0 is what makes this the PURE arm rather than the PRACTICAL one.
: "${GMM_MODE:=cfg_delta}"
: "${LAMBDA_CLS:=0.0}"
: "${LAMBDA_ENT:=1.0}"
: "${CFG_NORMALIZATION:=none}"
# Inert while LAMBDA_CLS=0; kept at the 100-class arm's value so a PRACTICAL
# variant launched from this script starts from the same place.
: "${CLS_NORMALIZATION:=self}"
: "${CLS_CAP:=-0.69}"
: "${GMM_WARMUP:=0}"
: "${GMM_RAMP:=500}"
: "${P_SHRINK:=0.75}"
: "${Q_SHRINK:=0.25}"
# 1000 appearances/class at beta=0.999 spans 1000*1000/96 ~ 10,400 steps here,
# which is what made q stale in the first 1000-class attempt. It is kept anyway:
# on the samples-per-class clock 10,400 steps is 1,000 samples/class, the same
# staleness the 100-class arm carried, and lowering beta raises the spread
# noise floor and makes the gate unreadable.
: "${GMM_MEAN_EMA_BETA:=0.999}"
: "${GMM_COV_EMA_BETA:=0.999}"
: "${GMM_BOOTSTRAP:=50000}"
: "${FD_EMA_BETA:=0.99}"

: "${EPOCHS:=400}"
: "${STEPS_PER_EPOCH:=1250}"
: "${BATCH_SIZE:=24}"          # per GPU; 24 x 4 = global 96, do not change
: "${QUEUE_SIZE:=50000}"
: "${NUM_EVAL_IMAGES:=50000}"
: "${EVAL_FREQ:=20}"           # 20 * 1250 = every 25,000 steps
: "${SAVE_FREQ:=8}"
: "${PRINT_FREQ:=20}"
: "${VIS_FREQ:=8}"
: "${ONLINE_EVAL:=1}"
: "${DISABLE_VIS:=0}"
: "${AUTO_RESUME:=0}"
: "${RUN_FOREGROUND:=0}"

# -- takeoff gate, on the samples-per-class clock --
# The 100-class cfg_delta arm read spread/noise 4.30 at 24,000 samples/class,
# so a 2.0x gate there carries a 2.15x margin. The gate step is derived from
# the per-class budget, never typed in.
: "${TAKEOFF_SAMPLES_PER_CLASS:=24000}"
: "${TAKEOFF_SPREAD_MULT:=2.0}"
: "${TAKEOFF_COS_WINDOW:=50}"
: "${TAKEOFF_LOGIC:=any}"
# The 100-class arm ran with the cosine criterion disabled; keep that.
: "${TAKEOFF_NO_COS:=1}"

NUM_TRAIN_CLASSES=1000
# Visualisation only. Training and evaluation still sample all 1000 labels.
VIS_CLASSES=(0 9 88 130 207 279 281 340 360 387 404 417 444 555 569 817 920 949 974 979)
FD_MODELS=(vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception)
FD_POOLS=(cls cls cls)
FD_SIZES=(224 224 256)
FD_STATS=("${SIGLIP_STATS}" "${MAE_STATS}" "${INCEPTION_STATS}")

die() { echo "ERROR: $*" >&2; exit 2; }
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
is_bool "${TAKEOFF_NO_COS}" || die "TAKEOFF_NO_COS must be 0 or 1"
[[ "${GMM_MODE}" == "density" || "${GMM_MODE}" == "posterior" || "${GMM_MODE}" == "cfg_delta" ]] \
    || die "GMM_MODE must be density, posterior or cfg_delta (got '${GMM_MODE}')"
[[ "${CFG_NORMALIZATION}" == "none" || "${CFG_NORMALIZATION}" == "rms" ]] \
    || die "CFG_NORMALIZATION must be none or rms (got '${CFG_NORMALIZATION}')"
[[ "${AUTO_RESUME}" == "0" ]] || die "this is a fresh-run launcher; AUTO_RESUME must remain 0"

[[ -n "${GMM_TEMP}" ]] || die "GMM_TEMP is unset. It does NOT transfer across class
counts: the 100-class cfg_delta arm ran at T=45, which on the 1000-class fit
puts the median real image at log p -3.72 with 1.1% above the cap -- a dead
term. Measure it:
  CUDA_VISIBLE_DEVICES=5 ${PY_BIN} scripts/validate_class_gmm.py \\
      --stats ${GMM_STATS} --shrinkage 0.75 --temperature 8 9 10 11 12 13 14 16
and pick the T matching the reference arm's median log p / >cap fractions.
Measured 2026-08-25: T=10 -> median -0.178, 2.1% saturated, 70.5% above cap."
[[ -n "${WEIGHT}" ]] || die "WEIGHT is unset. Calibrate it on grad_ratio_p_fd:
  CALIBRATION=1 WEIGHT=0.00218 GMM_TEMP=${GMM_TEMP} bash ${BASH_SOURCE[0]}
then relaunch with WEIGHT = <seed> * 0.352 / <median ratio over steps 600-1500>.
The target is 0.352 in the CALIBRATION WINDOW, not the documented sustained
0.25-0.28 operating band: |grad_x_fd| roughly doubles over a run while the GMM
term's does not, so the window reads ~1.27x the sustained value. The reference
arm read 0.3516 in its window and settled at 0.2769.
The seed 0.00218 is the 100-class weight 0.00643 rescaled by the ratio of
de-conditioned teacher-field RMS (0.0493 at C=100/T=45 vs 0.1456 at
C=1000/T=10); it does not account for the FD reference also changing."

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
NPROC_PER_NODE="${#GPU_IDS[@]}"
GLOBAL_BATCH=$(( BATCH_SIZE * NPROC_PER_NODE ))

if [[ "${CALIBRATION}" == "1" ]]; then
    EPOCHS=2
    STEPS_PER_EPOCH=750          # 1,500 steps: ~1,000 past the ramp
    ONLINE_EVAL=0
    DISABLE_VIS=1
    TAKEOFF_GATE_STEP=-1
    : "${EXP_NAME:=jitB_gmm1000_cfgdelta_PURE_CAL_w${WEIGHT}}"
else
    : "${EXP_NAME:=jitB_gmm1000_cfgdelta_PURE_w${WEIGHT}}"
fi

TOTAL_STEPS=$(( EPOCHS * STEPS_PER_EPOCH ))
SAMPLES_PER_CLASS_AT_END=$(( TOTAL_STEPS * GLOBAL_BATCH / NUM_TRAIN_CLASSES ))
if [[ "${CALIBRATION}" != "1" ]]; then
    TAKEOFF_GATE_STEP=$(( TAKEOFF_SAMPLES_PER_CLASS * NUM_TRAIN_CLASSES / GLOBAL_BATCH ))
    (( TAKEOFF_GATE_STEP > GMM_RAMP )) \
        || die "derived gate step ${TAKEOFF_GATE_STEP} is inside the ${GMM_RAMP}-step ramp"
    (( TAKEOFF_GATE_STEP < TOTAL_STEPS )) || die "\
the takeoff gate needs ${TAKEOFF_SAMPLES_PER_CLASS} samples/class = step
${TAKEOFF_GATE_STEP}, but this run is only ${TOTAL_STEPS} steps
(${SAMPLES_PER_CLASS_AT_END} samples/class at global batch ${GLOBAL_BATCH}).
The run cannot reach a decidable point. Either raise EPOCHS/STEPS_PER_EPOCH or
lower TAKEOFF_SAMPLES_PER_CLASS deliberately."
fi

(( GLOBAL_BATCH == 96 )) || echo "WARNING: global batch is ${GLOBAL_BATCH}, not the 96 \
every reference arm used. The samples-per-class clock still holds, but the \
weight calibration and the gate margins in this header do not."

[[ -x "${PY_BIN}" ]] || die "Python executable not found: ${PY_BIN}"
[[ -f conditional_main_fd_gmm.py ]] || die "missing training entry point: conditional_main_fd_gmm.py"
[[ -d "${DATA_PATH}/train" ]] || die "ImageNet train split not found: ${DATA_PATH}/train"
[[ -f "${START_CKPT}" ]] || die "unconditional JiT-B checkpoint not found: ${START_CKPT}"
[[ -f "${GMM_STATS}" ]] || die "1000-class GMM stats not found: ${GMM_STATS}"
for stats_path in "${FD_STATS[@]}"; do
    [[ -f "${stats_path}" ]] || die "full-ImageNet FD reference not found: ${stats_path}"
done

# The GMM reference must cover all 1000 drawable labels; a subset fit here
# would silently score 1000-way draws against a 100-way posterior.
"${PY_BIN}" - "${GMM_STATS}" <<'PYEOF' || die "GMM stats file is not a full 1000-class fit"
import sys, numpy as np
d = np.load(sys.argv[1])
if d["class_mu"].shape[0] != 1000:
    raise SystemExit(f"class_mu has {d['class_mu'].shape[0]} classes, expected 1000")
if "class_ids" in d.files:
    raise SystemExit("stats carry a class_ids subset mask; expected the full fit")
PYEOF

for required_flag in \
    --fd_gmm_cls_normalization \
    --fd_gmm_cfg_normalization \
    --fd_gmm_takeoff_gate_step \
    --fd_gmm_takeoff_spread_mult \
    --fd_gmm_takeoff_cos_window \
    --fd_gmm_takeoff_logic \
    --fd_gmm_takeoff_no_cos; do
    grep -q -- "${required_flag}" conditional_main_fd_gmm.py \
        || die "conditional_main_fd_gmm.py does not support ${required_flag}"
done
grep -q -- 'posterior_score_delta' frechet_distance/gmm.py \
    || die "frechet_distance/gmm.py does not implement posterior_score_delta"
grep -q -- 'gmm_class_mean_spread_noise_floor' frechet_distance/gmm.py \
    || die "frechet_distance/gmm.py does not expose the finite-EMA spread noise floor"

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
    --class_of_interest "${VIS_CLASSES[@]}"
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
    --fd_gmm_takeoff_cos_window "${TAKEOFF_COS_WINDOW}"
    --fd_gmm_takeoff_logic "${TAKEOFF_LOGIC}"
    --cond_probe
    --disable_wandb
)

if [[ "${TAKEOFF_NO_COS}" == "1" ]]; then CMD+=(--fd_gmm_takeoff_no_cos); fi
if [[ "${ONLINE_EVAL}" == "1" ]]; then CMD+=(--online_eval); fi
if [[ "${DISABLE_VIS}" == "1" ]]; then CMD+=(--disable_vis); fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA <<< "${EXTRA_ARGS}"
    CMD+=("${EXTRA[@]}")
fi

MODE_LABEL=$([[ "${CALIBRATION}" == "1" ]] && echo "WEIGHT CALIBRATION" || echo "full 1000-class run")
echo "Experiment:       ${PROJECT}/${EXP_NAME} (${MODE_LABEL})"
echo "GPUs:             ${GPUS} (${NPROC_PER_NODE} ranks)"
echo "Batch:            ${BATCH_SIZE}/GPU, global ${GLOBAL_BATCH}"
echo "Classes:          all 1000 (VIS_CLASSES are visualisation-only)"
echo "Iterations:       ${TOTAL_STEPS} = ${SAMPLES_PER_CLASS_AT_END} samples/class"
echo "GMM:              mode=${GMM_MODE}, weight=${WEIGHT}, cls=${LAMBDA_CLS}, ent=${LAMBDA_ENT}"
if [[ "${GMM_MODE}" == "cfg_delta" ]]; then
    variant=$([[ "${LAMBDA_CLS}" == "0" || "${LAMBDA_CLS}" == "0.0" ]] \
        && echo "PURE chain-rule (no explicit class driver)" \
        || echo "PRACTICAL (explicit -log p(c|z) retained -- NOT the pure objective)")
    echo "CFG-delta:        ${variant}, vector_norm=${CFG_NORMALIZATION}"
fi
echo "GMM calibration:  T=${GMM_TEMP}, cap=${CLS_CAP}, p/q shrink=${P_SHRINK}/${Q_SHRINK}"
echo "GMM estimators:   mean_beta=${GMM_MEAN_EMA_BETA}, cov_beta=${GMM_COV_EMA_BETA}, bootstrap=${GMM_BOOTSTRAP}"
echo "GMM reference:    ${GMM_STATS}"
echo "FD reference:     full ImageNet (siglip/mae/inception), eval draws range(1000)"
if [[ "${CALIBRATION}" == "1" ]]; then
    echo "Takeoff gate:     disabled (calibration run)"
    echo "READ:             median grad_ratio_p_fd over steps 600-1500; target 0.352"
else
    echo "Takeoff gate:     ${TAKEOFF_SAMPLES_PER_CLASS} samples/class -> step ${TAKEOFF_GATE_STEP}"
    echo "                  spread/noise>=${TAKEOFF_SPREAD_MULT}, cos disabled=${TAKEOFF_NO_COS}, abort=${TAKEOFF_LOGIC} failure"
    echo "                  (reference: 100-class cfg_delta read 4.30 at that budget)"
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
