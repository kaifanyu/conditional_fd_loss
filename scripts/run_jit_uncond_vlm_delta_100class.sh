#!/usr/bin/env bash
# The single VLM posterior-delta trial, 100 classes, from the de-conditioned JiT-B.
#
# THE OBJECTIVE, and nothing else:
#
#   L = L_FD + lambda * E[ log q(c | x) - log p(c | x) ]
#
#   * one frozen VLM (SigLIP-SO400M) -- it is ALREADY an FD judge here, so z is
#     computed once per step and the conditional term costs one linear layer
#   * p: a linear head trained offline on REAL images, then frozen forever
#   * q: the same architecture, initialised from p, trained online by cross
#     entropy on DETACHED generated (image, sampled label) pairs
#   * the generator sees the current student; Q_USE_EMA=1 opts into an EMA teacher
#
# It is NOT the class-summed posterior KL (that is --fd_gmm_mode posterior in
# conditional_main_fd_gmm.py; it is label-blind and died at its gate on
# 2026-08-27) and NOT a second explicit -log p(c|z) driver. Do not add arms here.
#
# WHY THE CALIBRATION IS DIFFERENT FROM EVERY GMM ARM
#
# At initialisation q == p exactly, so log q - log p == 0 and its gradient is
# EXACTLY ZERO -- grad_ratio_vlm_fd starts at 0 and grows only as q drifts.
# The GMM recipe ("read the 600-1500 window, target 0.352") does NOT apply:
# there is nothing to read at step 600. CALIBRATION=1 therefore runs long enough
# for q to have separated, and the weight is set from the *late* window.
#
# Global batch MUST stay 96 -- the samples-per-class clock every number in
# docs/gmm.md is quoted against is defined at that batch.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# shellcheck source=scripts/imagenet100_class_ids.sh
source "${SCRIPT_DIR}/imagenet100_class_ids.sh"

: "${GPUS:=4,5,6,7}"
: "${MASTER_PORT:=29572}"
: "${PY_BIN:=/home/nvidia/miniconda3/envs/fdloss/bin/python}"
: "${DATA_PATH:=/data/dataset/imagenet}"
: "${START_CKPT:=checkpoints/base/JiT-B-uncond.pth}"
: "${STATS_DIR:=data/fid_stats/imagenet100_v1}"
: "${P_HEAD:=work_dirs/vlm_p_head_siglip_c100/p_head.pt}"
: "${OUTPUT_DIR:=work_dirs}"
: "${PROJECT:=JiT_uncond_vlm_delta}"
: "${LOG_DIR:=sweep_logs}"

# -- calibration mode (Step D) --
# CALIBRATION=1 runs CAL_STEPS steps with the gate, eval and visualisation off.
# Read the result with
#   python scripts/analyze_vlm_delta_run.py --calibration --weight <seed> <run_dir>
# and relaunch at the printed WEIGHT.
: "${CALIBRATION:=0}"
: "${CAL_STEPS:=3000}"

# -- the one number that has to be measured, never carried over --
# Calibrated on the SUSTAINED grad_ratio_vlm_fd, never on the loss value.
# Target band 0.22-0.30 (both GMM successes sat at ~0.25).
: "${WEIGHT:=}"

# -- shared logit temperature for BOTH heads --
# Empty = use the temperature fitted on real held-out validation data and stored
# in the p-head checkpoint. p and q are never tuned separately, and generator
# performance must never be used to pick it.
: "${HEAD_TEMPERATURE:=}"

# -- q head: deliberately slow --
: "${Q_LR:=1e-4}"
: "${Q_OPTIMIZER:=adamw}"
: "${Q_BETA1:=0.0}"
: "${Q_BETA2:=0.999}"
: "${Q_USE_EMA:=0}"
# NOT optional. An unregularised online head on a signal-free stream random-walks
# away from p, so ||W_q - W_p|| -- and with it the injected field and
# grad_ratio_vlm_fd -- grows without bound and no weight stays calibrated.
# Measured on the first CAL run: teacher drift 0.44 -> 0.97 of ||W_p|| between
# steps 900 and 2640 with the ratio tracking it all the way up. Decay pulls W
# toward 0, i.e. toward the uniform posterior that IS the correct q at a
# de-conditioned start, and makes the drift stationary.
: "${Q_WEIGHT_DECAY:=3.0}"
: "${Q_GRAD_CLIP:=1.0}"
: "${Q_UPDATES_PER_STEP:=1}"
# 0 = the global batch, which puts the per-sample reuse factor at 1.0:
#   reuse = Q_BATCH_SIZE * Q_UPDATES_PER_STEP / GLOBAL_BATCH
# At the original 512 (reuse 5.3x) the head reached 40% top-1 on its own replay
# buffer while staying at chance on fresh samples -- pure memorisation, so
# log q_teacher(c|z) was noise exactly where the generator loss reads it.
: "${Q_BATCH_SIZE:=0}"
# 0.999 over 1 update/step is a ~1000-step teacher horizon: ~960 samples/class at
# 100 classes and global batch 96, i.e. the teacher describes the generator of
# roughly the last thousand steps, not the current minibatch.
: "${Q_EMA_BETA:=0.999}"
: "${Q_BUFFER_SIZE:=20000}"        # 200/class at 100 classes
: "${Q_BOOTSTRAP:=50000}"
# Non-zero breaks the q == p initialisation the whole design rests on. Keep 0.
: "${Q_BOOTSTRAP_UPDATES:=0}"

: "${DELTA_WARMUP:=0}"
: "${DELTA_RAMP:=500}"
# OPT-IN emergency clamp on log q - log p. 0 = off, which is the default on
# purpose: do not silently change the objective. If you turn it on, report
# vlm_delta_clamp_frac with the result.
: "${DELTA_CLAMP:=0.0}"

: "${FD_EMA_BETA:=0.99}"
# Keep the historical generator AdamW LR unless explicitly overridden.
# Averaging gradients across ranks does not make AdamW's effective LR equal
# to LR/world_size: its moment normalization also depends on gradient scale.
: "${LR:=1e-5}"
: "${EPOCHS:=40}"
: "${STEPS_PER_EPOCH:=1250}"   # 40 x 1250 = 50,000 steps = 48,000 samples/class
: "${BATCH_SIZE:=24}"          # per GPU; default 24 x 4 = global 96
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
# Slurm may supply physical IDs or GPU UUIDs. Keep its visibility mapping when
# requested; GPUS then only enumerates logical ranks for the batch calculation.
: "${PRESERVE_CUDA_VISIBLE_DEVICES:=0}"
: "${DRY_RUN:=0}"
# RESUME=1 continues an existing EXP_NAME from its newest checkpoint instead of
# refusing to touch an existing work dir. Everything -- generator, EMA,
# optimizer, FD queues, q student/teacher, q optimizer and the q replay buffer --
# comes back exactly (tests/test_vlm_delta.py + scripts/smoke_vlm_delta.sh pin
# this), and the trainer re-checks the p-head identity before continuing.
: "${RESUME:=0}"
# Bounds how much work an eviction destroys. Lower on contended GPUs.
: "${CKPT_TARGET_MINUTES:=10}"
: "${INIT_DIAG_SAMPLES:=2048}"
: "${PER_CLASS_EVERY:=2500}"

# -- takeoff gate: OPT-IN, off by default --
# Set TAKEOFF_SAMPLES_PER_CLASS to a positive number to enable it; the step is
# then DERIVED from the per-class budget, never typed in (docs/gmm.md §7). The
# criteria are generator-side and cannot be gamed by q: cond_delta (does the
# class token change the image at all?) and the frozen p head's view of the
# sampled label's rank as a fraction of chance.
: "${TAKEOFF_SAMPLES_PER_CLASS:=0}"
: "${TAKEOFF_COND_DELTA_MIN:=0.10}"
: "${TAKEOFF_RANK_RATIO_MAX:=0.90}"
: "${TAKEOFF_LOGIC:=any}"

CLASS_IDS=("${IMAGENET100_CLASS_IDS[@]}")
NUM_TRAIN_CLASSES="${#CLASS_IDS[@]}"
mapfile -t VIS_CLASSES < <(printf '%s\n' "${CLASS_IDS[@]}" | awk 'NR % 5 == 1')

# -- answer-state backend (only read when the p head declares it) --
# Nothing here selects the backend: the p-head checkpoint does, because the head
# IS the definition of z. These are only the runtime knobs for the 7B path.
# Measured on A100-80GB: microbatch 12 -> 29 GB peak / ~18 img/s, 24 -> 41 GB /
# ~20 img/s, both alongside the three FD judges.
: "${Q_LORA:=0}"
if [[ "${Q_LORA}" == "1" ]]; then
    : "${VLM_MICROBATCH:=1}"
    : "${VLM_DTYPE:=fp32}"
else
    : "${VLM_MICROBATCH:=12}"
fi
# 0 = score the whole per-rank batch. A smaller K trades gradient variance for
# step time and stays unbiased (generated batch elements are exchangeable).
: "${VLM_SAMPLES_PER_STEP:=0}"
# bf16 is the only affordable option: fp32 is ~37 GB and ~3.5x slower. Read
# docs/vlm_delta.md §3.3 before trusting cos_update_fd_vlm on this backend --
# the bf16 image gradient has cosine ~0.03 with the fp32 model's.
: "${VLM_DTYPE:=bf16}"

# The VLM must be the FIRST judge below and its short name must be VLM_JUDGE.
FD_MODELS=(vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception)
FD_POOLS=(cls cls cls)
FD_SIZES=(224 224 256)
VLM_JUDGE=siglip
SIGLIP_STATS="${STATS_DIR}/siglip_cls.npz"
MAE_STATS="${STATS_DIR}/mae_cls.npz"
INCEPTION_STATS="${STATS_DIR}/inception.npz"
FD_STATS=("${SIGLIP_STATS}" "${MAE_STATS}" "${INCEPTION_STATS}")

die() { echo "ERROR: $*" >&2; exit 2; }
is_nonnegative_int() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_bool() { [[ "$1" == "0" || "$1" == "1" ]]; }

[[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] \
    || die "GPUS must be a comma-separated list of numeric GPU IDs (got '${GPUS}')"
[[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] && (( MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) \
    || die "MASTER_PORT must be between 1 and 65535 (got '${MASTER_PORT}')"
is_nonnegative_int "${EPOCHS}" && (( EPOCHS > 0 )) || die "EPOCHS must be a positive integer"
is_nonnegative_int "${STEPS_PER_EPOCH}" && (( STEPS_PER_EPOCH > 0 )) \
    || die "STEPS_PER_EPOCH must be a positive integer"
is_nonnegative_int "${BATCH_SIZE}" && (( BATCH_SIZE > 0 )) \
    || die "BATCH_SIZE must be a positive integer"
is_nonnegative_int "${TAKEOFF_SAMPLES_PER_CLASS}" \
    || die "TAKEOFF_SAMPLES_PER_CLASS must be a non-negative integer"
[[ "${TAKEOFF_LOGIC}" == "any" || "${TAKEOFF_LOGIC}" == "all" ]] \
    || die "TAKEOFF_LOGIC must be any or all"
is_bool "${ONLINE_EVAL}" || die "ONLINE_EVAL must be 0 or 1"
is_bool "${DISABLE_VIS}" || die "DISABLE_VIS must be 0 or 1"
is_bool "${AUTO_RESUME}" || die "AUTO_RESUME must be 0 or 1"
is_bool "${RESUME}" || die "RESUME must be 0 or 1"
is_bool "${RUN_FOREGROUND}" || die "RUN_FOREGROUND must be 0 or 1"
is_bool "${PRESERVE_CUDA_VISIBLE_DEVICES}" || die "PRESERVE_CUDA_VISIBLE_DEVICES must be 0 or 1"
is_bool "${DRY_RUN}" || die "DRY_RUN must be 0 or 1"
is_bool "${CALIBRATION}" || die "CALIBRATION must be 0 or 1"
is_bool "${Q_LORA}" || die "Q_LORA must be 0 or 1"
is_bool "${Q_USE_EMA}" || die "Q_USE_EMA must be 0 or 1"
if [[ "${Q_LORA}" == "1" ]]; then
    (( Q_BOOTSTRAP_UPDATES == 0 )) || die "LoRA q starts equal to p; Q_BOOTSTRAP_UPDATES must be 0"
    Q_BOOTSTRAP=0
fi
if [[ "${RESUME}" == "1" ]]; then
    AUTO_RESUME=1
elif [[ "${AUTO_RESUME}" != "0" ]]; then
    die "this is a fresh-run launcher; set RESUME=1 to continue an existing EXP_NAME"
fi
[[ "${Q_OPTIMIZER}" == "adamw" || "${Q_OPTIMIZER}" == "sgd" ]] \
    || die "Q_OPTIMIZER must be adamw or sgd"
(( Q_BOOTSTRAP_UPDATES == 0 )) || echo "WARNING: Q_BOOTSTRAP_UPDATES=${Q_BOOTSTRAP_UPDATES} \
breaks the q == p initialisation; the conditional term will NOT start at zero gradient."

[[ -n "${WEIGHT}" ]] || die "WEIGHT is unset. Calibrate it on the SUSTAINED
grad_ratio_vlm_fd (target 0.22-0.30, aim 0.25). The term is exactly 0 at step 0
because q == p, so a short GMM-style 600-1500 window reads nothing:

  CALIBRATION=1 WEIGHT=<seed> EXP_NAME=<unique> bash ${BASH_SOURCE[0]}
  ${PY_BIN} scripts/analyze_vlm_delta_run.py --calibration --weight <seed> \\
      ${OUTPUT_DIR}/${PROJECT}/<EXP_NAME>

then relaunch with the WEIGHT that command prints."

IFS=',' read -r -a GPU_IDS <<< "${GPUS}"
NPROC_PER_NODE="${#GPU_IDS[@]}"
GLOBAL_BATCH=$(( BATCH_SIZE * NPROC_PER_NODE ))

if [[ "${CALIBRATION}" == "1" ]]; then
    STEPS_PER_EPOCH="${CAL_STEPS}"
    EPOCHS=1
    ONLINE_EVAL=0
    DISABLE_VIS=1
    TAKEOFF_GATE_STEP=-1
    : "${EXP_NAME:=jitB_uncond_vlmdelta100_CAL_w${WEIGHT}}"
else
    TAKEOFF_GATE_STEP=-1
    : "${EXP_NAME:=jitB_uncond_vlmdelta100_w${WEIGHT}}"
fi

TOTAL_STEPS=$(( EPOCHS * STEPS_PER_EPOCH ))
SAMPLES_PER_CLASS_AT_END=$(( TOTAL_STEPS * GLOBAL_BATCH / NUM_TRAIN_CLASSES ))

if [[ "${CALIBRATION}" != "1" ]] && (( TAKEOFF_SAMPLES_PER_CLASS > 0 )); then
    TAKEOFF_GATE_STEP=$(( TAKEOFF_SAMPLES_PER_CLASS * NUM_TRAIN_CLASSES / GLOBAL_BATCH ))
    (( TAKEOFF_GATE_STEP > DELTA_RAMP )) \
        || die "derived gate step ${TAKEOFF_GATE_STEP} is inside the ${DELTA_RAMP}-step ramp"
    (( TAKEOFF_GATE_STEP < TOTAL_STEPS )) || die "\
the takeoff gate needs ${TAKEOFF_SAMPLES_PER_CLASS} samples/class = step
${TAKEOFF_GATE_STEP}, but this run is only ${TOTAL_STEPS} steps
(${SAMPLES_PER_CLASS_AT_END} samples/class at global batch ${GLOBAL_BATCH})."
fi

(( GLOBAL_BATCH == 96 )) || echo "NOTE: global batch is ${GLOBAL_BATCH}; sample counts below are computed for this batch."

if [[ "${DRY_RUN}" == "1" ]]; then
    # Preview needs neither CUDA nor checkpoints. Actual launches always perform
    # the p-head precheck below and the trainer's full representation validation.
    P_HEAD_BACKEND=$([[ "${Q_LORA}" == "1" ]] && echo qwen_answer_state || echo timm)
else
[[ -x "${PY_BIN}" ]] || die "Python executable not found: ${PY_BIN}"
[[ -f conditional_main_fd_vlm_delta.py ]] || die "missing entry point: conditional_main_fd_vlm_delta.py"
[[ -d "${DATA_PATH}/train" ]] || die "ImageNet train split not found: ${DATA_PATH}/train"
if [[ "${RESUME}" == "0" ]]; then
    [[ -f "${START_CKPT}" ]] || die "unconditional JiT-B checkpoint not found: ${START_CKPT}"
fi
[[ -f "${P_HEAD}" ]] || die "frozen p-head checkpoint not found: ${P_HEAD}
Run first:
  CUDA_VISIBLE_DEVICES=... ${PY_BIN} -m torch.distributed.run --standalone \\
      --nproc_per_node=4 train_vlm_p_head.py \\
      --output_dir \$(dirname ${P_HEAD}) --class_ids \$(seq 0 10 990 | tr '\\n' ' ')"
for stats_path in "${FD_STATS[@]}"; do
    [[ -f "${stats_path}" ]] || die "missing 100-class FD reference: ${stats_path}
Run first: bash scripts/compute_imagenet100_fd_stats.sh"
done

# The p head must agree with this run's class subset and VLM before we pay for a
# queue fill. The training script re-checks all of it at startup and refuses.
P_HEAD_BACKEND="$("${PY_BIN}" - "${P_HEAD}" "${FD_MODELS[0]}" "${FD_SIZES[0]}" \
    "${FD_POOLS[0]}" "${NUM_TRAIN_CLASSES}" "${CLASS_IDS[@]}" <<'PYEOF'
import sys
from vlm_linear_heads import (P_HEAD_BACKEND_QWEN, load_p_head_checkpoint,
                              p_head_backend)
path, model, target, pool, n_classes = sys.argv[1:6]
class_ids = [int(c) for c in sys.argv[6:]]
ckpt = load_p_head_checkpoint(path)
backend = p_head_backend(ckpt)
problems = []
# The class subset is the run's contract whatever produced z.
if [int(c) for c in ckpt["class_ids"]] != sorted(class_ids):
    problems.append("p head class_ids != this run's --train_class_ids")
if int(ckpt["num_classes"]) != int(n_classes):
    problems.append(f"p head num_classes {ckpt['num_classes']} != {n_classes}")
if backend == P_HEAD_BACKEND_QWEN:
    # z comes from a separately loaded prompted VLM, not from an FD judge, so
    # there is nothing here to compare against FD_MODELS[0]. The training script
    # instantiates that VLM and checks model/layer/prompt-hash itself.
    detail = (f"layer {ckpt['vlm_layer']}, prompt sha "
              f"{ckpt['vlm_prompt_sha256'][:16]}, {ckpt['vlm_prompt']!r}")
else:
    if ckpt["vlm_model_name"] != model:
        problems.append(f"p head VLM {ckpt['vlm_model_name']!r} != judge {model!r}")
    if int(ckpt["vlm_target_size"]) != int(target):
        problems.append(f"p head target_size {ckpt['vlm_target_size']} != {target}")
    if ckpt["vlm_pool_type"] != pool:
        problems.append(f"p head pool {ckpt['vlm_pool_type']!r} != {pool!r}")
    detail = f"pool {ckpt['vlm_pool_type']}, target {ckpt['vlm_target_size']}px"
if problems:
    print("ERROR: " + "\n       ".join(problems), file=sys.stderr)
    raise SystemExit(1)
best = (ckpt.get("train_metadata") or {}).get("best") or {}
print(f"p head OK [{backend}]: {ckpt['num_classes']}-way on "
      f"{ckpt['vlm_model_name']} (d={ckpt['feature_dim']}), {detail}, "
      f"T={float(ckpt['temperature']):.4f}, real val top-1 {best.get('top1')}",
      file=sys.stderr)
print(backend)
PYEOF
)" || die "p-head precheck failed"
fi
if [[ "${Q_LORA}" == "1" && "${P_HEAD_BACKEND}" != "qwen_answer_state" ]]; then
    die "Q_LORA=1 requires a Qwen answer-state P_HEAD"
fi

RUN_DIR="${OUTPUT_DIR}/${PROJECT}/${EXP_NAME}"
LOG_PATH="${LOG_DIR}/${EXP_NAME}.out"
if [[ "${DRY_RUN}" == "1" ]]; then
    RESUME_STEP=preview
elif [[ "${RESUME}" == "1" ]]; then
    [[ -d "${RUN_DIR}/checkpoints" ]] \
        || die "RESUME=1 but no checkpoints under ${RUN_DIR}"
    LATEST_CKPT=""
    for candidate in "${RUN_DIR}"/checkpoints/step_*.pth; do
        [[ -f "${candidate}" ]] || continue
        if [[ -z "${LATEST_CKPT}" || "${candidate}" -nt "${LATEST_CKPT}" ]]; then
            LATEST_CKPT="${candidate}"
        fi
    done
    [[ -n "${LATEST_CKPT}" ]] || die "RESUME=1 but no step_*.pth in ${RUN_DIR}/checkpoints"
    RESUME_STEP="$(basename "${LATEST_CKPT}" .pth | sed 's/step_0*//')"
else
    [[ ! -e "${RUN_DIR}" ]] || die "work directory already exists: ${RUN_DIR}; choose a unique EXP_NAME (or set RESUME=1)"
    [[ ! -e "${LOG_PATH}" ]] || die "log path already exists: ${LOG_PATH}; choose a unique EXP_NAME (or set RESUME=1)"
fi

CMD=(
    "${PY_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_port="${MASTER_PORT}"
    conditional_main_fd_vlm_delta.py
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
    --lr "${LR}" --lr_sched constant --min_lr 0.0
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
    --vlm_p_head "${P_HEAD}"
    --vlm_judge "${VLM_JUDGE}"
    --vlm_delta_weight "${WEIGHT}"
    --vlm_delta_warmup_steps "${DELTA_WARMUP}"
    --vlm_delta_ramp_steps "${DELTA_RAMP}"
    --vlm_delta_clamp "${DELTA_CLAMP}"
    --vlm_q_lr "${Q_LR}"
    --vlm_q_optimizer "${Q_OPTIMIZER}"
    --vlm_q_beta1 "${Q_BETA1}"
    --vlm_q_beta2 "${Q_BETA2}"
    --vlm_q_weight_decay "${Q_WEIGHT_DECAY}"
    --vlm_q_grad_clip "${Q_GRAD_CLIP}"
    --vlm_q_updates_per_step "${Q_UPDATES_PER_STEP}"
    --vlm_q_batch_size "${Q_BATCH_SIZE}"
    --vlm_q_ema_beta "${Q_EMA_BETA}"
    --vlm_q_buffer_size "${Q_BUFFER_SIZE}"
    --vlm_q_bootstrap "${Q_BOOTSTRAP}"
    --vlm_q_bootstrap_updates "${Q_BOOTSTRAP_UPDATES}"
    --vlm_init_diag_samples "${INIT_DIAG_SAMPLES}"
    --vlm_per_class_every "${PER_CLASS_EVERY}"
    --vlm_takeoff_gate_step "${TAKEOFF_GATE_STEP}"
    --vlm_takeoff_cond_delta_min "${TAKEOFF_COND_DELTA_MIN}"
    --vlm_takeoff_rank_ratio_max "${TAKEOFF_RANK_RATIO_MAX}"
    --vlm_takeoff_logic "${TAKEOFF_LOGIC}"
    --cond_probe
    --ckpt_target_minutes "${CKPT_TARGET_MINUTES}"
    --disable_wandb
)

if [[ "${RESUME}" == "1" ]]; then CMD+=(--auto_resume); fi
if [[ "${Q_USE_EMA}" == "1" ]]; then CMD+=(--vlm_q_use_ema); fi
if [[ "${Q_LORA}" == "1" ]]; then
    CMD+=(--vlm_q_lora --vlm_disable_tf32)
fi

if [[ -n "${HEAD_TEMPERATURE}" ]]; then
    CMD+=(--vlm_head_temperature "${HEAD_TEMPERATURE}")
fi
if [[ "${P_HEAD_BACKEND}" == "qwen_answer_state" ]]; then
    CMD+=(
        --vlm_microbatch "${VLM_MICROBATCH}"
        --vlm_samples_per_step "${VLM_SAMPLES_PER_STEP}"
        --vlm_dtype "${VLM_DTYPE}"
    )
    # The queue fill cannot seed this backend's replay buffer -- the extractor is
    # not one of the FD judges, so none of its features are computed there. The
    # training script runs a standalone bootstrap pass instead, and at ~46 img/s
    # the default 50,000 is ~18 min PER RANK to overwrite a 20,000-entry buffer
    # ten times over.
    if (( Q_BOOTSTRAP > 4 * Q_BUFFER_SIZE / NPROC_PER_NODE )); then
        echo "NOTE: Q_BOOTSTRAP=${Q_BOOTSTRAP} costs ~$(( Q_BOOTSTRAP / 46 / 60 )) min/rank of 7B \
forward to fill a ${Q_BUFFER_SIZE}-entry buffer that ${NPROC_PER_NODE} ranks fill \
${NPROC_PER_NODE}x faster. Q_BOOTSTRAP=$(( Q_BUFFER_SIZE / NPROC_PER_NODE )) is enough to fill it once."
    fi
fi
if [[ "${ONLINE_EVAL}" == "1" ]]; then CMD+=(--online_eval); fi
if [[ "${DISABLE_VIS}" == "1" ]]; then CMD+=(--disable_vis); fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA <<< "${EXTRA_ARGS}"
    CMD+=("${EXTRA[@]}")
fi

MODE_LABEL=$([[ "${CALIBRATION}" == "1" ]] && echo "WEIGHT CALIBRATION" \
    || { [[ "${RESUME}" == "1" ]] && echo "full trial (RESUMING from step ${RESUME_STEP})" || echo "full trial"; })
echo "Experiment:       ${PROJECT}/${EXP_NAME} (${MODE_LABEL})"
echo "Objective:        L_FD + ${WEIGHT} * E[log q(c|x) - log p(c|x)]"
echo "GPUs:             ${GPUS} (${NPROC_PER_NODE} ranks)"
echo "Batch:            ${BATCH_SIZE}/GPU, global ${GLOBAL_BATCH}"
echo "LR:               ${LR} (generator AdamW; ${NPROC_PER_NODE} ranks)"
echo "Classes:          ${NUM_TRAIN_CLASSES} (${CLASS_IDS[0]} ${CLASS_IDS[1]} ... ${CLASS_IDS[-1]}), stride-10 subset"
echo "Iterations:       ${TOTAL_STEPS} = ${SAMPLES_PER_CLASS_AT_END} samples/class"
if [[ "${DISABLE_VIS}" == "0" ]]; then
    echo "Visualization:    initial grid and every $(( VIS_FREQ * STEPS_PER_EPOCH )) steps; ${#VIS_CLASSES[@]} classes, online + EMA"
fi
if [[ "${P_HEAD_BACKEND}" == "qwen_answer_state" ]]; then
    if [[ "${Q_LORA}" == "1" ]]; then
        echo "VLM:              Qwen2.5-VL frozen base + trainable q LoRA adapters"
        echo "                  two image-VJP branches plus student CE backward per step"
    else
        echo "VLM:              Qwen2.5-VL frozen answer state, separate from FD judges"
        echo "                  adds a 7B forward+backward per scored image every step"
    fi
    echo "                  (microbatch ${VLM_MICROBATCH}, ${VLM_DTYPE}, samples/rank/step \
${VLM_SAMPLES_PER_STEP:-all}). Check s/step in the first 50 log lines before"
    echo "                  committing. This precision configuration requires fresh timing/memory measurements."
else
    echo "VLM:              ${FD_MODELS[0]} (judge '${VLM_JUDGE}', pool ${FD_POOLS[0]}, ${FD_SIZES[0]}px) -- FROZEN"
fi
echo "p head:           ${P_HEAD} -- FROZEN"
Q_BATCH_EFF=$(( Q_BATCH_SIZE > 0 ? Q_BATCH_SIZE : GLOBAL_BATCH ))
Q_REUSE=$(awk -v b="${Q_BATCH_EFF}" -v u="${Q_UPDATES_PER_STEP}" -v g="${GLOBAL_BATCH}" \
    'BEGIN{printf "%.2f", b*u/g}')
echo "q head:           init from p; ${Q_OPTIMIZER} lr=${Q_LR} wd=${Q_WEIGHT_DECAY}, ${Q_UPDATES_PER_STEP} update(s)/step,"
if [[ "${Q_LORA}" == "1" ]]; then
    echo "                  fresh scored images with recomputed adapted features; no replay buffer"
else
    echo "                  batch ${Q_BATCH_EFF} from a ${Q_BUFFER_SIZE}-entry class-balanced buffer"
fi
if [[ "${Q_LORA}" == "1" ]]; then
    echo "                  -> ${Q_UPDATES_PER_STEP} pass(es) per fresh scored batch"
else
    echo "                  -> sample reuse ${Q_REUSE}x (keep near 1; >2 memorises the buffer),"
fi
echo "                  AdamW betas=(${Q_BETA1}, ${Q_BETA2})"
if [[ "${Q_USE_EMA}" == "1" ]]; then
    echo "                  generator reads EMA teacher, beta=${Q_EMA_BETA}"
else
    echo "                  generator reads current student directly; no q EMA updates"
fi
echo "Temperature:      ${HEAD_TEMPERATURE:-from the p-head checkpoint} (shared by p and q)"
if awk -v v="${DELTA_CLAMP}" 'BEGIN{exit !(v+0 > 0)}'; then
    echo "Clamp:            ${DELTA_CLAMP} (ON -- CHANGES the objective; report vlm_delta_clamp_frac with the result)"
else
    echo "Clamp:            off -- the objective is unmodified"
fi
if [[ "${CALIBRATION}" == "1" ]]; then
    echo "Takeoff gate:     disabled (calibration run)"
    echo "READ:             ${PY_BIN} scripts/analyze_vlm_delta_run.py --calibration \\"
    echo "                      --weight ${WEIGHT} ${RUN_DIR}"
    echo "                  grad_ratio_vlm_fd starts at EXACTLY 0 (q == p) and grows;"
    echo "                  take the median of the LAST window, not the ramp."
elif (( TAKEOFF_GATE_STEP >= 0 )); then
    echo "Takeoff gate:     ${TAKEOFF_SAMPLES_PER_CLASS} samples/class -> step ${TAKEOFF_GATE_STEP}"
    echo "                  cond_delta>=${TAKEOFF_COND_DELTA_MIN}, p_rank/chance<=${TAKEOFF_RANK_RATIO_MAX}, abort=${TAKEOFF_LOGIC} failure"
else
    echo "Takeoff gate:     disabled (set TAKEOFF_SAMPLES_PER_CLASS>0 to enable)"
fi
echo "Log:              ${LOG_PATH}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run: command only; no files created or GPU processes started."
    exit 0
fi
mkdir -p "${LOG_DIR}"
if [[ "${PRESERVE_CUDA_VISIBLE_DEVICES}" == "0" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPUS}"
fi
if [[ "${RUN_FOREGROUND}" == "1" ]]; then
    "${CMD[@]}" 2>&1 | tee -a "${LOG_PATH}"
else
    setsid "${CMD[@]}" \
        >>"${LOG_PATH}" 2>&1 < /dev/null &
    pid=$!
    echo "Started PID ${pid}. Follow with: tail -f ${LOG_PATH}"
fi
