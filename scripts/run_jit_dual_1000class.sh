#!/usr/bin/env bash
# Dual semantic judge (frozen MAE linear probe + frozen Qwen2.5-VL) on the
# standard full-ImageNet setup: 1000 classes, the clean unconditional JiT-B
# base, and the default 1000-class FD reference statistics.
#
#   L = L_FD + lambda_mae(s) * L_mae + lambda_vlm(s) * L_vlm
#
# Differences from run_jit_dual_20class.sh, and why:
#
#   * Starts from checkpoints/base/JiT-B-uncond.pth rather than r8's step
#     71799. r8 was itself 72k steps of FD training against 1000-class stats
#     with a conditional term so weak (ratio 0.03) it learned nothing, so
#     inheriting it only muddied the picture -- and for the 20-class runs it
#     was actively wrong, since its FD target was the 1000-class marginal.
#   * No --train_class_ids / --force_class_of_interest: the generator sees all
#     1000 labels and online eval samples all 1000, so the reported FID is the
#     standard ImageNet number against guided_diffusion_stats.npz.
#   * No --fd_repr_stats_paths: judges.py::infer_stats_path resolves each
#     judge's default 1000-class stats automatically. Do not pass the
#     imagenet20_v1 paths here.
#   * --class_of_interest is visualization only now. Without
#     --force_class_of_interest it does not touch eval.
#   * --vlm_distractor_pool all, because there is no training subset to draw a
#     distractor from.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

: "${CUDA_GPUS:=4,5,6,7}"
: "${MASTER_PORT:=29581}"
: "${PY_BIN:=/home/nvidia/miniconda3/envs/fdloss/bin/python}"
: "${DATA_PATH:=/data/dataset/imagenet}"
: "${START_CKPT:=checkpoints/base/JiT-B-uncond.pth}"
: "${MAE_PROBE_CKPT:=work_dirs/mae_probe_vitl_cls/best.pt}"
: "${VLM_MODEL:=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
: "${INCEPTION_STATS:=data/fid_stats/guided_diffusion_stats.npz}"
: "${OUTPUT_DIR:=work_dirs}"
: "${PROJECT:=JiT_cond_sweep}"
: "${EXP_NAME:=r13_dual_1000class}"
: "${LOG_DIR:=sweep_logs}"

# -- FD statistics window --
# 0.99 (~100-step EMA) rather than the repo default 0.999. At 0.999 the
# covariance spans ~1000 steps, so a collapsed generator whose single mode
# drifts still looks diverse -- that is how r11 reached in-loss inception FD
# 12.5 while its frozen-checkpoint eval read 120.7. See DUAL_JUDGE.md.
: "${FD_EMA_BETA:=0.99}"

# -- MAE probe term --
# CALIBRATE BEFORE TRUSTING. Both lambdas are setup-specific: they depend on
# the FD window, the number of FD judges, the reference stats, and the start
# checkpoint. Every value carried over from a previous run so far has been
# wrong by 8-15x. The number below was measured on this exact configuration;
# re-measure if you change any of those four things.
# Measured on this config: lambda_mae 1e-4 -> grad_ratio_mae_fd 0.062 at step 0.
# So 5.7e-4 would give 0.35 *at step 0* -- and that is the trap r8 fell into.
# The FD term is self-normalized (fid / (fid.detach() + eps)), so as the raw FD
# falls its gradient grows: r8 watched its ratio decay 0.305 -> 0.037 over 72k
# steps on this exact setup and ended up learning nothing. This base starts at
# fd_loss_raw 559 (inception FID 330), so there is a lot of that decay coming.
#
# Hence a long ramp instead of a fixed lambda: lambda rises linearly while
# grad_x_fd rises too, which keeps the ratio roughly in band instead of letting
# it bleed out. 2.0e-3 over 20k steps is calibrated for grad_x_fd growing ~5x;
# the envelope is ratio ~0.6 if it only doubles and ~0.15 if it grows 8x.
# CHECK grad_ratio_mae_fd AT 5k / 15k / 30k AND ADJUST -- this is an
# extrapolation, not a measurement.
#
# The ramp is independently justified: the base model's FD is poor, and
# pushing semantics onto a generator that cannot yet make realistic images is
# the r5 mistake. Let FD fix realism first.
: "${LAMBDA_MAE:=2.0e-3}"
: "${COND_WARMUP:=0}"
: "${COND_RAMP:=20000}"
: "${TARGET_LOGP:=-2.302585}"   # stop pushing a sample at p(target)=10%
: "${MIN_PROBE_VAL_TOP1:=0.50}"
# This base reads noise_delta ~0.38, not the ~0.75 of the r8 checkpoint, so the
# 20-class run's 0.35 threshold would fire from step 0. Half of base, as there.
: "${NOISE_DELTA_WARN:=0.20}"

# -- VLM term --
# Warmup is longer than the 20-class run (10k). Starting from an unconditional
# base on 1000 classes, "Is the subject of this image a golden retriever?" is
# correctly answered No for a long time; r5 ran a VLM from step 0 in exactly
# that regime for 100k steps and p(Yes|target) fell from 0.0014 to 0.0003.
# Measured: lambda_vlm 1e-5 -> grad_ratio_vlm_fd 0.069 at step 0. Scaled by the
# same ~5x grad_x_fd growth expected by the time it switches on at step 20k, so
# 6e-5 targets ~0.10 there. Subordinate to the MAE term by design.
: "${LAMBDA_VLM:=6e-5}"
: "${VLM_WARMUP:=20000}"
: "${VLM_RAMP:=10000}"
: "${VLM_TARGET_LOGP:=-0.69}"   # binary chance is -log(2); cap at p=50%
: "${VLM_SAMPLES:=2}"
: "${VLM_MICROBATCH:=1}"
: "${VLM_QUESTION_MODE:=pairwise}"
: "${VLM_DISTRACTOR_POOL:=all}"

: "${EPOCHS:=60}"
: "${STEPS_PER_EPOCH:=1250}"    # 60 x 1250 = 75,000 iterations
: "${QUEUE_SIZE:=50000}"
: "${NUM_EVAL_IMAGES:=50000}"   # standard ImageNet FID sample count
: "${EVAL_FREQ:=10}"
: "${SAVE_FREQ:=5}"
: "${PRINT_FREQ:=20}"
: "${AUTO_RESUME:=1}"
: "${ONLINE_EVAL:=1}"
: "${RUN_FOREGROUND:=0}"

# Visualization only -- 20 recognizable classes so the grids stay readable.
# Not passed with --force_class_of_interest, so eval still covers all 1000.
VIS_CLASSES=(0 9 88 130 207 279 281 340 360 387 404 417 444 555 569 817 920 949 974 979)
FD_MODELS=(vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception)
FD_POOLS=(cls cls cls)
FD_SIZES=(224 224 256)

IFS=',' read -r -a GPU_IDS <<< "${CUDA_GPUS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"

for v in LAMBDA_MAE LAMBDA_VLM; do
    if [[ "${!v}" == "__CALIBRATE__" ]]; then
        echo "ERROR: ${v} is unset. Run the calibration first (see header) and" >&2
        echo "       pass ${v}=<value>, or edit the default in this script." >&2
        exit 2
    fi
done
if [[ ! -x "${PY_BIN}" ]]; then
    echo "ERROR: Python executable not found: ${PY_BIN}" >&2
    exit 2
fi
for f in "${START_CKPT}" "${MAE_PROBE_CKPT}" "${INCEPTION_STATS}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required file not found: ${f}" >&2
        exit 2
    fi
done
if [[ ! -d "${VLM_MODEL}" ]]; then
    echo "ERROR: local VLM checkpoint not found: ${VLM_MODEL}" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${EXP_NAME}.out"

CMD=(
    "${PY_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_port="${MASTER_PORT}"
    conditional_main_fd_dual.py
    --data_path "${DATA_PATH}"
    --load_from "${START_CKPT}"
    --output_dir "${OUTPUT_DIR}"
    --project "${PROJECT}"
    --exp_name "${EXP_NAME}"
    --batch_size 21
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0
    --ema_type edm --num_sampling_steps 1
    --eval_bsz 256 --num_images_for_eval_and_search "${NUM_EVAL_IMAGES}"
    --vis_freq 2 --eval_freq "${EVAL_FREQ}"
    --print_freq "${PRINT_FREQ}" --milestone_interval 10 --save_freq "${SAVE_FREQ}"
    --epochs "${EPOCHS}" --steps_per_epoch "${STEPS_PER_EPOCH}"
    --warmup_epochs 0
    --lr 1e-5 --lr_sched cosine --min_lr 0.0
    --fd_eigvalsh --fd_ema_beta "${FD_EMA_BETA}"
    --queue_size "${QUEUE_SIZE}"
    --fd_repr_models "${FD_MODELS[@]}"
    --fd_repr_pool_types "${FD_POOLS[@]}"
    --fd_target_sizes "${FD_SIZES[@]}"
    --fid_stats_path "${INCEPTION_STATS}"
    --class_of_interest "${VIS_CLASSES[@]}"
    --lambda_cond "${LAMBDA_MAE}"
    --mae_probe_checkpoint "${MAE_PROBE_CKPT}"
    --mae_probe_min_val_top1 "${MIN_PROBE_VAL_TOP1}"
    --mae_probe_grad_check
    --cond_warmup_steps "${COND_WARMUP}"
    --cond_ramp_steps "${COND_RAMP}"
    --cond_target_logp "${TARGET_LOGP}"
    --noise_delta_warn "${NOISE_DELTA_WARN}"
    --mae_eot_views 1
    --mae_eot_noise_std 0.0
    --mae_eot_crop_min 1.0
    --lambda_vlm "${LAMBDA_VLM}"
    --cond_vlm_model "${VLM_MODEL}"
    --vlm_warmup_steps "${VLM_WARMUP}"
    --vlm_ramp_steps "${VLM_RAMP}"
    --vlm_target_logp "${VLM_TARGET_LOGP}"
    --vlm_samples_per_step "${VLM_SAMPLES}"
    --vlm_microbatch "${VLM_MICROBATCH}"
    --vlm_question_mode "${VLM_QUESTION_MODE}"
    --vlm_distractor_pool "${VLM_DISTRACTOR_POOL}"
    --vlm_dtype bf16
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

echo "Experiment:  ${PROJECT}/${EXP_NAME}"
echo "GPUs:        ${CUDA_GPUS} (${NPROC_PER_NODE} ranks)"
echo "Base model:  ${START_CKPT}  (clean unconditional JiT-B)"
echo "Classes:     all 1000; FD vs default 1000-class stats; eval FID vs ${INCEPTION_STATS}"
echo "Iterations:  $(( EPOCHS * STEPS_PER_EPOCH ))"
echo "FD window:   fd_ema_beta=${FD_EMA_BETA}"
echo "MAE term:    lambda=${LAMBDA_MAE} warmup=${COND_WARMUP} ramp=${COND_RAMP} cap=${TARGET_LOGP}"
echo "VLM term:    lambda=${LAMBDA_VLM} warmup=${VLM_WARMUP} ramp=${VLM_RAMP} cap=${VLM_TARGET_LOGP}"
echo "Log:         ${LOG_PATH}"

if [[ "${RUN_FOREGROUND}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_GPUS}" "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
else
    CUDA_VISIBLE_DEVICES="${CUDA_GPUS}" setsid "${CMD[@]}" \
        >"${LOG_PATH}" 2>&1 < /dev/null &
    echo "Started PID $!. Follow with: tail -f ${LOG_PATH}"
fi
