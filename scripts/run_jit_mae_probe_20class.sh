#!/usr/bin/env bash
# 10k-step diagnostic: restrict generator labels and all real/eval targets to
# 20 ImageNet classes while retaining the frozen 1000-way MAE linear probe.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

: "${CUDA_GPUS:=4,5,6,7}"
: "${MASTER_PORT:=29521}"
: "${PY_BIN:=/home/nvidia/miniconda3/envs/fdloss/bin/python}"
: "${DATA_PATH:=/data/dataset/imagenet}"
: "${START_CKPT:=work_dirs/JiT_cond_sweep/r8_mae_probe_inception_lam1e5/checkpoints/step_0071799.pth}"
: "${MAE_PROBE_CKPT:=work_dirs/mae_probe_vitl_cls/best.pt}"
: "${STATS_DIR:=data/fid_stats/imagenet20_v1}"
: "${OUTPUT_DIR:=work_dirs}"
: "${PROJECT:=JiT_cond_sweep}"
: "${EXP_NAME:=r9_mae_probe_20class_lam5e5}"
: "${LOG_DIR:=sweep_logs}"

# 5e-5 is a deliberate midpoint between the ineffective 1e-5 run and the
# 1e-4 run that overpowered FD. A short ramp avoids an abrupt gradient jump.
: "${LAMBDA_MAE:=5e-5}"
: "${COND_WARMUP:=0}"
: "${COND_RAMP:=500}"
: "${TARGET_LOGP:=-2.302585}"  # stop per-sample pushing at p(target)=10%
: "${MIN_PROBE_VAL_TOP1:=0.50}"
: "${EPOCHS:=8}"
: "${STEPS_PER_EPOCH:=1250}"
: "${QUEUE_SIZE:=50000}"
: "${NUM_EVAL_IMAGES:=20000}"
: "${RUN_FOREGROUND:=1}"

CLASS_IDS=(0 9 88 130 207 279 281 340 360 387 404 417 444 555 569 817 920 949 974 979)
FD_MODELS=(vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception)
FD_POOLS=(cls cls cls)
FD_SIZES=(224 224 256)
FD_STATS=(
    "${STATS_DIR}/siglip_cls.npz"
    "${STATS_DIR}/mae_cls.npz"
    "${STATS_DIR}/inception.npz"
)

IFS=',' read -r -a GPU_IDS <<< "${CUDA_GPUS}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"

if [[ ! -x "${PY_BIN}" ]]; then
    echo "ERROR: Python executable not found: ${PY_BIN}" >&2
    exit 2
fi
if [[ ! -f "${START_CKPT}" ]]; then
    echo "ERROR: starting JiT checkpoint not found: ${START_CKPT}" >&2
    exit 2
fi
if [[ ! -f "${MAE_PROBE_CKPT}" ]]; then
    echo "ERROR: trained MAE probe not found: ${MAE_PROBE_CKPT}" >&2
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
    conditional_main_fd_mae.py
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
    --vis_freq 1 --eval_freq 2 --online_eval
    --print_freq 20 --milestone_interval 2 --save_freq 2
    --epochs "${EPOCHS}" --steps_per_epoch "${STEPS_PER_EPOCH}"
    --warmup_epochs 0
    --lr 1e-5 --lr_sched cosine --min_lr 0.0
    --fd_eigvalsh --fd_ema_beta 0.999
    --queue_size "${QUEUE_SIZE}"
    --fd_repr_models "${FD_MODELS[@]}"
    --fd_repr_pool_types "${FD_POOLS[@]}"
    --fd_target_sizes "${FD_SIZES[@]}"
    --fd_repr_stats_paths "${FD_STATS[@]}"
    --fid_stats_path "${FD_STATS[2]}"
    --train_class_ids "${CLASS_IDS[@]}"
    --class_of_interest "${CLASS_IDS[@]}"
    --force_class_of_interest
    --lambda_cond "${LAMBDA_MAE}"
    --mae_probe_checkpoint "${MAE_PROBE_CKPT}"
    --mae_probe_min_val_top1 "${MIN_PROBE_VAL_TOP1}"
    --mae_probe_grad_check
    --cond_warmup_steps "${COND_WARMUP}"
    --cond_ramp_steps "${COND_RAMP}"
    --cond_target_logp "${TARGET_LOGP}"
    --mae_eot_views 1
    --mae_eot_noise_std 0.0
    --mae_eot_crop_min 1.0
    --cond_probe
    --disable_wandb
)

echo "Experiment: ${PROJECT}/${EXP_NAME}"
echo "Start checkpoint: ${START_CKPT}"
echo "Classes (${#CLASS_IDS[@]}): ${CLASS_IDS[*]}"
echo "Lambda: ${LAMBDA_MAE}; warmup=${COND_WARMUP}; ramp=${COND_RAMP}"
echo "Target cap: log(p)=${TARGET_LOGP} (p=10%)"
echo "Log: ${LOG_PATH}"

if [[ "${RUN_FOREGROUND}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_GPUS}" "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
else
    CUDA_VISIBLE_DEVICES="${CUDA_GPUS}" setsid "${CMD[@]}" \
        >"${LOG_PATH}" 2>&1 < /dev/null &
    echo "Started PID $!. Follow with: tail -f ${LOG_PATH}"
fi
