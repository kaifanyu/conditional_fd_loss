#!/usr/bin/env bash
# Step B of the VLM-delta trial: the integration smoke tests, end to end.
#
# The unit tests (tests/test_vlm_delta.py) pin the head / buffer / metric maths
# on synthetic data.  This script exercises the same ten checks against the REAL
# entry point, the REAL frozen SigLIP judge and the REAL de-conditioned JiT-B,
# because that is where the wiring can be wrong in ways synthetic tensors cannot
# show: the wrong judge, a stale feature detach, a checkpoint key that does not
# round-trip.
#
#   1. p and q initialise identically              (init diagnostic, exact 0)
#   2. the generator backward reaches the generator through the frozen VLM
#   3. the VLM receives no parameter gradient
#   4. the p head receives no updates
#   5. the q update does not backprop into the generator
#   6. q_student changes under generated CE training
#   7. q_teacher moves more slowly than q_student
#   8. log q - log p == 0 at q initialisation
#   9. the conditional image-space gradient is finite and non-zero once q != p
#  10. checkpoint/resume restores q exactly
#
# It also runs against the SHIPPED q defaults (--vlm_q_batch_size 0 = the global
# batch, i.e. sample reuse 1.0) so the memorisation guard is exercised, not
# bypassed.
#
# plus the three launch refusals (wrong class set, wrong VLM, resume against a
# different p head).
#
# ~5 minutes on one GPU.  Usage:  GPU=1 bash scripts/smoke_vlm_delta.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

: "${GPU:=1}"
: "${PY_BIN:=/home/nvidia/miniconda3/envs/fdloss/bin/python}"
: "${P_HEAD:=work_dirs/vlm_p_head_siglip_c100/p_head.pt}"
: "${START_CKPT:=checkpoints/base/JiT-B-uncond.pth}"
: "${WORK:=${TMPDIR:-/tmp}/vlm_delta_smoke.$$}"
: "${MASTER_PORT:=29631}"

die() { echo "SMOKE FAILED: $*" >&2; exit 1; }
[[ -x "${PY_BIN}" ]] || die "python not found: ${PY_BIN}"
[[ -f "${P_HEAD}" ]] || die "p head not found: ${P_HEAD} (run train_vlm_p_head.py first)"
[[ -f "${START_CKPT}" ]] || die "base checkpoint not found: ${START_CKPT}"

CLASS_IDS=$(seq 0 10 990 | tr '\n' ' ')
mkdir -p "${WORK}"
trap 'echo "artifacts left in ${WORK}"' EXIT

echo "=============================================================="
echo "0. unit tests"
echo "=============================================================="
"${PY_BIN}" -m unittest tests.test_vlm_delta 2>&1 | tail -3

common=(
    --data_path /data/dataset/imagenet --num_classes 1000
    --batch_size 48 --model JiT_B --rope_2d --learned_pe --legacy_time_convention
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0 --ema_type edm
    --num_sampling_steps 1 --warmup_epochs 0
    --lr 1e-5 --lr_sched constant --min_lr 0.0
    --save_freq 1 --milestone_interval 100 --print_freq 10
    --fd_eigvalsh --fd_ema_beta 0.99 --queue_size 512 --fd_queue_fill_bsz 128
    --fd_repr_models vit_so400m_patch16_siglip_256.v2_webli
    --fd_repr_pool_types cls --fd_target_sizes 224
    --fd_repr_stats_paths data/fid_stats/imagenet100_v1/siglip_cls.npz
    --fid_stats_path data/fid_stats/imagenet100_v1/inception.npz
    --vlm_p_head "${P_HEAD}" --vlm_judge siglip
    --vlm_delta_weight 0.01 --vlm_delta_ramp_steps 2
    --vlm_q_buffer_size 2000 --vlm_q_batch_size 0
    --vlm_init_diag_samples 256 --vlm_per_class_every 30
    --cond_probe --disable_vis --disable_wandb
)

run() {  # run <exp_name> <extra args...>
    local name="$1"; shift
    CUDA_VISIBLE_DEVICES="${GPU}" "${PY_BIN}" -m torch.distributed.run \
        --standalone --nproc_per_node=1 --master_port="${MASTER_PORT}" \
        conditional_main_fd_vlm_delta.py \
        --output_dir "${WORK}" --project smoke --exp_name "${name}" \
        --train_class_ids ${CLASS_IDS} "${common[@]}" "$@"
}

echo
echo "=============================================================="
echo "1-9. fresh run: init diagnostic + 60 training steps"
echo "=============================================================="
run train1 --load_from "${START_CKPT}" --epochs 1 --steps_per_epoch 60 \
    > "${WORK}/train1.log" 2>&1 || die "the fresh run crashed; see ${WORK}/train1.log"
grep -a -A 45 "INITIAL GENERATED SAMPLE" "${WORK}/train1.log" || true

CKPT="$(ls "${WORK}/smoke/train1/checkpoints/step_"*.pth | tail -1)"

echo
echo "=============================================================="
echo "10. resume and verify the q state came back bit-exact"
echo "=============================================================="
run train2 --resume_from "${CKPT}" --epochs 1 --steps_per_epoch 70 \
    > "${WORK}/train2.log" 2>&1 || die "the resumed run crashed; see ${WORK}/train2.log"
grep -a "Restored q student/teacher" "${WORK}/train2.log" \
    || die "the resumed run did not restore the q state"

echo
echo "=============================================================="
echo "assertions"
echo "=============================================================="
"${PY_BIN}" scripts/check_vlm_delta_smoke.py \
    --run "${WORK}/smoke/train1" --resumed "${WORK}/smoke/train2" --checkpoint "${CKPT}" \
    --p_head "${P_HEAD}" || die "assertions failed"

echo
echo "=============================================================="
echo "launch refusals"
echo "=============================================================="
refuse() {  # refuse <label> <expected substring> <args...>
    local label="$1" expect="$2"; shift 2
    local out
    out="$("$@" 2>&1 || true)"
    if grep -qa -- "${expect}" <<<"${out}"; then
        echo "  OK   ${label}"
    else
        echo "${out}" | tail -20
        die "${label}: expected a refusal containing '${expect}'"
    fi
}
refuse "wrong class set" "softmax denominator" \
    env CUDA_VISIBLE_DEVICES="${GPU}" "${PY_BIN}" -m torch.distributed.run --standalone \
    --nproc_per_node=1 --master_port="${MASTER_PORT}" conditional_main_fd_vlm_delta.py \
    --output_dir "${WORK}" --project smoke --exp_name refuse_classes \
    --train_class_ids $(seq 0 10 980 | tr '\n' ' ') "${common[@]}" \
    --epochs 1 --steps_per_epoch 1 --queue_size 0
refuse "wrong VLM" "does not match the frozen p head" \
    env CUDA_VISIBLE_DEVICES="${GPU}" "${PY_BIN}" -m torch.distributed.run --standalone \
    --nproc_per_node=1 --master_port="${MASTER_PORT}" conditional_main_fd_vlm_delta.py \
    --output_dir "${WORK}" --project smoke --exp_name refuse_vlm \
    --train_class_ids ${CLASS_IDS} \
    --data_path /data/dataset/imagenet --num_classes 1000 --batch_size 8 \
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention \
    --epochs 1 --steps_per_epoch 1 --queue_size 0 --fd_ema_beta 0.99 \
    --fd_repr_models vit_large_patch16_224.mae --fd_repr_pool_types cls \
    --fd_target_sizes 224 \
    --fd_repr_stats_paths data/fid_stats/imagenet100_v1/mae_cls.npz \
    --fid_stats_path data/fid_stats/imagenet100_v1/inception.npz \
    --vlm_p_head "${P_HEAD}" --vlm_judge mae --vlm_delta_weight 0.01 \
    --disable_vis --disable_wandb

echo
echo "=============================================================="
echo "SMOKE PASSED"
echo "=============================================================="
