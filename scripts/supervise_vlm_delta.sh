#!/usr/bin/env bash
# Keep a VLM-delta trial alive across GPU evictions.
#
# WHY THIS EXISTS
#
# Both full trials of this experiment were killed mid-run by an external SIGTERM
# when another job claimed the GPUs:
#
#   w=0.000429  killed 2026-08-28 15:52 at step 37,020 / 50,000  (6 h 19 m in)
#   w=0.000592  killed 2026-08-28 18:20 at step 14,180 / 50,000  (2 h 25 m in)
#
# The run needs ~8 uninterrupted hours; the GPUs do not offer them. Everything
# needed to continue is already in each checkpoint -- generator, EMA, optimizer,
# FD queues, q student and teacher, q optimizer, q replay buffer, and the p-head
# identity the trainer re-verifies before continuing. So an eviction should cost
# one checkpoint interval, not the run.
#
# This wrapper relaunches from the newest checkpoint whenever the trainer exits
# for any reason other than a deliberate one, and stops on:
#
#   exit 0  training complete (or a clean preemption save -- see below)
#   exit 4  takeoff gate aborted the run on purpose
#   MAX_RESTARTS reached, or a stop file appears
#
# NOTE on exit 0: the trainer also returns 0 after saving on a caught SIGTERM.
# The two are distinguished by comparing the reached step against the target, so
# a preemption-save is resumed rather than mistaken for completion.
#
# Usage:
#   GPUS=1,2,3 BATCH_SIZE=32 LR=7.5e-6 WEIGHT=0.000592 \
#     EXP_NAME=jitB_uncond_vlmdelta100_w0000592 \
#     bash scripts/supervise_vlm_delta.sh
#
# Stop it deliberately with:  touch <RUN_DIR>/STOP_SUPERVISOR
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

: "${EXP_NAME:?EXP_NAME is required}"
: "${OUTPUT_DIR:=work_dirs}"
: "${PROJECT:=JiT_uncond_vlm_delta}"
: "${LOG_DIR:=sweep_logs}"
: "${MAX_RESTARTS:=40}"
: "${RESTART_BACKOFF:=60}"
# Wait for the GPUs to be free again rather than thrashing against whoever
# evicted us. 0 disables the wait.
: "${WAIT_FOR_GPUS:=1}"
: "${GPU_FREE_MIB:=60000}"
: "${GPU_WAIT_MAX_MIN:=180}"

RUN_DIR="${OUTPUT_DIR}/${PROJECT}/${EXP_NAME}"
SUP_LOG="${LOG_DIR}/${EXP_NAME}.supervisor.log"
STOP_FILE="${RUN_DIR}/STOP_SUPERVISOR"
mkdir -p "${LOG_DIR}"

say() { echo "[supervisor $(date -u +%H:%M:%S)] $*" | tee -a "${SUP_LOG}"; }

reached_step() {  # newest checkpoint's step, or -1 when there is none
    local latest n
    latest="$(ls -t "${RUN_DIR}"/checkpoints/step_*.pth 2>/dev/null | head -1)"
    [[ -z "${latest}" ]] && { echo -1; return; }
    # step_0013699.pth -> 13699; step_0000000.pth must give 0, not "".
    n="$(basename "${latest}" .pth | sed 's/^step_//' | sed 's/^0*//')"
    echo "${n:-0}"
}

target_step() {  # EPOCHS * STEPS_PER_EPOCH, matching the launcher's defaults
    echo $(( ${EPOCHS:-40} * ${STEPS_PER_EPOCH:-1250} ))
}

wait_for_gpus() {
    [[ "${WAIT_FOR_GPUS}" != "1" ]] && return 0
    local deadline=$(( SECONDS + GPU_WAIT_MAX_MIN * 60 ))
    IFS=',' read -r -a ids <<< "${GPUS:-0}"
    while (( SECONDS < deadline )); do
        local ok=1
        for g in "${ids[@]}"; do
            local used total free
            used=$(nvidia-smi -i "${g}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
            total=$(nvidia-smi -i "${g}" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
            [[ -z "${used}" || -z "${total}" ]] && { ok=0; break; }
            free=$(( total - used ))
            (( free >= GPU_FREE_MIB )) || { ok=0; break; }
        done
        (( ok == 1 )) && return 0
        say "GPUs ${GPUS} not free (need ${GPU_FREE_MIB} MiB each); waiting 120 s"
        sleep 120
    done
    say "WARNING: gave up waiting for free GPUs after ${GPU_WAIT_MAX_MIN} min; trying anyway"
    return 0
}

TARGET="$(target_step)"
say "supervising ${PROJECT}/${EXP_NAME} -> target step ${TARGET}, max ${MAX_RESTARTS} restarts"
say "stop deliberately with: touch ${STOP_FILE}"

attempt=0
while (( attempt <= MAX_RESTARTS )); do
    if [[ -f "${STOP_FILE}" ]]; then
        say "stop file present; exiting without relaunching"; exit 0
    fi

    step="$(reached_step)"
    if (( step >= TARGET - 1 )); then
        say "reached step ${step} >= target ${TARGET}; done"; exit 0
    fi

    wait_for_gpus

    if (( step < 0 )); then
        # A work dir with no checkpoint means an earlier attempt died before its
        # first save. The fresh launcher refuses to touch an existing dir, and
        # deleting someone's run dir automatically is not this script's call.
        if [[ -d "${RUN_DIR}" ]] && (( attempt > 0 )); then
            say "ERROR: ${RUN_DIR} exists but holds no checkpoint -- the first"
            say "       attempt died before saving. Inspect ${LOG_DIR}/${EXP_NAME}.out,"
            say "       then either remove the dir or fix the launch. Not looping."
            exit 1
        fi
        say "attempt ${attempt}: no checkpoint yet -> FRESH start"
        RESUME=0 RUN_FOREGROUND=1 bash scripts/run_jit_uncond_vlm_delta_100class.sh
    else
        say "attempt ${attempt}: resuming from step ${step} (target ${TARGET})"
        RESUME=1 RUN_FOREGROUND=1 bash scripts/run_jit_uncond_vlm_delta_100class.sh
    fi
    rc=$?

    new_step="$(reached_step)"
    say "attempt ${attempt} exited rc=${rc}; checkpoint step ${step} -> ${new_step}"

    if (( rc == 4 )); then
        say "takeoff gate aborted the run deliberately; not relaunching"; exit 4
    fi
    if (( rc == 0 )) && (( new_step >= TARGET - 1 )); then
        say "training complete at step ${new_step}"; exit 0
    fi
    if (( rc == 0 )); then
        say "exited 0 below target (preemption save) -- relaunching"
    else
        say "exited rc=${rc} (killed or crashed) -- relaunching"
    fi

    if (( new_step <= step )) && (( attempt > 0 )); then
        say "WARNING: no forward progress since the last attempt (still step ${new_step})."
        say "         backing off ${RESTART_BACKOFF}s x3 to avoid a crash loop"
        sleep $(( RESTART_BACKOFF * 3 ))
    else
        sleep "${RESTART_BACKOFF}"
    fi
    attempt=$(( attempt + 1 ))
done

say "ERROR: exhausted ${MAX_RESTARTS} restarts without reaching ${TARGET}"
exit 1
