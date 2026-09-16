#!/usr/bin/env bash
# Run on the destination server with network access; no GPU is needed.
# Background example (from the repository root):
#   mkdir -p logs
#   nohup bash scripts/cache_fd_models.sh > logs/cache_fd_models.log 2>&1 < /dev/null &
#   tail -n 50 -f logs/cache_fd_models.log
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/cache_fd_models.sh [--imports-only]

Check Python imports, then load/cache SigLIP, MAE, Inception, and the
ResNet-50 probe on CPU. --imports-only stops after the startup checks.
This does not launch training or prepare the separate Qwen snapshot.

Environment overrides:
  PY_BIN                  Python executable (default: repo/.venv/bin/python)
  HF_HOME                 Default: /mnt/projects/jg/kaifany/.hf
  HF_HUB_CACHE            Default: $HF_HOME/hub
  TORCH_HOME              Default: /mnt/projects/jg/kaifany/.torch
  CACHE_CPU_THREADS       CPU threads (default: 2)
  CACHE_TRACEBACK_SECONDS Dump a diagnostic stack for a slow stage
                          (default: 120; 0 disables it)

Hugging Face online access is enabled inside this script only.
HELP
}

imports_only=0
if (( $# > 1 )); then
    usage >&2
    exit 2
fi
case "${1:-}" in
    "") ;;
    --imports-only) imports_only=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

printf '[cache] Wrapper started (PID %s); preparing cache environment.\n' "$$"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PY_BIN="${PY_BIN:-${repo_root}/.venv/bin/python}"
export HF_HOME="${HF_HOME:-/mnt/projects/jg/kaifany/.hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/mnt/projects/jg/kaifany/.torch}"
export HF_HUB_OFFLINE=0
export CACHE_CPU_THREADS="${CACHE_CPU_THREADS:-2}"
export CACHE_TRACEBACK_SECONDS="${CACHE_TRACEBACK_SECONDS:-120}"
[[ "$CACHE_CPU_THREADS" =~ ^[1-9][0-9]*$ ]] || {
    echo 'ERROR: CACHE_CPU_THREADS must be a positive integer.' >&2
    exit 2
}
[[ "$CACHE_TRACEBACK_SECONDS" =~ ^[0-9]+$ ]] || {
    echo 'ERROR: CACHE_TRACEBACK_SECONDS must be a nonnegative integer.' >&2
    exit 2
}
export OMP_NUM_THREADS="$CACHE_CPU_THREADS"
export MKL_NUM_THREADS="$CACHE_CPU_THREADS"
command -v "$PY_BIN" >/dev/null 2>&1 || {
    printf 'ERROR: Python executable not found: %s\n' "$PY_BIN" >&2
    exit 2
}
printf '[cache] Repository: %s\n[cache] Python: %s\n' "$repo_root" "$PY_BIN"
printf '[cache] HF_HOME=%s\n[cache] HF_HUB_CACHE=%s\n[cache] TORCH_HOME=%s\n' \
    "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME"
echo '[cache] Starting Python in unbuffered mode (HF_HUB_OFFLINE=0).'

# exec preserves the background PID and Python's exit status.
exec "$PY_BIN" -u -X faulthandler - "$imports_only" <<'PY'
print("[cache] Python started; enabling startup diagnostics.", flush=True)
import faulthandler
import os
import sys
import time

traceback_seconds = int(os.environ["CACHE_TRACEBACK_SECONDS"])
started = time.monotonic()
stage_started = started
current_stage = "startup"


def report(message):
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{timestamp}] {message}", flush=True)


def begin_stage(name):
    global current_stage, stage_started
    faulthandler.cancel_dump_traceback_later()
    current_stage = name
    stage_started = time.monotonic()
    report(f"START: {name}")
    if traceback_seconds:
        faulthandler.dump_traceback_later(traceback_seconds, repeat=False)


def end_stage():
    faulthandler.cancel_dump_traceback_later()
    report(f"DONE: {current_stage} ({time.monotonic() - stage_started:.1f}s)")


try:
    report(f"PID={os.getpid()}; CPU threads={os.environ['CACHE_CPU_THREADS']}")
    if traceback_seconds:
        report(
            f"A stage taking over {traceback_seconds}s triggers one diagnostic "
            "stack dump. That Timeout message does not stop the process."
        )

    begin_stage("Importing torch")
    import torch
    end_stage()

    begin_stage("Setting CPU threads")
    torch.set_num_threads(int(os.environ["CACHE_CPU_THREADS"]))
    end_stage()

    begin_stage("Importing FD feature-model loader")
    from frechet_distance.repr_models import load_repr_model
    end_stage()

    begin_stage("Importing torchvision probe")
    from torchvision.models import ResNet50_Weights, resnet50
    end_stage()

    if sys.argv[1] == "1":
        report("SUCCESS: Import/setup checks completed; no models were loaded.")
    else:
        import gc

        for name, size in (
            ("vit_so400m_patch16_siglip_256.v2_webli", 224),
            ("vit_large_patch16_224.mae", 224),
            ("inception", 256),
        ):
            begin_stage(f"Loading/caching on CPU: {name}")
            model = load_repr_model(name, device="cpu", target_size=size)[0]
            del model
            gc.collect()
            end_stage()

        begin_stage("Loading/caching on CPU: ResNet-50 IMAGENET1K_V2 probe")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
        del model
        gc.collect()
        end_stage()
        report("SUCCESS: FD models and probe cached successfully.")
        report("The separate Qwen snapshot and p-head assets are not checked here.")

    report(f"Finished in {time.monotonic() - started:.1f}s.")
except BaseException:
    report(f"FAILED during: {current_stage}; traceback follows.")
    raise
finally:
    faulthandler.cancel_dump_traceback_later()
PY
