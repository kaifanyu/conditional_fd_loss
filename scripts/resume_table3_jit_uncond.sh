#!/usr/bin/env bash
# Resume the table_3_JiT (UNCONDITIONAL) run "JiT_B-fd-inception" from its latest
# checkpoint, detached on GPUs 0-3. Survives SSH disconnect (setsid + nohup).
#
# Usage:   bash scripts/resume_table3_jit_uncond.sh
# Monitor: tail -f work_dirs/table_3_JiT/JiT_B-fd-inception/resume_*.out
#          tail -f work_dirs/table_3_JiT/JiT_B-fd-inception/log.txt
# Stop:    kill -INT $(cat work_dirs/table_3_JiT/JiT_B-fd-inception/resume.pid)
set -euo pipefail

REPO=/data/jgu/kai/FD-Loss
cd "$REPO"

# Same conda env the run was trained in (torch 2.6, diffusers, open_clip, ...).
export PATH="/home/nvidia/miniconda3/envs/fdloss/bin:$PATH"
export OMP_NUM_THREADS=8

# 4 GPUs -> local ranks 0..3 map to physical GPUs 0,1,2,3.
export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS=4
MASTER_PORT="${MASTER_PORT:-29500}"   # change if you hit "address already in use"

RUN_DIR="$REPO/work_dirs/table_3_JiT/JiT_B-fd-inception"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$RUN_DIR/resume_${TS}.out"
PIDFILE="$RUN_DIR/resume.pid"

# Exactly the original launch (args.json: world_size=4, batch_size=128, compile=off,
# wandb=on), plus --auto_resume. --auto_resume picks the newest checkpoint
# (step_0089899.pth) and restores model/EMA/optimizer/step/FD-queue; --load_from is
# ignored while resuming. Global batch = 128 * 4 = 512, identical to before.
CMD=(torchrun
  --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port="$MASTER_PORT"
  --nproc_per_node="$GPUS"
  main_fd.py
  --project table_3_JiT --exp_name JiT_B-fd-inception
  --batch_size 128
  --data_path /data/dataset/imagenet
  --load_from ./checkpoints/base/JiT-B.pth
  --model JiT_B --rope_2d --learned_pe --legacy_time_convention
  --cfg 3.0 --interval_min 0.1 --interval_max 1.0
  --ema_type edm
  --num_sampling_steps 1
  --eval_bsz 256 --num_images_for_eval_and_search 50000
  --vis_freq 100 --online_eval --eval_freq 10000
  --print_freq 20 --milestone_interval 10 --save_freq 5
  --epochs 100 --steps_per_epoch 1250 --warmup_epochs 5
  --lr 1e-5 --lr_sched cosine --min_lr 0.0
  --fd_eigvalsh --fd_ema_beta 0.999
  --enable_wandb
  --auto_resume
  --fd_repr_models inception
)

echo "[resume] GPUs        : $CUDA_VISIBLE_DEVICES (global batch 512)"
echo "[resume] master_port : $MASTER_PORT"
echo "[resume] log         : $LOG"

# setsid -> new session (no controlling terminal); nohup -> ignore SIGHUP;
# </dev/null -> no terminal stdin. Training keeps running after you close SSH.
setsid nohup "${CMD[@]}" >"$LOG" 2>&1 </dev/null &
echo $! > "$PIDFILE"

echo "[resume] pid         : $(cat "$PIDFILE")  (saved to $PIDFILE)"
echo "[resume] monitor     : tail -f \"$LOG\""
echo "[resume] stop        : kill -INT \$(cat \"$PIDFILE\")"
