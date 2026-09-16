# BF16 generator, FD extractors, and Qwen LoRA-q training on Slurm

The active GRASP preset starts a **fresh 15,000-step training run** from
`JiT-B-uncond.pth`, using six RTX A6000 GPUs on `enough-oryx.grasp.maas`, global
batch 48, and sample grids every 1,500 steps. The standalone shared launcher
still defaults to a separate 3,000-step weight calibration with four GPUs and
global batch 96.

The script filename, job/log names, and experiment prefix retain `blackwell`
for continuity. That historical name does not describe the current GPU target.

Both configurations retain the same 100-class conditional FD/VLM objective.
The generator, frozen FD feature networks, Qwen backbone, p/q classifier
weights, and LoRA parameters use BF16. Trainable parameter gradients and AdamW
first/second moments also use BF16, with no FP32 master weights or FP32 parameter
updates. It passes `--dtype bf16 --parameter_dtype bf16 --vlm_dtype bf16`.

"Full BF16" describes the neural compute and parameter-update policy. FD
covariance statistics and eigendecomposition retain the original FP64 path;
log probabilities and image-VJP reductions retain FP32. Classifier feature
normalization follows the BF16 parameter dtype.
These numerical reductions do not maintain or update FP32 model parameters.

| Run setting | Active GRASP RTX A6000 preset | Standalone calibration default |
|---|---|---|
| Launcher | `sbatch_vlm_lora_q_bf16_blackwell.sh` | `run_vlm_lora_q_bf16_calibration.sh` |
| Mode | `CALIBRATION=0` | `CALIBRATION=1` |
| Steps | 15,000 (`EPOCHS=10`, `STEPS_PER_EPOCH=1500`) | 3,000 (`CAL_STEPS=3000`) |
| GPU processes / global batch / images per rank | 6 / 48 / 8 | 4 / 96 / 24 |
| Generator LR | `1e-5`, fixed default | `1e-5` at four ranks |
| Sample grids | Step 0, then every 1,500 steps through 15,000 | Disabled |
| Online FID evaluation | Disabled | Disabled |

| Shared setting | Value |
|---|---|
| Conditional weight | `3.5e-5`, ramped over 500 steps |
| Qwen microbatch | 2 images |
| VLM image-VJP loss scale | 1,024 |
| q cross-entropy loss scale | 1,024 |
| q source / update policy | Direct student; one update on each fresh batch |
| q head / LoRA LR | `1e-4` / `1e-5` |
| LoRA | rank 8, alpha 16, vision and language attention |
| Neural compute / parameter dtype | BF16 / BF16 |
| AdamW parameter moments | BF16, no FP32 master weights |
| FD moments / eigensolve | FP64 |
| Classifier feature normalization | BF16 |
| Log probabilities / image VJP accumulation | FP32 |

`3.5e-5` is an initial estimate from the earlier FP32 run's late gradient ratio
of about 0.072 at weight `1e-5`. It is **not a validated BF16 weight**. Read the
sustained ratio again alongside the new sample grids; 0.22-0.30 remains a
calibration target, not a guarantee of conditioning.

The active run retains the default name
`qwen_lora_fullbf16_blackwell_b48_15k_vis1500_JOB_ID`. Its six-rank batch-48
configuration changes training dynamics relative to the previous four-rank
batch-96 experiment. Generator LR stays explicitly at `1e-5`; it is not
automatically raised to `1.5e-5` when adding two ranks. q head/adapter learning
rates, weight `3.5e-5`, 500-step ramp, fixed scales of 1024, and microbatches of
2 are retained. Equivalent updates or faster wall-clock training are not
established by changing GPU count.

The standalone calibration keeps the `qwen_lora_fullbf16_scaled_cal_` default
name prefix. Use a fresh experiment name for either configuration.

## Relationship to the original FD-Loss paper

Tables B.2 and B.3 of the [FD-Loss paper](https://arxiv.org/html/2604.28190)
report BF16 training. They do not specify that all parameters, optimizer states,
or matrix calculations use BF16. The released code keeps
[FD EMA moments in FP64](https://github.com/Jiawei-Yang/FD-Loss/blob/main/frechet_distance/queue.py)
and uses those statistics in its
[FD eigensolve](https://github.com/Jiawei-Yang/FD-Loss/blob/main/frechet_distance/losses.py).
PyTorch's [eigvalsh](https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigvalsh.html)
does not support BF16 inputs.

There is also a difference between the paper's precision setting and the
released implementation: [model construction](https://github.com/Jiawei-Yang/FD-Loss/blob/main/utils/builders.py)
keeps default FP32 parameters, and the released
[training step](https://github.com/Jiawei-Yang/FD-Loss/blob/main/main_fd.py)
has no training autocast scope. BF16 autocast appears in the
[sampling helper](https://github.com/Jiawei-Yang/FD-Loss/blob/main/utils/sampling_util.py).
The preset here explicitly enables BF16 neural training and BF16 parameter
updates. It is not an exact reproduction of the released implementation, and
the conditional VLM/LoRA objective is an addition to the original method.

## What loss scaling does

The existing trainer already implements two independent fixed scales:

```text
VLM VJP:  grad_x(S * log_probability) / S
q update: backward(S * CE); parameter_grad /= S; all-reduce; clip; optimizer.step()
```

The p and q image gradients are unscaled and subtracted in FP32 before the
conditional weight is applied. q parameter gradients are accumulated, unscaled,
and reduced in their BF16 storage dtype. Scaling protects the magnitude of
intermediate backward values; it does not multiply the final update or replace
calibration of `WEIGHT`. Both scales default to 1024 here and can be overridden separately.
Nonfinite gradients trigger the trainer's existing error checks; these are
fixed scales, not an adaptive GradScaler.

BF16 has a much wider exponent range than FP16 and usually does not need scaling
for underflow. Scaling cannot recover precision lost in BF16 forward operations
or guarantee agreement with FP32 image-gradient directions. The repository's
earlier measurements found substantial direction differences, so evaluate this
as a new numerical configuration. See [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)
and [the precision notes](vlm_lora_q.md#precision-and-loss-scaling).

BF16 parameter storage also rounds small optimizer increments. At the small
generator/adapter learning rates used here, some increments can disappear when
added to a BF16 parameter. Loss scaling is undone before the optimizer and
does not recover those increments. This preset intentionally uses that update
policy; finite losses or gradients alone do not establish that every parameter
receives an effective update.
Generator EMA shadows also use BF16, so small EMA changes can round away too.
The preset reads the q student directly and does not maintain a q EMA.

## Prepare the destination server

Copy or sync this updated repository to a filesystem visible from the compute
node. Submit from that repository root. Slurm transfers the batch script itself,
not the Python modules, models, data, or other scripts; see
[sbatch documentation](https://slurm.schedmd.com/sbatch.html).

Use the working training environment from the earlier server when possible,
with CUDA PyTorch and the packages in `requirements.txt`. The Windows workstation's
Python environment is not the environment used by the job. Select the server
environment with `PY_BIN`; the default is `<repo>/.venv/bin/python`.

Required assets, **not included in the copied `work_dirs/vlm_delta` logs**:

- Deconditioned generator checkpoint `checkpoints/base/JiT-B-uncond.pth`.
- Real-trained Qwen answer-state `p_head.pt` for classes 0, 10, ..., 990.
- FD references `siglip_cls.npz`, `mae_cls.npz`, and `inception.npz` under
  `data/fid_stats/imagenet100_v1`.
- ImageNet with a `train/` split under `DATA_PATH`.
- The original Qwen model snapshot, including processor/tokenizer assets and
  weight shards; the recorded revision was
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Cached weights for the three FD models and the independent ResNet-50 probe.

The launcher defaults to `HF_HUB_OFFLINE=1`. Transfer/populate the model caches
before submitting, and set `HF_HOME` and `TORCH_HOME` if using nondefault cache
locations. A Hugging Face snapshot directory can contain symlinks into `blobs/`:
copy the complete cache, or dereference symlinks when transferring the snapshot.

The p head stores an absolute Qwen path from its original machine. If that path
differs on the destination, create a relocated copy **on the destination**:

```bash
cd /path/to/conditional_fd_loss
export PY_BIN=/path/to/environment/bin/python

"$PY_BIN" scripts/relocate_vlm_p_head.py \
  --input /path/to/copied/p_head.pt \
  --output /path/to/p_head_local.pt \
  --qwen-model /path/to/qwen/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
```

The helper preserves the original file and learned head content, changes only
the model path in a new file, checks local assets, and rejects known snapshot
revision mismatches. It does not verify weight bytes. Use the same model
revision, prompt, and layer as the original head. This new p-head identity is
for the fresh run; do not resume an old generator run against a relocated head.

## Preview and submit the active six-GPU run

Set destination paths once in the shell used to call `sbatch`:

```bash
cd /path/to/conditional_fd_loss
export PY_BIN=/path/to/environment/bin/python
# Parent directory: the launcher checks/appends train/ itself.
export DATA_PATH=/mnt/projects/jg/kaifany/dataset/imagenet
export START_CKPT=/path/to/JiT-B-uncond.pth
export STATS_DIR=/path/to/imagenet100_v1
export P_HEAD=/path/to/p_head_local.pt
export EXP_NAME=qwen_lora_fullbf16_blackwell_b48_15k_vis1500_v1
# If needed:
export HF_HOME=/path/to/huggingface/cache
export TORCH_HOME=/path/to/torch/cache

# Exercise the Slurm wrapper's defaults without allocating GPUs or writing files.
DRY_RUN=1 SLURM_JOB_ID=preview SLURM_SUBMIT_DIR="$PWD" \
  bash scripts/sbatch_vlm_lora_q_bf16_blackwell.sh

sbatch --export=ALL scripts/sbatch_vlm_lora_q_bf16_blackwell.sh
```

The wrapper targets `enough-oryx.grasp.maas` in partition `batch`, using account
`gu-account` and QOS `normal`, requesting
**six GPUs, 48 CPUs, 256 GB host memory, and 24 hours**. It uses
`--gres=gpu:6` on this RTX A6000 node. GPU availability and queue wait still
depend on other jobs. Its default `DATA_PATH` is
`/mnt/projects/jg/kaifany/dataset/imagenet`, so the training split path is
`/mnt/projects/jg/kaifany/dataset/imagenet/train`. An exported `DATA_PATH`
overrides that default; use the parent directory, not the `train` directory.

Copy the updated repository to the server before the next submission. Editing
these local files does not change the allocation or settings of an already
submitted or running Slurm job.

To preview the same training settings directly through the shared launcher:

```bash
NPROC_PER_NODE=6 GLOBAL_BATCH=48 CALIBRATION=0 \
EPOCHS=10 STEPS_PER_EPOCH=1500 VIS_FREQ=1 DISABLE_VIS=0 ONLINE_EVAL=0 LR=1e-5 \
  bash scripts/run_vlm_lora_q_bf16_calibration.sh --dry-run
```

The preview should show six ranks, batch 8 per rank, global batch 48, 15,000
iterations, `--epochs 10 --steps_per_epoch 1500 --vis_freq 1`, and `--lr 1e-5`.
It must omit `--disable_vis` and `--online_eval`. Visualization frequency is
expressed in launcher epochs: `1 * 1500` steps. The independent per-class
diagnostics and conditional probe remain enabled.

Use a CUDA PyTorch environment compatible with RTX A6000 and this repository's
required packages. The retained `blackwell` script name
does not impose a Blackwell-specific CUDA requirement on this allocation.
The existing allocation check runs BF16 backward arithmetic and NCCL before
loading the training models. The retained Slurm log name is
`slurm-vlm-blackwell-JOB_ID.out`. Full-model memory use and throughput on this
six-GPU configuration still need measurement.

The launcher preserves Slurm's `CUDA_VISIBLE_DEVICES` mapping, including UUIDs
and physical IDs. Do not replace it with `0,1,2,3` in the batch script. A startup
check tests native BF16 backward arithmetic and NCCL collectives on each rank.
It checks exactly the requested process count, but does not prove full-model
memory fit or gradient fidelity. See [Slurm GPU management](https://slurm.schedmd.com/gres.html#GPU_Management).

The preview requires no CUDA, Python environment, data, or checkpoints and
does not create output files. Actual training validates the assets and p-head
identity before filling the FD queues. Shell launch scripts are stored with LF
line endings via `.gitattributes` for Slurm compatibility.

The preview must include `--dtype bf16`, `--parameter_dtype bf16`, and
`--vlm_dtype bf16`. Its precision summary reports BF16 neural parameters and
AdamW moments, FP64 FD statistics/eigensolve, and FP32 VJP/log-probability
reductions. `--dtype` controls generator/FD neural autocast;
`--parameter_dtype` selects parameter storage and update dtype; `--vlm_dtype`
selects the frozen Qwen backbone dtype.

## Separate 3,000-step calibration

Calling the shared launcher without the node-specific wrapper retains its original
four-rank, global-batch-96 calibration defaults. This mode disables sample
grids and online evaluation and uses `CAL_STEPS=3000`:

```bash
EXP_NAME=qwen_fullbf16_scaled_cal_v1 \
  bash scripts/run_vlm_lora_q_bf16_calibration.sh --dry-run

EXP_NAME=qwen_fullbf16_scaled_cal_v1 \
  sbatch --export=ALL --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  scripts/sbatch_vlm_lora_q_bf16.sh
```

The generic Slurm wrapper requests four GPUs, 32 CPUs, 256 GB host memory, and
24 hours. Select the account/partition and GPU type for that destination. The
node-specific wrapper deliberately overrides this shared launcher's run mode,
batch, duration, visualization settings, and default generator LR.

## Follow the run and inspect sample grids

```bash
squeue -u "$USER"
tail -f slurm-vlm-blackwell-JOB_ID.out
```

With the example `EXP_NAME`, output paths are:

```text
work_dirs/JiT_uncond_vlm_delta/qwen_lora_fullbf16_blackwell_b48_15k_vis1500_v1/training_metrics.json
work_dirs/JiT_uncond_vlm_delta/qwen_lora_fullbf16_blackwell_b48_15k_vis1500_v1/checkpoints/
work_dirs/JiT_uncond_vlm_delta/qwen_lora_fullbf16_blackwell_b48_15k_vis1500_v1/visualization/
sweep_logs/qwen_lora_fullbf16_blackwell_b48_15k_vis1500_v1.out
sweep_logs/qwen_lora_fullbf16_blackwell_b48_15k_vis1500_v1.allocation.json
sweep_logs/qwen_lora_fullbf16_blackwell_b48_15k_vis1500_v1.training.txt
```

Grids are saved at initialization (step 0 for a fresh run), then steps 1,500,
3,000, ..., 15,000. The existing grid format is retained: 20 class columns
for ImageNet labels `0, 50, 100, ..., 950`, generator EMA variants plus online
weights, and both shared-noise and independent-noise modes. Each rank supplies
one row, so the six-GPU run has six rows per grid. RNG resets make the sampled
noise comparable across snapshots; "independent noise" means across images,
not a new unrelated seed at every checkpoint. Filenames include the step,
CFG, EMA label, sampling-step count, and noise mode.

These sample sheets do not run the 50,000-image online FID evaluation. The
training budget is 720,000 generated images, averaging 7,200 per class; images
generated for FD warm-start and visualization are additional.

The shared launcher writes a training report after the 15,000-step run.
Standalone calibration writes a `.calibration.txt` report instead. To regenerate
the active training report:

```bash
"$PY_BIN" scripts/analyze_vlm_delta_run.py --weight 3.5e-5 \
  work_dirs/JiT_uncond_vlm_delta/qwen_lora_fullbf16_blackwell_b48_15k_vis1500_v1
```

Use the actual weight if overridden. Read the late-window gradient ratio,
per-class p/q accuracy, independent probe, q pre-update CE, and nonfinite/spike
diagnostics together. A finite scaled backward does not establish correct
semantic gradients. The instant calibration meters now keep their window size
after a resumed step; this fixes the smoothing issue in the earlier run.

## Overrides and restart

All overrides are environment variables exported before `sbatch`. For example,
an independent calibration comparison can use the generic wrapper:

```bash
# Compare against the old weight, with smaller scales or a different microbatch:
export WEIGHT=1e-5 VLM_VJP_LOSS_SCALE=128 VLM_Q_LOSS_SCALE=128
export VLM_MICROBATCH=4
export EXP_NAME=qwen_fullbf16_cal_scale128_v1
sbatch --export=ALL --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION \
  scripts/sbatch_vlm_lora_q_bf16.sh
```

Raise `VLM_MICROBATCH` only after checking memory use; set it to 1 if necessary.
It changes activation memory and throughput, not global batch size. For a short
pipeline check with the standalone calibration, set
`CALIBRATION=1 CAL_STEPS=50 INIT_DIAG_SAMPLES=8` and a unique `EXP_NAME`.
It still fills the normal FD queue and is too short to calibrate the final weight.
For the active training mode (`CALIBRATION=0`), duration is controlled by
`EPOCHS * STEPS_PER_EPOCH`; `CAL_STEPS` does not change its 15,000-step budget.

When changing GPU count, set both `NPROC_PER_NODE` and Slurm's `--gres` to
match. `GLOBAL_BATCH` must divide evenly across ranks. The standalone launcher's
historical LR convention applies only when `LR` is unset; the node-specific wrapper
explicitly defaults to `LR=1e-5`. Changing rank count or global batch still
requires fresh timing and calibration measurements.

Automatic Slurm requeue resumes the latest checkpoint under the same experiment.
If preempted before any checkpoint, the incomplete attempt is renamed and a
fresh attempt starts. To manually resume a **BF16 run with the same settings**:

```bash
export EXP_NAME=qwen_lora_fullbf16_blackwell_b48_15k_vis1500_v1 RESUME=1
sbatch --export=ALL scripts/sbatch_vlm_lora_q_bf16_blackwell.sh
```

Use the same launcher and experiment name as the original run. A standalone
calibration resumes through the generic wrapper instead of the node-specific
training wrapper.

Start the new BF16 experiment with `RESUME=0` and a new name. Both the old FP32
run and the Qwen-only BF16 preset used FP32 trainable parameters; their
checkpoints have a different precision identity and are not full-BF16 resumes.
Do not change the model path, dtype, optimizer settings, scales, or weight
mid-run if the goal is an interpretable single-configuration trajectory.

## Verification

On the server's training environment:

```bash
python -m unittest tests.test_vlm_lora_q tests.test_vlm_delta \
  tests.test_relocate_vlm_p_head
```

Local validation covers shell syntax and dry-run argument routing, CPU tests
for scaled/unscaled first derivatives and q updates, checkpoint relocation,
and the resume-meter regression. No full Qwen GPU calibration was run on the
Windows workstation; measure actual memory, speed, and gradient behavior on
the allocated server.
