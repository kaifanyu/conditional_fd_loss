# Learnable Qwen q with LoRA

For the current BF16 calibration preset with BF16 generator/FD neural compute,
BF16 trainable parameters and AdamW moments, fixed VJP/q loss scaling, and a
portable Slurm launcher, see [BF16 calibration on Slurm](vlm_bf16_calibration.md).
That preset has no FP32 master weights or FP32 parameter updates. FD matrix
statistics/eigensolve remain FP64, and log-probability/image-VJP reductions
remain FP32. Classifier feature normalization follows the BF16 parameter dtype.

The existing entry point `conditional_main_fd_vlm_delta.py` now accepts
`--vlm_q_lora`. This mode supports p checkpoints with
`vlm_backend=qwen_answer_state`. The default, without the flag, retains the
frozen extractor and linear-head feature replay experiment. A timm/SigLIP
p checkpoint is rejected in LoRA mode; the FD judges must remain fixed.

**Current training defaults (2026-09-09):** the generator uses the current q
student directly, without q EMA. Both q optimizer groups use AdamW with
betas=(0.0, 0.999); head LR=1e-4 and adapter LR=1e-5. One q update is applied
per generator step. `--vlm_q_updates_per_step 5` enables five optimizer steps
later; in LoRA mode these reuse the same fresh scored images and recompute
features after every optimizer step. `--vlm_q_use_ema` explicitly enables the
earlier EMA variant. Generator-weight EMA and FD-statistics EMA are separate.

The implementation uses native PyTorch LoRA in `vlm_lora_q.py`. It does not
require PEFT or load three copies of Qwen. Its adapter checkpoint format is
specific to this repository, not a PEFT `adapter_model.safetensors` file.

## What is learned

Write the frozen backbone as E0, the pretrained real-data head as Hp, the
student adapters as As/Bs, and the teacher adapters as At/Bt. For each targeted
linear projection:

```
base:     y = W0 x + b0
student:  y = W0 x + b0 + (alpha / rank) Bs As x
teacher:  y = W0 x + b0 + (alpha / rank) Bt At x
```

W0 and b0 never receive optimizer updates. Each branch selects its own adapter
bank; the base branch bypasses adapters. No weights are merged into W0.

| Component | Student CE update | Generator update |
|---|---|---|
| Qwen base weights, normalization, embeddings | Frozen | Frozen; differentiate with respect to images |
| Real-data p head | Frozen | Frozen; image derivatives retained |
| Student LoRA A/B and q linear head | Trainable | Current parameters detached; image derivatives retained |
| Optional teacher LoRA A/B and q linear head | EMA only with `--vlm_q_use_ema` | Used only with `--vlm_q_use_ema` |
| Generator | Detached images; no gradient | Trainable |

The three distributions are:

```
p(c | x)  = softmax(Hp(normalize(E0(x))) / T)[c]
qs(c | x) = softmax(Hs(normalize(Es(x))) / T)[c]
qt(c | x) = softmax(Ht(normalize(Et(x))) / T)[c]
```

All use the p checkpoint's class ordering, input preprocessing, prompt, selected
answer layer, feature normalization statistics, and shared temperature. The
normalization statistics stay frozen even as adapted features change. Feeding
adapted features to Hp would change p despite freezing its head; this
implementation always computes p from the unadapted branch.

After adaptation there is no single shared feature z. The ratio is interpreted
as `log qs(c | x) - log p(c | x)` by default (`qt` only with EMA enabled), with each classifier computing its own
features of the same image. This extends q's function class and changes the
shared-representation assumption of the earlier experiment.

## Initialization and EMA

Copy the entire p head to Hs and Ht, including weight, bias, feature mean/std,
and temperature. Initialize As with Kaiming uniform, Bs with zero, and copy both
to At/Bt. This makes both adapted functions equal E0 initially. Initializing
both A and B to zero would give zero gradients to both factors and prevent
learning. This is the conventional no-op LoRA initialization described in the
[Hugging Face LoRA documentation](https://huggingface.co/docs/peft/package_reference/lora).

With deterministic forwards and identical settings, initially:

```
qs = qt = p
log qt(c | x) - log p(c | x) = 0
d/dx [log qt(c | x) - log p(c | x)] = 0
```

The new CPU tests assert zero output difference and zero image gradient at
initialization, including zero output difference on a tiny real Transformers
Qwen model. Full GPU kernels can introduce roundoff; the existing initial
diagnostic checks max absolute log-probability difference with tolerance 1e-4.

With `--vlm_q_use_ema`, after each outer step's student updates, EMA both the
head and adapter factors. Without that flag, the teacher weights are inactive
and receive no EMA updates; the generator reads the latest student directly:

```
Ht <- beta Ht + (1 - beta) Hs
At <- beta At + (1 - beta) As
Bt <- beta Bt + (1 - beta) Bs
```

This is EMA of the trainable parameters. In general `EMA(B) EMA(A)` does not
equal `EMA(BA)`, and neither is an exact EMA of predictions. The teacher stays
rank-r and has the same architecture as the student. With beta=0.999 and one
EMA update per generator step, the approximate averaging horizon is 1000 steps.

## Training data and gradients

The old replay buffer stores detached features. Such a buffer cannot train a
backbone: it contains no image-to-feature graph, and the features become stale
as adapters move. LoRA mode therefore trains on the **fresh scored generated
images** from the current step. Images are detached from the generator and
features are recomputed under the student adapter bank with autograd enabled.
There is no image replay allocation and no uint8 image quantization.

`--vlm_q_batch_size`, `--vlm_q_buffer_size`, `--vlm_q_bootstrap`,
`--vlm_q_recompute_features`, and buffer checkpoint options do not control this
mode. `--vlm_q_bootstrap_updates` must be zero so q still equals p before the
first generator update. Set the scored batch through `--batch_size` and
`--vlm_samples_per_step`; microbatching controls memory, not the effective batch.
Default `--vlm_q_updates_per_step 1` uses each selected image once per step.
Multiple updates reuse that same fresh batch and can overfit it.

This is a data-policy change as well as a capacity change. To attribute an
improvement specifically to adapters, also compare against a head-only run
trained with a matched fresh-image data policy. The default head-only replay
run is not a pure capacity ablation.

Student loss is the mean sampled-label cross entropy. Every rank trains on its
own equally sized image shard, accumulates microbatch gradients using the full
local batch denominator, then averages parameter gradients across ranks in
their storage dtype (`--parameter_dtype`, FP32 by default; BF16 in the preset).
The random adapter initialization is broadcast from rank zero; the existing
rank-dependent seeds otherwise produce different A matrices. Head and adapter
optimizer groups have separate learning rates and weight decay.

Generator loss retains the sampled-label objective and existing FD behavior:

```
LG = LFD + lambda(step) mean[log qs(c | G(eps,c)) - log p(c | G(eps,c))]
```

The two VLM branches run sequentially in microbatches. Each branch immediately
computes its image gradient and releases its activation graph. Their gradients
are subtracted in FP32 and injected into the original generated images with:

```
surrogate = stopgrad(value) + sum((x - stopgrad(x)) * stopgrad(image_gradient))
```

This preserves the first derivative, not higher derivatives or differentiation
through the q optimizer. The generator step uses the current student, with
the head weights and both adapter factors detached in this forward path.
The image/features remain differentiable. Only after its backward/optimizer
step do student CE updates run. The next generator step sees those latest
student parameters. The optional EMA variant reads/updates the teacher instead.
If the optional delta clamp is enabled, its per-image derivative mask is
applied to the image gradients. Qwen processes images independently, so this
per-image mask agrees with differentiating the clamped sum directly.

The original trainer's global denominator and rank-averaged generator gradients
are retained. As its launcher already reports, this produces an effective
generator learning rate proportional to 1/world_size for these global losses.
Keep world size fixed for comparisons. Student CE uses a local mean followed
by rank averaging and therefore has the usual global-mean update.

## Starting configuration

The implementation defaults are rank=8, alpha=16, adapter LR=1e-5, adapter
weight decay=0.01, and no adapter dropout. Qwen stays in eval mode for all
branches; eval mode disables stochastic layers but permits parameter autograd.
The head LR/decay defaults are 1e-4/3.0. Both head and adapter groups use
AdamW with beta1=0.0 and beta2=0.999, configurable through `--vlm_q_beta1` and
`--vlm_q_beta2`. There is no q parameter EMA by default.
These are starting settings, not measured optima.

By default `--vlm_q_lora_scope both` targets:

* language attention `q_proj`, `k_proj`, `v_proj`, `o_proj`, only in decoder
  blocks contributing to the selected answer state;
* vision attention `qkv`, `proj` throughout the visual tower.

`--vlm_q_lora_scope language` or `vision` isolates a component. Targets can be
changed with `--vlm_q_lora_targets`. Only matching Linear modules in eligible
vision/decoder blocks are wrapped; convolutional patch embedding and LM output
head are not adapted. Default targets do not include MLPs or the visual merger.

Use the existing 100-class launcher with a **Qwen** p checkpoint. The current
BF16 preset is `scripts/run_vlm_lora_q_bf16_calibration.sh`. For an independent
FP32 reference smoke run, replace the paths/GPU selection in this example:

```bash
cd /mnt/projects/jg/kaifany/conditional_fd_loss
P_HEAD=/absolute/path/to/qwen_answer_state/p_head.pt \
Q_LORA=1 Q_USE_EMA=0 GPUS=0,1 Q_LR=1e-4 Q_BETA1=0.0 Q_BETA2=0.999 \
VLM_DTYPE=fp32 VLM_MICROBATCH=1 \
Q_BOOTSTRAP_UPDATES=0 \
CALIBRATION=1 CAL_STEPS=100 WEIGHT=0.01 \
INIT_DIAG_SAMPLES=32 PRINT_FREQ=10 \
EXP_NAME=qwen_lora_fp32_smoke RUN_FOREGROUND=1 \
EXTRA_ARGS="--vlm_q_lora_rank 8 --vlm_q_lora_alpha 16 --vlm_q_lora_lr 1e-5 --vlm_attn_implementation eager" \
bash scripts/run_jit_uncond_vlm_delta_100class.sh
```

Set `PY_BIN`, `DATA_PATH`, `START_CKPT`, and `STATS_DIR` if the launcher's
existing environment defaults do not match your machine. Q_LORA=1 defaults
to fp32 and microbatch=1, enables process-wide TF32 disabling, and turns off
feature-buffer bootstrap. `WEIGHT=0.01` above is an illustrative smoke weight;
100 steps checks operation and memory, not convergence or a calibrated weight.

### GRASP queue preset

`scripts/sbatch_vlm_lora_q_grasp.sh` requests one node with four nominal 48 GB
GPUs (L40/L40S/A40/A6000), 32 CPUs, 256 GB host RAM and a two-day limit on
`batch`, account `gu-account`, QOS `normal`. This is data parallel training:
each GPU holds Qwen and the judges; GPU memories are not pooled. The free
24 GB cards cannot hold this FP32 configuration.

The preset starts a 3,000-step **calibration training run** at weight 1e-5,
global batch 96, with all the direct-q defaults above, rank 8/alpha 16, one
q update, FP32 throughout, TF32 disabled and both loss scales set to 1.
This is a conservative calibration seed, not a measured long-run weight. The
late gradient-ratio window is analyzed after training before selecting a
weight for a longer experiment. No 50,000-step run is submitted automatically.

To accommodate the shared node's per-GPU memory, it enables generator
microbatch 2 with activation checkpointing
(`--generator_microbatch 2 --grad_checkpointing`), FD feature microbatch 2
with non-reentrant checkpointing (`--fd_feature_microbatch 2
--fd_feature_checkpoint`), Qwen microbatch 1 and queue-fill batch 8. The
effective FD/generated/scored training batch and objective remain unchanged.
These FD memory controls default off for other launchers. CPU tests compare
full-batch values, image VJPs and generator parameter gradients, including
the diagnostic's repeated backward and an uneven final microbatch. Actual
peak VRAM still needs to be measured on the allocated GPUs.

Job 544545 passed allocation, model loading, FD queue filling and q=p
initialization, but exhausted the L40S's 44.53 GiB during the first generator
backward. Checkpointing the entire 24-image generator forward still recreated
the full activation graph at backward time. The corrected preset checkpoints
each two-image generator chunk separately, concatenates all generated images
before computing the full FD/conditional loss, and accumulates parameter
gradients into the same single generator update. CPU regression tests verify
global-loss image/parameter gradients and bound every backward recomputation
to the requested microbatch size.

GPU smoke job 547194 completed three full training steps and saved a checkpoint
on an A40 with peak reserved memory 40.68 GiB (44.53 GiB usable). It exercised
the same 24-image per-rank batch, all models, both VLM image VJPs, q LoRA/head
updates, generator updates and diagnostics. This one-GPU smoke initialized FD
EMA statistics from 4,096 images and disabled the conditional ramp; it does
not validate the four-rank training runtime or convergence. At weight 0.01
its conditional/FD image-gradient ratio reached 59.2 then 113.8 after the first
q updates. That motivates reducing the **calibration seed** to 1e-5, retaining
the 500-step ramp and measuring the later sustained ratio before choosing a
long-run weight. The three-step smoke does not provide that final calibration.

Before loading the training models, `scripts/check_vlm_lora_allocation.py`
checks four visible GPUs with at least 44,000 MiB each, CUDA backward and
NCCL collectives; the allocation report is saved in `sweep_logs`. The trainer
then checks q=p before its first update and aborts on non-finite gradients.
All model files must already be cached; the job runs with HF offline mode.

The ten-minute checkpoint target is now enforced from the start of training,
at completed step boundaries, with a rank-zero decision broadcast to all
workers. It no longer depends on reaching the first 1,000-step timing
estimate. Slurm requeues resume from saved training state; if preempted
before any checkpoint exists, the launch preserves the incomplete attempt
and restarts from the original generator and p initialization. Zero-grace
preemption can still lose work since the last completed checkpoint.
The inherited ramp is 500 steps, so this short smoke does not establish the
mature conditional force. Extend the pilot past the ramp before using
`grad_ratio_vlm_fd` to choose lambda; there is no teacher lag in direct-q mode. Retain the
independent probe and visual inspection to judge actual conditioning.

The equivalent new entry-point arguments are:

```text
--vlm_q_lora
--vlm_q_lora_rank 8 --vlm_q_lora_alpha 16 --vlm_q_lora_scope both
--vlm_q_lora_lr 1e-5 --vlm_q_lora_weight_decay 0.01
--vlm_q_lr 1e-4 --vlm_q_weight_decay 3.0 --vlm_q_grad_clip 1.0
--vlm_q_optimizer adamw --vlm_q_beta1 0.0 --vlm_q_beta2 0.999
--vlm_q_updates_per_step 1
--vlm_q_bootstrap_updates 0
--vlm_dtype fp32 --vlm_disable_tf32 --vlm_microbatch 1
--vlm_vjp_loss_scale 1 --vlm_q_loss_scale 1
```

Only one base model is resident, but LoRA does not remove its activation
backward cost. A nonzero-weight step adds two Qwen image VJPs plus one student
CE backward, with extra forward passes on diagnostic steps. Microbatch=1 is
the starting point for measuring memory; no full 7B performance estimate has
been validated for this change. This implementation does not enable VLM
activation checkpointing or quantized base weights.

Checkpoints save both adapter banks/configuration, both q heads, optimizer
state, and the scored-image cursor. The immutable base model is reloaded from
the p checkpoint's model path. Resume rejects a mismatched LoRA configuration,
including dtype, and rejects switching between head-only and LoRA q modes.
Resume also checks the q EMA mode and optimizer learning rates, weight decay,
betas and momentum, preventing old settings from silently replacing new ones.
Old checkpoints without an EMA-mode field are interpreted as EMA runs.
For a new LoRA experiment from an older generator, use a new experiment name
and `--load_from`; do not resume the old q optimizer state.

## Precision and loss scaling

Feature cosine 0.9999 says two feature vectors point in almost the same
direction. It does **not** imply that their Jacobians with respect to pixels
are similar. It also does not prove the normalized head logits are unchanged:
feature standardization and the head can amplify small feature differences.

Your gradient cosines are evidence that the reduced-precision gradients are
poor matches to that fp32 reference. They do not isolate the cause. Potential
contributors include rounded weights, activations/backward intermediates,
fp16 underflow or overflow, and subtracting two nearly equal gradients when
q is close to p. Measure gradient norms as well: cosine of an almost-zero
delta gradient is not a reliable diagnostic. Identical prompts/layers, heads,
images, batch shapes, attention implementation and TF32 settings are essential.

Loss scaling helps when small backward intermediates underflow. It cannot
recover mantissa bits lost in the forward pass or undo changes to the model
caused by casting its weights. BF16 has FP32-like exponent range but fewer
significand bits, so increasing scale is generally less promising for BF16
rounding than for FP16 underflow. FP16 may also overflow, especially for
BF16-pretrained models; scaling upward can worsen that. See the
[PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)
for gradient scaling and the FP16 range limitation.

For the image VJP, the implemented experiment is:

```
g_scaled = autograd.grad(S * branch_loss, detached_fp32_image)
g_image  = g_scaled.float() / S
# Subtract q and p image gradients in FP32, then multiply by lambda.
```

`--vlm_vjp_loss_scale S` works for both the original Qwen head-only path and
the LoRA path. Scaling the final generator loss would be too late to repair
the VLM backward: its VJP has already been computed and detached.

For student updates, `--vlm_q_loss_scale S` in LoRA mode scales CE before
backward, then divides every parameter gradient by S before all-reduce,
clipping, and the optimizer. Both scale defaults are 1. These are fixed scales
for controlled experiments; non-finite gradients abort rather than silently
changing the scale. This is not a dynamic GradScaler implementation.
Parameter gradients retain the configured storage dtype, including BF16 for
the full-BF16 preset.

Changing `--vlm_delta_weight` without unscaling changes the objective's
strength. It is a separate intervention. Loss magnitude alone does not measure
gradient fidelity. In particular, a small `log q - log p` can simply reflect
successful p initialization or cancellation, not underflow.

The precision controls are independent:

| Argument | What it controls | Full-BF16 preset |
|---|---|---|
| `--dtype` | Generator and FD feature-network neural autocast | `bf16` |
| `--parameter_dtype` | Generator, FD-network, p/q-head, and LoRA parameter storage; trainable parameter gradients and AdamW moments | `bf16` |
| `--vlm_dtype` | Frozen Qwen base weights and compute | `bf16` |

`--parameter_dtype` defaults to `fp32` for other configurations. Selecting
`bf16` removes FP32 parameter updates and FP32 master weights; any enabled
parameter EMA uses the selected parameter dtype. It does not change FD moments
and eigensolve (FP64), or log probabilities, image leaves and VJP accumulation
(FP32). Classifier feature normalization follows the parameter dtype. These
higher-precision reductions are numerical computations rather than FP32
parameter updates. Native normalization/attention kernels can also internally
accumulate in higher precision.

BF16 parameter storage can round away optimizer increments that are small
relative to the existing weight. Loss scaling is removed before the optimizer,
so it does not correct this effect. Check actual parameter drift as well as
finite gradients and losses when evaluating this configuration.

The [original FD-Loss paper](https://arxiv.org/html/2604.28190) reports BF16
precision, but its released code keeps FP64 FD statistics and default FP32
model parameters, and lacks training autocast in `main_fd.py`. This BF16
parameter-update configuration is therefore an explicit extension, not an
exact reproduction; see [the source comparison](vlm_bf16_calibration.md#relationship-to-the-original-fd-loss-paper).

For a precision reference comparison, load the base checkpoint directly in
FP32 and disable TF32; casting an already-rounded BF16 model to FP32 does not
restore original weight information. FP32 adapters alone cannot make a BF16
base Jacobian equivalent to FP32.

Use `scripts/check_vlm_gradient_precision.py` on saved representative images:

```python
# Once, at the point where you have generated images in [0,1] and global labels:
torch.save({"images": images.detach().float().cpu(),
            "labels": labels.detach().long().cpu()}, "/tmp/vlm_images.pt")
```

```bash
.venv/bin/python scripts/check_vlm_gradient_precision.py \
  --p_head /absolute/path/to/qwen_answer_state/p_head.pt \
  --images_pt /tmp/vlm_images.pt --microbatch 1 \
  --dtypes fp32 bf16 fp16 --scales 1 128 1024 8192 \
  --attn_implementation eager --output /tmp/vlm_precision.json
```

The script loads one VLM dtype at a time and fixes the same p head plus a small,
seeded q-head perturbation to provide a nonzero delta. It reports per-image
cosines, norms, relative error, finite status and zero fractions for features,
log probabilities and individual p/q/delta image gradients. It diagnoses
numerics with a synthetic q head; it is not validation of a learned adapter
checkpoint. If scaling fails to improve agreement, treat fp32 as the baseline
and investigate precision/kernels/cancellation rather than raising lambda.

## Pseudocode

```python
E0 = load_qwen(dtype=vlm_dtype).eval().freeze_base_weights()
Hp, normalization, T, class_map = load_real_data_p_checkpoint()
Hp.to(dtype=parameter_dtype)            # includes feature normalization buffers
Hp.freeze()
Hs = deepcopy(Hp)
Hs.enable_grad()
As = kaiming_init(rank=8, dtype=parameter_dtype)
Bs = zeros(dtype=parameter_dtype)
broadcast_adapter_initialization_from_rank_zero()
q_optimizer = AdamW([
    {"params": Hs.parameters(), "lr": head_lr, "weight_decay": head_decay},
    {"params": [As, Bs], "lr": adapter_lr, "weight_decay": adapter_decay},
], betas=(0.0, 0.999))

for step in training_steps:
    c, noise = sample_labels_and_noise()
    x = G(noise, c)                      # keep generator graph
    chosen = round_robin_scored_subset(x)
    fd_loss = fixed_fd_judges(x)

    # Current student values; detach parameter tensors, not image inputs.
    for xb, cb in microbatches(x[chosen], c[chosen]):
        xp = stopgrad(xb).float().requires_grad_()
        lp = logprob(Hp(E0(xp)), cb, normalization, T)
        gp = grad(S_vjp * sum(lp) / global_scored_count, xp) / S_vjp
        release_p_graph()

        xq = stopgrad(xb).float().requires_grad_()
        zq = E0_with_adapters(xq, stopgrad(As), stopgrad(Bs))
        lq = logprob(linear_with_detached_parameters(Hs, zq), cb, normalization, T)
        gq = grad(S_vjp * sum(lq) / global_scored_count, xq) / S_vjp
        release_q_graph()

        d = stopgrad(lq - lp)
        gx = stopgrad(gq.float() - gp.float())
        if clamp_enabled:
            gx *= per_image_mask(abs(d) <= clamp)
            d = clip(d, -clamp, clamp)
        conditional += sum(d) / global_scored_count + sum((xb - stopgrad(xb)) * gx)

    backward(fd_loss + lambda_schedule(step) * conditional)
    average_generator_gradients_across_ranks()
    clip_and_step_generator()

    for _ in range(q_updates_per_generator_step):
        q_optimizer.zero_grad()
        for xb, cb in microbatches(stopgrad(x[chosen]), c[chosen]):
            zs = E0_with_adapters(xb, As, Bs)  # autograd ON for adapters
            ce = sum_cross_entropy(Hs(normalize(zs)) / T, cb) / local_scored_count
            backward(S_q * ce)
        divide_q_parameter_gradients_by(S_q)
        average_q_parameter_gradients_across_ranks()
        check_finite_and_clip_q_gradients()
        q_optimizer.step()
    # No q EMA: the next generator step reads the updated Hs, As and Bs.
```

## What this experiment establishes

Additional capacity may improve how well q estimates the current generator's
label posterior. It can also increase memorization and drift. If the generator
is independent of its sampled label, the population-optimal q is simply the
label prior (uniform for this sampler), regardless of q's capacity. Frozen p
can still have exploitable gradients. Seeing the same failure with GMM and VLM
does not by itself prove learnable q is the unique cause or that LoRA will
prevent exploitation.

Track fresh-batch student CE **before** training on that batch, active generator
q CE/entropy (`vlm_q_generator_*`), effective adapter drift (`q_student_lora_delta_l2`,
plus `q_teacher_lora_delta_l2` only with EMA), image-gradient norms/ratios, held-out probe accuracy,
and actual generated images. The old `q_generalization_gap` replay metric is
not emitted for LoRA because there is no replay buffer. Head drift meters only
measure the head; the new adapter meters measure the effective low-rank weight
updates. Better training CE alone is insufficient evidence of better q.

Validation: CPU adapter tests cover p isolation, q=p and zero gradient at init,
clamped/subsampled VJP equivalence, fresh-image gradient isolation, scaling and
uneven microbatch equivalence, EMA, checkpoint continuation, and forward/backward
with a tiny real Qwen from the installed Transformers library. A two-rank Gloo
test also verifies initialization synchronization and equivalence to a single
global-batch update. All 14 LoRA tests passed, including the direct-student
image VJP, absence of q parameter gradients during generator backward,
disabled EMA updates, five q optimizer steps, AdamW betas, and resume checks;
the existing head/answer-state
suites ran 80 tests with 16 GPU/model-dependent skips and no failures. Run the
new tests with `GLOO_SOCKET_IFNAME=lo .venv/bin/python -m unittest
tests.test_vlm_lora_q -v` in an environment allowing local sockets. Full 7B CUDA
training and the precision sweep require a GPU and cached model and were not
run in this workspace.
