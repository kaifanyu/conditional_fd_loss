# The VLM sampled-label `log q(c|z) - log p(c|z)` conditional term

**Status: implemented and validated 2026-08-28; one trial, at 100 classes.**

For the opt-in Qwen LoRA q extension, fresh-image updates, precision controls,
and pseudocode, see [Learnable Qwen q with LoRA](vlm_lora_q.md).

**2026-09-09 update:** the entry point and launcher now default to direct student
q, head LR=1e-4, and AdamW betas=(0.0, 0.999), with one q update per generator
step. The EMA trial described below is available through `--vlm_q_use_ema`
(`Q_USE_EMA=1` in the launcher); it is no longer the training default.

A single new experiment. It does not touch, re-run or reinterpret the fitted-GMM
work in [`docs/gmm.md`](gmm.md) — that code path is unchanged and its tests still
pass.

---

## 1. The objective, exactly

$$
L \;=\; L_{\mathrm{FD}} \;+\; w(s)\,
\mathbb{E}\!\left[\log q_{\psi_{\text{teacher}}}(c\mid z) - \log p_\phi(c\mid z)\right],
\qquad z = E_{\mathrm{VLM}}\!\left(G_\theta(\epsilon, c)\right)
$$

with `c` the **sampled** conditioning label, one frozen VLM, and two linear heads
on the same feature:

```
                            frozen p head  ->  p(c|z)      trained once on REAL images
                           /
image -> frozen VLM -> z
                           \
                            slow q teacher ->  q(c|z)      EMA of a student trained
                                                           online on DETACHED generated
                                                           (image, sampled label) pairs
```

Descent on this term moves samples along

$$
\nabla_x\!\left[\log p(c\mid z) - \log q(c\mid z)\right]
$$

i.e. **toward** what the real-data classifier calls class `c` and **away from**
where the generator's own current classifier already puts class `c`. The first
half is class fidelity; the second is the anti-mode-seeking counterweight, which
is why this is not simply `-log p(c|z)` guidance.

### What it is deliberately not

| | why it is excluded |
|---|---|
| class-summed posterior KL `E_z Σ_c q_c (log q_c − log p_c)` | that is `--fd_gmm_mode posterior` in `conditional_main_fd_gmm.py`. It is **label-blind** and died at its takeoff gate on 2026-08-27 (probe_top1 0.000 at 36k steps, eval FID 57.4). The whole point here is the *sampled* target-class field |
| a second explicit `-log p(c|z)` driver | would emphasise class fidelity twice — `∇[log q − log p]` already contains `−∇log p(c|z)` |
| `KL(q(·|z) ‖ p(·|z))` | a non-negative surrogate, not the sampled-label scalar |
| self-normalising the term (`t/(|t|+ε)`, as density mode does) | would destroy the per-sample structure the trial is about |

### The prior that makes this worth running

`docs/gmm_posterior_loss.md` §3(a) records that a sampled-label
`log q(c|x) − log p(c|x)` scalar **was tried once and diverged** — it reached −25
and grew `grad_norm` 8× in ten steps. That `q` was a *fitted generative GMM*:
not a normalised posterior, free to assign near-zero mass to `c` on samples it
had never modelled, so minimising the scalar rewarded producing samples its own
`q` misclassified. Off the manifold that is arbitrarily easy.

Everything about `q` here is different, and each difference targets that failure:

* `q` is a **discriminative, normalised softmax head**, not a density ratio;
* it is **initialised from `p`**, so the term starts at exactly 0 with exactly
  zero gradient rather than at some arbitrary fitted offset;
* it is trained by **cross entropy on the generator's actual samples**, so it
  tracks the generator rather than modelling it with 100 Gaussians;
* the generator sees a **slow EMA teacher**, so `q` cannot race the generator;
* the term lives on a **1152-d VLM feature the FD loss already computes**, not a
  128-d whitened projection.

It is still unbounded below. §7 is the watch list.

---

## 2. Files

| file | role |
|---|---|
| `vlm_linear_heads.py` | `VLMLinearHead`, `VLMDeltaHeads` (p / q_student / q_teacher + EMA + drift), `ClassBalancedFeatureBuffer`, all metric helpers, `LocalClassMap` |
| `qwen_answer_state.py` | the second feature backend: Qwen2.5-VL-7B's *answer state*, plus the first-order image-VJP surrogate that keeps its 7B graph out of the generator backward |
| `train_vlm_p_head.py` | Phase 1: cache frozen-VLM features on real images (one head per candidate layer), train the `p` head, fit and freeze the temperature, write the checkpoint |
| `validate_vlm_p_head.py` | the mandatory `p` validation report (also called at the end of training) |
| `conditional_main_fd_vlm_delta.py` | the training entry point |
| `scripts/run_jit_uncond_vlm_delta_100class.sh` | launcher, with `CALIBRATION=1` mode |
| `scripts/analyze_vlm_delta_run.py` | trajectory, milestones, diagnosis, per-class table, and `--calibration` |
| `scripts/sweep_vlm_q_regularization.py` | offline q learning-rate / weight-decay sweep on a saved replay buffer; run before Step D |
| `scripts/smoke_vlm_delta.sh` + `scripts/check_vlm_delta_smoke.py` | Step B integration smoke |
| `scripts/train_vlm_p_head_qwen.sh` | Phase 1 launcher for the answer-state backend |
| `tests/test_vlm_delta.py` | 57 unit tests |
| `tests/test_qwen_answer_state.py` | 22 tests for the answer state: VJP exactness in fp32, layer indexing, checkpoint identity, and the reduced-precision measurement |
| `docs/vlm_delta.md` | this file |

The one change to shared code is a new keyword-only `feature_collector=None`
argument on `frechet_distance.judges.fill_all_queues`, so the `q` replay buffer
can be seeded from the queue fill instead of paying for its own 50,000-image
generation pass. With the default it is a no-op; the GMM tests still pass.

---

## 3. The representation

`vit_so400m_patch16_siglip_256.v2_webli`, `cls` pooling, images fed at 256 px
and resized to 224 inside `TimmReprModel`. It is **already the first FD judge**
in the 100-class recipe, so `z` is computed once per step and the conditional
term costs one 1152×100 matmul. Frozen throughout — parameters never receive a
gradient, only the input does.

Both heads are `Linear(1152, C)` on a **frozen affine feature normalisation**
`(z − μ)/σ` fitted on the real training features. Not a BatchNorm: its running
statistics would drift differently for `p` and `q` and would make the p/q
comparison depend on something other than the two weight matrices. With this
choice `p` and `q` differ in exactly `(W, b)` and nothing else, which is what
makes `q_teacher_weight_delta_l2` a meaningful drift meter.

### 3.1 The second backend — the VLM's answer state

`--vlm_backend qwen_answer_state` replaces SigLIP's CLS token with the internal
state Qwen2.5-VL-7B is about to answer from. One chat turn per image,

```
[<image>]  "What is the ImageNet class of this image? Answer:"
```

rendered through `apply_chat_template(add_generation_prompt=True)`, one forward
(never `generate`), and `z = hidden_states[layer][:, -1, :]` ∈ ℝ^3584 — the
vector the LM head would multiply to emit the first token of the class name.
At 256 px Qwen sees 81 visual tokens and the whole prompt is 114 tokens.

The motivation is that the feature is **label-aligned by construction**: the
model has been asked which class this is, and `z` is its answer before
verbalisation. SigLIP's embedding is generically contrastive and knows nothing
about the question. Nothing downstream changes — same delta term, same replay
buffer, same EMA teacher, same calibration — only `z`.

The prompt names no class, so one rendered prompt serves the whole batch, every
sequence has identical length, and there is no padding to reason about. It must
stay that way: a prompt that named the class would make `z` a re-encoding of a
label handed to the model rather than the model's own answer. The **rendered**
prompt's SHA-256 is part of the p-head identity, so a reworded question — or a
changed chat template — invalidates the head rather than silently shifting `z`.

**Which layer is measured, not chosen.** `hidden_states` indices follow
HuggingFace's convention: `0` is the embedding output, `k` the output of decoder
layer `k−1`, and `28` (also `-1`) the final post-RMSNorm state the LM head
consumes. The last layer is next-token-specialised; a late-middle layer is often
the better linear probe. Every candidate in `--vlm_probe_layers` comes out of the
*same* forward, so the sweep costs one fp16 array per layer and zero extra VLM
time, and the winner is picked on the calibration half of real val.

**Cost.** This backend is *not* an FD judge: it owns no reference statistics and
no Frechet term, so unlike SigLIP its forward is not already being paid for.
Measured on one A100-80GB at 256 px:

| | throughput | peak memory |
|---|---|---|
| forward only (feature extraction) | ~46 img/s | 22 GB at batch 128 |
| forward + image VJP, microbatch 12 | ~18 img/s | 29 GB |
| forward + image VJP, microbatch 24 | ~20 img/s | 41 GB |

At 24 images/rank/step that is ~1.2 s/step of added VLM work, against a
measured ~40.8 GB peak for the existing dual-judge runs. Use
`--vlm_samples_per_step K` to score a round-robin subset instead of the whole
batch; it is unbiased because generated batch elements are exchangeable, and it
trades gradient variance for step time.

### 3.2 Memory-safe gradient injection

The 7B activation graph must never be alive at `loss.backward()` alongside the FD
judges'. `QwenAnswerStateExtractor.vjp_surrogate` therefore does what
`vlm_judge.BinaryVLMJudge.conditional_loss` already did: per microbatch it makes
a detached leaf, evaluates the caller's term on the live `z`, takes the image VJP
immediately, releases the 7B graph, and returns

    stopgrad(term) + <x − stopgrad(x), stopgrad(g)>.

At the current images this has exactly the term's value and exactly its first
derivative with respect to the generator's images. `tests/test_qwen_answer_state.py`
checks that against the direct computation **in fp32**, where it agrees to
`cos > 0.9999` and five decimal places regardless of microbatch size. The
detached `z` from that same forward feeds the `q` replay buffer for free.

### 3.3 Reduced precision: similar features, different image gradients

Measured on real ImageNet val images at 256 px, against an fp32 reference:

| quantity | bf16 vs fp32 | fp16 vs fp32 |
|---|---|---|
| `z` itself (cosine) | **0.9999** | 1.0000 |
| `z` relative error | 1.4 % | 0.14 % |
| image gradient (cosine) | **0.03** | −0.48 |
| image gradient magnitude ratio | 1.05 | 5.07 |

These measurements show similar feature directions but substantially different
image-gradient directions. They do not establish identical p-head predictions
or identify the source of error: weight/activation/backward rounding,
cancellation, and fp16 underflow or overflow need controlled comparisons.
In particular, a 5× gradient magnitude does not diagnose an unscaled backward.
The surrogate's fp32 equivalence tests validate its gradient wiring within a
fixed precision setting.

Repeated bf16 calls were self-consistent (`cos = 0.99998`), which establishes
repeatability rather than agreement with fp32. Historical fp32 measurements
were ~37 GB and ~3.5× slower for their tested setup; measure memory again at a
smaller microbatch before concluding it cannot fit. The new
[precision sweep](vlm_lora_q.md#precision-and-loss-scaling) compares fixed
inputs and separate p/q/delta gradients at multiple loss scales, with TF32 off.

Nothing here says the bf16 gradient carries no signal — what is measured is
per-sample direction agreement between two models, not the expectation over
samples and steps that training actually integrates. But it is the reason to
read `cos_update_fd_vlm` and `grad_ratio_vlm_fd` sceptically on this backend, and
to treat a short CAL run as the decision point rather than a formality.

---

## 4. Phase 1 — the `p` head

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 \
/home/nvidia/miniconda3/envs/fdloss/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node=4 --master_port=29591 \
    train_vlm_p_head.py \
    --data_path /data/dataset/imagenet \
    --output_dir work_dirs/vlm_p_head_siglip_c100 \
    --class_ids $(seq 0 10 990 | tr '\n' ' ') \
    --train_views 2 --epochs 100
```

Features are extracted once (center-cropped 256 px full frames plus their
mirrors — deliberately the same geometry as the generated images the head will
have to score) and cached, so the head itself trains in ~20 s and the slow VLM
forward is paid exactly once. Re-validate any time with

```bash
CUDA_VISIBLE_DEVICES=1 python validate_vlm_p_head.py \
    --p_head work_dirs/vlm_p_head_siglip_c100/p_head.pt \
    --feature_cache work_dirs/vlm_p_head_siglip_c100/cache/*_val.pt
```

For the answer-state backend, `scripts/train_vlm_p_head_qwen.sh` does the same
thing on the same class subset:

```bash
GPUS=1,2,3 bash scripts/train_vlm_p_head_qwen.sh
```

It fits one head per layer in `PROBE_LAYERS` from a single extraction pass and
prints the comparison table before saving the winner. `--head_init lm_head`
additionally seeds each head from the LM-head row of every class name's first
token and reports that head's top-1 *before* fitting — a zero-shot readout of
how label-aligned the answer state already is.

### Measured, 100-class stride-10 subset (2026-08-28)

```
p_real_top1                  0.9840        chance 0.0100
p_real_top5                  0.9984
p_real_ce                    0.0639        uniform 4.6052
p_real_target_logp_mean     -0.0639
p_real_target_logp_median   -0.0001
p_real_entropy               0.0807
mean top-1 confidence        0.9782
target rank mean / median    1.039 / 1.0   chance 50.5
target prob quantiles        p10 0.980  p25 0.999  p50 1.000  p75 1.000  p90 1.000
frac > 0.99                  0.8716
per-class counts             50 each; worst per-class top-1 0.88 (class 810)
temperature (fitted, frozen) 0.8639
```

For reference the fitted-GMM classifier on the same subset reached 94.9% top-1
(`docs/gmm.md` §4). The linear probe is meaningfully stronger.

The checkpoint stores VLM identity, feature dim, class ids, the frozen
normalisation, `p_head.weight/bias`, the temperature, and SHA-256 digests of the
class set and of the head itself. The generator run **refuses to launch** if any
of it disagrees with the run (verified: wrong class set, wrong VLM, and resume
against a different `p` head all abort at start-up).

### Temperature

`--vlm_head_temperature` sets one shared temperature for **both** heads; empty
means "use the one in the checkpoint". It is fitted by standard temperature
scaling (NLL) on one half of the real validation split and then frozen; the
report's headline numbers come from the other half. `p` and `q` are never tuned
separately, and generator performance is never used to pick it.

**Note the saturation.** At `T = 0.864`, 87% of *real* images sit above
`p(c|z) = 0.99`, i.e. `log p` is flat there. That is not a problem at the start —
the Step C diagnostic below measures `p`'s entropy on *generated* samples at
3.17 nats against a uniform 4.61, with `log p(c|z) = −5.87`, so there is plenty
of gradient — but it is where the fidelity half of the field goes quiet once
samples reach real-data confidence. If the term goes quiet late in the run,
`T` is the lever, and the criterion for moving it is the real-validation table
printed by `validate_vlm_p_head.py`, never the generator's score.

---

## 5. Phase 2 — the `q` head

`q_student` and `q_teacher` are both initialised from `p`, so at step 0

```
max |log q_student − log p| = 0.000e+00
max |log q_teacher − log p| = 0.000e+00
max |W_q − W_p|             = 0.000e+00
```

(exact, asserted at start-up: the run aborts if it is not).

Each generator step, after the generator update:

1. `q_student` is evaluated on the current batch → `*_pre` metrics;
2. the batch's **detached** VLM features are pushed into a class-balanced replay
   buffer (per-class ring, so no class can dominate `q`'s training data);
3. `--vlm_q_updates_per_step` cross-entropy updates are taken on a batch drawn
   from the buffer;
4. `q_student` is re-evaluated → `*_post` metrics;
5. `psi_teacher <- beta*psi_teacher + (1-beta)*psi_student`.

The generator's loss uses `q_teacher` as it stood **before** this update.

Defaults: `adamw`, `lr 1e-3`, **`weight_decay 3.0`**, `grad_clip 1.0`, 1
update/step, batch = the global batch, from a 20,000-entry buffer (200/class at
100 classes), `beta = 0.999`. Measured on the smoke run, the teacher moves ~130×
less than the student.

Two of those defaults are not cosmetic. Both were set from measurements taken on
the first calibration run and its own saved replay buffer, and getting either
wrong silently turns the conditional term into noise.

### 5.1 There is nothing for q to learn at the start — and that matters

Pulling the replay buffer out of a step-2000 checkpoint and sweeping q offline
on an 80/20 split of it (`scripts/sweep_vlm_q_regularization.py`):

| | held-out CE | held-out top-1 |
|---|---|---|
| uniform | **4.6052** | 0.0100 |
| the frozen `p` head | 7.7028 | 0.0098 |
| best of 23 swept q settings | 4.7071 | 0.0105 |
| worst (lr 1e-3, no decay) | 8.0644 | 0.0103 |

**No setting beats uniform, and held-out top-1 is chance everywhere**, while
train top-1 reaches 40%. At a de-conditioned start the generated images carry
essentially **no linearly-decodable label information** in SigLIP space, so the
true `q(c|z)` *is* uniform. Consistent with everything else at step 0:
`probe_top1` 0.000, p at rank 50.49 of a chance 50.5, `cond_delta` 0.0004.

The consequence for the objective is not a defect, it is the theory working:
with `q` uniform, `∇_z log q(c|z) ≈ 0` and the field is `−∇_z log p(c|z)`, so
descent ascends `log p(c|z)`. `docs/gmm.md` §12.4 records exactly the same thing
for `cfg_delta` ("at the de-conditioned start `Δ_q` is essentially zero … the
field *is* the teacher delta"). This is **not** the forbidden second `−log p`
driver — it is the one term's own correct behaviour, and the `q` half only
starts contributing once conditioning actually emerges. q's real job right now
is the ~3-nat move from p (CE 7.70, confidently *wrong* on generated images)
down to uniform.

### 5.2 Sample reuse must be ~1, or q memorises instead of estimating

A sample lives in the buffer for `buffer_size / global_batch` steps and each
update draws `q_batch` of `buffer_size`, so

```
reuse = q_batch * updates_per_step / global_batch
```

— independent of the buffer size, which controls *staleness*, not reuse. The
first calibration run used `q_batch 512` at global batch 96, i.e. **5.3×**, and
`q_train_accuracy` climbed to 6% on buffer samples while
`vlm_q_student_top1_pre` — the honest read, on a batch that has not yet entered
the buffer — sat at exactly chance and `vlm_q_student_ce_pre` stayed *above*
uniform. That is memorisation, and it makes `log q_teacher(c|z)` noise on
precisely the fresh samples the generator loss evaluates it at.

`--vlm_q_batch_size` therefore defaults to `0`, meaning the global batch, giving
reuse 1.0. The startup log prints the factor and warns above 2×, and
`q_generalization_gap` (`vlm_q_student_ce_pre − q_ce`) makes it visible per step.

### 5.3 Weight decay is what makes the term calibratable at all

An unregularised online head on a signal-free stream random-walks: its
cross-entropy gradient is unbiased noise, so `‖W_q − W_p‖` grows like
`sqrt(steps)` **forever**. Observed directly on the first calibration run —
`q_teacher_weight_delta_rel` 0.44 → 0.97 of `‖W_p‖` between steps 900 and 2640,
with `grad_ratio_vlm_fd` tracking it 0.60 → 1.04 the whole way. A weight
calibrated at one step is wrong at the next.

AdamW's decoupled decay pulls `W` toward 0 — which is the uniform posterior that
*is* the correct q here — turning the walk into an Ornstein-Uhlenbeck process
with a stationary distribution. Measured, batch 96, 20,000 updates, teacher
`beta = 0.999`, drift in units of `‖W_p‖`:

| lr | wd | d@2k | d@5k | d@10k | d@20k | plateau? | held-out CE | ‖∇_z δ‖ |
|---|---|---|---|---|---|---|---|---|
| 1e-3 | 0 | 1.86 | 2.87 | 3.99 | 5.59 | **no (walks)** | 8.064 | 0.0449 |
| 1e-3 | 0.1 | 1.76 | 2.45 | 2.93 | 3.22 | partly | 6.331 | 0.0252 |
| 1e-3 | 1 | 1.32 | 1.38 | 1.38 | 1.38 | yes | 5.023 | 0.0105 |
| **1e-3** | **3** | **1.12** | **1.11** | **1.12** | **1.11** | **yes** | **4.769** | **0.0086** |
| 1e-3 | 10 | 1.03 | 1.03 | 1.03 | 1.03 | yes | 4.654 | 0.0081 |
| 3e-4 | 0 | 1.01 | 1.58 | 2.22 | 3.10 | **no (walks)** | 6.446 | 0.0241 |
| 1e-4 | 0 | 0.58 | 0.91 | 1.28 | 1.81 | **no (walks)** | 5.566 | 0.0138 |
| 1e-4 | 3 | 0.71 | 0.97 | 1.07 | 1.09 | yes | 4.765 | 0.0086 |
| 1e-4 | 10 | 0.92 | 1.01 | 1.02 | 1.01 | yes | 4.654 | 0.0081 |

Three things to read off it:

* **`wd = 0` never plateaus at any learning rate.** Reject it.
* At `wd ≥ 3` the equilibrium is **independent of the learning rate** — 1e-4,
  3e-4 and 1e-3 give the same drift, the same held-out CE and the same field.
  Decay sets the operating point; lr only sets how fast q tracks a *changing*
  generator, so the fastest stable lr is the right one.
* **`wd = 3` is the chosen default.** `wd = 10` is marginally closer to uniform
  now (4.654 vs 4.769 against a uniform 4.605) but constrains q's equilibrium
  weights ~3× harder, which would shrink the `log q` counterweight later,
  exactly when the experiment needs it. `wd = 1` is stationary but leaves 0.42
  nats of overconfident noise. Both are one flag away.

A drift of ~1.0 `‖W_p‖` with `cos_to_p ≈ 0.01` is not q running away — it is q
sitting at `W ≈ 0`, i.e. uniform, which is `‖W_p‖` away from p by construction.

Re-run the sweep on any new checkpoint, class count or generator:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/sweep_vlm_q_regularization.py \
    --checkpoint work_dirs/<run>/checkpoints/step_0001999.pth \
    --p_head work_dirs/vlm_p_head_siglip_c100/p_head.pt
```

### 5.4 Why detaching rather than a second VLM forward

The spec's formulation runs the frozen VLM a second time on
`generated_images.detach()`. Detaching the graph features gives numerically
identical values (same frozen function, same input, eval mode) at zero cost, so
that is the default. `--vlm_q_recompute_features` runs the literal second
forward if you want to exercise that path.

### 5.5 Distributed

Every rank trains `q` on the *same globally gathered* detached batch with the
same seeded buffer RNG, so `q` stays identical across ranks with no data-parallel
wrapper. The gradient all-reduce in `q_update_step` is exactness insurance
against float non-determinism accumulating over 50k steps, and
`q_param_rank_max_dev` is logged at diagnostic steps to prove it stayed zero.

---

## 6. Calibrating `grad_ratio_vlm_fd` — this differs from every GMM arm

The weight is set on the **image-space gradient ratio**, never on the loss value:

```
grad_ratio_vlm_fd = ||∇_x (w · L_vlm_delta)|| / ||∇_x L_FD||
```

Target the same band the GMM work established, **0.22–0.30, aim 0.25**
(`docs/gmm.md` §0: both successes sat at 0.250 and 0.248; ≤ 0.14 was dead).

**The GMM recipe does not carry over.** Because `q == p` at initialisation the
term is exactly 0 and its gradient is *exactly* 0 — `grad_ratio_vlm_fd` starts at
0.0000 and grows only as `q` drifts. There is nothing to read in the GMM's
600–1500 window, and the "target a calibration-window median of 0.352 to land at
0.25 sustained" rule from [[fdloss-grad-ratio-drift]] measures a decay that does
not exist here. So:

```bash
# 1. calibration run: long enough for q to have separated
GPUS=4,5,6,7 CALIBRATION=1 CAL_STEPS=3000 WEIGHT=<seed> \
  EXP_NAME=<unique> bash scripts/run_jit_uncond_vlm_delta_100class.sh

# 2. read the trajectory, not a single window
python scripts/analyze_vlm_delta_run.py --calibration --weight <seed> \
    work_dirs/JiT_uncond_vlm_delta/<EXP_NAME>
```

`--calibration` prints the ratio by window (0–250, 250–500, … ) so you can see
whether it has plateaued, then the rescale `WEIGHT * target / observed`. The
injected gradient is exactly linear in `--vlm_delta_weight` at a fixed `q`, so
the rescale is exact in that sense — but `q` keeps drifting and `||∇_x L_FD||`
grows over a long run (×1.77 across the 100-class GMM arm), so re-read the
sustained value after launching and expect one iteration.

### What the ratio actually does here — read the whole curve, not one window

Measured, 100 classes, median `grad_ratio_vlm_fd` per window:

| q settings | w | 0–250 | 250–500 | 500–1k | 1k–1.5k | 1.5k–2k | 2k–3k | 3k–4k |
|---|---|---|---|---|---|---|---|---|
| `wd = 0`, reuse 5.3× | 0.002 | 0.016 | 0.128 | 0.602 | 0.713 | 0.933 | 0.993 | **1.043 — still climbing** |
| **`wd = 3`, reuse 1.0** | 0.0006 | 0.006 | 0.059 | 0.229 | 0.294 | 0.315 | 0.313 | **0.350 — flat from ~1.5k** |

The first never plateaus, so no weight calibrated from it survives more than a
few hundred steps. The second flattens after ~1500 steps, which is what makes
the rescale meaningful. **The shipped operating point is
`--vlm_delta_weight 0.000429`**, from `0.0006 × 0.25 / 0.3496`.

Confirmed on the launched trial at that weight — flat for 3,000 steps and inside
the band:

| window | 500–1k | 1k–1.5k | 1.5k–2k | 2k–3k | 3k–4k | 4k–4.7k |
|---|---|---|---|---|---|---|
| median `grad_ratio_vlm_fd` | 0.149 | 0.207 | 0.223 | 0.228 | 0.229 | 0.223 |

**sustained 0.225.** The rescale's linearity also checks out directly: against
CAL2 at matched windows the trial reads 0.65 / 0.70 / 0.71 of CAL2's values,
against a weight ratio of `0.000429/0.0006 = 0.715`. The landed value is 11%
under the 0.25 target — inside the 0.22–0.30 band and well inside its own noise,
so it was not worth restarting for. Do not chase the target to more precision
than the band: as conditioning emerges q starts fitting real signal and the field
changes anyway.

Two lessons, both learned the expensive way:

1. **A short window reads far too low.** The very first seed here was `0.05`; it
   looked plausible for 250 steps and then ran away to **3.2** by step 400. The
   term is *exactly* zero at step 0 and the 500-step weight ramp is still
   opening, so anything read before ~step 1500 is meaningless.
2. **The ratio decays over a long run, exactly as it does for the GMM arms.**
   Over the first ~4,000 steps `grad_x_fd` looks flat and the ratio looks
   stationary — that is a *transient*, and calibrating on it under-drives the
   run. Measured over the full 37,000 steps of the first trial:

   | window | 0–2k | 2k–5k | 5k–10k | 10k–15k | 15k–20k | 20k–25k | 25k–30k | 30k–37k |
   |---|---|---|---|---|---|---|---|---|
   | `grad_x_fd` | 0.0086 | 0.0099 | 0.0119 | 0.0134 | 0.0148 | 0.0152 | 0.0162 | **0.0168** |
   | ratio | 0.163 | 0.229 | 0.234 | 0.216 | 0.208 | 0.199 | 0.197 | **0.189** |

   `grad_x_fd` grows **×1.75** and the ratio drifts **×0.79** — against the GMM's
   documented ×1.77 and 0.70–0.79 ([[fdloss-grad-ratio-drift]]). The same
   correction therefore applies here: **target a calibration-window median of
   ~0.32, not 0.25**, or the sustained value lands below the working band.

   (An earlier revision of this document claimed the opposite, on the strength of
   the first 4,000 steps alone. It was wrong, and the run calibrated from it
   sustained 0.181 — under-driven.)

Because the ratio is driven by q's drift, `--vlm_q_weight_decay` (§5.3) is what
makes it plateau at all. With `wd = 0` it never does, and no weight is
calibrated for more than a few hundred steps.

---

## 7. What to watch

All metrics land in `training_metrics.json`, one JSON object per line. The
calibration quartet, the tail meters and the drift meters are logged with a
window of **1** (the value at that step); everything else is a trailing median,
as in every GMM arm.

### The three columns that decide the result

| meter | null | reading |
|---|---|---|
| `probe_top1` / `probe_rank` | `1/C` / 500.5 | held-out ResNet-50, **never in the loss. The only honest arbiter** |
| `vlm_p_top1` / `vlm_p_target_rank` | `1/C` / `(C+1)/2` | the frozen real-data head's view of the *sampled* label |
| `vlm_q_teacher_top1` / `..._target_rank` | `1/C` | what the generator's loss actually sees |

`vlm_p_top1` rising while `probe_top1` stays at chance is **VLM-space
exploitation, not conditioning**, and `analyze_vlm_delta_run.py` says so
explicitly under `EXPLOITATION WARNING`.

### The pathology watch (§16 of the spec)

The fitted-GMM sampled-label scalar went to −25 and blew up `grad_norm`. Logged
every step, unsmoothed:

```
vlm_delta_logqp_mean  _std  _min  _max      vlm_delta_abs_p95  vlm_delta_abs_p99
vlm_delta_logqp_p10 _p25 _p50 _p75 _p90     grad_ratio_vlm_fd  generator_grad_norm
```

A non-finite `log q − log p` **raises immediately** rather than being clamped, and
`--vlm_max_nonfinite_steps` consecutive non-finite generator gradients abort the
run. `--vlm_delta_clamp` is opt-in and defaults to off; when on,
`vlm_delta_clamp_frac` records how much of the experiment it changed.

### q health and drift

```
q_ce  q_train_accuracy  q_grad_norm  q_lr  q_weight_norm  q_bias_norm
q_student_weight_delta_l2   q_teacher_weight_delta_l2   (+ _rel, _bias_, _cos_to_p)
q_student_update_norm       q_teacher_ema_update_norm
q_teacher_class_weight_delta_mean / _max        # is a few classes dominating q?
q_teacher_student_logit_mse  q_teacher_student_kl
q_buffer_size  _class_coverage  _min_samples_per_class  _max_samples_per_class
vlm_q_student_{top1,ce,...}_pre / _post         # learning, or just fitting this batch?
```

`q_teacher_weight_delta_l2` near 0 means the term is inert. `q_teacher` drifting
*more* than `q_student` means the EMA is misconfigured.

### p vs q

```
vlm_delta_logqp   vlm_logq_c   vlm_logp_c   vlm_probq_c   vlm_probp_c
vlm_pq_full_kl_qp   vlm_pq_full_kl_pq   vlm_pq_js
vlm_pq_top1_agreement   vlm_pq_argmax_disagreement   vlm_target_prob_gap
```

The `vlm_pq_*` family is **diagnostic only** — it is not what is optimised. It
separates "q has learned something about the generator" from "p and q are still
making identical decisions".

### Label sensitivity

`cond_delta` (relative pixel change from swapping the class token under the same
noise — exactly as the GMM entry point computes it) and `vlm_cond_feature_delta`
(the same question in VLM feature space, which can move before pixels do).

### Per-class

`vlm_per_class_step_XXXXX.json` every `--vlm_per_class_every` steps: per class,
`p_top1`, `q_top1`, `probe_top1`, mean target probabilities, target ranks, mean
`logq−logp`, and the sample count. This is what answers *which classes learned
first* and *is the average being driven by a handful of classes*.

---

## 8. Running it

### Step A — the p head

§4 above. Do not continue unless the printed verdict is `OK`.

### Step B — integration smoke

```bash
GPU=1 bash scripts/smoke_vlm_delta.sh
```

57 unit tests, then a real 60-step run + resume against the real frozen SigLIP
judge and the real de-conditioned JiT-B, then the launch refusals. It asserts all
ten Step-B checks — plus `q_generalization_gap` and the `vlm_delta_logqp ==
vlm_logq_c - vlm_logp_c` identity — against what the training script actually
wrote to disk, and it runs with the **shipped** q defaults so the memorisation
guard is exercised rather than bypassed.

It deliberately uses a global batch of 48 rather than the smallest that fits.
Reuse 1.0 pins the q batch to the global batch, and at a global batch of 8 each
CE step sees fewer than a tenth of a sample per class: q overfits those few
samples and `q_ce` *rises* (measured: 5.3 → 7.5 in twelve steps). The training
script warns when the q batch falls below `C/2`. **This is the constraint to
watch when scaling the recipe** — at 1000 classes, reuse 1.0 at global batch 96
would leave q under a tenth of a sample per class per step, and that
configuration needs either a larger global batch or a deliberate reuse > 1.

### Step C — the initial diagnostic

Produced automatically at step 0 of any fresh run and written to
`vlm_init_diagnostics.json`. Measured on the de-conditioned JiT-B, 100 classes:

```
                       p head     q student   q teacher     chance
target top1            0.0117     0.0117      0.0117        0.0100
target top5            0.0508     0.0508      0.0508        0.0500
target rank (mean)     50.49      50.49       50.49         50.5
target logp           -5.8716    -5.8716     -5.8716       -4.6052 (uniform)
entropy                3.1710     3.1710      3.1710        4.6052 (uniform)

mean logq-logp   +0.000e+00      top1 agreement 1.0000     full KL 0.000e+00
held-out probe top1 0.0000
VERDICT: p is essentially CLUELESS about the sampled label
```

Read this carefully: entropy 3.17 ≪ 4.61 means `p` **is** confident about *some*
class — the generator makes recognisable images — but rank 50.49 against a chance
of 50.5 means that class has nothing to do with the **requested** one. That
distinction is the whole point of the diagnostic, and `log p(c|z) = −5.87`
against a uniform of −4.61 says the sampled label is actively *dis*preferred, so
the fidelity half of the field starts with real work to do.

### Step D — calibration

§6 above. Two preconditions, both cheap and both learned from a wasted run:

1. Check q's regularisation offline first
   (`scripts/sweep_vlm_q_regularization.py`, §5.3). A weight calibrated against
   a random-walking q is wrong within a few hundred steps.
2. Read the *whole* window table, not one number. Seed low: over-driving
   distorts the run you are measuring, under-driving costs nothing but a rescale.

### Step E — the single trial

```bash
GPUS=4,5,6,7 WEIGHT=0.000429 EXP_NAME=<unique> \
  bash scripts/run_jit_uncond_vlm_delta_100class.sh
```

`0.000592` is the calibrated 100-class value for the shipped q settings
(`wd 3.0`, reuse 1.0, `T 0.8639`, SigLIP-SO400M, the stride-10 subset), derived
from the first trial's measured **sustained** ratio: `0.000429 × 0.25 / 0.1813`.

The first trial was launched at `0.000429`, calibrated to 0.225 in the
3,000–4,660 window. It decayed to **0.181 sustained** — below the 0.22–0.30
band — and showed no conditioning in 35,540 samples/class. Calibrate against the
*sustained* target via the ×0.79 drift, not against the window reading.

None of this transfers to another class count, another temperature, or a
different `--vlm_q_weight_decay` — recalibrate from §6 whenever any of those
move.

50,000 steps at global batch 96 = 48,000 samples/class, matching the 100-class
GMM arms exactly so the trajectories are comparable. **Global batch must stay 96**
— every samples-per-class number in `docs/gmm.md` is defined against it, and
changing it invalidates the weight calibration.

An opt-in takeoff gate is available (`TAKEOFF_SAMPLES_PER_CLASS=35000`, derived
to a step from the per-class budget, never typed in) but is **off by default**.
Its criteria are generator-side and cannot be gamed by `q`: `cond_delta`, and the
frozen `p` head's view of the sampled label's rank as a fraction of chance.

### Analysis

```bash
python scripts/analyze_vlm_delta_run.py work_dirs/JiT_uncond_vlm_delta/<run>
```

Prints the §19 trajectory table, the first-crossing milestones (probe / p / q
top-1 above chance, ranks below chance, `cond_delta` rise), the diagnosis
(gradient band, exploitation warning, `logq−logp` trend and tails, q drift,
clamp), real FID from `eval_summary.csv`, and the latest per-class table sorted
by which classes learned first. Point it at several runs for a matched
comparison.

---

## 9. How to call the result

A success is **joint**:

```
probe_top1 ↑     probe_rank ↓     cond_delta ↑     vlm_p_target_rank ↓
```

without pathological growth in `|logq − logp|`, `grad_ratio_vlm_fd`, q drift, or
FID. VLM accuracy alone is not a result: if `vlm_p_top1` becomes high while
`probe_top1` stays at chance, that is VLM exploitation and must be reported as
such.

Reference points at 48,000 samples/class on the same 100 classes and the same
de-conditioned checkpoint (`docs/gmm.md` §8):

| arm | `probe_top1` | `probe_rank` | `cond_delta` | best FID |
|---|---|---|---|---|
| GMM density, `w` correct | 0.198 | 314 | 0.383 | 14.94 |
| GMM density, `w` too low | 0.010 | 493 | 0.115 | 11.92 |
| GMM posterior (class-summed KL) | 0.000 | ~500 | 0.09 | 57.4 |
| chance | 0.010 | 500.5 | 0 | — |

## 10. Known limitations

* **The term is unbounded below.** `log q(c|z)` has no lower bound, so a
  generator that produces samples its own `q` assigns zero mass to can drive the
  objective arbitrarily negative. Nothing here proves it will not; §7's tail
  meters exist to catch it early, and `--vlm_delta_clamp` is the opt-in
  emergency brake.
* **`T ≠ 1` changes the objective.** As in `cfg_delta` mode (`docs/gmm.md`
  §12.1), the implemented field at `T ≠ 1` is the *tempered* posterior
  difference. `T` is fitted on real data and frozen, but the weight must be
  recalibrated if it ever moves.
* **`q` is a linear probe.** It can only represent class structure that is
  linearly decodable from the frozen VLM feature. If the generator's conditioning
  lives somewhere else, `q` cannot see it and the term will not push on it.
* **One class count.** Nothing here is tested at 20 or 1000 classes, and neither
  the temperature nor the weight transfers across class counts (`docs/gmm.md`
  §6, [[fdloss-cfg-delta-mode]]).
* **`--compile` is unsupported**: the calibration this run is built around needs
  per-term eager autograd through the frozen judge at every diagnostic step.
* **q is regularised toward uniform, and that is a modelling choice.** `wd = 3`
  was picked because it makes the drift stationary and lands q within 0.16 nats
  of the true (uniform) posterior at a de-conditioned start. It also caps how
  sharp q can become later. If conditioning takes off and
  `q_generalization_gap` stays near zero while `vlm_q_student_top1_pre` climbs
  well above chance, q is genuinely learning and a smaller `wd` may extract more
  counterweight — but re-calibrate `--vlm_delta_weight` if you change it.
