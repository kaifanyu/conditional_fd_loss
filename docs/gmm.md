# The class-conditional GMM loss — complete working record

**Status as of 2026-08-18.** Consolidated reference for the `log p` / `log q`
GMM term: what it is, how the reference GMM was built, every parameter, every
run, what has been tuned and why, and where things stand.

Companion documents, kept for their derivations and not duplicated here:

| doc | contents |
|---|---|
| [`docs/gmm_posterior_loss.md`](docs/gmm_posterior_loss.md) | the objective's derivation, the three bring-up failure modes, the pMF-B/256 2×75k matched-control result |
| [`docs/gmm_uncond_20class.md`](docs/gmm_uncond_20class.md) | the 20-class pilot record (the one unambiguous success) |
| [`docs/GMM_INTEGRATION_README.md`](docs/GMM_INTEGRATION_README.md) | provenance: what was moved from `yhpark/FD-Loss` and what never merged |

---

## 0. TL;DR — the one thing that matters

Six runs have been executed against the de-conditioned JiT-B. Sorting them by
the sustained image-space gradient ratio of the GMM term over the FD term
(`grad_ratio_p_fd`, median over steps ≥ 6000) explains **every** outcome:

| run | sustained `grad_ratio_p_fd` | final `probe_top1` | chance | verdict |
|---|---|---|---|---|
| 20c arm C | **0.250** | 0.9271 | 0.050 | **conditioning** |
| 100c #2 | **0.248** | 0.1979 | 0.010 | **conditioning** (ended mid-takeoff) |
| 1000c v2 | 0.215 | 0.0000 | 0.001 | inconclusive — gate killed it at step 10k |
| 100c #1 | 0.140 | 0.0104 | 0.010 | dead |
| 1000c v1 | 0.088 | 0.0000 | 0.001 | dead |
| 20c arm B | 0.045 | 0.0521 | 0.050 | dead |

Both successes sat at ~0.25. Everything at ≤ 0.14 was dead flat. The documented
"target band" of 0.10–0.40 is **too wide at its bottom** — the working range
looks like 0.22–0.30, and the months of "it doesn't scale past 20 classes" were
a weight-calibration problem, not a class-count problem.

The 1000-class question is reopened and untested at a correct weight.

---

## 1. What the term is for

`compute_frechet_distance_loss` sees exactly two things about the generator: the
pooled `mu` and `sigma` of its features — `d + d(d+1)/2` scalar constraints. A
distribution supported on `2d+1` points satisfies all of them exactly, so
**FD → 0 is reachable by a generator with essentially no diversity.** Collapse
lives in the null space of the objective.

Adding `-log p(c|x)` from a real-data classifier does not fix it: that term's
per-sample minimiser is a *point*, so it supplies a sharper attractor for
collapse. The missing ingredient is a force that pushes samples apart, and the
only place it can come from is the generator's own current distribution — hence
a second, online GMM `q`.

In the current (de-conditioned JiT-B) setting the emphasis is inverted from the
original design: the model starts with **no** class conditioning at all, so
`E[-log p(c|x)]` is the *driver* that has to carve class structure from nothing,
and the `log q` term is its counterweight.

---

## 2. The objective

Everything lives in a **whitened PCA space** fitted once on real features:

```
z = (f - mean_p) @ P        with  P = V_k S_k^{-1/2}
```

so the real pooled covariance is the identity in `z`-space and both sides shrink
toward a common, meaningful target.

```
L = sum_j FD_j / (FD_j.detach() + eps)                    # siglip + mae + inception
  + w(s) * [ lambda_cls * E[-log p(c|x)]                  # class fidelity   (driver)
           + lambda_ent * E[log q(x|c) - log p(x|c)] ]    # anti-collapse    (counterweight)
```

The second term is `KL(q(·|c) || p(·|c))` in expectation. Its gradient in
whitened space is exactly the two forces the design is built around:

```
d/dz  =  Lambda_c^p (z - mu_c^p)   -   Lambda^q (z - mu_c^q)
         \_ pull toward the real _/   \_ push off the generated _/
            class mean                   class mean
```

As the generator collapses, `Sigma^q` shrinks, `Lambda^q` grows, and the
repulsion **strengthens** — the term is strongest exactly when it is needed.

### Sample-count asymmetry (why p and q are modelled differently)

| side | per-class samples | model |
|---|---|---|
| `p` (real) | ~1300 | full per-class covariance (QDA), shrunk 0.75 toward pooled within |
| `q` (generated) | ~50 at bootstrap, ~1000 at EMA steady state | **tied** covariance + per-class means (LDA) |

A `k × k` covariance cannot be estimated from 50 samples; a `k`-dim mean can.
Defining `q`'s tied covariance as the pooled *within-class* scatter makes
`tr(Sigma_within^q) / tr(Sigma_within^p)` a free per-step collapse meter.

All GMM parameters are detached; gradients flow only through
`x → features → z → quadratic form`. For `q` this is unbiased, not an
approximation — with reparameterised sampling the parameter-side term has zero
expectation by the score identity.

---

## 3. Code map

### Core

| file | lines | role |
|---|---|---|
| [`frechet_distance/gmm.py`](frechet_distance/gmm.py) | 965 | `ClassGMMReference` (p), `OnlineClassStats` (q), `gmm_posterior_loss`, `_self_normalize` |
| [`conditional_main_fd_gmm.py`](conditional_main_fd_gmm.py) | 1309 | training entry point: args, judge attachment, loss in the train step, takeoff gate, diagnostics, checkpointing |
| [`compute_class_stats.py`](compute_class_stats.py) | 348 | one-pass offline fit of the `p` side: whitening PCA + per-class Gaussians |
| [`scripts/validate_class_gmm.py`](scripts/validate_class_gmm.py) | — | held-out accuracy, shrinkage sweep, **temperature sweep** (added 2026-08-13) |

Key entry points inside `gmm.py`:

```
ClassGMMReference          .from_npz  .project  .logits(z, temperature)  .log_likelihood
                           .to_local  .validate_labels        # class-subset support
                           .posterior_score_delta             # grad_z log p_T(c|z)  (cfg_delta)
OnlineClassStats           .update  .refresh_cache  .logits  .log_likelihood
                           .class_means  .coverage  .mean_ema_effective_samples
                           .diagnostics(reference)            # all the collapse meters
                           .posterior_score_delta             # grad_z log q_T(c|z)  (cfg_delta)
gmm_posterior_loss(...)                                       # the loss itself
```

`conditional_main_fd_gmm.py` — not `main_fd.py`. The original integration in
`docs/gmm_reference_sources/` was never merged into the working-tree
`main_fd.py`; this is a separate entry point.

### Reference-statistics builders

| file | produces |
|---|---|
| `scripts/compute_class_gmm_20class.sh` | `data/fid_stats/inception_in256_t256_classgmm_k128_c20.npz` |
| `scripts/compute_class_gmm_100class.sh` | `..._classgmm_k128_c100.npz` |
| `scripts/compute_imagenet20_fd_stats.sh` | `data/fid_stats/imagenet20_v1/{siglip_cls,mae_cls,inception}.npz` |
| `scripts/compute_imagenet100_fd_stats.sh` | `data/fid_stats/imagenet100_v1/{...}` |
| `scripts/imagenet100_class_ids.sh` | single source of truth for the 100-class subset |

### Run launchers

| file | run |
|---|---|
| `scripts/run_jit_uncond_gmm_20class.sh` | the 20-class pilot (arms A/B/C via `ARM=`) |
| `scripts/run_jit_uncond_gmm_1000class.sh` | 1000-class v1 |
| `scripts/run_jit_uncond_gmm_1000class_v2.sh` | 1000-class v2 (introduced the takeoff gate) |
| `scripts/run_jit_uncond_gmm_100class.sh` | 100-class probe; gate step **derived** from samples/class; `CALIBRATION=1` mode |
| `scripts/analyze_gmm_run.py` | prints every meter for one or several arms side by side |

### Tests

```
tests/test_gmm_posterior.py       19 synthetic-Gaussian correctness checks
tests/test_gmm_cfg_delta.py       11 cfg_delta checks (analytic-vs-autograd, sign, no-leak)
tests/test_gmm_takeoff_gate.py     8 gate-logic checks
tests/test_class_subset.py           class-subset label mapping
```

---

## 4. How the separate GMM "classifier" was built

This is the `p` side — fitted **once, offline, on real images**, then frozen.
It is a generative classifier (per-class Gaussians), not a trained discriminative
head; nothing is learned by gradient descent.

### Why a whitened PCA space

A full `2048 × 2048` covariance per class is both unstorable
(`1000 × 2048² × 8B = 33 TB`) and unestimable (~1300 images per ImageNet class
against 2048 dimensions → singular). Projecting onto the leading `k=128`
whitened principal directions of the *real* features fixes both: storage drops
to `C × k²`, and `1300 >> 128` makes the per-class covariances well posed.
Whitening rather than plain PCA makes the real pooled covariance the identity in
`z`-space, giving both sides a common shrinkage target.

### The procedure (`compute_class_stats.py`)

1. One distributed pass over ImageNet train extracting judge features; each rank
   caches its own features in CPU fp16 and accumulates global sufficient statistics.
2. Rank 0 eigendecomposes the pooled covariance, builds `P = V_k S_k^{-1/2}`,
   broadcasts it.
3. Each rank projects its cached features and accumulates per-class sums and
   outer products in `k` dims; reduced to rank 0 → per-class means/covariances
   plus the pooled within-class covariance.

Output `.npz` keys: `feat_mean`, `pca_basis`, `class_mu`, `class_cov`,
`class_count`, `within_cov`, `meta`, and `class_ids` when a subset was fitted.

```bash
# 1000-class (the original, ~4 min on 3 A100s over 1.28M images)
CUDA_VISIBLE_DEVICES=4,5,6 torchrun --nproc_per_node=3 compute_class_stats.py \
    --model inception --data_path /data/dataset/imagenet --img_size 256 --pca_dim 128

# subsets
bash scripts/compute_class_gmm_20class.sh
bash scripts/compute_class_gmm_100class.sh
```

### The p side must be refitted per subset, never masked

The whitening PCA, the pooled within-class covariance and the log_softmax
denominator are all defined **over the subset**. Masking the 1000-class file down
to 20 would make `-log p(c|x)` a 1000-way problem against 980 classes the
generator never draws, and would leave those classes' `mu_c^q` parked on
`mu_c^p` as spurious attractors in the softmax.

`ClassGMMReference.validate_labels` enforces this at launch for the GMM. **It
does not check the FD references** — a subset mismatch there is silent, which is
why `scripts/imagenet100_class_ids.sh` exists as a single sourced list.

### Fitted geometry — this turned out to matter a lot

`tr(·)/k` in the whitened space; within + between ≈ 1 by construction.

| fit | tr(within)/k | tr(between)/k | held-out top-1 @ p_shrink 0.75 |
|---|---|---|---|
| 20-class | **0.8588** | **0.1418** | 98.6% |
| 100-class | 0.4016 | 0.5998 | 94.9% (top-5 99.0%) |
| 1000-class | 0.4423 | 0.5583 | 76.6% |

**The 20-class diagnostic set is the geometric outlier**, not the larger fits.
Its class means sit close together relative to within-class spread (between =
0.14); at 100 and 1000 classes they are far apart (0.56–0.60) and those two
resemble each other closely. This is why the 100-class subset was chosen as a
stride-10 sample of the label space rather than an extension of the hand-picked
"visually distinct" 20 — the probe has to predict 1000 classes, and it does.

---

## 5. Every parameter

### The GMM term

| flag | default | used | what it does |
|---|---|---|---|
| `--fd_gmm` | off | on | master switch; everything below is inert without it |
| `--fd_gmm_stats_path` | None | per-subset `.npz` | the fitted `p` |
| `--fd_gmm_judge` | None | `inception` | which judge's space the GMM lives in; must match the judge's target size |
| `--fd_gmm_pca_dim` | None | 128 | `k`; must match the fit |
| `--fd_gmm_weight` | 0.002 | **0.0038–0.0103** | `w(s)`. **The single most important knob — see §7** |
| `--fd_gmm_lambda_cls` | 1.0 | 1.0 | weight of `E[-log p(c|x)]` |
| `--fd_gmm_lambda_ent` | 1.0 | 1.0 | weight of the anti-collapse term |
| `--fd_gmm_cls_cap` | 0.0 | **-0.69** | per-sample confidence cap; above it a sample contributes exactly zero `l_cls` gradient. `-0.69 = log 0.5` |
| `--fd_gmm_temp` | 1.0 | **100 / 45 / 10** | softmax temperature on `p(c|x)`. Changes no ranking. **Does not transfer across class counts — see §6** |
| `--fd_gmm_cls_normalization` | legacy | `self` \| `log_classes` | `self`: `l_cls/(abs(l_cls)+0.01)` ≈ 1, bounded. `log_classes`: `l_cls/ln(C)`, fixed divisor |
| `--fd_gmm_mode` | `density` | `density` | `posterior` degenerates under collapse (failure mode (b)); `cfg_delta` is the CFG-delta vector field — **§12** |
| `--fd_gmm_cfg_normalization` | `none` | `none` | `cfg_delta` only: `rms` rescales the injected vector by its detached batch RMS. Separate ablation |
| `--fd_gmm_clamp` | 3.0 | 3.0 | density-ratio clamp in batch-sigma units about the detached batch mean (~2% clipped) |
| `--fd_gmm_no_q` | off | arm B only | drop the anti-collapse term |
| `--fd_gmm_ema_beta` | 0.999 | 0.99 / **0.999** | EMA horizon for `q`'s class **means**, in *appearances of that class* |
| `--fd_gmm_cov_ema_beta` | = mean beta | 0.999 | same for `q`'s tied covariance |
| `--fd_gmm_p_shrinkage` | 0.75 | 0.75 | `p` toward pooled-within (QDA → LDA) |
| `--fd_gmm_q_shrinkage` | 0.25 | 0.25 | `q` toward pooled-within |
| `--fd_gmm_warmup_steps` | 0 | 0 | steps before the term switches on at all |
| `--fd_gmm_ramp_steps` | 500 | 500 | linear ramp of `w(s)` to full weight |
| `--fd_gmm_bootstrap` | 50000 | 50000 | samples used to pre-fill `q` before step 0 |

### The takeoff gate (added in 1000c v2)

| flag | default | what it does |
|---|---|---|
| `--fd_gmm_takeoff_gate_step` | -1 (off) | zero-based step at which to evaluate and possibly abort with exit code 4 |
| `--fd_gmm_takeoff_spread_mult` | 2.0 | require `gmm_class_mean_spread_to_noise >= this` |
| `--fd_gmm_takeoff_cos_threshold` | 0.0 | require windowed-mean `cos_update_fd_p < this` |
| `--fd_gmm_takeoff_cos_window` | 50 | number of diagnostic prints averaged for the cosine |
| `--fd_gmm_takeoff_logic` | `any` | `any` = abort if *either* criterion fails |
| `--fd_gmm_takeoff_no_cos` | off | disable the cosine criterion (required with `--compile`) |

Boundaries follow the prose exactly: spread equal to the multiple passes; cosine
equal to the threshold fails. Missing/non-finite values fail closed.

### Non-GMM settings held fixed across all runs

`JiT_B`, `--rope_2d --learned_pe --legacy_time_convention`, `lr 1e-5`,
`--fd_ema_beta 0.99` (0.999 was shown blind to mode collapse),
`--queue_size 50000`, judges = SigLIP-SO400M + MAE ViT-L + InceptionV3 (all `cls`
pooling), `cfg 3.0`, `--num_sampling_steps 1`, `--cond_probe`,
base checkpoint `checkpoints/base/JiT-B-uncond.pth`.

**Global batch is fixed at 96** in every run. Every samples-per-class number in
this document is defined against it. Three GPUs → batch 32/GPU; four → 24/GPU.

---

## 6. Calibration procedures

Two quantities must be measured per configuration and **never carried over**.

### 6.1 Temperature — measured on real held-out images

A `C`-way posterior over 128-dimensional Mahalanobis distances is extremely
sharp. At `T=1` most real images sit at exactly `log p = 0`, where `l_cls` has
**no gradient at all on anything that already looks real**. Temperature flattens
the softmax without changing any ranking (top-1 is identical at every `T`).

Rule: pick the `T` whose **median real image lands just above `--fd_gmm_cls_cap`**
with a saturated fraction near zero. Below that the term is inert on realistic
samples; above it the term keeps pushing past real-data confidence, which is the
adversarial regime the cap exists to prevent.

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/validate_class_gmm.py \
    --stats data/fid_stats/inception_in256_t256_classgmm_k128_c100.npz \
    --class_ids $(seq 0 10 990 | tr '\n' ' ') \
    --shrinkage 0.75 --temperature 1 5 10 20 30 40 45 50 60 100 200
```

**Measured operating points:**

| fit | chosen `T` | median `log p` | saturated | `> cap` |
|---|---|---|---|---|
| 20-class | **100** | -0.312 | 0.0% | 79.6% |
| 100-class | **45** | -0.236 | 0.7% | 70.0% |
| 1000-class | **10** | -0.179 | 17.1% | 70.4% |

Full 100-class sweep (p_shrink 0.75, cap -0.69, 5000 held-out val images):

| T | mean | median | saturated | > cap |
|---|---|---|---|---|
| 1 | -1.796 | 0.000 | 93.5% | 94.9% |
| 10 | -0.231 | -0.000 | 65.7% | 94.2% |
| 25 | -0.285 | -0.010 | 13.9% | 88.1% |
| 30 | -0.352 | -0.031 | 6.6% | 84.4% |
| 35 | -0.435 | -0.071 | 2.9% | 80.4% |
| 40 | -0.531 | -0.138 | 1.4% | 75.7% |
| **45** | **-0.638** | **-0.236** | **0.7%** | **70.0%** |
| 50 | -0.754 | -0.361 | 0.4% | 64.2% |
| 60 | -1.002 | -0.672 | 0.1% | 50.9% |
| 100 | -1.939 | -1.854 | 0.0% | 8.6% |

At `T=60` the median real image sits *at* the cap — that is the upper boundary
before the adversarial regime. `T=100`, the 20-class value, is catastrophic here.

Shrinkage sweep (100-class), showing there is room to move `p_shrink` without
losing classifier quality:

| p_shrink | top-1 | top-5 | `-log p` |
|---|---|---|---|
| 0.00 | 91.04% | 96.62% | 8.607 |
| 0.25 | 93.96% | 98.50% | 2.632 |
| 0.50 | 94.60% | 98.84% | 1.904 |
| **0.75** | **94.88%** | **98.96%** | **1.796** |
| 0.90 | 94.94% | 98.74% | 2.145 |
| 1.00 | 93.96% | 97.74% | 4.756 |

### 6.2 Weight — calibrated on `grad_ratio_p_fd`, never on the loss value

The two terms are self-normalised at incomparable scales, so the loss value is
meaningless for calibration. `grad_ratio_p_fd` is the **image-space** gradient of
the GMM term over the FD term's.

```bash
# 1500 steps, gate/eval/vis off
CALIBRATION=1 CLS_NORMALIZATION=self GMM_TEMP=45 WEIGHT=<seed> \
GPUS=2,3,4 BATCH_SIZE=32 bash scripts/run_jit_uncond_gmm_100class.sh
```

**The ratio decays after the ramp.** Measured on 100c #1: median 0.186 in the
600–840 window, settling to 0.140 sustained — a factor of **0.753**. Calibrate
against the *sustained* target, not the ramped reading.

```
WEIGHT_new = WEIGHT_seed * (sustained_target / 0.753) / ratio_measured_at_600-840
```

The ratio is not exactly linear in the weight (the FD gradient moves too), so one
iteration of this loop is normal.

---

## 7. Parameter history — what was changed, when, and why

### `--fd_gmm_weight` — the one that decided everything

| run | seed → final | ramped reading | sustained | outcome |
|---|---|---|---|---|
| 20c arm C | 0.02 → **0.0066** | 0.61 → 0.26–0.29 | 0.250 | success |
| 20c arm B | 0.0066 (**not recalibrated**) | — | 0.045 | dead |
| 1000c v1 | 0.0038 | — | 0.088 | dead |
| 1000c v2 | 0.0060 → 0.0050 | 0.42 → target 0.35 | 0.215 | gated at 10k |
| 100c #1 | 0.0050 | 0.186 | 0.140 | dead |
| 100c #2 | 0.0050 → **0.0103** | 0.195 → 0.367 | **0.248** | success |

Three distinct mistakes, all the same mistake:

* **arm B was launched at arm C's weight.** `docs/gmm_uncond_20class.md` §3
  explicitly says "**Recalibrate for arm B** — dropping the `q` term removes half
  the loss, so the same weight will not give the same ratio." It wasn't done, the
  ratio landed at 0.045, and the run died. **Arm B is therefore not a valid
  ablation of the `q` term** — it is a second demonstration that a weight below
  the band does nothing. The role of `log q` remains untested.
* **1000c v1 at 0.088** was below the band floor for the whole run.
* **100c #1 at 0.140** was nominally "in band" but at its bottom, and died.

The band inherited from `docs/gmm_uncond_20class.md` is **0.10–0.40**, anchored
on the classifier-ensemble run that reached 63.5% top-1 at 0.18 and r9's 0.34
recorded as the regime that collapsed. Against the six runs here, the true
working range is **0.22–0.30**, and 0.10–0.18 is dead. The r9 collapse anchor was
measured at 20 classes where between-class scatter is 0.142; at 100 classes it is
0.600, so there is much more room to separate class means before collapsing, and
0.25 showed no collapse whatsoever (`within` fell monotonically, never below 1.6).

### `--fd_gmm_temp`

Measured per fit; see §6.1. `T=100` (20c) → `T=10` (1000c) → `T=45` (100c). This
was correctly re-measured every time and is **not** implicated in any failure.

### `--fd_gmm_cls_normalization`

* 20c pilot: legacy `self`.
* 1000c v2 and 100c #1: switched to `log_classes`, rationale "a hard example
  cannot turn its own gradient down."
* 100c #2: switched **back** to `self`, to match the pilot.

Still a **live confound**: 100c #2 changed weight *and* normalizer together. The
weight is much the more likely driver (the dose-response in §0 is monotone in
`grad_ratio` across both normalizers), but `log_classes` + a correct weight has
never been run.

### `--fd_gmm_ema_beta`

`OnlineClassStats` decays per class by `beta ** counts`, only for classes present
in the batch, so the horizon is 1000 *appearances of that class*, not 1000 total
samples. What changes with class count is the **wall-clock span** those
appearances cover, at global batch 96:

| classes | 1000 appearances ≈ |
|---|---|
| 20 | 208 steps |
| 100 | 1,040 steps |
| 1000 | 10,400 steps |

`docs/gmm_uncond_20class.md` §6 recommended lowering toward 0.99 at 1000 classes
so the anti-collapse repulsion is not computed against a `q` describing the
generator from 10k steps ago. **1000c v1 used 0.99; v2 reverted to 0.999** on the
grounds that it puts the spread noise floor 10× lower and makes the gate metric
readable. Both died, so this is not the binding cause and the conflict is
unresolved. At 100 classes the span is 1,040 steps and 0.999 is fine.

### `--fd_gmm_cls_cap`

`-0.69` throughout. Not implicated: `gmm_cls_sat_frac` was **0.0000** for the
entire length of every failed run, so the cap never withheld gradient from
anything. It only began to bite in 100c #2 (0.198 by the end), which is the
intended behaviour — samples reaching real-data class confidence.

### `lr_sched`

20c and 1000c v1 used `cosine`; v2 onward uses `constant`, so an end-of-run
readout is not confounded by a decaying LR. Not implicated in any result.

### The takeoff gate — added, then re-clocked

Introduced in 1000c v2 to stop paying for dead runs. Its criteria are sound and
class-count-normalised; its **clock** was wrong.

* **v2 gated on absolute step 10,000.** At 1000 classes that is 960
  samples/class. Reconstructing the metric on the successful 20-class run shows
  it read 0.71 at the same per-class budget and would have failed its own gate
  even harder. The gate was asking a question with no information in it.
* **Re-expressed in samples/class** in `scripts/run_jit_uncond_gmm_100class.sh`:

  ```
  gate_step = TAKEOFF_SAMPLES_PER_CLASS * num_classes / global_batch
  ```

  Derived in the launcher, so no training-code change was needed, and the
  existing `validate_gmm_takeoff_gate_args` check turns a too-short run into a
  **launch-time refusal** rather than a mid-run abort. Verified: a 12,500-step
  run refuses with the per-class arithmetic in the error message.
* Default target **35,000 samples/class** — the 20-class pilot first passed both
  criteria at 27,552, so this carries a 1.27× margin.

**The cosine criterion is not specific.** 100c #1 passed it (-0.130) on a
completely dead run: "the GMM gradient has become an independent force" and "the
GMM gradient is fighting FD and losing" have the same sign. Only `spread` caught
that run. `logic=any` is doing real work; do not weaken it.

---

## 8. Every run

All against `checkpoints/base/JiT-B-uncond.pth`, global batch 96, arm C
(`log p` + `log q`) unless noted. FID is the best EMA at the best step, against
each run's own subset reference — **not comparable across rows with different
class counts.**

| run | date | C | steps | `w` | `T` | cls_norm | `beta` | sust. `grad_ratio` | `probe_top1` | `probe_rank` | `spread` | `within` | best FID |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20c arm C | 08-11 | 20 | 50,000 | 0.0066 | 100 | self | 0.999 | 0.250 | **0.9271** | **1.3** | 1.0351 | 0.905 | 4.66 |
| 20c arm B | 08-12 | 20 | 50,000 | 0.0066 | 100 | self | 0.999 | 0.045 | 0.0521 | 455.3 | 0.0035 | 1.015 | 4.87 |
| 1000c v1 | 08-12 | 1000 | 50,000 | 0.0038 | 10 | self | 0.99 | 0.088 | 0.0000 | 497.8 | 0.0082 | 2.027 | 16.40 |
| 1000c v2 | 08-13 | 1000 | 10,000† | 0.0050 | 10 | log_classes | 0.999 | 0.215 | 0.0000 | 514.1 | 0.0018 | 2.038 | 26.52 |
| 100c #1 | 08-13 | 100 | 36,458† | 0.0050 | 45 | log_classes | 0.999 | 0.140 | 0.0104 | 494.3 | 0.0008 | 1.987 | 11.92 |
| **100c #2** | 08-17 | 100 | **50,000** | **0.0103** | 45 | **self** | 0.999 | **0.248** | **0.1979** | **313.6** | **0.1444** | 1.680 | 14.94 |

† aborted by the takeoff gate.

### Gate verdicts

```
1000c v2  ABORT  step 10000 : spread/noise=1.044 (<2)  cos=+0.076 (>0)   both failed
100c #1   ABORT  step 36458 : spread/noise=1.293 (<2)  cos=-0.130 ( ok)  spread failed
100c #2   PASS   step 36458 : spread/noise=12.89       cos=-0.128        both passed
```

### The failure signature shared by every dead run

`gmm_nll_p` — the quantity being minimised — **rose monotonically, past uniform,
and never came back:**

| run | start | end | uniform `ln C` |
|---|---|---|---|
| 1000c v1 | 9.27 | 14.20 | 6.91 |
| 100c #1 | 5.00 | 6.47 | 4.61 |
| **100c #2** | 4.92 | peak **6.09** @ 36,960 → **4.41** | 4.61 |

Mechanism: FD makes images realistic but *unconditional*, so each sample lands
near *some* class mean and not its assigned one, and the Mahalanobis distance to
the assigned mean grows. At init the images are garbage sitting near the global
mean, roughly equidistant from all class means, so `nll_p ≈ ln C`.

100c #2 is the only run where this reversed — it peaked just after the gate and
then fell through uniform. **That reversal is the signature of the term actually
working**, and it is the single most diagnostic number in the whole project.

### The samples-per-class clock

`samples/class = step × global_batch / num_classes`. Takeoff onset, measured as
`spread_to_noise` first crossing 2.0:

| run | onset |
|---|---|
| 20c arm C | ~15,800 samples/class |
| 100c #2 | 20,409 samples/class |

Within ~30% — **the clock roughly holds once the weight is right.** It appeared
refuted by 100c #1 only because that run was under-driven and never took off at
all, so there was no onset to time. What *is* genuinely slower at 100 classes is
the climb *after* onset, not its onset.

Note the pilot needed **240,000** samples/class (its full 50,000 steps at 20
classes) to reach `probe_top1` 0.93. 100c #2 got 48,000.

---

## 9. The latest run in detail — `jitB_uncond_gmm100_armC_v2_self_w0103`

100 classes (stride-10), 50,000 steps, 11 h 48 m on 3× A100 at batch 32/GPU,
`w=0.0103`, `T=45`, `cls_norm=self`. Completed; gate passed at 36,458.

```
  step  smp/cls   s/noise    spread      cos    gmm_t1   prb_t1  prb_rank   cond_d   within    nll_p      sat
     0        0    0.9052    0.0007   0.0000    0.0104   0.0000     523.0   0.0000   0.5934   4.9179   0.0000
 10000     9600    1.3033    0.0007  -0.0042    0.0104   0.0000     487.9   0.0718   1.6757   5.1622   0.0000
 20000    19200    1.7841    0.0010  -0.0877    0.0104   0.0000     478.2   0.1311   1.7013   5.4804   0.0000
 30000    28800    3.8968    0.0023  -0.1229    0.0208   0.0104     477.0   0.2032   1.7999   5.6094   0.0000
 35000    33600    8.3366    0.0050  -0.1307    0.0312   0.0104     454.6   0.2415   1.8076   5.6870   0.0104
 40000    38400   35.5032    0.0214  -0.1511    0.0938   0.0417     434.3   0.2956   1.8167   5.4142   0.0521
 45000    43200  139.7882    0.0816  -0.1538    0.1354   0.0938     411.0   0.3365   1.7693   5.1862   0.1042
 49999    47999  259.7053    0.1444  -0.1363    0.2292   0.1979     313.6   0.3831   1.6799   4.4126   0.1979
```

### What worked

* **`probe_top1` = 0.198 against a chance of 0.01 — 20× chance.** This is the
  held-out ResNet-50, the only honest arbiter. `probe_top5` 0.250,
  `probe_rank` 523 → 314 (chance 500).
* `gmm_top1` (0.229) and `probe_top1` (0.198) move **together**. Divergence
  between them is the documented signature of the GMM being gamed; there is none.
* `spread` 0.0007 → 0.1444, a 206× rise, and `spread_to_noise` 0.91 → 259.7
  against a gate threshold of 2.0.
* `cond_delta` 0.000 → 0.383 — the class token genuinely changes the image.
* `gmm_cls_sat_frac` 0 → 0.198: a fifth of samples are now as class-confident as
  real images, so the cap has started doing its job.
* `grad_ratio_p_fd` held **0.23–0.27** for the entire run — the calibration was
  correct and stable, exactly as designed.

### Versus the pilot at matched budget (48,000 samples/class)

| | spread | gmm_t1 | probe_t1 | probe_rank | cond_d | within |
|---|---|---|---|---|---|---|
| 20c pilot | 0.370 | 0.375 | 0.323 | 227 | 0.409 | 0.968 |
| **100c #2** | 0.144 | 0.229 | 0.198 | 314 | 0.383 | 1.680 |
| 100c #1 | 0.0008 | 0.010 | 0.010 | 493 | 0.115 | 1.975 |

40–60% of the pilot's progress at the same per-class budget. The previous 100c
run was at 0.2%.

### What it cost, and what is unfinished

* **FID regressed.** Bottomed at 14.94 (step 40,000), then 15.08 → 15.93 — a
  +6.6% give-back beginning exactly where conditioning took off. The pilot's FID
  fell monotonically the whole way with no regression at all. For scale, the
  classifier-ensemble baseline *doubled* FID (10.68 → 21.49), so 6.6% is mild —
  but it is not zero, and this is the tradeoff `log q` exists to prevent.
* **`within` = 1.680** against an ideal of 1.0 (`within_p` = 0.402 for this fit,
  so a fully de-conditioned generator reads 2.49). Falling, and read jointly with
  a rising `spread` that is the "classes learned, diversity kept" quadrant — not
  collapse, which would be < 0.8. But well behind the pilot's 0.968: the
  conditionals are still ~68% too wide.
* **It ended mid-takeoff.** Growth per 5,000 steps: 2.14× → 4.31× → 3.81× →
  1.77×, with `probe_top1` still doubling in the final window (0.094 → 0.198).
  Nothing had plateaued.

---

## 10. How to read the meters

From `training_metrics.json` (one JSON object per line) or
`scripts/analyze_gmm_run.py`.

| meter | null value | target | notes |
|---|---|---|---|
| `probe_top1` / `probe_rank` | chance = `1/C` / 500 | ↑ / → 1 | held-out ResNet-50. **The only honest arbiter.** `probe_rank` moves continuously where top-1 is quantised at `1/batch` |
| `gmm_top1` | `1/C` | → 1 | the in-loss classifier's own opinion. **Diverging upward from `probe_top1` is the signature of the GMM being gamed** |
| `grad_ratio_p_fd` | — | **0.22–0.30** | read *sustained*, not at the ramp. Outside it the run is uninterpretable |
| `gmm_nll_p` | `ln C` | ↓ below `ln C` | rising past uniform = realistic-but-unconditional. **The earliest reliable failure signal** |
| `gmm_class_mean_spread` | 0 | → ~1.0+ | between-class scatter of q over p |
| `gmm_class_mean_spread_to_noise` | 1.0 | ≥ 2 to gate | spread over its finite-EMA noise floor. Values near 1 are pure estimator noise |
| `gmm_within_trace_ratio` | `1/tr(within_p/k)` | → 1.0 | **only readable jointly with `spread`.** De-conditioned = 2.49 (100c) / 2.26 (1000c) / 1.16 (20c); < 0.8 = collapse |
| `cos_update_fd_p` | 0 | slightly < 0 | **ambiguous.** ≈ 0 = orthogonal/healthy; strongly negative can mean either "independent force" or "fighting FD and losing" |
| `cond_delta` | 0 | ↑ | pixel change from swapping the class token; 0 means still unconditional |
| `gmm_cls_sat_frac` | 0 | < 0.8 | fraction above the cap. → 1.0 means the term has run out of gradient |
| `gmm_clamp_frac` | ~0.02 | stable | if it climbs, the clamp is reshaping the loss instead of guarding it |
| `gmm_class_coverage` | 1.0 | 1.0 | must stay 1.0 |
| `gmm_class_mean_ema_n_eff` | — | → `1/(1-beta)` | Kish ESS of each per-class EMA mean |

The noise floor is reconstructible for runs that predate the metric:

```
floor = (1 - 1/C) * within_trace_ratio * tr(within_p) / n_eff / tr(between_p)
n_eff = Kish ESS = (1+b)/(1-b) * (1-b^n)/(1+b^n),  n = (bootstrap + step*batch)/C
```

Verified against the logged metric to within 5%.

---

## 11. Where things stand

### Settled

* The term produces real conditioning from a de-conditioned model at 100 classes.
* `grad_ratio_p_fd` ≈ 0.25 sustained is the operating point; ≤ 0.14 is dead.
* Temperature must be re-measured per fit; the procedure in §6.1 is reliable.
* The gate belongs on a samples-per-class clock; the spread criterion is
  load-bearing and the cosine criterion is not specific.
* No collapse risk observed at 0.25 — `within` fell monotonically and never
  approached the < 0.8 danger zone.

### Open

1. **100c #2 ended mid-takeoff.** Extending it is the highest-value next step —
   resume from the step-50,000 checkpoint for another 50–100k steps. Note
   `run_jit_uncond_gmm_100class.sh` forces `AUTO_RESUME=0` by design, so this
   needs `--resume_from` plus a higher `EPOCHS`.
2. **The `log q` term has never been validly ablated.** Arm B was run at an
   uncalibrated weight (0.045). A correctly-weighted `--fd_gmm_no_q` arm is the
   missing control, and without it none of the run-to-run movement can be
   attributed to the anti-collapse term specifically.
3. **`self` vs `log_classes` is confounded** with the weight change in 100c #2.
4. **1000 classes is untested at a correct weight.** Predicted takeoff onset
   ~20,000 samples/class ≈ 208,000 steps at batch 96. Expensive, and the
   precondition is the weight calibration, not the class count.
5. **`within` = 1.68 vs an ideal of 1.0.** If FID keeps climbing past ~17 while
   conditioning improves, rebalancing `lambda_ent` against `lambda_cls` is the
   lever that targets it directly — weight alone will not fix it.
6. **`--fd_gmm_ema_beta` 0.99 vs 0.999 at 1000 classes** remains unresolved
   (staleness vs. a readable noise floor).

### Reproducing the current best result

```bash
# 1. references (once per subset)
bash scripts/compute_class_gmm_100class.sh
bash scripts/compute_imagenet100_fd_stats.sh

# 2. temperature
CUDA_VISIBLE_DEVICES=4 python scripts/validate_class_gmm.py \
    --stats data/fid_stats/inception_in256_t256_classgmm_k128_c100.npz \
    --class_ids $(seq 0 10 990 | tr '\n' ' ') \
    --shrinkage 0.75 --temperature 25 30 35 40 45 50 60

# 3. weight
CALIBRATION=1 CLS_NORMALIZATION=self GMM_TEMP=45 WEIGHT=0.0050 \
  GPUS=2,3,4 BATCH_SIZE=32 EXP_NAME=<unique> bash scripts/run_jit_uncond_gmm_100class.sh

# 4. run
CLS_NORMALIZATION=self GMM_TEMP=45 WEIGHT=0.0103 \
  GPUS=2,3,4 BATCH_SIZE=32 EXP_NAME=<unique> bash scripts/run_jit_uncond_gmm_100class.sh
```

---

## 12. `cfg_delta` — the CFG-delta conditional mode (added 2026-08-22)

Implemented from [`CFG_DELTA_GMM_IMPLEMENTATION_SPEC.md`](../CFG_DELTA_GMM_IMPLEMENTATION_SPEC.md).
**An opt-in ablation.** `density` remains the default and the anti-collapse
baseline; nothing above changed.

### What it is

The KL chain rule splits the joint class-conditional KL into a marginal part
and a conditional part:

```
KL(q(x,c) || p(x,c)) = KL(q(x) || p(x)) + E_x KL(q(c|x) || p(c|x))
```

Hand the marginal to FD and only the conditional residual is left. Its pathwise
gradient is a *vector field* in feature space, which by Bayes is a difference of
classifier-free-guidance deltas:

```
g_cfg(z,c) = [s_q(z|c) - s_q(z)] - [s_p(z|c) - s_p(z)]
           =  grad_z log q(c|z)  -  grad_z log p(c|z)
           =  Delta_q            -  Delta_p          # fake CFG delta - teacher CFG delta
```

Descent moves samples along `Delta_p - Delta_q`: teach the generator's current
conditioning effect to match the real data's. Both deltas are computed
analytically (§12.2), evaluated at a **detached** `z`, and injected through the
stop-gradient linear surrogate `mean_i z_i . stopgrad(g_i)`, whose gradient is
exactly `g_cfg / B`.

Relative to density mode:

```
g_cfg = g_density - [s_q(z) - s_p(z)]
```

i.e. cfg_delta is density mode with the *marginal* score mismatch subtracted out
and delegated to FD. `tests/test_gmm_cfg_delta.py` verifies this identity
numerically — it is what pins the sign convention.

### 12.1 Four things it is *not*

These are the distinctions that make it an ablation rather than an upgrade.

1. **FD is not `KL(q(x)||p(x))`.** It constrains pooled `mu` and `sigma` only, so
   the marginal terms do not actually cancel. The whole objective is a hybrid
   surrogate, not the exact joint KL.
2. **`q_hat` is not `q*`.** The derivation needs the generator's true posterior;
   the code has a fitted online GMM. The injected direction inherits that model
   bias. What the vector form buys is that nothing pretends to be a KL and no
   gradient leaks into the fitted parameters — unlike the old sampled-label
   scalar `log q(c_i|z_i) - log p(c_i|z_i)`, which is unbounded below and
   rewards samples its own `q` misclassifies. **Do not reintroduce that.**
3. **`T > 1` changes the objective.** The Bayes identity is exact only at `T=1`;
   above it the implemented field is the *tempered* posterior gradient
   `(1/T)[s_c - sum_j r_T,j s_j]`. Necessary in practice (untempered
   high-dimensional posteriors saturate to zero gradient) but it means the
   weight must be recalibrated whenever `T` moves. Note density mode is
   *unaffected* by temperature; cfg_delta is directly affected.
4. **No guaranteed anti-collapse repulsion.** Density mode's repulsion grows as
   `Sigma^q` shrinks — strongest exactly under collapse. Nothing in `g_cfg` does
   that. Read `gmm_within_trace_ratio` and FID, not just the CFG meters.

It is also *not* the existing `posterior` mode: that differentiates a
non-negative scalar KL through the full posterior vector, this injects only the
sampled label's posterior score difference.

### 12.2 The analytic formulas (no autograd in the train step)

Real side is QDA (per-class precision), so the mixture score keeps its
per-class precision:

```
s_{p,j}(z) = P_j mu_j - P_j z
Delta_p    = (1/T) [ s_{p,c}(z) - sum_j r^p_{T,j}(z) s_{p,j}(z) ]
```

Generated side is LDA (one tied precision), so the class-independent `-P z` half
cancels between the selected class and the posterior average and only natural
parameters survive:

```
Delta_q    = (1/T) [ P mu_c - sum_j r^q_{T,j}(z) P mu_j ]
```

The QDA average is computed as `(probs @ prec_flat).view(B,k,k)` followed by a
`bmm` — two matmuls, intermediate `(B,k,k)`, never the `(B,C,k)` tensor a naive
per-class difference would build. (The spec suggested an `einsum`; this is the
same value at lower memory, and matches how `logits` already contracts against
`prec_flat`.)

Verified against `autograd.grad(log_softmax(logits))` at `T=1` and `T>1`:
**max absolute error 4.77e-07** on both sides, i.e. float32 round-off.

### 12.3 Two experiments, and why they are different objectives

| | `--fd_gmm_lambda_cls` | what it is |
|---|---|---|
| **PURE** | `0` | the actual chain-rule derivation. The cleanest test, and the one that may simply fail to move a fully de-conditioned generator |
| **PRACTICAL** | `1.0` | CFG-delta *plus* an explicit `-log p(c|z)` driver |

They must never be reported as the same objective. `Delta_p` already contains
`-grad log p(c|z)`, so the practical arm emphasises class fidelity **twice**
(spec §16, failure D). The startup log says which one is running:

```
[GMM] mode=cfg_delta [PURE: no explicit class driver] ...
[GMM] mode=cfg_delta [PRACTICAL: explicit -log p(c|z) retained] ...
```

### 12.4 Calibration — a density weight does NOT transfer

Density mode's `q` term is self-normalised (`term/(|term|+0.01)`), so its
gradient scale is set by that divisor. The cfg_delta surrogate is **not**
self-normalised and must not be: its scalar value is origin-dependent, so
dividing by it would tie the gradient to an arbitrary coordinate choice.
The raw field is injected as-is.

Consequence: recalibrate `WEIGHT` on sustained `grad_ratio_p_fd` from scratch.
A useful shortcut for the *seed* weight — both modes reach the image through the
same frozen judge, so the ratio of their whitened-feature gradient norms is the
ratio of their image-space norms, and can be measured offline in seconds against
the real reference plus a synthetic de-conditioned `q`.

Measured that way at `T=45`, `k=128`, `B=96` on the 100-class fit:

| | `||grad_z||` |
|---|---|
| density `q` term (self-normalised) | 2.57e-2 |
| cfg_delta field (raw) | 5.79e-2 |
| `-log p(c\|z)` driver (self-normalised) | 1.14e-2 |

At the de-conditioned start `Delta_q` is essentially zero (`fake_rms` 1.2e-3 vs
`teacher_rms` 5.0e-2): `q`'s class means all sit on the pooled mean, so the
field *is* the teacher delta and nothing else. Conditioning only becomes a
two-sided match once `q`'s class means separate.

### 12.5 New diagnostics

| meter | meaning | reading |
|---|---|---|
| `gmm_cfg_teacher_rms` | RMS of `Delta_p` | finite; scales as `1/T` |
| `gmm_cfg_fake_rms` | RMS of `Delta_q` | ~0 at a de-conditioned start; must rise |
| `gmm_cfg_error_rms` | RMS of `Delta_q - Delta_p` | should fall if matching works |
| `gmm_cfg_relative_error` | error / teacher | comparable across temperatures |
| `gmm_cfg_alignment_cos` | cosine(fake, teacher) | → 1 |
| `gmm_cfg_vector_scale` | applied RMS normaliser | 1.0 unless `rms` |
| `gmm_cfg_surrogate` | the injected scalar | **not an objective.** Origin-dependent; never compare across runs |

Interpretation warnings: falling CFG error with a flat held-out `probe_top1`
means the fitted GMM is being gamed, not satisfied; `gmm_top1` running well
above `probe_top1` is the same warning. A class-conditioned generator can still
collapse *within* each class, so read these together with
`gmm_within_trace_ratio` and FID.
