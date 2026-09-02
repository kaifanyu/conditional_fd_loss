# Teaching classification to an unconditional JiT-B with `log p` / `log q`

Experiment record for the 20-class pilot. Started 2026-08-11.

Companion to [`gmm_posterior_loss.md`](gmm_posterior_loss.md), which derives the
term and records the pMF-B/1000-class result where it did **not** pay off. This
document covers the setting the same doc named as the right next step: *"find a
configuration that does collapse, and re-run this comparison there."*

---

## 1. Why this setting, and not the previous one

The pMF-B run's verdict was that FD post-training raises within-class diversity
on its own, so the anti-collapse term had nothing to fix. Starting from a
**de-conditioned** model reverses that, for a structural reason: the class
signal has to come from `E[-log p(c|x)]`, whose per-sample minimiser is a point.
The contractive term is no longer a 0.1-weighted regulariser, it is the driver.

The cost is already measured, on this exact checkpoint (`JiT-B-uncond.pth`,
1000 classes, 100k steps, 3 GPUs):

| run | FID @100k | held-out top-1 | s/iter |
|---|---|---|---|
| [`JiT_uncond_baseline`](../work_dirs/JiT_uncond_baseline/JiT_B-fd-sim-uncond/eval_summary.csv) — FD-SIM only | **10.68** | — | 4.8 |
| [`JiT_uncond_baseline_logp_multiple`](../work_dirs/JiT_uncond_baseline_logp_multiple/JiT_B-fd-sim-uncond/eval_summary.csv) — + 3-classifier `-log p` | **21.49** | 0.635 | 14.2 |

Teaching classification doubled FID. That doubling is the thing `log q` exists
to counteract.

Two further reasons to prefer a GMM over the classifier ensemble here:

* **It is nearly free.** The judge features already carry gradient for the FD
  term; the posterior is one `(B,k)` projection plus two matmuls. The ensemble
  needs three extra frozen-backbone forward+backward passes per step — the
  4.8 → 14.2 s/iter difference above.
* **It is a generative classifier.** Raising `log p(c|x)` requires moving toward
  `mu_c^p` in a Mahalanobis sense; there is no direction that increases a
  Gaussian's density without moving toward its mode. That removes one class of
  adversarial solution and concentrates the risk on the one `log q` handles.

## 2. Setup

```
L = sum_j FD_j / (FD_j.detach() + eps)                     # siglip + mae + inception
  + w(s) * [ lambda_cls * E[-log p(c|x)]                   # class fidelity  (driver)
           + lambda_ent * E[log q(x|c) - log p(x|c)] ]     # anti-collapse
```

| | |
|---|---|
| base model | `checkpoints/base/JiT-B-uncond.pth` (`make_uncond_jit.py` on JiT-B) |
| classes | the 20-class diagnostic set (`0 9 88 130 207 279 281 340 360 387 404 417 444 555 569 817 920 949 974 979`) |
| FD judges | SigLIP-SO400M, MAE ViT-L, InceptionV3, all vs `data/fid_stats/imagenet20_v1/` |
| GMM judge | inception, whitened PCA k=128, `p_shrink=0.75`, `q_shrink=0.25` |
| GMM stats | `data/fid_stats/inception_in256_t256_classgmm_k128_c20.npz` |
| GPUs | 3,4 · batch 48/GPU · 50,000 steps · lr 1e-5 cosine |
| FD window | `--fd_ema_beta 0.99` (~100 steps; 0.999 was shown blind to mode collapse) |

Entry points: [`conditional_main_fd_gmm.py`](../conditional_main_fd_gmm.py),
[`scripts/run_jit_uncond_gmm_20class.sh`](../scripts/run_jit_uncond_gmm_20class.sh).

### The `p` side must be refitted on the subset, not masked

`compute_class_stats.py --class_ids` fits the whitening PCA, the pooled
within-class covariance and the posterior denominator over the 20 classes only.
Masking the 1000-class file instead would make `-log p(c|x)` a 1000-way problem
against 980 classes the generator never draws, and would leave those classes'
`mu_c^q` parked on `mu_c^p` as spurious attractors in the softmax.

Measured on the subset fit: `tr(within)/k = 0.859`, `tr(between)/k = 0.142`
(versus 0.442 / 0.558 for the 1000-class fit — with only 20 class means to
separate, most of the whitened variance is within-class).

### Arms

| arm | flag | question |
|---|---|---|
| A | `--fd_gmm_weight 0` | control: identical code path, all meters, zero gradient. Does conditioning appear without the term? What does FD-only do to the meters? |
| B | `--fd_gmm_no_q` | `-log p` only — the cheap drop-in for the classifier ensemble. Does it reach 63.5% top-1 at 1/3 the wall clock, and does FID still double? |
| C | *(default)* | the proposal. Does `within_trace_ratio` hold instead of falling, and does FID recover at matched top-1? |

`ARM=A|B|C bash scripts/run_jit_uncond_gmm_20class.sh`. **C was launched first**
(2026-08-11); A and B are one flag away and must be run before any of C's
movement is attributed to the term — the pMF-B write-up is explicit that every
pre-control reading of "the term is working" was the confound doing the work.

## 3. Calibration measured before launch

### The posterior saturates at T=1 — this was the one real blocker

`scripts/validate_class_gmm.py` on 1000 held-out val images of the 20 classes:

| `p_shrink` | top-1 | top-5 | `-log p(c\|x)` | ideal-generator postKL |
|---|---|---|---|---|
| 0.00 | 98.2% | 99.5% | 4.153 | 3.565 |
| 0.25 | 98.8% | 99.6% | 1.616 | 1.028 |
| 0.50 | 98.7% | 99.7% | 1.393 | 0.805 |
| **0.75** | **98.6%** | **99.6%** | **1.384** | **0.796** |
| 0.90 | 98.7% | 99.5% | 1.500 | 0.912 |
| 1.00 | 98.7% | 99.5% | 1.811 | 1.223 |

`0.75` wins again, matching the 1000-class result. But the mean `-log p` of
1.384 is misleading: the **median is exactly 0.000**, and **98.6% of real images
sit at exactly `log p = 0`**. The mean is entirely carried by the 1.4% of
catastrophic misses. In other words, at T=1 the class-fidelity term has *no
gradient at all* on anything that already looks like a real image — the failure
mode §9 of `gmm_posterior_loss.md` calls "the term needs residual class
ambiguity to bite."

A 20-way posterior over 128-dimensional Mahalanobis distances is simply too
sharp. The fix is a softmax temperature, exactly analogous to the existing
`--clip_logit_scale`. It changes no ranking, so top-1 is identical at every T:

| T | median `log p` | fraction saturated | fraction above the -0.69 cap |
|---|---|---|---|
| 1 | 0.000 | 98.6% | 98.6% |
| 50 | -0.010 | 49.4% | 97.2% |
| **100** | **-0.312** | **0.0%** | **79.6%** |
| 200 | -1.258 | 0.0% | 1.7% |

**`--fd_gmm_temp 100`** is the operating point: real images land just *above*
`--fd_gmm_cls_cap -0.69`, so the term stops pushing exactly when a sample is as
class-confident as a real one. T=200 would keep pushing past real-data
confidence, which is the adversarial regime the cap exists to prevent.

The density-ratio term reads `log_likelihood`, not the softmax, so it is
unaffected by any of this.

### The cap

`gmm_posterior_loss` had no per-sample confidence cap — correct when
`lambda_cls=0.1` is a regulariser, wrong when it is the driver. Every
classifier-ensemble run in this repo uses `cond_target_logp=-0.69` for the same
reason. Added as `--fd_gmm_cls_cap`, verified by
`test_cls_cap_stops_gradient_on_confident_samples`: samples above the cap
contribute exactly zero class-fidelity gradient, samples below are unchanged.

Note the cap also bounds the self-normalisation escalation. `_self_normalize`
divides by `|l_cls| + 0.01`, so an uncapped `l_cls` falling toward 0 would
amplify the effective weight without limit; with the cap, `l_cls >= 0.69` and
the amplification is bounded at ~1.4x. `--fd_gmm_no_normalize_cls` removes it
entirely if that is still unwanted.

### Weight

Calibrated on `grad_ratio_p_fd` (image-space gradient of the GMM term over the
FD term's), never on the loss value — the two terms are both self-normalised at
incomparable scales. Two reference points bracket the usable band: the
classifier-ensemble run that reached 63.5% held-out top-1 sat at **0.18**, and
r9's **0.34** on this same 20-class diagnostic is recorded as the regime that
collapsed. Target 0.10–0.40, read at the first fully-ramped print.

The first launch at `--fd_gmm_weight 0.02` measured **0.61**, stable over steps
500–640 — hotter than the collapse regime. Killed at step 640 and relaunched at
`0.02 * 0.20 / 0.61 = 0.0066`, which measures **0.26–0.29** at steps 680–720:
inside the band and clear of 0.34. (The ratio is not exactly linear in the
weight — the FD gradient moves too — so one iteration of this loop is normal.)
The calibration log is kept at
`sweep_logs/jitB_uncond_gmm20_armC_CALIBRATION_w0.02.out`; 0.0066 is now the
script default. **Recalibrate for arm B** — dropping the `q` term removes half
the loss, so the same weight will not give the same ratio.

### Cost

0.94 s/iter at batch 48/GPU on 2 A100s (46.7 GB peak), i.e. ~13 h for 50,000
steps. For scale, the 1000-class classifier-ensemble run was 14.2 s/iter on 3
GPUs.

## 4. What to watch

Ordered by what would kill the run first. `scripts/analyze_gmm_run.py` prints
all of these for one or several arms.

| meter | at init | target | reading |
|---|---|---|---|
| `grad_ratio_p_fd` | — | 0.10–0.40 | outside the band, relaunch; the run is otherwise uninterpretable |
| `probe_top1` | 0.00 | > 0.05 (chance), ideally ≫ | held-out ResNet-50, **the only honest arbiter** |
| `probe_rank` | ~500 | → 1 | moves continuously where top-1 is quantised at 0 |
| `gmm_top1` | ~0.05 | → 1.0 | the in-loss classifier's own opinion. **Diverging from `probe_top1` is the signature of the GMM being gamed** |
| `cond_delta` | 0.0003 | rising | pixel change from swapping the class token; 0 means still unconditional |
| `gmm_class_mean_spread` | ~0.03 | → ~1.16 | between-class scatter of q over p. The continuous measure of learned class structure |
| `gmm_within_trace_ratio` | ~0.86 | → ~0.98 | within-class scatter of q over p. **Only readable jointly with `spread` — see below** |
| `gmm_cls_sat_frac` | 0.00 | < 0.8 | fraction above the cap. Approaching 1.0 means the term has run out of gradient |
| `gmm_clamp_frac` | ~0.00 | ~0.02 | if it climbs, the density clamp is reshaping the loss instead of guarding it |
| real FID | — | vs 10.68 / 21.49 | the cost |

### `within_trace_ratio` is not readable on its own here

This is the one meter whose interpretation does **not** carry over from the
conditional-model setting, and getting it wrong would be easy.

The whitening makes the real pooled covariance the identity, and pooled =
within + between. For the 20-class fit that splits as `0.859 + 0.142 = 1.0`.
Meanwhile the FD term is actively driving the generator's *pooled* covariance
toward the same identity. So:

| generator state | between (`spread`) | within | `within_trace_ratio` |
|---|---|---|---|
| de-conditioned, FD converged | 0 | 1.0 | **1.16** |
| classes learned, diversity kept | 0.142 | 0.859 | **1.00** |
| classes learned by collapsing them | 0.142 | ≪ 0.859 | **< 1.0** |

Two very different outcomes — "learned nothing" and "learned classes by
collapsing them" — can produce the same reading. **`within_trace_ratio` must be
read jointly with `class_mean_spread`**, which is 0 for an unconditional
generator and ~1.16 for an ideal one (the floor is above 1.0 because q's class
means come from ~50 samples each, which inflates the between-class scatter;
measured on an ideal generator by `validate_class_gmm.py`).

The three readings that matter:

* `spread → ~1.16` **and** `within → ~0.98`: classes learned, diversity kept.
  This is the result the term is claimed to produce.
* `spread → ~1.16` **and** `within` falling well below: classes learned by
  collapsing them. This is the failure arm C exists to prevent, and the one
  arm B is expected to show.
* `spread` flat near 0: no conditioning learned at all, whatever `within` does.

Note the *starting* value is below 1.16 (the smoke run read 0.86) simply because
FD has not converged yet at step 0 — this checkpoint begins with a train-time
inception FD near 300 against the 20-class reference, i.e. its pooled covariance
is nowhere near the identity. Read the trend, jointly, not the level.

## 5. Reading the results

```bash
# one arm
python scripts/analyze_gmm_run.py work_dirs/JiT_uncond_gmm_20class/jitB_uncond_gmm20_armC_logp_logq

# matched comparison once A and B have run
python scripts/analyze_gmm_run.py work_dirs/JiT_uncond_gmm_20class/* --window 5000 --csv gmm20.csv
```

It prints, per arm, every meter in §4 averaged over the first and last `window`
steps plus the delta, and the real FID trajectory from `eval_summary.csv`; with
several arms it adds a side-by-side table of the final values. Raw per-step data
stays in each run's `training_metrics.json` (one JSON object per line).

Live: `tail -f sweep_logs/jitB_uncond_gmm20_armC_logp_logq.out`.

### Log

| date | event |
|---|---|
| 2026-08-11 07:54 | pipeline smoke test passes end to end; 13/13 unit tests pass |
| 2026-08-11 07:56 | arm C launched at `--fd_gmm_weight 0.02` |
| 2026-08-11 08:09 | `grad_ratio_p_fd` = 0.61 at step 640 — too hot. Killed. |
| 2026-08-11 08:12 | arm C relaunched at `--fd_gmm_weight 0.0066` |
| 2026-08-11 08:34 | `grad_ratio_p_fd` = 0.26–0.29 at steps 680–720. In band; run left to complete |
| 2026-08-11 21:45 | arm C complete: 50,000 steps in 13 h 32 m. `probe_top1` 0.928, FID 4.66 |
| 2026-08-11 22:0x | 1000-class temperature re-measured (§6): T=100 does **not** transfer, T≈10 is the equivalent point |

### Results — arm C, complete (50,000 steps, 13 h 32 m on 2 A100s)

| step | `probe_top1` | `probe_rank` | `gmm_top1` | `spread` | `within` | `mean_mse` | `cond_delta` | `sat_frac` | `grad p/fd` | real FID |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.000 | 620 | 0.042 | 0.002 | 0.579 | 1.694 | 0.0000 | 0.00 | — | — |
| 2,500 | 0.010 | 373 | 0.052 | 0.005 | 1.086 | 0.324 | 0.033 | 0.00 | 0.28 | — |
| 6,250 | — | — | — | — | — | — | — | — | — | 79.49 |
| 10,000 | 0.323 | 227 | 0.375 | 0.370 | 0.968 | 0.133 | 0.409 | 0.29 | 0.25 | — |
| 12,500 | — | — | — | — | — | — | — | — | — | 16.90 |
| 15,000 | 0.719 | 58.0 | 0.792 | 0.818 | **0.820** | 0.052 | 0.506 | 0.66 | 0.46 | — |
| 18,750 | — | — | — | — | — | — | — | — | — | 11.07 |
| 20,000 | 0.885 | 2.01 | 0.969 | 1.052 | 0.870 | 0.019 | 0.526 | 0.79 | 0.33 | — |
| 25,000 | 0.906 | 1.68 | 0.979 | 1.038 | 0.890 | 0.013 | 0.514 | 0.83 | 0.26 | 8.06 |
| 31,250 | 0.896 | 1.22 | 0.990 | 1.031 | 0.910 | 0.011 | 0.509 | 0.87 | 0.27 | 6.42 |
| 37,500 | 0.927 | 1.24 | 1.000 | 1.032 | 0.902 | 0.009 | 0.519 | 0.90 | 0.24 | 5.92 |
| 43,750 | 0.938 | 1.20 | 0.990 | 1.038 | 0.914 | 0.009 | 0.516 | 0.84 | 0.23 | 5.08 |
| **50,000** | **0.928** | **1.24** | **0.990** | **1.036** | **0.914** | **0.008** | **0.515** | **0.85** | **0.23** | **4.66** |

Last row is the mean over the final 5,000 steps (251 logged points); FID is the
best EMA at that step. `probe_top5` ends at **0.989**.

FID is 20k images at cfg 3.0, best EMA (`edm_1000`), against
`data/fid_stats/imagenet20_v1/inception.npz`. **It is not comparable to the
10.68 / 21.49 numbers in §1** — those are 50k images against the 1000-class ADM
reference.

FID never regressed at any eval: 79.5 → 16.9 → 11.1 → 8.06 → 6.42 → 5.92 → 5.08
→ 4.66, and was **still falling when the cosine schedule ran out**. Longer
training is leaving something on the table.

Observations, none of them yet attributable to the `q` term:

* **The conditioning works.** Held-out ResNet-50 top-1 goes 0.000 → 0.927 (chance
  0.05) and `probe_rank` 620 → 1.24 out of 1000. `cond_delta` 0.0002 → 0.52. The
  de-conditioned label table has been fully re-carved.
* **FID falls monotonically the whole way** while that happens — 79.5 → 6.00,
  still falling at 37.5k. There is no realism-vs-conditioning tradeoff visible in
  this arm. `cos_update_fd_p` sits at -0.02 to -0.04, i.e. the two gradients are
  essentially orthogonal rather than fighting.
* **`within` dips and recovers.** It bottoms at 0.820 around step 15k — exactly
  the window where classification is being learned fastest (`probe_top1`
  0.32 → 0.72 → 0.89 over 10k–20k) — then climbs back to 0.90. This is precisely
  the dynamic the `q` term is designed to produce. **Arm B is the only way to
  know whether it would have recovered without it.**
* **The class-fidelity term has largely switched itself off.** `sat_frac` = 0.90,
  so 90% of samples are above the cap and contribute no `l_cls` gradient. Yet
  `grad_ratio_p_fd` is still 0.24 — from step ~20k onward the GMM's contribution
  is mostly the density-ratio (`q`) half. That is the intended handover.
* **Residual diversity deficit.** `spread` 1.03 and `within` 0.90 against the
  ideal-generator floors of 1.16 and 0.98 measured in §3, i.e. ~8–11% short of
  real class structure. Not collapse, but not matched either.
* **Watch the `gmm_top1` / `probe_top1` gap.** The in-loss classifier is at 1.000
  while the held-out probe has been flat at 0.90–0.93 since step 25k. A 7-point
  gap is not alarming on its own (the GMM only reaches 98.8% on *real* images,
  and the probe is a different architecture at a different resolution), but the
  direction is the one that would signal gaming, so it is the thing to re-read at
  50k and to compare against arm B.

## 6. Does this transfer to 1000 classes? Measured, not assumed

The pilot's hyperparameters were calibrated *against the 20-class fit*, and the
same measurement re-run on the existing 1000-class GMM
(`inception_in256_t256_classgmm_k128.npz`, `p_shrink=0.75`, 25,000 held-out val
images) shows they do **not** carry over. Top-1 is 0.766 at every temperature —
only the confidence scale moves:

| T | mean `log p` | median | saturated | `> -0.69` | `> -2.30` |
|---|---|---|---|---|---|
| 1 | -6.886 | 0.000 | 71.5% | 76.6% | 78.9% |
| 5 | -1.520 | -0.014 | 48.1% | 75.5% | 84.2% |
| **10** | **-1.029** | **-0.179** | **17.1%** | **70.4%** | **86.8%** |
| 20 | -1.511 | -1.134 | 1.2% | 26.7% | 82.0% |
| 100 | -5.349 | -5.411 | 0.0% | 0.2% | 0.5% |

**`--fd_gmm_temp 100` — the pilot's value — is catastrophic at 1000 classes.**
It puts the median real image at -5.41 against a uniform of -6.91, i.e. the term
would push every sample, including already-correct ones, straight into the
adversarial regime. The equivalent operating point is **T ≈ 10** (median -0.18,
70% above the -0.69 cap, matching the 20-class T=100 figure of 79.6%).

Note also the shape difference: at T=1 the 1000-class posterior is *bimodal* —
71.5% saturated at exactly 0 and the remaining ~23% near uniform, which is why
the mean (-6.886) is barely better than uniform (-6.908) despite 76.6% top-1.
A temperature is needed at 1000 classes too, just a much smaller one.

### The other thing that changes: `q` staleness

`OnlineClassStats` decays per class by `beta ** counts`, only for classes present
in the batch, so the EMA horizon is 1000 *appearances of that class* at
`beta=0.999` — not 1000 total samples. Per-class sample counts are therefore
fine at 1000 classes. What changes is **how much wall-clock those appearances
span**: at 20 classes and global batch 96, 1000 appearances ≈ 210 steps; at 1000
classes it is ≈ 10,400 steps. The anti-collapse repulsion would be computed
against a `q` describing the generator from 10k steps ago.

This is the same trap as `--fd_ema_beta`, where 0.999 was found blind to mode
collapse for exactly this reason. Lower `--fd_gmm_ema_beta` toward 0.99 for a
1000-class run (≈1,040 steps of span, ~100 samples/class) and verify against
`gmm_class_mean_mse`, which should stay low rather than lagging.

## 7. Follow-ups not in this pilot

* **Multi-judge GMMs.** `compute_class_stats.py` works on any repr model;
  fitting SigLIP and MAE too and averaging the per-judge `log p` is the GMM
  analogue of `--cond_combine mean`, and is what makes single-space adversarial
  exploitation hard. Still ~free. Do this before the 1000-class run.
* **`K > 1` components per class**, the natural extension where a per-class
  Gaussian is too coarse (a Gaussian-shaped cloud of near-duplicates satisfies
  both `p` and `q`).
* **The 1000-class run**, once the 20-class arms have separated.
