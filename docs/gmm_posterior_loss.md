# Class-conditional `log p` / `log q` loss

An anti-collapse term for FD post-training, built from two class-conditional
GMMs in the frozen judge's representation space: `p` fitted offline on the
dataset, `q` fitted online on the generator's own samples.

---

## 1. Why the Fréchet term alone permits collapse

`compute_frechet_distance_loss` sees exactly two things about the generator:
the pooled `mu` and `sigma` of its features. That is `d + d(d+1)/2` scalar
constraints. A distribution supported on `2d+1` points satisfies all of them
exactly, so **FD → 0 is reachable by a generator with essentially no
diversity**. Collapse is in the null space of the objective, not a bug in the
implementation.

`tests/test_gmm_posterior.py::test_pooled_frechet_is_blind_to_what_the_posterior_term_catches`
makes this concrete: it collapses every sample of class `c` onto `mu_c`
(within-class variance identically zero), recolours the result so the pooled
mean and covariance match the real ones exactly, and confirms `FD < 1e-3`.
The Fréchet term cannot see the difference. The new term reads it immediately.

Adding `-log p(c|x)` from a real-data classifier does **not** fix this. Its
per-sample minimiser is a point — it is a contractive force that supplies a
*sharper attractor* for collapse. `test_q_term_flips_the_sign_of_the_diversity_update`
verifies that a descent step on `-log p` alone shrinks within-class spread on a
collapsed batch.

The missing ingredient is a term whose gradient pushes samples apart, and the
only place that can come from is the generator's own current distribution.

## 2. The objective

Everything lives in a **whitened PCA space** fitted once on real features:
`z = (f - mean_p) @ P` with `P = V_k S_k^{-1/2}`, so the real pooled covariance
is the identity in `z`-space. Both sides then shrink toward a common, meaningful
target — the real pooled within-class covariance.

```
L_gmm = lambda_cls * E[-log p(c|x)]   +   lambda_ent * E[ log q(x|c) - log p(x|c) ]
        \_______ class fidelity ______/       \_______ anti-collapse _________/
```

The second term is `KL(q(.|c) || p(.|c))` in expectation. Its gradient in
whitened space is exactly the two forces the design is built around:

```
d/dz  =  Lambda_c^p (z - mu_c^p)   -   Lambda^q (z - mu_c^q)
         \_ pull toward the real _/   \_ push off the generated _/
            class mean                    class mean
```

As the generator collapses, `Sigma^q` shrinks, `Lambda^q` grows, and the
repulsion **strengthens** — the term is strongest exactly when it is needed.

The first term is what makes a sample match the class it was *asked* for, which
the Fréchet term cannot see at all. It is deliberately weighted low (§4).

### Sample-count asymmetry

| side | per-class samples | model |
|---|---|---|
| `p` (real) | ~1300 | full per-class covariance (QDA), shrunk 0.75 toward pooled within |
| `q` (generated) | ~50 at bootstrap, ~1000 at EMA steady state | **tied** covariance + per-class means (LDA) |

A `k x k` covariance cannot be estimated from 50 samples; a `k`-dim mean can.
This split is the statistically honest one, and it costs nothing essential: if
class `c` collapses to a point, `mu_c^q` sits on that point, the sample's
Mahalanobis distance to its own class mean goes to zero, and the penalty is
maximal.

Defining `q`'s tied covariance as the pooled **within-class** scatter makes
`tr(Sigma_within^q) / tr(Sigma_within^p)` a free, per-step collapse meter.

### Gradients

All GMM parameters are detached; gradients flow only through
`x -> features -> z -> quadratic form`. For `q` this is not an approximation —
with reparameterised sampling the parameter-side term has zero expectation by
the score identity (`int q d_theta log q = d_theta int q = 0`), so dropping it
is unbiased.

## 3. Three failure modes found and fixed during bring-up

These were all found by running, not by reading. They are the substance of the
implementation and are recorded here so they are not reintroduced.

**(a) The sampled-label estimator is unbounded below.** The first version used
`log q(c_i|x_i) - log p(c_i|x_i)`. That is an unbiased estimator of the
posterior KL *only if* `q(c|x)` is the exact posterior of the joint `x` was
drawn from. A fitted `q` is not, so the estimator loses its non-negativity, and
minimising it rewards producing samples its own model **misclassifies** — off
the manifold that is arbitrarily easy. Measured on a real run: the term reached
`-25` and grew `grad_norm` 8x within ten steps.

**(b) The bounded posterior KL degenerates under collapse.** Contracting the KL
over the full class axis (`sum_c q_c (log q_c - log p_c)`) restores
non-negativity by Gibbs and is very stable. But as `q(.|x)` saturates toward a
delta — *exactly as collapse sets in* — the entropy part flattens and the term
reduces to `-log p(c|x)`, the contractive classifier-guidance signal. On
synthetic collapse it flips the sign of the diversity update. It is retained as
`--fd_gmm_mode posterior` for ablation but is not the default;
`test_posterior_mode_degenerates_under_collapse` pins the behaviour.

**(c) Scale and clamping.** The density ratio carries a systematic negative
offset because `q`'s per-class means are estimated from `n` samples, inflating
each Mahalanobis distance by `~k/n`. The offset does not affect the gradient
(it is a constant) but it decides which samples a fixed clamp hits: a `+-20` nat
clamp caught **44%** of the batch, so the clamp was reshaping the objective
rather than guarding it. Fixed by clamping about the detached batch mean, in
units of the batch standard deviation (3 sigma → ~2% clipped).

Separately, the term needed self-normalisation, mirroring the Fréchet term's
`fid / (fid.detach() + eps)`. The two are otherwise incomparable: FD's gradient
is diluted `~1/queue_size` per sample and divided by the FID magnitude, while
this is a direct per-sample loss over 128-dimensional Gaussian log-densities.
Unnormalised at weight 0.1 it produced `grad_norm` 81 against an FD-only
baseline of 0.057.

## 4. Calibration (all measured on pMF-B/256 + inception)

Reference GMM quality on 25k held-out **val** images, sweeping the `p`-side
shrinkage (`scripts/validate_class_gmm.py`):

| `p_shrink` | top-1 | top-5 | `-log p(c\|x)` | ideal-generator residual |
|---|---|---|---|---|
| 0.00 | 71.9% | 87.8% | 19.88 | 13.46 |
| 0.25 | 75.0% | 91.1% | 9.22 | 2.80 |
| 0.50 | 76.3% | 91.8% | 7.29 | 0.87 |
| **0.75** | **76.7%** | **92.5%** | **6.89** | **0.48** |
| 0.90 | 76.7% | 92.7% | 7.50 | 1.08 |
| 1.00 | 75.6% | 92.0% | 10.22 | 3.81 |

`0.75` wins on all three criteria simultaneously. Note that at `0.00` the mean
`-log p` is *worse than uniform* (6.91) despite 72% accuracy — unshrunk QDA in
128 dimensions is wildly overconfident, and its rare catastrophic misses
dominate the mean.

Gradient budget, 3 GPUs at batch 96/GPU:

| configuration | `grad_norm` |
|---|---|
| FD only (baseline) | 0.0125 |
| `+ GMM, weight 0.002` | 0.021 |
| `+ GMM, weight 0.02` | 0.144 |

At parity, `l_cls` contributes **92%** of the GMM gradient (5.84 of 6.34) — and
it is the contractive term. Hence `--fd_gmm_lambda_cls 0.1`: it is a
regulariser here, not the driver.

`--fd_gmm_weight 0.002` is chosen so the total is ~1.7x the FD-only baseline:
real influence, FD still a co-equal driver. **Recalibrate if the model, judge,
or batch size changes** — measure `grad_norm` with `--fd_gmm` off and target ~2x.

## 5. Files

| file | role |
|---|---|
| `frechet_distance/gmm.py` | `ClassGMMReference` (p), `OnlineClassStats` (q), `gmm_posterior_loss` |
| `compute_class_stats.py` | one-pass offline fit: whitening PCA + per-class Gaussians |
| `scripts/validate_class_gmm.py` | held-out accuracy / calibration sweep before spending GPU time |
| `scripts/gmm_posterior_75k.sh` | the run |
| `tests/test_gmm_posterior.py` | 11 synthetic-Gaussian correctness checks |
| `main_fd.py` | args, judge attachment, loss in the train step, diagnostics, checkpointing |
| `frechet_distance/judges.py` | `q` bootstrap folded into the existing queue fill (no second generation pass) |
| `frechet_distance/losses.py` | `all_gather_plain` for labels |

Two incidental fixes to existing code: `ClassGMMReference` does its batched
Cholesky on GPU (minutes → sub-second at startup), and `run_sanity_check` now
passes `sigma_ref_sqrt` in EMA mode, removing a ~7-minute `eigvals` stall on a
2048x2048 non-symmetric product at every launch.

## 6. Reproducing

```bash
# 1. Fit the real-data GMM (one pass over ImageNet train, ~4 min on 3 A100s)
CUDA_VISIBLE_DEVICES=4,5,6 torchrun --nproc_per_node=3 compute_class_stats.py \
    --model inception --data_path /data/dataset/imagenet --img_size 256 --pca_dim 128

# 2. Sanity-check it before spending GPU-days
CUDA_VISIBLE_DEVICES=4 python scripts/validate_class_gmm.py \
    --stats data/fid_stats/inception_in256_t256_classgmm_k128.npz \
    --shrinkage 0.25 0.5 0.75 1.0

# 3. Run
GPUS=4,5,6 bash scripts/gmm_posterior_75k.sh
```

## 7. What to watch

`gmm_within_trace_ratio` is the headline metric: generated within-class feature
variance over real. **It reads 0.893 at initialisation** — the base pMF-B model
already has ~11% less within-class diversity than real ImageNet. Falling means
collapse; rising toward 1.0 means the term is working.

Supporting meters: `gmm_class_mean_mse` (per-class mean fidelity, whitened),
`gmm_class_mean_spread` (between-class scatter vs real; falls if distinct
classes are merging), `gmm_class_coverage` (must stay 1.0),
`gmm_clamp_frac` (must stay ~0.02; if it climbs, the clamp is shaping the loss).

`gmm_cond_kl` sits around `-17` and that is expected, not a bug — it is the
finite-sample offset from (c) above. Watch its *trend*, not its sign.

## 8. Results: 2 x 75k steps, pMF-B/256, matched arms

Both arms ran the full 75,000 steps. The control is `--fd_gmm --fd_gmm_weight 0`:
identical code path, identical label/noise stream, all meters logged, exactly
zero gradient contribution. Both bootstrapped to `within_trace_ratio = 0.8934`
bit-for-bit, confirming the arms are matched.

### The premise did not hold

**The control's `within_trace_ratio` rises from 0.891 to 0.992 on its own.**
Plain FD post-training *restores* within-class diversity in this configuration
rather than destroying it. There was no collapse here for the term to fix.

This is the single most important result, and it is only visible because the
control was run. Every earlier reading of the GMM arm's rising curve as "the
term is working" was wrong — the confound was doing the work.

### What the term did do (mean over the final 25k steps, 1001 logged points)

| metric | control | GMM | delta | GMM better at |
|---|---|---|---|---|
| within-class ratio | 0.9920 | 1.0013 | **+0.0093** | 913/1001 |
| between-class spread | 0.9882 | 0.9932 | **+0.0050** | 1001/1001 |
| class-mean MSE | 0.00490 | 0.00475 | **-0.00015** | 1001/1001 |
| train-time FD | 0.2463 | 0.2510 | +0.0048 | 0/1001 |

Every class-structure metric moves in the designed direction, and the
between-class and mean-MSE effects are unambiguous: the control's own
step-to-step noise floor is 0.0004 and 0.0000 respectively, so a +0.0050 shift
that holds at 1001 of 1001 points is ~12 sigma. The within-class effect is
noisier (control std 0.0075) but holds at 91% of points, which is decisive as a
sign test.

**So the mechanism is real and measurable — the magnitudes are just tiny, because
the quantity it targets was already at ~0.99 without it.**

### The cost

Real 10k-sample FID, cfg 8.5, best EMA per step:

| step | control | GMM |
|---|---|---|
| 12,500 | 3.589 | 3.640 |
| 25,000 | 3.507 | 3.552 |
| 37,500 | 3.462 | 3.494 |
| 50,000 | 3.447 | 3.465 |
| 62,500 | 3.442 | 3.452 |
| **75,000** | **3.441** | **3.467** |

The control ends marginally ahead (3.441 vs 3.467). At 10k samples that gap is
within FID's own sampling noise and should not be read as a real regression, but
there is certainly no gain. Train-time FD is consistently worse for the GMM arm
at 1001/1001 points, which is expected — it is optimising an extra objective.

The GMM arm's *online* (non-EMA) FID was better early (3.944 vs 4.404 at 12.5k)
and the arms converged by 50k. That is a transient, not a result.

### Verdict

The term does exactly what it was designed and tested to do, at a small but
statistically unambiguous magnitude, and pays a small FID cost for it. On
**pMF-B/256 with a single inception judge it is not worth enabling**, because FD
post-training does not collapse within-class diversity in this configuration.

It remains worth testing where the premise actually holds. The right next step
is to find a configuration that *does* collapse — higher LR, longer training,
stronger CFG, or a weaker/lower-dimensional judge — and re-run this comparison
there. The `within_trace_ratio` meter is now available on any run at zero cost
(`--fd_gmm --fd_gmm_weight 0`) and is the cheapest way to look for one.

**That next step is now running: see [`gmm_uncond_20class.md`](gmm_uncond_20class.md).**
Starting from a *de-conditioned* JiT-B makes `lambda_cls` the driver rather than
a regulariser, so the premise holds by construction — and the cost is already
measured on that checkpoint: teaching classification with a frozen classifier
ensemble took FID from 10.68 to 21.49. That doubling is what `lambda_ent` is
being asked to counteract. Note the training entry point there is
`conditional_main_fd_gmm.py`, not `main_fd.py`: the integration in
`gmm_reference_sources/` was never merged into the working-tree `main_fd.py`.

## 9. Limitations

- **The matched baseline has now been run** (§8) and it overturned the working hypothesis: FD post-training raises within-class diversity on its own here. Remaining untested arms: `--fd_gmm_lambda_ent 0` (classifier guidance only) and `--fd_gmm_mode posterior`.
- **A per-class Gaussian is still second-order.** If the generator produces, per
  class, a Gaussian-shaped cloud of near-duplicate images, both `p` and `q`
  match and the loss is satisfied. The upgrade path is `K > 1` components per
  class, which the GMM framing extends to naturally.
- **The term needs residual class ambiguity to bite.** Where the posteriors
  saturate, both `log p` and `log q` sit at 0 and there is no gradient. This is
  a real property, not a test artifact — it is why the synthetic tests use
  deliberately overlapping classes.
- **`q`'s covariance is tied while `p`'s is per-class**, so the fixed point is
  slightly biased. `p_shrink=0.75` keeps `p` 75% of the way to tied, which makes
  the bias small and benign.
- **Feature-space diversity is not pixel-space diversity.** The existing
  multi-judge ensembling is the mitigation; this run uses a single judge.
- **The `p` stats are self-extracted** (center-crop 256, ImageFolder), while the
  FD term keeps using ADM's `guided_diffusion_stats.npz`. The GMM term is
  internally self-consistent; the two references differ slightly in
  preprocessing.
