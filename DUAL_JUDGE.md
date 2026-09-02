# Two semantic judges in one conditional log-p loss

`conditional_main_fd_dual.py` runs the frozen MAE linear probe and the frozen
Qwen2.5-VL judge **at the same time**, both feeding the conditional correction.
`conditional_main_fd_ponly.py` explicitly refuses this (`--cond_vlm_model
replaces the classifier ensemble`) and `conditional_main_fd_mae.py` has no VLM
path at all, so this is a third entrypoint rather than a flag on either.

## The objective

```text
L = L_FD  +  lambda_mae(s) * L_mae  +  lambda_vlm(s) * L_vlm

L_mae = -mean over B samples   of  min(log p_MAE(y | x),            cap_mae)
L_vlm = -mean over 2K questions of min(log p_VLM(correct | x, q),   cap_vlm)
```

### Why add capped log-probabilities instead of blending them

Adding two capped negative log-probabilities is the log of a weighted product
of experts. That is the AND-semantics `classifier_ensemble.py` argues for: a
judge that reaches its own cap contributes zero gradient while the unsatisfied
one keeps pushing, so the generator cannot buy its way out by over-satisfying
whichever judge is cheapest to fool. Averaging *probabilities* would be
OR-semantics and does the opposite.

The ensemble's `mean` combiner normalizes member weights and shares one cap.
That is wrong here, because the two judges are not on a common scale:

|                      | MAE linear probe            | Qwen2.5-VL judge              |
|----------------------|-----------------------------|-------------------------------|
| head                 | 1000-way softmax            | binary Yes/No token margin    |
| chance log p         | `-log 1000 = -6.91`         | `-log 2 = -0.69`              |
| samples scored /step | all `B = 21` per rank       | `K = 2` per rank              |
| cost                 | free (reuses FD features)   | 4 forwards of a 7B model      |
| gradient route       | straight through autograd   | image-space VJP surrogate     |

So each judge gets its own `lambda`, its own cap, and its own warmup/ramp.
`--cond_target_logp` caps the MAE term, `--vlm_target_logp` caps the VLM term.

### Why the MAE term costs nothing extra

The MAE ViT-L is already FD judge #1. `build_mae_probe` finds the FD judge whose
`(model_name, pool_type, target_size, feat_dim)` matches the probe checkpoint
and attaches the head to it, so `log p(y|x)` reuses features the FD loss already
computed. `cond_feature_reuse: 1.0` in the log confirms this; it drops to 0 if
you ask for EOT views, which need their own forwards.

### Why the VLM does not blow up memory

`BinaryVLMJudge.conditional_loss` runs before the FD feature extractors. It takes
an image-space VJP on a detached microbatch, frees the VLM graph, and attaches

```text
L_sur = stopgrad(L_vlm) + <x, g> - stopgrad(<x, g>)
```

which has the same value as `L_vlm` and the same first-order gradient. The VLM
activations are gone by the time SigLIP/MAE/Inception run, so the two never
coexist. Measured peak: **40.8 GB of 80 GB** with 4 ranks, batch 21, 3 FD judges.

## The evidence this design is built on

| run | judges | outcome |
|---|---|---|
| r8 | MAE probe, 1000 classes, `lambda 1e-5` | dead. `grad_ratio` fell to 0.03, held-out rank 507 -> 520 |
| r9 | MAE probe, **20 classes**, `lambda 5e-5`, cap 10% | **worked.** held-out rank 505 -> 171, top-1 0 -> 0.071, top-5 0 -> 0.19 |
| r10 | MAE probe, 20 classes, `lambda 2e-5`, cap 5% | weaker than r9; rank only 505 -> 417 |
| r5 | Qwen VLM alone, 1000 classes, `lambda 5e-5` | failed. `vlm_p_yes_target` 0.0014 -> **0.0003** over 100k steps |

Two things follow, and they set the whole configuration:

**1. The VLM cannot lead.** In r5 the VLM's gradient *dominated* FD
(`grad_ratio` 3.8 -> 1.2) and the target probability still went down. A binary
judge looking at unrecognizable 1-step JiT_B samples answers "No" to every
question and has no usable direction to give. Meanwhile `vlm_neg_cap_fraction`
was exactly **1.0 at every logged step** — every distractor question sat at the
cap contributing zero gradient, so half the VLM compute was wasted for 100k
steps. Hence `--vlm_warmup_steps 10000`: the MAE term runs alone until samples
are recognizable, then the VLM phases in.

**2. The MAE probe is being partially gamed, which is why a second judge is
worth its cost.** In r9 the training judge reached `mae_probe_top1 = 0.26` while
the held-out ResNet-50 scored the same samples at `0.071`. That gap is the
signature `classifier_ensemble.py` was written to close. The VLM differs from
the probe in architecture *and* training paradigm, which is exactly the axis
along which a shared adversarial direction is unlikely to survive.

## r11 collapsed, and why the metrics did not show it

The first dual run (`r11_dual_mae_vlm_20class`) was stopped at step 30,000. Every
number in the training log was improving — held-out `probe_top1` 0 -> 0.43,
`probe_rank` 540 -> 34, in-loss `fid_inception` 90.7 -> 12.5 — while the model
quietly stopped using its noise input at all. Measured off the saved
visualization grids as cross-noise spread at fixed label:

| | intra-class diversity | inter-class distance |
|---|---|---|
| base checkpoint | 0.804 | 0.180 |
| r9 @ 10,000 (MAE alone) | 0.206 | 0.110 |
| r11 @ 10,000 (MAE alone, VLM still off) | 0.201 | 0.140 |
| r11 @ 30,000 (dual) | **0.098** | 0.266 |

**The VLM did not cause this.** At step 10,000 `lambda_vlm_eff` was still exactly
0 and r11 (0.201) matches r9 (0.206). The MAE probe log-p term alone collapses
the generator; r9 did it too and nobody noticed, because r9 ran only 10,000
steps and its held-out numbers looked good.

The reason the in-loss FD did not object is in
`frechet_distance/queue.py::_build_feats_stats_ema`:

```python
mu = beta * self.mu_ema.detach() + (1.0 - beta) * new_d.mean(0)
m2 = beta * self.m2_ema.detach() + (1.0 - beta) * (new_d.T @ new_d) / B
```

At `beta = 0.999` the covariance FD scores against the reference is accumulated
over ~1000 steps. A generator that is collapsed at every individual step but
whose single mode *drifts* across training still accumulates a broad covariance:
the estimator cannot separate variation over training time from diversity within
a batch. Hence in-loss inception FD 12.5 against a frozen-checkpoint eval FID of
120.7 on the same weights — a 10x gap, and the eval is the honest one.

So the only term that could have penalized collapse was structurally blind to
it, and nothing else in the objective rewards diversity. Once collapsed there is
no restoring force. This is not a lambda-tuning problem.

Two changes followed, both in `r12_dual_fdbeta99`:

1. **`--fd_ema_beta 0.99`** (~100-step window, ~8.4k effective samples at 84
   gathered per step — still enough for a 2048-dim covariance, 10x more
   current). Side effect: the live-batch FD gradient is scaled by `(1 - beta)`,
   so `grad_x_fd` grows ~10x and every conditional ratio shrinks ~10x at fixed
   lambda. Both lambdas were re-measured, not rescaled by hand.
2. **`noise_delta`**, the mirror of `cond_delta`: hold the label, roll the noise,
   measure relative pixel change. Logged every `print_freq` steps, with a
   `MODE COLLAPSE` warning below `--noise_delta_warn` (default 0.35; the base
   checkpoint reads ~0.75). This is the metric r11 needed and did not have.

Note the cap does **not** protect against this. By step 30,000 r11's MAE term
was almost entirely clamped (`l_cond` 2.385 against a cap of 2.3026) and
collapse continued regardless — a confidence cap bounds per-sample confidence,
it does not require diversity.

## Calibrating the two lambdas

Loss *values* are not comparable across terms, so tune on image-space gradient
norms. The dual entrypoint logs each judge separately:

| key | meaning |
|---|---|
| `grad_ratio_mae_fd` | MAE pull / FD pull — target ~0.35-0.50 (r9 sat at 0.34) |
| `grad_ratio_vlm_fd` | VLM pull / FD pull — target ~0.05-0.20, deliberately subordinate |
| `grad_ratio_p_fd`   | combined conditional pull / FD pull |
| `cos_update_mae_vlm`| **do the two judges agree?** the whole premise rides on this |
| `cos_update_fd_mae`, `cos_update_fd_vlm` | conflict between each judge and realism |
| `probe_rank`, `probe_top1`, `probe_top5` | held-out ResNet-50; the only honest meter |

Short calibrations off the r8 start checkpoint measured:

| | `fd_ema_beta` 0.999 (r11) | `fd_ema_beta` 0.99 (r12) |
|---|---|---|
| `grad_x_fd` | ~0.0006 | ~0.0060 |
| `lambda_mae 5e-5` -> ratio | 0.51 | 0.067 |
| `lambda_vlm 8e-7` -> ratio | ~0.099 | 0.0064 |
| lambda for the target ratio | 5e-5 / 8e-7 | **2.2e-4 / 1.0e-5** |

Two things to carry away. The VLM pulls **~12x harder per unit lambda** than the
MAE probe, because its entire gradient lands on the `K=2` scored images instead
of spreading over all 21 — never port a lambda between the two terms. And
lowering `fd_ema_beta` by 10x raises `grad_x_fd` by 10x, so *both* lambdas need
re-measuring whenever that window changes. r12 targets ~0.30 for the MAE term
(deliberately under r9's 0.34, which is the regime that collapsed, and well over
r8's 0.03, which learned nothing) and ~0.08 for the VLM.

Do not trust r5's ratio for anything: it was measured with two FD judges, not
three, and at the old beta.

## Distractor pool

`--vlm_distractor_pool train_classes` draws the pairwise distractor from
`--train_class_ids` instead of all 1000 classes, so the negative question is a
discrimination among classes the generator actually produces.

Honest caveat: this does **not** fix the wasted-distractor problem in the early
regime. The 12-step calibration still showed `vlm_p_no_distractor = 0.999` and
`vlm_neg_cap_fraction = 1.0` with the restricted pool, because the VLM currently
rejects *every* class for these samples, not because the distractor was far
away. It should start to bind once `vlm_p_yes_target` lifts off the floor. If
`vlm_neg_cap_fraction` is still pinned at 1.0 well after the VLM ramp finishes,
switch to `--vlm_question_mode target_only` and halve the VLM cost instead of
paying for a question that never produces gradient.

## Running it

```bash
CUDA_GPUS=4,5,6,7 MASTER_PORT=29551 EXP_NAME=r11_dual_mae_vlm_20class \
  bash scripts/run_jit_dual_20class.sh
```

Defaults: 60 x 1250 = **75,000 iterations**, batch 21 per rank (global 84),
starting from `r8/step_0071799.pth` — the same checkpoint r9 started from. Steps
0-10,000 use the MAE term with r9's exact loss settings (`lambda 5e-5`, ramp
500, cap 10%) and the VLM ramps in over steps 10,000-15,000.

One caveat on comparing to r9, since it is easy to get wrong: the loss
configuration matches for the first 10,000 steps but **the learning-rate
schedule does not**. Both use cosine `1e-5 -> 0`, but r9 spread that decay over
its full 10,000 steps while this run spreads it over 75,000. At step 5,000 r9
was at roughly `0.5e-5` and this run is at roughly `1.0e-5`. So the first 10,000
steps are a same-loss, higher-LR run, not a replication — expect the MAE-only
phase here to move *faster* than r9's trace, and do not read that as an effect
of the dual judge. The genuinely controlled comparison is r9's final state
against this run's state at step 10,000, both as single-judge MAE results.

Expect roughly 0.50 s/it before the VLM engages and ~1.65 s/it after, so about
**30-33 hours** including six 20,000-image online evaluations.

### What to check, in order

1. **`noise_delta`, from step 500 onward.** This is now the first thing to look
   at, not the last. It starts near 0.75. r11 was already at 0.20 by step
   10,000 and 0.10 by 30,000, with every other metric improving. Anything below
   0.35 logs a `MODE COLLAPSE` warning — grep for it. If it appears and stays,
   the run is dead regardless of what the accuracies say.
2. **Step ~15,000, the moment the VLM ramp completes.** Read
   `grad_ratio_vlm_fd`. Outside 0.05-0.20, relaunch with
   `LAMBDA_VLM = 1.0e-5 * 0.08 / <observed>`.
3. **`probe_rank` must keep falling.** Held-out ResNet-50, never in the loss.
4. **The gap `mae_probe_top1 - probe_top1` must not widen.** Narrowing versus r9
   is the hypothesis. Caveat: r11 narrowed it from r9's 3.7x to 1.4x, but there
   is no 20-class MAE-only run past 10,000 steps, so training length and LR are
   confounded with the second judge. A clean claim needs that control run.
5. **`cos_update_mae_vlm`.** r11 held it at ~0.0001 for 30,000 steps: the two
   judges were essentially orthogonal in image space, so whatever helped, it was
   not gradient-level consensus. Report that rather than tuning around it.
6. **Frozen-checkpoint eval FID, not the in-loss `fid_*`.** r11 is the reason:
   in-loss 12.5 versus eval 120.7. The in-loss values are the training objective
   and cannot be read as realism.
7. **`vlm_neg_cap_fraction`.** Pinned at 1.000 for all of r11 even with the
   restricted distractor pool — half the VLM compute still produces no gradient.
   If it stays pinned past the ramp, switch to `--vlm_question_mode target_only`.

Kill it if MAE log-p and VLM p(correct) both improve while `probe_rank` stays
near 500. That is both judges being fooled at once, which is a more interesting
failure than one being fooled — but it is still a dead run.
