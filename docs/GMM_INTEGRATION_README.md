# Class-conditional `log p` / `log q` GMM loss — moved here from `yhpark/FD-Loss`

The design, calibration, and 2 x 75k results are in
[`gmm_posterior_loss.md`](gmm_posterior_loss.md). **Read §8 first** — the matched
control overturned the working hypothesis, and the verdict is "not worth
enabling on pMF-B/256 as configured".

## What landed cleanly

New files, moved in whole. Nothing in this tree was overwritten:

| path | role |
|---|---|
| `frechet_distance/gmm.py` | `ClassGMMReference` (p), `OnlineClassStats` (q), `gmm_posterior_loss` |
| `compute_class_stats.py` | one-pass offline fit: whitening PCA + per-class Gaussians |
| `scripts/validate_class_gmm.py` | held-out accuracy / calibration sweep |
| `scripts/gmm_posterior_75k.sh` | the run |
| `tests/test_gmm_posterior.py` | 11 synthetic-Gaussian correctness checks |
| `data/fid_stats/inception_in256_t256_classgmm_k128.npz` | fitted real-data GMM (129 MB, 1.28M images) |
| `work_dirs/fd_gmm/` | both 75k runs: checkpoints, `eval_summary.csv`, metrics (32 GB) |
| `work_dirs/fd_gmm/_logs/` | raw stdout of both runs + the class-stats extraction |

`tests/test_gmm_posterior.py` is self-contained and should pass as-is:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/test_gmm_posterior.py     # expect 11/11
```

## What did NOT land — and why

The loss also needs integration edits in five shared files. **This tree has its
own uncommitted work in four of them**, so nothing was overwritten. A dry run of
the patch failed 10 of 18 hunks, almost all in `main_fd.py`:

| file | status against this tree |
|---|---|
| `frechet_distance/losses.py` | applies cleanly |
| `frechet_distance/judges.py` | 2 of 6 hunks conflict |
| `frechet_distance/evaluator.py` | 2 of 4 hunks conflict |
| `utils/eval_util.py` | conflicts |
| `main_fd.py` | 8 of 12 hunks conflict — diverged the most |

Two artifacts are provided so the merge can be done by hand:

* **`gmm_posterior_loss.patch`** — the diff containing *only* the GMM changes,
  cleanly separated from `yhpark`'s unrelated uncommitted work.
* **`gmm_reference_sources/`** — verbatim copies of the files as they were when
  they produced the 75k results. These are the ground truth; `yhpark/FD-Loss`
  has since been reverted, so this is the only surviving copy.
  `main_fd.PRE_GMM.py` is the same file with the GMM changes removed, so
  `diff main_fd.PRE_GMM.py main_fd.py` isolates the integration exactly.

`gmm_reference_sources/` also includes `queue.py` and `checkpoint_util.py`.
Those carry `yhpark`'s pre-existing uncommitted refinements, not GMM changes —
included only because the run executed against them. The GMM code does not
depend on them: the APIs it uses (`ema_stats`, `build_feats_stats`,
`extra_keys`, `AsyncCheckpointSaver`) are all present at `HEAD`.

## Integration points, if you do merge it

The `main_fd.py` changes are additive and gated behind `--fd_gmm` (default off),
so they cannot affect existing runs when the flag is absent:

1. imports — `all_gather_plain`, and `frechet_distance.gmm`
2. `get_fd_train_step(...)` — two extra kwargs, and the loss term before `backward()`
3. `setup_gmm_judge` / `bootstrap_gmm_stats` — new module-level functions
4. `train_and_evaluate` — judge attachment, queue-fill bootstrap, checkpoint save/load
5. training loop — warm-up ramp, `q` statistics update, diagnostics
6. argparse — the `--fd_gmm*` block

## Two unrelated bug fixes worth taking regardless

Both are in the patch and are independent of the GMM work:

1. **Online eval crashed on every run.** `FDEvaluator` sets `has_logits=False`,
   so `inception_score` is legitimately `None`, but three call sites consumed it
   as a number: `f"{None:.2f}"` in `eval_util.py`, `round(None, 4)` in
   `evaluator.append_eval_csv`, and `broadcast_scalar(None)` in the cached path.
   This killed the first 75k run at step 12,500.
2. **~7-minute startup stall.** `run_sanity_check` did not pass `sigma_ref_sqrt`
   in EMA mode, falling back to `eigvals` on a 2048x2048 non-symmetric product
   at every launch.
