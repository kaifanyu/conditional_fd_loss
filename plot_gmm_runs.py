#!/usr/bin/env python
"""Plot the results of a GMM-posterior FD post-training run (conditional_main_fd_gmm.py).

Reads everything a run directory already writes — no re-generation, no GPU:

  args_*.json          -> class count, GMM weight/temp, the run's own config
  training_metrics.json-> JSONL, one record per --print_freq steps (train curves)
  eval_summary.csv     -> the real FID sweep (all EMA copies at every eval step)

Produces, per run, `<label>_overview.png` (12 panels: FID, held-out class probe,
GMM Bayes terms, class structure, gradient balance), and, when several runs are
given, `compare_overview.png` plus `summary.md` with the headline numbers.

Metric semantics worth remembering while reading the plots:
  probe_*                held-out ResNet-50 on generated samples. Never in the
                         loss -> the honest conditioning number.
  gmm_top1               the in-loss GMM posterior grading itself: optimistic.
  gmm_class_mean_spread  between-class scatter of q over p. 0 = de-conditioned,
                         1.0 = matches the data.
  gmm_within_trace_ratio within-class scatter of q over p. Falls toward 1.0 as
                         classes separate; below 1.0 is intra-class collapse.
  fid_inception/siglip/mae  the *online queue* estimate used by the FD loss, not
                         the eval FID. Compare shapes, not absolute values.

Usage:
    python plot_gmm_runs.py work_dirs/.../jitB_uncond_gmm20_armC_logp_logq \\
                            work_dirs/.../jitB_uncond_gmm100_armC_v2_self_w0103 \\
                            --labels 20-class 100-class --out_dir kai_results/plots_gmm
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe: these runs live on GPU nodes
import matplotlib.pyplot as plt
import numpy as np

EMA_ORDER = ["online", "edm_250", "edm_500", "edm_1000", "edm_2000"]
RAW_ALPHA = 0.22          # unsmoothed trace behind every smoothed curve
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_run(run_dir, label=None):
    """Collect args / training curves / eval FIDs for one run directory."""
    run_dir = Path(run_dir)
    args_files = sorted(run_dir.glob("args_*.json"))
    cfg = json.load(open(args_files[-1])) if args_files else {}

    metrics_path = run_dir / "training_metrics.json"
    rows = load_jsonl(metrics_path) if metrics_path.exists() else []

    eval_rows = []
    eval_path = run_dir / "eval_summary.csv"
    if eval_path.exists():
        for r in csv.DictReader(open(eval_path)):
            try:
                eval_rows.append({
                    "step": int(r["step"]),
                    "ema_label": r["ema_label"],
                    "cfg": float(r["cfg"]),
                    "num_imgs": int(r["num_imgs"]),
                    "fid": float(r["fid"]),
                })
            except (KeyError, ValueError):
                continue

    class_ids = cfg.get("train_class_ids") or cfg.get("class_of_interest") or []
    return {
        "dir": run_dir,
        "label": label or cfg.get("exp_name") or run_dir.name,
        "cfg": cfg,
        "rows": rows,
        "eval": eval_rows,
        "n_classes": len(class_ids) or cfg.get("num_classes", 0),
    }


def series(rows, key):
    """(steps, values) for one metric, dropping missing/NaN records."""
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        xs.append(r.get("iteration", len(xs)))
        ys.append(v)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def smooth(y, window):
    """Centered moving average; window<=1 is a no-op. Keeps the array length."""
    if window <= 1 or y.size < 3:
        return y
    window = int(min(window, max(3, y.size // 3)))
    kernel = np.ones(window) / window
    padded = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------

def curve(ax, rows, key, window, *, label=None, color=None, ls="-", lw=1.6,
          scale=1.0, raw=True):
    """Smoothed curve with the raw trace ghosted behind it. Returns True if drawn."""
    xs, ys = series(rows, key)
    if xs.size == 0:
        return False
    ys = ys * scale
    if raw and window > 1:
        ax.plot(xs, ys, color=color, alpha=RAW_ALPHA, lw=0.8, ls=ls, zorder=1)
    ax.plot(xs, smooth(ys, window), color=color, ls=ls, lw=lw,
            label=label or key, zorder=2)
    return True


def twin_legend(ax, twin, loc="best"):
    """One legend for a twinx pair — separate legends collide, since each `best`
    placement is blind to the other axes' artists."""
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    if h1 or h2:
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc=loc, framealpha=0.9)


def acc_ylim(ax, rows, keys, floor=0.35):
    """Accuracy panels: keep 0..1 when the run gets there, else zoom in so a
    0.2-max curve is not a flat line at the bottom."""
    top = 0.0
    for k in keys:
        _, ys = series(rows, k)
        if ys.size:
            top = max(top, float(ys.max()))
    ax.set_ylim(-0.03 * max(floor, top), min(1.03, 1.08 * max(floor, top)))


def finish(ax, title, *, xlabel="step", ylabel=None, legend=True, ncol=1):
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7.5, ncol=ncol, framealpha=0.85)


def eval_by_label(eval_rows):
    """{ema_label: (steps, fids)} sorted by step."""
    out = {}
    for lab in sorted({r["ema_label"] for r in eval_rows},
                      key=lambda l: EMA_ORDER.index(l) if l in EMA_ORDER else 99):
        pts = sorted([(r["step"], r["fid"]) for r in eval_rows if r["ema_label"] == lab])
        out[lab] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return out


def best_ema_curve(eval_rows):
    """Per step, the best FID across EMA copies (the number you would report)."""
    steps = sorted({r["step"] for r in eval_rows})
    fids = [min(r["fid"] for r in eval_rows if r["step"] == s) for s in steps]
    return np.array(steps, dtype=float), np.array(fids)


def plot_eval_fid(ax, run):
    if not run["eval"]:
        ax.text(0.5, 0.5, "no eval_summary.csv", ha="center", va="center")
        finish(ax, "eval FID", legend=False)
        return
    for i, (lab, (xs, ys)) in enumerate(eval_by_label(run["eval"]).items()):
        online = lab == "online"
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.4 if online else 1.8,
                ls="--" if online else "-",
                color="0.35" if online else COLORS[i % len(COLORS)],
                label=lab, zorder=3 if online else 2)
    best = min(run["eval"], key=lambda r: r["fid"])
    ax.plot([best["step"]], [best["fid"]], marker="*", ms=14, color="crimson", zorder=4)
    ax.annotate(f"best {best['fid']:.2f}\n{best['ema_label']} @ {best['step']}",
                (best["step"], best["fid"]), textcoords="offset points",
                xytext=(-6, 16), fontsize=7.5, ha="right", color="crimson")
    ax.set_yscale("log")
    n_imgs = run["eval"][0]["num_imgs"]
    finish(ax, f"eval FID ({n_imgs:,} imgs, cfg {run['eval'][0]['cfg']:g}) — lower is better",
           ylabel="FID (log)", ncol=2)


def plot_overview(run, out_path, window):
    rows, n_cls = run["rows"], run["n_classes"]
    chance = 1.0 / n_cls if n_cls else None
    # cfg_delta logs a vector-field diagnostic set the other modes have no
    # analogue for. Give it its own row rather than leaving three blank panels
    # in every density/posterior overview.
    has_cfg = bool(series(rows, "gmm_cfg_alignment_cos")[0].size)
    n_rows = 5 if has_cfg else 4
    fig, axes = plt.subplots(n_rows, 3, figsize=(19, 3.9 * n_rows),
                             constrained_layout=True)
    ax = axes.ravel()

    # --- 1. the headline result -------------------------------------------
    plot_eval_fid(ax[0], run)

    # --- 2. online FD-queue estimates (the training-time signal) ----------
    for i, k in enumerate(["fid_inception", "fid_siglip", "fid_mae"]):
        curve(ax[1], rows, k, window, color=COLORS[i], label=k)
    ax[1].set_yscale("log")
    finish(ax[1], "online FD-queue estimate per judge (training signal, not eval FID)",
           ylabel="FD (log)")

    # --- 3. conditioning: held-out probe vs the loss's own grader ---------
    curve(ax[2], rows, "probe_top1", window, color=COLORS[0], label="probe top-1 (held-out)")
    curve(ax[2], rows, "probe_top5", window, color=COLORS[1], label="probe top-5 (held-out)")
    curve(ax[2], rows, "gmm_top1", window, color=COLORS[3], ls="--",
          label="gmm top-1 (in-loss, optimistic)")
    if chance:
        ax[2].axhline(chance, ls=":", c="grey", lw=1.0,
                      label=f"chance = 1/{n_cls} = {chance:.3f}")
    acc_ylim(ax[2], rows, ["probe_top1", "probe_top5", "gmm_top1"])
    finish(ax[2], "class conditioning learned from a de-conditioned start",
           ylabel="accuracy")

    # --- 4. probe log p and rank ------------------------------------------
    curve(ax[3], rows, "probe_logp", window, color=COLORS[0], label="probe log p(y|x)")
    if chance:
        ax[3].axhline(math.log(chance), ls=":", c="grey", lw=1.0,
                      label=f"uniform over {n_cls} = {math.log(chance):.2f}")
    ax[3].set_ylabel("log p(y|x)", fontsize=9)
    twin = ax[3].twinx()
    curve(twin, rows, "probe_rank", window, color=COLORS[3], label="probe rank of y")
    twin.set_yscale("symlog", linthresh=1.0)
    twin.set_ylabel("rank of intended class (log, 0 = argmax)", fontsize=8)
    twin.tick_params(labelsize=8)
    twin_legend(ax[3], twin, loc="center right")
    finish(ax[3], "held-out probe confidence and rank of the intended class", legend=False)

    # --- 5. the two halves of the GMM objective ---------------------------
    curve(ax[4], rows, "gmm_logp_c", window, color=COLORS[0], label="log p(c|x)  (class fidelity)")
    if chance:
        ax[4].axhline(math.log(chance), ls=":", c="grey", lw=1.0,
                      label=f"uniform = {math.log(chance):.2f}")
    cap = run["cfg"].get("fd_gmm_cls_cap")
    if cap is not None and cap < 0:
        ax[4].axhline(cap, ls="--", c="grey", lw=0.9, label=f"cls cap = {cap}")
    ax[4].set_ylabel("log p(c|x)", fontsize=9)
    twin = ax[4].twinx()
    if curve(twin, rows, "gmm_cond_kl", window, color=COLORS[3],
             label="log q(x|c) - log p(x|c)  (anti-collapse)"):
        twin.axhline(0.0, ls=":", c=COLORS[3], lw=0.8)
        twin.set_ylabel("nats", fontsize=8)
        twin.tick_params(labelsize=8)
    else:
        # cfg_delta logs no scalar anti-collapse term -- its surrogate is
        # origin-dependent and not comparable. See the cfg_delta row instead.
        twin.set_axis_off()
    twin_legend(ax[4], twin, loc="lower right")
    finish(ax[4], "GMM objective: class fidelity vs anti-collapse term", legend=False)

    # --- 6. class structure in judge space ---------------------------------
    curve(ax[5], rows, "gmm_class_mean_spread", window, color=COLORS[0],
          label="between-class spread q/p")
    curve(ax[5], rows, "gmm_within_trace_ratio", window, color=COLORS[3],
          label="within-class scatter q/p")
    ax[5].axhline(1.0, ls="--", c="grey", lw=1.0, label="data-matched = 1.0")
    finish(ax[5], "class structure in judge space (spread up to 1.0 = good; "
                  "within below 1.0 = collapse)", ylabel="ratio to data")

    # --- 7. the FD (marginal) term ----------------------------------------
    curve(ax[6], rows, "fd_loss_raw", window, color=COLORS[0], label="fd_loss_raw (Frechet)")
    ax[6].set_yscale("log")
    ax[6].set_ylabel("raw FD (log)", fontsize=9)
    twin = ax[6].twinx()
    curve(twin, rows, "fd_loss_norm", window, color=COLORS[3], label="fd_loss_norm (self-normalised)")
    twin.set_ylabel("normalised", fontsize=8)
    twin.tick_params(labelsize=8)
    twin_legend(ax[6], twin, loc="upper right")
    finish(ax[6], "FD term: raw distance falls, normalised term stays ~n_judges", legend=False)

    # --- 8. who drives the update -----------------------------------------
    curve(ax[7], rows, "grad_x_fd", window, color=COLORS[0], label="|dL/dx| from FD")
    curve(ax[7], rows, "grad_x_p", window, color=COLORS[1], label="|dL/dx| from GMM p-term")
    ax[7].set_yscale("symlog", linthresh=1e-4)
    ax[7].set_ylabel("image-grad norm (symlog)", fontsize=9)
    twin = ax[7].twinx()
    curve(twin, rows, "grad_ratio_p_fd", window, color=COLORS[3], label="ratio p / FD")
    twin.set_ylabel("ratio", fontsize=8)
    twin.tick_params(labelsize=8)
    twin_legend(ax[7], twin, loc="lower right")
    finish(ax[7], "gradient balance: conditional term vs marginal FD term", legend=False)

    # --- 9. conflict and measured label effect -----------------------------
    curve(ax[8], rows, "cos_update_fd_p", window, color=COLORS[0], label="cos(update_FD, update_p)")
    ax[8].axhline(0.0, ls=":", c="grey", lw=0.9)
    ax[8].set_ylim(-1.05, 1.05)
    ax[8].set_ylabel("cosine", fontsize=9)
    twin = ax[8].twinx()
    curve(twin, rows, "cond_delta", window, color=COLORS[3],
          label="cond_delta (image change when label is rolled)")
    twin.set_ylabel("relative image delta", fontsize=8)
    twin.tick_params(labelsize=8)
    twin_legend(ax[8], twin, loc="lower right")
    finish(ax[8], "term conflict (cosine) and how much the label actually moves the image",
           legend=False)

    # --- 10. estimator health ---------------------------------------------
    curve(ax[9], rows, "gmm_cls_sat_frac", window, color=COLORS[0], label="cls-cap saturated frac")
    curve(ax[9], rows, "gmm_clamp_frac", window, color=COLORS[1], label="clamped frac")
    curve(ax[9], rows, "cond_delta_valid", window, color=COLORS[2], label="cond_delta valid frac")
    ax[9].set_ylim(-0.03, 1.03)
    finish(ax[9], "estimator health: saturation / clamping", ylabel="fraction of batch")

    # --- 11. optimisation --------------------------------------------------
    curve(ax[10], rows, "grad_norm", window, color=COLORS[0], label="grad_norm")
    ax[10].set_ylabel("grad norm", fontsize=9)
    twin = ax[10].twinx()
    if curve(twin, rows, "lr", window, color=COLORS[3], label="lr", raw=False):
        twin.set_yscale("log")
        twin.set_ylabel("lr (log)", fontsize=8)
        twin.tick_params(labelsize=8)
    twin_legend(ax[10], twin, loc="upper left")
    finish(ax[10], f"optimisation ({run['cfg'].get('lr_sched', '?')} schedule)", legend=False)

    # --- 12. what enters the loss -----------------------------------------
    curve(ax[11], rows, "loss", window, color=COLORS[0], label="total loss (left)")
    curve(ax[11], rows, "fd_loss_norm", window, color=COLORS[1], label="fd_loss_norm (left)")
    ax[11].set_ylabel("normalised FD scale", fontsize=9)
    twin = ax[11].twinx()
    curve(twin, rows, "p_loss_weighted", window, color=COLORS[2],
          label="weighted GMM term (right)")
    curve(twin, rows, "gmm_scale_eff", window, color=COLORS[3], ls="--",
          label="effective GMM weight (right)", raw=False)
    twin.axhline(0.0, ls=":", c="grey", lw=0.8)
    twin.set_ylabel("weighted GMM contribution", fontsize=8)
    twin.tick_params(labelsize=8)
    twin_legend(ax[11], twin, loc="center right")
    finish(ax[11], "loss decomposition — the GMM term is ~1e-2 against an FD term of ~3",
           legend=False)

    # --- 13-15. cfg_delta only: the injected vector field ------------------
    if has_cfg:
        curve(ax[12], rows, "gmm_cfg_alignment_cos", window, color=COLORS[0],
              label="cos(grad log q(c|z), grad log p(c|z))")
        curve(ax[12], rows, "gmm_cfg_relative_error", window, color=COLORS[3],
              label="||dq - dp|| / ||dp||   (1.0 = q says nothing yet)")
        ax[12].axhline(1.0, ls="--", c="grey", lw=0.9)
        ax[12].axhline(0.0, ls=":", c="grey", lw=0.8)
        ax[12].set_ylim(-0.1, 1.35)
        finish(ax[12], "CFG-delta field: is q catching the teacher? "
                       "(cosine up / rel-error down = yes)",
               ylabel="cosine / ratio")

        curve(ax[13], rows, "gmm_cfg_teacher_rms", window, color=COLORS[0],
              label="||grad log p(c|z)||  (teacher, frozen)")
        curve(ax[13], rows, "gmm_cfg_fake_rms", window, color=COLORS[1],
              label="||grad log q(c|z)||  (online)")
        curve(ax[13], rows, "gmm_cfg_error_rms", window, color=COLORS[3], ls="--",
              label="||injected field||")
        ax[13].set_yscale("log")
        finish(ax[13], "field magnitudes in whitened judge space — q lifts off ~0 "
                       "as its class means separate", ylabel="RMS (log)")

        curve(ax[14], rows, "gmm_cls_objective", window, color=COLORS[0],
              label="lambda_cls * cls term")
        curve(ax[14], rows, "gmm_q_objective", window, color=COLORS[3],
              label="lambda_ent * q term (surrogate)")
        ax[14].axhline(0.0, ls=":", c="grey", lw=0.8)
        finish(ax[14], "objective split — a flat zero cls term is the PURE "
                       "(lambda_cls=0) arm", ylabel="loss contribution")

    cfg = run["cfg"]
    fig.suptitle(
        f"{run['label']}  —  {run['dir']}\n"
        f"{cfg.get('model')} / {n_cls} classes / gmm_weight={cfg.get('fd_gmm_weight')} "
        f"temp={cfg.get('fd_gmm_temp')} mode={cfg.get('fd_gmm_mode')} "
        f"cls_norm={cfg.get('fd_gmm_cls_normalization', 'n/a')} | "
        f"bsz {cfg.get('batch_size')}x{cfg.get('world_size')} | lr {cfg.get('lr')} | "
        f"{cfg.get('total_steps', '?')} steps | judges: {', '.join(cfg.get('fd_repr_models', []))}",
        fontsize=11)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_compare(runs, out_path, window):
    fig, axes = plt.subplots(2, 3, figsize=(19, 9.5), constrained_layout=True)
    ax = axes.ravel()

    for i, run in enumerate(runs):
        c = COLORS[i % len(COLORS)]
        lab, n_cls = run["label"], run["n_classes"]
        chance = 1.0 / n_cls if n_cls else None
        rows = run["rows"]

        # 1. eval FID, best EMA per step
        if run["eval"]:
            xs, ys = best_ema_curve(run["eval"])
            n_imgs = run["eval"][0]["num_imgs"]
            ax[0].plot(xs, ys, marker="o", ms=4, lw=1.8, color=c,
                       label=f"{lab} (best EMA, {n_imgs // 1000}k imgs)")
            j = int(np.argmin(ys))
            ax[0].annotate(f"{ys[j]:.2f}", (xs[j], ys[j]), textcoords="offset points",
                           xytext=(0, -14), fontsize=8, ha="center", color=c)

        # 2. held-out probe top-1, with each run's own chance level
        curve(ax[1], rows, "probe_top1", window, color=c, label=f"{lab} probe top-1")
        if chance:
            ax[1].axhline(chance, ls=":", lw=1.0, color=c,
                          label=f"{lab} chance = 1/{n_cls}")

        # 3. lift over chance — the class-count-fair comparison
        xs, ys = series(rows, "probe_top1")
        if xs.size and chance:
            lift = smooth(ys / chance, window)
            # probe_top1 is exactly 0 for tens of thousands of steps on the harder
            # run; on a log axis that is -inf, so mask it rather than plot spikes.
            lift = np.where(lift > 0, lift, np.nan)
            ax[2].plot(xs, lift, lw=1.7, color=c, label=lab)

        # 4. between-class structure
        curve(ax[3], rows, "gmm_class_mean_spread", window, color=c, label=lab)

        # 5. class fidelity term
        curve(ax[4], rows, "gmm_logp_c", window, color=c, label=f"{lab} log p(c|x)")
        if chance:
            ax[4].axhline(math.log(chance), ls=":", lw=1.0, color=c,
                          label=f"{lab} uniform = {math.log(chance):.2f}")

        # 6. the trade-off: conditioning bought at what FID
        if run["eval"] and rows:
            xs_e, ys_e = best_ema_curve(run["eval"])
            xs_p, ys_p = series(rows, "probe_top1")
            if xs_p.size:
                probe_at_eval = np.interp(xs_e, xs_p, smooth(ys_p, window))
                ax[5].plot(probe_at_eval, ys_e, marker="o", ms=4, lw=1.4, color=c, label=lab)
                for xi, yi, si in zip(probe_at_eval, ys_e, xs_e):
                    if si in (xs_e[0], xs_e[-1]) or yi == ys_e.min():
                        ax[5].annotate(f"{int(si / 1000)}k", (xi, yi),
                                       textcoords="offset points", xytext=(5, 4),
                                       fontsize=7, color=c)

    ax[0].set_yscale("log")
    finish(ax[0], "eval FID per run (best EMA copy at each step)\n"
                  "NB: different reference stats + sample counts — compare trends, not absolutes",
           ylabel="FID (log)")
    ax[1].set_ylim(-0.03, 1.03)
    finish(ax[1], "held-out probe top-1 on generated samples", ylabel="top-1 accuracy", ncol=2)
    ax[2].set_yscale("log")
    ax[2].set_ylim(bottom=0.5)
    ax[2].axhline(1.0, ls="--", c="grey", lw=1.0, label="chance")
    finish(ax[2], "probe top-1 as a multiple of chance (fair across class counts;\n"
                  "gaps = probe never got one right at that step)",
           ylabel="x chance (log)")
    ax[3].axhline(1.0, ls="--", c="grey", lw=1.0, label="data-matched = 1.0")
    finish(ax[3], "between-class mean spread q/p (0 = de-conditioned)", ylabel="ratio to data")
    finish(ax[4], "GMM class-fidelity term log p(c|x)", ylabel="log p(c|x)", ncol=2)
    finish(ax[5], "cost of conditioning: FID vs probe top-1 (labels = step)",
           xlabel="held-out probe top-1", ylabel="eval FID (best EMA)")

    fig.suptitle("GMM-posterior FD post-training — run comparison ("
                 + "  vs  ".join(r["label"] for r in runs) + ")", fontsize=13)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_per_class_clock(runs, out_path, window, global_batch=96):
    """The decision figure: everything against samples/class, not steps.

    Class count only enters this loss through how often each class's q-statistics
    get refreshed, so `step * global_batch / C` is the axis on which runs at
    different class counts are actually comparable — and the axis on which the
    cost of going to 1000 classes becomes obvious.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True)

    for i, run in enumerate(runs):
        c, n_cls = COLORS[i % len(COLORS)], run["n_classes"]
        if not n_cls:
            continue
        per_class = global_batch / n_cls
        lab = f"{run['label']} (C={n_cls})"

        xs, ys = series(run["rows"], "probe_top1")
        if xs.size:
            axes[0].plot(xs * per_class, smooth(ys, window), lw=1.9, color=c, label=lab)
            axes[0].axhline(1.0 / n_cls, ls=":", lw=1.0, color=c)
            axes[0].annotate(f"end of run: {smooth(ys, window)[-1]:.3f}",
                             (xs[-1] * per_class, smooth(ys, window)[-1]),
                             textcoords="offset points",
                             xytext=(6, -2) if n_cls > 20 else (-6, 6),
                             fontsize=8.5, ha="left" if n_cls > 20 else "right",
                             color=c, fontweight="bold")
        if run["eval"]:
            es, ef = best_ema_curve(run["eval"])
            axes[1].plot(es * per_class, ef, marker="o", ms=4, lw=1.8, color=c, label=lab)

    # what one more 50k-step block would buy the 100-class run, on this axis
    hundred = next((r for r in runs if r["n_classes"] == 100), None)
    if hundred and hundred["rows"]:
        reached = hundred["rows"][-1]["iteration"] * global_batch / 100
        for ax in axes:
            ax.axvspan(reached, reached + 50000 * global_batch / 100,
                       color="tab:orange", alpha=0.10, lw=0)
        axes[0].annotate("+50k steps\n(~12 h on 3 GPUs)",
                         (reached + 24000, 0.62), fontsize=8.5, color="tab:orange",
                         ha="center", va="center")

    # where the pilot stopped paying: its own saturation point
    for ax in axes:
        ax.axvline(87000, ls="--", lw=1.2, color="0.45")
    axes[0].annotate("pilot saturates\n~87k smp/class", (89000, 0.30),
                     fontsize=8.5, color="0.35", ha="left", va="center")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlim(0, 250000)
    axes[1].set_xlim(0, 250000)
    finish(axes[0], "held-out probe top-1 against the per-class clock\n"
                    "(dotted = each run's chance level)",
           xlabel="samples per class  =  step x global_batch / C",
           ylabel="probe top-1")
    axes[1].set_yscale("log")
    finish(axes[1], "eval FID against the same clock\n"
                    "(each vs its own subset reference — shapes, not levels)",
           xlabel="samples per class", ylabel="FID (log)")

    fig.suptitle("The per-class clock — the axis that decides whether 1000 classes is affordable",
                 fontsize=13)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def write_summary(runs, out_path):
    """Headline numbers, so the plots do not have to be squinted at."""
    lines = ["# GMM-posterior FD post-training — run summary", ""]
    for run in runs:
        cfg, rows = run["cfg"], run["rows"]
        last = rows[-1] if rows else {}
        best = min(run["eval"], key=lambda r: r["fid"]) if run["eval"] else None
        # first eval = the best EMA copy at the earliest eval step, so the
        # "start -> best" pair below compares like with like
        first_eval = (min((r for r in run["eval"]
                           if r["step"] == min(e["step"] for e in run["eval"])),
                          key=lambda r: r["fid"]) if run["eval"] else None)

        def fin(k, fmt="{:.4g}"):
            v = last.get(k)
            return fmt.format(v) if isinstance(v, (int, float)) else "n/a"

        def peak(k, fmt="{:.4g}"):
            _, ys = series(rows, k)
            return fmt.format(ys.max()) if ys.size else "n/a"

        lines += [
            f"## {run['label']}",
            f"`{run['dir']}`",
            "",
            f"- classes trained: **{run['n_classes']}** | judges: "
            f"{', '.join(cfg.get('fd_repr_models', []))} | gmm_weight "
            f"{cfg.get('fd_gmm_weight')} | temp {cfg.get('fd_gmm_temp')} | "
            f"cls_norm {cfg.get('fd_gmm_cls_normalization', 'n/a')}",
            f"- steps: {cfg.get('total_steps', '?')} | bsz "
            f"{cfg.get('batch_size')}x{cfg.get('world_size')} | lr {cfg.get('lr')} "
            f"({cfg.get('lr_sched')})",
        ]
        if best:
            lines.append(
                f"- **eval FID: {first_eval['fid']:.2f} (step {first_eval['step']}, "
                f"{first_eval['ema_label']}) -> best {best['fid']:.2f} "
                f"(step {best['step']}, {best['ema_label']}, "
                f"{best['num_imgs']:,} imgs, cfg {best['cfg']:g})**")
            final_step = max(r["step"] for r in run["eval"])
            final_best = min(r["fid"] for r in run["eval"] if r["step"] == final_step)
            if final_step != best["step"]:
                lines.append(f"- FID at the final eval (step {final_step}): "
                             f"{final_best:.2f} — best was earlier, at step {best['step']}")
        chance = 1.0 / run["n_classes"] if run["n_classes"] else None
        lines += [
            f"- held-out probe top-1: final **{fin('probe_top1')}** "
            f"(peak {peak('probe_top1')}"
            + (f", chance {chance:.3f}, i.e. {float(fin('probe_top1')) / chance:.0f}x chance)"
               if chance and last.get("probe_top1") else ")"),
            f"- held-out probe top-5: final {fin('probe_top5')} | "
            f"mean rank of intended class: {fin('probe_rank')}",
            f"- in-loss gmm top-1: final {fin('gmm_top1')} | log p(c|x): {fin('gmm_logp_c')}",
            f"- class structure: between-class spread {fin('gmm_class_mean_spread')} "
            f"(target 1.0) | within-class scatter {fin('gmm_within_trace_ratio')} "
            f"(target 1.0, <1 = collapse)",
            f"- raw FD: {fin('fd_loss_raw')} | grad ratio p/FD: {fin('grad_ratio_p_fd')} | "
            f"cos(FD, p): {fin('cos_update_fd_p')}",
            "",
        ]
    out_path.write_text("\n".join(lines))
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="run directories (log_dir of each run)")
    ap.add_argument("--labels", nargs="+", default=None, help="short label per run")
    ap.add_argument("--out_dir", default="kai_results/plots_gmm", help="where PNGs go")
    ap.add_argument("--smooth", type=int, default=15,
                    help="moving-average window in log records (records are every "
                         "--print_freq steps; 1 = raw)")
    args = ap.parse_args()

    labels = args.labels or [None] * len(args.run_dirs)
    if len(labels) != len(args.run_dirs):
        ap.error("--labels must give one label per run dir")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [load_run(d, l) for d, l in zip(args.run_dirs, labels)]
    for run in runs:
        print(f"[load] {run['label']}: {len(run['rows'])} train records, "
              f"{len(run['eval'])} eval rows, {run['n_classes']} classes")
        if not run["rows"] and not run["eval"]:
            print(f"[warn] {run['dir']} has no metrics to plot")
            continue
        slug = run["label"].replace(" ", "_").replace("/", "_")
        print(f"[plot] {plot_overview(run, out_dir / f'{slug}_overview.png', args.smooth)}")

    if len(runs) > 1:
        print(f"[plot] {plot_compare(runs, out_dir / 'compare_overview.png', args.smooth)}")
        print(f"[plot] {plot_per_class_clock(runs, out_dir / 'per_class_clock.png', args.smooth)}")
    print(f"[plot] {write_summary(runs, out_dir / 'summary.md')}")


if __name__ == "__main__":
    main()
