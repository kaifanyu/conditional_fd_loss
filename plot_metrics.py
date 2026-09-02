#!/usr/bin/env python
"""Plot the no-wandb training metrics that conditional_main_fd.py already dumps.

MetricLogger writes one JSON object per print_freq steps to
``<log_dir>/training_metrics.json`` (see logging_util.py:dump_in_output_file).
This reads that JSONL file and graphs the conditional-correction signals.

Usage:
    python plot_metrics.py work_dirs/.../training_metrics.json
    python plot_metrics.py <file> --keys logp_c logq_raw loss fid_inception
    python plot_metrics.py <file> --out curves.png          # save instead of show
    python plot_metrics.py <a.json> <b.json> --labels v3 v4  # overlay runs
"""
import argparse
import json
import math

import matplotlib.pyplot as plt


# Keys are grouped onto panels by shared y-scale (log-prob vs [0,1] prob vs nats
# vs rank), so curves on the same panel are directly comparable.
DEFAULT_PANELS = [
    ("conditional log-probs: log p(y|x) vs log q(y|x) (no floor)",
        ["logp_c", "logq_raw", "l_cond"]),
    ("q_phi confidence q(y|x): fresh vs buffer (overfit gap)",
        ["q_prob_y", "q_prob_y_buf", "q_top1"]),
    ("q_phi fresh-batch accuracy (compare to qphi_acc on buffer)",
        ["q_acc_fresh"]),
    ("q_phi entropy (nats; uniform = ln(1000) = 6.91)", ["q_entropy"]),
    ("q_phi rank of intended class (0 = argmax-correct)", ["q_rank_y"]),
    ("loss / fid", ["loss", "fid_inception"]),
    ("FD loss: raw (Frechet dist) vs self-normalized", ["fd_loss_raw", "fd_loss_norm"]),
    ("weighted loss terms (what actually enters the loss sum)",
        ["fd_loss_norm", "p_loss_weighted", "q_loss_weighted"]),
    ("per-term image-grad norm: who drives the update (symlog)",
        ["grad_x_fd", "grad_x_p", "grad_x_q"]),
    ("update cosines: p/q cancellation (-1) & conflict with FD pull",
        ["cos_update_p_q", "cos_update_fd_p", "cos_update_fd_q"]),
    ("schedules", ["lambda_eff", "alpha_eff"]),
    ("q_phi head (buffer): CE / acc", ["qphi_ce", "qphi_acc"]),
    ("optimization", ["grad_norm"]),
]

# helpful horizontal references for the log-prob panel
UNIFORM_1000 = -math.log(1000)   # -6.908  (p/q can't tell the class: log(1/1000))
CAP_50PCT = math.log(0.5)        # -0.693  (cond_target_logp example: 50% conf)


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None and not (isinstance(r[key], float) and math.isnan(r[key])):
            xs.append(r.get("iteration", len(xs)))
            ys.append(r[key])
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="one or more training_metrics.json")
    ap.add_argument("--labels", nargs="+", default=None, help="legend label per file")
    ap.add_argument("--keys", nargs="+", default=None,
                    help="flat list of keys -> one panel (overrides default panels)")
    ap.add_argument("--out", default=None, help="save to this path instead of showing")
    ap.add_argument("--start", type=float, default=None,
                    help="drop iterations < START (e.g. 10000 to skip the early spike)")
    ap.add_argument("--end", type=float, default=None, help="drop iterations > END")
    args = ap.parse_args()

    def in_range(r):
        it = r.get("iteration", 0)
        return ((args.start is None or it >= args.start) and
                (args.end is None or it <= args.end))

    runs = [(lbl or f, [r for r in load(f) if in_range(r)])
            for f, lbl in zip(args.files,
                              args.labels or [None] * len(args.files))]

    panels = ([("metrics", args.keys)] if args.keys else DEFAULT_PANELS)
    # drop panels whose keys have no data in any run (e.g. q_phi panels for a
    # *_ponly run) so the figure isn't padded with blank axes.
    def has_data(keys):
        return any(series(rows, k)[0] for _, rows in runs for k in keys)
    panels = [p for p in panels if has_data(p[1])]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.1 * len(panels)),
                             squeeze=False)
    axes = axes[:, 0]

    for ax, (title, keys) in zip(axes, panels):
        for key in keys:
            for lbl, rows in runs:
                xs, ys = series(rows, key)
                if xs:
                    tag = f"{key}" if len(runs) == 1 else f"{key} [{lbl}]"
                    ax.plot(xs, ys, label=tag, lw=1.3)
        if "logp_c" in keys or "logq_raw" in keys:
            ax.axhline(UNIFORM_1000, ls=":", c="grey", lw=0.8,
                       label="uniform (-6.91)")
            ax.axhline(CAP_50PCT, ls="--", c="grey", lw=0.8, label="50% conf (-0.69)")
        # per-term grad norms span orders of magnitude (FD is choked, p/q are not),
        # so a symlog axis keeps them all readable while tolerating exact zeros.
        if any(k.startswith("grad_x") for k in keys):
            ax.set_yscale("symlog", linthresh=1e-4)
        # cosines are bounded in [-1,1]; mark the cancel (-1) and orthogonal (0) lines.
        if any(k.startswith("cos_") for k in keys):
            ax.axhline(0.0, ls=":", c="grey", lw=0.8)
            ax.axhline(-1.0, ls="--", c="red", lw=0.8, label="perfect oppose (-1)")
            ax.set_ylim(-1.1, 1.1)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=130)
        print(f"saved -> {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
