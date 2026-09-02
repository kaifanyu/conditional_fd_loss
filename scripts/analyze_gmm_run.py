"""Summarise a conditional_main_fd_gmm.py run for the log p / log q experiment.

Reads ``training_metrics.json`` (one JSON object per line) and ``eval_summary.csv``
from one or more work_dirs and prints the meters that decide whether the term is
doing what it was designed to do.  Point it at several arms to get a matched
comparison.

The four questions it answers, in order of what would kill the run first:

1. **Is the conditional term actually driving?**  ``grad_ratio_p_fd`` -- the
   image-space gradient of the GMM term over the FD term's.  << 0.1 means the
   term is a rounding error; >> 1 means realism has lost.
2. **Is classification being learned?**  ``probe_top1`` (held-out ResNet-50,
   never in the loss) is the arbiter.  ``gmm_top1`` is the in-loss classifier's
   own opinion; the two diverging is the signature of the GMM being gamed.
3. **Is diversity surviving?**  ``gmm_within_trace_ratio`` should hold or rise;
   falling is collapse.  ``gmm_class_mean_spread`` should rise from ~0 toward
   the "ideal generator" floor measured by scripts/validate_class_gmm.py.
4. **What did it cost?**  Real FID from ``eval_summary.csv``.

Usage:
    python scripts/analyze_gmm_run.py work_dirs/JiT_uncond_gmm_20class/*
    python scripts/analyze_gmm_run.py --window 2000 --csv out.csv <dirs...>
"""

import argparse
import csv
import json
import os
import sys

KEYS = [
    ("grad_ratio_p_fd", "grad p/fd"),
    ("gmm_logp_c", "log p(c|x)"),
    ("gmm_top1", "gmm top1"),
    ("gmm_cls_sat_frac", "cap frac"),
    ("probe_top1", "probe top1"),
    ("probe_top5", "probe top5"),
    ("probe_rank", "probe rank"),
    ("cond_delta", "cond delta"),
    ("gmm_within_trace_ratio", "within"),
    ("gmm_class_mean_spread", "spread"),
    ("gmm_class_mean_mse", "mean mse"),
    ("gmm_cond_kl", "cond KL"),
    ("fid_inception", "train FD inc"),
    ("grad_norm", "grad norm"),
]


def load_metrics(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a partially flushed final line while the run is live
    return rows


def mean_of(rows, key):
    vals = [r[key] for r in rows if key in r and r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(value, width=11):
    if value is None:
        return " " * (width - 3) + "n/a"
    return f"{value:>{width}.4f}"


def load_eval(path):
    """Best (lowest) FID per step across EMA labels."""
    if not os.path.exists(path):
        return {}
    best = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                step, fid = int(row["step"]), float(row["fid"])
            except (ValueError, TypeError, KeyError):
                continue
            if step not in best or fid < best[step][0]:
                best[step] = (fid, row.get("ema_label", ""))
    return best


def summarise(run_dir, window):
    metrics_path = os.path.join(run_dir, "training_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    rows = load_metrics(metrics_path)
    if not rows:
        return None
    last_step = max(r.get("iteration", 0) for r in rows)
    recent = [r for r in rows if r.get("iteration", 0) >= last_step - window]
    first = [r for r in rows if r.get("iteration", 0) <= window]
    return {
        "name": os.path.basename(run_dir.rstrip("/")),
        "steps": last_step,
        "n": len(rows),
        "first": {k: mean_of(first, k) for k, _ in KEYS},
        "last": {k: mean_of(recent, k) for k, _ in KEYS},
        "eval": load_eval(os.path.join(run_dir, "eval_summary.csv")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--window", type=int, default=2000,
                    help="steps averaged at each end of the run (default 2000)")
    ap.add_argument("--csv", default=None, help="also write the table to this path")
    args = ap.parse_args()

    runs = [s for s in (summarise(d, args.window) for d in args.run_dirs) if s]
    if not runs:
        sys.exit("no run directory contained a readable training_metrics.json")

    label_w = max(12, max(len(r["name"]) for r in runs) + 2)
    for run in runs:
        print(f"\n=== {run['name']}  ({run['steps']:,} steps, {run['n']} logged points) ===")
        print(f"{'meter':>{label_w}} {'first ' + str(args.window):>11} "
              f"{'last ' + str(args.window):>11} {'delta':>11}")
        print("-" * (label_w + 36))
        for key, label in KEYS:
            a, b = run["first"][key], run["last"][key]
            delta = None if (a is None or b is None) else b - a
            print(f"{label:>{label_w}} {fmt(a)} {fmt(b)} {fmt(delta)}")
        if run["eval"]:
            print(f"\n{'real FID (best EMA per step)':>{label_w}}")
            for step in sorted(run["eval"]):
                fid, ema = run["eval"][step]
                print(f"{step:>{label_w},} {fid:>11.4f}   [{ema}]")

    if len(runs) > 1:
        print(f"\n\n=== matched comparison (last {args.window} steps) ===")
        header = f"{'meter':>{label_w}}" + "".join(f"{r['name'][:14]:>15}" for r in runs)
        print(header)
        print("-" * len(header))
        for key, label in KEYS:
            cells = "".join(
                ("            n/a" if r["last"][key] is None else f"{r['last'][key]:>15.4f}")
                for r in runs
            )
            print(f"{label:>{label_w}}{cells}")
        print(f"{'final FID':>{label_w}}" + "".join(
            ("            n/a" if not r["eval"]
             else f"{r['eval'][max(r['eval'])][0]:>15.4f}")
            for r in runs
        ))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "steps", "meter", f"first_{args.window}", f"last_{args.window}"])
            for run in runs:
                for key, label in KEYS:
                    w.writerow([run["name"], run["steps"], key,
                                run["first"][key], run["last"][key]])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
