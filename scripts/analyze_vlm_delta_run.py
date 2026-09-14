"""Summarise a conditional_main_fd_vlm_delta.py run.

Reads ``training_metrics.json`` (one JSON object per line) plus, when present,
``eval_summary.csv`` and the periodic ``vlm_per_class_step_*.json`` dumps, and
prints the trajectory that decides whether the sampled-label
``log q(c|z) - log p(c|z)`` field taught conditioning.

The three columns to read together are

    vlm_p_top1        the frozen real-data VLM head's view of the sampled label
    vlm_q_generator_top1 the online generator-side head the loss actually uses
    probe_top1        the held-out ResNet-50, never in the loss -- the arbiter

``vlm_*`` rising while ``probe_top1`` stays at chance is VLM-space exploitation,
not conditioning, and is reported as such.

Usage:
    python scripts/analyze_vlm_delta_run.py work_dirs/JiT_uncond_vlm_delta/<run>
    python scripts/analyze_vlm_delta_run.py --calibration <run>      # Step D
    python scripts/analyze_vlm_delta_run.py --rows 20 --csv out.csv <run> [<run> ...]
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys


# (metric key, column header, format width) -- the §19 trajectory table
TRAJECTORY = [
    ("iteration", "step", 8),
    ("samples_per_class", "smp/cls", 9),
    ("probe_top1", "prb_t1", 8),
    ("probe_rank", "prb_rnk", 10),
    ("cond_delta", "cond_d", 8),
    ("vlm_cond_feature_delta", "vlm_cd", 8),
    ("vlm_p_top1", "p_t1", 8),
    ("vlm_p_target_rank", "p_rank", 8),
    ("vlm_p_target_logp", "p_logp", 8),
    ("vlm_q_generator_top1", "q_t1", 8),
    ("vlm_q_generator_target_rank", "q_rank", 8),
    ("vlm_q_generator_target_logp", "q_logp", 8),
    ("vlm_delta_logqp", "dlt_qp", 8),
    ("vlm_pq_full_kl_qp", "KL(q|p)", 8),
    ("q_generator_weight_delta_l2", "q_drft", 8),
    ("grad_ratio_vlm_fd", "g_ratio", 8),
    ("cos_update_fd_vlm", "cos", 9),
]

# meters printed in the "first/last window" comparison
SUMMARY_KEYS = [
    ("grad_ratio_vlm_fd", "grad vlm/fd"),
    ("cos_update_fd_vlm", "cos fd/vlm"),
    ("grad_x_fd", "|grad_x fd|"),
    ("grad_x_vlm_delta", "|grad_x vlm|"),
    ("vlm_delta_logqp", "logq-logp"),
    ("vlm_delta_logqp_std", "logq-logp std"),
    ("vlm_delta_logqp_p10", "logq-logp p10"),
    ("vlm_delta_abs_p99", "|delta| p99"),
    ("vlm_logp_c", "log p(c|z)"),
    ("vlm_logq_c", "log q(c|z)"),
    ("vlm_p_top1", "p top1"),
    ("vlm_p_target_rank", "p rank"),
    ("vlm_q_generator_top1", "generator q top1"),
    ("vlm_q_generator_target_rank", "generator q rank"),
    ("vlm_q_student_top1_pre", "q_student t1 fresh"),
    ("vlm_q_student_ce_pre", "q_student ce fresh"),
    ("vlm_pq_full_kl_qp", "KL(q||p)"),
    ("vlm_pq_top1_agreement", "p/q top1 agree"),
    ("q_generator_weight_delta_l2", "generator q drift"),
    ("q_student_weight_delta_l2", "q_student drift"),
    ("q_teacher_student_kl", "KL(stud||teach)"),
    ("q_train_accuracy", "q train acc"),
    ("q_generalization_gap", "q gen gap"),
    ("q_buffer_min_samples_per_class", "buf min/class"),
    ("probe_top1", "probe top1"),
    ("probe_top5", "probe top5"),
    ("probe_rank", "probe rank"),
    ("cond_delta", "cond_delta"),
    ("vlm_cond_feature_delta", "vlm feat delta"),
    ("fid_siglip", "train FD siglip"),
    ("fid_inception", "train FD incep"),
    ("generator_grad_norm", "gen grad norm"),
    ("vlm_delta_clamp_frac", "clamp frac"),
]


def load_metrics(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                # Current runs identify the scoring q explicitly. Older logs
                # always used the teacher; direct-q drift comes from the student.
                for suffix in ("top1", "target_rank", "target_logp"):
                    key = "vlm_q_generator_" + suffix
                    if key not in row and "vlm_q_teacher_" + suffix in row:
                        row[key] = row["vlm_q_teacher_" + suffix]
                for suffix in ("weight_delta_l2", "weight_delta_rel", "weight_cos_to_p"):
                    key = "q_generator_" + suffix
                    source = "q_teacher_" + suffix if "q_teacher_" + suffix in row else "q_student_" + suffix
                    if key not in row and source in row:
                        row[key] = row[source]
                rows.append(row)
            except json.JSONDecodeError:
                continue  # partially flushed final line on a live run
    return rows


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


def series(rows, key):
    return [(r["iteration"], float(r[key])) for r in rows
            if key in r and r[key] is not None and math.isfinite(float(r[key]))]


def window_median(rows, key, lo=None, hi=None):
    vals = [v for s, v in series(rows, key)
            if (lo is None or s >= lo) and (hi is None or s <= hi)]
    return statistics.median(vals) if vals else None


def first_crossing(rows, key, threshold, above=True, sustain=3):
    """First step where *key* crosses *threshold* and stays across for *sustain* points."""
    points = series(rows, key)
    for i, (step, _) in enumerate(points):
        window = points[i:i + sustain]
        if len(window) < sustain:
            return None
        if all((v > threshold) if above else (v < threshold) for _, v in window):
            return step
    return None


def fmt(value, width=8, places=4):
    if value is None:
        return " " * (width - 3) + "n/a"
    if abs(value) >= 1e5 or (value != 0 and abs(value) < 1e-4):
        return f"{value:>{width}.1e}"
    return f"{value:>{width}.{places}f}"


def find_num_classes(run_dir, rows):
    for row in rows:
        if row.get("vlm_chance_top1"):
            return int(round(1.0 / float(row["vlm_chance_top1"])))
    init = os.path.join(run_dir, "vlm_init_diagnostics.json")
    if os.path.exists(init):
        with open(init) as f:
            return int(json.load(f)["num_classes"])
    for path in sorted(glob.glob(os.path.join(run_dir, "vlm_per_class_step_*.json"))):
        with open(path) as f:
            return int(json.load(f)["num_classes"])
    return None


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def print_trajectory(rows, rows_wanted):
    points = [r for r in rows if "grad_ratio_vlm_fd" in r] or rows
    if not points:
        return
    stride = max(1, len(points) // max(1, rows_wanted))
    selected = points[::stride]
    if points[-1] is not selected[-1]:
        selected.append(points[-1])
    header = "".join(f"{h:>{w}}" for _, h, w in TRAJECTORY)
    print(header)
    print("-" * len(header))
    for row in selected:
        cells = []
        for key, _, width in TRAJECTORY:
            value = row.get(key)
            if key == "iteration":
                cells.append(f"{int(value):>{width},}" if value is not None
                             else " " * width)
            elif key == "samples_per_class":
                cells.append(f"{int(value):>{width},}" if value is not None
                             else " " * (width - 3) + "n/a")
            else:
                cells.append(fmt(None if value is None else float(value), width))
        print("".join(cells))


def print_milestones(rows, num_classes):
    """First step each conditioning signal clears chance and stays clear.

    Top-1 on a ~96-sample batch against 100 classes is quantised at 1/96, so
    "2x chance" is 2 correct samples -- which happens by luck often enough to
    fire on a completely dead run.  Top-1 milestones therefore need 3x chance
    sustained over 5 consecutive diagnostic points; the ranks are continuous and
    far less noisy, so they keep the tighter rule.
    """
    if not num_classes:
        print("\n  (class count unknown; milestone thresholds skipped)")
        return
    chance_top1 = 1.0 / num_classes
    chance_rank = (num_classes + 1) / 2.0
    print(f"\n  milestones (C={num_classes}, chance top1={chance_top1:.4f}, "
          f"chance rank={chance_rank:.1f}):")
    baseline_cd = window_median(rows, "cond_delta", hi=1000) or 0.0
    t1 = lambda k: first_crossing(rows, k, 3 * chance_top1, sustain=5)  # noqa: E731
    rank = lambda k: first_crossing(rows, k, 0.9 * chance_rank, above=False)  # noqa: E731
    checks = [
        ("probe_top1 > 3x chance", t1("probe_top1")),
        ("probe_rank < 0.9x chance", rank("probe_rank")),
        ("vlm_p_top1 > 3x chance", t1("vlm_p_top1")),
        ("vlm_p_target_rank < 0.9x chance", rank("vlm_p_target_rank")),
        ("vlm_q_generator_top1 > 3x chance", t1("vlm_q_generator_top1")),
        ("vlm_q_generator_target_rank < 0.9x chance", rank("vlm_q_generator_target_rank")),
        # the _pre read is the honest one: that batch has not entered the
        # replay buffer yet, so the student cannot have trained on it
        ("vlm_q_student_top1_pre > 3x chance (fresh batch)",
         t1("vlm_q_student_top1_pre")),
        (f"cond_delta > 2x its first-1k value ({baseline_cd:.4f})",
         first_crossing(rows, "cond_delta", max(2 * baseline_cd, 0.02), sustain=5)),
    ]
    for label, step in checks:
        print(f"    {label:<48} {'step ' + format(step, ',') if step is not None else 'never'}")


def print_diagnosis(rows, num_classes, window):
    last = max((r["iteration"] for r in rows), default=0)
    lo = max(0, last - window)
    med = lambda k, a=None, b=None: window_median(rows, k, a, b)  # noqa: E731

    print("\n  diagnosis:")
    ratio = med("grad_ratio_vlm_fd", lo)
    if ratio is None:
        print("    grad_ratio_vlm_fd    not logged")
    else:
        band = ("IN BAND" if 0.22 <= ratio <= 0.30 else
                "below band (the term may be a rounding error)" if ratio < 0.22 else
                "above band (realism may be losing)")
        print(f"    grad_ratio_vlm_fd    {ratio:.4f} sustained  -> {band} (target 0.22-0.30)")

    p_top1, probe_top1 = med("vlm_p_top1", lo), med("probe_top1", lo)
    if p_top1 is not None and probe_top1 is not None and num_classes:
        chance = 1.0 / num_classes
        if p_top1 > 4 * chance and probe_top1 < 2 * chance:
            print(f"    EXPLOITATION WARNING  vlm_p_top1={p_top1:.4f} is well above "
                  f"chance while probe_top1={probe_top1:.4f} is not. The generator is "
                  f"likely gaming the VLM feature space rather than learning classes.")
        elif p_top1 <= 2 * chance and probe_top1 <= 2 * chance:
            print(f"    conditioning           not started: vlm_p_top1 {p_top1:.4f}, "
                  f"probe_top1 {probe_top1:.4f}, chance {chance:.4f}")
        else:
            print(f"    conditioning           vlm_p_top1 {p_top1:.4f} and probe_top1 "
                  f"{probe_top1:.4f} (chance {chance:.4f}) -- moving together")

    first_d, last_d = med("vlm_delta_logqp", None, window), med("vlm_delta_logqp", lo)
    p10_last = med("vlm_delta_logqp_p10", lo)
    min_last = med("vlm_delta_logqp_min", lo)
    p99_last, p99_first = med("vlm_delta_abs_p99", lo), med("vlm_delta_abs_p99", None, window)
    if last_d is not None:
        trend = ("falling" if first_d is not None and last_d < first_d - 1e-6
                 else "stable/rising")
        print(f"    logq-logp            {first_d if first_d is None else round(first_d, 4)}"
              f" -> {last_d:.4f}  ({trend})")
        # The documented pathology is specifically the NEGATIVE tail: the fitted-GMM
        # sampled-label scalar reached -25 because minimising log q(c|z) - log p(c|z)
        # rewards samples q assigns near-zero mass to, which is unbounded below.
        # A large POSITIVE tail is the opposite situation -- p is confidently wrong
        # about a hard sample and the driver is pulling on it, which is the term
        # working. Report the two separately or the flag cries wolf.
        if p10_last is not None and p99_last is not None:
            grew = p99_first not in (None, 0) and p99_last > 3 * p99_first
            if p10_last < -1.0 or (min_last is not None and min_last < -10.0):
                note = ("  <-- NEGATIVE TAIL: this is the documented divergence mode "
                        "(unbounded below). Watch generator_grad_norm and consider "
                        "--vlm_delta_clamp")
            elif grew:
                note = ("  (positive tail grew >3x: p is confidently wrong on hard "
                        "samples and the driver is pulling on them -- not the "
                        "divergence mode, but check FID and generator_grad_norm)")
            else:
                note = ""
            min_txt = "" if min_last is None else f", min {min_last:.4f}"
            print(f"    tails                p10 {p10_last:.4f}{min_txt}, "
                  f"|delta| p99 {p99_last:.4f}{note}")

    drift_t, drift_s = med("q_generator_weight_delta_l2", lo), med("q_student_weight_delta_l2", lo)
    cos_t = med("q_generator_weight_cos_to_p", lo)
    if drift_t is not None:
        rel = med("q_generator_weight_delta_rel", lo)
        note = ("q head is near p; inspect LoRA drift too" if drift_t < 1e-3 else
                "q head has moved from p")
        rel_txt = "" if rel is None else f" ({rel:.1%} of ||W_p||)"
        student_txt = "n/a" if drift_s is None else f"{drift_s:.4f}"
        cos_txt = "" if cos_t is None else f", generator q cos_to_p {cos_t:.4f}"
        print(f"    q drift from p       generator q {drift_t:.4f}{rel_txt}, "
              f"student {student_txt}{cos_txt}  -- {note}")

    gap = med("q_generalization_gap", lo)
    ce_fresh = med("vlm_q_student_ce_pre", lo)
    if gap is not None and num_classes:
        uniform = math.log(num_classes)
        verdict = ("MEMORISING -- q scores far better on its replay buffer than on "
                   "fresh samples, so log q(c|z) is noise where the loss reads it. "
                   "Lower the reuse factor (--vlm_q_batch_size) or raise "
                   "--vlm_q_weight_decay" if gap > 1.0 else "healthy")
        extra = ("" if ce_fresh is None else
                 f", fresh CE {ce_fresh:.3f} vs uniform {uniform:.3f}")
        print(f"    q generalisation     gap {gap:.4f}{extra}  -- {verdict}")

    clamp = med("vlm_delta_clamp_frac", lo)
    if clamp:
        print(f"    CLAMP ACTIVE         {clamp:.4f} of samples clamped -- the "
              f"objective was changed; report this")


def print_per_class(run_dir, top_n):
    paths = sorted(glob.glob(os.path.join(run_dir, "vlm_per_class_step_*.json")))
    if not paths:
        return
    with open(paths[-1]) as f:
        payload = json.load(f)
    records = [r for r in payload["per_class"] if r.get("count", 0) > 0]
    if not records:
        return
    records.sort(key=lambda r: (-(r.get("p_top1") or 0.0), r.get("p_target_rank") or 1e9))
    print(f"\n  per-class at step {payload['step']:,} "
          f"({len(paths)} dumps, window {payload['window_steps']} diag steps):")
    learned = [r for r in records if (r.get("p_top1") or 0) > 0]
    print(f"    classes with any p top-1 hit: {len(learned)}/{len(records)}")
    print(f"    {'class':>7} {'n':>5} {'p_t1':>7} {'q_t1':>7} {'probe':>7} "
          f"{'p_rank':>8} {'delta':>9}")
    for record in records[:top_n]:
        print(f"    {record['class_id']:>7} {int(record['count']):>5} "
              f"{fmt(record.get('p_top1'), 7)} {fmt(record.get('q_top1'), 7)} "
              f"{fmt(record.get('probe_top1'), 7)} "
              f"{fmt(record.get('p_target_rank'), 8, 2)} "
              f"{fmt(record.get('delta_logqp'), 9)}")


def print_calibration(rows, args):
    """Step D: pick the conditional weight from the sustained gradient ratio."""
    points = series(rows, "grad_ratio_vlm_fd")
    if not points:
        sys.exit("no grad_ratio_vlm_fd in this run -- was it launched with diagnostics?")
    last = points[-1][0]
    print("\n  grad_ratio_vlm_fd trajectory (the term starts at exactly 0 because "
          "q == p):")
    print(f"    {'window':>18} {'n':>5} {'median':>9} {'mean':>9} {'min':>9} {'max':>9}")
    edges = [0, 250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 10 ** 9]
    for lo, hi in zip(edges, edges[1:]):
        vals = [v for s, v in points if lo <= s < hi]
        if not vals:
            continue
        label = f"{lo:,}-{min(hi, last):,}"
        print(f"    {label:>18} {len(vals):>5} {statistics.median(vals):>9.4f} "
              f"{statistics.fmean(vals):>9.4f} {min(vals):>9.4f} {max(vals):>9.4f}")
    lo = args.calibration_from if args.calibration_from is not None else max(0, last - 1000)
    window = [v for s, v in points if s >= lo]
    if not window:
        sys.exit(f"no grad_ratio_vlm_fd samples at or after step {lo}")
    observed = statistics.median(window)
    print(f"\n    calibration window   steps >= {lo:,}  ->  median {observed:.4f} "
          f"({len(window)} samples)")
    if args.weight is None:
        print("    pass --weight <the WEIGHT this run used> to get the rescale")
        return
    if observed <= 0:
        sys.exit("observed ratio is 0 -- q has not moved from p yet; run longer")
    for target in (0.22, args.target, 0.30):
        print(f"    target {target:.3f}  ->  WEIGHT = {args.weight:g} * {target:.3f} / "
              f"{observed:.4f} = {args.weight * target / observed:.6g}")
    print("\n    The injected gradient is exactly linear in --vlm_delta_weight, so the "
          "\n    rescale is exact *at a fixed q*. It is not exact across the run: q keeps"
          "\n    drifting and |grad_x_fd| grows, so re-read the sustained value after"
          "\n    launching and be ready for one iteration.")


# ---------------------------------------------------------------------------

def summarise(run_dir, window):
    metrics_path = os.path.join(run_dir, "training_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    rows = load_metrics(metrics_path)
    if not rows:
        return None
    last = max(r.get("iteration", 0) for r in rows)
    recent = [r for r in rows if r.get("iteration", 0) >= last - window]
    first = [r for r in rows if r.get("iteration", 0) <= window]
    return {
        "dir": run_dir,
        "name": os.path.basename(run_dir.rstrip("/")),
        "rows": rows,
        "steps": last,
        "n": len(rows),
        "num_classes": find_num_classes(run_dir, rows),
        "first": {k: window_median(first, k) for k, _ in SUMMARY_KEYS},
        "last": {k: window_median(recent, k) for k, _ in SUMMARY_KEYS},
        "eval": load_eval(os.path.join(run_dir, "eval_summary.csv")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--window", type=int, default=2000,
                    help="steps averaged at each end of the run (default 2000)")
    ap.add_argument("--rows", type=int, default=25,
                    help="trajectory rows to print (default 25)")
    ap.add_argument("--per_class_top", type=int, default=15)
    ap.add_argument("--calibration", action="store_true",
                    help="Step D mode: report the grad_ratio_vlm_fd trajectory and "
                         "the rescaled weight")
    ap.add_argument("--calibration_from", type=int, default=None,
                    help="first step of the calibration window (default: last 1000)")
    ap.add_argument("--weight", type=float, default=None,
                    help="--vlm_delta_weight this run used, for the rescale")
    ap.add_argument("--target", type=float, default=0.25,
                    help="target sustained grad_ratio_vlm_fd (default 0.25)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    runs = [s for s in (summarise(d, args.window) for d in args.run_dirs) if s]
    if not runs:
        sys.exit("no run directory contained a readable training_metrics.json")

    label_w = max(18, max(len(r["name"]) for r in runs) + 2)
    for run in runs:
        classes = ("" if not run["num_classes"] else f", C={run['num_classes']}")
        print(f"\n=== {run['name']}  ({run['steps']:,} steps, "
              f"{run['n']} logged points{classes}) ===")
        init_path = os.path.join(run["dir"], "vlm_init_diagnostics.json")
        if os.path.exists(init_path):
            with open(init_path) as f:
                init = json.load(f)
            print(f"  init: p top1 {init['p']['p_top1']:.4f} / rank "
                  f"{init['p']['p_target_rank']:.1f} (chance {init['chance_top1']:.4f} / "
                  f"{init['chance_mean_rank']:.1f}), "
                  f"max|logq-logp| {max(init['init_equality'].values()):.2e}")

        if args.calibration:
            print_calibration(run["rows"], args)
            continue

        print()
        print_trajectory(run["rows"], args.rows)
        print(f"\n{'meter':>{label_w}} {'first ' + str(args.window):>13} "
              f"{'last ' + str(args.window):>13} {'delta':>13}")
        print("-" * (label_w + 42))
        for key, lbl in SUMMARY_KEYS:
            a, b = run["first"][key], run["last"][key]
            delta = None if (a is None or b is None) else b - a
            print(f"{lbl:>{label_w}} {fmt(a, 13)} {fmt(b, 13)} {fmt(delta, 13)}")
        if run["eval"]:
            print(f"\n{'real FID (best EMA per step)':>{label_w}}")
            for step in sorted(run["eval"]):
                fid, ema = run["eval"][step]
                print(f"{step:>{label_w},} {fid:>13.4f}   [{ema}]")
        print_milestones(run["rows"], run["num_classes"])
        print_diagnosis(run["rows"], run["num_classes"], args.window)
        print_per_class(run["dir"], args.per_class_top)

    if len(runs) > 1 and not args.calibration:
        print(f"\n\n=== matched comparison (last {args.window} steps) ===")
        header = f"{'meter':>{label_w}}" + "".join(f"{r['name'][:14]:>15}" for r in runs)
        print(header)
        print("-" * len(header))
        for key, lbl in SUMMARY_KEYS:
            cells = "".join(fmt(r["last"][key], 15) for r in runs)
            print(f"{lbl:>{label_w}}{cells}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", "steps", "meter", f"first_{args.window}",
                             f"last_{args.window}"])
            for run in runs:
                for key, _ in SUMMARY_KEYS:
                    writer.writerow([run["name"], run["steps"], key,
                                     run["first"][key], run["last"][key]])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
