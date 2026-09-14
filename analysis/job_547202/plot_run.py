"""Reproduce CPU-only analysis of job 547202; never load or modify model weights."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import statistics

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-job-547202")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / "work_dirs/JiT_uncond_vlm_delta/qwen_lora_direct_fp32_cal_547202"
raw = [json.loads(s) for s in (RUN / "training_metrics.json").read_text().splitlines() if s.strip()]
# The resumed trajectory supersedes the preempted attempt's duplicate step 440.
by_step = {r["iteration"]: r for r in raw}
rows = [by_step[s] for s in sorted(by_step)]
x = np.array([r["iteration"] for r in rows])
last = rows[-1]
tail = [r for r in rows if r["iteration"] >= 2000]

def export_csv(path, records):
    fields = list(dict.fromkeys(k for r in records for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

export_csv(OUT / "metrics.csv", rows)
pc_rows, windows = [], []
for file in sorted(RUN.glob("vlm_per_class_step_*.json")):
    d = json.loads(file.read_text())
    records = d["per_class"]
    n = sum(c["count"] for c in records)
    window = {"step": d["step"], "diagnostic_batches": d["window_steps"], "samples": n}
    for k in ("p_top1", "q_top1", "probe_top1", "p_target_rank", "q_target_rank", "p_target_prob", "q_target_prob", "delta_logqp"):
        window[k] = sum(c[k] * c["count"] for c in records) / n
    windows.append(window)
    pc_rows.extend({"step": d["step"], **c} for c in records)
export_csv(OUT / "per_class.csv", pc_rows)
export_csv(OUT / "accuracy_windows.csv", windows)

checks = {
    "raw_rows": len(raw), "unique_steps": len(rows),
    "duplicate_steps": sorted(s for s in by_step if sum(r["iteration"] == s for r in raw) > 1),
    "max_abs_delta_identity_error": max(abs(r["vlm_delta_logqp"] - (r["vlm_logq_c"] - r["vlm_logp_c"])) for r in rows),
    "max_abs_weighted_identity_error": max(abs(r["vlm_delta_loss_weighted"] - r["vlm_scale_eff"] * r["vlm_delta_logqp"]) for r in rows),
    "nonfinite_values": [(r["iteration"], k) for r in rows for k, v in r.items() if isinstance(v, (int, float)) and not math.isfinite(v)],
}
assert checks["max_abs_delta_identity_error"] < 3e-6, checks
assert checks["max_abs_weighted_identity_error"] < 1e-9, checks
assert not checks["nonfinite_values"], checks
assert rows[0]["iteration"] == 0 and rows[-1]["iteration"] == 2999

summary = {
    "source": str(RUN), "source_sha256": hashlib.sha256((RUN / "training_metrics.json").read_bytes()).hexdigest(),
    "checks": checks, "final": last,
    "last_1000_logged_points": len(tail),
    "last_1000_mean": {k: statistics.mean(r[k] for r in tail) for k in last},
    "last_1000_median": {k: statistics.median(r[k] for r in tail) for k in last},
    "accuracy_windows": windows,
    "final_geometric_mean_q": math.exp(last["vlm_logq_c"]),
    "final_geometric_mean_p": math.exp(last["vlm_logp_c"]),
    "final_geometric_mean_q_over_p": math.exp(last["vlm_delta_logqp"]),
    "calibration_weight_for_025": 1e-5 * .25 / statistics.median(r["grad_ratio_vlm_fd"] for r in tail),
    "metric_semantics": {
        "vlm_logq_c/logp_c/delta/prob": "Instantaneous global means over 96 scored images, before q update; natural logs.",
        "other_metrics": "Usually trailing medians of 20 updates. Diagnostics update every 10 training steps; final step 2999 reuses diagnostics through 2990.",
        "resume_bug": "At resumed step 436, absent instant diagnostic meters were removed and later recreated with default 20-value median windows. Gradient ratios and q drift/count meters after resume are therefore smoothed.",
        "per_class": "Count-weighted actual sample aggregates, pre-update q, evaluated on training-time generated images. Not a held-out checkpoint evaluation.",
        "window_means": "Means of exported logged values, not all 3000 training batches or independent samples.",
        "fd": "Training queue feature distances, not a fresh held-out 50k-image FID evaluation.",
    },
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.grid": True,
                     "grid.alpha": .18, "savefig.facecolor": "white"})
BLUE, RED, GREEN, GRAY = "#2463a8", "#bd3d44", "#198267", "#636b75"

def series(k):
    return np.array([r.get(k, np.nan) for r in rows])

def moving(y, width=11):
    # Reset the visual smoother at the hardware/restart boundary.
    return np.array([np.mean(y[max(0, i-width+1, int(np.searchsorted(x,436)) if s>=436 else 0):i+1]) for i,s in enumerate(x)])

def axis_style(ax, ylabel, title, restart=True):
    ax.set(title=title, xlabel="Training step (zero-based)", ylabel=ylabel)
    ax.set_xlim(0, 3000)
    if restart:
        ax.axvline(436, color=GRAY, lw=1, ls=":", alpha=.7)

def finish(fig, name, foot):
    fig.text(.02, .013, foot, ha="left", va="bottom", fontsize=9, color=GRAY)
    fig.tight_layout(rect=(0, .06, 1, .94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=180)
    plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(11.8, 8.2))
fig.suptitle("Job 547202 • LoRA q approaches uniform; frozen p target scores fall", fontsize=15, fontweight="bold", y=.985)
for key, label, color in [("vlm_logq_c", "LoRA q", BLUE), ("vlm_logp_c", "Frozen p", RED)]:
    y = series(key)
    axes[0].plot(x, y, color=color, alpha=.24, lw=.8)
    axes[0].plot(x, moving(y), color=color, lw=2, label=label)
axes[0].axhline(-math.log(100), color=GRAY, ls="--", lw=1.2, label="Uniform 100-class: −4.6052")
axis_style(axes[0], "Mean log probability (nats)", "Target-class scores on the same generated images")
axes[0].legend(loc="lower left", ncol=3, fontsize=9)
axes[0].text(.985,.92, f"Last batch\nlog q = {last['vlm_logq_c']:.4f}\nlog p = {last['vlm_logp_c']:.4f}", transform=axes[0].transAxes, ha="right", va="top", bbox=dict(facecolor="white", edgecolor="#ddd", alpha=.95))
y = series("vlm_delta_logqp")
axes[1].plot(x,y,color=GREEN,alpha=.25,lw=.8)
axes[1].plot(x,moving(y),color=GREEN,lw=2,label="Mean [log q(c|x) − log p(c|x)]")
axes[1].axhline(0,color=GRAY,lw=1)
axis_style(axes[1], "Log-probability difference (nats)", "A larger positive gap does not establish better conditioning")
axes[1].text(.985,.09, f"Last batch: {last['vlm_delta_logqp']:.4f}\nMean of logged steps 2000–2999: {summary['last_1000_mean']['vlm_delta_logqp']:.4f}", transform=axes[1].transAxes, ha="right", bbox=dict(facecolor="white", edgecolor="#ddd", alpha=.95))
axes[1].legend(loc="upper left", fontsize=9)
finish(fig, "log_values", "Thin lines: recorded 96-image batch means. Bold lines: trailing mean of 11 logged points.\nDotted vertical line: resume at step 436. Duplicate step 440 keeps the resumed record. No per-image visuals were saved.")

fig, axes = plt.subplots(3,2,figsize=(13,11))
fig.suptitle("Job 547202 • Training improved feature distances, with weak class conditioning", fontsize=15,fontweight="bold",y=.988)
ax=axes[0,0]
for k,l,c in [("vlm_q_generator_entropy","q",BLUE),("vlm_p_entropy","p",RED)]:
    ax.plot(x,series(k),label=l,color=c)
ax.axhline(math.log(100),ls="--",color=GRAY,label="Uniform entropy 4.6052")
axis_style(ax,"Entropy (nats)","q spreads its probability across almost all classes")
ax.legend(fontsize=9)
ax=axes[0,1]
sx=[w["step"] for w in windows]
for k,l,c in [("q_top1","q · 100 classes",BLUE),("p_top1","p · 100 classes",RED),("probe_top1","ResNet · 1000 classes",GREEN)]:
    ax.plot(sx,[100*w[k] for w in windows],marker="o",label=l,color=c)
ax.axhline(1,ls="--",color=GRAY,label="p/q chance 1%")
ax.axhline(.1,ls=":",color=GRAY,label="Probe uniform chance 0.1%")
axis_style(ax,"Top-1 accuracy (%)","Actual counts from saved diagnostic windows",restart=False)
ax.set_ylim(0,1.65)
ax.legend(fontsize=8,ncol=2)
ax=axes[1,0]
for k,l,c in [("vlm_q_generator_target_rank","q before batch update",BLUE),("vlm_p_target_rank","p",RED),("vlm_q_student_target_rank_post","q after fitting same batch",GREEN)]:
    ax.plot(x,series(k),label=l,color=c)
ax.axhline(50.5,ls="--",color=GRAY,label="Chance rank 50.5 / 100")
axis_style(ax,"Target-class rank (lower is better)","Same-batch q fitting does not transfer to fresh batches")
ax.legend(fontsize=8,loc="lower right")
ax=axes[1,1]
ax.plot(x,series("grad_ratio_vlm_fd"),color=GREEN,label="Logged gradient ratio")
ax.axhspan(.22,.30,color=BLUE,alpha=.10,label="Configured calibration target 0.22–0.30")
axis_style(ax,"Weighted VLM / FD image-gradient norm","Late-run median ratio ≈ 0.072 at weight 1e−5")
ax.legend(fontsize=8)
ax=axes[2,0]
for k,l,c in [("fid_siglip","SigLIP",BLUE),("fid_inception","Inception",RED),("fid_mae","MAE",GREEN)]:
    ax.plot(x,series(k),label=l,color=c)
ax.set_yscale("log")
axis_style(ax,"Training feature distance (log axis)","Feature distances decrease; no held-out FID was run")
ax.legend(fontsize=9)
ax=axes[2,1]
ax.plot(x,100*series("cond_delta"),color=BLUE,label="Relative pixel L2 change")
ax.plot(x,100*series("vlm_cond_feature_delta"),color=GREEN,label="Relative VLM feature L2 change")
axis_style(ax,"Relative L2 change (%)","Fixed-noise images change weakly with the class")
ax.legend(fontsize=8)
finish(fig,"diagnostics","Diagnostic curves show saved rolling medians; after resume, gradient ratio also became a 20-diagnostic median.\nAccuracy panel uses true count-weighted windows: 672 images at step 500; 4800 images at each later point. Probe uses all 1000 classes.")

pc=[r for r in pc_rows if r["step"]==2500]
fig,axes=plt.subplots(2,1,figsize=(12,7.6))
fig.suptitle("Job 547202 • Latest saved class breakdown (step 2500, 4800 images)",fontsize=15,fontweight="bold",y=.985)
ids=[r["class_id"] for r in pc]
for k,l,c in [("q_target_prob","q",BLUE),("p_target_prob","p",RED)]:
    axes[0].plot(ids,[100*r[k] for r in pc],"o-",markersize=3,lw=.7,label=l,color=c)
axes[0].axhline(1,ls="--",color=GRAY,label="Uniform target probability 1%")
axes[0].set(yscale="log",ylabel="Mean target probability (%, log axis)",title="q is near 1% across classes; p is uneven")
axes[0].legend(fontsize=9)
axes[1].bar(ids,[r["delta_logqp"] for r in pc],width=7,color=GREEN)
axes[1].set(xlabel="ImageNet class ID (trained subset: 0, 10, …, 990)",ylabel="Mean log q − log p (nats)",title="All 100 class-average gaps are positive")
for ax in axes: ax.set_xlim(-10,1000)
finish(fig,"per_class","Each class mean uses its saved sample count. q is scored before its update on those images.\nThese are numeric diagnostics of generated images; the images themselves were not saved. Last 499 training steps are absent from this class dump.")
print(json.dumps({"output":str(OUT),"checks":checks,"latest_accuracy_window":windows[-1],"geometric_p":summary['final_geometric_mean_p'],"geometric_q":summary['final_geometric_mean_q'],"geometric_q_over_p":summary['final_geometric_mean_q_over_p'],"calibration_weight":summary['calibration_weight_for_025']},indent=2))
