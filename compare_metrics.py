"""

python plot_metrics.py work_dirs/v3_table_3_JiT_conditional_v2/JiT_B-fd-inception-cond0.05-v2/training_metrics.json --out curves2.png

/data/jgu/kai/FD-Loss/


Print a clean diff between two eval_metrics.json files (from eval_samples.py).

Usage:
    python compare_metrics.py kai_results/eval_out/baseline_1step/eval_metrics.json kai_results/eval_out/cond_1step/eval_metrics.json
"""
import json
import sys


def fmt_row(name, va, vb):
    d = vb - va
    arrow = "UP  " if d > 0 else ("DOWN" if d < 0 else "----")
    sign = "+" if d >= 0 else ""
    return f"  {name:<42}  {va:>10.4f}  {vb:>10.4f}  {sign}{d:>9.4f}  {arrow}"


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} A.json B.json")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        a = json.load(f)
    with open(sys.argv[2]) as f:
        b = json.load(f)

    tag_a = a.get("tag", "A")
    tag_b = b.get("tag", "B")

    print(f"\n  {'metric':<42}  {tag_a:>10}  {tag_b:>10}  {'delta':>10}")
    print("  " + "-" * 84)

    if "resnet50_in1k" in a and "resnet50_in1k" in b:
        ra, rb = a["resnet50_in1k"], b["resnet50_in1k"]
        print(fmt_row("ResNet-50 top1  (held-out arch)", ra["top1"], rb["top1"]))
        print(fmt_row("ResNet-50 top5", ra["top5"], rb["top5"]))

    if "held_out_clip" in a and "held_out_clip" in b:
        ca, cb = a["held_out_clip"], b["held_out_clip"]
        name = ca.get("clip_model", "CLIP")
        print(fmt_row(f"{name} top1  (held-out CLIP)", ca["top1"], cb["top1"]))
        print(fmt_row(f"{name} top5", ca["top5"], cb["top5"]))
        print(fmt_row(f"{name} mean log p(c|x)", ca["mean_logp_c"], cb["mean_logp_c"]))

    print()
    print("  Reading the table:")
    print("    Both top1 UP                 -> real class-fidelity gain")
    print("    CLIP UP but ResNet-50 flat   -> suspect CLIP-family gaming")
    print("    Both flat                    -> conditional term ineffective")
    print("    (Always read alongside FID — a class-fidelity win that wrecks")
    print("     FID is usually not a win.)")
    print()


if __name__ == "__main__":
    main()
