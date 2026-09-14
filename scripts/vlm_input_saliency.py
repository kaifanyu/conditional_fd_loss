"""Pixel derivatives for frozen VLM heads; no model or image updates here."""

from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def input_gradients(x, targets_local, heads, extractor):
    """Differentiate target log probabilities with respect to RGB in [0, 1].

    Use a sum over independent examples so norms do not depend on microbatch
    size. All preprocessing remains in the graph. Compute the combined loss
    derivative directly as well as the difference of separate derivatives;
    the two can disagree numerically in reduced precision.
    """
    if any(p.requires_grad for model in (heads, extractor) for p in model.parameters()):
        raise ValueError("saliency requires frozen heads and extractor parameters")
    with torch.enable_grad():
        pixels = x.detach().clone().requires_grad_(True)
        z = extractor.answer_states(pixels)[extractor.layer]
        idx = targets_local.long().view(-1, 1)
        lp = heads.p_log_probs(z).gather(1, idx).squeeze(1)
        lq = heads.q_teacher_log_probs(z).gather(1, idx).squeeze(1)
        gp = torch.autograd.grad(lp.sum(), pixels, retain_graph=True)[0]
        gq = torch.autograd.grad(lq.sum(), pixels, retain_graph=True)[0]
        gd = torch.autograd.grad((lq - lp).sum(), pixels)[0]
    result = {"x": pixels, "log_p": lp, "log_q": lq,
              "grad_log_p": gp, "grad_log_q": gq,
              "grad_log_q_minus_log_p": gd, "separate_grad_difference": gq - gp}
    if not all(torch.isfinite(v).all() for v in result.values()):
        raise RuntimeError("non-finite image, probability, or input gradient")
    return {k: v.detach().float().cpu() for k, v in result.items()}


def gradient_metrics(sample):
    # Float64 norms/cosines avoid hiding agreement when gradients are tiny.
    gp = sample["grad_log_p"].double().flatten()
    gq = sample["grad_log_q"].double().flatten()
    gd = sample["grad_log_q_minus_log_p"].double().flatten()
    np_, nq = float(gp.norm()), float(gq.norm())
    return {
        "log_p": float(sample["log_p"]), "log_q": float(sample["log_q"]),
        "p_prob": float(sample["log_p"].double().exp()),
        "q_prob": float(sample["log_q"].double().exp()),
        "p_grad_l2": np_, "q_grad_l2": nq, "delta_grad_l2": float(gd.norm()),
        "p_q_cosine": float(torch.dot(gp / np_, gq / nq)) if np_ and nq else None,
        "separate_difference_l2": float((gq - gp).norm()),
        "combined_vs_separate_error_l2": float((gd - (gq - gp)).norm()),
    }


class InputGradientLogger:
    """Save exact tensors immediately; render shared-scale saliency at the end."""

    def __init__(self, out_dir, every, image_ids, targets_global, q_source):
        self.root = Path(out_dir) / "saliency"
        self.root.mkdir(parents=True, exist_ok=True)
        self.every = every
        self.image_ids = image_ids
        self.targets_global = targets_global.tolist()
        self.q_source = q_source
        self.rows = []
        self.paths = {}

    def record(self, x, targets_local, heads, extractor, step, arm, offset, dtype):
        result = input_gradients(x, targets_local, heads, extractor)
        folder = self.root / arm
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(x.shape[0]):
            index = offset + i
            sample = {k: v[i] for k, v in result.items()}
            metrics = gradient_metrics(sample)
            path = folder / f"{self.image_ids[index]}_step{step:04d}.npz"
            np.savez_compressed(path, **{k: v.numpy() for k, v in sample.items()})
            self.paths.setdefault((arm, index), []).append((step, path, metrics))
            self.rows.append({"arm": arm, "dtype": dtype, "image_index": index,
                              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                              "target_class_global": self.targets_global[index],
                              "step": step, **metrics})
        self._write_metrics()

    def _write_metrics(self):
        with (self.root / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def finish(self):
        for (arm, index), entries in self.paths.items():
            samples = []
            for step, path, metrics in entries:
                with np.load(path) as data:
                    samples.append((step, {k: data[k] for k in data.files}, metrics))
            prefix = self.root / arm / self.image_ids[index]
            self._plot(prefix, samples, arm, index, shared_time=True)
            self._plot(prefix, samples, arm, index, shared_time=False)
        metadata = {
            "q_source": self.q_source, "interval": self.every,
            "derivative": "target-class log probability with respect to original RGB x in [0,1]",
            "reduction": "sum over independent examples (no batch averaging)",
            "heatmap": "mean absolute gradient across RGB channels",
            "delta": "direct autograd derivative of log q - log p; descent uses its negative",
            "fixed_scale": "one common linear color scale across all heads and times for each image/arm",
            "step_scale": "common scale across p, q and delta within each step; annotated raw norms",
            "zero_gradient_cosine": "null/empty because direction is undefined",
            "parameters": "VLM, p, and q frozen throughout; only image pixels optimized",
        }
        (self.root / "metadata.json").write_text(json.dumps(metadata, indent=2))
        lines = [
            "# Input-gradient saliency", "",
            "The input gradient is d log p(c|E(x))/dx (and similarly for q),",
            "including differentiable resizing/normalization and the frozen VLM.",
            "Only pixels are optimized. q is the frozen EMA teacher head.", "",
            f"q source: `{self.q_source}`. Saved every {self.every} steps, plus the final step.", "",
            "`*_fixed_scale.png` shares one linear scale across all steps and heads.",
            "`*_step_scale.png` rescales each row jointly for p/q/delta to show spatial structure.",
            "Bright pixels in a rescaled map may still have tiny absolute gradients; read the norms.",
            "These are local sensitivities, not attention weights or evidence of semantic generation.", "",
            "Each `.npz` contains the exact floating-point input, signed RGB derivatives,",
            "log probabilities, the direct combined-loss derivative, and g_q - g_p.",
            "The combined loss uses log q - log p; the pixel ascent delta arm uses the opposite sign.",
            "Metrics use raw, unnormalized derivatives. No saliency result updates a weight or pixel.", "",
            "| Arm | Step | p | q | norm g_p | norm g_q | norm g_delta | cosine p,q |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in self.rows:
            cosine = row["p_q_cosine"]
            lines.append(f"| {row['arm']} | {row['step']} | {row['p_prob']:.5g} | "
                         f"{row['q_prob']:.5g} | {row['p_grad_l2']:.4g} | "
                         f"{row['q_grad_l2']:.4g} | {row['delta_grad_l2']:.4g} | "
                         + (f"{cosine:.6f}" if cosine is not None else "undefined") + " |")
        (self.root / "README.md").write_text("\n".join(lines) + "\n")

    def _plot(self, prefix, samples, arm, index, shared_time):
        keys = ("grad_log_p", "grad_log_q", "grad_log_q_minus_log_p")
        labels = ("grad log p", "grad log q", "grad (log q - log p)")
        maps = [[np.abs(data[k]).mean(axis=0) for k in keys] for _, data, _ in samples]
        common_max = max(float(h.max()) for row in maps for h in row)
        fig, axes = plt.subplots(len(samples), 4, squeeze=False,
                                 figsize=(12, 3 * len(samples)), layout="constrained")
        for row_index, ((step, data, metrics), heatmaps) in enumerate(zip(samples, maps)):
            row = axes[row_index]
            row[0].imshow(np.clip(data["x"].transpose(1, 2, 0), 0, 1))
            row[0].set_title(f"step {step}: current image\np={metrics['p_prob']:.5g}, q={metrics['q_prob']:.5g}", fontsize=10)
            vmax = common_max if shared_time else max(float(h.max()) for h in heatmaps)
            vmax = vmax if vmax > 0 else 1.0
            for ax, heatmap, label, key in zip(row[1:], heatmaps, labels, keys):
                im = ax.imshow(heatmap, cmap="inferno", vmin=0, vmax=vmax)
                norm = np.linalg.norm(data[key].astype(np.float64))
                ax.set_title(f"{label}\nraw L2={norm:.3e}", fontsize=10)
            for ax in row:
                ax.set_axis_off()
            if not shared_time:
                fig.colorbar(im, ax=row[1:].tolist(), shrink=0.8, format="%.1e")
        if shared_time:
            fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), shrink=0.8, format="%.1e")
        scale = "fixed_scale" if shared_time else "step_scale"
        fig.suptitle(f"{arm}, {self.image_ids[index]} | mean absolute RGB input gradient\n"
                     + ("Common scale across time and heads" if shared_time
                        else "Rescaled each step; raw norms show gradient strength"))
        fig.savefig(f"{prefix}_{scale}.png", dpi=130)
        plt.close(fig)
