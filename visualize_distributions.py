"""
PCA + t-SNE for N labeled feature sources. First source is treated as the
reference for the FD numbers (typically 'real').
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.decomposition import PCA

try:
    from openTSNE import TSNE
    USE_OPENTSNE = True
except ImportError:
    from sklearn.manifold import TSNE
    USE_OPENTSNE = False


def frechet_distance(mu_a, sig_a, mu_b, sig_b, eps=1e-6):
    diff = mu_a - mu_b
    covm, _ = linalg.sqrtm(sig_a @ sig_b, disp=False)
    if not np.isfinite(covm).all():
        off = np.eye(sig_a.shape[0]) * eps
        covm, _ = linalg.sqrtm((sig_a + off) @ (sig_b + off), disp=False)
    if np.iscomplexobj(covm):
        covm = covm.real
    return float(diff @ diff + np.trace(sig_a + sig_b - 2 * covm))


def gaussian_stats(X):
    return X.mean(0), np.cov(X, rowvar=False)


def fit_tsne(X, perplexity=30, seed=42):
    if USE_OPENTSNE:
        return np.asarray(TSNE(perplexity=perplexity, max_iter=750,
                               random_state=seed, n_jobs=-1).fit(X))
    return TSNE(perplexity=perplexity, max_iter=1000, random_state=seed,
                init="pca").fit_transform(X)


DEFAULT_COLORS = ["tab:green", "tab:blue", "tab:orange", "tab:red",
                  "tab:purple", "tab:brown", "tab:pink", "tab:olive"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", required=True)
    p.add_argument("--judges", nargs="+", required=True,
                   help="Sub-folder names under features_dir")
    p.add_argument("--sources", nargs="+", required=True,
                   help="Source labels in plotting order. First is reference for FD.")
    p.add_argument("--output_dir", default="./vis_plots_multi")
    p.add_argument("--tsne_perplexity", type=int, default=30)
    p.add_argument("--tsne_n_per_class", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    colors = DEFAULT_COLORS[:len(args.sources)]

    fig, axes = plt.subplots(len(args.judges), 2,
                             figsize=(13, 5 * len(args.judges)), squeeze=False)
    summary = []

    for r, judge in enumerate(args.judges):
        jdir = Path(args.features_dir) / judge
        feats = {s: np.load(jdir / f"{s}.npy").astype(np.float32) for s in args.sources}
        n = min(len(v) for v in feats.values())
        feats = {s: v[:n] for s, v in feats.items()}

        ref = args.sources[0]
        mu_r, sig_r = gaussian_stats(feats[ref])
        bits = [f"[{judge}]"]
        for s in args.sources[1:]:
            mu_s, sig_s = gaussian_stats(feats[s])
            fd = frechet_distance(mu_r, sig_r, mu_s, sig_s)
            bits.append(f"FD({ref}, {s})={fd:.2f}")
        line = "  ".join(bits)
        summary.append(line); print(line)

        X   = np.concatenate([feats[s] for s in args.sources], 0)
        lbl = np.concatenate([np.full(n, i) for i in range(len(args.sources))]).astype(int)

        # PCA on full data
        pca   = PCA(n_components=2, random_state=args.seed).fit(X)
        Z_pca = pca.transform(X)
        evr   = pca.explained_variance_ratio_
        ax = axes[r, 0]
        for i, s in enumerate(args.sources):
            pts = Z_pca[lbl == i]
            ax.scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.25,
                       c=colors[i], label=s, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{judge}  |  PCA  (EVR: {evr[0]*100:.1f}% / {evr[1]*100:.1f}%)")
        leg = ax.legend(markerscale=4, fontsize=8, loc="best")
        for h in leg.legend_handles: h.set_alpha(1.0)

        # t-SNE on subsample for visual clarity
        rng = np.random.default_rng(args.seed)
        per = args.tsne_n_per_class
        keep = np.concatenate([
            rng.choice(np.where(lbl == i)[0],
                       min(per, (lbl == i).sum()), replace=False)
            for i in range(len(args.sources))
        ])
        Z_tsne = fit_tsne(X[keep], perplexity=args.tsne_perplexity, seed=args.seed)
        ax = axes[r, 1]
        for i, s in enumerate(args.sources):
            pts = Z_tsne[lbl[keep] == i]
            ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.4,
                       c=colors[i], label=s, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{judge}  |  t-SNE  (perp={args.tsne_perplexity}, "
                     f"n={per}/source)")
        leg = ax.legend(markerscale=4, fontsize=8, loc="best")
        for h in leg.legend_handles: h.set_alpha(1.0)

    fig.suptitle(f"Distribution comparison: {'  '.join(args.sources)}", y=1.005)
    fig.tight_layout()
    fig.savefig(out / "distributions_multi.png", dpi=160, bbox_inches="tight")
    fig.savefig(out / "distributions_multi.pdf",            bbox_inches="tight")
    (out / "fd_summary.txt").write_text("\n".join(summary) + "\n")
    print(f"\nSaved {out/'distributions_multi.png'}")


if __name__ == "__main__":
    main()