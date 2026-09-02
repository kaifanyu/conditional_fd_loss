"""
Compare three feature distributions: real vs pre-train vs post-train.

Produces a comprehensive set of diagnostic plots for visualizing
distribution shift in feature space — useful for showing how a
generative model's features migrate toward the real distribution
during training.

Generates 10 individual PNGs plus a summary JSON:
    01_pca_scatter.png         - PCA scatter in real-Σ eigenbasis (3 colors)
    02_umap.png                - Joint UMAP (PCA-50 → fit on real)
    03_mahalanobis.png         - Mahalanobis distance histograms vs χ²(D)
    04_eigenspectrum.png       - Eigenvalue spectrum of Σ, log-y
    05_moments.png             - Per-dim mean and variance scatter
    06_sliced_wasserstein.png  - Per-direction 1D Wasserstein histogram
    07_nn_distances.png        - Nearest-neighbor distance distributions
    08_prdc.png                - Precision/Recall + Density/Coverage
    09_c2st.png                - Classifier two-sample test accuracy
    10_qq.png                  - Per-dim Q-Q plots for top-variance dims

Usage:
    python analyze_three_distributions.py \\
        --real path/to/real.npy \\
        --pre  path/to/pre_train_gen.npy \\
        --post path/to/post_train_gen.npy \\
        --out_dir analysis_output \\
        [--n_samples 20000]

Dependencies:
    numpy, scipy, scikit-learn, matplotlib  (required)
    umap-learn                              (optional; UMAP plot skipped if missing)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import sqrtm, solve_triangular
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

EPS = 1e-6

# Consistent palette across all plots
COLORS = {"real": "#444444", "pre": "#d62728", "post": "#1f77b4"}
LABELS = {"real": "real", "pre": "pre-train", "post": "post-train"}


# ============================================================
# Data loading
# ============================================================

def load_features(path):
    p = Path(path)
    if p.suffix == ".npz":
        d = np.load(p, allow_pickle=True)
        for k in ("features", "feat", "feats", "x", "data"):
            if k in d.files:
                return np.asarray(d[k]).astype(np.float64)
        return np.asarray(d[d.files[0]]).astype(np.float64)
    return np.load(p).astype(np.float64)


def subsample(X, n, seed=0):
    if X.shape[0] <= n:
        return X
    rng = np.random.default_rng(seed)
    return X[rng.choice(X.shape[0], n, replace=False)]


# ============================================================
# Sufficient statistics & FD
# ============================================================

def gaussian_stats(X):
    mu = X.mean(axis=0)
    Xc = X - mu
    sig = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    return mu, sig


def fd_decompose(mu_g, sig_g, mu_r, sig_r):
    D = mu_g.shape[0]
    mean_t = float(np.sum((mu_g - mu_r) ** 2))
    cm = sqrtm((sig_g + EPS * np.eye(D)) @ (sig_r + EPS * np.eye(D)))
    if np.iscomplexobj(cm):
        cm = cm.real
    cov_t = float(np.trace(sig_g + sig_r - 2 * cm))
    return mean_t + cov_t, mean_t, cov_t


def get_whitener(mu_r, sig_r):
    D = mu_r.shape[0]
    L = np.linalg.cholesky(sig_r + EPS * np.eye(D))
    return mu_r, L


def whiten(X, mu, L):
    return solve_triangular(L, (X - mu).T, lower=True).T


def mahalanobis_sq(X, mu, L):
    w = whiten(X, mu, L)
    return (w ** 2).sum(axis=1)


# ============================================================
# Plot 1: PCA scatter in real-Σ eigenbasis
# ============================================================

def plot_joint_pca(features, mu_r, sig_r, out_path, n_per=5000):
    eigvals, eigvecs = np.linalg.eigh(sig_r)
    order = np.argsort(eigvals)[::-1]
    V2 = eigvecs[:, order[:2]]
    pc_var = np.clip(eigvals[order[:2]], 1e-12, None)

    fig, ax = plt.subplots(figsize=(8, 7))
    for src in ("real", "pre", "post"):
        if src not in features:
            continue
        X = subsample(features[src], n_per, seed=42)
        Z = (X - mu_r) @ V2
        ax.scatter(Z[:, 0], Z[:, 1], s=6, alpha=0.35,
                   c=COLORS[src], label=f"{LABELS[src]} (N={X.shape[0]})")

    theta = np.linspace(0, 2 * np.pi, 200)
    for k in (1, 2, 3):
        ax.plot(k * np.sqrt(pc_var[0]) * np.cos(theta),
                k * np.sqrt(pc_var[1]) * np.sin(theta),
                "k-", lw=1.0, alpha=0.55 - 0.12 * (k - 1))
    ax.set_xlabel(f"PC1  (real λ={pc_var[0]:.2f})")
    ax.set_ylabel(f"PC2  (real λ={pc_var[1]:.2f})")
    ax.set_title("PCA in real-Σ eigenbasis   |   ellipses = real 1σ/2σ/3σ")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal", "datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plot 2: Joint UMAP (PCA-50 pre-reduce → fit on real)
# ============================================================

def plot_joint_umap(features, out_path, n_per=3000):
    try:
        import umap
    except ImportError:
        print("  [skip] umap-learn not installed (pip install umap-learn)")
        return

    samples = {s: subsample(features[s], n_per, seed=42)
               for s in ("real", "pre", "post") if s in features}
    if "real" not in samples:
        return

    # PCA pre-reduce (fit on real, transform all): speeds UMAP + denoises
    n_pca = min(50, samples["real"].shape[1])
    pca = PCA(n_components=n_pca, random_state=0)
    reduced = {"real": pca.fit_transform(samples["real"])}
    for src in ("pre", "post"):
        if src in samples:
            reduced[src] = pca.transform(samples[src])

    print(f"  [umap] PCA-{n_pca} → UMAP fit on {reduced['real'].shape[0]} real points ...")
    t0 = time.time()
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0)
    emb = {"real": reducer.fit_transform(reduced["real"])}
    for src in ("pre", "post"):
        if src in reduced:
            emb[src] = reducer.transform(reduced[src])
    print(f"  [umap] done in {time.time() - t0:.1f}s")

    fig, ax = plt.subplots(figsize=(9, 8))
    for src in ("real", "pre", "post"):
        if src in emb:
            ax.scatter(emb[src][:, 0], emb[src][:, 1],
                       s=4, alpha=0.45, c=COLORS[src], label=LABELS[src])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Joint UMAP  (PCA-50 pre-reduce, fit on real, transform others)")
    ax.legend(loc="upper right", fontsize=10, markerscale=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plot 3: Mahalanobis distance histograms vs χ²(D)
# ============================================================

def plot_mahalanobis(features, mu_r, L, out_path):
    D = mu_r.shape[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    # χ²(D) reference: theoretical target if gen ~ N(μ_r, Σ_r)
    lo = max(1, D - 6 * np.sqrt(2 * D))
    hi = D + 6 * np.sqrt(2 * D)
    xs = np.linspace(lo, hi, 400)
    ax.plot(xs, stats.chi2(D).pdf(xs), "k--", lw=1.5,
            label=f"χ²(D={D})  (target)")
    for src in ("real", "pre", "post"):
        if src not in features:
            continue
        d2 = mahalanobis_sq(features[src], mu_r, L)
        ax.hist(d2, bins=80, density=True, alpha=0.45, color=COLORS[src],
                label=f"{LABELS[src]}  med={np.median(d2):.0f}")
    ax.set_xlabel("Mahalanobis distance²  (under real Σ)")
    ax.set_ylabel("density")
    ax.set_title("Mahalanobis distance: gen vs real reference Gaussian")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plot 4: Eigenvalue spectrum
# ============================================================

def plot_eigenspectrum(features, sig_r, out_path):
    eigs = {"real": np.sort(np.linalg.eigvalsh(sig_r))[::-1]}
    for src in ("pre", "post"):
        if src in features:
            _, s = gaussian_stats(features[src])
            eigs[src] = np.sort(np.linalg.eigvalsh(s))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for src in ("real", "pre", "post"):
        if src not in eigs:
            continue
        axes[0].plot(eigs[src], c=COLORS[src], lw=1.3, label=LABELS[src])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("eigenvalue rank")
    axes[0].set_ylabel("eigenvalue")
    axes[0].set_title("Eigenvalue spectrum of Σ  (log-y)")
    axes[0].legend(fontsize=10)

    # Ratio plot: gen/real per rank
    for src in ("pre", "post"):
        if src in eigs:
            ratio = eigs[src] / np.maximum(eigs["real"], 1e-12)
            axes[1].plot(ratio, c=COLORS[src], lw=1.2, label=f"{LABELS[src]} / real")
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("eigenvalue rank")
    axes[1].set_ylabel("λ_gen / λ_real")
    axes[1].set_title("Spectrum ratio  (1.0 = match)")
    axes[1].legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plot 5: Per-dim mean and variance scatter
# ============================================================

def plot_moment_scatter(features, mu_r, sig_r, out_path):
    var_r = np.clip(np.diag(sig_r), 1e-12, None)
    panels = [s for s in ("pre", "post") if s in features]
    if not panels:
        return
    fig, axes = plt.subplots(2, len(panels), figsize=(5.5 * len(panels), 9),
                             squeeze=False)
    for i, src in enumerate(panels):
        mu_g, sig_g = gaussian_stats(features[src])
        var_g = np.clip(np.diag(sig_g), 1e-12, None)

        ax = axes[0, i]
        ax.scatter(mu_r, mu_g, s=3, alpha=0.45, c=COLORS[src])
        lim = [min(mu_r.min(), mu_g.min()), max(mu_r.max(), mu_g.max())]
        ax.plot(lim, lim, "k--", lw=1)
        delta = float(np.sum((mu_g - mu_r) ** 2))
        ax.set_xlabel("μ_real per dim")
        ax.set_ylabel(f"μ_{LABELS[src]} per dim")
        ax.set_title(f"per-dim mean: {LABELS[src]} vs real  |  ||Δ||²={delta:.2f}")

        ax = axes[1, i]
        ax.loglog(var_r, var_g, ".", ms=2, alpha=0.45, c=COLORS[src])
        lim = [min(var_r.min(), var_g.min()), max(var_r.max(), var_g.max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("var_real (diag Σ_r)")
        ax.set_ylabel(f"var_{LABELS[src]} (diag Σ_g)")
        ax.set_title(f"per-dim variance: {LABELS[src]} vs real")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plot 6: Sliced Wasserstein
# ============================================================

def sliced_wasserstein_per_direction(X, Y, n_proj=256, seed=0):
    """Returns array of per-direction 1D Wasserstein-1 distances."""
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    P = rng.standard_normal((D, n_proj))
    P /= np.linalg.norm(P, axis=0, keepdims=True)
    Xp = np.sort(X @ P, axis=0)
    Yp = np.sort(Y @ P, axis=0)
    n = min(Xp.shape[0], Yp.shape[0])
    if Xp.shape[0] != n:
        idx = np.linspace(0, Xp.shape[0] - 1, n).astype(int); Xp = Xp[idx]
    if Yp.shape[0] != n:
        idx = np.linspace(0, Yp.shape[0] - 1, n).astype(int); Yp = Yp[idx]
    return np.mean(np.abs(Xp - Yp), axis=0)


def plot_sliced_wasserstein(features, out_path, n=10000, n_proj=256):
    if "real" not in features:
        return None
    X_real = subsample(features["real"], n, seed=0)
    results = {}
    fig, ax = plt.subplots(figsize=(9, 5))
    for src in ("pre", "post"):
        if src not in features:
            continue
        X_gen = subsample(features[src], n, seed=1)
        per_dir = sliced_wasserstein_per_direction(X_real, X_gen, n_proj=n_proj)
        results[src] = float(per_dir.mean())
        ax.hist(per_dir, bins=40, alpha=0.55, color=COLORS[src],
                label=f"{LABELS[src]}  SWD={per_dir.mean():.3f}")
    ax.set_xlabel("per-direction 1D Wasserstein-1")
    ax.set_ylabel("count")
    ax.set_title(f"Sliced Wasserstein over {n_proj} random projections")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return results


# ============================================================
# Plot 7: Nearest-neighbor distance distributions
# ============================================================

def plot_nn_distances(features, out_path, n=5000):
    if "real" not in features:
        return
    X_real = subsample(features["real"], n, seed=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    nn = NearestNeighbors(n_neighbors=2).fit(X_real)
    # real → real (excluding self)
    d_rr = nn.kneighbors(X_real, return_distance=True)[0][:, 1]
    ax.hist(d_rr, bins=60, alpha=0.5, density=True, color=COLORS["real"],
            label=f"real → real  med={np.median(d_rr):.2f}")
    for src in ("pre", "post"):
        if src not in features:
            continue
        X_gen = subsample(features[src], n, seed=1)
        d_gr = nn.kneighbors(X_gen, return_distance=True)[0][:, 0]
        ax.hist(d_gr, bins=60, alpha=0.45, density=True, color=COLORS[src],
                label=f"{LABELS[src]} → real  med={np.median(d_gr):.2f}")
    ax.set_xlabel("distance to nearest real neighbor")
    ax.set_ylabel("density")
    ax.set_title("Nearest-neighbor distance distributions")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plot 8: Precision / Recall / Density / Coverage
# ============================================================

def compute_prdc(real, gen, k=5):
    """Kynkäänniemi P/R + Naeem D/C."""
    nn_r = NearestNeighbors(n_neighbors=k + 1).fit(real)
    nn_g = NearestNeighbors(n_neighbors=k + 1).fit(gen)
    rad_r = nn_r.kneighbors(real, return_distance=True)[0][:, k]
    rad_g = nn_g.kneighbors(gen,  return_distance=True)[0][:, k]
    DM_gr = pairwise_distances(gen, real)     # (Ng, Nr)
    DM_rg = DM_gr.T                            # (Nr, Ng)
    inside_real_ball = DM_gr < rad_r[None, :]  # gen inside any real ball
    inside_gen_ball  = DM_rg < rad_g[None, :]
    precision = float(inside_real_ball.any(axis=1).mean())
    density   = float(inside_real_ball.sum(axis=1).mean() / k)
    recall    = float(inside_gen_ball.any(axis=1).mean())
    coverage  = float((DM_rg.min(axis=1) < rad_r).mean())
    return {"precision": precision, "recall": recall,
            "density": density, "coverage": coverage}


def plot_prdc(features, out_path, n=5000, k=5):
    if "real" not in features:
        return None
    X_real = subsample(features["real"], n, seed=0)
    results = {}
    for src in ("pre", "post"):
        if src not in features:
            continue
        X_gen = subsample(features[src], n, seed=1)
        results[src] = compute_prdc(X_real, X_gen, k=k)
    if not results:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    pairs = [("recall", "precision", "Precision vs Recall"),
             ("coverage", "density", "Density vs Coverage")]
    for ax, (xl, yl, ti) in zip(axes, pairs):
        for src, d in results.items():
            ax.scatter(d[xl], d[yl], s=220, c=COLORS[src],
                       label=f"{LABELS[src]}  ({xl}={d[xl]:.2f}, {yl}={d[yl]:.2f})",
                       zorder=3, edgecolors="white", linewidths=1.5)
        ax.set_xlim(-0.02, 1.05)
        # density/coverage can exceed 1
        ymax = max(1.05, max(d[yl] for d in results.values()) * 1.1)
        ax.set_ylim(-0.02, ymax)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(ti)
        ax.axhline(1, color="g", lw=0.5, ls=":")
        ax.axvline(1, color="g", lw=0.5, ls=":")
        ax.legend(fontsize=10, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return results


# ============================================================
# Plot 9: Classifier two-sample test
# ============================================================

def compute_c2st(real, gen, hidden=64, seed=0):
    X = np.concatenate([real, gen], 0)
    y = np.concatenate([np.zeros(len(real)), np.ones(len(gen))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                           random_state=seed, stratify=y)
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=200,
                        random_state=seed, early_stopping=True)
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def plot_c2st(features, out_path, n=5000):
    if "real" not in features:
        return None
    X_real = subsample(features["real"], n, seed=0)
    results = {}
    for src in ("pre", "post"):
        if src not in features:
            continue
        X_gen = subsample(features[src], n, seed=1)
        results[src] = compute_c2st(X_real, X_gen)
    if not results:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    srcs = list(results.keys())
    vals = [results[s] for s in srcs]
    bars = ax.bar([LABELS[s] for s in srcs], vals,
                  color=[COLORS[s] for s in srcs], width=0.55)
    ax.axhline(0.5, color="g", lw=1.2, ls="--", label="indistinguishable (0.5)")
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("C2ST test accuracy")
    ax.set_title("Classifier two-sample test  (closer to 0.5 → distributions match)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return results


# ============================================================
# Plot 10: Q-Q plots for top-variance dims
# ============================================================

def plot_qq(features, sig_r, out_path, n_dims=6, n_samp=5000):
    if "real" not in features:
        return
    var_r = np.diag(sig_r)
    top = np.argsort(var_r)[::-1][:n_dims]
    cols = 3
    rows = (n_dims + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows),
                             squeeze=False)
    X_real = subsample(features["real"], n_samp, seed=0)
    samples_gen = {s: subsample(features[s], n_samp, seed=1)
                   for s in ("pre", "post") if s in features}
    for i, d in enumerate(top):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        real_sorted = np.sort(X_real[:, d])
        q_grid = np.linspace(0, 1, len(real_sorted))
        for src, X_gen in samples_gen.items():
            gen_sorted = np.sort(X_gen[:, d])
            q_gen = np.interp(q_grid, np.linspace(0, 1, len(gen_sorted)),
                              gen_sorted)
            ax.plot(real_sorted, q_gen, ".", ms=2, alpha=0.5,
                    c=COLORS[src], label=LABELS[src])
        lim = [real_sorted.min(), real_sorted.max()]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("real quantile")
        ax.set_ylabel("gen quantile")
        ax.set_title(f"dim {d} (λ-rank #{i + 1})")
        ax.legend(fontsize=8)
    # hide unused axes
    for j in range(n_dims, rows * cols):
        axes[j // cols, j % cols].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main orchestrator
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help=".npy file of real features")
    ap.add_argument("--pre",  required=True, help=".npy file of pre-train gen features")
    ap.add_argument("--post", required=True, help=".npy file of post-train gen features")
    ap.add_argument("--out_dir", default="dist_analysis")
    ap.add_argument("--n_samples", type=int, default=20000,
                    help="per-source cap for the heavier plots; FD uses everything")
    ap.add_argument("--skip_umap", action="store_true")
    ap.add_argument("--skip_c2st", action="store_true")
    ap.add_argument("--skip_prdc", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    print("Loading features ...")
    features_full = {}
    for src, path in [("real", args.real), ("pre", args.pre), ("post", args.post)]:
        X = load_features(path)
        features_full[src] = X
        print(f"  {src:5s}  shape={X.shape}  dtype={X.dtype}")

    # FD on full sets; everything else on the capped subset
    print("\nComputing Gaussian sufficient statistics (full sets) ...")
    stats_full = {s: gaussian_stats(X) for s, X in features_full.items()}
    mu_r, sig_r = stats_full["real"]
    mu_r_w, L = get_whitener(mu_r, sig_r)

    summary = {"shapes": {s: list(X.shape) for s, X in features_full.items()}}

    print("\nFD vs real  (mean / cov split):")
    for src in ("pre", "post"):
        mu_g, sig_g = stats_full[src]
        total, m_t, c_t = fd_decompose(mu_g, sig_g, mu_r, sig_r)
        summary[f"FD_{src}"] = {"total": total, "mean": m_t, "cov": c_t}
        print(f"  {src:5s}  total={total:>9.3f}   mean={m_t:>9.3f}   cov={c_t:>9.3f}")

    # ---- Subsample for plot routines ----
    features = {s: subsample(X, args.n_samples, seed=0)
                for s, X in features_full.items()}

    # ---- Plots ----
    print(f"\nGenerating plots → {out}/")
    plot_joint_pca(features, mu_r, sig_r, out / "01_pca_scatter.png")
    print("  [done] 01_pca_scatter")

    if not args.skip_umap:
        plot_joint_umap(features, out / "02_umap.png")
        print("  [done] 02_umap")

    plot_mahalanobis(features, mu_r_w, L, out / "03_mahalanobis.png")
    print("  [done] 03_mahalanobis")

    plot_eigenspectrum(features, sig_r, out / "04_eigenspectrum.png")
    print("  [done] 04_eigenspectrum")

    plot_moment_scatter(features, mu_r, sig_r, out / "05_moments.png")
    print("  [done] 05_moments")

    swd = plot_sliced_wasserstein(features, out / "06_sliced_wasserstein.png")
    print("  [done] 06_sliced_wasserstein")
    if swd:
        summary["SWD"] = swd

    plot_nn_distances(features, out / "07_nn_distances.png")
    print("  [done] 07_nn_distances")

    if not args.skip_prdc:
        prdc = plot_prdc(features, out / "08_prdc.png")
        print("  [done] 08_prdc")
        if prdc:
            summary["PRDC"] = prdc

    if not args.skip_c2st:
        c2st = plot_c2st(features, out / "09_c2st.png")
        print("  [done] 09_c2st")
        if c2st:
            summary["C2ST"] = c2st

    plot_qq(features, sig_r, out / "10_qq.png")
    print("  [done] 10_qq")

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary → {out}/summary.json")
    print(f"All figures → {out}/")
    print("\nKey numbers:")
    for src in ("pre", "post"):
        fd = summary.get(f"FD_{src}", {})
        swd_v = summary.get("SWD", {}).get(src, float("nan"))
        c2 = summary.get("C2ST", {}).get(src, float("nan"))
        prd = summary.get("PRDC", {}).get(src, {})
        print(f"  {LABELS[src]:11s}  FD={fd.get('total', float('nan')):7.2f}  "
              f"SWD={swd_v:6.3f}  C2ST={c2:.3f}  "
              f"P={prd.get('precision', float('nan')):.2f}  "
              f"R={prd.get('recall', float('nan')):.2f}  "
              f"D={prd.get('density', float('nan')):.2f}  "
              f"C={prd.get('coverage', float('nan')):.2f}")


if __name__ == "__main__":
    main()