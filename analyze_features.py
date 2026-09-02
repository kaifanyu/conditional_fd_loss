"""Analyze saved generated features against real reference statistics.

Produces a multi-panel figure with:
  1. PCA scatter of generated features projected onto real Σ's top-2 eigenvectors,
     with real Gaussian iso-contours (1σ, 2σ, 3σ) overlaid.
  2. Per-dim mean scatter:  μ_real (x-axis)  vs  μ_gen (y-axis).
  3. Per-dim variance scatter:  diag(Σ_real)  vs  diag(Σ_gen), log-log.
  4. Marginal histograms for the top-variance dims of real Σ, with the
     corresponding real-Gaussian density overlaid.

Also prints the FD decomposition into mean and covariance contributions.

Usage:
    python analyze_features.py \\
        --gen   path/to/gen_features.npz \\
        --real  data/fid_stats/guided_diffusion_stats.npz \\
        --out   analysis_inception.png
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

EPS = 1e-6


def load_features(path):
    """Load generated features as an (N, D) float64 array."""
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p, allow_pickle=True)
        for k in ("features", "feat", "feats", "x", "data"):
            if k in data.files:
                return np.asarray(data[k])
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]])
        raise KeyError(
            f"Could not find features in {p}. Keys: {data.files}. "
            "Pass the right key explicitly via the loader."
        )
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix in {".pt", ".pth"}:
        import torch
        data = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            for k in ("features", "feat", "feats", "x"):
                if k in data:
                    v = data[k]
                    return v.numpy() if hasattr(v, "numpy") else np.asarray(v)
            raise KeyError(f"Could not find features key in {p}. Keys: {list(data.keys())}")
        if hasattr(data, "numpy"):
            return data.numpy()
        return np.asarray(data)
    raise ValueError(f"Unsupported file type: {p.suffix}")


def load_real_stats(path):
    """Load (mu, sigma) from a stats .npz file."""
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    mu_key = next((k for k in ("mu", "mean") if k in keys), None)
    sig_key = next((k for k in ("sigma", "cov") if k in keys), None)
    if mu_key is None or sig_key is None:
        raise KeyError(f"Need mu/sigma in {path}. Found: {data.files}")
    mu = np.asarray(data[mu_key]).astype(np.float64).reshape(-1)
    sig = np.asarray(data[sig_key]).astype(np.float64)
    return mu, sig


def fid_decompose(mu_g, sig_g, mu_r, sig_r):
    """Return (total_FD, mean_term, cov_term)."""
    D = mu_g.shape[0]
    mean_term = float(np.sum((mu_g - mu_r) ** 2))
    sig_g_e = sig_g + np.eye(D) * EPS
    sig_r_e = sig_r + np.eye(D) * EPS
    covmean = sqrtm(sig_g_e @ sig_r_e)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    cov_term = float(np.trace(sig_g_e + sig_r_e - 2 * covmean))
    return mean_term + cov_term, mean_term, cov_term


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="generated features file (.npz/.npy/.pt)")
    ap.add_argument("--real", required=True, help="real stats .npz with mu/sigma")
    ap.add_argument("--out", default="analysis.png", help="output figure path")
    ap.add_argument("--top_dims", type=int, default=3,
                    help="number of top-variance real dims for histograms (max 3 shown)")
    args = ap.parse_args()

    X = load_features(args.gen).astype(np.float64)
    print(f"[gen]  shape={X.shape}")
    mu_r, sig_r = load_real_stats(args.real)
    print(f"[real] mu={mu_r.shape}  sigma={sig_r.shape}")

    if X.shape[1] != mu_r.shape[0]:
        raise ValueError(
            f"Dim mismatch: gen D={X.shape[1]} vs real D={mu_r.shape[0]}. "
            "Make sure you matched the right stats file to the right backbone."
        )

    N, D = X.shape
    if N < D:
        print(f"WARNING: N={N} < D={D}. Σ_g is rank-deficient; the FD value "
              "will be biased upward. Use ≥ D samples (e.g. 50k) for meaningful FD.")

    mu_g = X.mean(axis=0)
    Xc = X - mu_g
    sig_g = (Xc.T @ Xc) / max(N - 1, 1)

    total, mean_t, cov_t = fid_decompose(mu_g, sig_g, mu_r, sig_r)
    print(f"\nFD decomposition:")
    print(f"  ||μ_g - μ_r||²       = {mean_t:>12.4f}")
    print(f"  Tr(Σ_g + Σ_r - 2√(...)) = {cov_t:>12.4f}")
    print(f"  total FD              = {total:>12.4f}")

    # Project onto top-2 eigenvectors of real Σ
    eigvals_r, eigvecs_r = np.linalg.eigh(sig_r)
    order = np.argsort(eigvals_r)[::-1]
    V2 = eigvecs_r[:, order[:2]]
    Z = (X - mu_r) @ V2                       # gen samples in real-PCA coords
    pc_var = np.clip(eigvals_r[order[:2]], 1e-12, None)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Panel 1: PCA scatter
    ax = axes[0, 0]
    ax.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.45, label=f"gen  (N={N})")
    theta = np.linspace(0, 2 * np.pi, 200)
    for k in (1, 2, 3):
        ax.plot(k * np.sqrt(pc_var[0]) * np.cos(theta),
                k * np.sqrt(pc_var[1]) * np.sin(theta),
                "r-", lw=1.3, alpha=0.85 - 0.2 * (k - 1),
                label=f"real {k}σ" if k == 1 else None)
    ax.axhline(0, color="k", alpha=0.15, lw=0.5)
    ax.axvline(0, color="k", alpha=0.15, lw=0.5)
    ax.set_xlabel(f"PC1  (real λ={pc_var[0]:.2f})")
    ax.set_ylabel(f"PC2  (real λ={pc_var[1]:.2f})")
    ax.set_title("PCA on real-Σ basis")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal", "datalim")

    # Panel 2: mean per dim
    ax = axes[0, 1]
    ax.scatter(mu_r, mu_g, s=4, alpha=0.4)
    lim = [min(mu_r.min(), mu_g.min()), max(mu_r.max(), mu_g.max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("μ_real per dim")
    ax.set_ylabel("μ_gen per dim")
    ax.set_title(f"Mean per dim  |  ||Δ||² = {mean_t:.3f}")

    # Panel 3: variance per dim
    ax = axes[0, 2]
    var_r = np.clip(np.diag(sig_r), 1e-12, None)
    var_g = np.clip(np.diag(sig_g), 1e-12, None)
    ax.loglog(var_r, var_g, ".", ms=2, alpha=0.4)
    lim = [min(var_r.min(), var_g.min()), max(var_r.max(), var_g.max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("var_real (diag Σ_r)")
    ax.set_ylabel("var_gen (diag Σ_g)")
    ax.set_title("Per-dim variance (log-log)")

    # Panels 4-6: marginal histograms for top-variance real dims
    top_dims = order[: min(args.top_dims, 3)]
    for i, d in enumerate(top_dims):
        ax = axes[1, i]
        mu_d, var_d = mu_r[d], max(sig_r[d, d], 1e-12)
        std_d = np.sqrt(var_d)
        ax.hist(X[:, d], bins=30, density=True, alpha=0.55, label="gen")
        x_lo = min(X[:, d].min(), mu_d - 4 * std_d)
        x_hi = max(X[:, d].max(), mu_d + 4 * std_d)
        xs = np.linspace(x_lo, x_hi, 200)
        ys = np.exp(-((xs - mu_d) ** 2) / (2 * var_d)) / np.sqrt(2 * np.pi * var_d)
        ax.plot(xs, ys, "r-", lw=1.5, label="real  N(μ,σ²)")
        ax.set_title(f"dim {d}  (real λ-rank #{i+1})")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"{Path(args.gen).name}   vs   {Path(args.real).name}\n"
        f"N={N}   D={D}   FD={total:.3f}   "
        f"(mean={mean_t:.3f}, cov={cov_t:.3f})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nSaved figure: {args.out}")


if __name__ == "__main__":
    main()