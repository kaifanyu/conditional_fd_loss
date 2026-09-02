"""Class-conditional GMM posteriors in representation space.

Motivation
----------
The Frechet term in :mod:`frechet_distance.losses` constrains only the first
two moments of the *pooled* feature distribution: ``d + d(d+1)/2`` scalars.
A distribution supported on ``2d+1`` points can satisfy all of them exactly,
so FD -> 0 is reachable by a generator with essentially no diversity.  Adding
``-log p(c|x)`` from a real-data classifier does not fix this: its per-sample
minimiser is a point (the class mean), i.e. it is a *contractive* force that
supplies a sharper attractor for collapse rather than removing it.

This module adds the missing repulsive term by comparing two class-conditional
Gaussian models fitted in the same frozen representation space:

* ``p(c|x)`` -- posterior of a GMM fitted **once, offline, on the dataset**.
* ``q(c|x)`` -- posterior of a GMM fitted **online, on the generator's own
  samples**.

Because the generated label ``c`` is drawn first and ``x ~ q(x|c)`` second,
``q(c|x)`` is the true posterior of the generator's joint, so

    E_{(x,c)~q}[ log q(c|x) - log p(c|x) ] = E_{x~q}[ KL( q(c|x) || p(c|x) ) ] >= 0

is a genuine non-negative objective, zero iff the two models induce the same
class posteriors on the support of q.  By Bayes with a uniform class prior,

    log p(c|x) - log q(c|x) = [log p(x|c) - log q(x|c)] - [log p(x) - log q(x)]

so the marginal log-ratio cancels: this term measures class-conditional
mismatch *modulo* the global mismatch that the Frechet term already handles.
The two losses are complementary by construction, and together they raise the
resolution of the objective from ``d + d(d+1)/2`` constraints to roughly
``C x (d + d(d+1)/2)``.

Under collapse the generated per-class clusters are tight, so ``q(c|x)`` on the
generator's own samples approaches a delta while ``p(c|x)`` stays soft; the KL
grows exactly as diversity dies.

Geometry
--------
Everything lives in a **whitened PCA space** fitted once on real features:

    z = (f - mean_p) @ P,      P = V_k S_k^{-1/2}

so the real *pooled* covariance is the identity in z-space.  This is what makes
the per-class covariances well conditioned and lets both sides shrink toward a
common, meaningful target (the real pooled *within-class* covariance).

Sample-count asymmetry
----------------------
Real side: ~1300 images/class -> a full ``k x k`` per-class covariance is well
posed for ``k <= 128``.  Generated side: a 50k queue over 1000 classes is ~50
samples/class, which cannot fill a ``k x k`` covariance.  So ``q`` uses a
**tied** (pooled within-class) covariance plus per-class means, i.e. LDA on the
generated side and QDA on the real side.  Mean estimation needs far fewer
samples than covariance estimation, so this is the statistically honest split.
It also loses nothing essential: if class ``c`` collapses to a point then
``mu_c^q`` sits on that point, the Mahalanobis distance of the sample to its own
class mean goes to zero, ``log q(c|x)`` is maximal, and the penalty is maximal.

The tied ``q`` covariance is defined as the pooled **within-class** scatter,
which makes ``tr(Sigma_within^q) / tr(Sigma_within^p)`` a free, per-step
intra-class collapse meter (see :meth:`OnlineClassStats.diagnostics`).

Gradients
---------
All GMM parameters are detached; gradients flow only through ``x -> features ->
z -> quadratic form``.  For the ``q`` side this is not an approximation: with
reparameterised sampling,

    d/dtheta E[log q_theta(x)] = E[ d_x log q . d_theta x ] + E[ d_theta log q_theta ]

and the parameter-side term has zero expectation by the score identity
(``int q d_theta log q = d_theta int q = 0``).  Dropping it is unbiased.
"""

import logging
import math

import numpy as np
import torch

logger = logging.getLogger("FD_loss")


# =============================================================================
# Reference (real-data) class-conditional GMM  --  the "p" side
# =============================================================================

class ClassGMMReference(torch.nn.Module):
    """Frozen per-class Gaussians fitted offline on the dataset.

    Holds the whitening projection and QDA parameters produced by
    ``compute_class_stats.py``.  All buffers are non-trainable and the module is
    only ever used in inference mode w.r.t. its own parameters.

    Shapes: ``proj (d, k)``, ``feat_mean (d,)``, ``mu (C, k)``,
    ``prec (C, k, k)``, ``logdet (C,)``, ``log_prior (C,)``.
    """

    def __init__(self, feat_mean, proj, mu, cov, log_prior, within_cov, device="cuda",
                 class_ids=None):
        super().__init__()
        num_classes, k = mu.shape

        # Do the batched factorisation on the accelerator: 1000 Cholesky
        # decompositions of 128x128 float64 matrices take minutes on CPU (and
        # this runs at every training start-up) but well under a second on GPU.
        cov = cov.to(device)
        mu = mu.to(device)
        within_cov = within_cov.to(device)
        feat_mean, proj = feat_mean.to(device), proj.to(device)
        log_prior = log_prior.to(device)

        # Cholesky-based inverse/logdet: more stable than a raw inverse and
        # gives the log-determinant for free.
        chol = torch.linalg.cholesky(cov)
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        prec = torch.cholesky_inverse(chol)
        prec = 0.5 * (prec + prec.transpose(-1, -2))  # re-symmetrise

        self.num_classes = num_classes
        self.k = k
        self.register_buffer("feat_mean", feat_mean.float())
        self.register_buffer("proj", proj.float())
        self.register_buffer("mu", mu.float())
        self.register_buffer("logdet", logdet.float())
        self.register_buffer("log_prior", log_prior.float())

        # Expanded quadratic form, precomputed once:
        #   (z-mu_c)^T P_c (z-mu_c) = <zz^T, P_c> - 2 z.(P_c mu_c) + mu_c^T P_c mu_c
        # Contracting against the flattened precisions costs O(B k^2) memory
        # instead of the O(B C k) that a (B, C, k) difference tensor would need
        # -- an 8x saving at C=1000, k=128 -- and turns the whole thing into two
        # matmuls.
        prec_mu = torch.einsum("ckl,cl->ck", prec, mu)
        self.register_buffer("prec_flat", prec.reshape(num_classes, k * k).float())
        self.register_buffer("prec_mu", prec_mu.float())
        self.register_buffer("mu_prec_mu", (prec_mu * mu).sum(-1).float())
        # Pooled within-class covariance of the real data, in whitened space.
        # Used as the shrinkage target for the generated side and as the
        # denominator of the intra-class collapse meter.
        self.register_buffer("within_cov", within_cov.float())
        self.register_buffer("within_trace", within_cov.diagonal().sum().float())

        # Class-subset support. ``class_ids[j]`` is the *global* generator label
        # of local component j; ``label_map`` inverts it. A GMM fitted on the
        # full label set gets the identity map, so every existing stats file and
        # call site keeps working unchanged. Fitting on a subset is not the same
        # as masking a full GMM: the whitening PCA, the pooled within-class
        # covariance and the log_softmax denominator are all computed over the
        # subset only, which is what makes -log p(c|x) a genuine k-way problem
        # instead of a 1000-way one the generator can never win.
        if class_ids is None:
            class_ids = torch.arange(num_classes, dtype=torch.long)
        class_ids = class_ids.to(device).long()
        if class_ids.numel() != num_classes:
            raise ValueError(
                f"class_ids has {class_ids.numel()} entries but the GMM has {num_classes} "
                f"components"
            )
        label_map = torch.full((int(class_ids.max().item()) + 1,), -1,
                               dtype=torch.long, device=device)
        label_map[class_ids] = torch.arange(num_classes, device=device)
        self.register_buffer("class_ids", class_ids)
        self.register_buffer("label_map", label_map)
        self.is_subset = bool(class_ids.numel() != label_map.numel())

    # -- label space --------------------------------------------------------

    def to_local(self, labels: torch.Tensor) -> torch.Tensor:
        """Map global generator labels to local component indices.

        Identity when the GMM covers every class.  No validation here: it runs
        inside the training step (and possibly under ``torch.compile``), so the
        label set is checked once at start-up by :meth:`validate_labels`.
        """
        if not self.is_subset:
            return labels
        return self.label_map.index_select(0, labels)

    def validate_labels(self, class_ids) -> None:
        """Assert that every label the generator may draw has a component."""
        missing = [int(c) for c in class_ids
                   if c >= self.label_map.numel() or int(self.label_map[int(c)]) < 0]
        if missing:
            raise ValueError(
                f"the GMM has no component for generator label(s) {missing}; it was "
                f"fitted on {self.num_classes} class(es) "
                f"{self.class_ids.tolist()[:8]}{'...' if self.num_classes > 8 else ''}. "
                f"Refit with compute_class_stats.py --class_ids matching the run."
            )

    @classmethod
    def from_npz(cls, path: str, pca_dim: int | None = None, shrinkage: float = 0.25,
                 device="cuda"):
        """Load class stats written by ``compute_class_stats.py``.

        Args:
            path: ``.npz`` with ``feat_mean``, ``pca_basis``, ``class_mu``,
                ``class_cov``, ``class_count``, ``within_cov``.
            pca_dim: truncate to the leading ``pca_dim`` whitened dimensions.
                Must not exceed the stored dimensionality.
            shrinkage: convex weight of the pooled within-class covariance in
                ``Sigma_c <- (1-a) S_c + a Sigma_within``.  Guards the per-class
                covariances against the finite-sample noise of ~1300 images.
        """
        blob = np.load(path)
        required = ("feat_mean", "pca_basis", "class_mu", "class_cov",
                    "class_count", "within_cov")
        missing = [key for key in required if key not in blob]
        if missing:
            raise KeyError(
                f"{path} is missing {missing}. Regenerate it with compute_class_stats.py "
                f"(found: {list(blob.keys())})"
            )

        feat_mean = torch.from_numpy(blob["feat_mean"]).double()
        proj = torch.from_numpy(blob["pca_basis"]).double()          # (d, k_stored)
        mu = torch.from_numpy(blob["class_mu"]).double()             # (C, k_stored)
        cov = torch.from_numpy(blob["class_cov"]).double()           # (C, k, k)
        within = torch.from_numpy(blob["within_cov"]).double()       # (k, k)
        count = torch.from_numpy(blob["class_count"]).double()       # (C,)
        class_ids = (torch.from_numpy(blob["class_ids"]).long()
                     if "class_ids" in blob else None)

        k_stored = proj.shape[1]
        if pca_dim is not None:
            if pca_dim > k_stored:
                raise ValueError(
                    f"--fd_gmm_pca_dim={pca_dim} exceeds the {k_stored} dimensions stored "
                    f"in {path}"
                )
            proj = proj[:, :pca_dim]
            mu = mu[:, :pca_dim]
            cov = cov[:, :pca_dim, :pca_dim]
            within = within[:pca_dim, :pca_dim]

        cov = (1.0 - shrinkage) * cov + shrinkage * within.unsqueeze(0)
        # Uniform-ish prior taken from the empirical class frequencies. In the
        # posterior *difference* a uniform prior would cancel, but ImageNet
        # classes are not exactly balanced, so use the real frequencies for p
        # and the generator's sampling prior (uniform) for q.
        log_prior = torch.log(count / count.sum())

        module = cls(feat_mean, proj, mu, cov, log_prior, within, device=device,
                     class_ids=class_ids).to(device)
        module.eval().requires_grad_(False)
        logger.info(
            f"[GMM] Loaded reference GMM from {path}: C={module.num_classes}, "
            f"k={module.k}, shrinkage={shrinkage}, "
            f"tr(Sigma_within)/k={float(module.within_trace) / module.k:.4f}"
            + (f", class subset {module.class_ids.tolist()}" if module.is_subset else "")
        )
        return module

    # -- projection ---------------------------------------------------------

    def project(self, feats: torch.Tensor) -> torch.Tensor:
        """Map raw judge features (B, d) into whitened PCA space (B, k).

        Gradients flow through ``feats``; the projection itself is frozen.
        """
        return (feats.float() - self.feat_mean) @ self.proj

    # -- posterior ----------------------------------------------------------

    def logits(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Unnormalised per-class log-densities (B, C) for whitened features z.

        *temperature* > 1 flattens the softmax built from these logits without
        changing their ranking (so top-1 is untouched).  It exists for the same
        reason ``--clip_logit_scale`` does: in a k-dimensional whitened space the
        per-class Mahalanobis differences run to tens of nats, so the posterior
        saturates and ``-log p(c|x)`` has *no gradient at all* on anything
        resembling a real image.  Measured on the 20-class inception fit: 98.6%
        of held-out real images sit at exactly ``log p = 0`` at T=1, versus 0%
        at T=100 with top-1 unchanged.  Only the posterior is affected; the
        density-ratio term reads :meth:`log_likelihood`, which is untouched.
        """
        zz = (z.unsqueeze(2) * z.unsqueeze(1)).reshape(z.shape[0], self.k * self.k)
        maha = zz @ self.prec_flat.T - 2.0 * (z @ self.prec_mu.T) + self.mu_prec_mu
        logits = -0.5 * (maha + self.logdet) + self.log_prior
        return logits if temperature == 1.0 else logits / temperature

    def log_likelihood(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Proper Gaussian log-density ``log p(z|c)`` for the given labels, (B,).

        Unlike :meth:`logits` this includes the ``2 pi`` normaliser and excludes
        the class prior, so it is comparable across *different* Gaussians -- the
        density-ratio term needs that, the posterior does not.
        """
        diff = z - self.mu.index_select(0, labels)
        prec = self.prec_flat.index_select(0, labels).view(-1, self.k, self.k)
        maha = torch.einsum("bk,bkl,bl->b", diff, prec, diff)
        norm = self.k * math.log(2.0 * math.pi)
        return -0.5 * (maha + self.logdet.index_select(0, labels) + norm)

    @torch.no_grad()
    def posterior_score_delta(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """``grad_z log p_T(c|z)`` for the given *local* labels, (B, k).

        At ``temperature=1`` this is exactly the Bayes/CFG score delta
        ``s_p(z|c) - s_p(z)``: the prior cancels under the z-gradient, so the
        posterior score is the difference between the selected class score and
        the posterior-weighted mixture score.  Above 1 it is the corresponding
        *tempered*-posterior gradient, which is a different vector field (see
        :func:`gmm_posterior_loss`).

        Every parameter here is frozen, so the result is a detached vector
        field evaluated at the current z -- it is meant to be injected via a
        stop-gradient linear surrogate, not differentiated through.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        probs = torch.softmax(self.logits(z, temperature=temperature), dim=-1)

        # Selected-class score  s_{p,c}(z) = P_c mu_c - P_c z.
        prec_y = self.prec_flat.index_select(0, labels).view(-1, self.k, self.k)
        score_selected = (
            self.prec_mu.index_select(0, labels)
            - torch.bmm(prec_y, z.unsqueeze(-1)).squeeze(-1)
        )

        # Posterior-weighted mixture score  sum_j r_j (P_j mu_j - P_j z).
        # Contracting the responsibilities against the *flattened* precisions
        # first gives the averaged precision in one (B,C)x(C,k*k) matmul and
        # keeps the intermediate at (B, k, k) instead of the (B, C, k) tensor a
        # naive per-class difference would need.
        prec_bar = (probs @ self.prec_flat).view(-1, self.k, self.k)
        score_marginal = (
            probs @ self.prec_mu
            - torch.bmm(prec_bar, z.unsqueeze(-1)).squeeze(-1)
        )

        return (score_selected - score_marginal) / float(temperature)


# =============================================================================
# Online (generated) class-conditional statistics  --  the "q" side
# =============================================================================

class OnlineClassStats(torch.nn.Module):
    """EMA per-class means + tied within-class covariance of generated features.

    Mirrors :class:`~frechet_distance.queue.FeatureQueue`: state lives in
    registered buffers so it survives ``state_dict`` save/load and moves with
    ``.cuda()``.  Updates are ``@torch.no_grad`` and must run *outside* the
    compiled region, for the same reason the feature queue does.

    Per-class EMA with per-class debiasing: with C=1000 and a few hundred
    samples per step, an individual class is touched only every few steps.  A
    global decay would leave rarely-touched classes dominated by their
    initialisation, so each class carries its own effective sample count and is
    debiased by it (Adam-style).
    """

    def __init__(self, num_classes: int, k: int, ema_beta: float = 0.999,
                 shrinkage: float = 0.25, eps: float = 1e-4,
                 cov_ema_beta: float = None):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.ema_beta = ema_beta
        # Keep the historical shared-beta behaviour unless callers explicitly
        # request a longer horizon for the tied covariance.  This is a plain
        # attribute (like ``ema_beta``), not a buffer, so existing GMM
        # state_dicts retain exactly the same keys and strict-load unchanged.
        self.cov_ema_beta = ema_beta if cov_ema_beta is None else cov_ema_beta
        self.shrinkage = shrinkage
        self.eps = eps

        # Debiased EMA accumulators. ``mu_raw``/``w`` form a weighted running
        # mean; ``mu = mu_raw / w`` once ``w > 0``.
        self.register_buffer("mu_raw", torch.zeros(num_classes, k, dtype=torch.float64))
        self.register_buffer("w", torch.zeros(num_classes, dtype=torch.float64))
        # Tied within-class scatter: EMA of (z - mu_c)(z - mu_c)^T pooled over
        # classes, plus its own scalar weight.
        self.register_buffer("within_raw", torch.zeros(k, k, dtype=torch.float64))
        self.register_buffer("within_w", torch.zeros(1, dtype=torch.float64))
        self.register_buffer("total_seen", torch.zeros(1, dtype=torch.long))

        # Parameters of q are constants within a training step (they are
        # detached by construction), so they are materialised once per step by
        # ``refresh_cache`` *outside* the compiled region.  Keeping the Cholesky
        # out of the graph avoids an inductor fallback and means the in-place
        # EMA updates never alias a tensor the autograd graph is holding.
        self.register_buffer("mu_cache", torch.zeros(num_classes, k))
        self.register_buffer("prec_cache", torch.zeros(k, k))
        self.register_buffer("prec_mu_cache", torch.zeros(num_classes, k))
        self.register_buffer("mu_prec_mu_cache", torch.zeros(num_classes))
        self.register_buffer("logdet_cache", torch.zeros(()))

    # -- state --------------------------------------------------------------

    @property
    def initialized(self) -> bool:
        """True once every class has been observed at least once."""
        return bool((self.w > 0).all().item())

    @property
    def coverage(self) -> float:
        """Fraction of classes with at least one observation."""
        return float((self.w > 0).float().mean().item())

    def class_means(self) -> torch.Tensor:
        """Debiased per-class means (C, k). Unseen classes fall back to 0."""
        w = self.w.clamp(min=1e-12).unsqueeze(1)
        return self.mu_raw / w

    def within_covariance(self, target: torch.Tensor) -> torch.Tensor:
        """Debiased tied within-class covariance (k, k), shrunk toward *target*.

        *target* is the real pooled within-class covariance, so the shrinkage
        both regularises a small-sample estimate and floors the covariance:
        without it, a fully collapsed generator drives ``Sigma^q -> 0`` and the
        Mahalanobis term (hence the gradient) diverges at exactly the moment the
        loss matters most.
        """
        w = self.within_w.clamp(min=1e-12)
        cov = self.within_raw / w
        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * target.double()
        return cov + self.eps * torch.eye(self.k, dtype=cov.dtype, device=cov.device)

    @property
    def mean_ema_effective_samples(self) -> float:
        """Approximate finite-sample ESS of each per-class EMA mean.

        For ``n`` normalized exponential weights proportional to
        ``beta ** age``, Kish's effective sample size is

        ``(1 + beta) / (1 - beta) * (1 - beta**n) / (1 + beta**n)``.

        ``total_seen / num_classes`` supplies ``n``.  This is exact for equal
        per-class counts and is the appropriate uniform-label approximation
        when counts differ slightly.  Unlike the asymptotic expression alone,
        it does not claim a 20k-sample horizon after only ~50 observations per
        class during bootstrap.
        """
        beta = float(self.ema_beta)
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"ema_beta must be in [0, 1), got {beta}")

        n_avg = float(self.total_seen.item()) / max(1, self.num_classes)
        if n_avg <= 0.0:
            return 0.0
        if beta == 0.0:
            return 1.0

        decay_n = math.exp(n_avg * math.log(beta))
        asymptotic = (1.0 + beta) / (1.0 - beta)
        return asymptotic * (1.0 - decay_n) / (1.0 + decay_n)

    # -- update -------------------------------------------------------------

    @torch.no_grad()
    def update(self, z: torch.Tensor, y: torch.Tensor):
        """Fold a detached batch of whitened generated features into the EMA.

        Args:
            z: (B, k) whitened features, detached.
            y: (B,) class labels used to generate them.
        """
        z64 = z.detach().double()
        beta = self.ema_beta

        # Per-class decay applied only to classes present in this batch keeps
        # the effective horizon per class rather than per step.
        counts = torch.zeros(self.num_classes, dtype=torch.float64, device=z.device)
        counts.index_add_(0, y, torch.ones_like(y, dtype=torch.float64))
        sums = torch.zeros(self.num_classes, self.k, dtype=torch.float64, device=z.device)
        sums.index_add_(0, y, z64)

        decay = torch.where(counts > 0, beta ** counts, torch.ones_like(counts))
        self.mu_raw.mul_(decay.unsqueeze(1)).add_(sums * (1.0 - beta))
        self.w.mul_(decay).add_(counts * (1.0 - beta))

        # Within-class scatter uses the *current* (pre-update) class means so
        # the residual is well defined even for classes seen for the first time.
        mu = self.class_means()
        resid = z64 - mu.index_select(0, y)
        n = z64.shape[0]
        cov_beta = self.cov_ema_beta
        decay_w = cov_beta ** n
        self.within_raw.mul_(decay_w).addmm_(
            resid.T, resid, alpha=(1.0 - cov_beta),
        )
        self.within_w.mul_(decay_w).add_(n * (1.0 - cov_beta))
        self.total_seen += n

    # -- posterior ----------------------------------------------------------

    @torch.no_grad()
    def refresh_cache(self, reference: "ClassGMMReference"):
        """Materialise the current q parameters. Call once per step, outside compile."""
        cov = self.within_covariance(reference.within_cov)
        chol = torch.linalg.cholesky(cov)
        prec = torch.cholesky_inverse(chol)
        prec = 0.5 * (prec + prec.T)

        mu = self.class_means()
        # Classes not yet observed would sit at the origin and act as spurious
        # attractors in the softmax; park them on the real class mean instead,
        # which makes their q-term momentarily agree with p and contribute no
        # gradient bias.
        unseen = self.w <= 0
        if unseen.any():
            mu = torch.where(unseen.unsqueeze(1), reference.mu.double(), mu)

        self.mu_cache.copy_(mu.float())
        self.prec_cache.copy_(prec.float())
        self.prec_mu_cache.copy_((mu @ prec).float())
        self.mu_prec_mu_cache.copy_(((mu @ prec) * mu).sum(-1).float())
        self.logdet_cache.copy_(2.0 * torch.log(torch.diagonal(chol)).sum().float())

    def log_likelihood(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Proper Gaussian log-density ``log q(z|c)`` for the given labels, (B,).

        The covariance floor applied in :meth:`within_covariance` bounds this
        from above, which is what keeps the density-ratio term finite when the
        generator collapses.
        """
        diff = z - self.mu_cache.index_select(0, labels)
        maha = ((diff @ self.prec_cache) * diff).sum(-1)
        norm = self.k * math.log(2.0 * math.pi)
        return -0.5 * (maha + self.logdet_cache + norm)

    def logits(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Unnormalised per-class log-densities (B, C) under the tied-covariance q.

        Reads the cache filled by :meth:`refresh_cache`; gradients flow only
        through ``z``.  This is the unbiased pathwise gradient (see the module
        docstring): the parameter-side term has zero expectation.

        A tied covariance makes this an LDA posterior, so the quadratic form
        expands into two matmuls with no (B, C, k) intermediate:
            -0.5 (z-mu_c)^T P (z-mu_c) = z.(P mu_c) - 0.5 mu_c^T P mu_c - 0.5 z^T P z
        The ``z^T P z`` term is class-independent and cancels in the softmax, but
        is kept so the logged log-likelihoods stay interpretable.
        """
        cross = z @ self.prec_mu_cache.T                        # (B, C)
        quad_z = ((z @ self.prec_cache) * z).sum(-1, keepdim=True)
        # Uniform prior: training labels are drawn uniformly.
        logits = cross - 0.5 * self.mu_prec_mu_cache - 0.5 * quad_z - 0.5 * self.logdet_cache
        return logits if temperature == 1.0 else logits / temperature

    @torch.no_grad()
    def posterior_score_delta(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """``grad_z log q_T(c|z)`` for the given local labels, (B, k).

        q has a *tied* covariance, so the class-independent ``-P z`` half of
        each per-class score is identical across classes and cancels between
        the selected class and the posterior-weighted average.  What is left is
        a difference of natural parameters and needs no quadratic term at all.

        Reads the detached cache filled by :meth:`refresh_cache`.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        probs = torch.softmax(self.logits(z, temperature=temperature), dim=-1)
        natural_means = self.prec_mu_cache                      # (C, k) = P mu_c

        selected = natural_means.index_select(0, labels)
        expected = probs @ natural_means
        return (selected - expected) / float(temperature)

    # -- diagnostics --------------------------------------------------------

    @torch.no_grad()
    def diagnostics(self, reference: "ClassGMMReference") -> dict:
        """Collapse meters, computed from state already maintained for the loss.

        * ``within_trace_ratio``: tr(Sigma_within^q) / tr(Sigma_within^p).  The
          direct intra-class collapse meter -- < 1 and falling means the
          generator is losing within-class diversity.  Computed *before*
          shrinkage so it is not floored by the regulariser.
        * ``class_mean_mse``: mean_c ||mu_c^q - mu_c^p||^2 / k in whitened space.
          Per-class mean fidelity; rises if classes drift or merge.
        * ``class_mean_spread``: trace of the between-class scatter of q over
          that of p.  Falls if distinct classes are collapsing onto each other.
        * ``class_mean_spread_noise_floor``: spread contributed solely by
          finite-EMA noise in the generated class means.  A small positive
          spread is not evidence of class structure until it clears this floor.
        * ``class_mean_spread_to_noise``: measured spread divided by that floor.
          Values near 1 are estimator noise; values well above 1 indicate
          resolved between-class signal.
        """
        seen = self.w > 0
        if not seen.any():
            return {}

        raw_within = self.within_raw / self.within_w.clamp(min=1e-12)
        # The scatter is taken about a mean estimated from the same samples, so
        # it is biased low by (1 - 1/n) per class. n here is the effective
        # per-class sample count: the EMA horizon, or the number seen so far if
        # that is smaller. The correction matters only early -- once each class
        # has been seen ~1/(1-beta) times it is under a percent.
        horizon = 1.0 / max(1e-9, 1.0 - self.ema_beta)
        n_eff = min(float(self.total_seen.item()) / self.num_classes, horizon)
        debias = n_eff / (n_eff - 1.0) if n_eff > 1.5 else 1.0
        within_ratio = (raw_within.diagonal().sum() * debias
                        / reference.within_trace.double().clamp(min=1e-12))

        mu_q = self.class_means()[seen]
        mu_p = reference.mu.double()[seen]
        mean_mse = ((mu_q - mu_p) ** 2).sum(-1).mean() / self.k

        between_q = mu_q.var(dim=0, unbiased=False).sum()
        between_p = mu_p.var(dim=0, unbiased=False).sum()
        between_p_safe = between_p.clamp(min=1e-12)
        spread = between_q / between_p_safe

        # Each EMA class mean carries covariance Sigma_within^q / n_eff.  The
        # spread numerator is a population variance across the seen class
        # means, so centering them removes 1 / C_seen of independent estimator
        # noise.  q_within_trace uses the same unshrunk, debiased covariance as
        # the headline within-class diagnostic above.
        n_eff = self.mean_ema_effective_samples
        finite_c = 1.0 - 1.0 / float(seen.sum().item())
        q_within_trace = raw_within.diagonal().sum() * debias
        spread_noise_floor = (
            finite_c * q_within_trace / max(n_eff, 1e-12) / between_p_safe
        )
        spread_to_noise = spread / spread_noise_floor.clamp(min=1e-12)

        return {
            "gmm_within_trace_ratio": float(within_ratio),
            "gmm_class_mean_mse": float(mean_mse),
            "gmm_class_mean_spread": float(spread),
            "gmm_class_mean_spread_noise_floor": float(spread_noise_floor),
            "gmm_class_mean_spread_to_noise": float(spread_to_noise),
            "gmm_class_mean_ema_n_eff": n_eff,
            "gmm_class_coverage": self.coverage,
        }


# =============================================================================
# The loss
# =============================================================================

def gmm_posterior_loss(
    feats: torch.Tensor,
    labels: torch.Tensor,
    reference: ClassGMMReference,
    online: OnlineClassStats,
    lambda_ent: float = 1.0,
    lambda_cls: float = 1.0,
    use_q: bool = True,
    mode: str = "density",
    clamp: float = 3.0,
    normalize: bool = True,
    cls_cap: float = 0.0,
    normalize_cls: bool | None = None,
    temperature: float = 1.0,
    cls_normalization: str | None = None,
    cfg_delta_normalization: str = "none",
    cfg_delta_eps: float = 1e-6,
):
    """``lambda_cls * E[-log p(c|x)]  +  lambda_ent * <anti-collapse term>``.

    ``l_cls = E[-log p(c|x)]`` is the class-fidelity term: the sample must look
    like the class it was *asked* for, which the Frechet term cannot see at all.
    On its own it is contractive -- its per-sample minimiser is a point -- so it
    is never a substitute for the second term.

    Two anti-collapse terms are available.

    ``mode="density"`` (default): the class-conditional log density ratio

        E_{x~q(.|c)}[ log q(x|c) - log p(x|c) ]  =  KL( q(.|c) || p(.|c) ) >= 0

    whose gradient in whitened space is exactly the two forces this design is
    built around::

        d/dz = Lambda_c^p (z - mu_c^p)   -   Lambda^q (z - mu_c^q)
               \\_ pull toward the real _/   \\_ push off the generated _/
                  class mean                     class mean

    As the generator collapses, ``Sigma^q`` shrinks, ``Lambda^q`` grows, and the
    repulsion *strengthens* -- the term is strongest exactly when it is needed.
    Per-sample it can run negative (a sample far from both means), so it is
    clamped; in expectation under a q that tracks the generator it is a KL and
    therefore non-negative.

    ``mode="cfg_delta"``: not a scalar objective at all -- it injects the
    feature-space vector field

        g_cfg(z,c) = [s_q(z|c) - s_q(z)]  -  [s_p(z|c) - s_p(z)]
                   = grad_z log q(c|z)    -  grad_z log p(c|z)

    through a stop-gradient linear surrogate ``mean_i z_i . stopgrad(g_i)``.
    The two brackets are the generated and real *classifier-free-guidance
    deltas*; their difference is the conditional residual of the joint KL once
    the marginal part has been handed to the Frechet term (chain rule:
    ``KL(q(x,c)||p(x,c)) = KL(q(x)||p(x)) + E_x KL(q(c|x)||p(c|x))``).

    Four caveats an implementation must not lose:

    * FD is a moment surrogate, *not* ``KL(q(x)||p(x))``, so the marginal terms
      do not literally cancel. This mode is a hybrid, not Equation (1).
    * The derivation needs the generator's true posterior ``q*(c|x)``; the code
      only has the fitted ``q_hat(c|z)``. The direction inherits that model
      bias -- what the explicit vector form buys is that nothing here pretends
      to be a KL, and no gradient leaks into the fitted parameters.
    * At ``temperature > 1`` the Bayes identity above is exact only for the
      *tempered* posterior, i.e. it becomes ``(1/T)[s_c - sum_j r_T,j s_j]``.
      Useful (untempered high-dimensional posteriors saturate and give no
      gradient) but it changes the objective, so the outer weight must be
      recalibrated whenever T moves.
    * It does **not** inherit density mode's covariance-strengthened repulsion:
      nothing here grows as ``Sigma^q`` shrinks. It is an ablation, not a
      replacement for the anti-collapse default.

    It is also distinct from ``mode="posterior"``: that differentiates through
    the full posterior probabilities of a non-negative scalar KL, this injects
    only the sampled label's posterior *score* difference.

    ``mode="posterior"``: the class-posterior KL ``E_x[ KL(q(.|x) || p(.|x)) ]``,
    summed over the class axis so it is non-negative pointwise by Gibbs. Bounded
    and very stable, but it has a failure mode that makes it a poor default: as
    ``q(.|x)`` saturates toward a delta -- i.e. exactly as collapse sets in --
    the entropy part of the KL flattens and the term degenerates into
    ``-log p(c|x)``, the *contractive* classifier-guidance signal. Measured on
    synthetic collapse it flips the sign of the diversity update. Kept as an
    ablation and for comparison.

    Note on the sampled-label form
    ------------------------------
    An earlier version used ``log q(c_i|x_i) - log p(c_i|x_i)``. That is an
    unbiased estimator of the posterior KL only if ``q(c|x)`` is the exact
    posterior of the joint x was drawn from. The fitted q is not, so it loses
    non-negativity, is unbounded below, and rewards producing samples its own
    model *misclassifies*. On a real run it reached -25 and grew the gradient
    norm 8x within ten steps. Do not reintroduce it.

    Args:
        feats: (B, d) judge features, carrying gradient.
        labels: (B,) class labels used to generate them.
        reference: frozen real-data GMM.
        online: generated-side statistics (already bootstrapped).
        lambda_ent: weight of the anti-collapse term.
        lambda_cls: weight of the class-fidelity term.
        use_q: if False, skip the anti-collapse term (ablation).
        mode: ``"density"``, ``"posterior"`` or ``"cfg_delta"``.
        clamp: per-sample bound on the density ratio, in standard deviations
            about the batch mean.
        normalize: divide each term by its own detached magnitude, as the
            Frechet term does. Strongly recommended; see the note below.
        cls_cap: per-sample ceiling (< 0) on ``log p(c|x)``. Once a sample is
            this confident it stops contributing gradient. Necessary whenever
            ``lambda_cls`` is the *driver* rather than a regulariser: without it
            the objective is a race to adversarial certainty, which is what the
            classifier-ensemble runs guard against with ``--cond_target_logp``.
            0 disables the cap. e.g. -0.69 caps at p=50%.
        normalize_cls: legacy boolean override for the class-fidelity term.
            ``True`` selects ``cls_normalization="self"`` and ``False`` selects
            ``"none"``. Kept for API compatibility; new callers should use
            ``cls_normalization``.
        temperature: softmax temperature for the class posterior (see
            :meth:`ClassGMMReference.logits`). Above 1 it flattens a saturated
            posterior back into a regime where the class-fidelity term actually
            has a gradient. Does not affect ``mode="density"``.
        cls_normalization: class-objective scaling mode. ``"self"`` divides by
            the detached adaptive magnitude ``|l_cls| + 0.01`` (legacy default),
            ``"log_classes"`` divides by the fixed ``log(C)`` and therefore
            cannot weaken as the task gets harder, and ``"none"`` uses the raw
            negative log posterior. ``None`` preserves the legacy
            ``normalize_cls`` / ``normalize`` resolution.
        cfg_delta_normalization: ``mode="cfg_delta"`` only. ``"none"`` (default)
            injects the raw fitted-GMM field; ``"rms"`` divides the whole batch
            field by its detached RMS, which stabilises the scale but changes
            the field and is therefore a separate ablation. Unrelated to
            ``normalize``, which never applies to this mode -- the surrogate
            scalar is origin-dependent and must not be self-normalised.
        cfg_delta_eps: floor for the RMS and cosine denominators.

    Returns:
        ``(loss, parts, z_detached)`` where ``parts`` holds scalar diagnostics
        and ``z_detached`` is the whitened batch for the online update.
        ``labels`` are *global* generator labels; the mapping onto the GMM's
        (possibly subset) component axis happens here.
    """
    z = reference.project(feats)
    labels = reference.to_local(labels)
    if cls_normalization is None:
        if normalize_cls is None:
            normalize_cls = normalize
        cls_normalization = "self" if normalize_cls else "none"
    else:
        if cls_normalization not in {"self", "log_classes", "none"}:
            raise ValueError(
                "cls_normalization must be one of 'self', 'log_classes', or "
                f"'none' (got {cls_normalization!r})"
            )
        if normalize_cls is not None:
            legacy_mode = "self" if normalize_cls else "none"
            if cls_normalization != legacy_mode:
                raise ValueError(
                    f"conflicting class normalizers: normalize_cls={normalize_cls} "
                    f"selects {legacy_mode!r}, but cls_normalization="
                    f"{cls_normalization!r}"
                )

    logits_p = reference.logits(z, temperature=temperature)
    logp_post = torch.log_softmax(logits_p, dim=-1)
    idx = labels.unsqueeze(1)
    logp_c = logp_post.gather(1, idx).squeeze(1)
    capped = torch.clamp(logp_c, max=cls_cap) if cls_cap < 0 else logp_c
    l_cls = -capped.mean()

    if cls_normalization == "self":
        cls_term = _self_normalize(l_cls, True)
    elif cls_normalization == "log_classes":
        if reference.num_classes <= 1:
            raise ValueError("cls_normalization='log_classes' requires at least 2 classes")
        cls_term = l_cls / math.log(reference.num_classes)
    else:
        cls_term = l_cls

    # These two meters are the weighted components, so their sum is exactly
    # ``gmm_loss``. Previously only the saturated composite (~2 for two
    # self-normalised terms) was visible, hiding which objective had failed.
    cls_objective = lambda_cls * cls_term
    q_objective = torch.zeros((), dtype=cls_objective.dtype, device=cls_objective.device)
    loss = cls_objective
    parts = {
        "gmm_nll_p": l_cls.detach(),
        "gmm_cls_objective": cls_objective.detach(),
        # Raw (uncapped) mean log p(c|x): comparable across runs and directly
        # against the classifier-ensemble runs' ``logp_c``.
        "gmm_logp_c": logp_c.mean().detach(),
        # The in-loss classifier's own accuracy on the generated batch. Reads
        # ~1/C at init on a de-conditioned model and is the first thing to move;
        # divergence from the held-out probe's top-1 is the signature of the GMM
        # being gamed rather than satisfied.
        "gmm_top1": (logits_p.argmax(-1) == labels).float().mean().detach(),
        "gmm_cls_sat_frac": (logp_c.detach() > cls_cap).float().mean()
                            if cls_cap < 0 else torch.zeros((), device=z.device),
    }

    if use_q and lambda_ent != 0.0:
        if mode == "density":
            log_q = online.log_likelihood(z, labels)
            log_p = reference.log_likelihood(z, labels)
            ratio = log_q - log_p

            # The ratio carries a systematic negative offset: q's per-class
            # means are estimated from ~n samples, so each sample's Mahalanobis
            # distance to its own (noisy) mean is inflated by ~k/n, biasing
            # log q down. The offset does not affect the gradient -- it is a
            # constant -- but it does decide which samples a fixed clamp hits.
            # Clamping about the detached batch mean makes the clamp a genuine
            # outlier guard instead of a bias-dependent reshaping of the loss.
            # (Measured: a fixed +-20 clamp caught 44% of samples.)
            # The bound is in units of the batch's own standard deviation, not
            # raw nats. In k=128 dimensions a Mahalanobis distance has spread
            # ~sqrt(2k), so the per-sample log-ratio varies by tens of nats and
            # any fixed bound is either inert or catches most of the batch
            # (measured: +-10 nats clipped 70%). At 3 sigma this clips the tail
            # only, and stays correct if k or the feature scale changes.
            centre = ratio.detach().mean()
            width = clamp * ratio.detach().std().clamp(min=1e-6)
            deviation = ratio - centre
            term = (centre + deviation.clamp(-width, width)).mean()

            parts["gmm_cond_kl"] = term.detach()
            parts["gmm_log_qxc"] = log_q.mean().detach()
            parts["gmm_log_pxc"] = log_p.mean().detach()
            parts["gmm_clamp_frac"] = (deviation.abs() > width).float().mean().detach()
        elif mode == "posterior":
            logq_post = torch.log_softmax(online.logits(z, temperature=temperature), dim=-1)
            term = (logq_post.exp() * (logq_post - logp_post)).sum(-1).mean()
            parts["gmm_post_kl"] = term.detach()
        elif mode == "cfg_delta":
            # Both deltas are evaluated at a detached z: the field is a
            # *constant* vector at this point, injected below through a linear
            # surrogate. Differentiating through the field itself would add the
            # parameter-side term the derivation drops (it has zero
            # expectation) plus a second-order term nobody asked for.
            delta_p = reference.posterior_score_delta(
                z.detach(), labels, temperature=temperature)
            delta_q = online.posterior_score_delta(
                z.detach(), labels, temperature=temperature)
            cfg_grad_raw = delta_q - delta_p

            raw_rms = cfg_grad_raw.square().mean().sqrt()
            if cfg_delta_normalization == "none":
                cfg_grad = cfg_grad_raw
                cfg_scale = torch.ones((), device=z.device, dtype=z.dtype)
            elif cfg_delta_normalization == "rms":
                cfg_scale = raw_rms.clamp_min(cfg_delta_eps)
                cfg_grad = cfg_grad_raw / cfg_scale
            else:
                raise ValueError(
                    "cfg_delta_normalization must be 'none' or 'rms', got "
                    f"{cfg_delta_normalization!r}"
                )

            # First-order gradient injection. The scalar value below is NOT a
            # KL and has no standalone statistical meaning -- it is origin
            # dependent. Its gradient w.r.t. z is the point: each sample gets
            # exactly cfg_grad / B, matching the batch-mean convention of the
            # other two modes.
            term = (z * cfg_grad.detach()).sum(-1).mean()

            def _rms(v):
                return v.square().mean().sqrt()

            denom = (delta_q.norm(dim=-1)
                     * delta_p.norm(dim=-1)).clamp_min(cfg_delta_eps)
            alignment = ((delta_q * delta_p).sum(-1) / denom).mean()
            teacher_rms, fake_rms, error_rms = _rms(delta_p), _rms(delta_q), raw_rms

            parts["gmm_cfg_surrogate"] = term.detach()
            parts["gmm_cfg_teacher_rms"] = teacher_rms.detach()
            parts["gmm_cfg_fake_rms"] = fake_rms.detach()
            parts["gmm_cfg_error_rms"] = error_rms.detach()
            parts["gmm_cfg_relative_error"] = (
                error_rms / teacher_rms.clamp_min(cfg_delta_eps)).detach()
            parts["gmm_cfg_alignment_cos"] = alignment.detach()
            parts["gmm_cfg_vector_scale"] = cfg_scale.detach()
        else:
            raise ValueError(f"unknown fd_gmm_mode '{mode}'")

        # Self-normalise, exactly as the Frechet term does with
        # ``fid / (fid.detach() + eps)``. Without it the two terms are not
        # comparable: FD's gradient is diluted ~1/queue_size per sample and
        # divided by the FID magnitude, while this is a direct per-sample loss
        # over 128-dimensional Gaussian log-densities. Measured unnormalised at
        # weight 0.1, it produced grad_norm 81 against an FD-only baseline of
        # 0.057. The absolute value handles the sign: the term is biased
        # negative, and dividing by |term| preserves the descent direction while
        # fixing the scale.
        if mode == "cfg_delta":
            # No _self_normalize: dividing by |term| would tie the gradient
            # scale to a coordinate-dependent scalar. The vector field is
            # normalised explicitly (or not at all) above instead.
            q_objective = lambda_ent * term
        else:
            q_objective = lambda_ent * _self_normalize(term, normalize)
        loss = loss + q_objective

    parts["gmm_q_objective"] = q_objective.detach()
    parts["gmm_loss"] = loss.detach()
    return loss, parts, z.detach()


def _self_normalize(value: torch.Tensor, enabled: bool, eps: float = 1e-2):
    """``value / (|value| + eps)`` with the denominator detached. No-op if off."""
    if not enabled:
        return value
    return value / (value.detach().abs() + eps)
