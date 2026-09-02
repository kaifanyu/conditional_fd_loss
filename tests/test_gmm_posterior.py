"""Correctness checks for the class-conditional log p / log q term.

Runs on synthetic Gaussians where the right answer is known, so a failure here
is a bug in the implementation rather than a property of the generator.
"""

import os
import sys
import math
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compute_class_stats import accumulate_class_stats, build_whitening, finalize
from frechet_distance.gmm import ClassGMMReference, OnlineClassStats, gmm_posterior_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C, D, K, N_PER = 20, 64, 16, 400

# Class centers are drawn once and shared by every dataset: a "generated"
# dataset must differ from the reference in its *sampling*, not in where its
# classes live.
#
# SEP_WIDE gives near-perfectly separable classes (used to check that the
# reference classifier and the whitening are correct). SEP_OVERLAP is the
# regime the loss actually operates in: high-dimensional Gaussians are trivially
# separable, and once the posteriors saturate both log p and log q sit at 0 and
# the term has no gradient at all. That is a real property of the method, not a
# quirk of the test -- the mechanism needs residual class ambiguity to bite.
SEP_WIDE, SEP_OVERLAP = 3.0, 0.15


def _centers(separation):
    g = torch.Generator(device="cpu").manual_seed(12345)
    return torch.randn(C, D, generator=g) * separation


def _synthetic_dataset(seed=0, spread=1.0, separation=SEP_WIDE):
    """Class-conditional Gaussians in D dims: (feats, labels, class centers)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    centers = _centers(separation)
    # A shared anisotropic within-class covariance, so whitening has work to do.
    scale = torch.linspace(0.4, 1.6, D).unsqueeze(0)
    feats, labels = [], []
    for c in range(C):
        noise = torch.randn(N_PER, D, generator=g) * scale * spread
        feats.append(centers[c] + noise)
        labels.append(torch.full((N_PER,), c, dtype=torch.long))
    return torch.cat(feats), torch.cat(labels), centers


def _within_class_spread(z, y):
    """tr of the within-class covariance of a batch -- the collapse meter."""
    total = 0.0
    for c in y.unique():
        zc = z[y == c]
        if zc.shape[0] > 1:
            total += float(zc.var(dim=0, unbiased=False).sum()) * zc.shape[0]
    return total / z.shape[0]


def _fit_reference(feats, labels, shrinkage=0.1):
    """Run the offline fitting path and load it back as a ClassGMMReference."""
    f64 = feats.to(DEVICE).double()
    n = f64.shape[0]
    feat_sum = f64.sum(0)
    feat_outer = f64.T @ f64
    mean, proj, explained = build_whitening(feat_sum, feat_outer, n, K)

    count, csum, couter = accumulate_class_stats(
        feats.half(), labels, mean, proj, C, world_size=1,
    )
    class_mu, class_cov, within_cov = finalize(count, csum, couter)

    ref = ClassGMMReference(
        mean, proj, class_mu,
        (1 - shrinkage) * class_cov + shrinkage * within_cov.unsqueeze(0),
        torch.log(count / count.sum()), within_cov,
    ).to(DEVICE)
    ref.eval().requires_grad_(False)
    return ref, explained


def _fit_online(ref, feats, labels, beta=0.9999, shrinkage=0.1):
    online = OnlineClassStats(C, K, ema_beta=beta, shrinkage=shrinkage).to(DEVICE)
    z = ref.project(feats.to(DEVICE))
    perm = torch.randperm(z.shape[0])
    for start in range(0, z.shape[0], 256):
        idx = perm[start:start + 256]
        online.update(z[idx], labels.to(DEVICE)[idx])
    online.refresh_cache(ref)
    return online


def test_cov_ema_beta_default_is_bit_exact_legacy_behavior():
    """Omitting the new covariance decay must reproduce the shared-beta path."""
    legacy = OnlineClassStats(4, 3, ema_beta=0.8, shrinkage=0.1).to(DEVICE)
    explicit = OnlineClassStats(
        4, 3, ema_beta=0.8, shrinkage=0.1, cov_ema_beta=0.8,
    ).to(DEVICE)

    z = torch.arange(36, dtype=torch.float32, device=DEVICE).reshape(12, 3) / 7.0
    y = torch.tensor([0, 1, 1, 2, 3, 0, 2, 2, 3, 1, 0, 3], device=DEVICE)
    for start, end in ((0, 5), (5, 9), (9, 12)):
        legacy.update(z[start:end], y[start:end])
        explicit.update(z[start:end], y[start:end])

    assert legacy.cov_ema_beta == legacy.ema_beta == 0.8
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(
            value, explicit.state_dict()[name], rtol=0.0, atol=0.0,
            msg=lambda msg: f"legacy mismatch in {name}: {msg}",
        )


def test_mean_and_covariance_ema_decays_are_independent():
    """Class weights use the mean beta while tied covariance uses its own beta."""
    online = OnlineClassStats(
        2, 1, ema_beta=0.5, shrinkage=0.1, cov_ema_beta=0.9,
    ).to(DEVICE)

    online.update(
        torch.tensor([[0.0], [2.0], [4.0]], device=DEVICE),
        torch.tensor([0, 0, 1], device=DEVICE),
    )
    torch.testing.assert_close(
        online.w, torch.tensor([1.0, 0.5], dtype=torch.float64, device=DEVICE),
    )
    torch.testing.assert_close(
        online.within_w,
        torch.tensor([0.3], dtype=torch.float64, device=DEVICE),
    )

    online.update(
        torch.tensor([[6.0], [8.0]], device=DEVICE),
        torch.tensor([0, 1], device=DEVICE),
    )
    torch.testing.assert_close(
        online.w, torch.tensor([1.0, 0.75], dtype=torch.float64, device=DEVICE),
    )
    expected_cov_w = 0.3 * (0.9 ** 2) + 2 * (1.0 - 0.9)
    torch.testing.assert_close(
        online.within_w,
        torch.tensor([expected_cov_w], dtype=torch.float64, device=DEVICE),
    )


def test_split_ema_strict_loads_legacy_state_dict():
    """EMA configuration stays out of state_dict, preserving old checkpoints."""
    # Positional arguments through eps retain their historical meaning.
    legacy = OnlineClassStats(3, 2, 0.8, 0.2, 1e-5).to(DEVICE)
    legacy.update(
        torch.tensor([[0.0, 1.0], [2.0, 3.0]], device=DEVICE),
        torch.tensor([0, 1], device=DEVICE),
    )
    state = legacy.state_dict()
    assert not any("beta" in key for key in state)

    split = OnlineClassStats(
        3, 2, ema_beta=0.5, shrinkage=0.2, eps=1e-5, cov_ema_beta=0.95,
    ).to(DEVICE)
    incompatible = split.load_state_dict(state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert split.ema_beta == 0.5
    assert split.cov_ema_beta == 0.95


def test_class_mean_spread_noise_floor_uses_finite_ema_ess():
    """The floor must reflect samples actually seen, not only the EMA asymptote."""
    online = OnlineClassStats(4, 1, ema_beta=0.9, shrinkage=0.0).to(DEVICE)
    online.w.fill_(1.0)
    online.mu_raw.copy_(
        torch.tensor([[-1.0], [0.0], [0.0], [1.0]],
                     dtype=torch.float64, device=DEVICE)
    )
    online.within_raw.fill_(2.0)
    online.within_w.fill_(1.0)
    online.total_seen.fill_(40)  # ten observations per class

    reference = SimpleNamespace(
        mu=torch.tensor([[-3.0], [-1.0], [1.0], [3.0]], device=DEVICE),
        within_trace=torch.tensor(2.0, device=DEVICE),
    )
    diag = online.diagnostics(reference)

    beta, n_avg = 0.9, 10.0
    expected_ess = ((1.0 + beta) / (1.0 - beta)
                    * (1.0 - beta ** n_avg) / (1.0 + beta ** n_avg))
    # diagnostics debiases the raw within trace by n/(n-1), and population
    # variance over four class means removes one quarter of mean noise.
    expected_floor = (3.0 / 4.0) * (2.0 * 10.0 / 9.0) / expected_ess / 5.0

    assert expected_ess < (1.0 + beta) / (1.0 - beta)
    assert abs(diag["gmm_class_mean_ema_n_eff"] - expected_ess) < 1e-10
    assert abs(diag["gmm_class_mean_spread_noise_floor"] - expected_floor) < 1e-7
    assert abs(
        diag["gmm_class_mean_spread_to_noise"]
        - diag["gmm_class_mean_spread"] / expected_floor
    ) < 1e-6


def test_whitening_makes_pooled_covariance_identity():
    feats, labels, _ = _synthetic_dataset()
    ref, explained = _fit_reference(feats, labels)
    z = ref.project(feats.to(DEVICE)).double()
    cov = torch.cov(z.T)
    err = (cov - torch.eye(K, device=DEVICE, dtype=cov.dtype)).abs().max()
    assert err < 1e-3, f"whitened pooled covariance is not identity (max err {err:.2e})"
    assert 0.0 < explained <= 1.0


def test_within_plus_between_is_one():
    """Whitening implies tr(within)/k + tr(between)/k == 1 -- a cheap invariant
    that catches a mis-scaled projection or a wrong per-class denominator."""
    feats, labels, _ = _synthetic_dataset()
    ref, _ = _fit_reference(feats, labels)
    within = float(ref.within_trace) / K
    between = float(ref.mu.double().var(dim=0, unbiased=False).sum()) / K
    assert abs(within + between - 1.0) < 0.02, f"{within=} {between=}"


def test_reference_posterior_classifies():
    feats, labels, _ = _synthetic_dataset()
    ref, _ = _fit_reference(feats, labels)
    z = ref.project(feats.to(DEVICE))
    pred = ref.logits(z).argmax(-1).cpu()
    acc = (pred == labels).float().mean()
    assert acc > 0.95, f"reference GMM only reaches {acc:.3f} accuracy on its own fit"


def test_posterior_kl_is_near_zero_when_q_matches_p():
    """The objective is E_x[KL(q(c|x)||p(c|x))]; feeding q the same distribution
    p was fitted on must drive it to ~0."""
    feats, labels, _ = _synthetic_dataset()
    ref, _ = _fit_reference(feats, labels)
    online = _fit_online(ref, feats, labels)

    idx = torch.randperm(feats.shape[0])[:512]
    _, parts, _ = gmm_posterior_loss(
        feats[idx].to(DEVICE), labels[idx].to(DEVICE), ref, online, lambda_ent=1.0,
    )
    kl = float(parts["gmm_cond_kl"])
    assert abs(kl) < 0.5, f"conditional KL should vanish when q == p, got {kl:.4f}"


def test_collapse_raises_the_loss():
    """A generator whose per-class clusters are 10x too tight must score worse
    than one that matches, even though a *pooled* Gaussian would barely notice."""
    feats, labels, _ = _synthetic_dataset(separation=SEP_OVERLAP)
    ref, _ = _fit_reference(feats, labels)

    matched, matched_y, _ = _synthetic_dataset(seed=1, separation=SEP_OVERLAP)
    collapsed, collapsed_y, _ = _synthetic_dataset(seed=1, spread=0.1, separation=SEP_OVERLAP)

    online_ok = _fit_online(ref, matched, matched_y)
    online_bad = _fit_online(ref, collapsed, collapsed_y)

    idx = torch.randperm(matched.shape[0])[:1024]
    _, parts_ok, _ = gmm_posterior_loss(
        matched[idx].to(DEVICE), matched_y[idx].to(DEVICE), ref, online_ok, lambda_ent=1.0,
    )
    _, parts_bad, _ = gmm_posterior_loss(
        collapsed[idx].to(DEVICE), collapsed_y[idx].to(DEVICE), ref, online_bad, lambda_ent=1.0,
    )
    kl_ok, kl_bad = float(parts_ok["gmm_cond_kl"]), float(parts_bad["gmm_cond_kl"])
    assert kl_bad > kl_ok + 0.5, (
        f"collapse not penalised: matched={kl_ok:.4f} collapsed={kl_bad:.4f}"
    )


def test_pooled_frechet_is_blind_to_what_the_posterior_term_catches():
    """The premise of the whole design: a collapsed generator can leave the
    pooled (mu, sigma) almost untouched. Here per-class spread is destroyed and
    replaced by extra between-class spread, so the pooled covariance is
    preserved while the class-conditional structure is not."""
    from frechet_distance.losses import compute_frechet_distance_loss

    real, real_y, _ = _synthetic_dataset(separation=SEP_OVERLAP)
    ref, _ = _fit_reference(real, real_y)
    fake_y = real_y.to(DEVICE)

    # Total collapse: every sample of class c is placed exactly on mu_c, so the
    # within-class variance is identically zero. Then recolour the result with
    # an affine map so its pooled mean and covariance match the real ones
    # *exactly*. FD cannot tell the two apart; there is nothing left for it to
    # measure.
    z_real = ref.project(real.to(DEVICE)).double()
    mu = torch.stack([z_real[fake_y == c].mean(0) for c in range(C)])
    z_collapsed = mu.index_select(0, fake_y)

    def _sqrt_and_inv_sqrt(cov):
        evals, evecs = torch.linalg.eigh(0.5 * (cov + cov.T))
        evals = evals.clamp(min=1e-10)
        return (evecs @ torch.diag(evals.sqrt()) @ evecs.T,
                evecs @ torch.diag(evals.rsqrt()) @ evecs.T)

    sig_real, mu_real = torch.cov(z_real.T), z_real.mean(0)
    sig_coll, mu_coll = torch.cov(z_collapsed.T), z_collapsed.mean(0)
    sqrt_real, _ = _sqrt_and_inv_sqrt(sig_real)
    _, inv_sqrt_coll = _sqrt_and_inv_sqrt(sig_coll)
    z_fake = (z_collapsed - mu_coll) @ inv_sqrt_coll @ sqrt_real + mu_real

    fd = float(compute_frechet_distance_loss(mu_real, sig_real, all_feats=z_fake))
    assert fd < 1e-3, f"the fake was supposed to match the pooled Gaussian, FD={fd:.4f}"

    # ...yet the posterior term sees it immediately.
    online = OnlineClassStats(C, K, ema_beta=0.9999, shrinkage=0.1).to(DEVICE)
    for start in range(0, z_fake.shape[0], 256):
        online.update(z_fake[start:start + 256].float(), fake_y[start:start + 256])
    online.refresh_cache(ref)
    ratio = online.diagnostics(ref)["gmm_within_trace_ratio"]
    assert ratio < 0.1, (
        f"pooled FD={fd:.3f} (blind) but the collapse meter should still fire; got {ratio:.4f}"
    )


def test_collapse_meter_tracks_within_class_variance():
    """tr(Sigma_within^q)/tr(Sigma_within^p) must read ~1 when matched and fall
    toward 0 under collapse -- this is the metric the run is judged on."""
    feats, labels, _ = _synthetic_dataset()
    ref, _ = _fit_reference(feats, labels)

    matched, matched_y, _ = _synthetic_dataset(seed=1)
    collapsed, collapsed_y, _ = _synthetic_dataset(seed=1, spread=0.1)

    ratio_ok = _fit_online(ref, matched, matched_y).diagnostics(ref)["gmm_within_trace_ratio"]
    ratio_bad = _fit_online(ref, collapsed, collapsed_y).diagnostics(ref)["gmm_within_trace_ratio"]

    assert 0.85 < ratio_ok < 1.15, f"matched ratio should be ~1, got {ratio_ok:.4f}"
    assert ratio_bad < 0.2, f"collapsed ratio should be near 0, got {ratio_bad:.4f}"


def _spread_after_step(lambda_ent, rel_step=0.1, seed=7, mode="density",
                       lambda_cls=1.0):
    """Within-class spread of a collapsed batch before/after one descent step.

    The step is normalised to a fixed fraction of the batch's own scale so the
    test measures the *direction* of the update, not the raw gradient magnitude
    (which depends on how saturated the posteriors happen to be).
    """
    feats, labels, _ = _synthetic_dataset(separation=SEP_OVERLAP)
    ref, _ = _fit_reference(feats, labels)
    collapsed, collapsed_y, _ = _synthetic_dataset(seed=1, spread=0.2, separation=SEP_OVERLAP)
    online = _fit_online(ref, collapsed, collapsed_y)

    idx = torch.randperm(collapsed.shape[0],
                         generator=torch.Generator().manual_seed(seed))[:1024]
    x = collapsed[idx].to(DEVICE).clone().requires_grad_(True)
    y = collapsed_y[idx].to(DEVICE)

    loss, _, _ = gmm_posterior_loss(x, y, ref, online, lambda_ent=lambda_ent,
                                    lambda_cls=lambda_cls, mode=mode)
    loss.backward()

    scale = x.detach().std() / x.grad.norm(dim=-1).mean().clamp(min=1e-12)
    stepped = x.detach() - rel_step * scale * x.grad
    before = _within_class_spread(ref.project(x.detach()), y)
    after = _within_class_spread(ref.project(stepped), y)
    return before, after


def test_q_term_flips_the_sign_of_the_diversity_update():
    """The central mechanism claim, tested as a sign flip on one fixed batch.

    On a collapsed batch, a descent step on ``-log p`` alone *contracts*
    within-class spread -- which is exactly why classifier guidance cannot fix
    collapse. Adding the density-ratio term reverses that: the step now
    *expands* it. The sign, not the magnitude, is the claim.
    """
    before_p, after_p = _spread_after_step(lambda_ent=0.0)
    before_d, after_d = _spread_after_step(lambda_ent=1.0, mode="density")

    assert after_p < before_p, (
        f"-log p alone should contract, but spread grew: {before_p:.5f} -> {after_p:.5f}"
    )
    assert after_d > before_d, (
        f"the density-ratio term should expand, but spread shrank: "
        f"{before_d:.5f} -> {after_d:.5f}"
    )


def test_posterior_mode_degenerates_under_collapse():
    """Documents *why* 'density' is the default.

    The posterior KL is bounded and stable, but on a collapsed batch q(.|x) is
    already near-deterministic, its entropy gradient flattens, and the term
    reduces to the contractive -log p signal. Isolate it (lambda_cls=0) and it
    still fails to expand -- so it cannot carry the anti-collapse role alone.
    """
    before, after = _spread_after_step(lambda_ent=1.0, mode="posterior", lambda_cls=0.0)
    before_d, after_d = _spread_after_step(lambda_ent=1.0, mode="density", lambda_cls=0.0)

    rel_post = (after - before) / before
    rel_dens = (after_d - before_d) / before_d
    assert rel_dens > rel_post, (
        f"density mode should spread more than posterior mode: "
        f"{rel_dens=:.5f} {rel_post=:.5f}"
    )


def test_lambda_zero_is_pure_classifier_guidance():
    """The lambda_ent=0 ablation must be attractive: a step reduces the distance
    to the target class mean, and the q term is not evaluated at all."""
    feats, labels, _ = _synthetic_dataset(separation=SEP_OVERLAP)
    ref, _ = _fit_reference(feats, labels)
    online = _fit_online(ref, feats, labels)

    x = _synthetic_dataset(seed=2, spread=1.5, separation=SEP_OVERLAP)[0][:512]
    x = x.to(DEVICE).clone().requires_grad_(True)
    y = torch.randint(0, C, (512,), device=DEVICE)
    loss, parts, _ = gmm_posterior_loss(x, y, ref, online, lambda_ent=0.0)
    loss.backward()

    assert "gmm_cond_kl" not in parts, "q term should be skipped at lambda_ent=0"
    scale = x.detach().std() / x.grad.norm(dim=-1).mean().clamp(min=1e-12)
    target = ref.mu.index_select(0, y)
    before = (ref.project(x.detach()) - target).norm(dim=-1).mean()
    after = (ref.project(x.detach() - 0.1 * scale * x.grad) - target).norm(dim=-1).mean()
    assert after < before, (
        f"classifier guidance is not mode-seeking: {float(before):.4f} -> {float(after):.4f}"
    )


def test_q_parameters_receive_no_gradient():
    """Gradients must flow only through the sample. If the q buffers ever
    started requiring grad, the pathwise-gradient argument would break."""
    feats, labels, _ = _synthetic_dataset(separation=SEP_OVERLAP)
    ref, _ = _fit_reference(feats, labels)
    online = _fit_online(ref, feats, labels)

    x = feats[:128].to(DEVICE).clone().requires_grad_(True)
    loss, _, _ = gmm_posterior_loss(x, labels[:128].to(DEVICE), ref, online)
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for module, name in ((online, "online"), (ref, "reference")):
        for buf_name, buf in module.named_buffers():
            assert not buf.requires_grad, f"{name}.{buf_name} requires grad"
            assert buf.grad is None, f"{name}.{buf_name} accumulated a gradient"


def test_class_subset_maps_global_labels():
    """A GMM fitted on a class subset is addressed with *global* labels.

    ``compute_class_stats.py --class_ids`` compacts the component axis, so the
    loss has to invert that mapping. Getting it wrong is silent: every sample
    would simply be scored against the wrong class and the run would look like
    slow-but-real learning.
    """
    feats, labels, _ = _synthetic_dataset()
    subset = [2, 5, 11]
    keep = torch.isin(labels, torch.tensor(subset))
    feats_s, labels_s = feats[keep], labels[keep]

    # Refit from scratch on the subset only: whitening, within-covariance and
    # the posterior denominator must all be defined over the 3 classes.
    f64 = feats_s.to(DEVICE).double()
    mean, proj, _ = build_whitening(f64.sum(0), f64.T @ f64, f64.shape[0], K)
    remap = torch.full((max(subset) + 1,), -1, dtype=torch.long)
    remap[torch.tensor(subset)] = torch.arange(len(subset))
    local = remap[labels_s]
    count, csum, couter = accumulate_class_stats(
        feats_s.half(), local, mean, proj, len(subset), world_size=1,
    )
    class_mu, class_cov, within_cov = finalize(count, csum, couter)
    ref = ClassGMMReference(
        mean, proj, class_mu, class_cov, torch.log(count / count.sum()), within_cov,
        class_ids=torch.tensor(subset),
    ).to(DEVICE)
    ref.eval().requires_grad_(False)

    assert ref.is_subset and ref.num_classes == len(subset)
    assert ref.to_local(torch.tensor(subset, device=DEVICE)).tolist() == [0, 1, 2]
    ref.validate_labels(subset)
    try:
        ref.validate_labels(subset + [7])
        raise AssertionError("validate_labels accepted a label with no component")
    except ValueError:
        pass

    online = OnlineClassStats(len(subset), K, ema_beta=0.9999, shrinkage=0.1).to(DEVICE)
    z = ref.project(feats_s.to(DEVICE))
    online.update(z, ref.to_local(labels_s.to(DEVICE)))
    online.refresh_cache(ref)

    # Global labels in, correct classification out; the posterior is 3-way.
    _, parts, _ = gmm_posterior_loss(
        feats_s.to(DEVICE), labels_s.to(DEVICE), ref, online,
    )
    assert float(parts["gmm_top1"]) > 0.95, f"subset GMM top-1 {float(parts['gmm_top1']):.3f}"
    assert float(parts["gmm_logp_c"]) > -0.2, "3-way posterior should be sharp here"


def test_cls_cap_stops_gradient_on_confident_samples():
    """The cap is what makes lambda_cls safe as a driver: a sample already above
    the cap must contribute exactly zero class-fidelity gradient, so the
    objective cannot become a race to adversarial certainty."""
    # A separation that leaves a genuine mix of confident and unconfident
    # samples: at SEP_WIDE every posterior saturates and *neither* variant has a
    # gradient, which would make the test vacuous.
    feats, labels, _ = _synthetic_dataset(separation=0.5)
    ref, _ = _fit_reference(feats, labels)
    online = _fit_online(ref, feats, labels)

    x = feats[:256].to(DEVICE).clone().requires_grad_(True)
    y = labels[:256].to(DEVICE)
    kw = dict(lambda_ent=0.0, normalize=False)

    capped, parts, _ = gmm_posterior_loss(x, y, ref, online, cls_cap=-0.69, **kw)
    frac = float(parts["gmm_cls_sat_frac"])
    assert 0.2 < frac < 0.9, f"test setup gives a degenerate cap fraction ({frac:.3f})"
    g_capped = torch.autograd.grad(capped, x)[0]

    uncapped, _, _ = gmm_posterior_loss(x, y, ref, online, cls_cap=0.0, **kw)
    g_uncapped = torch.autograd.grad(uncapped, x)[0]

    z = ref.project(x.detach())
    logp_c = torch.log_softmax(ref.logits(z), -1).gather(1, y.unsqueeze(1)).squeeze(1)
    above, below = logp_c > -0.69, logp_c <= -0.69

    assert float(g_capped[above].abs().max()) < 1e-9, \
        "samples above the cap still contribute class-fidelity gradient"
    assert float(g_uncapped[above].abs().max()) > 0.0, \
        "uncapped, those same samples must push -- otherwise the cap is a no-op"
    # Below the cap the two must agree: the clamp is a gate, not a rescale.
    ratio = g_capped[below].abs().sum() / g_uncapped[below].abs().sum().clamp(min=1e-12)
    assert abs(float(ratio) - 1.0) < 1e-4, \
        f"the clamp changed the gradient of uncapped samples (ratio {float(ratio):.6f})"


def test_class_normalization_modes_have_fixed_values_and_gradients():
    """Fixed log(C) scaling must not inherit self-normalization feedback."""
    feats, labels, _ = _synthetic_dataset(separation=SEP_OVERLAP)
    ref, _ = _fit_reference(feats, labels)
    online = _fit_online(ref, feats, labels)
    y = labels[:256].to(DEVICE)

    results = {}
    for mode in ("self", "log_classes", "none"):
        x = feats[:256].to(DEVICE).clone().requires_grad_(True)
        loss, parts, _ = gmm_posterior_loss(
            x, y, ref, online, lambda_ent=0.0,
            cls_normalization=mode,
        )
        results[mode] = (loss.detach(), torch.autograd.grad(loss, x)[0], parts)

    raw = results["none"][0]
    torch.testing.assert_close(
        results["self"][0], raw / (raw.abs() + 0.01), rtol=1e-6, atol=1e-7,
    )
    torch.testing.assert_close(
        results["log_classes"][0], raw / math.log(C), rtol=1e-6, atol=1e-7,
    )
    torch.testing.assert_close(
        results["log_classes"][1], results["none"][1] / math.log(C),
        rtol=1e-5, atol=1e-8,
    )
    # The denominator in self mode is detached, so it scales the gradient by
    # the current raw objective instead of the fixed class-count constant.
    torch.testing.assert_close(
        results["self"][1], results["none"][1] / (raw.abs() + 0.01),
        rtol=1e-5, atol=1e-8,
    )

    for loss, _, parts in results.values():
        torch.testing.assert_close(parts["gmm_cls_objective"], loss)
        torch.testing.assert_close(parts["gmm_q_objective"], torch.zeros_like(loss))
        torch.testing.assert_close(
            parts["gmm_loss"],
            parts["gmm_cls_objective"] + parts["gmm_q_objective"],
        )


def test_legacy_normalize_cls_api_and_component_meters():
    """Legacy booleans remain exact aliases and meters expose both objectives."""
    feats, labels, _ = _synthetic_dataset(separation=SEP_OVERLAP)
    ref, _ = _fit_reference(feats, labels)
    online = _fit_online(ref, feats, labels)
    x = feats[:256].to(DEVICE)
    y = labels[:256].to(DEVICE)

    for legacy, mode in ((True, "self"), (False, "none")):
        old, old_parts, _ = gmm_posterior_loss(
            x, y, ref, online, lambda_ent=0.0, normalize_cls=legacy,
        )
        new, new_parts, _ = gmm_posterior_loss(
            x, y, ref, online, lambda_ent=0.0, cls_normalization=mode,
        )
        torch.testing.assert_close(old, new, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            old_parts["gmm_cls_objective"], new_parts["gmm_cls_objective"],
            rtol=0.0, atol=0.0,
        )

    _, parts, _ = gmm_posterior_loss(
        x, y, ref, online, cls_normalization="log_classes",
    )
    assert torch.isfinite(parts["gmm_q_objective"])
    torch.testing.assert_close(
        parts["gmm_loss"],
        parts["gmm_cls_objective"] + parts["gmm_q_objective"],
    )

    try:
        gmm_posterior_loss(
            x, y, ref, online, normalize_cls=False,
            cls_normalization="log_classes",
        )
        raise AssertionError("conflicting old/new class normalizers were accepted")
    except ValueError as exc:
        assert "conflicting class normalizers" in str(exc)


if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
