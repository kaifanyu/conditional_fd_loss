"""Correctness checks for ``mode="cfg_delta"`` -- the CFG-delta vector field.

Every check here is against an independently-computed answer: autograd through
``log_softmax(logits)`` for the analytic score deltas, and the algebraic
density/CFG/marginal identity for the sign convention. A failure is a bug in
the implementation, not a property of the generator.

The companion synthetic-collapse tests for the *existing* modes stay in
``test_gmm_posterior.py``; nothing here weakens them. cfg_delta is an ablation.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frechet_distance.gmm import (
    ClassGMMReference, OnlineClassStats, gmm_posterior_loss,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C, D, K, B = 6, 12, 5, 32


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _random_reference(seed=0, tied=False, generator_means=None):
    """A small QDA reference with a well-conditioned per-class covariance.

    ``tied=True`` forces one shared covariance and a uniform prior, which is the
    only regime in which p and q are the *same model class* and the CFG delta
    can be expected to vanish exactly.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    feat_mean = torch.randn(D, generator=g)
    proj = torch.linalg.qr(torch.randn(D, D, generator=g))[0][:, :K]
    mu = (torch.randn(C, K, generator=g) if generator_means is None
          else generator_means.clone())

    within = torch.randn(K, K, generator=g)
    within = within @ within.T / K + 0.6 * torch.eye(K)
    if tied:
        cov = within.unsqueeze(0).expand(C, K, K).contiguous()
        log_prior = torch.full((C,), -torch.tensor(float(C)).log().item())
    else:
        a = torch.randn(C, K, K, generator=g)
        cov = a @ a.transpose(-1, -2) / K + 0.6 * torch.eye(K)
        counts = torch.rand(C, generator=g) + 0.5
        log_prior = torch.log(counts / counts.sum())

    ref = ClassGMMReference(feat_mean, proj, mu, cov, log_prior, within,
                            device=DEVICE)
    return ref.to(DEVICE).eval().requires_grad_(False)


def _random_online(ref, seed=1, mean_scale=1.0, means=None):
    """An OnlineClassStats whose cache holds a genuine tied-covariance fit."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    online = OnlineClassStats(C, K, ema_beta=0.99, shrinkage=0.25).to(DEVICE)
    centres = (torch.randn(C, K, generator=g) * mean_scale if means is None
               else means.clone())
    z = []
    y = []
    for c in range(C):
        z.append(centres[c] + torch.randn(64, K, generator=g) * 0.8)
        y.append(torch.full((64,), c, dtype=torch.long))
    z = torch.cat(z).to(DEVICE)
    y = torch.cat(y).to(DEVICE)
    perm = torch.randperm(z.shape[0], generator=g).to(DEVICE)
    for start in range(0, z.shape[0], 48):
        idx = perm[start:start + 48]
        online.update(z[idx], y[idx])
    online.refresh_cache(ref)
    return online


def _batch(seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = (torch.randn(B, K, generator=g)).to(DEVICE)
    labels = torch.randint(0, C, (B,), generator=g).to(DEVICE)
    return z, labels


def _autograd_oracle(logits_fn, z, labels, temperature):
    """grad_z of the *selected* label's log posterior. The reference answer."""
    z_ref = z.detach().clone().requires_grad_(True)
    log_post = torch.log_softmax(logits_fn(z_ref, temperature=temperature), dim=-1)
    selected = log_post.gather(1, labels[:, None]).sum()
    return torch.autograd.grad(selected, z_ref)[0].detach()


# ---------------------------------------------------------------------------
# Test 1 / 2: the analytic deltas are the autograd answer
# ---------------------------------------------------------------------------

def _check_against_autograd(logits_fn, analytic_fn, name):
    worst = 0.0
    for temperature in (1.0, 7.5):
        z, labels = _batch()
        analytic = analytic_fn(z, labels, temperature=temperature)
        expected = _autograd_oracle(logits_fn, z, labels, temperature)
        worst = max(worst, float((analytic - expected).abs().max()))
        torch.testing.assert_close(
            analytic, expected, rtol=1e-5, atol=1e-6,
            msg=lambda m: f"{name} at T={temperature}: {m}",
        )
    print(f"      {name}: max |analytic - autograd| = {worst:.3e}")


def test_real_posterior_score_delta_matches_autograd():
    """QDA side: the per-class precision must survive the mixture average."""
    ref = _random_reference()
    _check_against_autograd(ref.logits, ref.posterior_score_delta, "reference (QDA)")


def test_online_posterior_score_delta_matches_autograd():
    """Tied-covariance side: dropping the cancelled -Pz term must be exact."""
    ref = _random_reference()
    online = _random_online(ref)
    _check_against_autograd(online.logits, online.posterior_score_delta,
                            "online (LDA)")


# ---------------------------------------------------------------------------
# Test 3: the linear surrogate injects exactly the requested gradient
# ---------------------------------------------------------------------------

def test_linear_surrogate_injects_requested_gradient():
    """mean_i z_i.g_i has gradient g/B -- the Monte Carlo mean field."""
    g = torch.randn(B, K, device=DEVICE)
    z = torch.randn(B, K, device=DEVICE, requires_grad=True)
    loss = (z * g).sum(-1).mean()
    grad = torch.autograd.grad(loss, z)[0]
    torch.testing.assert_close(grad, g / B, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# Test 4: identical fitted models give a zero field
# ---------------------------------------------------------------------------

def test_cfg_delta_vanishes_when_p_and_q_agree():
    """Deliberately tied p, so p and q are in the same model class."""
    g = torch.Generator(device="cpu").manual_seed(3)
    means = torch.randn(C, K, generator=g)
    ref = _random_reference(tied=True, generator_means=means)

    # Hand q exactly p's parameters instead of fitting it: a fitted q would
    # differ by estimator noise and this test is about the algebra.
    online = OnlineClassStats(C, K, ema_beta=0.99, shrinkage=0.25).to(DEVICE)
    prec = ref.prec_flat[0].view(K, K)
    mu = ref.mu
    online.mu_cache.copy_(mu)
    online.prec_cache.copy_(prec)
    online.prec_mu_cache.copy_(mu @ prec)
    online.mu_prec_mu_cache.copy_(((mu @ prec) * mu).sum(-1))

    z, labels = _batch()
    for temperature in (1.0, 4.0):
        delta_p = ref.posterior_score_delta(z, labels, temperature=temperature)
        delta_q = online.posterior_score_delta(z, labels, temperature=temperature)
        torch.testing.assert_close(
            delta_q, delta_p, rtol=1e-4, atol=1e-4,
            msg=lambda m: f"tied p/q disagree at T={temperature}: {m}",
        )


# ---------------------------------------------------------------------------
# Test 5: density / CFG / marginal identity -- this pins the sign
# ---------------------------------------------------------------------------

def test_density_minus_cfg_equals_marginal_score_mismatch():
    """g_density - g_cfg = s_q(z) - s_p(z), at T=1 where Bayes is exact.

    g_density is the *gradient of the density-mode scalar*, i.e.
    grad_z[log q(z|c) - log p(z|c)] = s_q(z|c) - s_p(z|c).
    """
    ref = _random_reference()
    online = _random_online(ref)
    z, labels = _batch()

    zr = z.detach().clone().requires_grad_(True)
    ratio = (online.log_likelihood(zr, labels) - ref.log_likelihood(zr, labels)).sum()
    g_density = torch.autograd.grad(ratio, zr)[0].detach()

    g_cfg = (online.posterior_score_delta(z, labels, temperature=1.0)
             - ref.posterior_score_delta(z, labels, temperature=1.0))

    # Marginal mixture scores, from autograd on the mixture log-densities. The
    # class prior is inside `logits` for both sides, so logsumexp of the logits
    # is log of the (prior-weighted) mixture up to a z-independent constant.
    def _marginal_score(logits_fn):
        zm = z.detach().clone().requires_grad_(True)
        return torch.autograd.grad(
            torch.logsumexp(logits_fn(zm, temperature=1.0), dim=-1).sum(), zm,
        )[0].detach()

    marginal_mismatch = _marginal_score(online.logits) - _marginal_score(ref.logits)
    torch.testing.assert_close(g_density - g_cfg, marginal_mismatch,
                               rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Test 6: no gradient reaches any GMM state
# ---------------------------------------------------------------------------

def test_cfg_delta_backward_leaves_gmm_state_clean():
    ref = _random_reference()
    online = _random_online(ref)
    feats = torch.randn(B, D, device=DEVICE, requires_grad=True)
    labels = torch.randint(0, C, (B,), device=DEVICE)

    loss, parts, _ = gmm_posterior_loss(
        feats, labels, ref, online, mode="cfg_delta", lambda_cls=0.0,
        lambda_ent=1.0, temperature=5.0,
    )
    loss.backward()

    assert feats.grad is not None and torch.isfinite(feats.grad).all()
    assert float(feats.grad.abs().max()) > 0.0, "no gradient reached the features"
    for module, tag in ((ref, "reference"), (online, "online")):
        for name, buf in list(module.named_buffers()) + list(module.named_parameters()):
            assert buf.grad is None, f"{tag} buffer {name} received a gradient"
            assert not buf.requires_grad, f"{tag} buffer {name} requires grad"
    for key in ("gmm_cfg_teacher_rms", "gmm_cfg_fake_rms", "gmm_cfg_error_rms",
                "gmm_cfg_relative_error", "gmm_cfg_alignment_cos",
                "gmm_cfg_vector_scale", "gmm_cfg_surrogate"):
        assert key in parts, f"missing diagnostic {key}"
        assert torch.isfinite(parts[key]).all(), f"{key} is not finite"


def test_cfg_delta_loss_gradient_is_the_field_over_batch():
    """End-to-end: d loss / d z must equal lambda_ent * g_cfg / B."""
    ref = _random_reference()
    online = _random_online(ref)
    feats = torch.randn(B, D, device=DEVICE)
    labels = torch.randint(0, C, (B,), device=DEVICE)
    lambda_ent, temperature = 0.7, 3.0

    z = ref.project(feats).detach().requires_grad_(True)
    expected = lambda_ent * (
        online.posterior_score_delta(z, labels, temperature=temperature)
        - ref.posterior_score_delta(z, labels, temperature=temperature)
    ) / B

    # Same computation the loss performs, driven through `project` so the whole
    # path from features is exercised.
    feats_g = feats.clone().requires_grad_(True)
    loss, _, _ = gmm_posterior_loss(
        feats_g, labels, ref, online, mode="cfg_delta", lambda_cls=0.0,
        lambda_ent=lambda_ent, temperature=temperature,
    )
    grad_feats = torch.autograd.grad(loss, feats_g)[0]
    # project() is affine with frozen basis P, so dL/dfeat = dL/dz @ P^T.
    torch.testing.assert_close(grad_feats, expected @ ref.proj.T,
                               rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 7: rms normalization, and that it is a *different* field
# ---------------------------------------------------------------------------

def test_rms_normalization_rescales_the_field_to_unit_rms():
    ref = _random_reference()
    online = _random_online(ref)
    feats = torch.randn(B, D, device=DEVICE, requires_grad=True)
    labels = torch.randint(0, C, (B,), device=DEVICE)

    raw = gmm_posterior_loss(feats, labels, ref, online, mode="cfg_delta",
                             lambda_cls=0.0, temperature=5.0)[1]
    rms = gmm_posterior_loss(feats, labels, ref, online, mode="cfg_delta",
                             lambda_cls=0.0, temperature=5.0,
                             cfg_delta_normalization="rms")[1]

    assert float(raw["gmm_cfg_vector_scale"]) == 1.0
    torch.testing.assert_close(rms["gmm_cfg_vector_scale"],
                               raw["gmm_cfg_error_rms"], rtol=1e-5, atol=1e-8)
    # The reported error RMS is of the *raw* field either way, so the two runs
    # agree on the diagnostic and differ only in the injected scale.
    torch.testing.assert_close(rms["gmm_cfg_error_rms"], raw["gmm_cfg_error_rms"],
                               rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# Test 8: surrogate is not self-normalised
# ---------------------------------------------------------------------------

def test_surrogate_is_not_self_normalized():
    """_self_normalize would pin |q_objective| near 1 regardless of the field."""
    ref = _random_reference()
    online = _random_online(ref)
    feats = torch.randn(B, D, device=DEVICE, requires_grad=True)
    labels = torch.randint(0, C, (B,), device=DEVICE)

    small = gmm_posterior_loss(feats, labels, ref, online, mode="cfg_delta",
                               lambda_cls=0.0, lambda_ent=1.0, temperature=5.0)[1]
    big = gmm_posterior_loss(feats, labels, ref, online, mode="cfg_delta",
                             lambda_cls=0.0, lambda_ent=4.0, temperature=5.0)[1]
    torch.testing.assert_close(big["gmm_q_objective"], 4.0 * small["gmm_q_objective"],
                               rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(small["gmm_q_objective"], small["gmm_cfg_surrogate"],
                               rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# Test 9: invalid inputs fail loudly
# ---------------------------------------------------------------------------

def test_invalid_temperature_and_normalization_are_rejected():
    ref = _random_reference()
    online = _random_online(ref)
    z, labels = _batch()

    for module in (ref, online):
        for bad in (0.0, -1.0):
            try:
                module.posterior_score_delta(z, labels, temperature=bad)
                raise AssertionError(f"{type(module).__name__} accepted T={bad}")
            except ValueError as exc:
                assert "temperature must be positive" in str(exc)

    feats = torch.randn(B, D, device=DEVICE, requires_grad=True)
    try:
        gmm_posterior_loss(feats, labels, ref, online, mode="cfg_delta",
                           lambda_cls=0.0, cfg_delta_normalization="l2")
        raise AssertionError("unknown cfg_delta_normalization was accepted")
    except ValueError as exc:
        assert "cfg_delta_normalization must be" in str(exc)

    try:
        gmm_posterior_loss(feats, labels, ref, online, mode="cfg_deltaa",
                           lambda_cls=0.0)
        raise AssertionError("unknown mode was accepted")
    except ValueError as exc:
        assert "unknown fd_gmm_mode" in str(exc)


# ---------------------------------------------------------------------------
# Test 10: the other two modes are untouched
# ---------------------------------------------------------------------------

def test_density_and_posterior_modes_are_bit_identical_to_before():
    """cfg_delta must be inert for every existing call: same loss, same parts."""
    ref = _random_reference()
    online = _random_online(ref)
    labels = torch.randint(0, C, (B,), device=DEVICE)

    for mode, expected_key in (("density", "gmm_cond_kl"), ("posterior", "gmm_post_kl")):
        torch.manual_seed(11)
        feats_a = torch.randn(B, D, device=DEVICE).requires_grad_(True)
        loss_a, parts_a, _ = gmm_posterior_loss(feats_a, labels, ref, online,
                                                mode=mode, temperature=5.0)
        # The new kwarg must not change anything when the mode is not cfg_delta.
        feats_b = feats_a.detach().clone().requires_grad_(True)
        loss_b, parts_b, _ = gmm_posterior_loss(feats_b, labels, ref, online,
                                                mode=mode, temperature=5.0,
                                                cfg_delta_normalization="rms")
        torch.testing.assert_close(loss_a, loss_b, rtol=0.0, atol=0.0)
        assert expected_key in parts_a
        assert not any(k.startswith("gmm_cfg_") for k in parts_a), \
            f"{mode} mode leaked cfg diagnostics"
        for key in parts_a:
            torch.testing.assert_close(parts_a[key], parts_b[key], rtol=0.0, atol=0.0)


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
