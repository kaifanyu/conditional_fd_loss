"""Low-precision feature extraction must retain supported FD matrix arithmetic."""

import unittest

import torch

from frechet_distance.losses import compute_frechet_distance_loss, precompute_sigma_ref_sqrt
from frechet_distance.queue import FeatureQueue


class FDPrecisionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(79)
        self.features = torch.randn(16, 4).to(torch.bfloat16)
        self.mu_ref = torch.randn(4, dtype=torch.float64)
        base = torch.randn(4, 4, dtype=torch.float64)
        self.sigma_ref = base @ base.T + torch.eye(4, dtype=torch.float64)
        self.sqrt_ref = precompute_sigma_ref_sqrt(self.sigma_ref)

    def test_bf16_features_across_queue_modes_match_explicit_float_cast(self):
        for mode in ("empty", "snapshot", "online", "ema"):
            for symmetric in (False, True):
                with self.subTest(mode=mode, symmetric=symmetric):
                    queue = FeatureQueue(
                        size=0 if mode == "empty" else 24, feat_dim=4,
                        online_accum=mode == "online", ema_beta=0.9 if mode == "ema" else 0.0)
                    if mode == "ema":
                        queue.accumulate_batch(torch.randn(24, 4))
                        queue._finalize_streaming_init()
                    elif mode != "empty":
                        queue.feats.copy_(torch.randn(24, 4))
                        queue.ptr.fill_(20)  # Also exercise a wrapped snapshot.
                        if mode == "online":
                            queue._init_accumulators()
                    features = self.features.clone().requires_grad_(True)
                    explicit = self.features.float().requires_grad_(True)

                    def loss(x):
                        if mode in ("online", "ema"):
                            mu, sigma = queue.build_feats_stats(x)
                            self.assertEqual(mu.dtype, torch.float64)
                            self.assertEqual(sigma.dtype, torch.float64)
                            stats = {"mu": mu, "sigma": sigma}
                        else:
                            stats = {"all_feats": queue.build_feats_snapshot(x)}
                        return compute_frechet_distance_loss(
                            self.mu_ref, self.sigma_ref,
                            sigma_ref_sqrt=self.sqrt_ref if symmetric else None, **stats)

                    expected = loss(explicit)
                    # Matrix products must remain FP32/FP64 even when the caller
                    # surrounds the loss with neural-network autocast.
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        actual = loss(features)
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(actual))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    actual_grad = torch.autograd.grad(actual, features)[0]
                    expected_grad = torch.autograd.grad(expected, explicit)[0]
                    self.assertTrue(torch.isfinite(actual_grad).all())
                    self.assertGreater(float(actual_grad.float().norm()), 0)
                    torch.testing.assert_close(
                        actual_grad, expected_grad.to(torch.bfloat16), rtol=0, atol=0)

    def test_low_precision_precomputed_moments_preserve_vjp(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                mu = torch.randn(4).to(dtype).requires_grad_(True)
                sigma = (torch.eye(4) * 2.0).to(dtype).requires_grad_(True)
                explicit_mu = mu.detach().float().requires_grad_(True)
                explicit_sigma = sigma.detach().float().requires_grad_(True)
                expected = compute_frechet_distance_loss(
                    self.mu_ref, self.sigma_ref, mu=explicit_mu, sigma=explicit_sigma,
                    sigma_ref_sqrt=self.sqrt_ref)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    actual = compute_frechet_distance_loss(
                        self.mu_ref, self.sigma_ref, mu=mu, sigma=sigma,
                        sigma_ref_sqrt=self.sqrt_ref)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for grad, ref in zip(torch.autograd.grad(actual, (mu, sigma)),
                                     torch.autograd.grad(expected, (explicit_mu, explicit_sigma))):
                    self.assertTrue(torch.isfinite(grad).all())
                    torch.testing.assert_close(grad, ref.to(dtype), rtol=0, atol=0)

    def test_reference_square_root_preserves_supported_precision(self):
        for dtype in (torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                sigma = self.sigma_ref.to(dtype)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    root = precompute_sigma_ref_sqrt(sigma)
                self.assertEqual(root.dtype, torch.float64 if dtype == torch.float64 else torch.float32)
                torch.testing.assert_close(root @ root.T, sigma.to(root.dtype), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
