"""CPU checks for the offline image-gradient diagnostic (no model downloads)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.vlm_input_saliency import InputGradientLogger, gradient_metrics, input_gradients
from vlm_linear_heads import VLMDeltaHeads, VLMLinearHead


class TinyExtractor(nn.Module):
    layer = 0

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4)
        self.eval().requires_grad_(False)

    def answer_states(self, x):
        # Stand-in for differentiable image preprocessing and a frozen encoder.
        return {0: self.projection((2 * x - 1).mean((2, 3))).tanh()}


class InputSaliencyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.encoder = TinyExtractor()
        self.heads = VLMDeltaHeads(VLMLinearHead(4, 3)).eval().requires_grad_(False)
        self.x = torch.rand(2, 3, 4, 4)
        self.targets = torch.tensor([0, 2])

    def test_cloned_q_has_matching_gradients_and_zero_delta_without_updates(self):
        before = {k: v.clone() for k, v in self.heads.state_dict().items()}
        result = input_gradients(self.x, self.targets, self.heads, self.encoder)
        torch.testing.assert_close(result["grad_log_p"], result["grad_log_q"], rtol=0, atol=0)
        self.assertGreater(float(result["grad_log_p"].norm()), 0)
        self.assertEqual(float(result["grad_log_q_minus_log_p"].norm()), 0)
        torch.testing.assert_close(result["x"], self.x, rtol=0, atol=0)
        for key, value in self.heads.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for m in (self.heads, self.encoder)
                            for p in m.parameters()))

    def test_different_q_matches_finite_difference_and_delta_sign(self):
        with torch.no_grad():
            self.heads.q_teacher.weight.mul_(-2)
        result = input_gradients(self.x, self.targets, self.heads, self.encoder)
        direction = torch.randn_like(self.x)
        direction /= direction.norm()
        eps = 0.005

        def scores(x):
            z = self.encoder.answer_states(x)[0]
            idx = self.targets[:, None]
            p = self.heads.p_log_probs(z).gather(1, idx).sum()
            q = self.heads.q_teacher_log_probs(z).gather(1, idx).sum()
            return torch.stack((p, q, q - p))

        with torch.no_grad():
            finite_difference = (scores(self.x + eps * direction)
                                 - scores(self.x - eps * direction)) / (2 * eps)
        keys = ("grad_log_p", "grad_log_q", "grad_log_q_minus_log_p")
        derivative = torch.stack([(result[k] * direction).sum() for k in keys])
        torch.testing.assert_close(derivative, finite_difference, rtol=0.01, atol=5e-5)
        torch.testing.assert_close(result[keys[2]], result[keys[1]] - result[keys[0]])

    def test_gradients_independent_of_microbatch_size(self):
        batched = input_gradients(self.x, self.targets, self.heads, self.encoder)
        single = input_gradients(self.x[:1], self.targets[:1], self.heads, self.encoder)
        torch.testing.assert_close(batched["grad_log_p"][:1], single["grad_log_p"])

    def test_tiny_gradient_cosine_and_zero_direction(self):
        result = input_gradients(self.x[:1], self.targets[:1], self.heads, self.encoder)
        sample = {k: v[0] for k, v in result.items()}
        sample["grad_log_p"] *= 1e-20
        sample["grad_log_q"] *= 1e-20
        self.assertAlmostEqual(gradient_metrics(sample)["p_q_cosine"], 1.0)
        sample["grad_log_q"].zero_()
        self.assertIsNone(gradient_metrics(sample)["p_q_cosine"])

    def test_raw_arrays_and_figures_are_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            logger = InputGradientLogger(folder, 100, ["test"], self.targets[:1], "q=p")
            logger.record(self.x[:1], self.targets[:1], self.heads, self.encoder,
                          0, "p", 0, "fp32")
            logger.finish()
            root = Path(folder) / "saliency"
            with np.load(root / "p/test_step0000.npz") as data:
                np.testing.assert_array_equal(data["x"], self.x[0].numpy())
                self.assertEqual(data["grad_log_p"].shape, (3, 4, 4))
            self.assertTrue((root / "p/test_fixed_scale.png").is_file())
            self.assertTrue((root / "p/test_step_scale.png").is_file())
            self.assertTrue((root / "metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main()
