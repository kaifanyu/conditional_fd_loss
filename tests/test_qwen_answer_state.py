"""Tests for the prompted-VLM answer-state feature and its first-order VJP.

The whole point of ``qwen_answer_state.QwenAnswerStateExtractor.vjp_surrogate``
is that the generator receives *exactly* the gradient it would have received
from differentiating through the 7B model directly, while none of that model's
activations survive into the generator's backward pass.  That claim is only
worth anything if it is checked against the direct computation, which is what
:class:`VJPSurrogateTest` does.

The heavy tests need a GPU and the local Qwen2.5-VL checkpoint and skip
themselves cleanly without either.  The cheap ones (layer indexing, prompt
hashing, checkpoint identity) always run.

Run:  /home/nvidia/miniconda3/envs/fdloss/bin/python -m unittest \
          tests.test_qwen_answer_state -v
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_answer_state import (  # noqa: E402
    DEFAULT_ANSWER_PROMPT,
    DEFAULT_QWEN_MODEL,
    QWEN_BACKEND,
    prompt_checksum,
)
from vlm_linear_heads import (  # noqa: E402
    P_HEAD_BACKEND_QWEN,
    P_HEAD_BACKEND_TIMM,
    P_HEAD_FORMAT_VERSION,
    VLMLinearHead,
    class_ids_hash,
    load_p_head_checkpoint,
    p_head_backend,
    p_head_checksum,
    p_head_identity,
)


def _has_model() -> bool:
    return torch.cuda.is_available() and os.path.isdir(DEFAULT_QWEN_MODEL)


_SKIP = unittest.skipUnless(
    _has_model(), "needs CUDA and the local Qwen2.5-VL-7B checkpoint")

_EXTRACTOR = None
_EXTRACTOR_DTYPE = None


def _extractor(dtype="bf16"):
    """One 7B load at a time; reloading on a dtype change costs ~30 s.

    The VJP tests deliberately run in fp32.  bf16 weight rounding changes the
    *function* being differentiated, so a bf16 gradient cannot be checked
    against an fp32 reference without conflating an implementation bug with a
    precision effect -- see :class:`Bf16PrecisionTest`, which measures that
    effect instead of asserting it away.
    """
    global _EXTRACTOR, _EXTRACTOR_DTYPE
    if _EXTRACTOR is not None and _EXTRACTOR_DTYPE != dtype:
        del _EXTRACTOR
        _EXTRACTOR, _EXTRACTOR_DTYPE = None, None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    if _EXTRACTOR is None:
        from qwen_answer_state import QwenAnswerStateExtractor
        _EXTRACTOR = QwenAnswerStateExtractor(
            DEFAULT_QWEN_MODEL, image_size=256, microbatch_size=2, dtype=dtype)
        _EXTRACTOR_DTYPE = dtype
    return _EXTRACTOR


# ---------------------------------------------------------------------------
# Cheap tests
# ---------------------------------------------------------------------------

class PromptChecksumTest(unittest.TestCase):
    def test_stable_and_sensitive(self):
        a = prompt_checksum("What is the ImageNet class of this image? Answer:")
        self.assertEqual(a, prompt_checksum(
            "What is the ImageNet class of this image? Answer:"))
        self.assertNotEqual(a, prompt_checksum(
            "What is the ImageNet class of this image? Answer: "))
        self.assertEqual(len(a), 64)


class CheckpointBackendTest(unittest.TestCase):
    """The identity of a legacy timm head must not shift under the new field."""

    def _payload(self, backend=None, **extra):
        # One fixed head for the whole class: two payloads that differ only in
        # the backend tag must otherwise hash identically, or the comparison
        # below would be measuring random initialisation.
        torch.manual_seed(7)
        head = VLMLinearHead(8, 3)
        payload = {
            "format_version": P_HEAD_FORMAT_VERSION,
            "vlm_model_name": "m",
            "vlm_pool_type": "cls",
            "vlm_target_size": 224,
            "vlm_input_size": 256,
            "feature_dim": 8,
            "num_classes": 3,
            "class_ids": [0, 1, 2],
            "class_ids_sha256": class_ids_hash([0, 1, 2]),
            "feature_norm": "standardize",
            "feature_mean": head.feature_mean,
            "feature_std": head.feature_std,
            "head_state_dict": dict(head.linear.state_dict()),
            "temperature": 1.0,
        }
        if backend is not None:
            payload["vlm_backend"] = backend
        payload.update(extra)
        return payload

    def test_absent_backend_defaults_to_timm(self):
        payload = self._payload()
        self.assertEqual(p_head_backend(payload), P_HEAD_BACKEND_TIMM)

    def test_legacy_identity_gains_no_keys(self):
        legacy = p_head_identity(self._payload())
        self.assertNotIn("vlm_backend", legacy)
        self.assertNotIn("vlm_layer", legacy)
        self.assertEqual(legacy, p_head_identity(
            self._payload(backend=P_HEAD_BACKEND_TIMM)))

    def test_answer_state_identity_carries_layer_and_prompt(self):
        identity = p_head_identity(self._payload(
            backend=P_HEAD_BACKEND_QWEN, vlm_layer=24,
            vlm_prompt_sha256="a" * 64))
        self.assertEqual(identity["vlm_backend"], P_HEAD_BACKEND_QWEN)
        self.assertEqual(identity["vlm_layer"], 24)
        self.assertEqual(identity["vlm_prompt_sha256"], "a" * 64)

    def test_answer_state_checkpoint_without_layer_is_refused(self):
        import tempfile
        payload = self._payload(backend=P_HEAD_BACKEND_QWEN)
        payload["p_head_sha256"] = p_head_checksum(payload)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p_head.pt")
            torch.save(payload, path)
            with self.assertRaises(ValueError) as caught:
                load_p_head_checkpoint(path)
            self.assertIn("vlm_layer", str(caught.exception))

    def test_unknown_backend_is_refused(self):
        import tempfile
        payload = self._payload(backend="something_else")
        payload["p_head_sha256"] = p_head_checksum(payload)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p_head.pt")
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                load_p_head_checkpoint(path)


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------

@_SKIP
class ExtractorBasicsTest(unittest.TestCase):
    def setUp(self):
        self.ex = _extractor()

    def test_geometry_and_identity(self):
        self.assertEqual(self.ex.feat_dim, 3584)
        self.assertEqual(self.ex.num_hidden_states, 29)
        self.assertEqual(self.ex.layer, 28)  # -1 resolves to the post-norm state
        identity = self.ex.identity()
        self.assertEqual(identity["vlm_backend"], QWEN_BACKEND)
        self.assertEqual(identity["vlm_prompt_sha256"],
                         prompt_checksum(self.ex.rendered_prompt))
        self.assertGreater(self.ex.num_visual_tokens, 0)

    def test_prompt_does_not_leak_a_label(self):
        self.assertNotIn("{", DEFAULT_ANSWER_PROMPT)
        self.assertIn(DEFAULT_ANSWER_PROMPT, self.ex.rendered_prompt)

    def test_layer_indexing_matches_hidden_states(self):
        self.assertEqual(self.ex.normalize_layer(-1), 28)
        self.assertEqual(self.ex.normalize_layer(-29), 0)
        self.assertEqual(self.ex.normalize_layer(14), 14)
        with self.assertRaises(ValueError):
            self.ex.normalize_layer(29)
        with self.assertRaises(ValueError):
            self.ex.normalize_layer(-30)

    def test_forward_returns_the_same_feature_for_either_pool(self):
        images = torch.rand(2, 3, 256, 256, device="cuda")
        with torch.no_grad():
            primary, secondary = self.ex(images)
        self.assertEqual(primary.shape, (2, 3584))
        self.assertIs(primary, secondary)

    def test_wrong_resolution_is_refused(self):
        with self.assertRaises(ValueError):
            with torch.no_grad():
                self.ex(torch.rand(1, 3, 224, 224, device="cuda"))

    def test_multiple_layers_come_from_one_forward(self):
        images = torch.rand(2, 3, 256, 256, device="cuda")
        with torch.no_grad():
            many = self.ex.answer_states(images, layers=[14, 24, 28])
            one = self.ex.answer_states(images, layers=[24])
        self.assertEqual(sorted(many), [14, 24, 28])
        torch.testing.assert_close(many[24], one[24], rtol=0, atol=0)

    def test_vlm_takes_no_parameter_gradient(self):
        images = torch.rand(2, 3, 256, 256, device="cuda", requires_grad=True)
        z = self.ex.answer_states(images)[self.ex.layer]
        z.float().sum().backward()
        leaked = [n for n, p in self.ex.vlm.named_parameters() if p.grad is not None]
        self.assertEqual(leaked, [])
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertGreater(float(images.grad.norm()), 0.0)


@_SKIP
class VJPSurrogateTest(unittest.TestCase):
    """The surrogate must have the term's value and the term's image gradient."""

    def setUp(self):
        self.ex = _extractor("fp32")
        torch.manual_seed(0)
        self.num_classes = 5
        self.head = VLMLinearHead(self.ex.feat_dim, self.num_classes).cuda()
        self.head.set_feature_norm_stats(
            torch.zeros(self.ex.feat_dim).cuda(),
            torch.ones(self.ex.feat_dim).cuda() * 10.0)
        self.head.eval().requires_grad_(False)
        self.labels = torch.tensor([0, 3, 1, 4], device="cuda")
        self.base = torch.rand(4, 3, 256, 256, device="cuda")

    def _term_fn(self, scale):
        def term_fn(z, mb_labels):
            logp = self.head.log_probs(z)
            return logp.gather(1, mb_labels.view(-1, 1)).squeeze(1).sum() * scale
        return term_fn

    def _direct(self, images, scale):
        """The same quantity with the whole 7B graph alive, in one forward."""
        z = self.ex.answer_states(images)[self.ex.layer]
        term = self._term_fn(scale)(z, self.labels)
        grad = torch.autograd.grad(term, images)[0]
        return term.detach(), grad

    def test_value_and_gradient_match_the_direct_computation(self):
        scale = 1.0 / self.labels.numel()
        direct_images = self.base.clone().requires_grad_(True)
        direct_value, direct_grad = self._direct(direct_images, scale)

        sur_images = self.base.clone().requires_grad_(True)
        surrogate, z_detached, selected = self.ex.vjp_surrogate(
            sur_images, self.labels, self._term_fn(scale), microbatch_size=2)
        sur_grad = torch.autograd.grad(surrogate, sur_images)[0]

        self.assertEqual(tuple(z_detached.shape), (4, self.ex.feat_dim))
        self.assertEqual(selected.tolist(), [0, 1, 2, 3])
        self.assertGreater(float(direct_grad.norm()), 0.0)
        # In fp32 the surrogate is the term, to the last few bits: same value,
        # same image gradient, computed two images at a time instead of four.
        self.assertAlmostEqual(float(surrogate), float(direct_value), places=5)
        cos = torch.nn.functional.cosine_similarity(
            sur_grad.reshape(1, -1).float(), direct_grad.reshape(1, -1).float())
        self.assertGreater(float(cos), 0.9999)
        rel = float((sur_grad - direct_grad).norm() / direct_grad.norm())
        self.assertLess(rel, 1e-3)

    def test_microbatching_does_not_change_the_answer(self):
        scale = 1.0 / self.labels.numel()
        grads, values = [], []
        for mb in (1, 2, 4):
            images = self.base.clone().requires_grad_(True)
            surrogate, _, _ = self.ex.vjp_surrogate(
                images, self.labels, self._term_fn(scale), microbatch_size=mb)
            grads.append(torch.autograd.grad(surrogate, images)[0])
            values.append(float(surrogate))
        for value, other in zip(values[1:], grads[1:]):
            self.assertAlmostEqual(value, values[0], places=5)
            cos = torch.nn.functional.cosine_similarity(
                grads[0].reshape(1, -1).float(), other.reshape(1, -1).float())
            self.assertGreater(float(cos), 0.9999)

    def test_no_vlm_graph_survives(self):
        """After the call, nothing in the surrogate references a 7B activation."""
        images = self.base.clone().requires_grad_(True)
        surrogate, _, _ = self.ex.vjp_surrogate(
            images, self.labels, self._term_fn(1.0), microbatch_size=2)
        before = torch.cuda.memory_allocated()
        surrogate.backward()
        after = torch.cuda.memory_allocated()
        # A retained 7B graph would free hundreds of MB here; the surrogate's
        # own graph is a handful of tensors the size of the image batch.
        self.assertLess(abs(after - before), 64 * 1024 * 1024)
        self.assertTrue(torch.isfinite(images.grad).all())

    def test_zero_gradient_when_the_term_is_constant(self):
        """A term that ignores z must inject exactly nothing."""
        def constant_term(z, mb_labels):
            return z.sum() * 0.0
        images = self.base.clone().requires_grad_(True)
        surrogate, _, _ = self.ex.vjp_surrogate(
            images, self.labels, constant_term, microbatch_size=4)
        grad = torch.autograd.grad(surrogate, images)[0]
        self.assertEqual(float(grad.abs().max()), 0.0)

    def test_subset_selection_scores_k_and_round_robins(self):
        self.ex.max_samples_per_step = 2
        try:
            seen = []
            for _ in range(3):
                images = self.base.clone().requires_grad_(True)
                _, z, selected = self.ex.vjp_surrogate(
                    images, self.labels, self._term_fn(0.5), microbatch_size=2)
                self.assertEqual(z.shape[0], 2)
                seen.append(selected.tolist())
            self.assertEqual(seen, [[0, 1], [2, 3], [0, 1]])
        finally:
            self.ex.max_samples_per_step = 0
            self.ex._sample_cursor.zero_()

    def test_detached_images_are_refused(self):
        with self.assertRaises(RuntimeError):
            self.ex.vjp_surrogate(self.base, self.labels, self._term_fn(1.0))

    def test_out_of_range_images_are_refused(self):
        images = (self.base * 3.0).requires_grad_(True)
        with self.assertRaises(ValueError):
            self.ex.vjp_surrogate(images, self.labels, self._term_fn(1.0))


@_SKIP
class Bf16PrecisionTest(unittest.TestCase):
    """What reduced precision does to the answer state, and to its gradient.

    Measured on six real ImageNet val images at 256px, bf16 against an fp32
    reference (see the module docstring of ``qwen_answer_state``):

        quantity                       bf16 vs fp32      fp16 vs fp32
        z (the feature itself)         cos 0.9999        cos 1.0000
        dz-driven image gradient       cos 0.03          cos -0.48

    High feature cosine does not establish equivalent classifier probabilities
    or image gradients. These observations alone do not identify weight
    rounding, activation/backward rounding, cancellation, or under/overflow as
    the cause. In particular a larger fp16 gradient does not diagnose underflow.
    Compare fixed inputs/heads/layers, per-branch gradients and loss scales
    against fp32 before attributing the discrepancy. VJPSurrogateTest checks
    the surrogate itself within a fixed dtype.

    This test asserts only what must hold: the forward is faithful, and one
    dtype is self-consistent at a fixed batch shape.  The gradient divergence is
    printed rather than bounded, because a threshold on it would be a threshold
    on a numerical accident.
    """

    def setUp(self):
        self.labels = torch.tensor([0, 3, 1, 4], device="cuda")
        torch.manual_seed(0)
        self.head = VLMLinearHead(3584, 5).cuda()
        self.head.set_feature_norm_stats(torch.zeros(3584).cuda(),
                                         torch.ones(3584).cuda() * 100.0)
        self.head.eval().requires_grad_(False)
        torch.manual_seed(1)
        self.base = torch.rand(4, 3, 256, 256, device="cuda")

    def _term_fn(self, z, mb_labels):
        logp = self.head.log_probs(z)
        return logp.gather(1, mb_labels.view(-1, 1)).squeeze(1).mean()

    def test_forward_feature_survives_bf16(self):
        z = {}
        for dtype in ("fp32", "bf16"):
            with torch.no_grad():
                extractor = _extractor(dtype)
                z[dtype] = extractor.answer_states(self.base)[extractor.layer].float()
        cos = float(torch.nn.functional.cosine_similarity(
            z["bf16"], z["fp32"], dim=1).mean())
        rel = float((z["bf16"] - z["fp32"]).norm() / z["fp32"].norm())
        print(f"\n  [bf16 forward] cos={cos:.5f} rel_err={rel:.4f}")
        self.assertGreater(cos, 0.99)
        self.assertLess(rel, 0.05)

    def test_bf16_gradient_is_self_consistent(self):
        """Two identical bf16 calls must agree in direction.

        They are not bit-identical: the attention backward reduces with atomics,
        so repeated calls differ by ~1e-5 in absolute terms.  What matters for
        training is that the direction is stable, which it is -- unlike the
        comparison across dtypes.
        """
        extractor = _extractor("bf16")
        grads = []
        for _ in range(2):
            images = self.base.clone().requires_grad_(True)
            surrogate, _, _ = extractor.vjp_surrogate(
                images, self.labels, self._term_fn, microbatch_size=2)
            grads.append(torch.autograd.grad(surrogate, images)[0])
        cos = float(torch.nn.functional.cosine_similarity(
            grads[0].reshape(1, -1).float(), grads[1].reshape(1, -1).float()))
        max_abs = float((grads[0] - grads[1]).abs().max())
        print(f"\n  [bf16 repeat] cos={cos:.6f} max_abs_diff={max_abs:.2e}")
        self.assertGreater(cos, 0.999)


if __name__ == "__main__":
    unittest.main(verbosity=2)
