"""Unit + smoke tests for the VLM sampled-label log q - log p conditional term.

Covers the ten integration checks from the trial spec (Step B) plus the head,
buffer, metric and launch-refusal logic:

 1. p and q initialise identically
 2. the generator backward reaches the generator through a frozen encoder
 3. the encoder receives no parameter gradient
 4. the p head receives no updates
 5. the q update does not backprop into the generator
 6. q_student changes under generated CE training
 7. q_teacher moves more slowly than q_student
 8. log q - log p == 0 immediately at q initialisation
 9. the conditional image-space gradient is finite and non-zero once q != p
10. checkpoint/resume restores q exactly

Run:  /home/nvidia/miniconda3/envs/fdloss/bin/python -m unittest tests.test_vlm_delta -v
"""

import math
import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_linear_heads import (  # noqa: E402
    ClassBalancedFeatureBuffer,
    LocalClassMap,
    PerClassAccumulator,
    P_HEAD_FORMAT_VERSION,
    VLMDeltaHeads,
    VLMLinearHead,
    build_head_from_checkpoint,
    class_ids_hash,
    delta_distribution_metrics,
    head_metrics,
    load_p_head_checkpoint,
    p_head_checksum,
    p_head_identity,
    pq_agreement_metrics,
)

D, C = 16, 5


def make_p_head(seed=0, feature_dim=D, num_classes=C):
    torch.manual_seed(seed)
    head = VLMLinearHead(feature_dim, num_classes)
    with torch.no_grad():
        head.linear.weight.normal_(0, 0.5)
        head.linear.bias.normal_(0, 0.1)
    head.set_feature_norm_stats(torch.randn(feature_dim),
                                torch.rand(feature_dim) + 0.5)
    return head.eval().requires_grad_(False)


def make_heads(seed=0, temperature=1.0, ema_beta=0.9):
    return VLMDeltaHeads(make_p_head(seed), temperature=temperature,
                         ema_beta=ema_beta)


class FrozenEncoder(nn.Module):
    """Stand-in for the frozen VLM: differentiable in x, frozen in its params."""

    def __init__(self, out_dim=D):
        super().__init__()
        torch.manual_seed(3)
        self.conv = nn.Conv2d(3, out_dim, 3, stride=2, padding=1)
        self.eval().requires_grad_(False)

    def forward(self, x):
        return self.conv(x).mean(dim=(2, 3))


class TinyGenerator(nn.Module):
    def __init__(self, num_classes=C, size=8):
        super().__init__()
        torch.manual_seed(4)
        self.embed = nn.Embedding(num_classes, 3 * size * size)
        self.size = size

    def forward(self, noise, labels):
        return (noise + self.embed(labels).view(-1, 3, self.size, self.size)).sigmoid()


# ---------------------------------------------------------------------------
# Head basics
# ---------------------------------------------------------------------------

class VLMLinearHeadTest(unittest.TestCase):
    def test_log_probs_normalised(self):
        head = make_p_head()
        z = torch.randn(7, D)
        lp = head.log_probs(z)
        self.assertEqual(lp.shape, (7, C))
        self.assertTrue(torch.allclose(lp.exp().sum(-1), torch.ones(7), atol=1e-5))

    def test_temperature_preserves_ranking(self):
        head = make_p_head()
        z = torch.randn(32, D)
        a = head.logits(z, 1.0).argsort(-1)
        b = head.logits(z, 7.5).argsort(-1)
        self.assertTrue(torch.equal(a, b))

    def test_frozen_feature_norm_applied(self):
        head = make_p_head()
        z = torch.randn(4, D)
        expected = head.linear((z - head.feature_mean) / head.feature_std)
        self.assertTrue(torch.allclose(head.logits(z), expected, atol=1e-6))

    def test_feature_norm_buffers_are_not_parameters(self):
        head = make_p_head()
        names = {n for n, _ in head.named_parameters()}
        self.assertEqual(names, {"linear.weight", "linear.bias"})

    def test_rejects_wrong_feature_dim(self):
        with self.assertRaises(ValueError):
            make_p_head().logits(torch.randn(3, D + 1))


# ---------------------------------------------------------------------------
# Step B, checks 1 / 7 / 8
# ---------------------------------------------------------------------------

class InitialisationTest(unittest.TestCase):
    def test_q_initialised_from_p_exactly(self):
        heads = make_heads()
        for q in (heads.q_student, heads.q_teacher):
            self.assertTrue(torch.equal(q.weight, heads.p_head.weight))
            self.assertTrue(torch.equal(q.bias, heads.p_head.bias))
            self.assertTrue(torch.equal(q.feature_mean, heads.p_head.feature_mean))
            self.assertTrue(torch.equal(q.feature_std, heads.p_head.feature_std))

    def test_q_does_not_share_parameters_with_p(self):
        heads = make_heads()
        self.assertIsNot(heads.q_student.linear.weight, heads.p_head.linear.weight)
        self.assertIsNot(heads.q_teacher.linear.weight, heads.q_student.linear.weight)
        with torch.no_grad():
            heads.q_student.linear.weight.add_(1.0)
        self.assertFalse(torch.equal(heads.q_student.weight, heads.p_head.weight))
        self.assertTrue(torch.equal(heads.q_teacher.weight, heads.p_head.weight))

    def test_delta_is_exactly_zero_at_init(self):
        heads = make_heads()
        z = torch.randn(64, D)
        labels = torch.randint(0, C, (64,))
        delta, logq, logp = heads.delta_log_qp(z, labels)
        self.assertLess(float(delta.abs().max()), 1e-6)
        self.assertTrue(torch.allclose(logq, logp, atol=1e-6))

    def test_init_equality_check_reports_zero(self):
        heads = make_heads()
        report = heads.init_equality_check(torch.randn(32, D))
        for key, value in report.items():
            self.assertLess(value, 1e-6, key)

    def test_pq_metrics_degenerate_when_q_equals_p(self):
        heads = make_heads()
        z = torch.randn(48, D)
        labels = torch.randint(0, C, (48,))
        m = pq_agreement_metrics(heads.p_log_probs(z), heads.q_teacher_log_probs(z),
                                 labels)
        self.assertAlmostEqual(m["vlm_pq_full_kl_qp"], 0.0, places=6)
        self.assertAlmostEqual(m["vlm_pq_full_kl_pq"], 0.0, places=6)
        self.assertAlmostEqual(m["vlm_pq_js"], 0.0, places=6)
        self.assertEqual(m["vlm_pq_top1_agreement"], 1.0)
        self.assertAlmostEqual(m["vlm_target_prob_gap"], 0.0, places=6)

    def test_temperature_must_be_positive(self):
        with self.assertRaises(ValueError):
            VLMDeltaHeads(make_p_head(), temperature=0.0)
        with self.assertRaises(ValueError):
            VLMDeltaHeads(make_p_head(), ema_beta=1.0)


# ---------------------------------------------------------------------------
# Step B, checks 4 / 5 / 6 / 7
# ---------------------------------------------------------------------------

class QTrainingTest(unittest.TestCase):
    def _train_q(self, heads, steps=25, lr=0.1, batch=128):
        optimizer = torch.optim.SGD(heads.q_student.linear.parameters(), lr=lr)
        torch.manual_seed(11)
        # A generator-like distribution: features that DO carry class structure,
        # so q has something to learn and must move away from p.
        centers = torch.randn(C, D) * 3.0
        for _ in range(steps):
            y = torch.randint(0, C, (batch,))
            z = centers[y] + 0.3 * torch.randn(batch, D)
            loss = F.cross_entropy(heads.q_student.logits(z, heads.temperature), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            heads.q_train_steps.add_(1)
            heads.ema_update()
        return heads

    def test_q_student_changes_under_generated_ce(self):
        heads = self._train_q(make_heads())
        drift = heads.drift_metrics()
        self.assertGreater(drift["q_student_weight_delta_l2"], 1e-3)

    def test_q_teacher_moves_more_slowly_than_student(self):
        heads = self._train_q(make_heads(ema_beta=0.99))
        drift = heads.drift_metrics()
        self.assertGreater(drift["q_student_weight_delta_l2"],
                           drift["q_teacher_weight_delta_l2"])
        self.assertGreater(drift["q_student_bias_delta_l2"],
                           drift["q_teacher_bias_delta_l2"])

    def test_slower_ema_means_slower_teacher(self):
        fast = self._train_q(make_heads(ema_beta=0.5)).drift_metrics()
        slow = self._train_q(make_heads(ema_beta=0.99)).drift_metrics()
        self.assertGreater(fast["q_teacher_weight_delta_l2"],
                           slow["q_teacher_weight_delta_l2"])

    def test_p_head_never_updates(self):
        heads = make_heads()
        p_before = heads.p_head.weight.clone()
        b_before = heads.p_head.bias.clone()
        self._train_q(heads)
        self.assertTrue(torch.equal(heads.p_head.weight, p_before))
        self.assertTrue(torch.equal(heads.p_head.bias, b_before))
        self.assertFalse(heads.p_head.linear.weight.requires_grad)

    def test_q_update_does_not_reach_the_generator(self):
        """Detached features must carry no path back to the generator."""
        generator = TinyGenerator()
        encoder = FrozenEncoder()
        heads = make_heads()
        noise = torch.randn(8, 3, 8, 8)
        labels = torch.randint(0, C, (8,))
        images = generator(noise, labels)
        z_detached = encoder(images).detach()
        self.assertFalse(z_detached.requires_grad)
        loss = F.cross_entropy(heads.q_student.logits(z_detached), labels)
        loss.backward()
        for name, param in generator.named_parameters():
            self.assertIsNone(param.grad, f"generator param {name} got a q gradient")
        for name, param in encoder.named_parameters():
            self.assertIsNone(param.grad, f"encoder param {name} got a q gradient")
        self.assertIsNotNone(heads.q_student.linear.weight.grad)

    def test_ema_update_matches_the_formula(self):
        heads = make_heads(ema_beta=0.8)
        teacher_before = heads.q_teacher.weight.clone()
        with torch.no_grad():
            heads.q_student.linear.weight.add_(torch.full_like(
                heads.q_student.linear.weight, 2.0))
        heads.ema_update()
        expected = 0.8 * teacher_before + 0.2 * heads.q_student.weight
        self.assertTrue(torch.allclose(heads.q_teacher.weight, expected, atol=1e-6))


# ---------------------------------------------------------------------------
# Step B, checks 2 / 3 / 9
# ---------------------------------------------------------------------------

class GeneratorGradientTest(unittest.TestCase):
    def _setup(self):
        generator = TinyGenerator()
        encoder = FrozenEncoder()
        heads = make_heads()
        noise = torch.randn(16, 3, 8, 8)
        labels = torch.randint(0, C, (16,))
        return generator, encoder, heads, noise, labels

    def test_gradient_is_exactly_zero_while_q_equals_p(self):
        generator, encoder, heads, noise, labels = self._setup()
        images = generator(noise, labels)
        delta, _, _ = heads.delta_log_qp(encoder(images), labels)
        grad = torch.autograd.grad(delta.mean(), images, retain_graph=True)[0]
        self.assertLess(float(grad.abs().max()), 1e-6)

    def test_gradient_is_finite_and_nonzero_once_q_differs(self):
        generator, encoder, heads, noise, labels = self._setup()
        with torch.no_grad():
            heads.q_teacher.linear.weight.add_(0.2 * torch.randn_like(
                heads.q_teacher.linear.weight))
        images = generator(noise, labels)
        delta, _, _ = heads.delta_log_qp(encoder(images), labels)
        grad_x = torch.autograd.grad(delta.mean(), images, retain_graph=True)[0]
        self.assertTrue(torch.isfinite(grad_x).all())
        self.assertGreater(float(grad_x.norm()), 0.0)

        # ... and it reaches the generator's own parameters
        delta.mean().backward()
        grads = [p.grad for p in generator.parameters() if p.grad is not None]
        self.assertTrue(grads, "no generator parameter received a gradient")
        self.assertGreater(sum(float(g.norm()) for g in grads), 0.0)

    def test_frozen_encoder_and_heads_receive_no_parameter_gradient(self):
        generator, encoder, heads, noise, labels = self._setup()
        with torch.no_grad():
            heads.q_teacher.linear.weight.add_(0.2)
        delta, _, _ = heads.delta_log_qp(encoder(generator(noise, labels)), labels)
        delta.mean().backward()
        for name, param in encoder.named_parameters():
            self.assertIsNone(param.grad, f"frozen encoder param {name} got a gradient")
        for head_name, head in (("p", heads.p_head), ("q_teacher", heads.q_teacher)):
            for name, param in head.named_parameters():
                self.assertIsNone(param.grad, f"{head_name}.{name} got a gradient")

    def test_delta_sign_convention(self):
        """Descending the term must push z toward higher log p(c|z)."""
        heads = make_heads()
        with torch.no_grad():
            # make q flat so grad(log q) ~ 0 and the field is -grad log p
            heads.q_teacher.linear.weight.zero_()
            heads.q_teacher.linear.bias.zero_()
        z = torch.randn(64, D, requires_grad=True)
        labels = torch.randint(0, C, (64,))
        delta, _, logp_c = heads.delta_log_qp(z, labels)
        grad_z = torch.autograd.grad(delta.mean(), z)[0]
        # a small descent step on the loss should raise log p(c|z)
        with torch.no_grad():
            z_new = z - 1e-2 * grad_z
            logp_new = heads.p_log_probs(z_new).gather(
                1, labels.view(-1, 1)).squeeze(1)
        self.assertGreater(float(logp_new.mean()), float(logp_c.mean()))


# ---------------------------------------------------------------------------
# Step B, check 10
# ---------------------------------------------------------------------------

class CheckpointTest(unittest.TestCase):
    def test_q_state_round_trip_is_exact(self):
        heads = make_heads(ema_beta=0.97)
        with torch.no_grad():
            heads.q_student.linear.weight.add_(0.3 * torch.randn_like(
                heads.q_student.linear.weight))
        heads.ema_update()
        heads.q_train_steps.add_(123)
        state = heads.q_state_dict()

        restored = make_heads(ema_beta=0.97)
        restored.load_q_state_dict(state)
        self.assertTrue(torch.equal(restored.q_student.weight, heads.q_student.weight))
        self.assertTrue(torch.equal(restored.q_student.bias, heads.q_student.bias))
        self.assertTrue(torch.equal(restored.q_teacher.weight, heads.q_teacher.weight))
        self.assertTrue(torch.equal(restored.q_teacher.bias, heads.q_teacher.bias))
        self.assertEqual(int(restored.q_train_steps.item()), 123)
        self.assertTrue(restored.q_student.linear.weight.requires_grad)
        self.assertFalse(restored.q_teacher.linear.weight.requires_grad)

    def test_restore_refuses_a_different_temperature(self):
        state = make_heads(temperature=2.0).q_state_dict()
        with self.assertRaises(ValueError):
            make_heads(temperature=3.0).load_q_state_dict(state)

    def test_restore_refuses_a_different_ema_beta(self):
        state = make_heads(ema_beta=0.9).q_state_dict()
        with self.assertRaises(ValueError):
            make_heads(ema_beta=0.99).load_q_state_dict(state)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class BufferTest(unittest.TestCase):
    def _buffer(self, per_class=4):
        return ClassBalancedFeatureBuffer(C, D, per_class, device="cpu",
                                          dtype=torch.float32, seed=5)

    def test_push_is_class_balanced_and_bounded(self):
        buf = self._buffer(per_class=4)
        # deliberately imbalanced input: class 0 pushed 100x, others 1x
        y = torch.cat([torch.zeros(100, dtype=torch.long),
                       torch.arange(1, C)])
        buf.push(torch.randn(y.numel(), D), y)
        stats = buf.stats()
        self.assertEqual(stats["q_buffer_class_coverage"], 1.0)
        self.assertEqual(stats["q_buffer_max_samples_per_class"], 4.0)
        self.assertEqual(stats["q_buffer_min_samples_per_class"], 1.0)
        self.assertEqual(stats["q_buffer_size"], 4.0 + (C - 1))

    def test_push_stores_the_right_features(self):
        buf = self._buffer(per_class=8)
        z = torch.randn(6, D)
        y = torch.tensor([0, 1, 2, 0, 1, 2])
        buf.push(z, y)
        self.assertTrue(torch.allclose(buf.feats[0, 0], z[0]))
        self.assertTrue(torch.allclose(buf.feats[0, 1], z[3]))
        self.assertTrue(torch.allclose(buf.feats[2, 1], z[5]))

    def test_ring_wraps(self):
        buf = self._buffer(per_class=2)
        z = torch.randn(5, D)
        y = torch.zeros(5, dtype=torch.long)
        buf.push(z, y)
        self.assertEqual(int(buf.fill[0]), 2)
        self.assertTrue(torch.allclose(buf.feats[0, 0], z[4]))
        self.assertTrue(torch.allclose(buf.feats[0, 1], z[3]))

    def test_sample_is_balanced_over_present_classes(self):
        buf = self._buffer(per_class=8)
        y = torch.cat([torch.zeros(200, dtype=torch.long), torch.ones(1, dtype=torch.long)])
        buf.push(torch.randn(y.numel(), D), y)
        _, classes = buf.sample(4000)
        frac_class1 = float((classes == 1).float().mean())
        self.assertGreater(frac_class1, 0.35)  # ~0.5 despite 200:1 push imbalance
        self.assertLess(frac_class1, 0.65)

    def test_sample_returns_none_when_empty(self):
        self.assertIsNone(self._buffer().sample(8))

    def test_state_dict_round_trip(self):
        buf = self._buffer(per_class=4)
        buf.push(torch.randn(20, D), torch.randint(0, C, (20,)))
        state = buf.state_dict()
        other = self._buffer(per_class=4)
        other.load_state_dict(state)
        self.assertTrue(torch.equal(other.feats, buf.feats))
        self.assertTrue(torch.equal(other.fill, buf.fill))
        self.assertTrue(torch.equal(other.ptr, buf.ptr))
        self.assertEqual(other.total_pushed, buf.total_pushed)

    def test_load_refuses_a_different_shape(self):
        buf = self._buffer(per_class=4)
        state = buf.state_dict()
        with self.assertRaises(ValueError):
            self._buffer(per_class=8).load_state_dict(state)

    def test_two_ranks_stay_identical(self):
        """Same gathered data + same seed => bit-identical buffers, no collective."""
        a, b = self._buffer(6), self._buffer(6)
        torch.manual_seed(0)
        for _ in range(5):
            z, y = torch.randn(24, D), torch.randint(0, C, (24,))
            a.push(z, y)
            b.push(z, y)
        za, ya = a.sample(32)
        zb, yb = b.sample(32)
        self.assertTrue(torch.equal(ya, yb))
        self.assertTrue(torch.equal(za, zb))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class MetricTest(unittest.TestCase):
    def test_head_metrics_on_a_known_case(self):
        log_probs = torch.log(torch.tensor([[0.7, 0.2, 0.1],
                                            [0.1, 0.2, 0.7]]))
        labels = torch.tensor([0, 0])
        m = head_metrics(log_probs, labels, "x")
        self.assertAlmostEqual(m["x_top1"], 0.5, places=6)
        self.assertAlmostEqual(m["x_target_prob"], 0.4, places=5)
        self.assertAlmostEqual(m["x_target_rank"], 2.0, places=5)  # ranks 1 and 3
        self.assertAlmostEqual(m["x_ce"], -math.log(0.7) / 2 - math.log(0.1) / 2,
                               places=5)
        self.assertAlmostEqual(m["x_margin"],
                               ((math.log(0.7) - math.log(0.2))
                                + (math.log(0.1) - math.log(0.7))) / 2, places=5)

    def test_rank_is_optimistic_on_ties(self):
        """Matches ProbeClassifier: rank counts strictly-better classes only."""
        log_probs = torch.log(torch.tensor([[0.25, 0.25, 0.25, 0.25]]))
        m = head_metrics(log_probs, torch.zeros(1, dtype=torch.long), "x")
        self.assertAlmostEqual(m["x_target_rank"], 1.0, places=6)

    def test_rank_at_chance_for_random_logits(self):
        torch.manual_seed(0)
        n_cls = 100
        log_probs = torch.log_softmax(torch.randn(20000, n_cls), dim=-1)
        labels = torch.randint(0, n_cls, (20000,))
        m = head_metrics(log_probs, labels, "x")
        self.assertAlmostEqual(m["x_target_rank"], (n_cls + 1) / 2, delta=2.0)
        self.assertAlmostEqual(m["x_top1"], 1.0 / n_cls, delta=0.01)

    def test_uniform_entropy(self):
        n_cls = 8
        log_probs = torch.full((4, n_cls), -math.log(n_cls))
        m = head_metrics(log_probs, torch.zeros(4, dtype=torch.long), "x")
        self.assertAlmostEqual(m["x_entropy"], math.log(n_cls), places=5)

    def test_delta_distribution_quantiles(self):
        d = torch.linspace(-1.0, 1.0, 1001)
        m = delta_distribution_metrics(d)
        self.assertAlmostEqual(m["vlm_delta_logqp_p50"], 0.0, places=5)
        self.assertAlmostEqual(m["vlm_delta_logqp_p10"], -0.8, places=3)
        self.assertAlmostEqual(m["vlm_delta_logqp_p90"], 0.8, places=3)
        self.assertAlmostEqual(m["vlm_delta_logqp_min"], -1.0, places=5)
        self.assertAlmostEqual(m["vlm_delta_logqp_max"], 1.0, places=5)
        self.assertGreater(m["vlm_delta_abs_p99"], m["vlm_delta_abs_p95"])

    def test_pq_metrics_detect_disagreement(self):
        logp = torch.log_softmax(torch.tensor([[3.0, 0.0, 0.0]]), dim=-1)
        logq = torch.log_softmax(torch.tensor([[0.0, 3.0, 0.0]]), dim=-1)
        m = pq_agreement_metrics(logp, logq, torch.zeros(1, dtype=torch.long))
        self.assertGreater(m["vlm_pq_full_kl_qp"], 1.0)
        self.assertEqual(m["vlm_pq_top1_agreement"], 0.0)
        self.assertEqual(m["vlm_pq_argmax_disagreement"], 1.0)
        self.assertGreater(m["vlm_target_prob_gap"], 0.0)

    def test_per_class_accumulator(self):
        acc = PerClassAccumulator(C, list(range(0, 10 * C, 10)), device="cpu")
        heads = make_heads()
        z = torch.randn(64, D)
        y = torch.randint(0, C, (64,))
        acc.update(y, logp_all=heads.p_log_probs(z), logq_all=heads.q_teacher_log_probs(z))
        records = acc.to_records()
        self.assertEqual(len(records), C)
        self.assertEqual(sum(r["count"] for r in records), 64)
        self.assertEqual(records[1]["class_id"], 10)
        for record in records:
            if record["count"]:
                self.assertAlmostEqual(record["delta_logqp"], 0.0, places=5)

    def test_per_class_probe_is_nan_when_never_supplied(self):
        acc = PerClassAccumulator(C, list(range(C)), device="cpu")
        heads = make_heads()
        z, y = torch.randn(32, D), torch.randint(0, C, (32,))
        acc.update(y, logp_all=heads.p_log_probs(z), logq_all=heads.q_teacher_log_probs(z))
        for record in acc.to_records():
            if record["count"]:
                self.assertTrue(math.isnan(record["probe_top1"]))
        acc.update(y, logp_all=heads.p_log_probs(z), logq_all=heads.q_teacher_log_probs(z),
                   probe_correct=torch.ones(32))
        for record in acc.to_records():
            if record["count"]:
                self.assertFalse(math.isnan(record["probe_top1"]))

    def test_per_class_dump(self):
        acc = PerClassAccumulator(C, list(range(C)), device="cpu")
        heads = make_heads()
        z, y = torch.randn(16, D), torch.randint(0, C, (16,))
        acc.update(y, logp_all=heads.p_log_probs(z), logq_all=heads.q_student_log_probs(z))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vlm_per_class_step_00100.json")
            acc.dump(path, step=100)
            import json
            with open(path) as f:
                payload = json.load(f)
        self.assertEqual(payload["step"], 100)
        self.assertEqual(len(payload["per_class"]), C)


# ---------------------------------------------------------------------------
# Class mapping and p-head checkpoint contract
# ---------------------------------------------------------------------------

class ClassMapTest(unittest.TestCase):
    def test_round_trip(self):
        ids = [0, 10, 20, 30, 40]
        cmap = LocalClassMap(ids, num_global=1000, device="cpu")
        globals_ = torch.tensor([30, 0, 40])
        local = cmap.to_local(globals_)
        self.assertTrue(torch.equal(local, torch.tensor([3, 0, 4])))
        self.assertTrue(torch.equal(cmap.to_global(local), globals_))

    def test_validate_rejects_a_missing_class(self):
        cmap = LocalClassMap([0, 10], num_global=1000, device="cpu")
        with self.assertRaises(ValueError) as ctx:
            cmap.validate([0, 10, 20])
        self.assertIn("no output unit", str(ctx.exception))

    def test_validate_rejects_a_smaller_drawable_set(self):
        cmap = LocalClassMap([0, 10, 20], num_global=1000, device="cpu")
        with self.assertRaises(ValueError) as ctx:
            cmap.validate([0, 10])
        self.assertIn("softmax denominator", str(ctx.exception))


class PHeadCheckpointTest(unittest.TestCase):
    def _payload(self, class_ids=(0, 10, 20, 30, 40)):
        head = make_p_head(num_classes=len(class_ids))
        payload = {
            "format_version": P_HEAD_FORMAT_VERSION,
            "vlm_model_name": "test-vlm",
            "vlm_pool_type": "cls",
            "vlm_target_size": 224,
            "vlm_input_size": 256,
            "feature_dim": D,
            "num_classes": len(class_ids),
            "class_ids": list(class_ids),
            "class_ids_sha256": class_ids_hash(list(class_ids)),
            "feature_norm": "standardize",
            "feature_mean": head.feature_mean.clone(),
            "feature_std": head.feature_std.clone(),
            "head_state_dict": {k: v.clone() for k, v in head.linear.state_dict().items()},
            "temperature": 1.5,
        }
        payload["p_head_sha256"] = p_head_checksum(payload)
        return payload

    def test_load_and_rebuild(self):
        payload = self._payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p_head.pt")
            torch.save(payload, path)
            loaded = load_p_head_checkpoint(path)
        head = build_head_from_checkpoint(loaded, device="cpu")
        self.assertTrue(torch.equal(head.linear.weight,
                                    payload["head_state_dict"]["weight"]))
        self.assertTrue(torch.equal(head.feature_std, payload["feature_std"]))

    def test_missing_key_is_rejected(self):
        payload = self._payload()
        del payload["temperature"]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p_head.pt")
            torch.save(payload, path)
            with self.assertRaises(ValueError) as ctx:
                load_p_head_checkpoint(path)
        self.assertIn("missing keys", str(ctx.exception))

    def test_tampered_class_ids_are_rejected(self):
        payload = self._payload()
        payload["class_ids"] = [0, 10, 20, 30, 50]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p_head.pt")
            torch.save(payload, path)
            with self.assertRaises(ValueError) as ctx:
                load_p_head_checkpoint(path)
        self.assertIn("hash mismatch", str(ctx.exception))

    def test_identity_changes_with_the_weights(self):
        a = self._payload()
        b = self._payload()
        b["head_state_dict"]["weight"] = b["head_state_dict"]["weight"] + 1.0
        self.assertNotEqual(p_head_identity(a)["p_head_sha256"],
                            p_head_identity(b)["p_head_sha256"])

    def test_identity_changes_with_the_class_set(self):
        a = self._payload()
        b = self._payload(class_ids=(0, 10, 20, 30, 50))
        self.assertNotEqual(p_head_identity(a)["class_ids_sha256"],
                            p_head_identity(b)["class_ids_sha256"])


# ---------------------------------------------------------------------------
# Training-script logic (no GPU / no model needed)
# ---------------------------------------------------------------------------

class TrainScriptLogicTest(unittest.TestCase):
    def setUp(self):
        from conditional_main_fd_vlm_delta import (  # noqa: F401
            evaluate_vlm_takeoff_gate, get_args_parser, validate_train_class_ids,
            validate_vlm_takeoff_gate_args, vlm_scale_schedule,
        )
        self.mod = sys.modules["conditional_main_fd_vlm_delta"]

    def test_instant_metrics_stay_instant_after_nondiagnostic_resume(self):
        import json

        from utils.logging_util import MetricLogger

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "metrics.jsonl")
            metrics = MetricLogger(output_file=path)
            # Resume at step 436, before the next diagnostic at step 440.
            self.mod.update_training_metrics(
                metrics, self.mod.INSTANT_METERS,
                loss=3.0, vlm_logp_c=-10.0)
            metrics.dump_in_output_file(436, 1.0, 0.0)
            for step, ratio in ((440, 0.07), (450, 0.5)):
                self.mod.update_training_metrics(
                    metrics, self.mod.INSTANT_METERS,
                    loss=2.0, grad_ratio_vlm_fd=ratio, q_train_steps=step + 1)
                metrics.dump_in_output_file(step, 1.0, 0.0)
            with open(path) as f:
                rows = [json.loads(line) for line in f]

        self.assertNotIn("grad_ratio_vlm_fd", rows[0])
        self.assertAlmostEqual(rows[-1]["grad_ratio_vlm_fd"], 0.5)
        self.assertEqual(rows[-1]["q_train_steps"], 451)
        self.assertEqual(metrics.meters["grad_ratio_vlm_fd"].deque.maxlen, 1)
        self.assertEqual(metrics.meters["loss"].deque.maxlen, 20)

    def test_unproduced_instant_metrics_do_not_create_empty_windows(self):
        from utils.logging_util import MetricLogger

        metrics = MetricLogger()
        self.mod.update_training_metrics(
            metrics, self.mod.INSTANT_METERS,
            loss=3.0, grad_x_fd=None, vlm_logq_c=-4.6)
        self.assertNotIn("grad_x_fd", metrics.meters)
        self.assertNotIn("q_buffer_size", metrics.meters)
        self.assertNotIn("q_teacher_weight_delta_l2", metrics.meters)
        self.assertIn("vlm_logq_c", str(metrics))

    def test_backend_instant_metrics_keep_latest_value(self):
        from utils.logging_util import MetricLogger

        metrics = MetricLogger()
        names = list(self.mod.INSTANT_METERS)
        for backend_names in self.mod.BACKEND_INSTANT_METERS.values():
            names.extend(backend_names)
        for value in (24.0, 12.0, 48.0):
            self.mod.update_training_metrics(
                metrics, names, vlm_answer_state_samples=value)
        self.assertEqual(metrics.meters["vlm_answer_state_samples"].median, 48.0)

    def test_scale_schedule(self):
        from argparse import Namespace
        args = Namespace(vlm_delta_weight=0.4, vlm_delta_warmup_steps=100,
                         vlm_delta_ramp_steps=100)
        self.assertEqual(self.mod.vlm_scale_schedule(0, args), 0.0)
        self.assertEqual(self.mod.vlm_scale_schedule(99, args), 0.0)
        self.assertAlmostEqual(self.mod.vlm_scale_schedule(150, args), 0.2)
        self.assertAlmostEqual(self.mod.vlm_scale_schedule(200, args), 0.4)
        self.assertAlmostEqual(self.mod.vlm_scale_schedule(5000, args), 0.4)

    def test_takeoff_gate_logic(self):
        good = {"cond_delta": 0.2, "vlm_p_target_rank_ratio_to_chance": 0.8}
        aborted, failures = self.mod.evaluate_vlm_takeoff_gate(good)
        self.assertFalse(aborted)
        self.assertEqual(failures, {"cond_delta": False, "p_rank": False})

        aborted, _ = self.mod.evaluate_vlm_takeoff_gate(
            {"cond_delta": 0.05, "vlm_p_target_rank_ratio_to_chance": 0.8})
        self.assertTrue(aborted)

        aborted, _ = self.mod.evaluate_vlm_takeoff_gate(
            {"cond_delta": 0.2, "vlm_p_target_rank_ratio_to_chance": 0.95})
        self.assertTrue(aborted)

    def test_takeoff_gate_fails_closed(self):
        self.assertTrue(self.mod.evaluate_vlm_takeoff_gate({})[0])
        self.assertTrue(self.mod.evaluate_vlm_takeoff_gate(
            {"cond_delta": float("nan"),
             "vlm_p_target_rank_ratio_to_chance": 0.1})[0])

    def test_takeoff_gate_all_logic(self):
        aborted, _ = self.mod.evaluate_vlm_takeoff_gate(
            {"cond_delta": 0.05, "vlm_p_target_rank_ratio_to_chance": 0.8},
            logic="all")
        self.assertFalse(aborted)

    def test_class_id_validation(self):
        self.assertIsNone(self.mod.validate_train_class_ids(None, 1000))
        self.assertEqual(self.mod.validate_train_class_ids([1, 2], 1000), [1, 2])
        with self.assertRaises(ValueError):
            self.mod.validate_train_class_ids([1, 1], 1000)
        with self.assertRaises(ValueError):
            self.mod.validate_train_class_ids([1, 1000], 1000)

    def test_gate_arg_validation(self):
        from argparse import Namespace
        base = dict(vlm_takeoff_gate_step=5000, vlm_takeoff_cond_delta_min=0.1,
                    vlm_takeoff_no_rank=False, epochs=8, steps_per_epoch=1250,
                    vlm_delta_warmup_steps=0, vlm_delta_ramp_steps=500)
        self.mod.validate_vlm_takeoff_gate_args(Namespace(**base))
        with self.assertRaises(ValueError):
            self.mod.validate_vlm_takeoff_gate_args(
                Namespace(**{**base, "vlm_takeoff_gate_step": 5000, "epochs": 4}))
        with self.assertRaises(ValueError):
            self.mod.validate_vlm_takeoff_gate_args(
                Namespace(**{**base, "vlm_takeoff_gate_step": 100}))

    def test_parser_defaults_are_the_documented_ones(self):
        args = self.mod.get_args_parser().parse_args([])
        self.assertEqual(args.vlm_delta_clamp, 0.0)          # no silent clamping
        self.assertEqual(args.vlm_q_bootstrap_updates, 0)    # q == p at step 0
        self.assertEqual(args.vlm_q_ema_beta, 0.999)
        self.assertEqual(args.vlm_takeoff_gate_step, -1)     # gate opt-in
        self.assertIsNone(args.vlm_head_temperature)         # taken from the p head

    def test_backend_specific_meters_are_not_registered_unconditionally(self):
        """Declare backend-only diagnostics separately from shared metrics."""
        unconditional = set(self.mod.INSTANT_METERS)
        for backend, names in self.mod.BACKEND_INSTANT_METERS.items():
            overlap = unconditional.intersection(names)
            self.assertEqual(
                overlap, set(),
                f"{backend} meters {sorted(overlap)} are registered for every run "
                f"but only produced by that backend")

    def test_fill_all_queues_collector_is_optional(self):
        import inspect

        from frechet_distance.judges import fill_all_queues
        params = inspect.signature(fill_all_queues).parameters
        self.assertIn("feature_collector", params)
        self.assertIsNone(params["feature_collector"].default)
        self.assertIsNone(params["gmm_judge"].default)


if __name__ == "__main__":
    unittest.main(verbosity=2)
