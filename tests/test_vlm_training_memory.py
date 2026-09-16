"""Check that memory controls preserve FD values and generator derivatives."""
import copy
import unittest

import torch
from torch.utils.checkpoint import checkpoint

from conditional_main_fd_vlm_delta import training_generated_images, training_judge_features
from frechet_distance.judges import extract_judge_features


class FrozenJudge(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.layer = torch.nn.Conv2d(3, 5, 1)
        self.eval().requires_grad_(False)

    def forward(self, images):
        features = self.layer(images).tanh().mean((-2, -1))
        return features, features.square()


class TinyGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Conv2d(3, 3, 1)
        self.embedding = torch.nn.Embedding(3, 3)
        self.forward_sizes = []

    def sample_images_with_grad(self, noise, labels, sampling_args=None):
        self.forward_sizes.append(noise.shape[0])
        return (self.layer(noise) + self.embedding(labels)[:, :, None, None]).tanh()


class TrainingMemoryTest(unittest.TestCase):
    def test_judge_precision_applies_to_queue_fill_and_training(self):
        judge = {"model": FrozenJudge(11).to(dtype=torch.bfloat16),
                 "pool_type": "cls", "amp_dtype": torch.bfloat16}
        images = torch.rand(5, 3, 4, 4, requires_grad=True)
        # Queue fill calls extract_judge_features directly, without an AMP
        # argument or an outer autocast scope. It must share training precision.
        with torch.no_grad():
            queue_features = extract_judge_features(judge, images)
        train_features = training_judge_features(
            judge, images, microbatch=2, checkpoint_features=True)
        self.assertEqual(queue_features.dtype, torch.bfloat16)
        self.assertEqual(train_features.dtype, torch.bfloat16)
        torch.testing.assert_close(train_features, queue_features, rtol=0.02, atol=0.004)
        train_features.double().square().sum().backward()
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertGreater(float(images.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in judge["model"].parameters()))

    def test_bf16_generator_and_judges_preserve_checkpointed_gradients(self):
        torch.manual_seed(31)
        generator = TinyGenerator().to(dtype=torch.bfloat16)
        small_generator = copy.deepcopy(generator)
        compute_dtypes = []
        for model in (generator, small_generator):
            model.layer.register_forward_hook(
                lambda _module, _inputs, output: compute_dtypes.append(output.dtype))
        judge = {"model": FrozenJudge(7).to(dtype=torch.bfloat16), "pool_type": "cls"}
        # Float inputs deliberately exercise autocast at both model boundaries.
        noise = torch.rand(5, 3, 4, 4)
        labels = torch.tensor([0, 1, 2, 1, 0])
        reference = training_generated_images(
            generator, noise, labels, {}, amp_dtype=torch.bfloat16)
        images = training_generated_images(
            small_generator, noise, labels, {}, microbatch=2,
            checkpoint_generator=True, amp_dtype=torch.bfloat16)
        # Model arithmetic is BF16; the returned image leaf is FP32 for VJPs.
        self.assertEqual(reference.dtype, torch.float32)
        self.assertEqual(images.dtype, torch.float32)
        torch.testing.assert_close(images, reference, rtol=0.02, atol=0.004)
        ref_features = training_judge_features(
            judge, reference.float(), amp_dtype=torch.bfloat16)
        features = training_judge_features(
            judge, images.float(), microbatch=2, checkpoint_features=True,
            amp_dtype=torch.bfloat16)
        self.assertEqual(features.dtype, torch.bfloat16)
        torch.testing.assert_close(features, ref_features, rtol=0.02, atol=0.004)
        # FD accumulates statistics above BF16 precision after feature extraction.
        # Keep this batch-coupled loss global to detect accidental per-chunk loss.
        ref_features, features = ref_features.double(), features.double()
        ref_loss = ref_features.mean(0).square().sum() + ref_features.var(0).sum()
        loss = features.mean(0).square().sum() + features.var(0).sum()
        expected = torch.autograd.grad(ref_loss, reference, retain_graph=True)[0]
        actual = torch.autograd.grad(loss, images, retain_graph=True)[0]
        torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.0003)
        ref_loss.backward()
        loss.backward()
        for a, b in zip(generator.parameters(), small_generator.parameters()):
            self.assertEqual(b.grad.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(b.grad).all())
            torch.testing.assert_close(b.grad, a.grad, rtol=0.04, atol=0.0003)
        self.assertGreater(sum(float(p.grad.float().norm())
                               for p in small_generator.parameters()), 0)
        self.assertTrue(all(p.grad is None for p in judge["model"].parameters()))
        self.assertGreater(len(small_generator.forward_sizes), 3)
        self.assertLessEqual(max(small_generator.forward_sizes), 2)
        self.assertTrue(compute_dtypes)
        self.assertEqual(set(compute_dtypes), {torch.bfloat16})

    def test_bf16_adamw_updates_use_bf16_parameters_and_moments(self):
        torch.manual_seed(43)
        generator = TinyGenerator().to(dtype=torch.bfloat16)
        judge = {"model": FrozenJudge(5).to(dtype=torch.bfloat16), "pool_type": "avg"}
        optimizer = torch.optim.AdamW(generator.parameters(), lr=0.02)
        before = [p.detach().clone() for p in generator.parameters()]
        noise = torch.rand(5, 3, 4, 4)
        labels = torch.tensor([0, 1, 2, 1, 0])
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            images = training_generated_images(
                generator, noise, labels, {}, microbatch=2,
                checkpoint_generator=True, amp_dtype=torch.bfloat16)
            features = training_judge_features(
                judge, images, microbatch=2, checkpoint_features=True,
                amp_dtype=torch.bfloat16)
            features.double().mean(0).square().sum().backward()
            optimizer.step()
        self.assertTrue(any(not torch.equal(p, old)
                            for p, old in zip(generator.parameters(), before)))
        for p in generator.parameters():
            self.assertEqual(p.dtype, torch.bfloat16)
            self.assertEqual(p.grad.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(p).all())
            state = optimizer.state[p]
            self.assertEqual(int(state["step"]), 2)
            for key in ("exp_avg", "exp_avg_sq"):
                self.assertEqual(state[key].dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(state[key]).all())

    def test_generator_microbatch_bounds_recompute_and_preserves_global_loss(self):
        generator = TinyGenerator()
        small_generator = copy.deepcopy(generator)
        noise = torch.rand(5, 3, 4, 4)  # The final microbatch has just one image.
        labels = torch.tensor([0, 1, 2, 1, 0])
        reference = training_generated_images(generator, noise, labels, {})
        images = training_generated_images(small_generator, noise, labels, {},
                                           microbatch=2, checkpoint_generator=True)
        torch.testing.assert_close(reference, images)
        judge = {"model": FrozenJudge(3), "pool_type": "cls"}
        ref_feats = training_judge_features(judge, reference)
        feats = training_judge_features(judge, images, microbatch=2, checkpoint_features=True)
        # Batch-coupled loss: splitting into independent losses would be wrong.
        ref_loss = ref_feats.mean(0).square().sum() + ref_feats.var(0).sum()
        loss = feats.mean(0).square().sum() + feats.var(0).sum()
        expected = torch.autograd.grad(ref_loss, reference, retain_graph=True)[0]
        actual = torch.autograd.grad(loss, images, retain_graph=True)[0]
        torch.testing.assert_close(actual, expected)
        ref_loss.backward()
        loss.backward()
        for a, b in zip(generator.parameters(), small_generator.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)
        self.assertEqual(small_generator.forward_sizes[:3], [2, 2, 1])
        self.assertGreater(len(small_generator.forward_sizes), 3)  # Backward recomputed.
        self.assertLessEqual(max(small_generator.forward_sizes), 2)
        with torch.no_grad():
            alternate = training_generated_images(small_generator, noise, labels, {}, microbatch=2)
        torch.testing.assert_close(alternate, reference.detach())

    def test_multiple_judges_microbatch_checkpoint_values_and_vjps(self):
        images = torch.rand(5, 3, 4, 4, requires_grad=True)
        judges = [{"model": FrozenJudge(i), "pool_type": pool}
                  for i, pool in enumerate(("cls", "avg"))]
        ref = [training_judge_features(j, images) for j in judges]
        small = [training_judge_features(j, images, microbatch=2,
                                         checkpoint_features=True) for j in judges]
        for a, b in zip(ref, small):
            torch.testing.assert_close(a, b)
        # Global statistics couple samples after extraction; the final short
        # microbatch must retain its original weight. Repeated backward also
        # exercises retain_graph=True used by the FD/VLM image diagnostics.
        ref_loss = sum(f.mean(0).square().sum() + f.var(0).sum() for f in ref)
        loss = sum(f.mean(0).square().sum() + f.var(0).sum() for f in small)
        expected = torch.autograd.grad(ref_loss, images)[0]
        actual = torch.autograd.grad(loss, images, retain_graph=True)[0]
        torch.testing.assert_close(actual, expected)
        loss.backward()
        torch.testing.assert_close(images.grad, expected)
        self.assertTrue(all(p.grad is None for j in judges for p in j["model"].parameters()))

    def test_generator_checkpoint_without_differentiable_noise(self):
        generator = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.Tanh())
        copy_generator = copy.deepcopy(generator)
        noise = torch.rand(5, 3, 4, 4)
        judge = {"model": FrozenJudge(2), "pool_type": "cls"}
        reference = training_judge_features(judge, generator(noise)).square().sum()
        images = checkpoint(copy_generator, noise, use_reentrant=False)
        loss = training_judge_features(judge, images, microbatch=2,
                                       checkpoint_features=True).square().sum()
        torch.autograd.grad(loss, images, retain_graph=True)
        reference.backward()
        loss.backward()
        for a, b in zip(generator.parameters(), copy_generator.parameters()):
            torch.testing.assert_close(a.grad, b.grad)


if __name__ == "__main__":
    unittest.main()
