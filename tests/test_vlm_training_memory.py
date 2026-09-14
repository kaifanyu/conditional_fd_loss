"""Check that memory controls preserve FD values and generator derivatives."""
import copy
import unittest

import torch
from torch.utils.checkpoint import checkpoint

from conditional_main_fd_vlm_delta import training_generated_images, training_judge_features


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
