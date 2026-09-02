import tempfile
import unittest
from pathlib import Path

import torch

from mae_linear_probe import (
    CHECKPOINT_FORMAT_VERSION,
    MAELinearProbe,
    build_linear_probe_head,
    load_probe_checkpoint,
    select_pooled_features,
)


class TinyBackbone(torch.nn.Module):
    feat_dim = 4
    target_size = 2

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images):
        rgb = images.mean(dim=(2, 3)) * self.scale
        cls = torch.cat((rgb, rgb.mean(dim=1, keepdim=True)), dim=1)
        return cls, cls + 1.0


def make_checkpoint(head_norm="none", pool_type="cls"):
    head = build_linear_probe_head(4, 3, head_norm)
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name": "tiny_mae",
        "pool_type": pool_type,
        "target_size": 2,
        "feature_dim": 4,
        "num_classes": 3,
        "head_norm": head_norm,
        "temperature": 1.0,
        "head_state_dict": head.state_dict(),
        "class_to_idx": {"class_a": 0, "class_b": 1, "class_c": 2},
        "best_val_top1": 0.75,
        "best_val_top5": 1.0,
    }


class MAELinearProbeTest(unittest.TestCase):
    def test_pool_selection(self):
        cls = torch.tensor([[1.0, 2.0]])
        avg = torch.tensor([[3.0, 4.0]])
        self.assertIs(select_pooled_features((cls, avg), "cls"), cls)
        self.assertIs(select_pooled_features((cls, avg), "avg"), avg)
        with self.assertRaises(ValueError):
            select_pooled_features((cls, None), "avg")

    def test_feature_and_image_paths_agree(self):
        probe = MAELinearProbe(TinyBackbone(), make_checkpoint(), device="cpu")
        images = torch.rand(2, 3, 2, 2)
        labels = torch.tensor([0, 2])
        features = probe.features_from_images(images)
        from_features = probe.log_p_c_given_features(features, labels)
        from_images = probe.log_p_c_given_x(images, labels)
        torch.testing.assert_close(from_features, from_images)

    def test_frozen_weights_still_pass_image_gradients(self):
        probe = MAELinearProbe(TinyBackbone(), make_checkpoint(), device="cpu")
        images = torch.rand(2, 3, 2, 2, requires_grad=True)
        labels = torch.tensor([0, 1])
        loss = -probe.log_p_c_given_x(images, labels).mean()
        loss.backward()
        self.assertIsNotNone(images.grad)
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertGreater(float(images.grad.abs().sum()), 0.0)
        for parameter in probe.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)

    def test_checkpoint_roundtrip_and_validation(self):
        checkpoint = make_checkpoint(head_norm="bn")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.pt"
            torch.save(checkpoint, path)
            loaded = load_probe_checkpoint(path)
            probe = MAELinearProbe.from_checkpoint(
                path,
                TinyBackbone(),
                device="cpu",
                expected_model_name="tiny_mae",
                expected_pool_type="cls",
                expected_target_size=2,
                expected_num_classes=3,
            )
            self.assertEqual(loaded["feature_dim"], 4)
            self.assertFalse(probe.training)
            self.assertFalse(probe.head.training)
            with self.assertRaises(ValueError):
                MAELinearProbe.from_checkpoint(
                    path,
                    TinyBackbone(),
                    device="cpu",
                    expected_num_classes=1000,
                )

    def test_rejects_noncontiguous_class_mapping(self):
        checkpoint = make_checkpoint()
        checkpoint["class_to_idx"] = {"class_a": 0, "class_b": 2, "class_c": 4}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.pt"
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                load_probe_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
