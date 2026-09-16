"""CPU checks for copying a p head without changing its learned content."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.relocate_vlm_p_head import relocate_p_head, validate_local_qwen
from vlm_linear_heads import class_ids_hash, load_p_head_checkpoint


class RelocatePHeadTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.revision = "a" * 40
        self.model = self.root / "snapshots" / self.revision
        self.model.mkdir(parents=True)
        for name, value in {
            "config.json": {"model_type": "qwen2_5_vl", "vision_config": {}},
            "preprocessor_config.json": {},
            "tokenizer_config.json": {"chat_template": "test"},
            "tokenizer.json": {},
            "model.safetensors.index.json": {"weight_map": {"weight": "model-1.safetensors"}},
        }.items():
            (self.model / name).write_text(json.dumps(value), encoding="utf-8")
        (self.model / "model-1.safetensors").write_bytes(b"test fixture")
        self.source = self.root / "p_head.pt"
        self.target = self.root / "new" / "p_head.pt"
        self.checkpoint = {
            "format_version": 1, "vlm_backend": "qwen_answer_state",
            "vlm_model_name": f"/old/cache/snapshots/{self.revision}",
            "vlm_pool_type": "answer_state", "vlm_target_size": 252,
            "vlm_input_size": 256, "vlm_layer": 24, "vlm_prompt_sha256": "prompt-hash",
            "feature_dim": 3, "num_classes": 2, "class_ids": [0, 10],
            "class_ids_sha256": class_ids_hash([0, 10]), "feature_norm": "standardize",
            "feature_mean": torch.tensor([1., 2., 3.]), "feature_std": torch.ones(3),
            "head_state_dict": {"weight": torch.arange(6.).reshape(2, 3),
                                "bias": torch.tensor([0.1, 0.2])},
            "temperature": 0.947615, "p_head_sha256": "preserved-stored-hash",
            "train_metadata": {"old_server": "/old/path", "heldout_top1": 0.96},
        }
        torch.save(self.checkpoint, self.source)

    def test_only_model_path_changes_and_source_bytes_survive(self):
        source_bytes = self.source.read_bytes()
        report = relocate_p_head(self.source, self.target, self.model)
        after = load_p_head_checkpoint(self.target)
        self.assertEqual(source_bytes, self.source.read_bytes())
        self.assertEqual(after["vlm_model_name"], str(self.model.resolve()))
        self.assertEqual(report["changed_fields"], ["vlm_model_name"])
        after.pop("vlm_model_name")
        expected = dict(self.checkpoint)
        expected.pop("vlm_model_name")
        for key in ("feature_mean", "feature_std"):
            self.assertTrue(torch.equal(after.pop(key), expected.pop(key)))
        for key, tensor in expected.pop("head_state_dict").items():
            self.assertTrue(torch.equal(after["head_state_dict"][key], tensor))
        after.pop("head_state_dict")
        self.assertEqual(after, expected)

    def test_existing_output_is_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            relocate_p_head(self.source, self.source, self.model)
        self.target.parent.mkdir()
        self.target.write_bytes(b"keep me")
        with self.assertRaises(FileExistsError):
            relocate_p_head(self.source, self.target, self.model)
        self.assertEqual(self.target.read_bytes(), b"keep me")

    def test_different_snapshot_revision_is_rejected(self):
        self.checkpoint["vlm_model_name"] = "/old/snapshots/" + "b" * 40
        torch.save(self.checkpoint, self.source)
        with self.assertRaisesRegex(ValueError, "revisions differ"):
            relocate_p_head(self.source, self.target, self.model)
        self.assertFalse(self.target.exists())

    def test_non_qwen_head_is_rejected(self):
        self.checkpoint["vlm_backend"] = "timm"
        torch.save(self.checkpoint, self.source)
        with self.assertRaisesRegex(ValueError, "qwen_answer_state"):
            relocate_p_head(self.source, self.target, self.model)

    def test_missing_weight_shard_is_rejected(self):
        (self.model / "model-1.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "weight shard"):
            validate_local_qwen(self.model)


if __name__ == "__main__":
    unittest.main()
