"""CPU coverage for precision-safe generator checkpoint continuation."""

from contextlib import nullcontext
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from utils.checkpoint_util import ckpt_resume, save_checkpoint


class CheckpointPrecisionTests(unittest.TestCase):
    def _args(self, directory, parameter_dtype):
        args = SimpleNamespace(
            ckpt_dir=str(directory),
            resume_from=None,
            auto_resume=False,
            load_from=None,
            current_step=4,
            samples_seen=12,
            start_epoch=0,
            steps_per_epoch=10,
            keep_n_ckpts=2,
            milestone_every=0,
            last_elapsed_time=0.0,
        )
        if parameter_dtype is not None:
            args.parameter_dtype = parameter_dtype
        return args

    def _model_and_optimizer(self, dtype):
        model = torch.nn.Linear(3, 2, dtype=dtype)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.25, -0.5, 0.125], [-0.25, 0.5, 0.375]]))
            model.bias.copy_(torch.tensor([0.0625, -0.125]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, betas=(0.9, 0.95))
        return model, optimizer

    def _step(self, model, optimizer):
        dtype = next(model.parameters()).dtype
        inputs = torch.tensor([[1.0, 0.5, -1.0], [-0.5, 1.0, 0.25]], dtype=dtype)
        targets = torch.tensor([[0.0, 0.25], [0.5, -0.25]])
        optimizer.zero_grad(set_to_none=True)
        loss = (model(inputs).float() - targets).square().mean()
        loss.backward()
        optimizer.step()

    def _save(self, args, model, optimizer):
        # Windows may require elevation to create the optional latest.pth alias.
        # Keep serialization, cleanup, resume, and optimizer math real.
        alias_context = (
            patch("utils.checkpoint_util.os.symlink") if os.name == "nt" else nullcontext()
        )
        with alias_context:
            save_checkpoint(args, 3, model, optimizer, None, elapsed_time=12.5)
        return Path(args.ckpt_dir) / "step_0000003.pth"

    def _assert_same_training_state(self, left, left_optimizer, right, right_optimizer):
        for left_param, right_param in zip(left.parameters(), right.parameters()):
            self.assertEqual(left_param.dtype, right_param.dtype)
            torch.testing.assert_close(left_param, right_param, rtol=0, atol=0)
            left_state = left_optimizer.state[left_param]
            right_state = right_optimizer.state[right_param]
            self.assertEqual(left_state.keys(), right_state.keys())
            for key in left_state:
                self.assertEqual(left_state[key].dtype, right_state[key].dtype)
                torch.testing.assert_close(left_state[key], right_state[key], rtol=0, atol=0)

    def _assert_rejected_without_mutation(self, path, directory, target_dtype_name, target_dtype):
        args = self._args(directory, target_dtype_name)
        args.resume_from = str(path)
        args.current_step = 999
        model, optimizer = self._model_and_optimizer(target_dtype)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(-7)
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        with self.assertRaisesRegex(ValueError, "Cannot resume parameter_dtype=.*use --load_from"):
            ckpt_resume(args, model, optimizer, model_ema=None)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        self.assertEqual(len(optimizer.state), 0)
        self.assertEqual(args.current_step, 999)
        self.assertEqual(args.last_elapsed_time, 0.0)

    def test_bf16_roundtrip_preserves_moments_and_next_update(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(directory, "bf16")
            model, optimizer = self._model_and_optimizer(torch.bfloat16)
            self._step(model, optimizer)
            path = self._save(args, model, optimizer)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(saved["parameter_dtype"], "bf16")
            self.assertIsNone(saved["model_ema"])
            self.assertTrue(all(value.dtype == torch.bfloat16 for value in saved["model"].values()))
            for state in saved["optimizer"]["state"].values():
                self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)
                self.assertEqual(state["exp_avg_sq"].dtype, torch.bfloat16)

            resumed_args = self._args(directory, "bf16")
            resumed_args.resume_from = str(path)
            resumed_model, resumed_optimizer = self._model_and_optimizer(torch.bfloat16)
            ckpt_resume(resumed_args, resumed_model, resumed_optimizer, model_ema=None)
            self.assertEqual(resumed_args.current_step, 4)
            self.assertEqual(resumed_args.samples_seen, 12)
            self.assertEqual(resumed_args.last_elapsed_time, 12.5)
            self._assert_same_training_state(model, optimizer, resumed_model, resumed_optimizer)

            self._step(model, optimizer)
            self._step(resumed_model, resumed_optimizer)
            self._assert_same_training_state(model, optimizer, resumed_model, resumed_optimizer)

    def test_legacy_fp32_resume_allowed_only_into_fp32(self):
        with tempfile.TemporaryDirectory() as directory:
            # Old callers omit the precision field entirely.
            args = self._args(directory, None)
            model, optimizer = self._model_and_optimizer(torch.float32)
            self._step(model, optimizer)
            path = self._save(args, model, optimizer)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertNotIn("parameter_dtype", saved)

            resumed_args = self._args(directory, "fp32")
            resumed_args.resume_from = str(path)
            resumed_model, resumed_optimizer = self._model_and_optimizer(torch.float32)
            ckpt_resume(resumed_args, resumed_model, resumed_optimizer, model_ema=None)
            self._assert_same_training_state(model, optimizer, resumed_model, resumed_optimizer)
            self._assert_rejected_without_mutation(path, directory, "bf16", torch.bfloat16)

    def test_bf16_resume_rejected_into_fp32_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(directory, "bf16")
            model, optimizer = self._model_and_optimizer(torch.bfloat16)
            self._step(model, optimizer)
            path = self._save(args, model, optimizer)
            self._assert_rejected_without_mutation(path, directory, "fp32", torch.float32)


if __name__ == "__main__":
    unittest.main()
