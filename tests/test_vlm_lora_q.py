"""CPU tests for adapter isolation, exact first derivatives, updates and resume."""

import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from qwen_answer_state import QwenAnswerStateExtractor
from vlm_linear_heads import VLMDeltaHeads, VLMLinearHead
from vlm_lora_q import (LoRAQ, build_lora_optimizer, load_q_optimizer_state,
                        q_lora_update_step)


class TinyExtractor(torch.nn.Module):
    def __init__(self, compute_dtype=torch.float32):
        super().__init__()
        self.vlm = torch.nn.Module()
        self.vlm.visual = torch.nn.Module()
        self.vlm.visual.proj = torch.nn.Linear(3, 6)
        self.vlm.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
        for layer in self.vlm.layers:
            layer.q_proj = torch.nn.Linear(6, 6)
        self.vlm.to(dtype=compute_dtype).requires_grad_(False)
        self.layer = 1
        self.compute_dtype = compute_dtype
        self.microbatch_size = 2
        self.max_samples_per_step = 0
        self.register_buffer("_sample_cursor", torch.zeros((), dtype=torch.long))

    select_indices = QwenAnswerStateExtractor.select_indices

    def answer_states(self, images):
        z = self.vlm.visual.proj(images.mean((-2, -1)).to(self.compute_dtype)).tanh()
        z = self.vlm.layers[0].q_proj(z).tanh()
        return {self.layer: z}


def make_system(parameter_dtype=torch.float32, compute_dtype=torch.float32):
    torch.manual_seed(123)
    extractor = TinyExtractor(compute_dtype=compute_dtype)
    heads = VLMDeltaHeads(VLMLinearHead(6, 3), ema_beta=0.6).to(dtype=parameter_dtype)
    lora = LoRAQ(extractor, heads, rank=2, alpha=4, parameter_dtype=parameter_dtype)
    return extractor, heads, lora


def train_args(**kwargs):
    defaults = dict(vlm_q_lr=0.1, vlm_q_lora_lr=0.03,
                    vlm_q_weight_decay=0.0, vlm_q_lora_weight_decay=0.01,
                    vlm_q_optimizer="adamw", vlm_q_momentum=0.9,
                    vlm_q_beta1=0.0, vlm_q_beta2=0.999,
                    vlm_q_loss_scale=1.0, vlm_q_updates_per_step=1,
                    vlm_microbatch=2, vlm_q_grad_clip=1.0)
    return SimpleNamespace(**(defaults | kwargs))


def distributed_worker(rank, store_path, output_dir):
    torch.distributed.init_process_group("gloo", init_method="file://" + store_path,
                                         rank=rank, world_size=2)
    try:
        torch.manual_seed(123)
        extractor = TinyExtractor()
        heads = VLMDeltaHeads(VLMLinearHead(6, 3), ema_beta=0.6)
        torch.manual_seed(100 + rank)  # Must be corrected by the constructor broadcast.
        lora = LoRAQ(extractor, heads, rank=2, alpha=4)
        torch.manual_seed(917)
        images = torch.rand(10, 3, 4, 4)
        labels = torch.arange(10) % 3
        args = train_args()
        q_lora_update_step(lora, build_lora_optimizer(lora, args),
                           images[rank * 5:(rank + 1) * 5], labels[rank * 5:(rank + 1) * 5], args)
        torch.save({"lora": lora.state_dict(), "heads": heads.q_state_dict()},
                   Path(output_dir) / f"rank{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


class LoRAQTest(unittest.TestCase):
    def setUp(self):
        self.extractor, self.heads, self.lora = make_system()
        self.x = torch.rand(5, 3, 4, 4, requires_grad=True)
        self.y = torch.tensor([0, 1, 2, 1, 0])

    def test_init_preserves_logits_and_image_gradient(self):
        p = self.lora.log_probs(self.x, "base", 2)
        for bank in ("student", "teacher"):
            torch.testing.assert_close(self.lora.log_probs(self.x, bank, 2), p, rtol=0, atol=0)
        term, _, _, _ = self.lora.generator_surrogate(self.x, self.y, denominator=5)
        torch.testing.assert_close(term, torch.zeros_like(term), rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(term, self.x)[0], torch.zeros_like(self.x), rtol=0, atol=0)
        self.assertEqual(list(self.lora.layers), ["visual.proj", "layers.0.q_proj"])

    def test_direct_student_updates_without_ema_and_guides_generator(self):
        self.heads.use_ema = False
        teacher_before = self.lora.state_dict()["adapters"]
        head_before = self.heads.q_teacher.weight.detach().clone()
        args = train_args(vlm_q_updates_per_step=5)
        opt = build_lora_optimizer(self.lora, args)
        result = q_lora_update_step(self.lora, opt, self.x, self.y, args, True)
        self.assertEqual(result["q_updates_applied"], 5)
        self.assertEqual(int(self.heads.q_train_steps), 5)
        self.assertTrue(all(int(state["step"]) == 5 for state in opt.state.values()))
        self.assertIsNone(self.x.grad)
        self.assertNotIn("q_teacher_lora_delta_l2", result)
        torch.testing.assert_close(self.heads.q_teacher.weight, head_before, rtol=0, atol=0)
        for name, layer in self.lora.layers.items():
            for key in ("teacher_a", "teacher_b"):
                torch.testing.assert_close(getattr(layer, key), teacher_before[name][key], rtol=0, atol=0)

        # Corrupt the unused teacher: the generator must still read the student.
        with torch.no_grad():
            self.heads.q_teacher.weight.fill_(100)
            for layer in self.lora.layers.values():
                layer.teacher_b.fill_(100)
        direct = (self.heads.q_student_log_probs(self.lora.features(self.x, "student"))
                  - self.heads.p_log_probs(self.lora.features(self.x, "base")))
        direct = direct.gather(1, self.y[:, None]).mean()
        expected_grad = torch.autograd.grad(direct, self.x)[0]
        surrogate, _, lp, _ = self.lora.generator_surrogate(self.x, self.y, denominator=5)
        torch.testing.assert_close(lp, self.lora.log_probs(self.x, "student", 2))
        torch.testing.assert_close(surrogate, direct)
        surrogate.backward()
        torch.testing.assert_close(self.x.grad, expected_grad, atol=1e-7, rtol=1e-4)
        self.assertGreater(float(self.x.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in self.extractor.parameters()))
        self.assertTrue(all(p.grad is None for p in self.heads.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.lora.trainable_parameters()))

    def test_direct_head_preserves_image_gradient_without_q_parameter_gradient(self):
        self.heads.use_ema = False
        z = torch.randn(5, 6, requires_grad=True)
        with torch.no_grad():
            self.heads.q_student.weight.normal_(std=0.2)
        expected = (self.heads.q_student_log_probs(z) - self.heads.p_log_probs(z))
        expected = expected.gather(1, self.y[:, None]).mean()
        expected_grad = torch.autograd.grad(expected, z)[0]
        delta, _, _ = self.heads.delta_log_qp(z, self.y)
        delta.mean().backward()
        torch.testing.assert_close(z.grad, expected_grad)
        self.assertTrue(all(p.grad is None for p in self.heads.parameters()))

    def test_training_defaults_and_both_adamw_optimizers(self):
        from conditional_main_fd_vlm_delta import get_args_parser, _build_q_optimizer
        args = get_args_parser().parse_args([])
        self.assertFalse(args.vlm_q_use_ema)
        self.assertEqual(args.vlm_q_updates_per_step, 1)
        self.assertEqual(args.vlm_q_lr, 1e-4)
        self.assertEqual(args.vlm_q_lora_lr, 1e-5)
        for opt in (_build_q_optimizer(self.heads, args), build_lora_optimizer(self.lora, args)):
            self.assertIsInstance(opt, torch.optim.AdamW)
            self.assertTrue(all(group["betas"] == (0.0, 0.999) for group in opt.param_groups))
        opt = build_lora_optimizer(self.lora, args)
        load_q_optimizer_state(opt, copy.deepcopy(opt.state_dict()))
        for key, value in (("betas", (0.9, 0.999)), ("lr", 1e-3)):
            state = copy.deepcopy(opt.state_dict())
            state["param_groups"][0][key] = value
            with self.assertRaisesRegex(ValueError, key):
                load_q_optimizer_state(opt, state)

    def test_direct_mode_checkpoint_and_legacy_mode_mismatch(self):
        self.heads.use_ema = False
        state = copy.deepcopy(self.heads.q_state_dict())
        self.heads.load_q_state_dict(state)
        legacy = copy.deepcopy(state)
        del legacy["use_ema"]
        with self.assertRaisesRegex(ValueError, "EMA mode"):
            self.heads.load_q_state_dict(legacy)
        _, other, _ = make_system()
        with self.assertRaisesRegex(ValueError, "EMA mode"):
            other.load_q_state_dict(state)

    def test_head_only_scaled_vjp_matches_direct(self):
        # Exercise the original extractor's changed scaling implementation on
        # CPU, since its existing 7B VJP tests require CUDA.
        extractor = TinyExtractor()
        extractor._validate_images = lambda images: None

        def term_fn(z, y):
            return self.heads.p_log_probs(z).gather(1, y[:, None]).sum() / len(self.y)

        expected = term_fn(extractor.answer_states(self.x)[1], self.y)
        expected_grad = torch.autograd.grad(expected, self.x)[0]
        for scale in (1.0, 1024.0):
            term, _, _ = QwenAnswerStateExtractor.vjp_surrogate(
                extractor, self.x, self.y, term_fn, loss_scale=scale)
            torch.testing.assert_close(term, expected)
            torch.testing.assert_close(torch.autograd.grad(term, self.x)[0], expected_grad)

    def test_head_stays_fp32_inside_autocast(self):
        z = torch.randn(4, self.heads.feature_dim)
        expected = self.heads.p_log_probs(z)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = self.heads.p_log_probs(z)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_bf16_head_normalizes_and_multiplies_in_bf16(self):
        head = VLMLinearHead(6, 3).to(dtype=torch.bfloat16)
        z = torch.randn(4, 6, requires_grad=True)
        observed = []
        handle = head.linear.register_forward_hook(
            lambda module, inputs, output: observed.append((inputs[0].dtype, output.dtype)))
        try:
            logits = head.logits(z, temperature=0.9)
        finally:
            handle.remove()
        self.assertEqual(observed, [(torch.bfloat16, torch.bfloat16)])
        self.assertEqual(head.feature_mean.dtype, torch.bfloat16)
        self.assertEqual(head.feature_std.dtype, torch.bfloat16)
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(head.log_probs(z).dtype, torch.float32)
        logits.square().sum().backward()
        self.assertEqual(head.weight.grad.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(float(z.grad.norm()), 0)

    def test_bf16_parameters_optimizer_updates_and_image_vjp(self):
        extractor, heads, lora = make_system(torch.bfloat16, torch.bfloat16)
        heads.use_ema = False
        p_before = lora.log_probs(self.x, "base", 2)
        torch.testing.assert_close(lora.log_probs(self.x, "student", 2), p_before, rtol=0, atol=0)
        initial, _, _, _ = lora.generator_surrogate(self.x, self.y, denominator=5)
        torch.testing.assert_close(initial, torch.zeros_like(initial), rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(initial, self.x)[0],
                                   torch.zeros_like(self.x), rtol=0, atol=0)
        p_state = {key: value.clone() for key, value in heads.p_head.state_dict().items()}
        frozen_base = {key: value.clone() for key, value in extractor.state_dict().items()
                       if ".base." in key}
        head_before = heads.q_student.weight.detach().clone()
        args = train_args(vlm_q_updates_per_step=2, vlm_q_loss_scale=1024.0)
        optimizer = build_lora_optimizer(lora, args)
        metrics = q_lora_update_step(lora, optimizer, self.x, self.y, args, True)
        self.assertEqual(metrics["q_updates_applied"], 2)
        self.assertGreater(metrics["q_student_lora_delta_l2"], 0)
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in lora.trainable_parameters()))
        for state in optimizer.state.values():
            self.assertEqual(state["exp_avg"].dtype, torch.bfloat16)
            self.assertEqual(state["exp_avg_sq"].dtype, torch.bfloat16)
        self.assertFalse(torch.equal(heads.q_student.weight, head_before))
        for layer in lora.layers.values():
            self.assertGreater(float(layer.student_b.float().norm()), 0)
        for key, value in p_state.items():
            torch.testing.assert_close(heads.p_head.state_dict()[key], value, rtol=0, atol=0)
        for key, value in frozen_base.items():
            torch.testing.assert_close(extractor.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(lora.log_probs(self.x, "base", 2), p_before, rtol=0, atol=0)
        self.assertIsNone(self.x.grad)
        surrogate, _, logq, _ = lora.generator_surrogate(
            self.x, self.y, denominator=5, loss_scale=1024.0)
        gradient = torch.autograd.grad(surrogate, self.x)[0]
        self.assertEqual(logq.dtype, torch.float32)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0)
        self.assertTrue(all(p.grad is None for p in extractor.parameters()))
        self.assertTrue(all(p.grad is None for p in heads.parameters()))

    def test_precision_checkpoints_restore_and_reject_mismatch(self):
        _, heads, lora = make_system(parameter_dtype=torch.bfloat16)
        args = train_args()
        optimizer = build_lora_optimizer(lora, args)
        q_lora_update_step(lora, optimizer, self.x, self.y, args)
        saved_lora = lora.state_dict()
        saved_heads = copy.deepcopy(heads.q_state_dict())
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        self.assertEqual(saved_lora["config"]["parameter_dtype"], "torch.bfloat16")
        self.assertEqual(saved_heads["parameter_dtype"], "torch.bfloat16")
        _, restored_heads, restored_lora = make_system(parameter_dtype=torch.bfloat16)
        restored_lora.load_state_dict(saved_lora)
        restored_heads.load_q_state_dict(saved_heads)
        restored_optimizer = build_lora_optimizer(restored_lora, args)
        load_q_optimizer_state(restored_optimizer, saved_optimizer)
        for model, opt in ((lora, optimizer), (restored_lora, restored_optimizer)):
            q_lora_update_step(model, opt, self.x, self.y, args)
        for a, b in zip(lora.trainable_parameters(), restored_lora.trainable_parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.lora.load_state_dict(saved_lora)
        with self.assertRaisesRegex(ValueError, "dtype"):
            self.heads.load_q_state_dict(saved_heads)
        with self.assertRaisesRegex(ValueError, "dtype"):
            load_q_optimizer_state(build_lora_optimizer(self.lora, args), saved_optimizer)
        with self.assertRaisesRegex(ValueError, "configuration"):
            lora.load_state_dict(self.lora.state_dict())
        with self.assertRaisesRegex(ValueError, "dtype"):
            heads.load_q_state_dict(self.heads.q_state_dict())
        # Legacy checkpoints omit parameter_dtype; an explicit FP32 spelling
        # must be equally compatible when both tensors and configuration match.
        self.assertNotIn("parameter_dtype", self.lora.state_dict()["config"])
        self.assertNotIn("parameter_dtype", self.heads.q_state_dict())
        explicit_fp32 = self.lora.state_dict()
        explicit_fp32["config"]["parameter_dtype"] = "torch.float32"
        self.lora.load_state_dict(explicit_fp32)

    def _drift(self):
        with torch.no_grad():
            for layer in self.lora.layers.values():
                layer.teacher_b.normal_(std=0.15)
            self.heads.q_teacher.weight.normal_(std=0.4)

    def test_vjp_matches_direct_with_clamp_subset_and_scale(self):
        self._drift()
        self.extractor.max_samples_per_step = 3
        for clamp in (0.0, 0.1):
            for scale in (1.0, 1024.0):
                self.extractor._sample_cursor.fill_(4)
                surrogate, _, _, indices = self.lora.generator_surrogate(
                    self.x, self.y, denominator=6, clamp=clamp, loss_scale=scale)
                x, y = self.x[indices], self.y[indices]
                zp, zq = self.lora.features(x, "base"), self.lora.features(x, "teacher")
                d = (self.heads.q_teacher_log_probs(zq) - self.heads.p_log_probs(zp))
                d = d.gather(1, y[:, None]).squeeze(1)
                if clamp:
                    self.assertTrue((d.abs() > clamp).any())
                    d = d.clamp(-clamp, clamp)
                direct = d.sum() / 6
                torch.testing.assert_close(surrogate, direct)
                actual = torch.autograd.grad(surrogate, self.x)[0]
                expected = torch.autograd.grad(direct, self.x)[0]
                torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-5)
                self.assertTrue(all(p.grad is None for p in self.extractor.parameters()))
                self.assertTrue(all(p.grad is None for p in self.heads.parameters()))

    def test_student_updates_adapters_and_head_without_changing_p(self):
        p_before = self.lora.log_probs(self.x, "base", 2).clone()
        base_before = {k: v.clone() for k, v in self.extractor.state_dict().items() if ".base." in k}
        head_before = self.heads.q_student.weight.detach().clone()
        args = train_args()
        optimizer = build_lora_optimizer(self.lora, args)
        metrics = q_lora_update_step(self.lora, optimizer, self.x, self.y, args, True)
        self.assertIsNone(self.x.grad)
        torch.testing.assert_close(self.lora.log_probs(self.x, "base", 2), p_before, rtol=0, atol=0)
        for k, v in base_before.items():
            torch.testing.assert_close(self.extractor.state_dict()[k], v, rtol=0, atol=0)
        self.assertFalse(torch.equal(self.heads.q_student.weight, head_before))
        for layer in self.lora.layers.values():
            self.assertGreater(float(layer.student_b.detach().norm()), 0)
            torch.testing.assert_close(layer.teacher_b, layer.student_b * 0.4)
        self.assertGreater(metrics["q_teacher_lora_delta_l2"], 0)
        self.assertEqual(metrics["q_updates_applied"], 1)

    def test_microbatch_scaled_updates_equal_full_batch(self):
        # An uneven final microbatch must still implement one batch mean.
        _, h2, l2 = make_system()
        args1 = train_args(vlm_microbatch=2, vlm_q_loss_scale=512.0)
        args2 = train_args(vlm_microbatch=5)
        o1, o2 = build_lora_optimizer(self.lora, args1), build_lora_optimizer(l2, args2)
        q_lora_update_step(self.lora, o1, self.x, self.y, args1)
        q_lora_update_step(l2, o2, self.x, self.y, args2)
        for a, b in zip(self.lora.trainable_parameters(), l2.trainable_parameters()):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_checkpoint_restores_next_optimizer_and_ema_step(self):
        args = train_args()
        opt = build_lora_optimizer(self.lora, args)
        q_lora_update_step(self.lora, opt, self.x, self.y, args)
        saved_lora = self.lora.state_dict()
        saved_heads = copy.deepcopy(self.heads.q_state_dict())
        saved_opt = copy.deepcopy(opt.state_dict())
        _, heads2, lora2 = make_system()
        lora2.load_state_dict(saved_lora)
        heads2.load_q_state_dict(saved_heads)
        opt2 = build_lora_optimizer(lora2, args)
        opt2.load_state_dict(saved_opt)
        for lora, optimizer in ((self.lora, opt), (lora2, opt2)):
            q_lora_update_step(lora, optimizer, self.x, self.y, args)
        for a, b in zip(self.lora.adapter_parameters(True), lora2.adapter_parameters(True)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(self.heads.q_teacher.weight, heads2.q_teacher.weight, rtol=0, atol=0)
        invalid = copy.deepcopy(saved_lora)
        invalid["config"]["alpha"] = 99
        with self.assertRaisesRegex(ValueError, "configuration"):
            lora2.load_state_dict(invalid)

    def test_selection_is_restored_after_exception(self):
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with self.lora.use("teacher"):
                raise RuntimeError("deliberate")
        self.assertTrue(all(layer.active == "base" for layer in self.lora.layers.values()))

    def test_real_qwen_module_layout_and_backward_on_cpu(self):
        # Tiny random Qwen uses the installed Transformers implementation,
        # without downloading weights or needing a processor/GPU.
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
        config = Qwen2_5_VLConfig(
            text_config=dict(vocab_size=32, hidden_size=24, intermediate_size=48,
                             num_hidden_layers=2, num_attention_heads=3, num_key_value_heads=1,
                             bos_token_id=5, eos_token_id=6,
                             rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 2]}),
            vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                               patch_size=2, temporal_patch_size=1, spatial_merge_size=2,
                               out_hidden_size=24, window_size=4, fullatt_block_indexes=[0]),
            image_token_id=3, video_token_id=4, vision_start_token_id=1, vision_end_token_id=2)
        config._attn_implementation = "eager"
        extractor = TinyExtractor()
        extractor.vlm = Qwen2_5_VLForConditionalGeneration(config).eval().requires_grad_(False)

        def answer_states(images):
            # 4 patches of 2x2 RGB pixels, grouped into one visual token.
            pixels = images.reshape(-1, 3, 2, 2, 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(-1, 12)
            batch = images.shape[0]
            outputs = extractor.vlm(input_ids=torch.tensor([[1, 3, 2, 5]]).repeat(batch, 1),
                                    pixel_values=pixels, image_grid_thw=torch.tensor([[1, 2, 2]]).repeat(batch, 1),
                                    use_cache=False, output_hidden_states=True, logits_to_keep=1)
            return {1: outputs.hidden_states[1][:, -1, :]}

        extractor.answer_states = answer_states
        p_head = VLMLinearHead(24, 3)
        heads = VLMDeltaHeads(p_head, ema_beta=0.6)
        images, labels = self.x[:2], self.y[:2]
        before = heads.p_log_probs(answer_states(images)[1]).detach()
        lora = LoRAQ(extractor, heads, rank=2, alpha=4)
        self.assertEqual(len(lora.layers), 6)  # vision qkv/proj + text q/k/v/o
        torch.testing.assert_close(lora.log_probs(images, "teacher", 2), before, rtol=0, atol=0)
        args = train_args()
        q_lora_update_step(lora, build_lora_optimizer(lora, args), images, labels, args)
        for layer in lora.layers.values():
            self.assertGreater(float(layer.student_b.detach().norm()), 0)
        surrogate, _, _, _ = lora.generator_surrogate(images, labels, denominator=2)
        direct = (heads.q_teacher_log_probs(lora.features(images, "teacher"))
                  - heads.p_log_probs(lora.features(images, "base")))
        direct = direct.gather(1, labels[:, None]).mean()
        torch.testing.assert_close(torch.autograd.grad(surrogate, self.x)[0],
                                   torch.autograd.grad(direct, self.x)[0], atol=1e-7, rtol=1e-4)

    @unittest.skipUnless(torch.distributed.is_gloo_available(), "requires Gloo")
    def test_two_rank_training_matches_single_global_batch(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(distributed_worker, args=(str(Path(tmp) / "store"), tmp),
                     nprocs=2, join=True)
            states = [torch.load(Path(tmp) / f"rank{rank}.pt", weights_only=True) for rank in range(2)]
        torch.manual_seed(123)
        extractor = TinyExtractor()
        heads = VLMDeltaHeads(VLMLinearHead(6, 3), ema_beta=0.6)
        torch.manual_seed(100)
        lora = LoRAQ(extractor, heads, rank=2, alpha=4)
        torch.manual_seed(917)
        images, labels = torch.rand(10, 3, 4, 4), torch.arange(10) % 3
        args = train_args()
        q_lora_update_step(lora, build_lora_optimizer(lora, args), images, labels, args)
        for name, layer_state in lora.state_dict()["adapters"].items():
            for key, value in layer_state.items():
                torch.testing.assert_close(states[0]["lora"]["adapters"][name][key],
                                           states[1]["lora"]["adapters"][name][key], rtol=0, atol=0)
                torch.testing.assert_close(states[0]["lora"]["adapters"][name][key], value, atol=1e-6, rtol=1e-5)
        for bank in ("q_student", "q_teacher"):
            for key, value in heads.q_state_dict()[bank].items():
                torch.testing.assert_close(states[0]["heads"][bank][key], value, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
