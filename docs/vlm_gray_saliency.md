# Gray-image optimization and input gradients

An input gradient is the derivative of a scalar target-class log probability
with respect to the original RGB image tensor `x` in `[0, 1]`:

```
g_p = d log p(c | E_VLM(x)) / dx
g_q = d log q_teacher(c | E_VLM(x)) / dx
g_delta = d [log q_teacher(c | E_VLM(x)) - log p(c | E_VLM(x))] / dx
```

The graph includes differentiable image resizing and normalization, the Qwen
answer-state extractor, and the linear head. The model parameters are frozen;
autograd still differentiates through their operations to the image. The image
is the only optimized variable in this offline test. The maps are local pixel
sensitivities, not attention weights or model-parameter gradients.

The default q is an exact copy of the real-data p head and stays frozen for the
whole diagnostic. Therefore `g_p` and `g_q` should match along the entire
trajectory, while `g_delta` should be zero. This checks initialization; it does
not measure q learning. In the main training loop, q_student instead learns on
detached generated features and updates its EMA teacher. A trained teacher can
be evaluated here with `--q_ckpt path/to/training_checkpoint.pth`. Its p/VLM
identity must agree. Checkpoints with LoRA q are rejected because the adapted
backbone requires a different gradient path.

## Run

From the repository root, using an allocated GPU:

```bash
.venv/bin/python scripts/test_vlm_p_gradient_semantics.py \
  --vlm_p_head work_dirs/vlm_p_head_qwen_answer_c100/p_head.pt \
  --synthetic_init gray --synthetic_class 340 \
  --num_steps 300 --alpha 0.5 \
  --checkpoint_steps 0 1 5 10 20 40 60 100 200 300 \
  --saliency_every 100 --save_every_step \
  --ascent_microbatch 1 --vlm_microbatch 1 \
  --vlm_dtype bf16 --precision_control --delta_mode \
  --out_dir work_dirs/diagnostics/qwen_gray_pq_saliency_s0_a05_n300
```

Class 340 is zebra, present in the fitted 100-class p head. The start is exactly
`0.5` in every pixel, with no noise. The `p` arm performs normalized pixel
ascent on log p; `delta` ascends log p - log q from the same initial canvas.
`p_fp32` repeats p ascent with float32 weights/operations and TF32 disabled.
The BF16 arm honors the training default for TF32 unless
`--vlm_disable_tf32` is set. Precision settings are saved with the metrics.
Neither arm updates a VLM, head, or generator weight.

Use `--synthetic_init gray_noise` to match the previous gray-plus-1/255-noise
test. Use a new output directory for every experiment. For BF16 only, omit
`--precision_control`; FP32 needs more GPU and host memory.

## Read the outputs

- `REPORT.md`, `metrics.csv`: class confidence, rank, held-out ResNet results,
  and pixel changes along the trajectory.
- `steps/<arm>/`: every iterate as a PNG when `--save_every_step` is enabled.
- `saliency/<arm>/*_fixed_scale.png`: current input, mean absolute RGB gradient
  of log p, log q, and log q - log p, all using one scale across times and heads.
- `saliency/<arm>/*_step_scale.png`: the same panels, rescaled jointly within
  each step so weak spatial patterns remain visible. Read the raw norms before
  interpreting a bright map as a strong gradient.
- `saliency/<arm>/*_stepNNNN.npz`: exact floating-point input, signed RGB
  gradients, and log probabilities, saved before the update at that step.
  Unlike a PNG, this preserves sub-1/255 changes.
- `saliency/metrics.csv`: raw p/q/combined gradient norms, p/q cosine, and
  discrepancy between a direct combined backward and subtracting the separate
  gradients. Cosine is undefined when either gradient is zero.

Saliency is measured at step 0, each multiple of `--saliency_every`, and the
final step even if it is not a multiple. It uses an independent autograd graph
and never changes an ascent update. The loss is summed across independent
examples, so raw per-image derivatives do not depend on microbatch size.

If p becomes certain while the image stays visually gray, and the independent
classifier does not recognize the target, that is evidence of a classifier
shortcut in this experiment. Normalized steps can magnify very small gradients
after confidence saturates; fixed-scale maps and raw norms make this visible.
The test alone does not establish whether a trained generator can make useful
images with an additional image prior.

## Verification

```bash
MPLCONFIGDIR=/tmp/vlm-saliency-mpl .venv/bin/python -m unittest \
  tests.test_vlm_input_saliency tests.test_vlm_gradient_verdict -v
```

Checks cover cloned-head cancellation, agreement with finite differences when
q differs, microbatch-independent scaling, tiny-gradient cosine handling, and
raw-array/figure output without model downloads.
Verdict checks also prevent improvements among low-ranked classes from being
reported as semantic generation and identify the expected q=p zero-field control.
