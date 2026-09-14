"""Is ``grad_x log p(c | z(x))`` a class direction, or an adversarial one?

OFFLINE diagnostic.  It trains nothing.  It launches no generator run, updates
no model parameter, and writes nothing into any existing run directory.  The
only optimised variable in this file is the image tensor ``x``.

    x_{t+1} = x_t + alpha * normalize(grad_x log p(c | z(x_t)))

``z`` is the Qwen2.5-VL answer state and ``p`` the frozen real-data linear head,
both obtained from ``setup_vlm_delta()`` in ``conditional_main_fd_vlm_delta.py``
-- the same call the training entry point makes, so the prompt, the layer, the
image size, the 3584-d feature, the class ids, the temperature and the frozen
weights are the run's, not a reconstruction.  The p-head checkpoint IS the
definition of z, and every identity check ``setup_vlm_delta`` performs is
inherited here; a mismatch aborts.

The gradient points locally toward increasing p; finite steps need not increase
p monotonically. The question is whether an INDEPENDENT classifier that has
never been in any loss -- torchvision ResNet-50 ``IMAGENET1K_V2``, the repo's
``ProbeClassifier`` canary -- agrees that the image became the requested class.

    p rises to ~1.0, probe stays at chance, pixels barely move
        -> the field is classifier-specific/adversarial, not semantic.
    p rises AND probe recognizes the target
        -> classifier transfer, requiring visual inspection to assess semantics.

Arms
----
``p``      (always)   ascend ``log p(c|z)``            -- the primary test
``delta``  (opt-in)   ascend ``log p(c|z) - log q_teacher(c|z)``
                      == descend the generator-facing ``log q - log p`` field.
                      Uses --q_ckpt (or --gen_ckpt); otherwise q=p is a zero-field control.
``fp32``   (opt-in)   the p arm again with an fp32 Qwen, plus the per-image
                      ``cos(g_bf16, g_fp32)`` at x0 (docs/vlm_delta.md Sec.3.3
                      measured 0.03 on real images -- this re-measures it on
                      the exact images the diagnostic uses).

Input modes
-----------
MODE A ``--gen_ckpt``  : sample N images from a frozen generator checkpoint at
                         fixed per-image seeds; the sampled labels are the
                         targets c.
MODE B ``--images``    : fixed external image files with ``--image_classes``
                         giving one global ImageNet id per file.
MODE C ``--synthetic_init`` : one deterministic gray/noise canvas with
                              ``--synthetic_class`` as its target.

Usage
-----
    # MODE A -- the Qwen run's own final checkpoint
    CUDA_VISIBLE_DEVICES=1 python scripts/test_vlm_p_gradient_semantics.py \
        --vlm_p_head work_dirs/vlm_p_head_qwen_answer_c100/p_head.pt \
        --run_dir  work_dirs/JiT_uncond_vlm_delta/qwen_CAL2_w0005 \
        --gen_ckpt work_dirs/JiT_uncond_vlm_delta/qwen_CAL2_w0005/checkpoints/latest.pth \
        --num_test_images 8 --num_steps 60 --alpha 0.1 \
        --out_dir work_dirs/diagnostics/qwen_p_grad_semantics

    # MODE B -- fixed external images
    CUDA_VISIBLE_DEVICES=1 python scripts/test_vlm_p_gradient_semantics.py \
        --vlm_p_head work_dirs/vlm_p_head_qwen_answer_c100/p_head.pt \
        --images a.png b.png --image_classes 207 360 \
        --out_dir work_dirs/diagnostics/external

    # MODE C -- start from one nearly gray canvas and save every iterate
    CUDA_VISIBLE_DEVICES=1 python scripts/test_vlm_p_gradient_semantics.py \
        --vlm_p_head work_dirs/vlm_p_head_qwen_answer_c100/p_head.pt \
        --synthetic_init gray_noise --synthetic_class 340 \
        --num_steps 200 --alpha 0.5 --save_every_step \
        --out_dir work_dirs/diagnostics/qwen_p_gray_zebra
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import conditional_main_fd_vlm_delta as vlm_delta_main
from classifier_ensemble import ProbeClassifier
from generate_samples import (
    ARCH_KEYS,
    SAMPLING_KEYS,
    _cli_overrides,
    _find_args_json,
    _imagenet_class_names,
    _load_font,
    load_checkpoint as load_generator_checkpoint,
)
from vlm_linear_heads import (
    P_HEAD_BACKEND_QWEN,
    load_p_head_checkpoint,
    p_head_backend,
    p_head_identity,
)
from scripts.vlm_input_saliency import InputGradientLogger

logger = logging.getLogger("p_grad_semantics")

DEFAULT_CHECKPOINT_STEPS = (0, 1, 5, 10, 20, 40, 60)
EPS = 1e-12


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def get_parser():
    """Built on the training parser, so every default that defines ``z`` is the
    training default and nothing about the representation can drift silently."""
    parser = vlm_delta_main.get_args_parser()
    parser.add_argument("-h", "--help", action="help", help="show this message and exit")

    group = parser.add_argument_group("p-gradient semantics diagnostic")
    group.add_argument("--out_dir", type=str, required=True,
                       help="output directory; must not be an existing training run dir")
    group.add_argument("--run_dir", type=str, default=None,
                       help="training run dir (or its args_*.json) whose architecture, "
                            "sampling config and train_class_ids are reused. Explicit "
                            "CLI flags still win.")

    # MODE A
    group.add_argument("--gen_ckpt", type=str, default=None,
                       help="MODE A: frozen generator checkpoint to sample x0 from")
    group.add_argument("--use_ema", action="store_true", help="MODE A: load EMA weights")
    group.add_argument("--ema_idx", type=int, default=0)
    group.add_argument("--ema_label", type=str, default=None)
    # NOT --num_images: the training parser already owns that for FID.
    group.add_argument("--num_test_images", type=int, default=8,
                       help="MODE A: how many images to sample and ascend")
    group.add_argument("--gen_seed", type=int, default=1234,
                       help="MODE A: base seed; image i uses gen_seed + 10007*i")
    group.add_argument("--classes", type=int, nargs="+", default=None,
                       help="MODE A: force these global class ids as the targets "
                            "(cycled to --num_test_images). Default: sampled uniformly "
                            "from the run's train_class_ids at --gen_seed.")

    # MODE B
    group.add_argument("--images", type=str, nargs="+", default=None,
                       help="MODE B: image files to use as x0 (generator never loaded)")
    group.add_argument("--image_classes", type=int, nargs="+", default=None,
                       help="MODE B: one global ImageNet class id per --images entry")

    # MODE C
    group.add_argument("--synthetic_init",
                       choices=("gray", "gray_noise", "uniform_noise"), default=None,
                       help="MODE C: optimize one synthetic canvas: constant 0.5 gray, "
                            "gray plus small Gaussian noise, or U[0,1] noise")
    group.add_argument("--synthetic_class", type=int, default=None,
                       help="MODE C: target global ImageNet class id")
    group.add_argument("--synthetic_noise_std", type=float, default=1.0 / 255.0,
                       help="MODE C gray_noise: Gaussian standard deviation in [0,1] "
                            "pixel units (default: 1/255)")

    # the ascent
    group.add_argument("--num_steps", type=int, default=60)
    group.add_argument("--alpha", type=float, default=0.1,
                       help="step size. In normalized mode this is EXACTLY the L2 "
                            "norm of the per-image pixel update per step, so the "
                            "per-step RMS drift is alpha*255/sqrt(3*H*W) /255 -- at "
                            "the default 0.1 and 256px that is 0.058/255 per step and "
                            "at most 3.5/255 after 60 steps, the scale the earlier "
                            "SigLIP measurement landed at.")
    group.add_argument("--grad_mode", choices=("normalized", "raw"), default="normalized",
                       help="normalized: x += alpha * g/||g||_2 per image (default, so "
                            "alpha has a units-free interpretation). raw: x += alpha * g.")
    group.add_argument("--objective_reduction", choices=("sum", "mean"), default="sum",
                       help="reduction over the microbatch before autograd. The two "
                            "differ only by the constant 1/B, which normalized mode "
                            "removes exactly; sum keeps raw mode microbatch-independent.")
    group.add_argument("--checkpoint_steps", type=int, nargs="+",
                       default=list(DEFAULT_CHECKPOINT_STEPS))
    group.add_argument("--ascent_microbatch", type=int, default=0,
                       help="images per Qwen forward/backward during ascent "
                            "(0 = --vlm_microbatch)")

    # arms
    group.add_argument("--delta_mode", action="store_true",
                       help="also run the full-delta arm: ascend log p - log q_teacher, "
                            "using --q_ckpt or --gen_ckpt's vlm_delta_state; otherwise "
                            "use the initial frozen q=p copy as a zero-gradient control")
    group.add_argument("--q_ckpt", type=str, default=None,
                       help="load frozen q_teacher from this training checkpoint's "
                            "vlm_delta_state, independently of the image input mode; "
                            "default for synthetic/external images is initial q=p")
    group.add_argument("--precision_control", action="store_true",
                       help="after the main arms, reload Qwen in fp32 (TF32 disabled) and report "
                            "cos(g_bf16, g_fp32) at x0 plus a full fp32 p-arm")

    # outputs
    group.add_argument("--amplify", type=int, nargs="+", default=[10, 20],
                       help="amplification factors for the perturbation visualisations")
    group.add_argument("--save_every_step", action="store_true",
                       help="save a PNG at every ascent step, not just checkpoint steps")
    group.add_argument("--saliency_every", type=int, default=0,
                       help="save raw input gradients of log p, log q, and log q-log p "
                            "plus saliency maps at step 0, every N steps and the final "
                            "step (0 disables; recommended 100)")
    group.add_argument("--no_heatmap", action="store_true")
    group.add_argument("--diag_seed", type=int, default=0)
    return parser


def apply_run_dir(args, argv):
    """Overlay a run's arch/sampling config, exactly as generate_samples.py does."""
    json_path = _find_args_json(args.run_dir)
    with open(json_path) as f:
        cfg = json.load(f)
    given = _cli_overrides(argv)
    applied = []
    # The VLM keys matter as much as the arch keys here: they are what makes the
    # extractor bit-identical to the run's.
    keys = ARCH_KEYS + SAMPLING_KEYS + (
        "vlm_p_head", "vlm_judge", "vlm_head_temperature", "vlm_q_ema_beta",
        "vlm_dtype", "vlm_attn_implementation", "vlm_microbatch",
        "vlm_q_buffer_size", "vlm_q_lr", "vlm_q_optimizer", "vlm_q_momentum",
        "vlm_q_weight_decay", "vlm_q_batch_size", "vlm_q_updates_per_step",
        "vlm_delta_weight", "vlm_delta_clamp", "num_classes", "seed",
    )
    for key in keys:
        if key not in cfg or key in given:
            continue
        if getattr(args, key, None) != cfg[key]:
            applied.append(f"{key}={cfg[key]}")
        setattr(args, key, cfg[key])
    logger.info("[cfg] loaded run config from %s", json_path)
    if applied:
        logger.info("[cfg] overrides from run config: %s", ", ".join(applied))
    return json_path, cfg.get("train_class_ids")


def resolve_args(argv):
    args = get_parser().parse_args(argv)

    run_class_ids = None
    if args.run_dir:
        _, run_class_ids = apply_run_dir(args, argv)

    if not args.vlm_p_head:
        raise SystemExit("--vlm_p_head is required (see train_vlm_p_head.py)")
    if args.vlm_q_lora:
        raise SystemExit("this diagnostic supports linear q heads only, not --vlm_q_lora")
    input_modes = sum(x is not None for x in
                      (args.gen_ckpt, args.images, args.synthetic_init))
    if input_modes != 1:
        raise SystemExit("give exactly one of --gen_ckpt (MODE A), --images "
                         "(MODE B), or --synthetic_init (MODE C)")
    if args.images and not args.image_classes:
        raise SystemExit("MODE B requires --image_classes, one global class id per image")
    if args.images and len(args.images) != len(args.image_classes):
        raise SystemExit(
            f"--images has {len(args.images)} entries but --image_classes has "
            f"{len(args.image_classes)}")
    if args.synthetic_init and args.synthetic_class is None:
        raise SystemExit("MODE C requires --synthetic_class")
    if not math.isfinite(args.synthetic_noise_std) or args.synthetic_noise_std < 0:
        raise SystemExit("--synthetic_noise_std must be finite and non-negative")
    if args.num_steps < 1 or args.saliency_every < 0:
        raise SystemExit("--num_steps must be positive and --saliency_every non-negative")
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        raise SystemExit("--alpha must be finite and positive")

    # The head's own class list is the authority; the run dir only has to agree.
    ckpt_class_ids = [int(c) for c in load_p_head_checkpoint(args.vlm_p_head)["class_ids"]]
    if args.train_class_ids is None:
        args.train_class_ids = list(run_class_ids) if run_class_ids else list(ckpt_class_ids)
    if sorted(int(c) for c in args.train_class_ids) != sorted(ckpt_class_ids):
        raise SystemExit(
            "the class set does not match the p head: run/CLI has "
            f"{len(args.train_class_ids)} ids, the head was fitted on "
            f"{len(ckpt_class_ids)}. The softmax denominator must be exactly the "
            "head's class set.")

    args.checkpoint_steps = sorted({int(s) for s in args.checkpoint_steps
                                    if 0 <= int(s) <= args.num_steps})
    if not args.checkpoint_steps:
        raise SystemExit("--checkpoint_steps is empty after clamping to [0, num_steps]")
    if args.checkpoint_steps[0] != 0:
        args.checkpoint_steps.insert(0, 0)
    args.checkpoint_steps = sorted(set(args.checkpoint_steps + [args.num_steps]))
    if args.saliency_every:
        args.checkpoint_steps = sorted(set(args.checkpoint_steps).union(
            range(0, args.num_steps + 1, args.saliency_every)))

    # Fields the training entry point sets in setup(); nothing here is distributed.
    args.world_size, args.rank, args.local_rank = 1, 0, 0
    args.global_bsz = args.batch_size
    args.total_steps = max(1, getattr(args, "total_steps", 1))
    args.vlm_samples_per_step = 0          # score every image handed over
    args.vlm_q_bootstrap_updates = 0       # q is never trained here
    if args.ascent_microbatch <= 0:
        args.ascent_microbatch = args.vlm_microbatch

    out = Path(args.out_dir)
    if (out / "checkpoints").exists() or list(out.glob("args_*.json")):
        raise SystemExit(
            f"--out_dir {out} looks like an existing training run directory; "
            "refusing to write diagnostics into it")
    return args


# ---------------------------------------------------------------------------
# Models -- all frozen
# ---------------------------------------------------------------------------

def build_vlm(args):
    """p head + Qwen extractor, through the training entry point's own setup.

    ``setup_vlm_delta`` is what enforces the checkpoint identity: model path,
    prompt sha, layer, feature dim, input size, class ids and temperature. It is
    called with an empty judge list because the answer-state backend is not an FD
    judge; a timm-backend head would need real judges and is rejected above.
    """
    ckpt = load_p_head_checkpoint(args.vlm_p_head)
    backend = p_head_backend(ckpt)
    if backend != P_HEAD_BACKEND_QWEN:
        raise SystemExit(
            f"this diagnostic is for the Qwen answer-state backend; {args.vlm_p_head} "
            f"declares vlm_backend={backend!r}. Point it at a qwen_answer_state head.")
    vlm_judge = vlm_delta_main.setup_vlm_delta([], [], args)
    heads = vlm_judge["vlm_heads"]
    heads.eval().requires_grad_(False)
    extractor = vlm_judge["vlm_extractor"]
    extractor.eval().requires_grad_(False)

    live = extractor.identity()
    stored = p_head_identity(ckpt)
    mismatch = {k: (stored.get(k), live.get(k)) for k in live if k in stored
                and stored[k] != live[k]}
    if mismatch:  # setup_vlm_delta already checks this; belt and braces
        raise SystemExit(f"extractor/p-head identity mismatch: {mismatch}")

    identity = dict(stored)
    identity.update({
        "vlm_target_size": int(extractor.target_size),
        "num_visual_tokens": int(extractor.num_visual_tokens),
        "vlm_rendered_prompt": extractor.rendered_prompt,
        "vlm_prompt": extractor.prompt,
        "vlm_dtype": str(args.vlm_dtype),
        "temperature": float(heads.temperature),
        "p_head_path": os.path.abspath(args.vlm_p_head),
        "p_head_val_top1": (ckpt.get("train_metadata", {}) or {}).get("best", {}).get("top1"),
    })
    return vlm_judge, identity


def load_q_teacher(vlm_judge, gen_ckpt_path):
    """Install q_teacher from a generator checkpoint's ``vlm_delta_state``.

    Mirrors the resume path in ``train_and_evaluate`` including its identity
    refusal, so the delta arm cannot silently be run against a q fitted on a
    different representation.
    """
    blob = torch.load(gen_ckpt_path, map_location="cpu", weights_only=False)
    state = blob.get("vlm_delta_state")
    if state is None:
        raise SystemExit(
            f"--delta_mode: {gen_ckpt_path} has no vlm_delta_state, so there is no "
            "q_teacher to read. Run the p arm alone.")
    if state.get("q_lora") is not None:
        raise SystemExit("this checkpoint uses LoRA q; loading only its linear head "
                         "would give the wrong q input gradient")
    saved_identity = state.get("p_identity", {})
    live_identity = vlm_judge["vlm_p_identity"]
    if saved_identity and saved_identity != live_identity:
        diffs = {k: (saved_identity.get(k), live_identity.get(k))
                 for k in live_identity if saved_identity.get(k) != live_identity.get(k)}
        raise SystemExit(
            "--delta_mode: the generator checkpoint's q was trained against a "
            f"different p head / VLM. Differences (saved, current): {diffs}")
    heads = vlm_judge["vlm_heads"]
    heads.load_q_state_dict(state["q"])
    heads.cuda().eval().requires_grad_(False)
    q_steps = int(heads.q_train_steps.item())
    w_teacher = heads.q_teacher.weight.detach()
    info = {
        "q_train_steps": q_steps,
        "q_teacher_weight_norm": float(w_teacher.norm()),
        "p_weight_norm": float(heads.p_weight_ref.norm()),
        "q_teacher_weight_rel_to_p": float(w_teacher.norm() / heads.p_weight_ref.norm()),
        "q_teacher_cos_to_p": float(F.cosine_similarity(
            w_teacher.flatten(), heads.p_weight_ref.flatten(), dim=0)),
        "weight_schedule": state.get("weight_schedule"),
    }
    logger.info("[delta] q_teacher restored after %d q steps: ||W_q||/||W_p|| = %.4f, "
                "cos(W_q, W_p) = %.4f  (both near 0 => q is uniform and the field is "
                "essentially -grad log p)",
                q_steps, info["q_teacher_weight_rel_to_p"], info["q_teacher_cos_to_p"])
    del blob
    return info


# ---------------------------------------------------------------------------
# x0
# ---------------------------------------------------------------------------

@torch.no_grad()
def mode_a_images(args):
    """Sample x0 from a frozen generator at fixed per-image seeds."""
    from utils.builders import create_generation_model, create_tokenizer

    tokenizer = create_tokenizer(args)
    model, _ = create_generation_model(args)
    model = model.cuda().eval().requires_grad_(False)
    if tokenizer is not None:
        tokenizer = tokenizer.cuda().eval().requires_grad_(False)
    weights_desc = load_generator_checkpoint(
        model, args.gen_ckpt, use_ema=args.use_ema,
        ema_idx=args.ema_idx, ema_label=args.ema_label)

    ids = [int(c) for c in args.train_class_ids]
    if args.classes:
        targets = [int(args.classes[i % len(args.classes)]) for i in range(args.num_test_images)]
        unknown = sorted({c for c in targets if c not in set(ids)})
        if unknown:
            raise SystemExit(f"--classes {unknown} are outside the p head's class set")
        label_source = f"--classes {args.classes}"
    else:
        g = torch.Generator().manual_seed(args.gen_seed)
        picks = torch.randint(0, len(ids), (args.num_test_images,), generator=g)
        targets = [ids[int(i)] for i in picks]
        label_source = f"uniform over {len(ids)} train_class_ids at seed {args.gen_seed}"

    shape = (model.in_channels, model.input_size, model.input_size)
    sampling_args = {"t_min": args.interval_min, "t_max": args.interval_max,
                     "cfg": args.cfg, "num_steps": args.num_sampling_steps}

    images, seeds = [], []
    for i, c in enumerate(targets):
        seed = args.gen_seed + 10007 * i
        seeds.append(seed)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        z = torch.randn(1, *shape, device="cuda", generator=gen) * args.noise_scale
        y = torch.full((1,), c, dtype=torch.long, device="cuda")
        out = model.sample_images_with_grad(z, y, sampling_args=sampling_args)
        if tokenizer is not None:
            out = tokenizer.decode(tokenizer.denormalize_z(out))
        images.append((out * 0.5 + 0.5).clamp(0, 1).float())
    x0 = torch.cat(images, dim=0)

    meta = {
        "mode": "A_generated",
        "gen_ckpt": os.path.abspath(args.gen_ckpt),
        "weights": weights_desc,
        "gen_seed": args.gen_seed,
        "per_image_seeds": seeds,
        "label_source": label_source,
        "sampling_args": sampling_args,
        "model": args.model,
        "img_size": args.img_size,
        "noise_scale": args.noise_scale,
    }
    del model, tokenizer
    torch.cuda.empty_cache()
    return x0, torch.tensor(targets, dtype=torch.long, device="cuda"), meta


def mode_b_images(args, size):
    """Load fixed external images, deterministically resized to the head's size."""
    tensors, records = [], []
    for path in args.images:
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            side = min(w, h)
            im = im.crop(((w - side) // 2, (h - side) // 2,
                          (w - side) // 2 + side, (h - side) // 2 + side))
            arr = torch.from_numpy(np.asarray(im, dtype=np.uint8)).permute(2, 0, 1)
        t = arr.float().div(255.0).unsqueeze(0).cuda()
        if t.shape[-1] != size or t.shape[-2] != size:
            t = F.interpolate(t, size=(size, size), mode="bicubic",
                              align_corners=False, antialias=True).clamp(0, 1)
        tensors.append(t)
        records.append({"path": os.path.abspath(path), "original_size": [w, h]})
    meta = {
        "mode": "B_external",
        "images": records,
        "preprocessing": f"center square crop -> bicubic antialias resize to {size} -> [0,1]",
    }
    return (torch.cat(tensors, 0),
            torch.tensor([int(c) for c in args.image_classes], dtype=torch.long,
                         device="cuda"),
            meta)


@torch.no_grad()
def mode_c_image(args, size):
    """Create one deterministic synthetic canvas directly on the VLM device."""
    shape = (1, 3, int(size), int(size))
    generator = torch.Generator(device="cuda").manual_seed(int(args.diag_seed))
    if args.synthetic_init == "gray":
        x0 = torch.full(shape, 0.5, dtype=torch.float32, device="cuda")
    elif args.synthetic_init == "gray_noise":
        x0 = torch.full(shape, 0.5, dtype=torch.float32, device="cuda")
        noise = torch.randn(shape, dtype=torch.float32, device="cuda",
                            generator=generator)
        x0 = (x0 + float(args.synthetic_noise_std) * noise).clamp(0, 1)
    elif args.synthetic_init == "uniform_noise":
        x0 = torch.rand(shape, dtype=torch.float32, device="cuda",
                        generator=generator)
    else:  # resolve_args guarantees one of the parser choices.
        raise AssertionError(f"unknown synthetic init {args.synthetic_init!r}")
    meta = {
        "mode": "C_synthetic",
        "initialization": args.synthetic_init,
        "target_global_class": int(args.synthetic_class),
        "seed": int(args.diag_seed),
        "gray_noise_std": (float(args.synthetic_noise_std)
                           if args.synthetic_init == "gray_noise" else None),
        "size": int(size),
    }
    targets = torch.tensor([int(args.synthetic_class)], dtype=torch.long,
                           device="cuda")
    return x0, targets, meta


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def head_per_sample(log_probs, targets_local):
    """Per-sample version of ``vlm_linear_heads.head_metrics`` (same formulas)."""
    idx = targets_local.long().view(-1, 1)
    num_classes = log_probs.shape[-1]
    target_logp = log_probs.gather(1, idx).squeeze(1)
    probs = log_probs.exp()
    target_prob = probs.gather(1, idx).squeeze(1)
    pred = log_probs.argmax(-1)
    rank = (log_probs > target_logp.view(-1, 1)).sum(-1) + 1
    best_other = log_probs.scatter(1, idx, float("-inf")).max(dim=-1).values
    entropy = -(probs * log_probs).sum(-1)
    k5 = min(5, num_classes)
    top5 = log_probs.topk(k5, dim=-1).indices.eq(idx).any(-1)
    return {
        "target_logp": target_logp, "target_prob": target_prob,
        "top1_correct": (pred == targets_local).float(), "pred": pred,
        "target_rank": rank.float(), "entropy": entropy,
        "margin": target_logp - best_other, "top5_correct": top5.float(),
    }


@torch.no_grad()
def probe_per_sample(probe, images01, targets_global):
    """Held-out ResNet-50, with ``ProbeClassifier.stats``' exact preprocessing.

    Copied rather than called because ``stats`` returns batch aggregates only;
    the two must never drift apart (same copy as ``probe_per_sample_correct``).
    """
    x = F.interpolate(images01.float(), size=(224, 224), mode="bicubic",
                      align_corners=False, antialias=True)
    logits = probe.model((x - probe.mean) / probe.std)
    log_probs = torch.log_softmax(logits, dim=-1)
    idx = targets_global.long().view(-1, 1)
    target_logp = log_probs.gather(1, idx).squeeze(1)
    rank = (logits > logits.gather(1, idx)).sum(-1) + 1
    return {
        "probe_target_logp": target_logp,
        "probe_target_prob": target_logp.exp(),
        "probe_top1_correct": (logits.argmax(-1) == targets_global).float(),
        "probe_pred": logits.argmax(-1),
        "probe_target_rank": rank.float(),
        "probe_top5_correct": logits.topk(5, dim=-1).indices.eq(idx).any(-1).float(),
    }


def perturbation_per_sample(x, x0):
    d = (x - x0).flatten(1)
    rms01 = d.pow(2).mean(dim=1).sqrt()
    return {
        "rel_l2": d.norm(dim=1) / x0.flatten(1).norm(dim=1).clamp_min(EPS),
        "abs_l2": d.norm(dim=1),
        "rms01": rms01,
        "rms255": 255.0 * rms01,
        "max_abs_change": d.abs().amax(dim=1),
        "mean_abs_change": d.abs().mean(dim=1),
    }


# ---------------------------------------------------------------------------
# The ascent
# ---------------------------------------------------------------------------

def _objective(heads, z, targets_local, arm):
    """Per-sample scalar being ascended."""
    logp_all = heads.p_log_probs(z)
    idx = targets_local.long().view(-1, 1)
    logp_c = logp_all.gather(1, idx).squeeze(1)
    if arm == "p":
        return logp_c, logp_all, None
    logq_all = heads.q_teacher_log_probs(z)
    logq_c = logq_all.gather(1, idx).squeeze(1)
    # ascending (log p - log q) == descending the generator's (log q - log p)
    return logp_c - logq_c, logp_all, logq_all


def _reduce(v, how):
    return v.sum() if how == "sum" else v.mean()


def ascend_group(x0, targets_global, targets_local, *, heads, extractor, probe,
                 args, arm, want_cross_cosine, saliency_logger=None,
                 output_arm=None, image_offset=0):
    """Full ascent for one microbatch-sized group.  Returns per-step records.

    Images are independent under this objective, so a group is exactly a batch
    of independent single-image experiments; only the Qwen forward is shared.
    """
    checkpoint_steps = set(args.checkpoint_steps)
    x = x0.clone().detach().requires_grad_(True)
    per_step = {}
    saved_images = {}
    grad_at_x0 = None

    for t in range(args.num_steps + 1):
        if saliency_logger is not None and (t % saliency_logger.every == 0
                                             or t == args.num_steps):
            saliency_logger.record(x, targets_local, heads, extractor, t,
                                   output_arm or arm, image_offset, str(args.vlm_dtype))
            logger.info("[%s] saved input gradients at step %d", output_arm or arm, t)
        need_grad = t < args.num_steps
        record = t in checkpoint_steps
        cross = want_cross_cosine and record and arm == "delta"

        if need_grad:
            z = extractor.answer_states(x)[extractor.layer]
            obj, logp_all, logq_all = _objective(heads, z, targets_local, arm)
            g = torch.autograd.grad(_reduce(obj, args.objective_reduction), x,
                                    retain_graph=cross)[0].detach()
            g_cross = None
            if cross:
                logp_c = logp_all.gather(1, targets_local.long().view(-1, 1)).squeeze(1)
                g_cross = torch.autograd.grad(
                    _reduce(logp_c, args.objective_reduction), x)[0].detach()
            logp_all, logq_all = logp_all.detach(), (
                None if logq_all is None else logq_all.detach())
        else:  # logging-only final step: forward is all that is needed
            # x.detach() is required, not cosmetic: _prepare_inputs refuses an
            # input that requires grad whose pixel_values do not, which is
            # exactly what no_grad would produce.
            with torch.no_grad():
                z = extractor.answer_states(x.detach())[extractor.layer]
                _, logp_all, logq_all = _objective(heads, z, targets_local, arm)
            g, g_cross = None, None

        if t == 0 and need_grad:
            grad_at_x0 = g.clone()

        if record:
            xd = x.detach()
            row = {k: v.detach() for k, v in head_per_sample(logp_all, targets_local).items()}
            row.update(probe_per_sample(probe, xd, targets_global))
            row.update(perturbation_per_sample(xd, x0))
            if logq_all is not None:
                idx = targets_local.long().view(-1, 1)
                logq_c = logq_all.gather(1, idx).squeeze(1)
                row["q_teacher_target_logp"] = logq_c
                row["delta_logqp"] = logq_c - row["target_logp"]
            row["grad_norm"] = (g.flatten(1).norm(dim=1) if g is not None
                                else torch.full((xd.shape[0],), float("nan"),
                                                device=xd.device))
            if g_cross is not None:
                row["cos_g_delta_g_p"] = F.cosine_similarity(
                    g.flatten(1), g_cross.flatten(1), dim=1)
            per_step[t] = {k: v.detach().float().cpu() for k, v in row.items()}
            saved_images[t] = xd.clone().cpu()
        elif args.save_every_step:
            saved_images[t] = x.detach().clone().cpu()

        if need_grad:
            with torch.no_grad():
                if args.grad_mode == "normalized":
                    step = g / g.flatten(1).norm(dim=1).clamp_min(EPS).view(-1, 1, 1, 1)
                else:
                    step = g
                x = (x + args.alpha * step).clamp(0, 1).detach().requires_grad_(True)

    return per_step, saved_images, grad_at_x0


def run_arm(x0, targets_global, targets_local, *, vlm_judge, probe, args, arm,
            want_cross_cosine=False, saliency_logger=None, output_arm=None):
    heads = vlm_judge["vlm_heads"]
    extractor = vlm_judge["vlm_extractor"]
    step = max(1, int(args.ascent_microbatch))
    n = x0.shape[0]
    records = {t: {} for t in args.checkpoint_steps}
    images = {}
    grads0 = []
    t_start = time.perf_counter()
    for start in range(0, n, step):
        sl = slice(start, min(start + step, n))
        logger.info("[%s] ascending images %d-%d of %d (%d steps, alpha=%g, %s)",
                    arm, start, sl.stop - 1, n, args.num_steps, args.alpha,
                    args.grad_mode)
        per_step, saved, g0 = ascend_group(
            x0[sl], targets_global[sl], targets_local[sl], heads=heads,
            extractor=extractor, probe=probe, args=args, arm=arm,
            want_cross_cosine=want_cross_cosine, saliency_logger=saliency_logger,
            output_arm=output_arm, image_offset=start)
        for t, row in per_step.items():
            for k, v in row.items():
                records[t].setdefault(k, []).append(v)
        for t, img in saved.items():
            images.setdefault(t, []).append(img)
        if g0 is not None:
            grads0.append(g0.cpu())
        torch.cuda.empty_cache()
    out = {t: {k: torch.cat(v) for k, v in row.items()} for t, row in records.items()}
    imgs = {t: torch.cat(v) for t, v in images.items()}
    logger.info("[%s] done in %.1f s", arm, time.perf_counter() - t_start)
    return out, imgs, (torch.cat(grads0) if grads0 else None)


# ---------------------------------------------------------------------------
# Visual output
# ---------------------------------------------------------------------------

def _to_uint8(t):
    return (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _label_block(lines, width, font, pad=6, line_h=None, bg=(255, 255, 255),
                 fg=(20, 20, 20)):
    from PIL import ImageDraw
    line_h = line_h or (getattr(font, "size", 12) + 3)
    h = pad * 2 + line_h * len(lines)
    block = Image.new("RGB", (width, h), bg)
    draw = ImageDraw.Draw(block)
    for i, text in enumerate(lines):
        w = draw.textlength(text, font=font)
        draw.text(((width - w) / 2, pad + i * line_h), text, font=font, fill=fg)
    return block


def _titled(strip, title, font, bg=(255, 255, 255), fg=(20, 20, 20)):
    from PIL import ImageDraw
    # Shrink to fit: a clipped title on a difference panel is how an amplified
    # perturbation gets mistaken for an image.
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    while (probe.textlength(title, font=font) > strip.width - 16
           and getattr(font, "size", 12) > 9):
        font = _load_font(getattr(font, "size", 12) - 1)
    head = Image.new("RGB", (strip.width, getattr(font, "size", 12) + 14), bg)
    ImageDraw.Draw(head).text((8, 6), title, font=font, fill=fg)
    out = Image.new("RGB", (strip.width, head.height + strip.height), bg)
    out.paste(head, (0, 0))
    out.paste(strip, (0, head.height))
    return out


def save_trajectory_grid(path, images_by_step, steps, records, i, *, title, gap=6):
    """original | step 1 | ... | step 60, each captioned with the numbers."""
    font = _load_font(15)
    title_font = _load_font(18)
    tiles = []
    for t in steps:
        img = Image.fromarray(_to_uint8(images_by_step[t][i]))
        r = records[t]
        head = "ORIGINAL (step 0)" if t == 0 else f"step {t}"
        lines = [
            head,
            f"p prob {float(r['target_prob'][i]):.3f}   p rank {int(r['target_rank'][i])}",
            f"probe rank {int(r['probe_target_rank'][i])}"
            f"   probe top1 {int(r['probe_top1_correct'][i])}",
            f"rel L2 {float(r['rel_l2'][i]):.4f}   RMS {float(r['rms255'][i]):.2f}/255",
        ]
        cap = _label_block(lines, img.width, font)
        tile = Image.new("RGB", (img.width, img.height + cap.height), (255, 255, 255))
        tile.paste(img, (0, 0))
        tile.paste(cap, (0, img.height))
        tiles.append(tile)
    w = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    strip = Image.new("RGB", (w, max(t.height for t in tiles)), (255, 255, 255))
    x = 0
    for tile in tiles:
        strip.paste(tile, (x, 0))
        x += tile.width + gap
    _titled(strip, title, title_font).save(path)


def save_perturbation_strips(prefix, images_by_step, steps, records, i, *,
                             amplify, title, want_heatmap, gap=6):
    """x_t - x0: symmetric-normalised, amplified, and (optionally) |d| heatmap."""
    font = _load_font(15)
    title_font = _load_font(18)
    x0 = images_by_step[0][i]

    def _strip(tiles):
        w = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
        s = Image.new("RGB", (w, max(t.height for t in tiles)), (255, 255, 255))
        x = 0
        for tile in tiles:
            s.paste(tile, (x, 0))
            x += tile.width + gap
        return s

    def _tile(img, lines):
        cap = _label_block(lines, img.width, font)
        tile = Image.new("RGB", (img.width, img.height + cap.height), (255, 255, 255))
        tile.paste(img, (0, 0))
        tile.paste(cap, (0, img.height))
        return tile

    # (1) symmetric per-image normalisation: grey = no change, full scale = max |d|
    tiles = []
    for t in steps:
        d = images_by_step[t][i] - x0
        peak = float(d.abs().amax())
        vis = torch.full_like(d, 0.5) if peak <= 0 else 0.5 + d / (2.0 * peak)
        tiles.append(_tile(Image.fromarray(_to_uint8(vis)), [
            f"step {t}  (normalised)",
            f"grey = 0, full scale = +-{255 * peak:.2f}/255",
            f"RMS {float(records[t]['rms255'][i]):.2f}/255",
        ]))
    _titled(_strip(tiles), title + "  |  x_t - x0, per-image symmetric normalisation "
            "(NOT an image: each panel has its own scale)",
            title_font).save(f"{prefix}_perturbation_normalized.png")

    # (2) fixed amplification, comparable across steps
    for amp in amplify:
        tiles = []
        for t in steps:
            d = images_by_step[t][i] - x0
            vis = (0.5 + amp * d).clamp(0, 1)
            sat = float((((0.5 + amp * d) < 0) | ((0.5 + amp * d) > 1)).float().mean())
            tiles.append(_tile(Image.fromarray(_to_uint8(vis)), [
                f"step {t}  (x{amp} AMPLIFIED)",
                f"0.5 + {amp} * (x_t - x0), clipped",
                f"clipped px {100 * sat:.1f}%   RMS {float(records[t]['rms255'][i]):.2f}/255",
            ]))
        _titled(_strip(tiles),
                title + f"  |  AMPLIFIED x{amp} DIFFERENCE -- NOT AN IMAGE, "
                        f"visibility only", title_font).save(
            f"{prefix}_perturbation_x{amp}.png")

    # (3) |d| heatmap
    if not want_heatmap:
        return
    fig, axes = plt.subplots(1, len(steps), figsize=(2.6 * len(steps), 3.2))
    axes = np.atleast_1d(axes)
    peak = max(float((images_by_step[t][i] - x0).abs().mean(0).amax()) for t in steps)
    peak = max(peak, 1e-8)
    for ax, t in zip(axes, steps):
        d = (images_by_step[t][i] - x0).abs().mean(0).numpy()
        im = ax.imshow(d * 255.0, cmap="inferno", vmin=0.0, vmax=peak * 255.0)
        ax.set_title(f"step {t}", fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.01,
                 label="|x_t - x0| mean over RGB (/255)")
    fig.suptitle(title + "  |  absolute perturbation heatmap", fontsize=11)
    fig.savefig(f"{prefix}_perturbation_heatmap.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables and report
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "arm", "image_index", "image_id", "target_class_global", "target_class_local",
    "target_class_name", "step", "p_target_logp", "p_target_prob", "p_top1_correct",
    "p_pred_class_local", "p_pred_class_global", "p_target_rank", "p_entropy",
    "p_margin", "p_top5_correct", "q_teacher_target_logp", "delta_logqp",
    "probe_target_logp", "probe_target_prob", "probe_top1_correct",
    "probe_pred_class", "probe_target_rank", "probe_top5_correct",
    "rel_l2", "abs_l2", "rms01", "rms255", "max_abs_change", "mean_abs_change",
    "grad_norm", "cos_g_delta_g_p",
]


def rows_for_arm(arm, records, steps, targets_global, class_ids, image_ids, class_names):
    rows = []
    for t in steps:
        r = records[t]
        n = r["target_prob"].shape[0]
        for i in range(n):
            g = int(targets_global[i])
            pred_local = int(r["pred"][i])
            row = {
                "arm": arm, "image_index": i, "image_id": image_ids[i],
                "target_class_global": g,
                "target_class_local": class_ids.index(g),
                "target_class_name": (class_names[g] if class_names else ""),
                "step": t,
                "p_target_logp": float(r["target_logp"][i]),
                "p_target_prob": float(r["target_prob"][i]),
                "p_top1_correct": float(r["top1_correct"][i]),
                "p_pred_class_local": pred_local,
                "p_pred_class_global": class_ids[pred_local],
                "p_target_rank": float(r["target_rank"][i]),
                "p_entropy": float(r["entropy"][i]),
                "p_margin": float(r["margin"][i]),
                "p_top5_correct": float(r["top5_correct"][i]),
                "probe_target_logp": float(r["probe_target_logp"][i]),
                "probe_target_prob": float(r["probe_target_prob"][i]),
                "probe_top1_correct": float(r["probe_top1_correct"][i]),
                "probe_pred_class": int(r["probe_pred"][i]),
                "probe_target_rank": float(r["probe_target_rank"][i]),
                "probe_top5_correct": float(r["probe_top5_correct"][i]),
                "rel_l2": float(r["rel_l2"][i]),
                "abs_l2": float(r["abs_l2"][i]),
                "rms01": float(r["rms01"][i]),
                "rms255": float(r["rms255"][i]),
                "max_abs_change": float(r["max_abs_change"][i]),
                "mean_abs_change": float(r["mean_abs_change"][i]),
                "grad_norm": float(r["grad_norm"][i]),
            }
            for key, col in (("q_teacher_target_logp", "q_teacher_target_logp"),
                             ("delta_logqp", "delta_logqp"),
                             ("cos_g_delta_g_p", "cos_g_delta_g_p")):
                row[col] = float(r[key][i]) if key in r else ""
            rows.append(row)
    return rows


def aggregate(records, steps, num_classes):
    """Mean and median over images at each saved step."""
    keys = ("target_logp", "target_prob", "top1_correct", "target_rank", "entropy",
            "margin", "top5_correct", "probe_target_logp", "probe_target_prob",
            "probe_top1_correct", "probe_target_rank", "probe_top5_correct",
            "rel_l2", "rms255", "max_abs_change", "mean_abs_change", "grad_norm",
            "q_teacher_target_logp", "delta_logqp", "cos_g_delta_g_p")
    out = []
    for t in steps:
        r = records[t]
        row = {"step": t, "n_images": int(r["target_prob"].shape[0])}
        for k in keys:
            if k not in r:
                continue
            v = r[k].float()
            row[f"{k}_mean"] = float(v.mean())
            row[f"{k}_median"] = float(v.median())
        row["frac_p_predicts_target"] = float(r["top1_correct"].mean())
        row["frac_probe_predicts_target"] = float(r["probe_top1_correct"].mean())
        row["p_chance_top1"] = 1.0 / num_classes
        row["p_chance_mean_rank"] = (num_classes + 1) / 2.0
        row["probe_chance_top1"] = 1.0 / 1000.0
        row["probe_chance_mean_rank"] = 500.5  # (1000 + 1) / 2
        out.append(row)
    return out


def verdict(summary, num_classes, *, initial_q_control=False):
    """The go/no-go signature, stated in the terms the experiment was posed in."""
    first, last = summary[0], summary[-1]
    p_chance = 1.0 / num_classes
    probe_chance_rank = 500.5
    p_rose = (last["target_prob_mean"] > 0.5
              and last["target_prob_mean"] > 4.0 * max(first["target_prob_mean"], p_chance))
    rank_collapsed = last["target_rank_mean"] <= max(2.0, 0.1 * first["target_rank_mean"])
    probe_moved_top1 = last["frac_probe_predicts_target"] > max(
        0.05, 3.0 * first["frac_probe_predicts_target"])
    # rank progress toward 1, measured as a fraction of the distance from chance
    span = max(probe_chance_rank - 1.0, 1.0)
    probe_rank_gain = (first["probe_target_rank_mean"] - last["probe_target_rank_mean"]) / span
    probe_moved = probe_moved_top1 or probe_rank_gain > 0.25
    probe_recognizes = probe_moved_top1 or (
        last["probe_top5_correct_mean"] > max(0.05, first["probe_top5_correct_mean"]))
    small_pixels = last["rms255_mean"] < 8.0
    if initial_q_control and last["rms255_mean"] == 0:
        label = "EXPECTED-ZERO-FIELD"
        text = ("q is an unchanged copy of p, so log p - log q cancels and the "
                "image stays exactly unchanged. This is an initialization "
                "control, not a test of a trained q.")
    elif (p_rose or rank_collapsed) and not probe_recognizes:
        label = "P-CONFIDENCE-WITHOUT-PROBE-RECOGNITION"
        text = ("p confidence/rank improves without new target recognition by the "
                "independent probe. Its rank may improve while remaining far "
                "from top-5; that alone does not establish semantic generation. "
                "Inspect the saved images for class structure or classifier shortcuts.")
    elif (p_rose or rank_collapsed) and probe_recognizes:
        label = "PROBE-TRANSFER"
        text = ("p confidence/rank improves and target recognition transfers to "
                "the independent probe. This can also occur for transferable "
                "adversarial patterns; inspect the images before claiming "
                "semantic generation.")
    elif not (p_rose or rank_collapsed):
        label = "INCONCLUSIVE-NO-ASCENT"
        text = ("the ascent did not reliably raise p(c|z); the diagnostic says "
                "nothing about semantics until it does. Raise --alpha or "
                "--num_steps and rerun.")
    else:
        label = "MIXED"
        text = "partial movement on both sides; read the per-step table."
    return {
        "label": label, "text": text,
        "p_rose": bool(p_rose), "p_rank_collapsed": bool(rank_collapsed),
        "probe_moved": bool(probe_moved),
        "probe_recognizes": bool(probe_recognizes),
        "probe_rank_gain_fraction_of_chance_span": float(probe_rank_gain),
        "perturbation_small": bool(small_pixels),
    }


def md_table(summary, arm, num_classes):
    head = ("| step | log p(c\\|z) | p target prob | p top1 | p rank | probe top1 | "
            "probe rank | rel L2 | RMS/255 |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [f"**Arm `{arm}` — mean over images**", "", head, sep]
    for r in summary:
        lines.append(
            f"| {r['step']} | {r['target_logp_mean']:.3f} | {r['target_prob_mean']:.4f} "
            f"| {r['frac_p_predicts_target']:.3f} | {r['target_rank_mean']:.2f} "
            f"| {r['frac_probe_predicts_target']:.3f} | {r['probe_target_rank_mean']:.1f} "
            f"| {r['rel_l2_mean']:.4f} | {r['rms255_mean']:.2f} |")
    lines += ["", f"**Arm `{arm}` — median over images**", "", head, sep]
    for r in summary:
        lines.append(
            f"| {r['step']} | {r['target_logp_median']:.3f} | {r['target_prob_median']:.4f} "
            f"| — | {r['target_rank_median']:.2f} | — "
            f"| {r['probe_target_rank_median']:.1f} | {r['rel_l2_median']:.4f} "
            f"| {r['rms255_median']:.2f} |")
    lines += ["",
              f"Chance for p: top-1 {1.0 / num_classes:.4f}, mean rank "
              f"{(num_classes + 1) / 2.0:.1f} ({num_classes}-way). "
              f"Chance for the probe: top-1 0.0010, mean rank 500.5 (1000-way)."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S")
    argv = list(sys.argv[1:] if argv is None else argv)
    args = resolve_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.vlm_disable_tf32 or args.vlm_dtype == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    base_precision_settings = {
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    torch.manual_seed(args.diag_seed)
    torch.cuda.manual_seed_all(args.diag_seed)

    out = Path(args.out_dir)
    (out / "originals").mkdir(parents=True, exist_ok=True)
    (out / "steps").mkdir(parents=True, exist_ok=True)
    (out / "visuals").mkdir(parents=True, exist_ok=True)

    logger.info("building the frozen VLM + p head from %s", args.vlm_p_head)
    vlm_judge, identity = build_vlm(args)
    heads = vlm_judge["vlm_heads"]
    class_map = vlm_judge["vlm_class_map"]
    class_ids = list(class_map.class_ids)
    num_classes = heads.num_classes
    size = int(identity["vlm_input_size"])

    q_info = None
    q_path = args.q_ckpt or (args.gen_ckpt if args.delta_mode or args.saliency_every else None)
    q_source = os.path.abspath(q_path) if q_path else "initial copy of p (q=p), frozen throughout"
    if q_path:
        q_info = load_q_teacher(vlm_judge, q_path)
    logger.info("q_teacher source: %s", q_source)

    logger.info("building the held-out ResNet-50 IMAGENET1K_V2 probe (never in any loss)")
    probe = ProbeClassifier(device="cuda")
    probe.eval().requires_grad_(False)

    if args.gen_ckpt:
        x0, targets_global, input_meta = mode_a_images(args)
    elif args.images:
        x0, targets_global, input_meta = mode_b_images(args, size)
    else:
        x0, targets_global, input_meta = mode_c_image(args, size)
    if x0.shape[-1] != size or x0.shape[-2] != size:
        raise SystemExit(
            f"x0 is {tuple(x0.shape[-2:])} but the p head defines z at {size}px")
    targets_local = class_map.to_local(targets_global)
    if int(targets_local.min()) < 0:
        bad = sorted({int(c) for c, l in zip(targets_global.tolist(),
                                             targets_local.tolist()) if l < 0})
        raise SystemExit(f"target class ids {bad} have no output unit in the p head")

    class_names = _imagenet_class_names()
    image_ids = [f"img{i:02d}_c{int(c):03d}"
                 for i, c in enumerate(targets_global.tolist())]
    for i, name in enumerate(image_ids):
        Image.fromarray(_to_uint8(x0[i].cpu())).save(out / "originals" / f"{name}.png")

    # every model frozen
    frozen = {
        "qwen_requires_grad": any(p.requires_grad for p in vlm_judge["vlm_extractor"].parameters()),
        "heads_requires_grad": any(p.requires_grad for p in heads.parameters()),
        "probe_requires_grad": any(p.requires_grad for p in probe.parameters()),
    }
    if any(frozen.values()):
        raise SystemExit(f"a model is not frozen: {frozen}")
    saliency_logger = (InputGradientLogger(out, args.saliency_every, image_ids,
                                          targets_global, q_source)
                       if args.saliency_every else None)

    if args.grad_mode == "normalized":
        per_step_rms = 255.0 * args.alpha / math.sqrt(3 * size * size)
        logger.info("[ascent] alpha=%g is the exact L2 norm of each per-image pixel "
                    "update: <= %.4f/255 RMS per step, <= %.2f/255 after %d steps "
                    "(the bound is tight only if every step points the same way)",
                    args.alpha, per_step_rms, per_step_rms * args.num_steps,
                    args.num_steps)

    steps = args.checkpoint_steps
    arms = {}
    grads0 = {}

    logger.info("=== arm p: ascend log p(c|z) ===")
    rec_p, imgs_p, g0_p = run_arm(x0, targets_global, targets_local,
                                  vlm_judge=vlm_judge, probe=probe, args=args, arm="p",
                                  saliency_logger=saliency_logger)
    arms["p"] = (rec_p, imgs_p)
    grads0["p"] = g0_p

    if args.delta_mode:
        logger.info("=== arm delta: ascend log p(c|z) - log q_teacher(c|z) ===")
        rec_d, imgs_d, g0_d = run_arm(x0, targets_global, targets_local,
                                      vlm_judge=vlm_judge, probe=probe, args=args,
                                      arm="delta", want_cross_cosine=True,
                                      saliency_logger=saliency_logger)
        arms["delta"] = (rec_d, imgs_d)
        grads0["delta"] = g0_d

    precision = None
    if args.precision_control:
        base_dtype = str(args.vlm_dtype)
        logger.info("=== precision control: reloading Qwen in fp32 ===")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        g0_base = grads0["p"].clone()
        del vlm_judge["vlm_extractor"], vlm_judge["model"]
        vlm_judge.pop("vlm_buffer", None)
        vlm_judge.pop("vlm_q_optimizer", None)
        gc.collect()
        torch.cuda.empty_cache()
        fp32_args = argparse.Namespace(**vars(args))
        fp32_args.vlm_dtype = "fp32"
        fp32_args.ascent_microbatch = max(1, args.ascent_microbatch // 3)
        fp32_judge, fp32_identity = build_vlm(fp32_args)
        if q_path:
            load_q_teacher(fp32_judge, q_path)
        rec_f, imgs_f, g0_f = run_arm(x0, targets_global, targets_local,
                                      vlm_judge=fp32_judge, probe=probe,
                                      args=fp32_args, arm="p",
                                      saliency_logger=saliency_logger, output_arm="p_fp32")
        arms["p_fp32"] = (rec_f, imgs_f)
        cos = F.cosine_similarity(g0_base.flatten(1), g0_f.flatten(1), dim=1)
        ratio = (g0_base.flatten(1).norm(dim=1)
                 / g0_f.flatten(1).norm(dim=1).clamp_min(EPS))
        precision = {
            "base_dtype": base_dtype, "compare_dtype": "fp32",
            "matmul_allow_tf32": False, "cudnn_allow_tf32": False,
            "cos_per_image": [float(v) for v in cos],
            "cos_mean": float(cos.mean()), "cos_median": float(cos.median()),
            "grad_norm_ratio_mean": float(ratio.mean()),
            "fp32_target_size": fp32_identity["vlm_target_size"],
        }
        logger.info("[precision] cos(g_%s, g_fp32) mean %.4f median %.4f, "
                    "|g_%s|/|g_fp32| mean %.3f", base_dtype, precision["cos_mean"],
                    precision["cos_median"], base_dtype, precision["grad_norm_ratio_mean"])
        vlm_judge = fp32_judge

    # -- write everything ---------------------------------------------------
    if saliency_logger is not None:
        saliency_logger.finish()
    all_rows = []
    summaries = {}
    for arm, (records, images) in arms.items():
        all_rows += rows_for_arm(arm, records, steps, targets_global, class_ids,
                                 image_ids, class_names)
        summaries[arm] = aggregate(records, steps, num_classes)
        arm_dir = out / "steps" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        for t, batch in images.items():
            for i, name in enumerate(image_ids):
                Image.fromarray(_to_uint8(batch[i])).save(
                    arm_dir / f"{name}_step{t:03d}.png")
        vis_dir = out / "visuals" / arm
        vis_dir.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(image_ids):
            g = int(targets_global[i])
            label = f"{g}: {class_names[g]}" if class_names else str(g)
            title = f"[{arm}] {name}  target = {label}  alpha={args.alpha} {args.grad_mode}"
            save_trajectory_grid(vis_dir / f"{name}_trajectory.png", images, steps,
                                 records, i, title=title)
            save_perturbation_strips(str(vis_dir / name), images, steps, records, i,
                                     amplify=args.amplify, title=title,
                                     want_heatmap=not args.no_heatmap)

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})

    with open(out / "summary.csv", "w", newline="") as f:
        cols = ["arm"] + sorted({k for s in summaries.values() for r in s for k in r})
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm, s in summaries.items():
            for r in s:
                w.writerow({"arm": arm, **r})

    verdicts = {arm: verdict(s, num_classes,
                            initial_q_control=arm == "delta" and q_path is None)
                for arm, s in summaries.items()}
    payload = {
        "command": "python " + " ".join([os.path.relpath(sys.argv[0], REPO_ROOT)] + argv),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "identity": identity,
        "input": input_meta,
        "targets_global": targets_global.tolist(),
        "targets_local": targets_local.tolist(),
        "image_ids": image_ids,
        "ascent": {
            "num_steps": args.num_steps, "alpha": args.alpha,
            "grad_mode": args.grad_mode, "objective_reduction": args.objective_reduction,
            "checkpoint_steps": steps, "ascent_microbatch": args.ascent_microbatch,
            "clamp": "[0,1] after every step",
        },
        "q_teacher": q_info,
        "q_source": q_source,
        "saliency_every": args.saliency_every,
        "frozen": frozen,
        "base_precision_settings": base_precision_settings,
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "slurm_job_id": os.getenv("SLURM_JOB_ID")},
        "precision_control": precision,
        "summary": summaries,
        "verdict": verdicts,
        "per_image": all_rows,
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(payload, f, indent=1)

    write_report(out, args, identity, input_meta, summaries, verdicts, num_classes,
                 q_info, precision, payload["command"], image_ids, q_source)
    logger.info("wrote %s", out / "REPORT.md")
    print(open(out / "REPORT.md").read())
    return 0


def write_report(out, args, identity, input_meta, summaries, verdicts, num_classes,
                 q_info, precision, command, image_ids, q_source):
    p_sum = summaries["p"]
    first, last = p_sum[0], p_sum[-1]
    v = verdicts["p"]
    lines = [
        "# Is `grad_x log p(c|z)` semantic or adversarial?",
        "",
        "Offline pixel-space ascent on a frozen Qwen2.5-VL answer state and the frozen",
        "real-data `p` head. No generator training; the only optimised variable is `x`.",
        "",
        "```",
        command,
        "```",
        "",
        "## Setup",
        "",
        f"- p head: `{identity['p_head_path']}` (sha256 `{identity['p_head_sha256'][:16]}...`,"
        f" real-val top-1 {identity.get('p_head_val_top1')})",
        f"- VLM: `{identity['vlm_model_name']}`",
        f"- layer {identity['vlm_layer']}, prompt sha `{identity['vlm_prompt_sha256'][:16]}...`,"
        f" d={identity['feature_dim']}, {identity['vlm_input_size']}px ->"
        f" {identity['vlm_target_size']}px, {identity['num_visual_tokens']} visual tokens",
        f"- prompt: `{identity['vlm_prompt']}`",
        f"- temperature {identity['temperature']:.4f}, {num_classes}-way",
        f"- Qwen dtype `{identity['vlm_dtype']}`",
        f"- input: {input_meta['mode']}, {len(image_ids)} images",
        f"- ascent: {args.num_steps} steps, alpha={args.alpha}, {args.grad_mode} gradient,"
        f" clamped to [0,1] after every step",
        f"- independent probe: torchvision ResNet-50 `IMAGENET1K_V2`, never in any loss",
        "",
    ]
    lines += [f"- q_teacher: {q_source}", ""]
    if args.saliency_every:
        lines += [
            f"Input gradients saved every {args.saliency_every} steps, plus step 0 and final.",
            "See [saliency maps, norms and explanation](saliency/README.md).",
            "If q is an initial copy of p and both remain frozen, their individual",
            "input gradients should agree at every image; the combined field should vanish.",
            "This initialization control does not test a q trained on generated images.", "",
        ]
    if q_info:
        lines += [
            "### q_teacher (delta arm)",
            "",
            f"- restored after {q_info['q_train_steps']} q steps",
            f"- `||W_q|| / ||W_p||` = {q_info['q_teacher_weight_rel_to_p']:.4f}, "
            f"`cos(W_q, W_p)` = {q_info['q_teacher_cos_to_p']:.4f} "
            "(both near 0 means q is uniform and the delta field reduces to "
            "`grad log p`)",
            "",
        ]
    lines += ["## Results", ""]
    for arm, s in summaries.items():
        lines += [md_table(s, arm, num_classes), ""]
    if "delta" in summaries:
        cos_key = [r.get("cos_g_delta_g_p_mean") for r in summaries["delta"]
                   if "cos_g_delta_g_p_mean" in r]
        if verdicts["delta"]["label"] == "EXPECTED-ZERO-FIELD":
            lines += ["### Does q change the field?", "",
                      "The combined gradient is zero and has no direction, so its cosine",
                      "with the p gradient is undefined (the legacy metrics use a zero placeholder).", ""]
        elif cos_key:
            lines += [
                "### Does q change the field?",
                "",
                f"`cos(grad_x[log p - log q_teacher], grad_x log p)` over saved steps: "
                f"mean {np.mean(cos_key):.4f}, min {np.min(cos_key):.4f}, "
                f"max {np.max(cos_key):.4f}.",
                "", ]
    if precision:
        lines += [
            "### Precision control",
            "",
            f"`cos(g_{precision['base_dtype']}, g_fp32)` at x0: mean "
            f"{precision['cos_mean']:.4f}, median {precision['cos_median']:.4f}; "
            f"magnitude ratio {precision['grad_norm_ratio_mean']:.3f}. "
            "Compare the `p` and `p_fp32` tables above for how fast each raises p "
            "and whether either moves the probe.",
            "", ]

    lines += [
        "## Answers",
        "",
        "**Does ascent reliably increase p(c|z)?** "
        f"log p(c|z) {first['target_logp_mean']:.2f} -> {last['target_logp_mean']:.2f}; "
        f"target prob {first['target_prob_mean']:.4f} -> {last['target_prob_mean']:.4f}; "
        f"p top-1 {first['frac_p_predicts_target']:.3f} -> "
        f"{last['frac_p_predicts_target']:.3f}; "
        f"p rank {first['target_rank_mean']:.1f} -> {last['target_rank_mean']:.2f} "
        f"(chance {(num_classes + 1) / 2:.1f}).",
        "",
        "**How much pixel perturbation is required?** "
        f"relative L2 {last['rel_l2_mean']:.4f}, "
        f"RMS {last['rms255_mean']:.2f}/255, "
        f"max abs change {255 * last['max_abs_change_mean']:.1f}/255, "
        f"mean abs change {255 * last['mean_abs_change_mean']:.2f}/255 at the final "
        "saved step.",
        "",
        "**Does the independent ResNet probe move toward the same target?** "
        f"probe top-1 {first['frac_probe_predicts_target']:.4f} -> "
        f"{last['frac_probe_predicts_target']:.4f} (chance 0.0010); "
        f"probe rank {first['probe_target_rank_mean']:.1f} -> "
        f"{last['probe_target_rank_mean']:.1f} (chance 500.5); "
        f"probe top-5 {first['probe_top5_correct_mean']:.4f} -> "
        f"{last['probe_top5_correct_mean']:.4f}.",
        "",
        f"**Is the p gradient semantic or adversarial?** **{v['label']}.** {v['text']}",
        "",
        "A gradient is a local derivative; normalized finite steps can lower p. "
        "Classifier confidence and rank alone do not establish semantic generation. "
        "Use the image trajectory together with the independent probe results.",
        "",
    ]
    if "delta" in verdicts:
        lines += [
            "**If full-delta mode was tested, does q materially change the field?** "
            f"delta arm verdict: **{verdicts['delta']['label']}**. {verdicts['delta']['text']}",
            "", ]
    lines += [
        "## Files",
        "",
        "- `metrics.csv` — one row per (arm, image, saved step)",
        "- `summary.csv`, `metrics.json` — per-step mean/median aggregates and metadata",
        "- `originals/` — x0 as lossless PNG",
        "- `steps/<arm>/` — every saved step as lossless PNG",
        "- `visuals/<arm>/*_trajectory.png` — original | step 1 | ... | final, captioned",
        "- `visuals/<arm>/*_perturbation_normalized.png` — x_t - x0, per-image symmetric scale",
        "- `visuals/<arm>/*_perturbation_x*.png` — AMPLIFIED difference, visibility only, "
        "not an image",
        "- `visuals/<arm>/*_perturbation_heatmap.png` — |x_t - x0| heatmap",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
