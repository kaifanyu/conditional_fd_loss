#!/usr/bin/env python3
"""Fixed-input Qwen image-gradient precision/scale sweep; no training or downloads.

Input: torch.save({'images': CPU float [N,3,H,W] in [0,1],
                   'labels': CPU long [N] GLOBAL class IDs}, path).
Use representative generated images. A controlled q-head perturbation provides
a nonzero delta for cancellation diagnostics; this is not a trained-q eval.
"""

import argparse
import gc
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_answer_state import QwenAnswerStateExtractor
from vlm_linear_heads import (LocalClassMap, VLMDeltaHeads, build_head_from_checkpoint,
                              load_p_head_checkpoint, p_head_backend)
from vlm_lora_q import validate_loss_scale


def comparison(value, reference):
    value, reference = value.detach().cpu().double(), reference.detach().cpu().double()
    a, b = value.flatten(1), reference.flatten(1)
    na, nb = a.norm(dim=1), b.norm(dim=1)
    valid = (na > 0) & (nb > 0) & torch.isfinite(a).all(1) & torch.isfinite(b).all(1)
    cos = (a[valid] * b[valid]).sum(1) / (na[valid] * nb[valid])
    finite = bool(torch.isfinite(value).all())
    return {
        "finite": finite,
        "cos_per_image": cos.tolist(),
        "valid_cos_images": int(valid.sum()),
        "norm": float(a.norm()) if finite else None,
        "reference_norm": float(b.norm()),
        "relative_error": float((a - b).norm() / b.norm()) if finite and b.norm() > 0 else None,
        "zero_fraction": float((value == 0).double().mean()),
    }


def measure(extractor, heads, images, labels, scale):
    features, logps, logqs, gp, gq = [], [], [], [], []
    for start in range(0, images.shape[0], extractor.microbatch_size):
        x = images[start:start + extractor.microbatch_size].detach().requires_grad_(True)
        y = labels[start:start + extractor.microbatch_size]
        z = extractor.answer_states(x)[extractor.layer]
        p, q = heads.p_log_probs(z), heads.q_teacher_log_probs(z)
        # Separate branches reveal when subtraction amplifies small errors.
        a = p.gather(1, y[:, None]).sum() / images.shape[0]
        b = q.gather(1, y[:, None]).sum() / images.shape[0]
        gp.append((torch.autograd.grad(a * scale, x, retain_graph=True)[0].float() / scale).cpu())
        gq.append((torch.autograd.grad(b * scale, x)[0].float() / scale).cpu())
        features.append(z.detach().cpu())
        logps.append(p.detach().cpu())
        logqs.append(q.detach().cpu())
        del z, p, q, a, b
    gp, gq = torch.cat(gp), torch.cat(gq)
    return {"features": torch.cat(features), "logp": torch.cat(logps),
            "logq": torch.cat(logqs), "grad_logp": gp, "grad_logq": gq,
            "grad_delta": gq - gp}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p_head", required=True)
    parser.add_argument("--images_pt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--scales", type=float, nargs="+", default=[1, 128, 1024, 8192])
    parser.add_argument("--dtypes", nargs="+", choices=["fp32", "bf16", "fp16"], default=["fp32", "bf16", "fp16"])
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--q_head_perturbation", type=float, default=1e-3)
    args = parser.parse_args()
    for scale in args.scales:
        validate_loss_scale(scale)
    if not torch.cuda.is_available():
        raise RuntimeError("This precision comparison requires CUDA and the cached Qwen checkpoint")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    ckpt = load_p_head_checkpoint(args.p_head)
    if p_head_backend(ckpt) != "qwen_answer_state":
        raise ValueError("A Qwen answer-state p head is required")
    payload = torch.load(args.images_pt, map_location="cpu", weights_only=True)
    images = payload["images"].float().cuda()
    if not bool(torch.isfinite(images).all()) or images.min() < 0 or images.max() > 1:
        raise ValueError("images must be finite and in [0,1]")
    labels = LocalClassMap(ckpt["class_ids"]).to_local(payload["labels"].long().cuda())
    if labels.shape != (images.shape[0],) or bool((labels < 0).any()):
        raise ValueError("labels must have one global class ID per image, within the p head's class subset")
    heads = VLMDeltaHeads(build_head_from_checkpoint(ckpt, "cuda"), temperature=ckpt["temperature"]).cuda()
    # Fixed synthetic q across every dtype and scale; never compare a zero delta.
    rng = torch.Generator(device="cpu").manual_seed(7717)
    with torch.no_grad():
        noise = torch.randn(heads.q_teacher.weight.shape, generator=rng).cuda()
        heads.q_teacher.weight.add_(noise * args.q_head_perturbation)
    report = {"p_head": str(args.p_head), "images_pt": str(args.images_pt),
              "q_head_perturbation": args.q_head_perturbation,
              "microbatch": args.microbatch, "attn_implementation": args.attn_implementation,
              "tf32": False, "measurements": []}
    reference = None
    for dtype in dict.fromkeys(["fp32"] + args.dtypes):
        extractor = QwenAnswerStateExtractor(
            ckpt["vlm_model_name"], prompt=ckpt["vlm_prompt"], layer=ckpt["vlm_layer"],
            image_size=ckpt["vlm_input_size"], dtype=dtype,
            microbatch_size=args.microbatch, attn_implementation=args.attn_implementation)
        for scale in dict.fromkeys([1.0] + args.scales):
            result = measure(extractor, heads, images, labels, scale)
            if reference is None:
                reference = result
            entry = {"dtype": dtype, "loss_scale": scale,
                     **{key: comparison(value, reference[key]) for key, value in result.items()}}
            report["measurements"].append(entry)
            print(json.dumps(entry), flush=True)
        del extractor
        gc.collect()
        torch.cuda.empty_cache()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
