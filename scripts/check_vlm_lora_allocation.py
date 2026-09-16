"""Fail early on an unsuitable Slurm allocation or broken GPU collectives."""
import argparse
import datetime
import json
import math
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--min-vram-mib", type=int, default=None,
                        help="Default 44000 for FP32; no asserted memory floor for BF16")
    parser.add_argument("--loss-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.expected_gpus < 1 or not math.isfinite(args.loss_scale) or args.loss_scale <= 0:
        parser.error("expected GPU count and finite loss scale must be positive")
    minimum = args.min_vram_mib
    if minimum is None:
        minimum = 44000 if args.dtype == "fp32" else 0
    if minimum < 0:
        parser.error("min-vram-mib must be non-negative")
    local_rank = int(os.environ["LOCAL_RANK"])
    if (torch.cuda.device_count() != args.expected_gpus
            or int(os.environ["WORLD_SIZE"]) != args.expected_gpus):
        raise RuntimeError(f"Expected {args.expected_gpus} visible GPUs/ranks; "
                           f"found {torch.cuda.device_count()} GPUs. Check the Slurm allocation.")
    torch.cuda.set_device(local_rank)
    props = torch.cuda.get_device_properties(local_rank)
    if props.total_memory < minimum * 1024**2:
        raise RuntimeError(f"Rank {local_rank}: {props.name} is below {minimum} MiB VRAM")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError(f"Rank {local_rank}: {props.name} does not support native BF16")
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank),
                            timeout=datetime.timedelta(minutes=3))
    try:
        value = torch.tensor(float(dist.get_rank() + 1), device="cuda")
        dist.all_reduce(value)
        expected = args.expected_gpus * (args.expected_gpus + 1) / 2
        torch.testing.assert_close(value, torch.tensor(expected, device="cuda"))
        # Verify a forward/backward and synchronized parameter gradient.
        weight = torch.eye(8, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.dtype == "bf16"):
            loss = (weight @ torch.ones(8, 8, device="cuda")).square().mean()
        (loss * args.loss_scale).backward()
        weight.grad.div_(args.loss_scale)
        dist.all_reduce(weight.grad)
        if not torch.isfinite(weight.grad).all() or weight.grad.norm() == 0:
            raise RuntimeError("GPU gradient/collective preflight failed")
        records = [None] * dist.get_world_size()
        dist.all_gather_object(records, {
            "rank": dist.get_rank(), "name": props.name,
            "vram_mib": props.total_memory // 1024**2,
            "hostname": socket.gethostname(),
            "compute_dtype": args.dtype,
        })
        if dist.get_rank() == 0:
            report = {"status": "PASS", "torch": torch.__version__,
                      "cuda": torch.version.cuda, "gpus": records,
                      "loss_scale": args.loss_scale,
                      "note": "Arithmetic/collective check; full-model memory and gradient accuracy are not validated."}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
