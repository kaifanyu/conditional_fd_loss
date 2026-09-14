"""Fail early on an unsuitable Slurm allocation or broken GPU collectives."""
import argparse
import datetime
import json
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.device_count() != 4 or int(os.environ["WORLD_SIZE"]) != 4:
        raise RuntimeError("This launch requires exactly four visible GPUs/ranks")
    torch.cuda.set_device(local_rank)
    props = torch.cuda.get_device_properties(local_rank)
    if props.total_memory < 44000 * 1024**2:
        raise RuntimeError(f"Rank {local_rank}: {props.name} has insufficient VRAM for FP32 Qwen")
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank),
                            timeout=datetime.timedelta(minutes=3))
    try:
        value = torch.tensor(float(dist.get_rank() + 1), device="cuda")
        dist.all_reduce(value)
        torch.testing.assert_close(value, torch.tensor(10.0, device="cuda"))
        # Verify a forward/backward and synchronized parameter gradient.
        weight = torch.eye(8, device="cuda", requires_grad=True)
        (weight @ torch.ones(8, 8, device="cuda")).square().mean().backward()
        dist.all_reduce(weight.grad)
        if not torch.isfinite(weight.grad).all() or weight.grad.norm() == 0:
            raise RuntimeError("GPU gradient/collective preflight failed")
        records = [None] * dist.get_world_size()
        dist.all_gather_object(records, {
            "rank": dist.get_rank(), "name": props.name,
            "vram_mib": props.total_memory // 1024**2,
            "hostname": socket.gethostname(),
        })
        if dist.get_rank() == 0:
            report = {"status": "PASS", "torch": torch.__version__,
                      "cuda": torch.version.cuda, "gpus": records}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
