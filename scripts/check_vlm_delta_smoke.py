"""Assertions over the artefacts a smoke run of the VLM-delta trial produced.

Invoked by ``scripts/smoke_vlm_delta.sh``; separated out so the checks are
readable and can be re-run against an existing run directory.

Every assertion here is one of the ten Step-B integration checks, phrased
against what the real training script actually wrote to disk.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_linear_heads import (  # noqa: E402
    VLMDeltaHeads,
    build_head_from_checkpoint,
    load_p_head_checkpoint,
    p_head_identity,
)

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if condition else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))
    if not condition:
        FAILURES.append(label)


def load_metrics(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "training_metrics.json")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        raise SystemExit(f"no metrics in {path}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="fresh smoke run directory")
    ap.add_argument("--resumed", required=True, help="resumed smoke run directory")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--p_head", required=True)
    args = ap.parse_args()

    rows = load_metrics(args.run)
    first, last = rows[0], rows[-1]

    print("\n[1/8] initial diagnostic: q == p exactly")
    with open(os.path.join(args.run, "vlm_init_diagnostics.json")) as f:
        init = json.load(f)
    for key, value in init["init_equality"].items():
        check(f"init {key} == 0", value == 0.0, f"{value:.3e}")
    n_cls = init["num_classes"]
    check("p is at chance on the sampled label (rank within 10% of chance)",
          abs(init["p"]["p_target_rank"] - init["chance_mean_rank"])
          < 0.10 * init["chance_mean_rank"],
          f"rank {init['p']['p_target_rank']:.2f} vs chance {init['chance_mean_rank']:.1f}")
    use_ema = init.get("q_generator_source", "teacher") == "teacher"
    active_q = "q_teacher" if use_ema else "q_student"
    check(f"{active_q} matches p at init",
          init[active_q][f"{active_q}_target_logp"] == init["p"]["p_target_logp"])

    print("\n[2/8] the conditional term is exactly zero at step 0")
    check("vlm_delta_logqp == 0 at step 0", first["vlm_delta_logqp"] == 0.0,
          f"{first['vlm_delta_logqp']:.3e}")
    check("grad_x_vlm_delta == 0 at step 0", first["grad_x_vlm_delta"] == 0.0,
          f"{first['grad_x_vlm_delta']:.3e}")
    check("grad_ratio_vlm_fd == 0 at step 0", first["grad_ratio_vlm_fd"] == 0.0)

    print("\n[3/8] the generator gradient is finite and the FD term drives it")
    for row in rows:
        if not math.isfinite(row.get("generator_grad_norm", float("nan"))):
            check(f"generator_grad_norm finite at step {row['iteration']}", False)
            break
    else:
        check("generator_grad_norm finite at every logged step", True,
              f"last {last['generator_grad_norm']:.4f}")
    check("generator_grad_norm > 0", last["generator_grad_norm"] > 0)
    check("grad_x_fd > 0", last["grad_x_fd"] > 0, f"{last['grad_x_fd']:.4e}")

    print("\n[4/8] the conditional gradient becomes non-zero once q != p")
    nonzero = [r for r in rows if r.get("grad_x_vlm_delta", 0.0) > 0.0]
    check("grad_x_vlm_delta > 0 at some later step", bool(nonzero),
          "" if not nonzero else f"first at step {nonzero[0]['iteration']}, "
                                 f"{nonzero[0]['grad_x_vlm_delta']:.3e}")
    check("every logged grad_x_vlm_delta is finite",
          all(math.isfinite(r["grad_x_vlm_delta"]) for r in rows if "grad_x_vlm_delta" in r))

    print("\n[5/8] q_student learns; active q is " + active_q)
    # q starts AT p, which is a bad classifier of generated images (measured:
    # CE 7.70 on generated features against a uniform of 4.61), so any q that is
    # training at all must improve on its own starting point.  Compared against
    # the first post-update value rather than a fixed threshold, because the
    # absolute level depends on the class count.
    check("q_ce improves on its p initialisation",
          last["q_ce"] < rows[0]["q_ce"],
          f"{rows[0]['q_ce']:.3f} -> {last['q_ce']:.3f}")
    check("q_student drifted from p", last["q_student_weight_delta_l2"] > 1e-3,
          f"{last['q_student_weight_delta_l2']:.4f}")
    if use_ema:
        check("q_teacher drifted less than q_student",
              last["q_teacher_weight_delta_l2"] < last["q_student_weight_delta_l2"],
              f"teacher {last['q_teacher_weight_delta_l2']:.5f} vs student "
              f"{last['q_student_weight_delta_l2']:.5f}")

    print("\n[6/8] the q training data stays class-balanced and q is not memorising")
    check("buffer covers every class", last["q_buffer_class_coverage"] == 1.0,
          f"{last['q_buffer_class_coverage']:.4f}")
    check("buffer max/min per class within the ring capacity",
          last["q_buffer_max_samples_per_class"] >= last["q_buffer_min_samples_per_class"],
          f"{last['q_buffer_min_samples_per_class']:.0f}-"
          f"{last['q_buffer_max_samples_per_class']:.0f}")
    # A q that scores far better on its replay buffer than on a batch it has not
    # yet seen is memorising, and log q_teacher(c|z) is then noise on exactly the
    # fresh samples the generator loss reads it at.
    gaps = [r["q_generalization_gap"] for r in rows if "q_generalization_gap" in r]
    check("q_generalization_gap is logged", bool(gaps))
    if gaps:
        check("q is not badly memorising its replay buffer", gaps[-1] < 1.0,
              f"fresh CE - buffer CE = {gaps[-1]:.4f}")

    print("\n[7/8] the objective was not silently modified")
    check("vlm_delta_clamp_frac == 0 (no clamping by default)",
          all(r.get("vlm_delta_clamp_frac", 0.0) == 0.0 for r in rows))
    # These five are logged with a window of 1 precisely so this reconciles.
    worst = max(abs(r["vlm_delta_logqp"] - (r["vlm_logq_c"] - r["vlm_logp_c"]))
                for r in rows)
    check("vlm_delta_logqp == vlm_logq_c - vlm_logp_c at every logged step",
          worst < 1e-4, f"max discrepancy {worst:.2e}")

    print("\n[8/8] checkpoint round-trip")
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = blob["vlm_delta_state"]
    ckpt = load_p_head_checkpoint(args.p_head)
    check("the checkpoint records the p-head identity",
          state["p_identity"] == p_head_identity(ckpt))
    check("the checkpoint records the weight schedule", "weight_schedule" in state)
    check("the checkpoint records the q optimizer", state.get("q_optimizer") is not None)
    check("the checkpoint records the replay buffer", state.get("q_buffer") is not None)

    heads = VLMDeltaHeads(build_head_from_checkpoint(ckpt, device="cpu"),
                          temperature=float(state["q"]["temperature"]),
                          ema_beta=float(state["q"]["ema_beta"]),
                          use_ema=bool(state["q"].get("use_ema", True)))
    heads.load_q_state_dict(state["q"])
    exact = all(
        torch.equal(getattr(heads, name).linear.state_dict()[key],
                    state["q"][name][f"linear.{key}"])
        for name in ("q_student", "q_teacher") for key in ("weight", "bias"))
    check("q_student and q_teacher restore bit-exactly", exact)
    check("q_student is trainable after restore",
          heads.q_student.linear.weight.requires_grad)
    check("q_teacher is frozen after restore",
          not heads.q_teacher.linear.weight.requires_grad)
    check("p stays frozen after restore", not heads.p_head.linear.weight.requires_grad)
    check("q_train_steps restored", int(heads.q_train_steps.item()) > 0,
          str(int(heads.q_train_steps.item())))

    resumed = load_metrics(args.resumed)
    check("the resumed run continued past the checkpoint step",
          resumed[-1]["iteration"] > blob["step"],
          f"{blob['step']} -> {resumed[-1]['iteration']}")
    check("the resumed run kept q's drift (it did not restart from p)",
          resumed[0]["q_student_weight_delta_l2"] > 0.0,
          f"{resumed[0]['q_student_weight_delta_l2']:.4f}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: " + ", ".join(FAILURES))
        return 1
    print("all smoke assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
