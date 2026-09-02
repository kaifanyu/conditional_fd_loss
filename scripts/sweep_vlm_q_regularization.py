"""Pick the q head's weight decay: does its drift -- and so the field -- plateau?

Run this against the replay buffer saved in any VLM-delta checkpoint, BEFORE
committing to a full trial.  It answers offline in a few minutes a question that
otherwise costs a wasted multi-hour run.

WHY IT EXISTS.  On the first 100-class calibration run
``q_teacher_weight_delta_rel`` grew 0.44 -> 0.97 of ||W_p|| between steps 900 and
2640 with no sign of stopping, and ``grad_ratio_vlm_fd`` tracked it 0.60 -> 1.04
all the way up.  That is not a bug: at a de-conditioned start the generated
images carry no linearly-decodable label information, so q's cross-entropy
gradient is unbiased noise and an unregularised online head RANDOM-WALKS away
from p.  ||W_q - W_p|| grows like sqrt(steps) forever, the injected field grows
with it, and no conditional weight stays calibrated.

AdamW's decoupled weight decay pulls W toward 0 -- which is exactly the uniform
posterior that IS the correct q at a de-conditioned start -- turning the walk
into an Ornstein-Uhlenbeck process with a stationary distribution.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/sweep_vlm_q_regularization.py \
        --checkpoint work_dirs/<run>/checkpoints/step_0001999.pth \
        --p_head work_dirs/vlm_p_head_siglip_c100/p_head.pt

How to read the table:
  * ``plateau? = no (walks)`` -- that setting cannot be calibrated. Reject it.
  * ``heldCE`` against the printed uniform CE -- how close q is to the TRUE
    posterior on generated features it has never trained on. Below uniform means
    q found real class signal; above means it is overconfident noise.
  * ``||g_z||`` -- the magnitude of the field the generator would receive, which
    is what --vlm_delta_weight is calibrated against.
"""
import math, sys, torch, torch.nn.functional as F
sys.path.insert(0, "/data/jgu/kai/FD-Loss")
from vlm_linear_heads import build_head_from_checkpoint, load_p_head_checkpoint

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True,
                help="any VLM-delta training checkpoint (it carries the q replay buffer)")
ap.add_argument("--p_head", default="work_dirs/vlm_p_head_siglip_c100/p_head.pt")
ap.add_argument("--batch", type=int, default=96,
                help="q batch size; equal to the run's global batch means reuse 1.0")
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--ema_beta", type=float, default=0.999)
args = ap.parse_args()
buf = torch.load(args.checkpoint, map_location="cpu",
                 weights_only=False)["vlm_delta_state"]["q_buffer"]
feats, fill = buf["feats"], buf["fill"]
C = feats.shape[0]
z = torch.cat([feats[c, :int(fill[c])].float() for c in range(C) if fill[c]]).cuda()
y = torch.cat([torch.full((int(fill[c]),), c, dtype=torch.long) for c in range(C) if fill[c]]).cuda()
g = torch.Generator().manual_seed(0)
perm = torch.randperm(z.shape[0], generator=g).cuda()
n_tr = int(0.8 * z.shape[0]); ztr, ytr = z[perm[:n_tr]], y[perm[:n_tr]]
zte, yte = z[perm[n_tr:]], y[perm[n_tr:]]
UNIF = math.log(C)
pck = load_p_head_checkpoint(args.p_head)
T = float(pck["temperature"])

BATCH, STEPS, BETA = args.batch, args.steps, args.ema_beta
MARKS = tuple(sorted({max(1, STEPS // 10), max(1, STEPS // 4), max(1, STEPS // 2), STEPS}))
print(f"batch {BATCH} (reuse 1.0 at that global batch), {STEPS} updates, "
      f"teacher EMA beta {BETA}, uniform CE {UNIF:.4f}\n")
print(f"{'lr':>7} {'wd':>6} | " + " ".join(f"{'d@'+str(s):>8}" for s in MARKS)
      + f" | {'plateau?':>9} | {'teach_d':>8} {'cos':>7} {'heldCE':>8} {'||g_z||':>8}")
print("-" * 108)
for lr, wd in [(1e-3, 0.0), (1e-3, 0.1), (1e-3, 1.0), (1e-3, 3.0), (1e-3, 10.0),
               (3e-4, 0.0), (3e-4, 1.0), (3e-4, 3.0), (3e-4, 10.0),
               (1e-4, 0.0), (1e-4, 3.0), (1e-4, 10.0)]:
    head = build_head_from_checkpoint(pck, device="cuda"); head.linear.requires_grad_(True)
    teacher = build_head_from_checkpoint(pck, device="cuda").eval().requires_grad_(False)
    p_head = build_head_from_checkpoint(pck, device="cuda").eval().requires_grad_(False)
    p_w = p_head.weight.detach().clone(); p_n = float(p_w.norm())
    opt = torch.optim.AdamW(head.linear.parameters(), lr=lr, weight_decay=wd)
    gg = torch.Generator().manual_seed(1)
    marks = {}
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, ztr.shape[0], (BATCH,), generator=gg).cuda()
        loss = F.cross_entropy(head.logits(ztr[idx], T), ytr[idx])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.linear.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for tp, sp in zip(teacher.linear.parameters(), head.linear.parameters()):
                tp.add_((1 - BETA) * (sp.detach() - tp))
        if step in MARKS:
            marks[step] = float((head.weight.detach() - p_w).norm()) / p_n
    with torch.no_grad():
        td = float((teacher.weight.detach() - p_w).norm()) / p_n
        cos = float(F.cosine_similarity(teacher.weight.detach().reshape(1, -1), p_w.reshape(1, -1)))
        ce = float(F.cross_entropy(teacher.log_probs(zte, T), yte))
    # magnitude of the field the generator would actually receive (needs grad)
    zz = zte[:512].clone().requires_grad_(True)
    yy = yte[:512]
    d = (teacher.log_probs(zz, T).gather(1, yy.view(-1, 1))
         - p_head.log_probs(zz, T).gather(1, yy.view(-1, 1))).mean()
    gnorm = float(torch.autograd.grad(d, zz)[0].norm(dim=1).mean())
    growth = marks[MARKS[-1]] / max(marks[MARKS[1]], 1e-9)
    plateau = "YES" if growth < 1.15 else ("partly" if growth < 1.6 else "no (walks)")
    print(f"{lr:>7.0e} {wd:>6.3g} | " + " ".join(f"{marks[s]:>8.3f}" for s in MARKS)
          + f" | {plateau:>9} | {td:>8.3f} {cos:>7.3f} {ce:>8.4f} {gnorm:>8.5f}")
