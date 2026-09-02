"""De-condition a class-conditional JiT checkpoint (Way 1).

Overwrites the 1000 class rows of the label embedding table with the null row
(index = num_classes) in the model AND both EMA copies, so every class index
0..999 initially produces the same generic (unconditional) image. Everything
else in the network — the trained "reader" (adaLN, in-context attention, MLPs,
patch embed) — is left untouched.

Then fine-tune with conditional_main_fd_ponly.py (FD-loss + lambda_cond * -log p(c|x)),
pointing --load_from at the produced checkpoint, and the CLIP gradient re-carves
rows 0..999 into class-specific vectors.

Usage:
    python make_uncond_jit.py \
        --in_ckpt checkpoints/base/JiT-B.pth \
        --out_ckpt checkpoints/base/JiT-B-uncond.pth \
        --num_classes 1000
"""
import argparse
import torch

EMB_SUFFIX = "y_embedder.embedding_table.weight"
SUBDICTS = ["model", "model_ema1", "model_ema2"]  # all weight copies in the JiT ckpt


def _find_emb_key(sd):
    keys = [k for k in sd if k.endswith(EMB_SUFFIX)]
    if len(keys) != 1:
        raise KeyError(f"expected exactly one '*{EMB_SUFFIX}', found {keys}")
    return keys[0]


def decondition_subdict(sd, num_classes, name):
    key = _find_emb_key(sd)
    W = sd[key].detach().clone()                     # detach: EMA copies require grad
    n_rows, dim = W.shape
    assert n_rows == num_classes + 1, (
        f"[{name}] table has {n_rows} rows but num_classes+1={num_classes + 1}; "
        f"is --num_classes correct?"
    )
    null = W[num_classes].clone()                    # the unconditional row
    before = W[:num_classes].std(0).mean().item()    # spread across class rows
    W[:num_classes] = null.unsqueeze(0)              # broadcast null into all classes
    sd[key] = W                                       # reassign the modified table
    after = W[:num_classes].std(0).mean().item()
    # sanity: every class row must now equal the null row exactly
    assert torch.equal(W[0], null) and torch.equal(W[num_classes - 1], null)
    print(f"[{name}] {key}: class-row std {before:.5f} -> {after:.5f} "
          f"(null row norm {null.norm().item():.4f}, untouched)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ckpt", required=True)
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--num_classes", type=int, default=1000)
    args = ap.parse_args()

    ck = torch.load(args.in_ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise TypeError(f"unexpected checkpoint type {type(ck)}")

    touched = 0
    for name in SUBDICTS:
        if name in ck and isinstance(ck[name], dict):
            decondition_subdict(ck[name], args.num_classes, name)
            touched += 1
        else:
            print(f"[{name}] absent — skipping")
    if touched == 0:
        raise RuntimeError("no model/EMA subdicts found to de-condition")

    torch.save(ck, args.out_ckpt)
    print(f"\nsaved de-conditioned checkpoint -> {args.out_ckpt} ({touched} weight copies reset)")


if __name__ == "__main__":
    main()
