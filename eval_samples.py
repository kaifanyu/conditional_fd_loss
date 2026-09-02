"""
Compute class-fidelity metrics on a folder of generated images.

Runs two HELD-OUT classifiers (neither was used during training):
  1. timm ResNet-50 (in1k) — completely different architectural family than CLIP.
     The strongest "is it actually class c?" test.
  2. CLIP ViT-B-32 (openai) — same family as the ViT-L-14 used in conditional
     training, but smaller. Tests whether the gain transfers within CLIP family.

Interpretation:
  ResNet-50 top1 UP  AND  ViT-B-32 top1 UP   -> real class-fidelity gain
  ViT-B-32 UP but ResNet-50 flat             -> suspect CLIP-family gaming
  Neither UP                                 -> conditional term ineffective
  (Read alongside FID. If FID got worse, even a real gain may not be worth it.)

Usage:
    python eval_samples.py --dir eval_out/baseline --tag baseline
    python eval_samples.py --dir eval_out/cond     --tag cond
    # then:
    python compare_metrics.py eval_out/baseline/eval_metrics.json \\
                              eval_out/cond/eval_metrics.json
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class GeneratedDataset(Dataset):
    """Reads `{root}/class_XXX/*.png`, labels by folder name."""
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.items = []
        for class_dir in sorted(self.root.glob("class_*")):
            try:
                c = int(class_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            for img_path in sorted(class_dir.glob("*.png")):
                self.items.append((img_path, c))
        if not self.items:
            raise RuntimeError(f"No class_*/*.png found under {root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def make_loader(dataset, batch_size, num_workers):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      shuffle=False, pin_memory=True)


def _per_class_acc(per_class_dict):
    """Convert {class: [correct, total]} -> {class: acc}, dropping empty classes."""
    return {k: v[0] / v[1] for k, v in per_class_dict.items() if v[1] > 0}


@torch.no_grad()
def eval_resnet50(root, batch_size=128, num_workers=4, device="cuda"):
    import timm
    print("[eval] loading timm/resnet50.a1_in1k ...")
    model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=1000)
    model = model.to(device).eval()

    # standard ImageNet preprocess
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = GeneratedDataset(root, transform=transform)
    loader = make_loader(ds, batch_size, num_workers)

    correct1, correct5, total = 0, 0, 0
    per_class = {}

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        top5 = logits.topk(5, dim=-1).indices
        c1 = (top5[:, 0] == labels)
        c5 = (top5 == labels.unsqueeze(1)).any(dim=1)
        correct1 += c1.sum().item()
        correct5 += c5.sum().item()
        total += labels.size(0)
        for l in labels.unique():
            k = int(l.item())
            mask = labels == l
            per_class.setdefault(k, [0, 0])
            per_class[k][0] += c1[mask].sum().item()
            per_class[k][1] += mask.sum().item()

    return {
        "top1": correct1 / total,
        "top5": correct5 / total,
        "n": total,
        "per_class_top1": _per_class_acc(per_class),
        "model": "timm/resnet50.a1_in1k",
    }


@torch.no_grad()
def eval_held_out_clip(root, batch_size=128, num_workers=4, device="cuda",
                       clip_model="ViT-B-32", clip_pretrained="openai"):
    import open_clip
    from open_clip import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES

    print(f"[eval] loading open_clip {clip_model} ({clip_pretrained}) ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained
    )
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer(clip_model)

    # Build text-embedding table for the 1000 ImageNet classes,
    # mirroring the OpenAI 80-template ensemble used in training.
    print(f"[eval] building text embeddings for {len(IMAGENET_CLASSNAMES)} classes ...")
    text_feats = []
    for cname in IMAGENET_CLASSNAMES:
        texts = [t(cname) for t in OPENAI_IMAGENET_TEMPLATES]
        tokens = tok(texts).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        text_feats.append(emb.mean(0))
    text_feats = torch.stack(text_feats)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    ds = GeneratedDataset(root, transform=preprocess)
    loader = make_loader(ds, batch_size, num_workers)

    sum_logp = 0.0
    correct1, correct5, total = 0, 0, 0
    per_class = {}
    per_class_logp = {}
    logit_scale = model.logit_scale.exp()

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        feats = model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = logit_scale * feats @ text_feats.t()
        log_probs = F.log_softmax(logits, dim=-1)

        # log p(c|x) at the conditioning label — the same quantity the
        # training objective optimizes (but here we use a HELD-OUT CLIP).
        logp_c = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        sum_logp += logp_c.sum().item()

        top5 = log_probs.topk(5, dim=-1).indices
        c1 = (top5[:, 0] == labels)
        c5 = (top5 == labels.unsqueeze(1)).any(dim=1)
        correct1 += c1.sum().item()
        correct5 += c5.sum().item()
        total += labels.size(0)

        for l in labels.unique():
            k = int(l.item())
            mask = labels == l
            per_class.setdefault(k, [0, 0])
            per_class[k][0] += c1[mask].sum().item()
            per_class[k][1] += mask.sum().item()
            per_class_logp.setdefault(k, [0.0, 0])
            per_class_logp[k][0] += logp_c[mask].sum().item()
            per_class_logp[k][1] += mask.sum().item()

    return {
        "top1": correct1 / total,
        "top5": correct5 / total,
        "mean_logp_c": sum_logp / total,
        "n": total,
        "per_class_top1": _per_class_acc(per_class),
        "per_class_logp": {k: v[0] / v[1] for k, v in per_class_logp.items() if v[1] > 0},
        "clip_model": clip_model,
        "clip_pretrained": clip_pretrained,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder produced by generate_samples.py")
    ap.add_argument("--tag", required=True, help="short label, e.g. 'baseline' or 'cond'")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--clip_model", type=str, default="ViT-B-32",
                    help="held-out CLIP backbone (must differ from training's --clip_model_name)")
    ap.add_argument("--clip_pretrained", type=str, default="openai")
    ap.add_argument("--skip_resnet50", action="store_true")
    ap.add_argument("--skip_clip", action="store_true")
    args = ap.parse_args()

    out_json = Path(args.dir) / "eval_metrics.json"
    metrics = {"tag": args.tag, "dir": str(args.dir)}

    if not args.skip_resnet50:
        print(f"\n[{args.tag}] === ResNet-50 (held-out architecture) ===")
        m = eval_resnet50(args.dir, args.batch_size, args.num_workers)
        print(f"  top1={m['top1']:.4f}  top5={m['top5']:.4f}  n={m['n']}")
        metrics["resnet50_in1k"] = m

    if not args.skip_clip:
        print(f"\n[{args.tag}] === CLIP {args.clip_model} (held-out CLIP) ===")
        m = eval_held_out_clip(args.dir, args.batch_size, args.num_workers,
                               clip_model=args.clip_model,
                               clip_pretrained=args.clip_pretrained)
        print(f"  top1={m['top1']:.4f}  top5={m['top5']:.4f}  "
              f"mean_logp_c={m['mean_logp_c']:.4f}  n={m['n']}")
        metrics["held_out_clip"] = m

    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nsaved {out_json}")


if __name__ == "__main__":
    main()
