"""External-classifier conditioning eval — the honest test of conditional FD.

FD-loss can fall while the generator ignores the class label c. The only honest
check is whether an *external* classifier (NOT CLIP — CLIP supplies the training
gradient, so judging with it would be circular) recognises the requested class
in the generated images.

This loads a trained generator, generates `--eval_per_class` images for each
requested class, classifies them with a torchvision ResNet50 (IMAGENET1K_V2,
80.8% top-1), and reports top-1 / top-5 accuracy. High accuracy => the generator
actually uses c. Accuracy near 0.1% (= 1/1000) => conditioning is broken,
regardless of how good FD looks.

Single GPU:
    python eval_class_accuracy.py --model JiT_H --img_size 256 \
        --resume_from checkpoints/.../step_XXXXXXX.pth \
        --cfg 1.0 --num_sampling_steps 50 --eval_per_class 50

Quick smoke (8 classes, 16 imgs each):
    python eval_class_accuracy.py --model JiT_H --resume_from <ckpt> \
        --eval_classes 207 360 387 974 88 979 417 279 --eval_per_class 16
"""

import argparse
import logging
import sys

import torch
import torch.nn.functional as F

from conditional_main_fd_ponly import get_args_parser
from utils.builders import create_generation_model, create_tokenizer
from utils.sampling_util import generate_images

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("FD_loss")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def add_eval_args(parser):
    parser.add_argument("--eval_per_class", type=int, default=50,
                        help="images generated per class")
    parser.add_argument("--eval_classes", type=int, nargs="+", default=None,
                        help="class subset to evaluate (default: all num_classes)")
    parser.add_argument("--eval_ema_label", type=str, default=None,
                        help="EMA label to swap in before eval (default: raw model)")
    parser.add_argument("--ext_classifier", type=str, default="resnet50",
                        choices=["resnet50", "vit_b_16"],
                        help="external torchvision classifier (not CLIP)")
    return parser


def _init_single_process(args):
    """Set the derived attrs the builders/sampler expect, without distributed."""
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    args.global_bsz = args.batch_size
    args.enable_amp = args.dtype != "fp32"
    args.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                      "fp32": torch.float32}[args.dtype]


def load_generator(args):
    tokenizer = create_tokenizer(args)
    model, ema_model = create_generation_model(args)
    ckpt_path = args.resume_from or args.load_from
    if ckpt_path is None:
        raise ValueError("pass --resume_from or --load_from with a checkpoint path")
    logger.info(f"loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    msg = model.load_state_dict(state, strict=False)
    logger.info(f"model load: missing={len(msg.missing_keys)} "
                f"unexpected={len(msg.unexpected_keys)}")
    if args.eval_ema_label is not None:
        if ckpt.get("model_ema") is None:
            raise ValueError("checkpoint has no model_ema but --eval_ema_label was set")
        ema_model.load_state_dict(ckpt["model_ema"])
        logger.info(f"will evaluate EMA label={args.eval_ema_label} "
                    f"(available: {ema_model.labels})")
    model.eval().cuda()
    return model, ema_model, tokenizer


def build_external_classifier(name):
    import torchvision
    if name == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        net = torchvision.models.resnet50(weights=weights)
        resize, crop = 232, 224
    else:  # vit_b_16
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        net = torchvision.models.vit_b_16(weights=weights)
        resize, crop = 256, 224
    net = net.eval().cuda().requires_grad_(False)
    logger.info(f"[ext] {name} ({weights}) — top-1 ref ~"
                f"{weights.meta.get('_metrics', {}).get('ImageNet-1K', {}).get('acc@1', '?')}%")
    return net, resize, crop


def _preprocess_for_classifier(imgs, resize, crop, mean, std):
    """imgs in [0,1], NCHW -> resize -> center crop -> ImageNet normalize."""
    imgs = F.interpolate(imgs, size=(resize, resize), mode="bilinear",
                         align_corners=False, antialias=True)
    top = (resize - crop) // 2
    imgs = imgs[:, :, top:top + crop, top:top + crop]
    return (imgs - mean) / std


@torch.inference_mode()
def evaluate(args):
    _init_single_process(args)
    model, ema_model, tokenizer = load_generator(args)
    clf, resize, crop = build_external_classifier(args.ext_classifier)
    mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)

    classes = args.eval_classes if args.eval_classes is not None else list(range(args.num_classes))
    bsz = args.eval_bsz

    def _run():
        total = top1 = top5 = 0
        per_class = {}
        for ci, c in enumerate(classes):
            c_correct = c_total = 0
            remaining = args.eval_per_class
            while remaining > 0:
                n = min(bsz, remaining)
                y = torch.full((n,), c, dtype=torch.long, device="cuda")
                imgs = generate_images(args, model, labels=y, cfg=args.cfg,
                                       tokenizer=tokenizer)           # [0,1] NCHW
                x = _preprocess_for_classifier(imgs, resize, crop, mean, std)
                logits = clf(x)
                top5_idx = logits.topk(5, dim=1).indices               # (n,5)
                hit1 = (top5_idx[:, 0] == c)
                hit5 = (top5_idx == c).any(dim=1)
                top1 += int(hit1.sum()); top5 += int(hit5.sum())
                c_correct += int(hit1.sum()); c_total += n; total += n
                remaining -= n
            per_class[c] = c_correct / max(c_total, 1)
            if (ci + 1) % 50 == 0 or len(classes) <= 16:
                logger.info(f"  class {c}: top1={per_class[c]*100:.1f}%  "
                            f"[{ci+1}/{len(classes)} classes, running "
                            f"top1={top1/total*100:.2f}%]")
        return total, top1, top5, per_class

    if args.eval_ema_label is not None:
        with ema_model.swap(model, label=args.eval_ema_label):
            total, top1, top5, per_class = _run()
    else:
        total, top1, top5, per_class = _run()

    logger.info("=" * 60)
    logger.info(f"External-classifier conditioning accuracy ({args.ext_classifier}, "
                f"cfg={args.cfg}, steps={args.num_sampling_steps})")
    logger.info(f"  classes evaluated : {len(classes)} "
                f"x {args.eval_per_class} imgs = {total}")
    logger.info(f"  top-1 : {top1/total*100:.2f}%   (chance = {100/args.num_classes:.2f}%)")
    logger.info(f"  top-5 : {top5/total*100:.2f}%")
    if len(classes) <= 16:
        for c in classes:
            logger.info(f"    class {c}: {per_class[c]*100:.1f}%")
    logger.info("=" * 60)
    return top1 / total


if __name__ == "__main__":
    parser = add_eval_args(get_args_parser())
    args = parser.parse_args()
    sys.exit(0 if evaluate(args) is not None else 1)
