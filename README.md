## Representation Fréchet Loss for Visual Generation

[![arXiv](https://img.shields.io/badge/arXiv-2604.28190-b31b1b.svg)](https://arxiv.org/abs/2604.28190)
[![Hugging Face checkpoints](https://img.shields.io/badge/HuggingFace-checkpoints-ffcc00.svg)](https://huggingface.co/jjiaweiyang/FD-Loss)

<p align="center">
  <img src="docs/fd_loss_dynamic.svg" width="760" alt="FD-Loss training dynamics">
</p>

This is a PyTorch/GPU implementation of the paper:
[Representation Fréchet Loss for Visual Generation](https://arxiv.org/abs/2604.28190).

```bibtex
@article{yang2026fdloss,
  title={Representation Fréchet Loss for Visual Generation},
  author={Yang, Jiawei and Geng, Zhengyang and Ju, Xuan and Tian, Yonglong and Wang, Yue},
  journal={arXiv:2604.28190},
  url={https://arxiv.org/abs/2604.28190},
  year={2026}
}
```

FD-Loss post-trains visual generators by matching generated-image feature
distributions to real-image feature distributions in frozen representation spaces.
This repository includes training, released-checkpoint evaluation, reference
statistics utilities, and scripts for the ImageNet experiments.

<p align="center">
  <img src="docs/visual.png" width="760" alt="FD-Loss visual overview">
</p>

### Dataset

Download ImageNet and place it in your `DATA_ROOT` using the standard
`ImageFolder` layout:

```bash
export DATA_ROOT=/path/to/imagenet
```

### Installation

Download the code:

```bash
git clone https://github.com/Jiawei-Yang/FD-Loss.git
cd FD-Loss
```

Create and activate a conda environment:

```bash
conda create -n fdloss python=3.11 -y
conda activate fdloss

pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -U huggingface_hub
```

### Checkpoints And Statistics

Released checkpoints and data files are hosted on
[Hugging Face](https://huggingface.co/jjiaweiyang/FD-Loss).

```bash
hf download jjiaweiyang/FD-Loss \
  --local-dir . \
  --include "checkpoints/**/*.pth" \
  --include "data/**"

python scripts/extract_paper_ref_stats.py
```

See [scripts/README.md](scripts/README.md) for the asset layout and lighter
download options.

### Evaluation

Evaluate the released FD-SIM models:

```bash
PRESET=pMF_H_256 \
CKPT_PATH=checkpoints/post-trained/pMF-H_FD-SIM.pth \
GPUS_PER_NODE=8 \
bash scripts/evaluate_released_ckpt.sh

PRESET=JiT_H \
CKPT_PATH=checkpoints/post-trained/JiT-H_FD-SIM.pth \
GPUS_PER_NODE=8 \
bash scripts/evaluate_released_ckpt.sh

PRESET=iMF_XL \
CKPT_PATH=checkpoints/post-trained/iMF-XL_FD-SIM.pth \
GPUS_PER_NODE=8 \
bash scripts/evaluate_released_ckpt.sh
```

Additional presets and smoke-test settings are documented in
[scripts/README.md](scripts/README.md).

### Training

Training starts from the released base checkpoints:

```bash
export CKPT_ROOT=./checkpoints/base
```

The experiment scripts under [scripts/](scripts/) reproduce the Table 1
ablations, Table 2 JiT repurposing, and Table 3 scalability runs. For example:

```bash
bash scripts/table_1a_queue_size.sh
bash scripts/table_2_repurpose_jit_L.sh
MODEL_SIZE=L RES=256 bash scripts/table_3_pMF.sh
MODEL_SIZE=XL bash scripts/table_3_iMF.sh
MODEL_SIZE=H bash scripts/table_3_JiT.sh
```

### Conditional FD-Loss (CLIP correction term)

The base FD-loss matches the **marginal** feature distribution, so it gives the
class label `c` no gradient: a generator trained on `L_FD` alone has no incentive
to use `c`. `conditional_main_fd.py` adds the classifier-guidance half of the
Bayes decomposition

```
∇_x log p(x|c) - ∇_x log q(x|c)
  = [∇_x log p(x)   - ∇_x log q(x)]     # marginal  -> handled by L_FD
  + [∇_x log p(c|x) - ∇_x log q(c|x)]   # conditional correction
```

as an **additive** term. We keep only the one-sided `p(c|x)` half (the `q(c|x)`
side is dropped, as in classifier guidance / NP-Edit) and approximate it with a
frozen **CLIP** zero-shot classifier:

```
L_total = L_FD + lambda_cond * L_cond,   L_cond = -E[ log p(c|x_gen) ]
log p(c_k|x) ≈ log softmax_k( s · <φ_img(x), φ_text(c_k)> )
```

The gradient flows back through CLIP's image encoder into the generator. `L_FD`
is **unchanged**; the generator must already be class-conditional (the JiT
backbone is, via `LabelEmbedder` + AdaLN — the same `y` is reused, no new
conditioning machinery is added).

CLIP needs the text tower, which the repo's timm wrapper does not expose, so we
use `open_clip` (already in `requirements.txt`):

```bash
pip install open_clip_torch
```

**Warm the text-embedding cache** (optional — training builds it lazily on first
run; this also runs the near-uniform smoke test):

```bash
python precompute_text_embeddings.py            # ViT-L-14, 80-prompt ensemble
```

**Train.** Same launch as `main_fd.py`, but run `conditional_main_fd.py` and add
the conditional args (verify gradient flow on the first run with
`--clip_grad_check`):

```bash
torchrun --nproc_per_node=8 conditional_main_fd.py \
    --model JiT_H --img_size 256 \
    --resume_from checkpoints/base/<base>.pth \
    --fd_repr_models inception --fd_eigvalsh \
    --lambda_cond 0.1 \
    --clip_model_name ViT-L-14 --clip_pretrained openai --clip_dtype bf16 \
    --clip_grad_check \
    --enable_wandb --exp_name jit_h_cond
```

`L_FD` (logged as `fid_<judge>`) and the conditional term (`l_cond`, and the raw
`logp_c`) are logged separately to the metric file and wandb so you can see
whether the conditional signal is moving. `logp_c` should start near
`-log(1000) ≈ -6.9` and rise toward 0 as conditioning kicks in.

**Honest eval (external classifier — *not* CLIP).** FD can fall while
conditioning is broken; the real test is a separate ImageNet classifier:

```bash
python eval_class_accuracy.py --model JiT_H --img_size 256 \
    --resume_from work_dirs/<exp>/checkpoints/step_XXXXXXX.pth \
    --cfg 1.0 --num_sampling_steps 50 --eval_per_class 50
# quick 8-class smoke:
python eval_class_accuracy.py --model JiT_H --resume_from <ckpt> \
    --eval_classes 207 360 387 974 88 979 417 279 --eval_per_class 16
```

Top-1 well above chance (`0.1%`) means the generator uses `c`.

**Hyperparameters to sweep:**

| arg | default | what it does / how to tune |
| --- | --- | --- |
| `--lambda_cond` | `0.1` | **most important, NEEDS TUNING.** `L_FD` terms are normalized to ~1.0 each; `L_cond` starts ~6.9. Too high → realism (FD) degrades and images look adversarial; too low → label ignored. Sweep `{0.02, 0.05, 0.1, 0.3, 1.0}` and watch FD **and** external top-1 together. |
| `--clip_model_name` | `ViT-L-14` | Signal quality is bounded by CLIP zero-shot accuracy (ViT-L/14 ≈ 75.5%, ViT-B/32 ≈ 63.3%). Drop to `ViT-B-32` only if memory-bound or for smoke tests. |
| `--clip_single_template` | off | 80-prompt ensemble (default) gives lower-variance class embeddings (≈+1.5% acc). The single `"a photo of a {c}."` prompt is faster to cache. |
| `--clip_dtype` | `bf16` | bf16 keeps the frozen CLIP cheap; use `fp32` only to debug numerics. |
| `--num_sampling_steps` | `50` | `L_cond` backprops through every step, so cost scales with it. Few-step (1–4) is much cheaper for the correction; raise for final quality. |
| `--cfg` (sampling) | — | the conditional term trains a clean conditional model; CFG at sample time is intentionally left out (kept as a follow-up). |

### License

This project is released under the MIT license. See [LICENSE](LICENSE) for details.

If you have any questions, feel free to contact me through email
([yangjiaw@usc.edu](mailto:yangjiaw@usc.edu)).
