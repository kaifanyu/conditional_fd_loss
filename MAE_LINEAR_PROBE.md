# Frozen MAE linear probe for conditional FD training

## What is pretrained

Do not pretrain MAE again. `vit_large_patch16_224.mae` is already a pretrained
self-supervised backbone. The offline step trains only a supervised ImageNet
head:

```text
image -> frozen MAE -> 1024-D CLS feature -> BN + Linear(1024, 1000)
```

The head is trained on real ImageNet `(image, class)` pairs. The generated-image
run then freezes both MAE and the head and minimizes:

```text
L_cond = -mean(log_softmax(head(MAE(image)))[requested_label])
```

MAE may be the only classifier inside the loss for this ablation. It must not
be the only evaluator: keep the existing ResNet-50 probe completely held out.

## 1. Train the head

```bash
bash scripts/train_mae_linear_probe.sh
```

Equivalent explicit command:

```bash
CUDA_VISIBLE_DEVICES=4,5 \
/home/nvidia/miniconda3/envs/fdloss/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 --master_port=29514 \
  train_mae_linear_probe.py \
  --data_path /data/dataset/imagenet \
  --output_dir work_dirs/mae_probe_vitl_cls \
  --model_name vit_large_patch16_224.mae \
  --pool_type cls --target_size 224 --head_norm bn \
  --batch_size 128 --epochs 90 --warmup_epochs 10 \
  --base_lr 0.1 --optimizer lars --weight_decay 0 --dtype bf16
```

This writes `last.pt` and the best real-validation checkpoint `best.pt`. The
checkpoint contains the small head, optimizer state, exact class mapping, MAE
model/pool/resolution metadata, and validation accuracy. It does not duplicate
the pretrained MAE weights.

Recheck validation at any time:

```bash
/home/nvidia/miniconda3/envs/fdloss/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=1 train_mae_linear_probe.py \
  --data_path /data/dataset/imagenet --eval_only \
  --resume work_dirs/mae_probe_vitl_cls/best.pt
```

Do not continue to JiT training unless real ImageNet validation accuracy is
strong and the class mapping matches. Training a head on generated requested
labels would be circular and is intentionally not supported.

## 2. Run a short JiT pilot

```bash
bash scripts/run_jit_mae_probe.sh
```

The new entrypoint is `conditional_main_fd_mae.py`; the original
`conditional_main_fd_ponly.py` is unchanged. The copy requires the probe
checkpoint whenever `--lambda_cond` is positive.

The copied main finds the exact MAE FD judge described by the checkpoint and
reuses its local 1024-D features. With `--mae_eot_views 1`, MAE runs once per
batch for both FD and log-p. EOT values above one require extra augmented MAE
forwards and set `cond_feature_reuse=0`.

The pilot defaults are intentionally short. Do not inherit the Qwen VLM
coefficient blindly: MAE log-probabilities have a different gradient scale.
Sweep `LAMBDA_MAE`, for example `1e-6`, `3e-6`, and `1e-5`, and inspect the
measured gradient ratio.

```bash
LAMBDA_MAE=1e-6 EXP_NAME=r6_mae_lam1e6 bash scripts/run_jit_mae_probe.sh
```

## Metrics that decide whether it works

- `logp_mae_probe` and `mae_probe_top1`: training-judge progress.
- `probe_rank`: held-out requested-class rank; chance is about 500.5. This
  must decrease consistently.
- `probe_top1` / `probe_top5`: held-out semantic transfer.
- `cond_delta`: labels affect pixels, but does not by itself prove semantics.
- `grad_ratio_p_fd`: conditional-to-realism image-gradient strength.
- `fid_siglip`, `fid_mae`, and external Inception FID: realism/diversity guard.
- `cond_feature_reuse`: should be `1` for the default clean-view configuration.

Kill the pilot if MAE log-p improves while held-out ResNet rank remains near
500. That means the generator is exploiting the MAE decision boundary rather
than learning transferable class content.

Finally, this is supervised: the MAE backbone is self-supervised, but the
linear head uses real ImageNet labels. If the research claim is strictly
data-free or label-free conditioning, this probe changes that assumption and
must be described as an external supervised semantic oracle.
