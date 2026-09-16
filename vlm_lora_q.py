"""Online Qwen q: one frozen backbone and student/EMA LoRA banks.

Native PyTorch LoRA keeps adapter selection explicit for the image VJP. No
weights are merged into the base model and no PEFT dependency is required.
"""

from contextlib import contextmanager
import math
import re

import torch
import torch.nn.functional as F

from vlm_linear_heads import head_metrics, pq_agreement_metrics


def validate_loss_scale(scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("loss scale must be finite and positive")
    return float(scale)


class BankedLoRALinear(torch.nn.Module):
    """W x + (alpha/r) B A x; B=0 makes both q banks equal the base at init."""

    def __init__(self, base, rank, alpha, *, parameter_dtype=torch.float32):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if parameter_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("LoRA parameter_dtype must be torch.float32 or torch.bfloat16")
        self.base = base.requires_grad_(False)
        self.scaling = float(alpha) / rank
        self.active = "base"
        self.student_a = torch.nn.Parameter(torch.empty(
            rank, base.in_features, device=base.weight.device, dtype=parameter_dtype))
        self.student_b = torch.nn.Parameter(torch.zeros(
            base.out_features, rank, device=base.weight.device, dtype=parameter_dtype))
        torch.nn.init.kaiming_uniform_(self.student_a, a=math.sqrt(5))
        self.teacher_a = torch.nn.Parameter(self.student_a.detach().clone(), requires_grad=False)
        self.teacher_b = torch.nn.Parameter(self.student_b.detach().clone(), requires_grad=False)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        out = self.base(x)
        if self.active == "base":
            return out
        a, b = ((self.student_a, self.student_b) if self.active in ("student", "student_frozen")
                else (self.teacher_a, self.teacher_b))
        if self.active == "student_frozen":
            a, b = a.detach(), b.detach()
        # Adapter storage determines its compute precision independently of the
        # backbone's autocast. BF16 mode has no FP32 master parameter copy.
        with torch.autocast(device_type=x.device.type, enabled=False):
            update = F.linear(F.linear(x.to(dtype=a.dtype), a), b) * self.scaling
        return out + update.to(out.dtype)


class LoRAQ:
    """Own adapter operations, leaving the extractor/head ownership unchanged."""

    def __init__(self, extractor, heads, *, rank=8, alpha=16.0,
                 scope="both", targets=("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "proj"),
                 parameter_dtype=torch.float32):
        if scope not in ("language", "vision", "both"):
            raise ValueError("LoRA scope must be language, vision, or both")
        self.extractor = extractor
        self.heads = heads
        self.layers = {}
        extractor.vlm.requires_grad_(False)
        for name, module in list(extractor.vlm.named_modules()):
            if not isinstance(module, torch.nn.Linear) or name.split(".")[-1] not in targets:
                continue
            vision = "visual" in name.split(".")
            text_layer = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            # Decoder blocks beyond h[layer] cannot affect this classifier.
            language = text_layer is not None and not vision and int(text_layer[1]) < extractor.layer
            if not ((vision and scope in ("vision", "both"))
                    or (language and scope in ("language", "both"))):
                continue
            parent_name, _, child_name = name.rpartition(".")
            parent = extractor.vlm.get_submodule(parent_name) if parent_name else extractor.vlm
            layer = BankedLoRALinear(module, rank, alpha, parameter_dtype=parameter_dtype)
            setattr(parent, child_name, layer)
            self.layers[name] = layer
        if not self.layers:
            raise ValueError("No active Qwen Linear modules matched the LoRA scope/targets/layer")
        self.config = {"rank": rank, "alpha": float(alpha), "scope": scope,
                       "targets": list(targets), "modules": list(self.layers),
                       "dtype": str(extractor.compute_dtype), "format_version": 1}
        # Preserve the legacy FP32 payload while recording a BF16 experiment's
        # actual parameter precision separately from backbone compute precision.
        if parameter_dtype != torch.float32:
            self.config["parameter_dtype"] = str(parameter_dtype)
        # setup() deliberately uses seed+rank. Random A must still be identical
        # across ranks: averaged gradients alone cannot synchronize unequal A.
        if torch.distributed.is_initialized():
            for p in self.adapter_parameters(include_teacher=True):
                torch.distributed.broadcast(p.data, src=0)

    def adapter_parameters(self, include_teacher=False):
        for layer in self.layers.values():
            yield layer.student_a
            yield layer.student_b
            if include_teacher:
                yield layer.teacher_a
                yield layer.teacher_b

    def trainable_parameters(self):
        return list(self.heads.q_student.parameters()) + list(self.adapter_parameters())

    @contextmanager
    def use(self, bank):
        if bank not in ("base", "student", "student_frozen", "teacher"):
            raise ValueError(f"Unknown adapter bank {bank!r}")
        previous = [layer.active for layer in self.layers.values()]
        try:
            for layer in self.layers.values():
                layer.active = bank
            yield
        finally:
            for layer, active in zip(self.layers.values(), previous):
                layer.active = active

    def features(self, images, bank):
        with self.use(bank):
            return self.extractor.answer_states(images)[self.extractor.layer]

    @property
    def generator_bank(self):
        return "teacher" if self.heads.use_ema else "student_frozen"

    @torch.no_grad()
    def log_probs(self, images, bank, microbatch):
        head = {"base": self.heads.p_head, "student": self.heads.q_student,
                "student_frozen": self.heads.q_student, "teacher": self.heads.q_teacher}[bank]
        chunks = []
        for x in images.split(microbatch):
            z = self.features(x, bank)
            chunks.append(head.log_probs(z, self.heads.temperature))
        return torch.cat(chunks)

    @torch.no_grad()
    def ema_update(self):
        if not self.heads.use_ema:
            return
        beta = self.heads.ema_beta
        for layer in self.layers.values():
            layer.teacher_a.lerp_(layer.student_a, 1 - beta)
            layer.teacher_b.lerp_(layer.student_b, 1 - beta)

    def state_dict(self):
        return {"config": dict(self.config),
                "sample_cursor": int(self.extractor._sample_cursor.item()), "adapters": {
            name: {key: getattr(layer, key).detach().cpu().clone()
                   for key in ("student_a", "student_b", "teacher_a", "teacher_b")}
            for name, layer in self.layers.items()}}

    @torch.no_grad()
    def load_state_dict(self, state):
        saved_config, current_config = dict(state["config"]), dict(self.config)
        saved_config.setdefault("parameter_dtype", str(torch.float32))
        current_config.setdefault("parameter_dtype", str(torch.float32))
        if saved_config != current_config:
            raise ValueError("Checkpoint LoRA configuration differs from this run")
        if state["adapters"].keys() != self.layers.keys():
            raise ValueError("Checkpoint LoRA module names differ from this run")
        self.extractor._sample_cursor.fill_(int(state["sample_cursor"]))
        for name, layer in self.layers.items():
            saved = state["adapters"][name]
            if set(saved) != {"student_a", "student_b", "teacher_a", "teacher_b"}:
                raise ValueError(f"Incomplete adapter checkpoint for {name}")
            for key, value in saved.items():
                target = getattr(layer, key)
                if target.shape != value.shape:
                    raise ValueError(f"LoRA shape mismatch for {name}.{key}")
                if target.dtype != value.dtype:
                    raise ValueError(f"LoRA parameter dtype mismatch for {name}.{key}")
                target.copy_(value)

    @torch.no_grad()
    def drift_metrics(self):
        out = {}
        for bank in (("student", "teacher") if self.heads.use_ema else ("student",)):
            # ||B A||_F^2 = tr((B^T B)(A A^T)); never materialize dense delta W.
            norm_sq = 0.0
            for layer in self.layers.values():
                a = getattr(layer, bank + "_a").float()
                b = getattr(layer, bank + "_b").float()
                norm_sq += float(((b.T @ b) * (a @ a.T)).sum()) * layer.scaling ** 2
            out[f"q_{bank}_lora_delta_l2"] = math.sqrt(max(0.0, norm_sq))
        return out

    def generator_surrogate(self, images, labels, *, denominator, clamp=0.0,
                            loss_scale=1.0, need_grad=True):
        """Two sequential VLM passes: frozen p and q with detached parameters.

        Returns local surrogate, p features, selected q log probabilities, indices.
        Each graph is released before the next branch. Clamp masks are applied
        to per-image VJPs (Qwen has no cross-image attention/normalization).
        """
        scale = validate_loss_scale(loss_scale)
        q_bank = self.generator_bank
        selected = self.extractor.select_indices(images.shape[0], images.device)
        surrogate = images.new_zeros((), dtype=torch.float32)
        p_features, q_log_probs = [], []
        for indices in selected.split(self.extractor.microbatch_size):
            y = labels.index_select(0, indices).long()
            values, grads = {}, {}
            for bank, head in (("base", self.heads.p_head), (q_bank, self.heads.generator_head)):
                leaf = images.index_select(0, indices).detach().float().requires_grad_(need_grad)
                with torch.set_grad_enabled(need_grad), self.use(bank):
                    z = self.extractor.answer_states(leaf)[self.extractor.layer]
                    lp = head.log_probs(z, self.heads.temperature, detach_parameters=True)
                    target = lp.gather(1, y[:, None]).squeeze(1)
                    if need_grad:
                        grad = torch.autograd.grad(target.sum() * (scale / denominator), leaf)[0]
                        grads[bank] = grad.detach().float() / scale
                    values[bank] = target.detach()
                    if bank == "base":
                        p_features.append(z.detach())
                    else:
                        q_log_probs.append(lp.detach())
                    del z, lp, target
            delta = values[q_bank] - values["base"]
            mask = (delta.abs() <= clamp) if clamp > 0 else torch.ones_like(delta, dtype=torch.bool)
            value = delta.clamp(-clamp, clamp) if clamp > 0 else delta
            surrogate = surrogate + value.sum() / denominator
            if need_grad:
                grad = (grads[q_bank] - grads["base"]) * mask[:, None, None, None]
                if not torch.isfinite(grad).all():
                    raise RuntimeError("Non-finite LoRA p/q image VJP; reduce --vlm_vjp_loss_scale or use fp32")
                original = images.index_select(0, indices).float()
                surrogate = surrogate + ((original - original.detach()) * grad).sum()
        z = torch.cat(p_features)
        self.extractor.last_stats = {
            "vlm_answer_state_samples": float(selected.numel()),
            "vlm_answer_state_microbatch": float(self.extractor.microbatch_size),
            "vlm_answer_state_norm": float(z.float().norm(dim=1).mean()),
        }
        return surrogate, z, torch.cat(q_log_probs), selected


def build_lora_optimizer(lora, args):
    groups = [
        {"params": list(lora.heads.q_student.parameters()), "lr": args.vlm_q_lr,
         "weight_decay": args.vlm_q_weight_decay},
        {"params": list(lora.adapter_parameters()), "lr": args.vlm_q_lora_lr,
         "weight_decay": args.vlm_q_lora_weight_decay},
    ]
    if args.vlm_q_optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=args.vlm_q_momentum)
    return torch.optim.AdamW(groups, betas=(args.vlm_q_beta1, args.vlm_q_beta2))


def load_q_optimizer_state(optimizer, state):
    """Do not silently restore old learning rates/betas over the requested setup."""
    saved_groups = state["param_groups"]
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("checkpoint q optimizer parameter groups differ from this run")
    for current, saved in zip(optimizer.param_groups, saved_groups):
        for key in ("lr", "weight_decay", "betas", "momentum"):
            if current.get(key) != saved.get(key):
                raise ValueError(f"checkpoint q optimizer {key}={saved.get(key)!r} differs "
                                 f"from configured {current.get(key)!r}; use --load_from "
                                 "for a new experiment or match the resume configuration")
        for parameter, saved_id in zip(current["params"], saved["params"]):
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"):
                value = state["state"].get(saved_id, {}).get(key)
                if value is not None and value.dtype != parameter.dtype:
                    raise ValueError(f"checkpoint q optimizer {key} dtype {value.dtype} differs "
                                     f"from parameter dtype {parameter.dtype}; use --load_from "
                                     "for a new precision experiment")
    optimizer.load_state_dict(state)


def q_lora_update_step(lora, optimizer, images, labels, args, collect_metrics=False):
    """CE on this rank's fresh detached images; average parameter grads across ranks."""
    images, labels = images.detach().float(), labels.detach().long()
    scale = validate_loss_scale(args.vlm_q_loss_scale)
    params = lora.trainable_parameters()
    heads = lora.heads
    before = heads.student_snapshot()
    out = {}
    ce_sum, acc_sum, gn_sum = 0.0, 0.0, 0.0
    updates = max(0, args.vlm_q_updates_per_step)
    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        ce, accuracy = 0.0, 0.0
        pre_lp = []
        for start in range(0, images.shape[0], args.vlm_microbatch):
            x, y = images[start:start + args.vlm_microbatch], labels[start:start + args.vlm_microbatch]
            # Eval mode disables dropout, but does not disable autograd.
            with lora.use("student"):
                z = lora.extractor.answer_states(x)[lora.extractor.layer]
                logits = heads.q_student.logits(z, heads.temperature)
                loss = F.cross_entropy(logits, y, reduction="sum") / images.shape[0]
                (loss * scale).backward()
            ce += float(loss.detach())
            accuracy += float((logits.argmax(-1) == y).sum()) / images.shape[0]
            if collect_metrics and update == 0:
                pre_lp.append(logits.detach().log_softmax(-1))
            del z, logits, loss
        for p in params:
            # Always participate, even if a parameter is unused on a rank.
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.div_(scale)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(p.grad)
                p.grad.div_(torch.distributed.get_world_size())
        norm = torch.nn.utils.clip_grad_norm_(
            params, args.vlm_q_grad_clip if args.vlm_q_grad_clip > 0 else float("inf"))
        if not torch.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("Non-finite q LoRA gradient; reduce --vlm_q_loss_scale or use fp32")
        optimizer.step()
        ce_sum += ce
        acc_sum += accuracy
        gn_sum += float(norm)
        if collect_metrics and update == 0:
            pre = head_metrics(torch.cat(pre_lp), labels, "vlm_q_student")
            out.update({f"{k}_pre": v for k, v in pre.items()})
    optimizer.zero_grad(set_to_none=True)
    if updates:
        heads.record_student_update(before)
        heads.q_train_steps.add_(updates)
        heads.ema_update()
        lora.ema_update()
        out.update(q_ce=ce_sum / updates, q_train_accuracy=acc_sum / updates,
                   q_grad_norm=gn_sum / updates, q_updates_applied=float(updates),
                   q_lr=float(optimizer.param_groups[0]["lr"]),
                   q_lora_lr=float(optimizer.param_groups[1]["lr"]))
    if collect_metrics:
        student = lora.log_probs(images, "student", args.vlm_microbatch)
        post = head_metrics(student, labels, "vlm_q_student")
        out.update(post)
        out.update({f"{k}_post": v for k, v in post.items()})
        if heads.use_ema:
            teacher = lora.log_probs(images, "teacher", args.vlm_microbatch)
            out.update(pq_agreement_metrics(student, teacher, labels, prefix="q_student_teacher"))
        out.update(heads.drift_metrics())
        out.update(lora.drift_metrics())
        out["q_fresh_batch_ce_pre"] = out.get("vlm_q_student_ce_pre", post["vlm_q_student_ce"])
        out["q_train_images_per_rank"] = float(images.shape[0])
    return out
