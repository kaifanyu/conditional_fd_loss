"""Rectified-Flow (score_sde NCSN++) denoiser wrapper for FD-Loss conditional
distillation.

The base checkpoint (`checkpoints/base/cifar_10_base.pth`) is the *unconditional*
1-rectified-flow CIFAR-10 model from gnobitab/RectifiedFlow — score_sde NCSN++
(`module.all_modules.*`, positional time embedding, nf=128, ch_mult (1,2,2,2),
attn@16, biggan resblocks, fir=False). This wrapper:

  * builds that exact NCSN++ from a plain-kwargs config shim (no ml_collections),
  * adds a zero-initialized class embedding into the time embedding so a class
    label `y` can steer the model (see models/score_sde/ncsnpp.py),
  * exposes the interface the FD pipeline expects: `sample_images_with_grad`,
    `generate`, `in_channels`, `input_size`, `num_classes`.

Flow / time convention (copied exactly from gnobitab RF, sde_lib.RectifiedFlow /
losses.get_rectified_flow_loss_fn):
    perturbed = t * data + (1 - t) * z0,   z0 ~ N(0, I) noise,   t in (eps, 1]
    target velocity = data - z0
    model(x, t * 999) predicts that velocity
    forward ODE (noise -> data):  x <- x + v * dt,  t: eps -> 1
Data is centered to [-1, 1] (config.data.centered = True), which is what the FD
loop expects before its own `* 0.5 + 0.5`.
"""

import logging
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from tqdm import trange

from .score_sde.ncsnpp import NCSNpp

logger = logging.getLogger("FD_loss")

# Rectified-flow integration endpoints (gnobitab RF: eps=1e-3, T=1).
_RF_EPS = 1e-3
_RF_T = 1.0
# score_sde feeds the model `t * 999` as the (continuous) time label.
_RF_TIME_SCALE = 999.0


def _cifar10_rf_config(img_size, num_channels, num_classes, dropout):
    """Plain-namespace replica of
    configs/rectified_flow/cifar10_rf_gaussian_ddpmpp.py (+ default_cifar10_configs)
    — exactly the values the released checkpoint was built with. `num_classes`
    is our addition (drives the zero-init label embedding in NCSNpp)."""
    model = SimpleNamespace(
        name="ncsnpp",
        scale_by_sigma=False,
        ema_rate=0.999999,
        dropout=dropout,                 # 0.0 by default -> deterministic sampling
        normalization="GroupNorm",
        nonlinearity="swish",
        nf=128,
        ch_mult=(1, 2, 2, 2),
        num_res_blocks=4,
        attn_resolutions=(16,),
        resamp_with_conv=True,
        conditional=True,                # time-conditional temb MLP
        fir=False,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=True,
        resblock_type="biggan",
        progressive="none",
        progressive_input="none",
        progressive_combine="sum",
        attention_type="ddpm",
        init_scale=0.0,
        embedding_type="positional",
        fourier_scale=16,
        conv_size=3,
        # noise-level grid for the `sigmas` buffer (default_cifar10_configs)
        sigma_min=0.01,
        sigma_max=50,
        num_scales=1000,
        beta_min=0.1,
        beta_max=20.0,
        # our class-conditioning add-on
        num_classes=num_classes,
    )
    data = SimpleNamespace(image_size=img_size, num_channels=num_channels, centered=True)
    training = SimpleNamespace(continuous=False, sde="rectified_flow")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return SimpleNamespace(model=model, data=data, training=training, device=device)


class RFDenoiser(nn.Module):
    """1-rectified-flow NCSN++ with an added (zero-init) class-conditioning path."""

    def __init__(
        self,
        img_size=32,
        in_channels=3,
        num_classes=10,
        dropout=0.0,
        grad_checkpoint=True,
        **_ignored,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.input_size = img_size
        self.num_classes = num_classes
        self.grad_checkpoint = grad_checkpoint

        config = _cifar10_rf_config(img_size, in_channels, num_classes, dropout)
        self.net = NCSNpp(config)

        n = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[RFDenoiser] NCSN++ params: {n:.2f}M, img={img_size}, "
                    f"num_classes={num_classes}, dropout={dropout}, "
                    f"grad_checkpoint={grad_checkpoint}")
        logger.info("[RFDenoiser] flow: t=0 noise -> t=1 data, x += v*dt, label=t*999")

    # -- velocity field -------------------------------------------------------
    def _velocity(self, x, t_scalar, y):
        """RF velocity v(x, t) = net(x, t*999, y). `t_scalar` is a python float."""
        t_vec = torch.full((x.shape[0],), t_scalar * _RF_TIME_SCALE,
                           device=x.device, dtype=torch.float32)
        if self.grad_checkpoint and torch.is_grad_enabled():
            # checkpoint each ODE step so backprop memory stays ~O(1) in #steps.
            return ckpt.checkpoint(self.net, x, t_vec, y, use_reentrant=False)
        return self.net(x, t_vec, y)

    def _euler_integrate(self, z, y, num_steps):
        """Forward Euler ODE noise->data, returns image in [-1, 1]. Keeps grad."""
        dt = 1.0 / num_steps
        x = z
        for i in range(num_steps):
            t = i / num_steps * (_RF_T - _RF_EPS) + _RF_EPS
            x = x + self._velocity(x, t, y) * dt
        return x

    # -- interface used by conditional_main_fd_ponly -------------------------
    def sample_images_with_grad(self, x, y, sampling_args=None):
        """Differentiable sampling. `x` is the initial noise (B,C,H,W)~N(0,I)*scale,
        `y` the class labels. Returns images in [-1, 1] (the FD loop then maps to
        [0,1]). `cfg`/`t_min`/`t_max` in sampling_args are ignored — RF here has no
        classifier-free guidance path."""
        sampling_args = sampling_args or {}
        num_steps = sampling_args.get("num_steps", 100)
        return self._euler_integrate(x, y, num_steps)

    @torch.inference_mode()
    def generate(self, n_samples, labels, cfg=4.0, args=None, verbose=True, z_t=None):
        """Inference sampler used by vis / eval / queue-fill. Mirrors the JiT
        denoiser signature. `cfg` is accepted but unused (no CFG for RF)."""
        device = labels.device
        num_steps = args.num_sampling_steps if args is not None else 100
        if z_t is not None:
            z = z_t
        elif args is not None and getattr(args, "same_noise", False):
            z = torch.randn(1, self.in_channels, self.img_size, self.img_size, device=device)
            z = z.repeat(n_samples, 1, 1, 1)
        else:
            z = torch.randn(n_samples, self.in_channels, self.img_size, self.img_size, device=device)

        dt = 1.0 / num_steps
        x = z
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        steps = (trange(num_steps, desc=f"[Rank{rank}] RF Euler (n={n_samples})")
                 if (n_samples > 32 and verbose) else range(num_steps))
        for i in steps:
            t = i / num_steps * (_RF_T - _RF_EPS) + _RF_EPS
            t_vec = torch.full((n_samples,), t * _RF_TIME_SCALE, device=device, dtype=torch.float32)
            x = x + self.net(x, t_vec, labels) * dt
        return x


def convert_rf_checkpoint(state_dict):
    """Map a gnobitab RectifiedFlow checkpoint (DataParallel-wrapped NCSN++:
    keys like `module.all_modules.*`, `module.sigmas`) onto RFDenoiser, whose
    backbone lives under `net.*`. The added `net.label_emb.*` is absent here and
    stays zero-initialized (loaded with strict=False)."""
    out = {}
    for k, v in state_dict.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        out["net." + nk] = v
    return out


# model registry (mirrors JiTDenoiser_models)
RFDenoiser_models = {
    "RF_cifar": lambda **kw: RFDenoiser(img_size=32, in_channels=3, **kw),
}
