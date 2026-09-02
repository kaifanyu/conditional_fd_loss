# Joint-KL / CFG-Delta Conditional Matching for FD Post-Training

**Implementation specification for an AI coding agent**

**Status:** Proposed experimental mode / ablation. Do **not** replace the current `density` mode or change existing defaults.

**Target repository components:**

- `frechet_distance/gmm.py` — real-data GMM `p`, online generator GMM `q`, and GMM losses
- `conditional_main_fd_gmm.py` — training integration, arguments, logging, state updates
- `compute_class_stats.py` — offline real-data reference fit
- `validate_class_gmm.py` — held-out validation and posterior-temperature calibration
- `tests/test_gmm_posterior.py` — synthetic correctness and collapse tests

---

## 0. Agent task

Add an **opt-in conditional CFG-delta mode** to the existing class-conditional GMM subsystem.

The new mode should implement the feature-space vector field

\[
 g_{\mathrm{cfg}}(z,c)
 =
 \underbrace{\nabla_z \log \hat q(c\mid z)}_{\text{fake CFG delta}}
 -
 \underbrace{\nabla_z \log p(c\mid z)}_{\text{teacher CFG delta}},
\]

and inject it into generator training alongside the existing FD loss.

The preferred mode name in this document is:

```text
--fd_gmm_mode cfg_delta
```

The implementation must:

1. Preserve existing `density` and `posterior` modes exactly.
2. Preserve `density` as the default.
3. Add analytic, first-order posterior-score methods to `ClassGMMReference` and `OnlineClassStats`.
4. Inject the desired feature gradient with a stop-gradient linear surrogate.
5. Keep all GMM parameters and online statistics detached.
6. Continue updating online `q` **after** the training backward pass, as the current code does.
7. Add diagnostics and correctness tests.
8. Clearly distinguish the pure chain-rule experiment from the practical variant that also retains `-log p(c|z)`.

Do **not** reintroduce the old sampled-label scalar posterior ratio as a default or recommended loss. It is included below only to explain the derivation and known failure mode.

---

## 1. Current system

### 1.1 Generator and representation space

The generator samples an image from noise and a requested class:

\[
 x = G_\theta(\epsilon,c).
\]

The GMM is not fitted directly in pixel space. A frozen FD judge extracts a feature vector, then the real-data whitening projection maps it into a lower-dimensional space:

\[
 f = \phi(x),
 \qquad
 z=(f-\mu_{p,\mathrm{feat}})P.
\]

The existing code calls this whitened feature `z`. Do not confuse it with the generator noise variable; this document uses `epsilon` for generator noise and `z` for the whitened judge feature.

### 1.2 Real and generated distributions

- `p`: frozen real-data class GMM, fitted offline by `compute_class_stats.py`.
- `q`: online GMM approximation to the generator feature distribution, maintained by `OnlineClassStats`.

Current model classes:

| Side | Statistics | Model form |
|---|---|---|
| Real `p` | many real samples per class | per-class full covariance, QDA-like |
| Generated `q` | online EMA, fewer samples per class | per-class means plus one tied within-class covariance, LDA-like |

Both operate in the same frozen whitened PCA space. The projection must match the selected FD judge exactly.

### 1.3 Current default objective

The current training recipe is

\[
\mathcal L
=
\mathcal L_{\mathrm{FD}}
+w(s)
\left[
\lambda_{\mathrm{cls}}\,\mathbb E[-\log p(c\mid z)]
+
\lambda_{\mathrm{ent}}\,\mathbb E[\log \hat q(z\mid c)-\log p(z\mid c)]
\right].
\]

The two GMM components have different roles:

- `-log p(c|z)` is the **class-fidelity driver**.
- `log q(z|c) - log p(z|c)` is the **density anti-collapse counterweight**.

The proposed CFG-delta mode is a different conditional signal. It should be implemented as an ablation, not silently substituted for the density term.

---

## 2. Proposed theoretical objective

Let \(q^*_\theta(x\mid c)\) denote the generator's **true induced distribution**, not the fitted online GMM. The user-stated ideal class-conditional matching objective is

\[
J(\theta)
=
\mathbb E_{c\sim p(c)}
D_{\mathrm{KL}}
\left(
q^*_\theta(x\mid c)\,\|\,p(x\mid c)
\right).
\tag{1}
\]

When the generator draws labels from the same prior, \(q(c)=p(c)\), this is exactly the joint KL

\[
J(\theta)
=
D_{\mathrm{KL}}
\left(
q^*_\theta(x,c)\,\|\,p(x,c)
\right).
\tag{2}
\]

Operationally, generator batches are drawn from the generator sampling prior \(q(c)\). If the two fixed class priors differ, then

\[
\mathbb E_{c\sim q(c)}
D_{\mathrm{KL}}(q^*(x\mid c)\|p(x\mid c))
=
D_{\mathrm{KL}}(q^*(x,c)\|p(x,c))
-
D_{\mathrm{KL}}(q(c)\|p(c)).
\]

The extra prior KL is constant with respect to \(\theta\), so the generator gradient is unchanged. Exact scalar equality, however, requires equal priors.

---

## 3. Derivation

### 3.1 Conditional KL to joint KL

Write

\[
q^*_\theta(x,c)=q(c)q^*_\theta(x\mid c),
\qquad
p(x,c)=p(c)p(x\mid c).
\]

Then

\[
\begin{aligned}
D_{\mathrm{KL}}(q^*_\theta(x,c)\|p(x,c))
&=
\mathbb E_{q^*_\theta(x,c)}
\left[
\log\frac{q(c)}{p(c)}
+
\log\frac{q^*_\theta(x\mid c)}{p(x\mid c)}
\right].
\end{aligned}
\]

When the priors match, the first term vanishes and Equation (1) equals Equation (2).

### 3.2 Refactor the joint in the other direction

Using

\[
q^*_\theta(x,c)=q^*_\theta(x)q^*_\theta(c\mid x),
\qquad
p(x,c)=p(x)p(c\mid x),
\]

KL chain rule gives

\[
\boxed{
D_{\mathrm{KL}}(q^*_\theta(x,c)\|p(x,c))
=
D_{\mathrm{KL}}(q^*_\theta(x)\|p(x))
+
\mathbb E_{x\sim q^*_\theta(x)}
D_{\mathrm{KL}}
\left(q^*_\theta(c\mid x)\|p(c\mid x)\right)
}.
\tag{3}
\]

Interpretation:

- The first term matches the **unconditional or pooled marginal** image distribution.
- The second term matches the **relationship between image and class label**.

A generator can have a good marginal while assigning the wrong class labels to images. The posterior term detects that mismatch.

### 3.3 Reparameterized gradient

Sample

\[
c\sim q(c),
\qquad
\epsilon\sim p(\epsilon),
\qquad
x=G_\theta(\epsilon,c).
\]

Starting from the joint KL,

\[
J(\theta)
=
\mathbb E_{\epsilon,c}
\left[
\log q^*_\theta(x,c)-\log p(x,c)
\right].
\]

Differentiation gives a pathwise term plus the explicit derivative of the generator density:

\[
\nabla_\theta J
=
\mathbb E
\left[
\left(
\nabla_x\log q^*_\theta(x,c)
-
\nabla_x\log p(x,c)
\right)
\frac{\partial x}{\partial\theta}
\right]
+
\mathbb E[\partial_\theta\log q^*_\theta(x,c)].
\]

The explicit score term has zero expectation:

\[
\begin{aligned}
\mathbb E_{q^*_\theta}
[\partial_\theta\log q^*_\theta]
&=
\int q^*_\theta(x)\frac{\partial_\theta q^*_\theta(x)}{q^*_\theta(x)}dx\\
&=
\partial_\theta\int q^*_\theta(x)dx\\
&=
\partial_\theta 1\\
&=0.
\end{aligned}
\]

Therefore

\[
\nabla_\theta J
=
\mathbb E
\left[
\left(
\nabla_x\log q^*_\theta(x,c)
-
\nabla_x\log p(x,c)
\right)
\frac{\partial x}{\partial\theta}
\right].
\tag{4}
\]

In the implementation, the same reasoning motivates detaching all fitted GMM parameters and allowing gradients to flow only through

```text
image -> frozen judge features -> whitening projection -> Gaussian score
```

### 3.4 Separate marginal and posterior gradients

Because

\[
\log q(x,c)=\log q(x)+\log q(c\mid x),
\]

Equation (4) can be written as

\[
\nabla_\theta J
=
\mathbb E
\left[
\left(
\underbrace{s_q(x)-s_p(x)}_{\text{marginal score mismatch}}
+
\underbrace{\nabla_x\log q(c\mid x)-\nabla_x\log p(c\mid x)}_{\text{conditional mismatch}}
\right)
\frac{\partial x}{\partial\theta}
\right],
\tag{5}
\]

where

\[
s_p(x)=\nabla_x\log p(x),
\qquad
s_q(x)=\nabla_x\log q(x).
\]

### 3.5 Bayes identity and the CFG interpretation

Bayes gives

\[
\log p(c\mid x)
=
\log p(x\mid c)+\log p(c)-\log p(x).
\]

The prior does not depend on \(x\), so

\[
\boxed{
\nabla_x\log p(c\mid x)
=
s_p(x\mid c)-s_p(x)
}.
\tag{6}
\]

Likewise,

\[
\boxed{
\nabla_x\log q(c\mid x)
=
s_q(x\mid c)-s_q(x)
}.
\tag{7}
\]

Define

\[
\Delta_p(x,c)=s_p(x\mid c)-s_p(x),
\qquad
\Delta_q(x,c)=s_q(x\mid c)-s_q(x).
\]

These are directly analogous to classifier-free-guidance score deltas:

- \(\Delta_p\): **teacher / real-data CFG delta**
- \(\Delta_q\): **fake / generator CFG delta**

The conditional correction is therefore

\[
\boxed{
 g_{\mathrm{cfg}}(x,c)=\Delta_q(x,c)-\Delta_p(x,c)
}.
\tag{8}
\]

Gradient descent moves in the opposite direction, \(\Delta_p-\Delta_q\), teaching the generator's current conditioning effect to match the real-data conditioning effect.

### 3.6 Exact cancellation sanity check

Substitute Equations (6) and (7) into Equation (5):

\[
\begin{aligned}
&s_q(x)-s_p(x)
+
[s_q(x\mid c)-s_q(x)]
-
[s_p(x\mid c)-s_p(x)]\\
&=s_q(x\mid c)-s_p(x\mid c).
\end{aligned}
\]

This recovers the gradient of the original conditional density KL. The algebra is therefore internally consistent.

### 3.7 Replace marginal KL with FD

The practical proposal is

\[
\boxed{
\nabla_\theta \mathcal L_{\mathrm{hybrid}}
\approx
\nabla_\theta \mathcal L_{\mathrm{FD}}
+
\mathbb E
\left[
(\Delta_q(z,c)-\Delta_p(z,c))
\frac{\partial z}{\partial\theta}
\right]
}.
\tag{9}
\]

This says:

- FD handles the pooled marginal distribution.
- The CFG-delta term handles only the conditional residual.

However, FD is a moment-based surrogate, **not** the exact marginal KL. Equation (9) is a hybrid surrogate and is not mathematically equal to Equation (1).

---

## 4. Exact statements versus practical approximations

An implementation agent must preserve the following distinctions.

### 4.1 FD is not `KL(q(x) || p(x))`

FD constrains pooled feature means and covariances. It does not provide the exact marginal density ratio or marginal score. Therefore the exact cancellation in Section 3.6 does not literally occur in training.

### 4.2 The GMM operates in feature space

The implemented scores are

\[
\nabla_z\log p(c\mid z),
\qquad
\nabla_z\log \hat q(c\mid z),
\]

not full pixel-space scores. The frozen judge and whitening projection carry these gradients back to pixels and then to generator parameters.

### 4.3 `q*` is not the fitted online `q_hat`

The derivation uses the generator's true posterior \(q^*_\theta(c\mid x)\). The code only has a fitted approximation \(\hat q(c\mid z)\).

This distinction is the reason the old sampled-label scalar

\[
\log \hat q(c_i\mid z_i)-\log p(c_i\mid z_i)
\]

is unsafe: it is a valid Monte Carlo estimator of a posterior KL only if \(\hat q(c\mid z)\) is the exact posterior of the joint that generated each sample. A fitted GMM does not satisfy that condition.

### 4.4 Full posterior KL and CFG-delta injection are different plug-in objectives

The existing `posterior` mode computes

\[
\mathbb E_z
D_{\mathrm{KL}}
\left(
\hat q(c\mid z)\|p(c\mid z)
\right).
\]

This is non-negative pointwise. Its direct gradient includes differentiation through the full posterior probabilities and is **not identical** to injecting only

\[
\nabla_z\log\hat q(c_i\mid z_i)-\nabla_z\log p(c_i\mid z_i)
\]

for the sampled label.

The proposed `cfg_delta` mode should therefore be a new branch, not an alias for the existing `posterior` branch.

### 4.5 Temperature changes the exact objective

At `temperature = 1`, Bayes identities in Section 3.5 are exact for the fitted mixture.

At `temperature > 1`, the implementation uses tempered posteriors:

\[
r^p_T(c\mid z)=\operatorname{softmax}(\ell^p_c(z)/T),
\]

so

\[
\nabla_z\log r^p_T(c\mid z)
=
\frac{1}{T}
\left[
 s_p(z\mid c)-\sum_j r^p_T(j\mid z)s_p(z\mid j)
\right].
\]

This is a **tempered CFG delta**, not the exact untempered joint-KL decomposition. Temperature is nevertheless useful in the current project because high-dimensional GMM posteriors otherwise saturate and provide almost no gradient.

Temperature changes both responsibilities and gradient scale, so the GMM weight must be recalibrated whenever temperature changes.

### 4.6 `p` and `q` use different covariance families

Real `p` is QDA-like and generated `q` is LDA-like. Even an ideal generator may leave a residual mismatch because the model classes differ. Do not assume the CFG-delta error or posterior KL must reach exactly zero.

---

## 5. Relationship to existing modes

Let

\[
g_{\mathrm{density}}
=
s_q(z\mid c)-s_p(z\mid c).
\]

The proposed field is

\[
\begin{aligned}
g_{\mathrm{cfg}}
&=
[s_q(z\mid c)-s_q(z)]
-
[s_p(z\mid c)-s_p(z)]\\
&=
g_{\mathrm{density}}-[s_q(z)-s_p(z)].
\end{aligned}
\]

Thus CFG-delta matching removes the marginal score mismatch from the conditional density mismatch and delegates the marginal part to FD.

| Mode | Implemented quantity | Main purpose | Main risk |
|---|---|---|---|
| `density` | `log q(z|c) - log p(z|c)` | direct class-density matching and anti-collapse repulsion | overlaps with marginal structure already addressed by FD |
| `posterior` | full `KL(q(c|z) || p(c|z))` | stable posterior agreement | under collapse, `q(c|z)` can saturate and the loss degenerates toward contractive classifier guidance |
| `cfg_delta` | sampled-label posterior **score** difference | match only conditional residual / CFG field | fitted `q_hat` is not true `q*`; posterior saturation and model bias remain |
| old sampled scalar | `log q(c_i|z_i) - log p(c_i|z_i)` | literal scalar giving the desired sampled score | unbounded below with fitted `q_hat`; can reward samples that `q_hat` misclassifies |

The current density mode remains the preferred anti-collapse baseline because, for a Gaussian,

\[
\nabla_z[\log q(z\mid c)-\log p(z\mid c)]
=
\Lambda^p_c(z-\mu^p_c)-\Lambda^q(z-\mu^q_c).
\]

As generated within-class covariance shrinks, \(\Lambda^q\) grows, strengthening the repulsive term exactly under collapse. The posterior-based alternatives do not have the same guarantee.

---

## 6. Experimental objectives

The implementation should make the following configurations explicit.

### 6.1 Existing density baseline

\[
\mathcal L_A
=
\mathcal L_{\mathrm{FD}}
+w
\left[
\lambda_{\mathrm{cls}}(-\log p(c\mid z))
+
\lambda_{\mathrm{ent}}(\log\hat q(z\mid c)-\log p(z\mid c))
\right].
\]

CLI concept:

```bash
--fd_gmm_mode density \
--fd_gmm_lambda_cls 1 \
--fd_gmm_lambda_ent 1
```

### 6.2 Pure chain-rule / CFG-delta experiment

\[
\mathcal L_B
\leadsto
\mathcal L_{\mathrm{FD}}
+
\text{feature gradient }g_{\mathrm{cfg}}.
\]

CLI concept:

```bash
--fd_gmm_mode cfg_delta \
--fd_gmm_lambda_cls 0 \
--fd_gmm_lambda_ent 1
```

This is the cleanest test of the proposed decomposition. It may fail to teach a de-conditioned generator because it removes the explicit class-fidelity driver.

### 6.3 Practical CFG-delta plus class driver

\[
\mathcal L_C
\leadsto
\mathcal L_{\mathrm{FD}}
+
\lambda_{\mathrm{cls}}[-\log p(c\mid z)]
+
\text{feature gradient }\lambda_{\mathrm{ent}}g_{\mathrm{cfg}}.
\]

CLI concept:

```bash
--fd_gmm_mode cfg_delta \
--fd_gmm_lambda_cls 1 \
--fd_gmm_lambda_ent 1
```

This is not the pure chain-rule objective: the teacher class-fidelity signal is emphasized separately. It is a practical experiment for de-conditioned training and must be labeled as such.

### 6.4 Existing full posterior-KL comparison

```bash
--fd_gmm_mode posterior \
--fd_gmm_lambda_cls 0
```

This compares the new explicit vector-field injection against the existing non-negative scalar posterior KL.

---

## 7. Gaussian posterior-score formulas

The new implementation should use analytic formulas rather than nested `autograd.grad` inside the compiled training step.

### 7.1 Real QDA mixture

For real class \(j\),

\[
\log p(z\mid j)
=-\frac12(z-\mu^p_j)^T\Lambda^p_j(z-\mu^p_j)+\text{const},
\]

so

\[
s_{p,j}(z)
=
\nabla_z\log p(z\mid j)
=
-\Lambda^p_j(z-\mu^p_j)
=
\Lambda^p_j\mu^p_j-\Lambda^p_j z.
\]

Let

\[
r^p_j(z)=p(j\mid z).
\]

The marginal mixture score is

\[
s_p(z)=\sum_j r^p_j(z)s_{p,j}(z).
\]

Therefore

\[
\boxed{
\Delta_p(z,c)
=
\frac{1}{T}
\left[
 s_{p,c}(z)-\sum_j r^p_{T,j}(z)s_{p,j}(z)
\right]
}.
\tag{10}
\]

At \(T=1\), this is exactly \(\nabla_z\log p(c\mid z)\).

### 7.2 Generated tied-covariance mixture

For online `q`, every class shares one precision \(\Lambda^q\):

\[
s_{q,j}(z)=\Lambda^q\mu^q_j-\Lambda^qz.
\]

The class-independent term \(-\Lambda^qz\) cancels between the selected class and the posterior-weighted average. Thus

\[
\boxed{
\Delta_q(z,c)
=
\frac{1}{T}
\left[
\Lambda^q\mu^q_c
-
\sum_j r^q_{T,j}(z)\Lambda^q\mu^q_j
\right]
}.
\tag{11}
\]

This can be computed with two matrix multiplications and no `(B, C, k)` tensor.

### 7.3 Desired injected gradient

\[
\boxed{
 g_{\mathrm{cfg}}(z,c)=\Delta_q(z,c)-\Delta_p(z,c)
}.
\tag{12}
\]

---

## 8. Implementation plan by file

## 8.1 `frechet_distance/gmm.py`

### A. Add a real posterior-score method

Add this method to `ClassGMMReference`. It expects **local** component labels, because `gmm_posterior_loss` already calls `reference.to_local(labels)`.

```python
@torch.no_grad()
def posterior_score_delta(
    self,
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return grad_z log p_T(c | z), shape (B, k).

    At temperature=1 this is the exact Bayes/CFG score delta
    s_p(z|c) - s_p(z). For temperature>1 it is the corresponding
    tempered-posterior gradient.

    All reference parameters are frozen. The returned tensor is a
    detached vector field evaluated at the current z.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    probs = torch.softmax(self.logits(z, temperature=temperature), dim=-1)
    precisions = self.prec_flat.view(self.num_classes, self.k, self.k)

    # Selected-class score: P_c mu_c - P_c z.
    p_y = precisions.index_select(0, labels)
    score_selected = (
        self.prec_mu.index_select(0, labels)
        - torch.bmm(p_y, z.unsqueeze(-1)).squeeze(-1)
    )

    # Posterior-weighted mixture score.
    weighted_prec_mu = probs @ self.prec_mu
    weighted_prec_z = torch.einsum(
        "bc,ckl,bl->bk", probs, precisions, z
    )
    score_marginal = weighted_prec_mu - weighted_prec_z

    return (score_selected - score_marginal) / float(temperature)
```

Notes:

- The `einsum` has QDA cost comparable to the existing real GMM logit evaluation.
- Do not construct a persistent `(B, C, k)` tensor.
- Use an autograd-based implementation only as a test oracle, not the production path.

### B. Add an online posterior-score method

Add this method to `OnlineClassStats`:

```python
@torch.no_grad()
def posterior_score_delta(
    self,
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return grad_z log q_T(c | z), shape (B, k).

    q has tied covariance, so the class-independent -Pz term cancels
    from the posterior gradient. Reads the detached cache produced by
    refresh_cache().
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    probs = torch.softmax(self.logits(z, temperature=temperature), dim=-1)
    natural_means = self.prec_mu_cache  # (C, k) = mu_c @ precision

    selected = natural_means.index_select(0, labels)
    expected = probs @ natural_means
    return (selected - expected) / float(temperature)
```

### C. Extend `gmm_posterior_loss`

Add `"cfg_delta"` to the accepted modes. Add an optional explicit vector normalization argument rather than reusing scalar self-normalization:

```python
cfg_delta_normalization: str = "none",
cfg_delta_eps: float = 1e-6,
```

Accepted values:

- `none` — exact raw fitted-GMM vector field; recommended default.
- `rms` — divide the whole batch vector field by its detached RMS; an explicit engineering surrogate, not the exact derived field.

Add a branch next to `density` and `posterior`:

```python
elif mode == "cfg_delta":
    delta_p = reference.posterior_score_delta(
        z.detach(), labels, temperature=temperature
    )
    delta_q = online.posterior_score_delta(
        z.detach(), labels, temperature=temperature
    )
    cfg_grad_raw = delta_q - delta_p

    raw_rms = cfg_grad_raw.square().mean().sqrt()
    if cfg_delta_normalization == "none":
        cfg_grad = cfg_grad_raw
        cfg_scale = torch.ones((), device=z.device, dtype=z.dtype)
    elif cfg_delta_normalization == "rms":
        cfg_scale = raw_rms.clamp_min(cfg_delta_eps)
        cfg_grad = cfg_grad_raw / cfg_scale
    else:
        raise ValueError(
            "cfg_delta_normalization must be 'none' or 'rms', got "
            f"{cfg_delta_normalization!r}"
        )

    # First-order gradient injection. The numeric value is not a KL and has
    # no standalone statistical interpretation. Its gradient w.r.t. z is
    # the correct Monte Carlo mean field: each sample receives cfg_grad / B
    # because the batch objective is averaged.
    term = (z * cfg_grad.detach()).sum(-1).mean()

    def _rms(v):
        return v.square().mean().sqrt()

    denom = (
        delta_q.norm(dim=-1) * delta_p.norm(dim=-1)
    ).clamp_min(cfg_delta_eps)
    alignment = ((delta_q * delta_p).sum(-1) / denom).mean()

    teacher_rms = _rms(delta_p)
    fake_rms = _rms(delta_q)
    error_rms = _rms(cfg_grad_raw)

    parts["gmm_cfg_surrogate"] = term.detach()
    parts["gmm_cfg_teacher_rms"] = teacher_rms.detach()
    parts["gmm_cfg_fake_rms"] = fake_rms.detach()
    parts["gmm_cfg_error_rms"] = error_rms.detach()
    parts["gmm_cfg_relative_error"] = (
        error_rms / teacher_rms.clamp_min(cfg_delta_eps)
    ).detach()
    parts["gmm_cfg_alignment_cos"] = alignment.detach()
    parts["gmm_cfg_vector_scale"] = cfg_scale.detach()
```

Then handle objective scaling as follows:

```python
if mode == "cfg_delta":
    # Do not call _self_normalize(term). The dot-product surrogate's scalar
    # value is origin-dependent and is not the objective being estimated.
    q_objective = lambda_ent * term
else:
    q_objective = lambda_ent * _self_normalize(term, normalize)
```

Important:

- `cfg_grad` must be detached before the linear surrogate.
- Do not allow gradients into GMM statistics or posterior responsibilities.
- Do not call `_self_normalize` on `gmm_cfg_surrogate`.
- Calibrate the outer `fd_gmm_weight` from image-space gradients.
- Keep `l_cls` behavior unchanged. Setting `lambda_cls=0` selects the pure CFG-delta experiment.

### D. Update the function docstring

The docstring must explain:

- `cfg_delta` injects a local vector field, not a scalar KL.
- At `T=1`, it is the fitted-GMM version of the Bayes/CFG posterior score difference.
- At `T>1`, it is tempered.
- It does not inherit density mode's guaranteed covariance-strengthened anti-collapse repulsion.
- It is distinct from the existing full posterior KL.

---

## 8.2 `conditional_main_fd_gmm.py`

### A. Parser

Extend:

```python
parser.add_argument(
    "--fd_gmm_mode",
    type=str,
    default="density",
    choices=["density", "posterior", "cfg_delta"],
)
```

Add:

```python
parser.add_argument(
    "--fd_gmm_cfg_normalization",
    choices=["none", "rms"],
    default="none",
    help=(
        "normalization for the explicit cfg_delta feature vector. "
        "'none' preserves the raw fitted-GMM field; 'rms' divides by "
        "its detached batch RMS and changes the objective. This is "
        "separate from scalar GMM loss self-normalization."
    ),
)
```

### B. Pass the new option into `gmm_posterior_loss`

Capture the argument in `get_fd_train_step`, then pass it through:

```python
cfg_delta_normalization=args.fd_gmm_cfg_normalization,
```

### C. Startup logging

When mode is `cfg_delta`, log whether the run is:

- pure chain-rule: `lambda_cls == 0`
- practical class-driver variant: `lambda_cls > 0`

Example:

```text
[GMM] mode=cfg_delta [PURE: no explicit class driver]
```

or

```text
[GMM] mode=cfg_delta [PRACTICAL: explicit -log p(c|z) retained]
```

Also log:

- temperature
- CFG vector normalization
- `lambda_cls`
- `lambda_ent`
- overall GMM weight

### D. Keep the current state-update ordering

Do not alter this order:

1. Generate samples.
2. Extract differentiable judge features.
3. Evaluate FD and GMM terms using the current detached `q` cache.
4. Backward through generator and judge features.
5. Optimizer / EMA update.
6. Update `OnlineClassStats` with detached current features and labels.
7. Refresh the `q` cache for the next step.

Updating `q` before the loss or inside the graph would make the estimator self-referential and could invalidate the intended score-identity treatment.

---

## 8.3 Tests

Add focused tests to `tests/test_gmm_posterior.py` or a new `tests/test_gmm_cfg_delta.py`.

### Test 1: real analytic delta matches autograd

For a small random QDA GMM:

1. Create `z.requires_grad_(True)`.
2. Compute `log_softmax(reference.logits(z, T))`.
3. Select labels and differentiate their summed log posterior with respect to `z`.
4. Compare against `reference.posterior_score_delta(z.detach(), labels, T)`.

Acceptance:

```python
torch.testing.assert_close(analytic, autograd_value, rtol=1e-5, atol=1e-6)
```

Run for `T=1` and one `T>1`.

### Test 2: online tied-covariance delta matches autograd

Repeat the same test for `OnlineClassStats.logits` after populating its cache.

### Test 3: linear surrogate injects exactly the requested gradient

Given a detached random vector `g`:

```python
z = torch.randn(B, k, requires_grad=True)
loss = (z * g).sum(-1).mean()
grad = torch.autograd.grad(loss, z)[0]
```

Verify:

\[
\nabla_z\text{loss}=g/B
\]

with the expected batch-mean factor.

### Test 4: CFG delta is zero when fitted models agree

Construct `p` and `q` with the same tied covariance, class means, and priors. For the same temperature:

\[
\Delta_q(z,c)-\Delta_p(z,c)\approx0.
\]

Because the production `p` and `q` model classes differ, this exact-zero test should use a deliberately tied `p` fixture.

### Test 5: density/CFG/marginal identity

Numerically verify at the fitted-model level:

\[
g_{\mathrm{density}}-g_{\mathrm{cfg}}
=s_q(z)-s_p(z).
\]

This pins the sign convention.

### Test 6: no GMM parameter gradients

After backward in `cfg_delta` mode:

- generator/judge input features must receive gradient;
- every GMM buffer or parameter must have no gradient;
- online EMA accumulators must remain detached.

### Test 7: existing behavior is unchanged

All existing density and posterior tests must pass bit-for-bit or within their current tolerances. The default mode must remain `density`.

### Test 8: known posterior-collapse test remains

Do not remove or weaken the existing synthetic test showing that full posterior KL degenerates under collapse. The new mode is an ablation, not a claim that posterior-based matching solved that failure mode.

### Test 9: invalid temperature and normalization fail clearly

Require positive temperature and reject unknown normalization strings.

---

## 9. Reference autograd oracle

Use the following only in tests or debugging. Do not use it as the production implementation if `torch.compile` or performance matters.

```python
def posterior_score_delta_autograd(logits_fn, z, labels, temperature):
    z_ref = z.detach().clone().requires_grad_(True)
    logits = logits_fn(z_ref, temperature=temperature)
    log_post = torch.log_softmax(logits, dim=-1)
    selected = log_post.gather(1, labels[:, None]).sum()
    return torch.autograd.grad(selected, z_ref, create_graph=False)[0].detach()
```

Compare this oracle against both analytic methods.

---

## 10. Why the old sampled-label scalar is not the recommended implementation

The scalar

\[
\ell_{\mathrm{sample}}
=
\log\hat q(c_i\mid z_i)-\log p(c_i\mid z_i)
\]

has the desired local gradient

\[
\nabla_z\ell_{\mathrm{sample}}
=
\Delta_q(z_i,c_i)-\Delta_p(z_i,c_i).
\]

For the true generator posterior, averaging over samples from the true joint gives the posterior KL. For a fitted online GMM, however, \(\hat q(c\mid z)\) is not guaranteed to be the posterior of the joint that generated the sample. The scalar then loses non-negativity and may be minimized by producing examples that the fitted `q` misclassifies.

The explicit vector-field implementation does **not** remove the model-bias risk in the direction itself, but it avoids pretending the sampled scalar is a valid KL and avoids accidental higher-order or parameter-side gradients. The full posterior-KL mode remains the non-negative scalar comparison.

---

## 11. Normalization and weight calibration

### 11.1 Do not normalize the surrogate scalar

The dot-product surrogate

\[
\mathcal L_{\mathrm{sur}}=z^T\operatorname{stopgrad}(g_{\mathrm{cfg}})
\]

exists only to inject a gradient. Its numeric value changes with the coordinate origin and is not a KL. Applying

```python
term / (abs(term.detach()) + eps)
```

would make the gradient scale depend on an arbitrary scalar value. Do not do this.

### 11.2 Calibrate with image-space gradient ratio

Use the existing diagnostic

```text
grad_ratio_p_fd = ||d L_GMM / d image|| / ||d L_FD / d image||
```

The current successful density runs suggest a sustained ratio near `0.22–0.30`, but that range is **not guaranteed to transfer** to CFG-delta mode. The vector field, temperature, and normalization differ.

Required procedure:

1. Start with a conservative small outer weight.
2. Run a short calibration job through the full warmup/ramp.
3. Read the sustained image-space gradient ratio, not the scalar loss.
4. Adjust weight and rerun calibration.
5. Record the final sustained ratio with the experiment.

Do not reuse a density-mode weight without measurement.

### 11.3 Optional RMS vector normalization

`cfg_delta_normalization=rms` produces

\[
\tilde g_{\mathrm{cfg}}
=
\frac{g_{\mathrm{cfg}}}
{\sqrt{\mathbb E[g_{\mathrm{cfg}}^2]}+\epsilon}.
\]

This may stabilize scale but changes the derived vector field. Treat it as a separate ablation. Default to `none` for the first correctness experiment.

---

## 12. Temperature calibration

The current project calibrates posterior temperature on held-out real images because high-dimensional Mahalanobis logits saturate at `T=1`.

For CFG-delta mode:

- Use the same temperature for `p` and `q` initially, matching the current API.
- Re-measure temperature for every class subset and GMM fit.
- Recognize that `T>1` makes the experiment a tempered surrogate.
- Recalibrate the outer weight after changing temperature.

The density mode's `log q(z|c)-log p(z|c)` is unaffected by posterior temperature; CFG-delta mode is directly affected.

---

## 13. Reference fitting and subset rules

The real reference must be fitted on exactly the label support used by training.

For a class subset, refit all of the following on that subset:

- whitening PCA;
- pooled within-class covariance;
- per-class means and covariances;
- posterior denominator / component set.

Do not mask a 1000-class GMM down to a smaller subset at runtime. Unused components would remain in the softmax and create incorrect posterior responsibilities and spurious CFG directions.

At startup, continue using `reference.validate_labels(drawable)` and require

```text
reference.num_classes == number of drawable labels
```

---

## 14. Online `q` state and staleness

`OnlineClassStats` updates per class and uses detached EMA statistics. Its effective wall-clock age depends on class count. A mean-EMA horizon measured in appearances per class can span far more training steps at 1000 classes than at 20 classes.

For every new class count:

- inspect `gmm_class_mean_ema_n_eff`;
- inspect `gmm_class_mean_mse`;
- ensure `q` is not so stale that the fake CFG field describes a generator from many thousands of steps earlier;
- recalibrate `fd_gmm_ema_beta` rather than copying it blindly.

The new mode depends directly on current posterior geometry, so stale `q` can rotate the injected vector field in the wrong direction.

---

## 15. Required diagnostics

Retain all existing diagnostics and add the following.

| Metric | Meaning | Desired reading |
|---|---|---|
| `gmm_cfg_teacher_rms` | strength of real-data conditional field | finite; depends on temperature |
| `gmm_cfg_fake_rms` | strength of current generator conditional field | should evolve from the de-conditioned baseline |
| `gmm_cfg_error_rms` | RMS of `Delta_q - Delta_p` | should eventually fall if matching succeeds |
| `gmm_cfg_relative_error` | error divided by teacher RMS | easier comparison across temperatures |
| `gmm_cfg_alignment_cos` | cosine between fake and teacher deltas | should rise toward `1` |
| `gmm_cfg_vector_scale` | applied RMS normalizer | `1` in raw mode |
| `grad_ratio_p_fd` | actual image-space contribution | primary weight-calibration metric |

Continue to monitor:

- `probe_top1` and `probe_rank` — held-out class signal;
- `gmm_top1` — in-loss GMM opinion;
- `gmm_class_mean_spread` and spread-to-noise;
- `gmm_within_trace_ratio` — read jointly with spread;
- `gmm_nll_p`;
- `cond_delta`;
- `cos_update_fd_p`;
- real FID.

### Interpretation warnings

- Falling CFG error without improvement in held-out probe accuracy may indicate the fitted GMM is being gamed.
- High `gmm_top1` with much lower `probe_top1` is a warning sign.
- A class-conditioned generator can still collapse within each class; always read CFG metrics together with `within_trace_ratio` and FID.
- `gmm_cfg_surrogate` has no standalone objective interpretation. Do not compare its scalar value across runs.

---

## 16. Expected failure modes and safeguards

### Failure A: no conditioning takeoff

Pure CFG-delta mode with `lambda_cls=0` may provide no usable driver from a fully de-conditioned generator.

Safeguards:

- run both pure and practical `+ class driver` variants;
- use `probe_rank`, class-mean spread-to-noise, and `gmm_nll_p` as early indicators;
- calibrate weight before a long run.

### Failure B: posterior saturation

If both fitted posteriors become nearly one-hot, conditional posterior gradients can flatten or become dominated by teacher classification.

Safeguards:

- held-out temperature sweep;
- log posterior entropy or saturation if added;
- preserve the density mode as the anti-collapse baseline.

### Failure C: fitted `q` is wrong or stale

The fake CFG delta can be misleading if online class means or covariance lag behind the generator.

Safeguards:

- bootstrap all classes;
- maintain full class coverage;
- log effective sample size and mean MSE;
- tune EMA per class count;
- update `q` only after the current gradient step.

### Failure D: double counting class fidelity

`cfg_delta` already contains `-grad log p(c|z)` inside the teacher delta. Adding a separate `-log p(c|z)` emphasizes class fidelity again.

Safeguard:

- label `lambda_cls=0` as the pure derivation;
- label `lambda_cls>0` as the practical de-conditioned variant;
- never present them as the same objective.

### Failure E: incorrect sign

The objective gradient is

\[
g_{\mathrm{cfg}}=\Delta_q-\Delta_p.
\]

Gradient descent therefore moves samples along

\[
\Delta_p-\Delta_q.
\]

Safeguard:

- pin the sign with the density/CFG/marginal identity test;
- include a toy two-class visualization or one-step synthetic test.

### Failure F: scalar self-normalization corrupts the vector field

Safeguard:

- never apply `_self_normalize` to the linear surrogate;
- normalize only the vector explicitly and log the choice.

---

## 17. Recommended experiment matrix

All arms must use matched model checkpoint, labels, FD judges, batch size, temperature fit, queue/bootstrap policy, and evaluation protocol. Calibrate each arm's outer weight independently by sustained `grad_ratio_p_fd`.

| Arm | Mode | `lambda_cls` | Purpose |
|---|---:|---:|---|
| A | `density` | current value | existing best baseline |
| B | `posterior` | `0` | full non-negative posterior KL |
| C | `cfg_delta` | `0` | pure chain-rule / conditional-residual test |
| D | `cfg_delta` | current driver value | practical de-conditioned version |
| E | `density` with `use_q=False` | calibrated | correctly weighted class-driver-only control |

Do not reuse the same raw `fd_gmm_weight` across arms. Match their actual image-space gradient contribution.

Primary comparison criteria:

1. held-out `probe_top1` / `probe_rank`;
2. real FID at matched conditioning accuracy;
3. within-class diversity and class-mean spread;
4. GMM-versus-probe agreement;
5. CFG error and alignment trajectories.

---

## 18. End-to-end pseudocode

```python
# Sample generator inputs.
epsilon = torch.randn(...)
y_global = sample_labels()
images = generator.sample_images_with_grad(epsilon, y_global)

# Frozen differentiable judge features.
features = judge(images)
features_all = differentiable_all_gather(features)

# Existing pooled FD objective.
fd_loss = compute_normalized_fd(features_all, reference_moments, queue)

# Whitened GMM features and local labels.
z = reference.project(features_all)
y = reference.to_local(all_gather_labels(y_global))

# Existing optional class driver.
logp_post = torch.log_softmax(reference.logits(z, temperature=T), dim=-1)
logp_c = logp_post.gather(1, y[:, None]).squeeze(1)
cls_loss = capped_negative_logp(logp_c)

if mode == "density":
    conditional_loss = mean(
        online.log_likelihood(z, y)
        - reference.log_likelihood(z, y)
    )
    conditional_objective = density_self_normalize(conditional_loss)

elif mode == "posterior":
    logq_post = torch.log_softmax(online.logits(z, temperature=T), dim=-1)
    conditional_loss = mean(sum(exp(logq_post) * (logq_post - logp_post), dim=-1))
    conditional_objective = posterior_self_normalize(conditional_loss)

elif mode == "cfg_delta":
    # Evaluate local fitted-GMM posterior fields without building a graph
    # through the field itself.
    delta_p = reference.posterior_score_delta(z.detach(), y, temperature=T)
    delta_q = online.posterior_score_delta(z.detach(), y, temperature=T)
    g_cfg = delta_q - delta_p

    if cfg_normalization == "rms":
        g_cfg = g_cfg / detached_rms(g_cfg)

    # The averaged objective gives each sample the correct Monte Carlo
    # contribution g_cfg / batch_size.
    conditional_objective = mean(sum(z * stop_gradient(g_cfg), dim=-1))

loss = (
    fd_loss
    + gmm_weight * (
        lambda_cls * normalized_or_fixed_scale(cls_loss)
        + lambda_ent * conditional_objective
    )
)

loss.backward()
optimizer.step()
optimizer.zero_grad()

# State mutation remains outside the loss graph and after the step.
queue.enqueue(features_all.detach())
online.update(z.detach(), y.detach())
online.refresh_cache(reference)
```

---

## 19. Acceptance criteria

The task is complete only when all of the following are true.

### Functional

- `--fd_gmm_mode cfg_delta` parses and runs.
- Existing default remains `density`.
- Pure mode works with `--fd_gmm_lambda_cls 0`.
- Practical mode works with `lambda_cls > 0`.
- Checkpoint save/load of online GMM state remains unchanged.
- Class-subset validation remains enforced.

### Mathematical

- Real analytic posterior delta matches autograd.
- Online analytic posterior delta matches autograd.
- The linear surrogate produces the exact requested first-order feature gradient.
- The sign convention passes the density/CFG/marginal identity test.
- GMM buffers receive no gradients.

### Regression

- Existing density and posterior tests pass.
- Existing collapse-failure tests remain active.
- Existing runs without `cfg_delta` are unaffected.

### Observability

- New CFG RMS, error, relative error, and alignment metrics are logged.
- Startup logs clearly identify pure versus practical mode.
- `grad_ratio_p_fd` remains available for calibration.
- The surrogate scalar is not mislabeled as a KL.

### Documentation

- The mode is documented as a hybrid feature-space surrogate.
- The distinction between true `q*` and fitted `q_hat` is explicit.
- The temperature approximation is explicit.
- The known posterior-collapse limitation is explicit.

---

## 20. Completion report expected from the coding agent

After implementation, report:

1. Files changed.
2. Exact CLI required for pure and practical CFG-delta modes.
3. Test commands and results.
4. Numerical maximum error between analytic and autograd posterior deltas.
5. Confirmation that current defaults are unchanged.
6. Any compile/performance impact of the real QDA `einsum`.
7. A short calibration run showing `grad_ratio_p_fd`, CFG error RMS, alignment, probe metrics, spread, within ratio, and FID if available.
8. Any deviations from this specification, with reasons.

---

## 21. Source map in the current project

Use these project files as the source of truth while implementing:

- `gmm.py`
  - `ClassGMMReference`
  - `OnlineClassStats`
  - `gmm_posterior_loss`
  - current density and posterior branches
- `conditional_main_fd_gmm.py`
  - `get_fd_train_step`
  - `setup_gmm_judge`
  - online-stat update ordering
  - parser and diagnostics
- `compute_class_stats.py`
  - whitening and real per-class statistics
- `validate_class_gmm.py`
  - reference validation and temperature calibration
- `gmm_posterior_loss.md`
  - derivation history and the three bring-up failure modes
- `gmm.md`
  - current experiment record, calibration history, and open controls
- `gmm_uncond_20class.md`
  - de-conditioned pilot setup and interpretation of class-spread / within-class meters

The current evidence supports implementing `cfg_delta` as a carefully instrumented ablation. It does **not** support replacing density mode as the default anti-collapse objective.
