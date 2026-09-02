"""The frozen VLM's *answer state* as a differentiable image feature.

Where ``vlm_judge.BinaryVLMJudge`` asks a Yes/No question and differentiates the
one-token answer *logit*, this module stops one step earlier and hands back the
hidden state the language model is about to decode from::

    [<image>] "What is the ImageNet class of this image? Answer:"   ->   z

``z`` is the last-layer (or a chosen intermediate layer) hidden state at the
final prompt position -- the vector the LM head would multiply to emit the first
token of the class name.  It is a classifier feature in the most literal sense:
the model has been asked the question, and ``z`` is its answer before
verbalisation.

Why this and not SigLIP's CLS token
-----------------------------------
The SigLIP feature the VLM-delta trial currently uses is a generic contrastive
embedding: nothing in it knows that a *question about the class* was asked.  The
answer state is label-aligned by construction, and it is where a VLM can use the
world knowledge (context, text in the image, part/whole reasoning) a pure image
encoder has no access to.  Whether that buys a better ``p`` head is an empirical
question, answered by the same criterion ``train_vlm_p_head.py`` already uses:
held-out real-val top-1.

What this module deliberately does NOT do
-----------------------------------------
It computes no Frechet distance and owns no reference statistics.  It is *not*
an FD judge; it is only the representation ``z = E_VLM(x)`` that the frozen ``p``
head and the online ``q`` head sit on top of.

Memory-safe first-order gradient injection
------------------------------------------
Identical in structure to ``vlm_judge.BinaryVLMJudge.conditional_loss``, and for
the same reason: a 7B activation graph must never coexist with the FD judges'
graphs at ``loss.backward()`` time.  :meth:`QwenAnswerStateExtractor.vjp_surrogate`

1. makes a detached leaf ``x' = stopgrad(x).requires_grad_(True)`` per microbatch;
2. runs the VLM, hands ``z`` to the caller's ``term_fn`` (the ``log q - log p``
   scalar), and immediately takes the image VJP ``g = d term / d x'``;
3. releases the 7B graph; and
4. returns ``stopgrad(term) + <x - stopgrad(x), stopgrad(g)>``,

which has exactly the term's value and exactly its first derivative with respect
to the generator's images, while no VLM activation survives into the generator
backward pass.  This is an exact first-order VJP surrogate, not a higher-order
differentiable loss.

Measured on one A100-80GB at 256px (81 visual tokens, 114 prompt tokens total):
~46 img/s forward-only, ~20 img/s with the image VJP at microbatch 24 (41 GB
peak), ~18 img/s at microbatch 12 (29 GB peak).
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch


logger = logging.getLogger("qwen_answer_state")


# The backend tag written into p-head checkpoints.  ``timm`` (the implicit value
# for every checkpoint that predates this file) means "a TimmReprModel that is
# also an FD judge"; this one means "a prompted VLM answer state".
QWEN_BACKEND = "qwen_answer_state"

DEFAULT_QWEN_MODEL = (
    "/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/"
    "snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
)

# The question. It names no class, so one rendered prompt serves the whole batch
# and every sequence has identical length -- there is no padding to reason about.
DEFAULT_ANSWER_PROMPT = "What is the ImageNet class of this image? Answer:"

# Candidate layers cached by the offline p-head fit.  The last layer is
# next-token-specialised (it has to encode "emit *this* token next"); a
# late-middle layer is often the better linear probe.  Caching several costs one
# extra fp16 array each and exactly zero extra VLM forwards, so the choice is
# measured rather than guessed.
DEFAULT_PROBE_LAYERS = (14, 18, 21, 24, 26, 28)

_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def prompt_checksum(rendered_prompt: str) -> str:
    """Content hash of the *rendered* chat prompt.

    Hashing the rendered string rather than the question alone means a changed
    chat template -- a different system message, a different generation prompt --
    invalidates a p head just as a changed question does.  They move ``z`` by
    exactly the same mechanism.
    """
    return hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()


class QwenAnswerStateExtractor(torch.nn.Module):
    """Frozen prompted VLM exposing ``z = h_layer[last prompt token]``.

    Exposes the same surface the FD judges do -- ``forward(images) -> (z, z)``,
    ``feat_dim``, ``target_size`` -- so ``extract_judge_features`` and the
    offline feature-cache path work on it unchanged, for either ``pool_type``.

    Parameters
    ----------
    model_name_or_path:
        Local model directory, or an already fully cached Hub ID.  Network
        downloads are disabled.
    prompt:
        The question text.  It must not contain the class name: ``z`` has to be
        the model's *answer*, not a re-encoding of a label handed to it.
    layer:
        Index into the model's ``hidden_states`` tuple, HuggingFace's own
        convention: ``0`` is the embedding output, ``k`` is the output of
        decoder layer ``k-1``, and the last index (28 for Qwen2.5-VL-7B, also
        reachable as ``-1``) additionally has the final RMSNorm applied -- that
        last one is exactly what the LM head consumes.
    image_size:
        Side length the images are expected at.  Only validated, never resized
        here: the processor owns the resize, and ``target_size`` reports the
        resolution it actually produces.
    microbatch_size:
        Images per VLM forward inside :meth:`vjp_surrogate`.
    max_samples_per_step:
        ``0`` scores every image handed to :meth:`vjp_surrogate`; ``K > 0``
        scores a round-robin subset of K, which is an unbiased estimate because
        generated batch elements are exchangeable.
    """

    is_answer_state_extractor = True
    backend = QWEN_BACKEND

    def __init__(
        self,
        model_name_or_path: Union[str, Path] = DEFAULT_QWEN_MODEL,
        *,
        prompt: str = DEFAULT_ANSWER_PROMPT,
        layer: int = -1,
        image_size: int = 256,
        dtype: str = "bf16",
        device: Optional[Union[str, torch.device]] = None,
        attn_implementation: Optional[str] = "sdpa",
        processor_kwargs: Optional[Dict[str, object]] = None,
        trust_remote_code: bool = False,
        microbatch_size: int = 12,
        max_samples_per_step: int = 0,
    ) -> None:
        super().__init__()

        if not torch.cuda.is_available():
            raise RuntimeError(
                "QwenAnswerStateExtractor requires CUDA: the VLM must share the "
                "current rank's GPU with the differentiable generator."
            )
        if device is None or str(device) == "cuda":
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"a CUDA device is required, got {device}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.index != torch.cuda.current_device():
            raise ValueError(
                f"Requested {device}, but this process's current CUDA device is "
                f"cuda:{torch.cuda.current_device()}. Set the distributed rank "
                "device before constructing the extractor."
            )

        dtype_key = str(dtype).lower()
        if dtype_key not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
        compute_dtype = _DTYPES[dtype_key]
        if compute_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "dtype='bf16' was requested but this device reports no bfloat16 "
                "support; use dtype='fp16'."
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if int(microbatch_size) < 1:
            raise ValueError("microbatch_size must be >= 1")
        if int(max_samples_per_step) < 0:
            raise ValueError("max_samples_per_step must be >= 0")
        if int(image_size) < 1:
            raise ValueError("image_size must be positive")

        self.vlm_device = device
        self.compute_dtype = compute_dtype
        self.prompt = prompt
        self.image_size = int(image_size)
        self.microbatch_size = int(microbatch_size)
        self.max_samples_per_step = int(max_samples_per_step)
        self.model_ref = str(model_name_or_path)
        self.last_stats: Dict[str, float] = {}
        self.register_buffer(
            "_sample_cursor", torch.zeros((), dtype=torch.long), persistent=False
        )

        self._load(trust_remote_code, attn_implementation, processor_kwargs)

        num_layers = int(self.vlm.config.text_config.num_hidden_layers)
        # hidden_states has num_layers + 1 entries: embeddings, then one per layer.
        self.num_hidden_states = num_layers + 1
        self.layer = self.normalize_layer(layer)

        self.rendered_prompt = self._render_prompt(prompt)
        self.prompt_sha256 = prompt_checksum(self.rendered_prompt)

        self.feat_dim = int(self.vlm.config.text_config.hidden_size)
        self.target_size, self.num_visual_tokens = self._probe_geometry()

        logger.info(
            "[QwenAnswerState] %s | layer %d/%d | d=%d | %dpx -> %dpx "
            "(%d visual tokens) | prompt sha %s | %s",
            self.model_ref, self.layer, num_layers, self.feat_dim,
            self.image_size, self.target_size, self.num_visual_tokens,
            self.prompt_sha256[:16], repr(self.prompt),
        )

    # -- construction helpers -------------------------------------------------

    def _load(self, trust_remote_code, attn_implementation, processor_kwargs) -> None:
        try:
            import transformers
            from packaging.version import Version
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "QwenAnswerStateExtractor needs transformers>=5.5, packaging and "
                "torchvision in the training environment."
            ) from exc
        try:
            installed = Version(transformers.__version__.split("+", 1)[0])
        except Exception as exc:
            raise RuntimeError(
                f"Could not parse transformers version {transformers.__version__!r}"
            ) from exc
        if installed < Version("5.5.0"):
            raise RuntimeError(
                "transformers>=5.5.0 is required for the torchvision processor "
                "backend and differentiable tensor preprocessing; found "
                f"{transformers.__version__}."
            )

        proc_kwargs = dict(processor_kwargs or {})
        forbidden = {"backend", "image_processor_backend", "local_files_only",
                     "trust_remote_code"}
        overlap = forbidden.intersection(proc_kwargs)
        if overlap:
            raise ValueError(
                "processor_kwargs may not override gradient/safety options: "
                + ", ".join(sorted(overlap))
            )

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_ref,
                image_processor_backend="torchvision",
                local_files_only=True,
                trust_remote_code=bool(trust_remote_code),
                **proc_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the local VLM processor from {self.model_ref!r}. "
                "Cache the complete processor first; runtime downloads are "
                "intentionally disabled."
            ) from exc

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            raise TypeError(
                f"{type(self.processor).__name__} has no image_processor and "
                "cannot serve as an image-to-text VLM processor."
            )
        backend = getattr(image_processor, "backend", None)
        if backend != "torchvision":
            raise RuntimeError(
                "The loaded image processor is not using Transformers' "
                f"torchvision backend (reported backend={backend!r}). A PIL/NumPy "
                "backend would detach the generated images."
            )
        if not hasattr(self.processor, "apply_chat_template"):
            raise TypeError(
                f"{type(self.processor).__name__} has no apply_chat_template; "
                "use an instruct/chat VLM checkpoint."
            )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise TypeError("The VLM processor has no tokenizer")
        # Every prompt here is identical so no padding is emitted in practice,
        # but left padding is what makes hidden_states[:, -1] the answer position
        # if that ever stops being true.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    "The VLM tokenizer has neither a pad token nor an EOS token "
                    "usable for left padding."
                )
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: Dict[str, object] = {
            "dtype": self.compute_dtype,
            "local_files_only": True,
            "trust_remote_code": bool(trust_remote_code),
        }
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        try:
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                self.model_ref, **load_kwargs
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the local image-to-text model from "
                f"{self.model_ref!r}. Confirm it is supported by "
                "AutoModelForImageTextToText and that all shards are present."
            ) from exc

        self.vlm.to(self.vlm_device)
        self.vlm.eval()
        self.vlm.requires_grad_(False)
        if hasattr(self.vlm.config, "use_cache"):
            self.vlm.config.use_cache = False

    def _render_prompt(self, question: str) -> str:
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question}],
        }]
        try:
            rendered = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as exc:
            raise RuntimeError(
                "The VLM chat template could not format a one-image message. Use "
                "an instruct checkpoint whose processor has a multimodal "
                "chat_template."
            ) from exc
        if not isinstance(rendered, str) or not rendered:
            raise RuntimeError("apply_chat_template returned no prompt string")
        return rendered

    @torch.inference_mode()
    def _probe_geometry(self) -> Tuple[int, int]:
        """Measure the resolution and token count the processor actually produces."""
        probe = torch.zeros(1, 3, self.image_size, self.image_size,
                            device=self.vlm_device)
        encoded = self.processor(
            text=[self.rendered_prompt], images=[probe[0]], padding=True,
            return_tensors="pt", do_rescale=False, device=self.vlm_device,
        )
        grid = dict(encoded).get("image_grid_thw")
        if grid is None:
            raise RuntimeError(
                "The processor returned no image_grid_thw; this extractor targets "
                "the Qwen2.5-VL family."
            )
        _, patch_h, patch_w = (int(v) for v in grid[0].tolist())
        if patch_h != patch_w:
            raise RuntimeError(
                f"The processor produced a non-square {patch_h}x{patch_w} patch "
                "grid for a square input; z's geometry would depend on the image."
            )
        patch_size = int(self.vlm.config.vision_config.patch_size)
        merge = int(getattr(self.vlm.config.vision_config, "spatial_merge_size", 2))
        return patch_h * patch_size, (patch_h * patch_w) // (merge * merge)

    def normalize_layer(self, layer: int) -> int:
        """Resolve a possibly negative ``hidden_states`` index to ``[0, L]``."""
        index = int(layer)
        if index < 0:
            index += self.num_hidden_states
        if not 0 <= index < self.num_hidden_states:
            raise ValueError(
                f"layer {layer} is outside the valid hidden_states range "
                f"[-{self.num_hidden_states}, {self.num_hidden_states - 1}]"
            )
        return index

    # -- frozen ---------------------------------------------------------------

    def train(self, mode: bool = True) -> "QwenAnswerStateExtractor":
        """Stay deterministic even if a parent module calls ``train()``."""
        super().train(False)
        self.vlm.eval()
        return self

    # -- the forward ----------------------------------------------------------

    def _prepare_inputs(self, images01: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch = images01.shape[0]
        try:
            encoded = self.processor(
                text=[self.rendered_prompt] * batch,
                images=list(images01.unbind(0)),
                padding=True,
                return_tensors="pt",
                do_rescale=False,
                device=self.vlm_device,
            )
        except Exception as exc:
            raise RuntimeError(
                "Differentiable VLM preprocessing failed. Inputs must be batched "
                "RGB torch tensors in [0,1] and the processor must use the "
                "torchvision backend. do_rescale=False is deliberate: generated "
                "images are already in [0,1] and must not be divided by 255."
            ) from exc

        inputs: Dict[str, torch.Tensor] = {}
        for key, value in dict(encoded).items():
            if torch.is_tensor(value):
                inputs[key] = value.to(self.vlm_device)
            else:
                raise TypeError(
                    f"Processor output {key!r} has unsupported type "
                    f"{type(value).__name__}; expected torch.Tensor"
                )

        pixel_keys = [k for k, v in inputs.items()
                      if k.startswith("pixel_values") and v.is_floating_point()]
        if not pixel_keys:
            raise RuntimeError("The VLM processor returned no floating pixel_values")
        if images01.requires_grad:
            detached = [k for k in pixel_keys if not inputs[k].requires_grad]
            if detached:
                raise RuntimeError(
                    f"The VLM processor detached the image graph for {detached}. "
                    "Verify transformers>=5.5, the torchvision image-processor "
                    "backend, tensor inputs and do_rescale=False. Continuing "
                    "would produce no generator gradient."
                )
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None and not bool(attention_mask[:, -1].all()):
            raise RuntimeError(
                "The processor produced trailing padding despite "
                "padding_side='left'; hidden_states[:, -1] would not be the "
                "answer position for every row."
            )
        return inputs

    def _validate_images(self, images: torch.Tensor) -> None:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must be [B,3,H,W], got {tuple(images.shape)}")
        if not images.is_floating_point():
            raise TypeError(f"images must be floating point, got {images.dtype}")
        if images.shape[0] == 0:
            raise ValueError("the image batch must not be empty")
        if images.device.type != self.vlm_device.type or (
            images.device.index is not None
            and images.device.index != self.vlm_device.index
        ):
            raise ValueError(
                f"Images are on {images.device} but the VLM is on {self.vlm_device}"
            )
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            raise ValueError(
                f"this extractor was configured for {self.image_size}px images; "
                f"got {images.shape[-2]}x{images.shape[-1]}. z depends on the "
                "input resolution, so the p head would not transfer."
            )

    def _autocast(self):
        if self.compute_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast(device_type="cuda", dtype=self.compute_dtype,
                                  enabled=True)
        return nullcontext()

    def answer_states(self, images: torch.Tensor,
                      layers: Optional[Iterable[int]] = None,
                      ) -> Dict[int, torch.Tensor]:
        """``{layer: z}`` for one batch, in a single VLM forward.

        Every requested layer comes from the same forward, so probing six
        candidate layers costs six fp16 arrays and zero extra VLM compute.
        """
        self._validate_images(images)
        wanted = ([self.layer] if layers is None
                  else [self.normalize_layer(int(k)) for k in layers])
        if not wanted:
            raise ValueError("no layers requested")

        inputs = self._prepare_inputs(images)
        with self._autocast():
            outputs = self.vlm(**inputs, use_cache=False, return_dict=True,
                               output_hidden_states=True, logits_to_keep=1)
        hidden = getattr(outputs, "hidden_states", None)
        if hidden is None:
            raise RuntimeError(
                "The VLM forward returned no hidden_states despite "
                "output_hidden_states=True."
            )
        if len(hidden) != self.num_hidden_states:
            raise RuntimeError(
                f"expected {self.num_hidden_states} hidden states, got "
                f"{len(hidden)}; the layer index convention would be wrong"
            )
        out = {}
        for index in wanted:
            state = hidden[index]
            if state.ndim != 3 or state.shape[0] != images.shape[0]:
                raise RuntimeError(
                    f"hidden_states[{index}] has shape {tuple(state.shape)} for "
                    f"{images.shape[0]} images"
                )
            out[index] = state[:, -1, :].float()
        return out

    def forward(self, images: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Judge interface: ``(primary, secondary)``, both the answer state.

        The two are the same tensor on purpose -- there is no ``cls``/``avg``
        distinction for a single hidden state -- so either ``pool_type`` selects
        the same feature and no caller has to special-case this backend.
        """
        z = self.answer_states(images)[self.layer]
        return z, z

    # -- first-order VJP surrogate -------------------------------------------

    def select_indices(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Which rows of a batch to score, round-robin across steps.

        Public because callers that skip the VJP (a zero-weight warmup step, say)
        still have to advance the same cursor, or the subset would stop being a
        round robin.
        """
        count = (batch_size if self.max_samples_per_step == 0
                 else min(batch_size, self.max_samples_per_step))
        if count == batch_size:
            return torch.arange(batch_size, device=device)
        start = int(self._sample_cursor.item()) % batch_size
        indices = (torch.arange(count, device=device) + start) % batch_size
        self._sample_cursor.fill_((start + count) % batch_size)
        return indices

    @torch._dynamo.disable
    def vjp_surrogate(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        term_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        microbatch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate ``term_fn(z, labels)`` and inject its first-order image VJP.

        ``term_fn`` receives the live ``z`` for one microbatch and must return a
        scalar **already divided by the caller's global denominator**, so that
        summing the per-microbatch terms over microbatches and over ranks gives
        the intended mean.  Its gradient with respect to ``z`` is what reaches
        the generator.

        Returns ``(surrogate, z_detached, selected)``:

        * ``surrogate`` -- value equal to this rank's share of the term, gradient
          with respect to ``images`` exactly the term's;
        * ``z_detached`` -- the same forward's features, free of charge, for the
          ``q`` replay buffer and every diagnostic;
        * ``selected`` -- which rows of ``images`` were scored.
        """
        self._validate_images(images)
        if not images.requires_grad:
            raise RuntimeError(
                "images.requires_grad is False: vjp_surrogate must receive the "
                "live generated image tensor before it is detached."
            )
        if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
            raise ValueError(
                f"labels must be [{images.shape[0]}], got {tuple(labels.shape)}"
            )
        if labels.is_floating_point() or labels.is_complex():
            raise TypeError(f"labels must be an integer tensor, got {labels.dtype}")
        if not bool(torch.isfinite(images).all()):
            raise ValueError("images contains NaN or Inf")
        image_min = float(images.detach().amin())
        image_max = float(images.detach().amax())
        if image_min < -1e-4 or image_max > 1.0001:
            raise ValueError(
                f"images must be in [0,1], observed [{image_min:.5g}, {image_max:.5g}]"
            )

        step = int(microbatch_size or self.microbatch_size)
        if step < 1:
            raise ValueError("microbatch_size must be >= 1")
        selected = self.select_indices(images.shape[0], images.device)

        surrogate = images.new_zeros(())
        z_chunks = []
        for start in range(0, selected.numel(), step):
            mb_indices = selected[start:start + step]
            mb_labels = labels.index_select(0, mb_indices).long()
            # This leaf owns only a tiny image microbatch, never the generator's
            # graph, so the 7B activations die at the end of this iteration.
            leaf = images.index_select(0, mb_indices).detach().requires_grad_(True)

            z = self.answer_states(leaf)[self.layer]
            term = term_fn(z, mb_labels)
            if term.ndim != 0:
                raise ValueError(
                    f"term_fn must return a scalar, got shape {tuple(term.shape)}"
                )
            leaf_vjp = torch.autograd.grad(term, leaf, create_graph=False,
                                           retain_graph=False)[0]
            chunk_value = term.detach()
            z_chunks.append(z.detach())
            del z, term

            original = images.index_select(0, mb_indices)
            coupling = (original * leaf_vjp.detach()).sum()
            # Numerically chunk_value; d/d original == leaf_vjp.
            surrogate = surrogate + chunk_value + coupling - coupling.detach()

        z_detached = torch.cat(z_chunks, dim=0)
        self.last_stats = {
            "vlm_answer_state_samples": float(selected.numel()),
            "vlm_answer_state_microbatch": float(step),
            "vlm_answer_state_norm": float(z_detached.float().norm(dim=1).mean()),
        }
        return surrogate, z_detached, selected

    # -- zero-shot head initialisation ---------------------------------------

    def class_first_token_ids(self, classnames: Sequence[str]) -> torch.Tensor:
        """First generated token id of each class name, as the answer would start.

        The answer state is the position that emits the *first* token of the
        class name, so the LM-head row for that token is the direction in
        ``z``-space that means "this class" to the frozen model.  Names are
        prefixed with a space because that is how a continuation after
        ``"Answer:"`` is actually tokenised.
        """
        tokenizer = self.processor.tokenizer
        ids = []
        for name in classnames:
            token_ids = tokenizer.encode(" " + str(name).strip(),
                                         add_special_tokens=False)
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.reshape(-1).tolist()
            if not token_ids:
                raise ValueError(f"class name {name!r} tokenised to nothing")
            ids.append(int(token_ids[0]))
        return torch.tensor(ids, dtype=torch.long, device=self.vlm_device)

    @torch.inference_mode()
    def lm_head_rows(self, token_ids: torch.Tensor) -> torch.Tensor:
        """``(C, d)`` LM-head rows for the given tokens -- a zero-shot classifier."""
        lm_head = self.vlm.get_output_embeddings()
        if lm_head is None:
            raise RuntimeError("The VLM exposes no output embedding matrix")
        weight = lm_head.weight
        if int(token_ids.max()) >= weight.shape[0]:
            raise ValueError("a class token id exceeds the output vocabulary")
        return weight.index_select(0, token_ids.to(weight.device)).float().clone()

    # -- identity -------------------------------------------------------------

    def identity(self) -> Dict[str, object]:
        """Everything a p head must agree with to be usable on this extractor."""
        return {
            "vlm_backend": QWEN_BACKEND,
            "vlm_model_name": self.model_ref,
            "vlm_layer": int(self.layer),
            "vlm_prompt_sha256": self.prompt_sha256,
            "feature_dim": int(self.feat_dim),
            "vlm_input_size": int(self.image_size),
            "vlm_target_size": int(self.target_size),
        }
