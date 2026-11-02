"""
Dynamic RoBERTa -> GPT-2 model.

Implements Sections 3 and 4 of the technical design:

* frozen RoBERTa features are consumed from the offline cache,
* a trainable ``Linear(768, 1024) + LayerNorm(1024)`` bridge,
* GPT-2 Medium with dynamic early exit (``NL <= 24``),
* up to 24 appended zero-residual blocks (``NL > 24``),
* dynamic attention-head masking without variance rescaling,
* last-valid-token pooling,
* a dropout + linear classification head.

Transformers 5.x removed ``head_mask`` from ``GPT2Attention``.  Head masking is
therefore implemented as a forward-pre-hook on each attention output
projection, which zeroes the flattened head slices before ``c_proj``.  This is
mathematically identical to zeroing head outputs directly and introduces no
variance rescaling.

Backbone contract
-----------------
The design pins ``d_model = 1024``, 16 heads and 24 base blocks (GPT-2 Medium).
A checkpoint that violates this contract cannot be bridged, so it is rejected
with :class:`DimensionMismatchError` instead of silently producing a
mixed-width model.  Tests and ablations can request a randomly initialised
backbone through ``load_pretrained=False``, which uses the same 1024/16/24
shape without downloading weights.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from src.config import Config, replace_file

LOGGER = logging.getLogger(__name__)

GPT2_BASE_LAYERS = 24
GPT2_HEADS = 16
GPT2_HEAD_DIM = 64
D_MODEL = 1024
MAX_TOTAL_LAYERS = 48


class DimensionMismatchError(ValueError):
    pass


class InvalidArchitectureError(ValueError):
    pass


class EmptyMaskError(ValueError):
    pass


class HeadDimensionError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Bridge and heads
# ---------------------------------------------------------------------------
class ContinuousBridge(nn.Module):
    """Linear(768 -> 1024) followed by LayerNorm(1024)."""

    def __init__(self, d_in: int = 768, d_out: int = 1024) -> None:
        super().__init__()
        self.projection = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out, eps=1e-5)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.projection.in_features:
            raise DimensionMismatchError(
                f"expected {self.projection.in_features} features, "
                f"got {features.shape[-1]}"
            )
        return self.norm(self.projection(features))


class ClassificationHead(nn.Module):
    def __init__(self, d_in: int = 1024, d_out: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.shape[-1] != self.linear.in_features:
            raise HeadDimensionError(
                f"expected pooled dimension {self.linear.in_features}, "
                f"got {pooled.shape[-1]}"
            )
        return self.linear(self.dropout(pooled))


# ---------------------------------------------------------------------------
# Backbone configuration
# ---------------------------------------------------------------------------
def _backbone_config(config: Config, load_pretrained: bool):
    """Return a GPT-2 config pinned to the design's 1024/16/24 contract."""
    from transformers import GPT2Config

    if load_pretrained:
        hf_config = GPT2Config.from_pretrained(
            config.gpt2_model, revision=config.gpt2_revision
        )
        problems = []
        if hf_config.n_embd != config.d_model:
            problems.append(f"n_embd={hf_config.n_embd} (expected {config.d_model})")
        if hf_config.n_head != GPT2_HEADS:
            problems.append(f"n_head={hf_config.n_head} (expected {GPT2_HEADS})")
        if hf_config.n_layer < GPT2_BASE_LAYERS:
            problems.append(
                f"n_layer={hf_config.n_layer} (expected >= {GPT2_BASE_LAYERS})"
            )
        if problems:
            raise DimensionMismatchError(
                f"{config.gpt2_model!r} does not satisfy the GPT-2 Medium "
                f"backbone contract: {', '.join(problems)}. Use "
                f"gpt2_model='gpt2-medium' or load_pretrained=False."
            )
    else:
        hf_config = GPT2Config(
            vocab_size=50257,
            n_positions=max(1024, config.max_seq_len),
            n_embd=config.d_model,
            n_layer=GPT2_BASE_LAYERS,
            n_head=GPT2_HEADS,
            n_inner=4 * config.d_model,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
        )

    hf_config.n_positions = max(int(hf_config.n_positions), int(config.max_seq_len))
    hf_config.embd_pdrop = 0.1
    hf_config.attn_pdrop = 0.1
    hf_config.resid_pdrop = 0.1
    # Eager attention keeps the masked-softmax path explicit and deterministic.
    hf_config._attn_implementation = "eager"
    return hf_config


# ---------------------------------------------------------------------------
# Dynamic GPT-2
# ---------------------------------------------------------------------------
class DynamicGPT2(nn.Module):
    def __init__(self, config: Config, load_pretrained: bool = True) -> None:
        super().__init__()
        self.config = config
        self.nah = GPT2_HEADS
        self.nl = min(GPT2_BASE_LAYERS, config.max_layers or GPT2_BASE_LAYERS)
        self.pooling = config.pooling
        self.head_rescale = config.head_rescale

        from transformers import GPT2Model

        hf_config = _backbone_config(config, load_pretrained)

        if load_pretrained:
            self.base: GPT2Model = GPT2Model.from_pretrained(
                config.gpt2_model,
                config=hf_config,
                revision=config.gpt2_revision,
                attn_implementation="eager",
            )
        else:
            self.base = GPT2Model(hf_config)

        # The token embedding matrix is never used because all inputs arrive
        # through `inputs_embeds`.  Dropping the module frees 51.46M params.
        self.base.wte = None  # type: ignore[assignment]

        self.bridge = ContinuousBridge(config.d_roberta, config.d_model)
        self.head_2 = ClassificationHead(config.d_model, 2, dropout=0.1)
        self.head_4 = ClassificationHead(config.d_model, 4, dropout=0.1)

        # One fixed set of up to 24 appended blocks.  Their output projections
        # start at exact zero, so they compute the identity at initialisation.
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block

        block_config = copy.deepcopy(hf_config)
        n_appended = max(0, int(config.max_layers or GPT2_BASE_LAYERS) - GPT2_BASE_LAYERS)
        n_appended = min(n_appended, MAX_TOTAL_LAYERS - GPT2_BASE_LAYERS)
        self.appended = nn.ModuleList(
            [
                GPT2Block(block_config, layer_idx=GPT2_BASE_LAYERS + i)
                for i in range(n_appended)
            ]
        )
        self._zero_residual_init(self.appended)

        self._install_head_mask_hooks()
        self.set_head_count(GPT2_HEADS)
        self.set_dropout(0.1)
        self.gradient_checkpointing = False
        self._causal_mask_fn = _resolve_causal_mask_fn()

    # -- initialisation helpers -------------------------------------------
    @staticmethod
    def _zero_residual_init(blocks: nn.ModuleList) -> None:
        for block in blocks:
            for module_name in ("attn", "mlp"):
                module = getattr(block, module_name)
                projection = getattr(module, "c_proj", None)
                if projection is not None:
                    nn.init.zeros_(projection.weight)
                    nn.init.zeros_(projection.bias)

    def _install_head_mask_hooks(self) -> None:
        def make_hook(model: "DynamicGPT2"):
            def hook(module, inputs):
                if not inputs:
                    return None
                x = inputs[0]
                if x.shape[-1] != D_MODEL:
                    return None
                if model.nah >= GPT2_HEADS and not model.head_rescale:
                    return None
                x = x.clone()
                active = model.nah * GPT2_HEAD_DIM
                if model.head_rescale and model.nah > 0:
                    x[..., :active] = x[..., :active] * (GPT2_HEADS / model.nah)
                x[..., active:] = 0.0
                return (x,) + tuple(inputs[1:])

            return hook

        hook = make_hook(self)
        for block in list(self.base.h) + list(self.appended):
            block.attn.c_proj.register_forward_pre_hook(hook)

    def set_head_count(self, nah: int) -> None:
        if not (1 <= nah <= GPT2_HEADS):
            raise InvalidArchitectureError(f"NAH must be in [1, 16], got {nah}")
        self.nah = int(nah)

    def set_layer_count(self, nl: int) -> None:
        if not (1 <= nl <= MAX_TOTAL_LAYERS):
            raise InvalidArchitectureError(f"NL must be in [1, 48], got {nl}")
        max_available = len(self.base.h) + len(self.appended)
        if nl > max_available:
            raise InvalidArchitectureError(
                f"NL={nl} requested but this instance only carries {max_available} blocks"
            )
        self.nl = int(nl)

    def set_dropout(self, dropout: float) -> None:
        self.base.drop.p = dropout
        self.base.config.embd_pdrop = dropout
        self.base.config.attn_pdrop = dropout
        self.base.config.resid_pdrop = dropout
        for block in list(self.base.h) + list(self.appended):
            block.attn.attn_dropout.p = dropout
            block.attn.resid_dropout.p = dropout
            mlp_dropout = getattr(block.mlp, "dropout", None)
            if mlp_dropout is not None:
                mlp_dropout.p = dropout
        self.head_2.dropout.p = dropout
        self.head_4.dropout.p = dropout

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def _active_blocks(self) -> List[nn.Module]:
        n_base = min(self.nl, len(self.base.h))
        blocks: List[nn.Module] = list(self.base.h[:n_base])
        if self.nl > len(self.base.h):
            blocks += list(self.appended[: self.nl - len(self.base.h)])
        return blocks

    # -- forward -----------------------------------------------------------
    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.config.d_roberta:
            raise DimensionMismatchError(
                f"expected last dim {self.config.d_roberta}, got {features.shape[-1]}"
            )
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must be (B, T)")
        if attention_mask.sum(dim=1).min() <= 0:
            raise EmptyMaskError("every sequence must contain at least one valid token")

        mask = attention_mask.to(features.dtype)
        embeds = self.bridge(features)
        embeds = embeds * mask.unsqueeze(-1)
        batch, length, _ = embeds.shape
        position_ids = torch.arange(length, device=embeds.device).unsqueeze(0)
        position_ids = position_ids.expand(batch, -1)

        hidden = self.base.wpe(position_ids) + embeds
        hidden = self.base.drop(hidden)

        causal_mask = self._causal_mask_fn(
            config=self.base.config,
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )

        for block in self._active_blocks():
            if self.gradient_checkpointing and self.training:

                def run_block(current_block, current_hidden):
                    return current_block(
                        current_hidden,
                        past_key_values=None,
                        attention_mask=causal_mask,
                        encoder_hidden_states=None,
                        encoder_attention_mask=None,
                        use_cache=False,
                    )

                hidden = torch.utils.checkpoint.checkpoint(
                    run_block, block, hidden, use_reentrant=False
                )
            else:
                hidden = block(
                    hidden,
                    past_key_values=None,
                    attention_mask=causal_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    use_cache=False,
                )

        hidden = self.base.ln_f(hidden)
        pooled = self._pool(hidden, attention_mask)
        if self.config.d_out == 2:
            return self.head_2(pooled)
        if self.config.d_out == 4:
            return self.head_4(pooled)
        raise HeadDimensionError(f"unsupported d_out={self.config.d_out}")

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            return summed / counts
        lengths = attention_mask.sum(dim=1).long()
        last_index = (lengths - 1).clamp(min=0)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[rows, last_index]

    # -- persistence -------------------------------------------------------
    def master_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if not key.startswith("base.wte")
        }

    def parameter_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def _resolve_causal_mask_fn():
    try:
        from transformers.masking_utils import create_causal_mask

        return create_causal_mask
    except Exception:  # pragma: no cover - older transformers fallback
        try:
            from transformers.modeling_attn_mask_utils import (
                _prepare_4d_causal_attention_mask,
            )

            def create_causal_mask(
                config,
                inputs_embeds,
                attention_mask,
                past_key_values=None,
                position_ids=None,
                **kwargs,
            ):
                return _prepare_4d_causal_attention_mask(
                    attention_mask, inputs_embeds.shape[:2], inputs_embeds, past_key_values
                )

            return create_causal_mask
        except Exception as exc:
            raise RuntimeError(
                "Could not locate a causal-mask constructor in this transformers "
                "version. Install transformers>=4.45 or <=5.x."
            ) from exc


# ---------------------------------------------------------------------------
# Checkpoint construction / loading
# ---------------------------------------------------------------------------
def save_master_checkpoint(model: DynamicGPT2, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.master_state_dict(),
        "meta": {
            "gpt2_model": model.config.gpt2_model,
            "gpt2_revision": model.config.gpt2_revision,
            "d_roberta": model.config.d_roberta,
            "d_model": model.config.d_model,
            "max_layers": model.config.max_layers,
            "seed_init": model.config.seed_init,
        },
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    replace_file(tmp, path)
    return path


def load_master_checkpoint(model: DynamicGPT2, path: Path | str) -> DynamicGPT2:
    """Load every master tensor that this candidate actually materialises.

    A master stores 48 blocks.  A candidate may materialise 1 base block and no
    appended blocks, so tensors are filtered by name and shape before loading.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if "state_dict" in payload else payload

    target = model.state_dict()
    filtered = {
        key: value
        for key, value in state.items()
        if key in target and tuple(target[key].shape) == tuple(value.shape)
    }
    missing = sorted(set(target) - set(filtered))
    if missing:
        raise DimensionMismatchError(
            f"master checkpoint {path} is missing {len(missing)} candidate "
            f"tensors, e.g. {missing[:5]}"
        )
    model.load_state_dict(filtered, strict=False)
    return model


def build_model(
    config: Config,
    master_path: Path | str | None = None,
    load_pretrained: bool = True,
    nl: int | None = None,
) -> DynamicGPT2:
    """Build a model with only the blocks needed for ``nl``.

    The master checkpoint always stores all 24 appended blocks.  Materialising
    only ``max(0, nl - 24)`` of them keeps a candidate's parameter footprint
    equal to the NL-block formula in Section 8.2 instead of the 48-block
    master footprint.
    """
    target_nl = (
        nl
        if nl is not None
        else min(GPT2_BASE_LAYERS, config.max_layers or GPT2_BASE_LAYERS)
    )
    build_config = Config.from_dict({**config.to_dict(), "max_layers": int(target_nl)})
    model = DynamicGPT2(build_config, load_pretrained=load_pretrained)

    # Keep only the base blocks this candidate can execute.  Without this a
    # NL=1 candidate would still carry all 24 pretrained blocks even though
    # Section 8.2's parameter formula charges it for one.
    if target_nl < GPT2_BASE_LAYERS:
        model.base.h = nn.ModuleList(list(model.base.h[:target_nl]))

    if master_path is not None and Path(master_path).exists():
        load_master_checkpoint(model, master_path)
    model.set_dropout(0.1)
    model.set_head_count(GPT2_HEADS)
    model.set_layer_count(target_nl)
    return model


def state_dict_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
