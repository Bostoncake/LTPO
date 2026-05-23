"""
visattnsink_patch.py — wire VisAttnSink redistribution into transformers'
Qwen2.5-VL and Qwen3-VL eager attention path.

We install two things per backbone:
  1. a custom `eager_attention_forward` (replacing the module's reference)
     that, after softmax, calls HeadFork+VARProcessor before the V matmul.
  2. forward_pre_hooks on each text decoder layer that call DimProspector
     on the incoming hidden state, tagged with the layer's `layer_idx`.

`install_visattnsink(model)` returns a callable that undoes the patch (we
never actually call it — the model is rebuilt per script invocation — but
it is useful for tests).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
from torch import nn

from visattnsink_core import (
    LogicEngine,
    MetadataStation,
    run_dim_prospector,
    run_head_fork,
    run_var_processor,
)


# ---------------------------------------------------------------------------
# Custom eager attention forward (parallels transformers' implementation for
# Qwen2.5-VL / Qwen3-VL text models; both share the same shape conventions).
# ---------------------------------------------------------------------------

def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    b, h, s, d = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(b, h, n_rep, s, d)
        .reshape(b, h * n_rep, s, d)
    )


def _make_eager_forward():
    def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
    ):
        key_states = _repeat_kv(key, module.num_key_value_groups)
        value_states = _repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query.dtype)

        # === VisAttnSink hook ===
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is not None and LogicEngine.enabled and MetadataStation.active:
            with torch.no_grad():
                run_head_fork(attn_weights, layer_idx)
                attn_weights = run_var_processor(attn_weights, layer_idx)
        # === end hook ===

        attn_weights = nn.functional.dropout(
            attn_weights, p=dropout, training=module.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    return eager_attention_forward


# ---------------------------------------------------------------------------
# Decoder-layer forward_pre_hook: capture hidden_states -> DimProspector.
# ---------------------------------------------------------------------------

def _make_layer_pre_hook(layer_idx: int) -> Callable:
    def hook(module, args, kwargs):
        if not (LogicEngine.enabled and MetadataStation.active):
            return None
        hs = None
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hs = args[0]
        elif "hidden_states" in kwargs:
            hs = kwargs["hidden_states"]
        if hs is not None:
            run_dim_prospector(hs.detach(), layer_idx)
        return None

    return hook


# ---------------------------------------------------------------------------
# Public installer.
# ---------------------------------------------------------------------------

def _is_text_decoder_layer(module: nn.Module) -> bool:
    name = module.__class__.__name__
    return name in (
        "Qwen2_5_VLDecoderLayer",
        "Qwen3VLTextDecoderLayer",
    )


def install_visattnsink(model: nn.Module) -> Callable[[], None]:
    """Replace eager attention + register pre-hooks on text decoder layers.

    Returns an `uninstall()` closure that reverses the changes (best-effort).
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []
    patched_modules: List[Tuple[object, str, Callable]] = []

    # Replace module-level eager_attention_forward in both backbones.
    new_forward = _make_eager_forward()
    try:
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as q25
        patched_modules.append((q25, "eager_attention_forward", q25.eager_attention_forward))
        q25.eager_attention_forward = new_forward
    except Exception:
        pass
    try:
        from transformers.models.qwen3_vl import modeling_qwen3_vl as q3
        patched_modules.append((q3, "eager_attention_forward", q3.eager_attention_forward))
        q3.eager_attention_forward = new_forward
    except Exception:
        pass

    # Walk the model, register pre-hook on each text decoder layer.
    for _, module in model.named_modules():
        if _is_text_decoder_layer(module):
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is None:
                # Try the attention sub-module's layer_idx.
                attn = getattr(module, "self_attn", None)
                layer_idx = getattr(attn, "layer_idx", None)
            if layer_idx is None:
                continue
            handle = module.register_forward_pre_hook(
                _make_layer_pre_hook(int(layer_idx)), with_kwargs=True
            )
            handles.append(handle)

    def _uninstall() -> None:
        for h in handles:
            h.remove()
        for mod, attr, orig in patched_modules:
            setattr(mod, attr, orig)

    return _uninstall
