"""Lightweight inference profiling utilities.

The profiler counts FLOPs analytically from the tensors that actually flow
through the model. Linear/Conv FLOPs are recorded with module hooks, while
attention matmul FLOPs are recorded explicitly from patched eager attention.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import nn


_ACTIVE_PROFILER: Optional["InferenceProfiler"] = None


def set_active_profiler(profiler: Optional["InferenceProfiler"]) -> None:
    global _ACTIVE_PROFILER
    _ACTIVE_PROFILER = profiler


def get_active_profiler() -> Optional["InferenceProfiler"]:
    return _ACTIVE_PROFILER


def _first_tensor(obj: Any) -> Optional[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(obj, dict):
        for item in obj.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _is_text_decoder_layer(module: nn.Module) -> bool:
    return module.__class__.__name__ in (
        "Qwen2_5_VLDecoderLayer",
        "Qwen3VLTextDecoderLayer",
    )


def _is_vision_root(name: str, module: nn.Module) -> bool:
    cls_name = module.__class__.__name__.lower()
    lname = name.lower()
    return (
        name in ("visual", "vision_tower", "vision_model")
        or "visiontransformer" in cls_name
        or "vision_model" in lname
    )


@dataclass
class InferenceProfiler:
    """Collect low-overhead generation FLOP/call counters."""

    enabled: bool = True
    total_flops: int = 0
    module_flops: int = 0
    linear_flops: int = 0
    conv_flops: int = 0
    attention_flops: int = 0
    vas_flops: int = 0
    model_forward_calls: int = 0
    decoder_layer_forward_calls: int = 0
    vision_forward_calls: int = 0
    linear_calls: int = 0
    conv_calls: int = 0
    attention_calls: int = 0
    vas_calls: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    vas_time_s: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def install(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        self.remove()
        self.handles.append(model.register_forward_pre_hook(self._model_pre_hook))
        seen_vision_roots = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self._linear_hook))
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                self.handles.append(module.register_forward_hook(self._conv_hook))
            if _is_text_decoder_layer(module):
                self.handles.append(
                    module.register_forward_pre_hook(self._decoder_layer_pre_hook)
                )
            elif _is_vision_root(name, module) and id(module) not in seen_vision_roots:
                seen_vision_roots.add(id(module))
                self.handles.append(module.register_forward_pre_hook(self._vision_pre_hook))
        set_active_profiler(self)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if get_active_profiler() is self:
            set_active_profiler(None)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_flops": int(self.total_flops),
            "module_flops": int(self.module_flops),
            "linear_flops": int(self.linear_flops),
            "conv_flops": int(self.conv_flops),
            "attention_flops": int(self.attention_flops),
            "vas_flops": int(self.vas_flops),
            "model_forward_calls": int(self.model_forward_calls),
            "decoder_layer_forward_calls": int(self.decoder_layer_forward_calls),
            "vision_forward_calls": int(self.vision_forward_calls),
            "linear_calls": int(self.linear_calls),
            "conv_calls": int(self.conv_calls),
            "attention_calls": int(self.attention_calls),
            "vas_calls": dict(self.vas_calls),
            "vas_time_s": dict(self.vas_time_s),
        }

    def diff(self, before: Dict[str, Any]) -> Dict[str, Any]:
        after = self.snapshot()
        diff: Dict[str, Any] = {}
        for key, value in after.items():
            if isinstance(value, dict):
                keys = set(value) | set(before.get(key, {}))
                diff[key] = {
                    k: value.get(k, 0) - before.get(key, {}).get(k, 0)
                    for k in sorted(keys)
                }
            else:
                diff[key] = value - before.get(key, 0)
        return diff

    def _add_flops(self, amount: int, bucket: str) -> None:
        amount = int(max(amount, 0))
        self.total_flops += amount
        if bucket == "linear":
            self.module_flops += amount
            self.linear_flops += amount
        elif bucket == "conv":
            self.module_flops += amount
            self.conv_flops += amount
        elif bucket == "attention":
            self.attention_flops += amount
        elif bucket == "vas":
            self.vas_flops += amount

    def _model_pre_hook(self, module: nn.Module, inputs: tuple) -> None:
        self.model_forward_calls += 1

    def _decoder_layer_pre_hook(self, module: nn.Module, inputs: tuple) -> None:
        self.decoder_layer_forward_calls += 1

    def _vision_pre_hook(self, module: nn.Module, inputs: tuple) -> None:
        self.vision_forward_calls += 1

    def _linear_hook(self, module: nn.Linear, inputs: tuple, output: Any) -> None:
        x = _first_tensor(inputs)
        if x is None:
            return
        in_features = int(module.in_features)
        out_features = int(module.out_features)
        if in_features <= 0:
            return
        batch_ops = int(x.numel() // in_features)
        flops = 2 * batch_ops * in_features * out_features
        if module.bias is not None:
            flops += batch_ops * out_features
        self.linear_calls += 1
        self._add_flops(flops, "linear")

    def _conv_hook(self, module: nn.modules.conv._ConvNd, inputs: tuple, output: Any) -> None:
        y = _first_tensor(output)
        if y is None:
            return
        kernel_mul = int(module.in_channels // module.groups)
        for k in module.kernel_size:
            kernel_mul *= int(k)
        flops = 2 * int(y.numel()) * kernel_mul
        if module.bias is not None:
            flops += int(y.numel())
        self.conv_calls += 1
        self._add_flops(flops, "conv")

    def record_attention(self, query: torch.Tensor, key_states: torch.Tensor) -> None:
        """Record QK^T and Attn*V matmul FLOPs for one attention call."""
        if not self.enabled:
            return
        bsz = int(query.shape[0])
        heads = int(query.shape[1])
        q_len = int(query.shape[2])
        head_dim = int(query.shape[3])
        k_len = int(key_states.shape[2])
        flops = 4 * bsz * heads * q_len * k_len * head_dim
        self.attention_calls += 1
        self._add_flops(flops, "attention")

    def record_vas(self, name: str, flops: int = 0, elapsed_s: float = 0.0) -> None:
        if not self.enabled:
            return
        self.vas_calls[name] += 1
        self.vas_time_s[name] += float(elapsed_s)
        self._add_flops(int(flops), "vas")
