"""
visattnsink_core.py — VisAttnSink (See What You Are Told) algorithm core,
ported from Other_baseline/VisAttnSink/src/logic/{logic,constants}.py for
use on Qwen2.5-VL / Qwen3-VL inside the LTPO framework.

Three components (paper Sec. 3):
  * DimProspector: per-layer flags sink tokens whose |rmsnorm(hidden)| is
    abnormally large. Original code uses fixed `DIM_SINK` indices for LLaMA;
    Qwen has no such table, so we use the *adaptive* variant — flag tokens
    whose max-over-hidden-dim normalised magnitude exceeds `tau`.
  * HeadFork: per layer, flags (head, query) coordinates whose visual
    attention over-concentrates on visual sink tokens
    (portion <= rho AND summation >= summ).
  * VARProcessor: on those coordinates, scales attention to sink keys by `p`
    and redistributes (1-p) * sink_mass proportionally to non-sink image keys.

The three operate via global class-level state, exactly as in the upstream
code, so the patched eager-attention forward in `visattnsink_patch.py` can
read DimProspector indices / HeadFork coordinates without threading extra
arguments through transformers internals.

State is reset per sample via `LogicEngine.clear()`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import copy
import time
import torch

from inference_profile import get_active_profiler


# ---------------------------------------------------------------------------
# Metadata for current sample (image span, model config). Mirrors upstream
# `src/stash.MetadataStation` but trimmed to the fields we actually use.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Outlier hidden-dim indices per Qwen-VL backbone.
#
# Calibrated offline by `calibrate_outlier_dims.py`: feed ~6 image+text
# prompts, capture mid-layer hidden states, RMS-normalise, rank dims by
# their median absolute value across all token positions. Every Qwen
# checkpoint we use has a clean two-dim outlier pattern (top-2 ≫ next).
#
# Paper (LLaMA-2-7B) used dims [2533, 1415] with tau=20; the gap between
# top-2 and the rest there is also ~3-4x, identical structure to ours.
DIM_SINK = {
    "Qwen2.5-VL-7B-Instruct": [2570, 458],
    "Qwen2.5-VL-3B-Instruct": [1874, 1819],
    "Qwen3-VL-8B-Instruct":   [1838, 2276],
    "Qwen3-VL-4B-Instruct":   [0, 4],
}


class MetadataStation:
    image_start: int = 0          # absolute key position of the first image patch
    image_end: int = 0            # one past last image patch position
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    active: bool = False

    @classmethod
    def activate(cls, image_start: int, image_end: int,
                 num_hidden_layers: int, num_attention_heads: int) -> None:
        cls.image_start = int(image_start)
        cls.image_end = int(image_end)
        cls.num_hidden_layers = int(num_hidden_layers)
        cls.num_attention_heads = int(num_attention_heads)
        cls.active = True

    @classmethod
    def deactivate(cls) -> None:
        cls.active = False
        cls.image_start = 0
        cls.image_end = 0


def _record_vas(name: str, start_s: float, flops: int = 0) -> None:
    profiler = get_active_profiler()
    if profiler is not None:
        profiler.record_vas(name, flops=flops, elapsed_s=time.perf_counter() - start_s)


# ---------------------------------------------------------------------------
# LogicEngine — central toggle + per-layer state.
# ---------------------------------------------------------------------------

class LogicEngine:
    enabled: bool = False
    tau: float = 20.0
    rho: float = 0.5
    summ: float = 0.2
    p: float = 0.6
    except_last_layer: bool = True
    # Outlier hidden-dim indices used by DimProspector. Set per backbone via
    # `activate(..., dim_sink=DIM_SINK[model_name])`.
    dim_sink: List[int] = [2570, 458]
    # First two layers in LLaVA-1.5's setup are skipped by the original
    # (`sink_select_layers = [2:]`). We replicate that — sinks are unstable
    # in the very first decoder layers.
    skip_first_layers: int = 2

    # DimProspector output: layer_idx -> 1-D tensor of sink-token absolute positions
    sink_indices: Dict[int, torch.Tensor] = {}
    # HeadFork output: layer_idx -> coords [N, 3] of (batch, head, query)
    forked_head: Dict[int, torch.Tensor] = {}

    @classmethod
    def activate(cls, tau: float = 20.0, rho: float = 0.5,
                 summ: float = 0.2, p: float = 0.6,
                 except_last_layer: bool = True,
                 dim_sink: Optional[List[int]] = None,
                 skip_first_layers: int = 2) -> None:
        cls.enabled = True
        cls.tau = float(tau)
        cls.rho = float(rho)
        cls.summ = float(summ)
        cls.p = float(p)
        cls.except_last_layer = bool(except_last_layer)
        cls.skip_first_layers = int(skip_first_layers)
        if dim_sink is not None:
            cls.dim_sink = list(dim_sink)
        cls.sink_indices = {}
        cls.forked_head = {}

    @classmethod
    def clear(cls) -> None:
        cls.sink_indices = {}
        cls.forked_head = {}


# ---------------------------------------------------------------------------
# DimProspector — adaptive sink-token detection.
# ---------------------------------------------------------------------------

def _rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-token RMS-normalisation, matching upstream `DimProspector.rmsnorm`."""
    x = x.to(torch.float32)
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps)


def run_dim_prospector(hidden_states: torch.Tensor, layer_idx: int) -> None:
    """Flag sink token positions for `layer_idx`, paper-faithful.

    Only inspects `LogicEngine.dim_sink` (the calibrated outlier dim indices
    for the current backbone). A token is a sink iff the max RMS-normalised
    |value| over those specific dims exceeds `tau`.

    The original adaptive replacement (max over the entire hidden dim) was
    incorrect: every Qwen token has at least one outlier-register dim with
    large magnitude, so every token got flagged as a sink, which collapsed
    VARProcessor's redistribution and produced fp16 attention overflow.
    """
    start_s = time.perf_counter()
    if not LogicEngine.enabled:
        return
    # First few layers: sink dimensions are unstable; skip (matches upstream).
    if layer_idx < LogicEngine.skip_first_layers:
        LogicEngine.sink_indices[layer_idx] = hidden_states.new_zeros(0, dtype=torch.long)
        _record_vas("dim_prospector", start_s, 0)
        return
    with torch.no_grad():
        norm = _rmsnorm(hidden_states).abs()       # [bsz, seq, dim]
        flops = 5 * int(hidden_states.numel())
        dim_sink = torch.as_tensor(
            LogicEngine.dim_sink, device=norm.device, dtype=torch.long
        )
        # Guard: drop any dim index that's out of range for this model.
        dim_sink = dim_sink[dim_sink < norm.size(-1)]
        if dim_sink.numel() == 0:
            LogicEngine.sink_indices[layer_idx] = norm.new_zeros(0, dtype=torch.long)
            _record_vas("dim_prospector", start_s, flops)
            return
        vals = norm[..., dim_sink]                 # [bsz, seq, K]
        flops += int(norm.size(0) * norm.size(1) * dim_sink.numel())
        peak = vals.max(dim=-1)[0]                  # [bsz, seq]
        mask = (peak > LogicEngine.tau).any(dim=0)  # [seq]
        idx = torch.nonzero(mask, as_tuple=False).flatten()  # [N]
    LogicEngine.sink_indices[layer_idx] = idx.detach()
    _record_vas("dim_prospector", start_s, flops)


# ---------------------------------------------------------------------------
# HeadFork — flag heads with excessive visual-sink concentration.
# ---------------------------------------------------------------------------

def run_head_fork(attn_weights: torch.Tensor, layer_idx: int) -> None:
    """Identify (batch, head, query) coords flagged for redistribution.

    `attn_weights` is [bsz, heads, q_len, k_len] **after softmax**. Sink
    indices for this layer must already be populated by `run_dim_prospector`.
    """
    start_s = time.perf_counter()
    if not LogicEngine.enabled or not MetadataStation.active:
        LogicEngine.forked_head[layer_idx] = attn_weights.new_zeros(0, 3, dtype=torch.long)
        _record_vas("head_fork", start_s, 0)
        return

    im = MetadataStation.image_start
    pa = MetadataStation.image_end - MetadataStation.image_start
    if pa <= 0:
        LogicEngine.forked_head[layer_idx] = attn_weights.new_zeros(0, 3, dtype=torch.long)
        _record_vas("head_fork", start_s, 0)
        return

    sink_inds = LogicEngine.sink_indices.get(layer_idx)
    if sink_inds is None or sink_inds.numel() == 0:
        LogicEngine.forked_head[layer_idx] = attn_weights.new_zeros(0, 3, dtype=torch.long)
        _record_vas("head_fork", start_s, 0)
        return

    # vis_sink = sink tokens inside image span
    sink_inds = sink_inds.to(attn_weights.device)
    vis_mask = (sink_inds >= im) & (sink_inds < im + pa)
    vis_sink_inds = sink_inds[vis_mask]
    if vis_sink_inds.numel() == 0:
        LogicEngine.forked_head[layer_idx] = attn_weights.new_zeros(0, 3, dtype=torch.long)
        _record_vas("head_fork", start_s, 0)
        return

    # k_len may exceed the prompt span during cached decode — fine, the image
    # span lies within the prefilled portion so attn_weights[..., im:im+pa]
    # is always valid.
    k_len = attn_weights.size(-1)
    if im + pa > k_len:
        LogicEngine.forked_head[layer_idx] = attn_weights.new_zeros(0, 3, dtype=torch.long)
        _record_vas("head_fork", start_s, 0)
        return

    image_attn = attn_weights[..., im:im + pa]                       # [b,h,q,pa]
    sink_local = vis_sink_inds - im
    bsz, heads, q_len, _ = image_attn.shape
    flops = int(bsz * heads * q_len * (pa + vis_sink_inds.numel() + 4))
    portion = image_attn[..., sink_local].sum(dim=-1) / (image_attn.sum(dim=-1) + 1e-6)
    summation = image_attn.sum(dim=-1)
    cond = (portion <= LogicEngine.rho) & (summation >= LogicEngine.summ)
    coords = torch.nonzero(cond, as_tuple=False)                     # [N,3]
    LogicEngine.forked_head[layer_idx] = coords.detach()
    _record_vas("head_fork", start_s, flops)


# ---------------------------------------------------------------------------
# VARProcessor — redistribute attention away from sink keys.
# ---------------------------------------------------------------------------

def run_var_processor(attn_weights: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Apply VisAttnSink's attention re-distribution in-place-style.

    Returns the modified attention tensor.
    """
    start_s = time.perf_counter()
    flops = 0
    if not LogicEngine.enabled or not MetadataStation.active:
        return attn_weights

    if (LogicEngine.except_last_layer
            and layer_idx == MetadataStation.num_hidden_layers - 1):
        _record_vas("var_processor", start_s, flops)
        return attn_weights

    im = MetadataStation.image_start
    pa = MetadataStation.image_end - MetadataStation.image_start
    if pa <= 0:
        _record_vas("var_processor", start_s, flops)
        return attn_weights

    coords = LogicEngine.forked_head.get(layer_idx)
    if coords is None or coords.numel() == 0:
        _record_vas("var_processor", start_s, flops)
        return attn_weights

    sink_inds = LogicEngine.sink_indices.get(layer_idx)
    if sink_inds is None or sink_inds.numel() == 0:
        _record_vas("var_processor", start_s, flops)
        return attn_weights

    sink_inds = sink_inds.to(attn_weights.device)
    vis_mask = (sink_inds >= im) & (sink_inds < im + pa)
    vis_sink = sink_inds[vis_mask]
    text_sink = sink_inds[~vis_mask]
    if vis_sink.numel() == 0:
        _record_vas("var_processor", start_s, flops)
        return attn_weights

    k_len = attn_weights.size(-1)
    text_sink = text_sink[(text_sink >= 0) & (text_sink < k_len)]

    p = LogicEngine.p

    # Group coords per head to mirror upstream's per-head loop.
    heads = coords[:, 1].unique()
    for h in heads.tolist():
        sub = coords[coords[:, 1] == h]
        b = sub[:, 0]
        q = sub[:, 2]
        if q.numel() == 0:
            continue
        flops += int(q.numel() * (attn_weights.size(-1) + 4 * pa + 4 * (vis_sink.numel() + text_sink.numel())))

        # selected [Q, K]
        sel = attn_weights[b, h, q, :].clone()
        copied = sel.clone()

        # Decrease attention on sink keys (visual + textual).
        if text_sink.numel() > 0:
            sel[:, text_sink] = sel[:, text_sink] * p
        sel[:, vis_sink] = sel[:, vis_sink] * p

        # Budget freed up.
        budget_vis = copied[:, vis_sink].sum(dim=1) * (1 - p)
        budget_text = (copied[:, text_sink].sum(dim=1) * (1 - p)
                       if text_sink.numel() > 0 else torch.zeros_like(budget_vis))

        # Compute non-sink image weight ratios.
        img_slice = copied[:, im:im + pa].clone()
        img_slice[:, vis_sink - im] = 0
        img_sum = img_slice.sum(dim=1, keepdim=True)
        # Guard against all-zero row.
        ratios = torch.where(
            img_sum > 0,
            img_slice / img_sum.clamp_min(1e-9),
            torch.zeros_like(img_slice),
        ).to(sel.dtype)

        sel[:, im:im + pa] = sel[:, im:im + pa] + (
            (budget_vis + budget_text).view(-1, 1) * ratios
        )

        attn_weights[b, h, q, :] = sel

    _record_vas("var_processor", start_s, flops)
    return attn_weights
