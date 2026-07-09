# VisAttnSink Baseline — Run Log

This doc tracks the port + dev evaluation of **VisAttnSink**
([See What You Are Told](https://openreview.net/forum?id=7uDI7w5RQA))
on the LTPO codebase. We run **VisAttnSink alone**, not LTPO+VisAttnSink:
the LTPO NES/visual-injection loop is disabled, only VisAttnSink's
visual-attention redistribution survives. The code shares the LTPO repo
only so prompt format / dataloaders match the rest of the paper's table.

## Targets

- Models (in order):
  - `Other_baseline/models/Qwen2.5-VL-7B-Instruct`
  - `Other_baseline/models/Qwen3-VL-8B-Instruct`
  - `Other_baseline/models/Qwen3-VL-4B-Instruct`
  - `Other_baseline/models/Qwen2.5-VL-3B-Instruct`
- Datasets (dev splits): `mmvp_dev`, `mmstar_dev`, `mm_math_dev`,
  `math_vista_dev`, `math_vision_dev`, `hallusion_dev`,
  `scienceqa_dev` (under `LTPO/mllm_data/`).
- Hyper-params (from `Other_baseline/VisAttnSink/A_exps/lv1.5_7b.yml`):
  `tau=20`, `rho=0.5`, `summ=0.2`, `p=0.6`, `except_last_layer=1`.

## Files added

| File | Purpose |
| --- | --- |
| `LTPO/visattnsink_core.py` | Algorithm core: DimProspector / HeadFork / VARProcessor + MetadataStation, ported from `Other_baseline/VisAttnSink/src/logic`. |
| `LTPO/visattnsink_patch.py` | Monkey-patches `eager_attention_forward` and decoder-layer pre-hooks for `qwen2_5_vl` and `qwen3_vl` text models. |
| `LTPO/main_vl_visattnsink.py` | Entry point, mirroring `main_vl_dmlr_final.py`'s `--eval_baseline` path. |
| `LTPO/scripts/run_visattnsink_dev.sh` | Sweep launcher over 4 models × 7 datasets. |

## Algorithm port (high-level)

Original (LLaVA-1.5 / LLaMA-2):
1. **DimProspector** flags *sink tokens* via fixed `DIM_SINK` indices.
2. **HeadFork** flags (head, query) coords that over-concentrate
   visual attention on visual sinks (portion ≤ ρ ∧ summation ≥ summ).
3. **VARProcessor** scales attn to sink keys by `p`, redistributes the
   freed `(1-p) · sink_mass` proportionally to non-sink visual tokens.

### Adaptations for Qwen2.5-VL / Qwen3-VL

- `DIM_SINK` is LLaMA-specific. We instead detect sink dims *adaptively*
  per forward: take `max_dim |rmsnorm(hs)|` and flag tokens whose value
  exceeds `tau`. Captures the "two outlier dims" empirical observation
  without requiring per-model calibration.
- `attn_implementation="eager"` is forced so we can monkey-patch
  `eager_attention_forward` in both `qwen2_5_vl` and `qwen3_vl`. After
  softmax we run HeadFork + VARProcessor on `attn_weights`, then matmul
  with V.
- Decoder layers get a `forward_pre_hook` that captures `hidden_states`
  and runs DimProspector for their `layer_idx`.
- Image-token span is recovered from positions of `image_token_id` in
  the original `input_ids` (stored once in `MetadataStation` at the
  beginning of each sample).
- During cached decoding (`q_len==1`) the algorithm still operates over
  the full key span, so sink positions stay at their absolute indices.

### Prompt format

Identical to `main_vl_dmlr_final.py --eval_baseline`:
- LTPO `SYSTEM_PROMPT` (the `<think>/<answer>` instruction).
- User turn: `{"type":"image"}` + `{"type":"text","text":question}`.
- Forced assistant prefix `\boxed{`.
- Answer extraction: prefers `<answer>...</answer>`, else last `\boxed{...}`.
- Verification: `verify_solution_equivalence` (LLM-as-judge, qwen-max).

## Status log

| Time | Event |
| --- | --- |
| 2026-05-22 17:31 | All 8× H200 saturated by root's `vllm serve Llama-3.1-70B` (8 ports, 100% util, 139GB/143GB each). Cannot launch GPU jobs yet. |
| 2026-05-22 17:35 | Read LTPO baseline path, VisAttnSink logic, transformers 4.57.1 Qwen2.5/3-VL attention internals. |
| 2026-05-22 17:40 | Started writing port. |
| 2026-05-22 17:41 | Port complete (`visattnsink_core.py`, `visattnsink_patch.py`, `main_vl_visattnsink.py`, `scripts/run_visattnsink_dev.sh`). Import smoke test passed. Background poller `scripts/wait_and_launch_visattnsink.sh` launched (PID 1890697); it waits until ≥4 GPUs have ≥30 GB free, then runs the 4-model × 7-dataset sweep. Currently 0/4 GPUs eligible. |
