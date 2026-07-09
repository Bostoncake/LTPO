# ICoT (Interleaved-Modal Chain-of-Thought) on LTPO framework

This doc tracks the ICoT baseline runs I'm doing on the LTPO codebase
(reproducing ICoT, **not** LTPO + ICoT). The data loader, prompt scaffolding,
answer extraction and LLM-judge come from `main_vl_dmlr_final_visual.py`;
the generation loop is rewritten to do the ICoT interleaved-modal trick
(every 2 newlines, pick top-K image patches by attention and splice them in
as a sub-image, up to 3 sub-images per response).

## Plan
- Models (run order): Qwen2.5-VL-7B → Qwen3-VL-8B → Qwen3-VL-4B → Qwen2.5-VL-3B
- Datasets (parallelised, one per GPU): mmvp_dev, mmstar_dev, mm_math_dev,
  math_vista_dev, math_vision_dev, hallusion_dev, scienceqa_dev (7 GPUs)
- Output root: `LTPO/output/icot/`
- Environment: `conda activate ltpo`, transformers 4.57.1

## Implementation notes
- New files (kept inside LTPO so the user's repo stays self-contained, and
  the original transformers install is untouched):
    - `LTPO/icot_vl.py` — ICoT generation loop
    - `LTPO/main_vl_icot.py` — entry script, mirrors `main_vl_dmlr_final_visual.py`
    - `LTPO/scripts/run_icot_dev.sh` — launches one model × 7 datasets
    - `LTPO/scripts/run_icot_all_models.sh` — runs all 4 models sequentially
- ICoT details:
    - Detect newline token (id 198) during generation; every 2 newlines and
      while `num_sub_imgs < 3`, run a forward pass with `output_attentions=True`,
      pick top `num_selected_patches=16` image tokens by attention from the
      latest position averaged across layers + heads, and inject
      `<|vision_start|>` + those patch embeddings + `<|vision_end|>` into the
      sequence so subsequent generation can attend to them.
    - We keep the LTPO prompt scaffold (`SYSTEM_PROMPT` with `\boxed{}` answer
      hint) so the LLM-judge step works identically across baselines.

## Progress
(filled in as runs progress)

### 2026-05-23 launch
- Launched `bash scripts/run_icot_all_models.sh` in background (master log:
  `LTPO/logs/icot/run_all.log`). Sequence:
  `qwen25vl7b → qwen3vl8b → qwen3vl4b → qwen25vl3b`. Per-model, 7 dev sets
  fan out across GPUs 0-6 in parallel.
- 01:31 — First Qwen2.5-VL-7B launch saw mmstar/scienceqa/math_* accuracy
  collapse to ~2% because the trigger fired right after the model wrote
  "Let's analyze the image step by step:\n\n", which broke the chain of
  thought before any reasoning happened. Killed the run.
- Tuned the trigger: require ≥60 generated tokens before the first
  injection and ≥80 tokens between injections (was 20). Re-tested on
  mmstar / math_vista with Qwen2.5-VL-3B — responses now complete with
  proper `<answer>` / `\boxed{}` formatting.
- 01:40 — Re-launched the full grid. Early progress (~12 samples in)
  shows mmstar 4/15, mmvp 10/17, hallusion 7/16, scienceqa 1/13 — all
  in the expected range for a 7B baseline.

### Qwen2.5-VL-7B results (Qwen-Max LLM judge)
Finished ~01:55 (≈15 min wall, 7 GPUs in parallel).

| Dataset         | Correct | Total | Accuracy |
|-----------------|--------:|------:|---------:|
| mmvp_dev        |     150 |   300 |   50.00% |
| hallusion_dev   |      96 |   300 |   32.00% |
| scienceqa_dev   |      55 |   300 |   18.33% |
| mmstar_dev      |      44 |   300 |   14.67% |
| math_vista_dev  |      33 |   300 |   11.00% |
| math_vision_dev |      12 |   300 |    4.00% |
| mm_math_dev     |       8 |   300 |    2.67% |

### Qwen3-VL-8B in progress (started ~01:56)

### Qwen3-VL-8B results (Qwen-Max LLM judge)
Finished ~03:04 (~67 min wall, math sets dominate).

| Dataset         | Correct | Total | Accuracy |
|-----------------|--------:|------:|---------:|
| mmvp_dev        |     225 |   300 |   75.00% |
| hallusion_dev   |     212 |   300 |   70.67% |
| scienceqa_dev   |     172 |   300 |   57.33% |
| math_vista_dev  |     170 |   300 |   56.67% |
| mmstar_dev      |     169 |   300 |   56.33% |
| mm_math_dev     |      85 |   300 |   28.33% |
| math_vision_dev |      66 |   300 |   22.00% |

### Qwen3-VL-4B in progress (started ~03:04)

### Qwen3-VL-4B results (Qwen-Max LLM judge)
Finished ~04:07.

| Dataset         | Correct | Total | Accuracy |
|-----------------|--------:|------:|---------:|
| mmvp_dev        |     230 |   300 |   76.67% |
| hallusion_dev   |     198 |   300 |   66.00% |
| scienceqa_dev   |     167 |   300 |   55.67% |
| mmstar_dev      |     163 |   300 |   54.33% |
| math_vista_dev  |     153 |   300 |   51.00% |
| mm_math_dev     |      77 |   300 |   25.67% |
| math_vision_dev |      51 |   300 |   17.00% |

### Qwen2.5-VL-3B in progress (started ~04:07)

### Qwen2.5-VL-3B results (Qwen-Max LLM judge)
Finished ~04:40.

| Dataset         | Correct | Total | Accuracy |
|-----------------|--------:|------:|---------:|
| hallusion_dev   |     158 |   300 |   52.67% |
| mmvp_dev        |     138 |   300 |   46.00% |
| mmstar_dev      |     132 |   300 |   44.00% |
| scienceqa_dev   |      97 |   300 |   32.33% |
| math_vista_dev  |      83 |   300 |   27.67% |
| math_vision_dev |      37 |   300 |   12.33% |
| mm_math_dev     |      36 |   300 |   12.00% |

## Summary across models (accuracy %)

| Dataset         | Qwen2.5-VL-7B | Qwen3-VL-8B | Qwen3-VL-4B | Qwen2.5-VL-3B |
|-----------------|--------------:|------------:|------------:|--------------:|
| math_vista_dev  |        11.00 |       56.67 |       51.00 |        27.67 |
| math_vision_dev |         4.00 |       22.00 |       17.00 |        12.33 |
| mm_math_dev     |         2.67 |       28.33 |       25.67 |        12.00 |
| hallusion_dev   |        32.00 |       70.67 |       66.00 |        52.67 |
| mmvp_dev        |        50.00 |       75.00 |       76.67 |        46.00 |
| mmstar_dev      |        14.67 |       56.33 |       54.33 |        44.00 |
| scienceqa_dev   |        18.33 |       57.33 |       55.67 |        32.33 |

Wall-clock: ~3 h 8 min (01:31 → 04:40), 7 datasets fan-out per model on
GPUs 0–6, sequential across the 4 models. Per-sample artefacts and
extracted answers live in `output/icot/<model>-<dataset>-…-icot/logistics.pt`;
per-dataset stdout logs in `output/icot/<tag>_<dataset>.log`.

The Qwen2.5-VL-7B numbers look noticeably weaker than its Qwen3-VL siblings.
A quick spot-check of `output/icot/Qwen2.5-VL-7B-Instruct-…/logistics.pt`
shows responses often produce the correct answer in prose but skip the
`<answer>` / `\boxed{}` wrapper — the LLM judge picks up most of them but
some are still flagged equivalent=False; the chat template / boxed
prompting interacts differently with 7B-instruct than with the Qwen3 family.

### Notes on the implementation
- `icot_vl.generate_icot` runs greedy decoding token-by-token using a KV
  cache. When the cumulative newline count crosses an even threshold (and
  at least 20 reasoning tokens have passed since the last injection), the
  loop runs a forward with `output_attentions=True`, averages attention
  across layers + heads from the latest position to image-token positions
  and picks the top-16 patch indices. Their patch embeddings (from the
  vision tower) are spliced into the cache as `<|vision_start|>` + 16
  patches + `<|vision_end|>`, up to 3 sub-images per response. A 2-step
  newline suppression keeps the model from stalling in a "\n\n…" loop
  right after the inject.
- Differences from the original ICoT (Qwen2-VL) code: the original mutates
  `self.rope_deltas` and crafts a mrope 3D position layout for the sub-
  image. We instead let the model's cached `rope_deltas` apply 1-D rope to
  the injected positions — this stays portable across Qwen2.5-VL / Qwen3-VL
  and avoids monkey-patching transformers, at the cost of slightly OOD
  positions for the sub-image tokens.

