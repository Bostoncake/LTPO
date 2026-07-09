# SLOT baseline run

Reproducing [SLOT](Other_baseline/SLOT) (test-time prompt-conditional shift of
the LM hidden states) on the 7 dev splits and 4 models that we use for the LTPO
paper.

## What SLOT does

From `Other_baseline/SLOT/modeling_qwen2_slot.py` (lines ~872–895):

1. Run a single forward pass over the prompt with the LM, **disabling SLOT** so
   we get the vanilla last hidden state `H ∈ R^{1,L,d}`.
2. Initialise `delta ∈ R^{1,1,d}` to zeros, then for `times` AdamW steps
   minimise the next-token CE loss on the prompt of `lm_head(H + delta)`.
3. Freeze `delta`. During generation, every time `lm_head` is invoked add
   `delta` to its input.

The original repo achieves step 3 by copying the entire `modeling_qwen2.py` and
injecting `hidden_states = hidden_states + self.delta` before the LM head.

## Why I did NOT duplicate the modeling files

The four target models (`Qwen2.5-VL-{3B,7B}`, `Qwen3-VL-{4B,8B}`) come from two
distinct `transformers` modeling files (`modeling_qwen2_5_vl.py`,
`modeling_qwen3_vl.py`). Copying and patching both is verbose and easy to break
during cleanup. Instead, the SLOT step "add delta to hidden states right before
`lm_head`" is implemented by **monkey-patching `model.lm_head.forward`** at
runtime:

```python
orig = model.lm_head.forward
def patched(x): return orig(x + delta.to(x.dtype))
model.lm_head.forward = patched
# ... generate ...
model.lm_head.forward = orig   # restored every sample
```

This is the exact same modification (mathematically) as the SLOT repo, works
unchanged across all four architectures, and leaves the upstream
`transformers` files untouched, so reverting only requires deleting the new
files in `LTPO/`.

## Files added (all under `LTPO/`, no modeling files touched)

- `slot_vl.py`             — prompt-optimisation + monkey-patch + `generate_vl`.
- `main_vl_slot.py`        — driver, mirroring `main_vl_dmlr_final_visual.py`.
- `scripts/run_slot_vl_dev.sh` — launcher (1 GPU per dataset).
- `docs/MY_RUN_SLOT.md`    — this file.

## Datasets

`mmvp_dev, mmstar_dev, mm_math_dev, math_vista_dev, math_vision_dev,
hallusion_dev, scienceqa_dev` — loaded via `data_vl.get_mllm_dataset` exactly
as the LTPO `--dataset` flag expects.

## Models (run in this order)

1. Qwen2.5-VL-7B-Instruct
2. Qwen3-VL-8B-Instruct
3. Qwen3-VL-4B-Instruct
4. Qwen2.5-VL-3B-Instruct

## Environment

`source /export/home/lanliwei.1/abcxyz/env/miniconda3/bin/activate && conda activate ltpo`
(torch 2.4.1+cu121, transformers 4.57.1, 8× H200, GPU 0 busy at run start so 7
free GPUs map 1:1 to the 7 datasets per model).

## Run log

### Smoke tests (verbose=1, 1–2 samples)

- Qwen2.5-VL-7B + mmvp_dev[0:2] → 1/2 correct, SLOT loss 7.35 → 7.13 over 3
  steps; first sample matched the GT "(a) Open", second flipped to "a"
  (wrong). End-to-end pipeline works.
- Qwen3-VL-4B + mmvp_dev[0:1] → 1/1 correct, SLOT loss 23.47 → 23.34. The
  higher loss is consistent with Qwen3-VL's tied input/output embeddings.

### Full run (300 samples each)

Launched `scripts/run_slot_vl_dev.sh` in the background at 15:56:25 local
(2026-05-22). Outputs and per-(model,dataset) logs in `LTPO/output/slot/`.

Sequential per model, 7 datasets in parallel across GPUs 1-7:

| # | Model | Status |
|---|---|---|
| 1 | Qwen2.5-VL-7B-Instruct | done (16:59:09) |
| 2 | Qwen3-VL-8B-Instruct   | done (19:58:48) |
| 3 | Qwen3-VL-4B-Instruct   | 5/7 done; math_vision & mm_math resuming from idx 139 / 177 |
| 4 | Qwen2.5-VL-3B-Instruct | resuming all 7 from early logistics checkpoints |

### Incident note (22:20 local)

The original launcher's bash `wait` returned spuriously on Qwen3-VL-4B
(most likely the shell that ran the original `nohup bash
run_slot_vl_dev.sh &` was terminated and its children were reaped). Two
Qwen3-VL-4B jobs (math_vision @139/300, mm_math @177/300) had not
finished, and the launcher then started Qwen2.5-VL-3B for 7 datasets
which only completed 2–23 samples each before being killed too.

Recovery: relaunched via `scripts/run_slot_resume.sh` with `--resume`
(logistics.pt intact on every interrupted job) and `setsid` so each
child is in its own process group, immune to parent-shell SIGHUP.
9 jobs running in parallel; GPU 0 shares Qwen3-VL-4B and Qwen2.5-VL-3B
math_vision.

Approx 3-5 s / sample → ~20 min per dataset → ~20 min per model.
Total ETA ≈ 1.5 h.

(Final accuracies will be appended once each model finishes.)

### Results — Qwen2.5-VL-7B-Instruct (DONE)

| Dataset       | correct / total | accuracy |
|---------------|------------------|----------|
| mmvp_dev      | 209 / 300        | 0.6967   |
| mmstar_dev    | 170 / 300        | 0.5667   |
| math_vista_dev| 171 / 300        | 0.5700   |
| math_vision_dev | 72 / 300       | 0.2400   |
| mm_math_dev   | 103 / 300        | 0.3433   |
| hallusion_dev | 202 / 300        | 0.6733   |
| scienceqa_dev | 177 / 300        | 0.5900   |

### Results — Qwen3-VL-8B-Instruct (DONE)

| Dataset       | correct / total | accuracy |
|---------------|------------------|----------|
| mmvp_dev      | 232 / 300        | 0.7733   |
| mmstar_dev    | 175 / 300        | 0.5833   |
| math_vista_dev| 194 / 300        | 0.6467   |
| math_vision_dev | 100 / 300      | 0.3333   |
| mm_math_dev   | 174 / 300        | 0.5800   |
| hallusion_dev | 214 / 300        | 0.7133   |
| scienceqa_dev | 184 / 300        | 0.6133   |

### Results — Qwen3-VL-4B-Instruct (5/7 done)

| Dataset       | correct / total | accuracy |
|---------------|------------------|----------|
| mmvp_dev      | 233 / 300        | 0.7767   |
| mmstar_dev    | 179 / 300        | 0.5967   |
| math_vista_dev| 182 / 300        | 0.6067   |
| math_vision_dev | — (running, 105/300) | —  |
| mm_math_dev   | — (running, 134/300) | —    |
| hallusion_dev | 209 / 300        | 0.6967   |
| scienceqa_dev | 181 / 300        | 0.6033   |




