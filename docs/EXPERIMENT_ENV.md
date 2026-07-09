# Experiment Environment

Run script: `scripts/run_ltpo_vl_dmlr_final_visual_dev.sh`
Entry point: `main_vl_dmlr_final_visual.py`

## Hardware

GPU:
- Model and memory: NVIDIA H200, 143771 MiB (≈141 GB) per GPU
- Number of GPUs: 8 (one dataset per GPU, launched in parallel via `CUDA_VISIBLE_DEVICES=0..7`)
- Same hardware for all experiments: yes
- NVIDIA driver: 580.105.08

## Software

- OS: VesselOS 2.0 (LTS-SP2), Linux kernel 6.6.0-100.jd_b003.x86_64
- Python: 3.10.20
- PyTorch: 2.4.1+cu121
- CUDA: 12.1 (runtime / torch build)
- transformers: 4.57.1
- accelerate: 1.13.0
- flash-attn: not installed (not used; attention implementation is `eager`)
- qwen-vl-utils: not installed in the `ltpo` conda env

Environment activation:
```bash
source /export/home/lanliwei.1/abcxyz/env/miniconda3/bin/activate
conda activate ltpo
```

## Model

- Backbone: Qwen2.5-VL-7B-Instruct (local path:
  `/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct`)

## Inference

- dtype: `torch.float32` (set in `main_vl_dmlr_final_visual.py:237`)
- quantization: none
- attention implementation: `eager` (`attn_implementation="eager"`,
  `main_vl_dmlr_final_visual.py:239`)
- device mapping: single-device per process (`model.to("cuda")`); no
  HuggingFace `device_map` sharding. Parallelism across datasets is achieved
  by launching one process per dataset with `CUDA_VISIBLE_DEVICES=i`.
- Decoding / generation: `max_new_tokens=2048`
- Image preprocessing: `min_pixels=128*28*28`, `max_pixels=256*28*28`
  (passed to the processor)
- Verifier: LLM verifier enabled (`--use_llm_verify`), backed by
  `qwen-max` via the OpenAI-compatible DashScope endpoint
  (`https://dashscope.aliyuncs.com/compatible-mode/v1`).

## Randomness

- seed(s): 42 (single seed, set via `--seed 42` → `set_seed(42)`)
- number of runs: 1 per (dataset, configuration)

## O-CAT / LTPO hyperparameters (from the script)

Mapping to the LTPO/O-CAT notation used in the paper, with the
corresponding CLI flags from
`scripts/run_ltpo_vl_dmlr_final_visual_dev.sh`:

- T (max optimization steps per sample): 15 — `--max_num_steps 15`
- L (number of latent / thought tokens): 2 — `--num_thought_tokens 2`
- k (top-k candidates considered per step): 10 — `--top_k 10`
- p (perturbation / search hyperparameter): not exposed as a separate
  flag in this script; the perturbation strength is controlled by `sigma`
  (with `sigma_decay 0.95`)
- sigma (initial perturbation scale): 25.0 — `--sigma 25.0`,
  with multiplicative decay `--sigma_decay 0.95`
- eta_c (learning rate, candidate / latent update): 0.01 — `--lr 0.01`
- eta_o (learning rate, outer / orchestrator update): not used as a
  separate value in this script — a single `--lr 0.01` governs the update
- N_stag (stagnation / early-stop window): not explicitly set on the
  command line in this script (uses the code default in
  `main_vl_dmlr_final_visual.py`)

Additional visual-injection / DMLR hyperparameters used by this variant:

- `--num_selected_patches 16`
- `--initial_patch_count 1`
- `--patch_increment 1`
- `--visual_insert_stride 1`
- `--visual_injection_start_step 0`
- `--visual_injection_interval 1`

## Evaluation

- Dataset splits (dev, 300-sample subsets — the `*_dev` MLLM benchmarks):
  `mmvp_dev`, `mmstar_dev`, `mm_math_dev`, `math_vista_dev`,
  `math_vision_dev`, `hallusion_dev`, `scienceqa_dev`
- Data root: `mllm_data`, image root: `.`
- Official evaluation scripts used: no — evaluation is performed by the
  in-repo pipeline (LLM-judge verification via `qwen-max` through the
  OpenAI-compatible API, gated by `--use_llm_verify`); we do not invoke
  the benchmarks' own official scorers.
- Decoding settings: greedy / model-default sampling, `max_new_tokens=2048`,
  `seed=42`.
- Total or per-sample runtime / GPU hours: not separately measured for
  this report; the seven dev splits run concurrently, one per H200.
