# VisAttnSink Inference Profiling

This run profiles VisAttnSink on the full MLLM datasets with Qwen2.5-VL-7B:

```bash
bash scripts/run_visattnsink_profile_qwen25vl7b_full.sh
```

Default model path:

```text
/home/xiongyizhe/hqs/storage/models/Qwen2.5-VL-7B-Instruct
```

Default datasets are the full splits, not dev splits:

```text
mmvp mmstar mm_math math_vista math_vision hallusion scienceqa
```

Run the script from an already activated environment, for example after
`conda activate ltpo`. The script uses the current shell's `python` by default
and does not activate or modify any conda environment. To override the Python
binary explicitly, set `PYTHON_BIN=/path/to/python`.

## LLM Judge

LLM judge is enabled by default. The script reads secrets from the environment:

```bash
export OPENAI_API_KEY="..."
export OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export MODEL_TYPE="${MODEL_TYPE:-qwen-max}"
```

Disable LLM judge only for debugging:

```bash
USE_LLM_VERIFY=0 bash scripts/run_visattnsink_profile_qwen25vl7b_full.sh
```

## Smoke Test

Run one example per dataset on one GPU:

```bash
GPUS=0 END_DATA_IDX=1 bash scripts/run_visattnsink_profile_qwen25vl7b_full.sh
```

Run a single dataset:

```bash
GPUS=0 DATASETS=mmvp END_DATA_IDX=1 bash scripts/run_visattnsink_profile_qwen25vl7b_full.sh
```

Resume an interrupted full run:

```bash
RESUME=1 bash scripts/run_visattnsink_profile_qwen25vl7b_full.sh
```

## Outputs

Default output root:

```text
./output/visattnsink_profile_qwen25vl7b_full/Qwen2.5-VL-7B-Instruct/
```

Each dataset has a config directory such as:

```text
Qwen2.5-VL-7B-Instruct-mmvp-visattnsink-tau20.0-rho0.5-summ0.2-p0.6/
```

Important files:

```text
results.log             accuracy and embedded profile summary
logistics.pt            resumable per-example records
profile_metrics.jsonl   one JSON row per evaluated example
profile_summary.json    aggregate profiling summary
```

## Profiling Fields

`profile_metrics.jsonl` records per-example metrics:

```text
total_flops                         total estimated inference FLOPs
module_flops                        Linear + Conv FLOPs from hooks
linear_flops / conv_flops           module-level breakdown
attention_flops                     QK^T + Attn*V matmul FLOPs
vas_flops                           estimated VisAttnSink extra FLOPs
model_forward_calls                 top-level model forward calls during generate
decoder_layer_forward_calls         decoder layer forward calls
vision_forward_calls                vision tower/root forward calls
attention_calls                     eager attention calls
prompt_tokens / generated_tokens    prompt and generated token counts
image_tokens                        image-token span length in the text input
preprocess_wall_s                   image/template/processor time
generate_wall_s                     synchronized generation wall clock
judge_wall_s                        LLM/rule judge time
total_example_wall_s                end-to-end example time
cuda_peak_memory_allocated_bytes    peak allocated CUDA memory during generation
vas_calls / vas_time_s              VisAttnSink call counts and CPU dispatch time
```

`profile_summary.json` contains sums, means, and selected p50/p90 statistics.
The most useful top-level fields are:

```text
total_flops_sum
total_flops_mean
generate_wall_s_sum
generate_wall_s_mean
model_forward_calls_sum
generated_tokens_sum
flops_per_generated_token_mean
tokens_per_second_mean
vas_calls_sum
vas_time_s_sum
```

## FLOPs Counting Scope

The FLOPs are lightweight analytical estimates from actual runtime tensor
shapes:

- Linear and Conv modules are counted by forward hooks.
- Decoder attention matmuls are counted in the patched Qwen eager attention:
  `QK^T` plus `Attn*V`.
- VisAttnSink extra work is estimated inside DimProspector, HeadFork, and
  VARProcessor.
- Tokenization, PIL image loading, Python control flow, softmax, dropout, and
  miscellaneous elementwise model ops outside the VAS estimate are not fully
  counted.

`generate_wall_s` is the reliable efficiency wall-clock number because the
script synchronizes CUDA around generation with `--profile_sync_cuda`.
`vas_time_s` is CPU dispatch time for the VAS hooks, useful for relative
debugging but not a clean isolated GPU kernel wall time.
