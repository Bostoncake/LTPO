# ICoT 全集效率统计运行说明

目标：在 Qwen2.5-VL-7B-Instruct 上跑 ICoT 全集，并统计 inference 开销。

默认模型：

```bash
/home/xiongyizhe/hqs/storage/models/Qwen2.5-VL-7B-Instruct
```

默认数据集是全集，不是 dev：

```text
mmvp, mmstar, mm_math, math_vista, math_vision, hallusion, scienceqa
```

## 一键运行

在仓库根目录运行：

```bash
export OPENAI_API_KEY=...
bash scripts/run_icot_profile_full_qwen25vl7b.sh
```

默认输出目录：

```text
output/icot_profile_full/
```

脚本会并行启动 7 个数据集，默认使用 GPU 0-6：

```bash
GPUS="0 1 2 3 4 5 6" bash scripts/run_icot_profile_full_qwen25vl7b.sh
```

如果只想先 smoke test，不跑全集：

```bash
conda activate ltpo
CUDA_VISIBLE_DEVICES=0 python main_vl_icot.py \
  --dataset mmvp \
  --data_root mllm_data \
  --image_root . \
  --model_name_or_path /home/xiongyizhe/hqs/storage/models/Qwen2.5-VL-7B-Instruct \
  --output_dir ./output/icot_profile_smoke \
  --device cuda \
  --start_data_idx 0 \
  --end_data_idx 2 \
  --max_new_tokens 64 \
  --profile_efficiency \
  --profile_flops \
  --profile_flops_every 1 \
  --verbose 1
```

这个 smoke test 只会使用原来的 `ltpo` 环境，不会安装或修改包。上面的 smoke test 没有加 `--use_llm_verify`，只用于检查本地 inference/profiling 能否跑通；正式结果请用一键脚本默认的 Qwen-Max judge。

## 输出文件

每个数据集会生成一个 run 目录，形如：

```text
output/icot_profile_full/Qwen2.5-VL-7B-Instruct-mmvp-patches16-subimgs3-icot/
```

其中重要文件：

| 文件 | 内容 |
| --- | --- |
| `logistics.pt` | 原有逐样本结果，entry 里新增 `efficiency` 字段 |
| `efficiency_profile.jsonl` | 逐样本效率统计，一行一个样本 |
| `efficiency_summary.json` | 单数据集效率汇总 |
| `efficiency_summary.md` | 单数据集 Markdown 汇总 |
| `results.log` | 原有 accuracy 日志，附带效率文件路径 |

全集聚合文件位于 `output/icot_profile_full/`：

| 文件 | 内容 |
| --- | --- |
| `efficiency_summary_all.json` | 7 个数据集总汇总 |
| `efficiency_summary_all.md` | 7 个数据集 Markdown 总汇总 |
| `qwen25vl7b_<dataset>_profile.log` | 每个数据集的 stdout/stderr |

如果需要重新汇总已有结果：

```bash
python scripts/summarize_icot_efficiency.py --output_root ./output/icot_profile_full
```

## 统计字段

逐样本 `efficiency_profile.jsonl` 中主要字段：

| 字段 | 含义 |
| --- | --- |
| `generation_wall_time_s` | 单样本 ICoT 生成耗时，不包含 judge |
| `judge_time_s` | 答案判断耗时；如果不用 LLM judge，基本是规则判断耗时 |
| `profiler_estimated_flops` | PyTorch profiler 估算 FLOPs |
| `profiled_flops` | 当前样本是否启用了 FLOPs profiler |
| `prompt_tokens` | prompt token 数 |
| `image_tokens` | 原图 image token 数 |
| `generated_tokens` | 生成 token 数，不含停止 token |
| `num_sub_imgs` | ICoT 插入子图次数 |
| `subimage_tokens_inserted` | 插入的视觉 token 总数；每次通常是 18 |
| `selected_patch_tokens` | 被选中的 patch token 总数；每次通常是 16 |
| `forward_passes` | 总 forward 次数 |
| `prefill_forward_passes` | prefill forward 次数 |
| `decode_forward_passes` | decode token forward 次数 |
| `subimage_forward_passes` | 插入子图后刷新 KV-cache 的 forward 次数 |
| `attention_forward_passes` | `output_attentions=True` 的 forward 次数 |
| `model_forward_input_tokens` | 所有 forward 输入 token 数之和 |
| `max_context_tokens` | 生成过程中达到的最大上下文长度 |

`forward_passes` 的口径：

```text
forward_passes = prefill_forward_passes
               + decode_forward_passes
               + subimage_forward_passes
```

其中 decode 阶段每生成一个非停止 token 会有一次 forward；每次成功插入子图后，还会额外有一次长度约 18 的 forward。

## FLOPs 口径

FLOPs 通过：

```python
torch.profiler.profile(with_flops=True)
```

统计。这个值是 PyTorch operator-level estimate，主要覆盖 matmul/addmm/conv 等算子。它适合比较不同 ICoT 设置或不同样本的相对开销，但不是严格硬件 FLOPs：

- attention softmax、cache bookkeeping、Python 循环开销可能不计入；
- 一些 fused kernel 可能没有 FLOPs 字段；
- profiler 本身会增加 wall clock time，所以开启 `--profile_flops` 后运行会比普通 inference 慢。

默认脚本 `PROFILE_FLOPS_EVERY=1`，即每个样本都 profile FLOPs。如果想降低 profiler 额外开销，可以改成抽样：

```bash
PROFILE_FLOPS_EVERY=10 bash scripts/run_icot_profile_full_qwen25vl7b.sh
```

此时汇总中的 `estimated_total_flops` 会用被 profile 样本的均值外推全集 FLOPs。

## Qwen-Max Judge 和时间口径

正式运行默认 `USE_LLM_VERIFY=1`，使用 Qwen-Max judge 判断 correctness。需要先设置：

```bash
export OPENAI_API_KEY=...
```

时间统计不会把 judge API 时间混进 inference 开销：

- `generation_wall_time_s`：只统计本地 Qwen2.5-VL-7B 的 ICoT 生成时间。
- `judge_time_s`：单独统计 Qwen-Max judge 的 API 时间。
- FLOPs、forward pass、token throughput 都只来自本地 ICoT generation。

如果只是 efficiency-only smoke test，不想调用外部 API，可以显式关闭 judge：

```bash
USE_LLM_VERIFY=0 bash scripts/run_icot_profile_full_qwen25vl7b.sh
```

正式脚本会默认设置：

```bash
OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_TYPE=qwen-max
```

注意：`judge_time_s` 会单独记录，不混入 `generation_wall_time_s`。

## 常用配置

```bash
MODEL_PATH=/home/xiongyizhe/hqs/storage/models/Qwen2.5-VL-7B-Instruct
OUTPUT_ROOT=./output/icot_profile_full
GPUS="0 1 2 3 4 5 6"
MAX_NEW_TOKENS=512
MIN_PIXELS=128
MAX_PIXELS=256
PROFILE_FLOPS_EVERY=1
USE_LLM_VERIFY=1
VERBOSE=1
```

例子：

```bash
OUTPUT_ROOT=./output/icot_profile_full_qwen25vl7b \
GPUS="0 1 2 3" \
PROFILE_FLOPS_EVERY=5 \
bash scripts/run_icot_profile_full_qwen25vl7b.sh
```

## 断点续跑

一键脚本默认传了 `--resume`。如果中间某个数据集断掉，重新执行同一条命令即可从对应 `logistics.pt` 的 `start_idx` 继续。

如果你想完全重跑，删除对应输出目录即可。
