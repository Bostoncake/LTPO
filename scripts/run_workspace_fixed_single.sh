#!/bin/bash
# run_workspace_single.sh — LTPO + Visual Workspace，单 GPU，遍历全部数据集。
#
# 用法:
#   bash scripts/run_workspace_single.sh
#
# 可调参数:
#   MODEL              模型路径
#   GPU                使用的 GPU 编号
#   NUM_WORKSPACE_SLOTS  workspace 容量 K（默认 8）
#   NUM_ROUTE_SLOTS      每步注入槽数 r（默认 2）
#   INJECT_MODE          "add" 或 "prepend"（默认 "add"）

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}
GPU=${GPU:-0}
NUM_WORKSPACE_SLOTS=${NUM_WORKSPACE_SLOTS:-8}
NUM_ROUTE_SLOTS=${NUM_ROUTE_SLOTS:-2}
INJECT_MODE=${INJECT_MODE:-prepend}

output=./output/workspace_${INJECT_MODE}_K${NUM_WORKSPACE_SLOTS}_r${NUM_ROUTE_SLOTS}
mkdir -p "${output}"

echo "=== Visual Workspace: K=${NUM_WORKSPACE_SLOTS}  r=${NUM_ROUTE_SLOTS}  mode=${INJECT_MODE} ==="
echo "=== Model: ${MODEL} ==="
echo "=== Output: ${output} ==="

for dataset in "mmvp" "mmstar" "mm_math" "math_vista" "math_vision" "hallusion" "scienceqa"; do
    echo "[$(date +%T)] Starting ${dataset} on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} python main_vl_workspace_fixed.py \
        --dataset "${dataset}" \
        --data_root mllm_data \
        --image_root . \
        --model_name_or_path "${MODEL}" \
        --output_dir "${output}" \
        --device cuda \
        --seed 42 \
        --max_new_tokens 2048 \
        --min_pixels 128 \
        --max_pixels 256 \
        --num_thought_tokens 2 \
        --sigma 25.0 \
        --sigma_decay 0.95 \
        --lr 0.01 \
        --max_num_steps 15 \
        --top_k 10 \
        --use_workspace \
        --num_workspace_slots "${NUM_WORKSPACE_SLOTS}" \
        --num_route_slots "${NUM_ROUTE_SLOTS}" \
        --workspace_inject_mode "${INJECT_MODE}" \
        --use_llm_verify \
        --verbose 1 \
        > "${output}/${dataset}.log" 2>&1 &
    GPU=$((GPU+1))

    # echo "[$(date +%T)] Done: ${dataset}  $(tail -1 "${output}/${dataset}.log")"
done

echo "=== All datasets finished. Results in ${output} ==="
