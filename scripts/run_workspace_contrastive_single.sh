#!/bin/bash
# run_workspace_contrastive_single.sh
# Single-config launcher for the Contrastive Workspace method — parallel
# over the 7 benchmark datasets on consecutive GPUs.
#
# Contrastive reward:  r = r2 - r1
#   r1 = confidence WITHOUT visual evidence injection
#   r2 = confidence WITH    visual evidence injection
#
# Usage:
#   bash scripts/run_workspace_contrastive_single.sh
#
# Overridable env vars:
#   MODEL                model path
#   GPU                  starting GPU index (default 0)
#   NUM_WORKSPACE_SLOTS  K (default 8)
#   NUM_ROUTE_SLOTS      r (default 2)
#   INJECT_MODE          "add" or "prepend" (default "prepend")
#   ROUTE_MODE           "avg" or "per_token" (default "per_token")

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}
GPU=${GPU:-0}
NUM_WORKSPACE_SLOTS=${NUM_WORKSPACE_SLOTS:-8}
NUM_ROUTE_SLOTS=${NUM_ROUTE_SLOTS:-2}
INJECT_MODE=${INJECT_MODE:-prepend}
ROUTE_MODE=${ROUTE_MODE:-per_token}

output=./output/workspace_contrastive_${INJECT_MODE}_${ROUTE_MODE}_K${NUM_WORKSPACE_SLOTS}_r${NUM_ROUTE_SLOTS}
mkdir -p "${output}"

echo "=== Contrastive Workspace  (reward = r2 - r1) ==="
echo "=== K=${NUM_WORKSPACE_SLOTS}  r=${NUM_ROUTE_SLOTS}  inject=${INJECT_MODE}  route=${ROUTE_MODE} ==="
echo "=== Model: ${MODEL} ==="
echo "=== Output: ${output} ==="

for dataset in "mmvp" "mmstar" "mm_math" "math_vista" "math_vision" "hallusion" "scienceqa"; do
    echo "[$(date +%T)] Starting ${dataset} on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} python main_vl_workspace_contrastive.py \
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
        --workspace_route_mode "${ROUTE_MODE}" \
        --use_llm_verify \
        --verbose 1 \
        > "${output}/${dataset}.log" 2>&1 &
    GPU=$((GPU+1))
done

wait
echo "=== All datasets finished. Results in ${output} ==="
