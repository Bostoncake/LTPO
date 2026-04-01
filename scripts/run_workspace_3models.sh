#!/bin/bash
# run_workspace_3models.sh — LTPO + Visual Workspace，多 GPU 动态调度。
#
# 遍历 3 个模型 × 全部数据集（3×7 = 21 个任务），动态分配到 N_GPUS 块 GPU。
# 某块 GPU 空出即立即分配下一个任务，避免 GPU 空闲等待。
#
# 用法:
#   N_GPUS=8 bash scripts/run_workspace_3models.sh
#
# 可调环境变量:
#   N_GPUS               可用 GPU 数量（默认 8）
#   MODEL_DIR            模型根目录
#   NUM_WORKSPACE_SLOTS  workspace 容量 K（默认 8）
#   NUM_ROUTE_SLOTS      每步注入槽数 r（默认 2）
#   INJECT_MODE          "add" 或 "prepend"（默认 "add"）

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=${N_GPUS:-8}
MODEL_DIR=${MODEL_DIR:-/WillDevExt/xiongyizhe/models}
NUM_WORKSPACE_SLOTS=${NUM_WORKSPACE_SLOTS:-8}
NUM_ROUTE_SLOTS=${NUM_ROUTE_SLOTS:-2}
INJECT_MODE=${INJECT_MODE:-add}

MODELS=(
    "Qwen2.5-VL-7B-Instruct"
    "Qwen3-VL-4B-Instruct"
    "Qwen3-VL-8B-Instruct"
)
DATASETS=("mmvp" "mmstar" "mm_math" "math_vista" "math_vision" "hallusion" "scienceqa")

output=./output/workspace_3models_${INJECT_MODE}_K${NUM_WORKSPACE_SLOTS}_r${NUM_ROUTE_SLOTS}
mkdir -p "${output}"

echo "=== Visual Workspace 3-Model Run ==="
echo "    K=${NUM_WORKSPACE_SLOTS}  r=${NUM_ROUTE_SLOTS}  mode=${INJECT_MODE}"
echo "    GPUs: ${N_GPUS}  Output: ${output}"

# 构建任务列表（model dataset 对）
jobs=()
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        jobs+=("${model} ${dataset}")
    done
done
total=${#jobs[@]}
echo "    Total jobs: ${total}"

# 动态 GPU 调度
declare -a gpu_pids
for ((g=0; g<N_GPUS; g++)); do gpu_pids[$g]=-1; done

job_idx=0
while [ $job_idx -lt $total ]; do
    for ((g=0; g<N_GPUS; g++)); do
        [ $job_idx -ge $total ] && break
        pid=${gpu_pids[$g]}
        if [ "$pid" -eq -1 ] || ! kill -0 "$pid" 2>/dev/null; then
            read -r model dataset <<< "${jobs[$job_idx]}"
            log="${output}/${model}_${dataset}.log"
            echo "[$(date +%T)] GPU ${g}  ←  ${model} / ${dataset}"
            CUDA_VISIBLE_DEVICES=${g} python main_vl_workspace.py \
                --dataset "${dataset}" \
                --data_root mllm_data \
                --image_root . \
                --model_name_or_path "${MODEL_DIR}/${model}" \
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
                > "${log}" 2>&1 &
            gpu_pids[$g]=$!
            job_idx=$((job_idx+1))
        fi
    done
    sleep 2
done

wait
echo "=== All ${total} jobs done. Results in ${output} ==="
