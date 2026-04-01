#!/bin/bash
# run_workspace_fixed_grid.sh — LTPO + Fixed Visual Workspace，grid-search ablation.
#
# Grid-searches NUM_WORKSPACE_SLOTS ∈ {8,16} × NUM_ROUTE_SLOTS ∈ {2,4}
# over 5 datasets, for a total of 20 jobs dynamically dispatched across GPUs.
#
# 用法:
#   N_GPUS=4 bash scripts/run_workspace_fixed_grid.sh
#
# 可调环境变量:
#   N_GPUS      可用 GPU 数量（默认 8）
#   MODEL       模型路径
#   INJECT_MODE "add" 或 "prepend"（默认 "prepend"）

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=${N_GPUS:-7}
MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}
INJECT_MODE=${INJECT_MODE:-prepend}

K_VALUES=(8 16)
R_VALUES=(2 4)
DATASETS=("mmvp" "mmstar" "math_vista" "hallusion" "scienceqa")

root_output=./output/workspace_fixed_grid_${INJECT_MODE}
mkdir -p "${root_output}"

echo "=== Fixed Workspace Grid Search ==="
echo "    K ∈ {${K_VALUES[*]}}  r ∈ {${R_VALUES[*]}}  mode=${INJECT_MODE}"
echo "    GPUs: ${N_GPUS}  Model: ${MODEL}"
echo "    Output root: ${root_output}"

# Build job list: "K r dataset"
jobs=()
for K in "${K_VALUES[@]}"; do
    for r in "${R_VALUES[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            jobs+=("${K} ${r} ${dataset}")
        done
    done
done
total=${#jobs[@]}
echo "    Total jobs: ${total}"

# Dynamic GPU scheduling
declare -a gpu_pids
for ((g=0; g<N_GPUS; g++)); do gpu_pids[$g]=-1; done

job_idx=0
while [ $job_idx -lt $total ]; do
    for ((g=0; g<N_GPUS; g++)); do
        [ $job_idx -ge $total ] && break
        pid=${gpu_pids[$g]}
        if [ "$pid" -eq -1 ] || ! kill -0 "$pid" 2>/dev/null; then
            read -r K r dataset <<< "${jobs[$job_idx]}"
            out_dir="${root_output}/K${K}_r${r}"
            mkdir -p "${out_dir}"
            log="${root_output}/K${K}_r${r}_${dataset}.log"
            echo "[$(date +%T)] GPU ${g}  ←  K=${K}  r=${r}  ${dataset}"
            CUDA_VISIBLE_DEVICES=${g} python main_vl_workspace_fixed.py \
                --dataset "${dataset}" \
                --data_root mllm_data \
                --image_root . \
                --model_name_or_path "${MODEL}" \
                --output_dir "${out_dir}" \
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
                --num_workspace_slots "${K}" \
                --num_route_slots "${r}" \
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
echo "=== All ${total} jobs done. Results in ${root_output} ==="
