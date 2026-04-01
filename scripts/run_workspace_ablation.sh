#!/bin/bash
# run_workspace_ablation.sh — 三组消融实验并行运行，用于对比：
#   Group A (GPU 0): LTPO-DMLR baseline（不启用 workspace）
#   Group B (GPU 1): workspace, inject_mode=add
#   Group C (GPU 2): workspace, inject_mode=prepend
#
# 每组跑全部数据集，结果分别写入独立 output 目录。
#
# 用法:
#   bash scripts/run_workspace_ablation.sh

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}
DATASETS=("mmvp" "mmstar" "mm_math" "math_vista" "math_vision" "hallusion" "scienceqa")

NUM_WORKSPACE_SLOTS=8
NUM_ROUTE_SLOTS=2

# ---- 公共参数 ----
COMMON_ARGS=(
    --data_root mllm_data
    --image_root .
    --model_name_or_path "${MODEL}"
    --device cuda
    --seed 42
    --max_new_tokens 2048
    --min_pixels 128
    --max_pixels 256
    --num_thought_tokens 2
    --sigma 25.0
    --sigma_decay 0.95
    --lr 0.01
    --max_num_steps 15
    --top_k 10
    --use_llm_verify
    --verbose 1
)

run_group() {
    local gpu=$1
    local out_dir=$2
    shift 2
    local extra_args=("$@")

    mkdir -p "${out_dir}"
    echo "=== [GPU ${gpu}] ${out_dir} ==="

    for dataset in "${DATASETS[@]}"; do
        echo "[$(date +%T)] GPU ${gpu}  ${dataset}"
        CUDA_VISIBLE_DEVICES=${gpu} python main_vl_workspace.py \
            --dataset "${dataset}" \
            --output_dir "${out_dir}" \
            "${COMMON_ARGS[@]}" \
            "${extra_args[@]}" \
            > "${out_dir}/${dataset}.log" 2>&1
        echo "[$(date +%T)] Done  ${dataset}  $(tail -1 "${out_dir}/${dataset}.log")"
    done
    echo "=== [GPU ${gpu}] All done → ${out_dir} ==="
}

# ---- 三组并行 ----
run_group 0 ./output/ablation_baseline &

run_group 1 ./output/ablation_ws_add \
    --use_workspace \
    --num_workspace_slots "${NUM_WORKSPACE_SLOTS}" \
    --num_route_slots "${NUM_ROUTE_SLOTS}" \
    --workspace_inject_mode add &

run_group 2 ./output/ablation_ws_prepend \
    --use_workspace \
    --num_workspace_slots "${NUM_WORKSPACE_SLOTS}" \
    --num_route_slots "${NUM_ROUTE_SLOTS}" \
    --workspace_inject_mode prepend &

wait
echo "=== Ablation complete. Compare results in ./output/ablation_* ==="
