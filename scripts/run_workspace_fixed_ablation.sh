#!/bin/bash
# run_workspace_fixed_ablation.sh — Ablation for the fixed visual workspace.
#
# Four groups run in parallel:
#   GPU 0: LTPO-DMLR baseline          (no workspace)
#   GPU 1: fixed workspace, mode=add   (pool → text-guided top-K → add)
#   GPU 2: fixed workspace, mode=prepend
#   GPU 3: original workspace, mode=add  (for direct comparison with fix)
#
# Tunable env vars:
#   MODEL                model path
#   NUM_WORKSPACE_SLOTS  K  (default 8)
#   NUM_POOLED_TOKENS    P  (default: auto = max(K*4,32))
#   NUM_ROUTE_SLOTS      r  (default 2)
#
# Usage:
#   bash scripts/run_workspace_fixed_ablation.sh

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}
NUM_WORKSPACE_SLOTS=${NUM_WORKSPACE_SLOTS:-8}
NUM_ROUTE_SLOTS=${NUM_ROUTE_SLOTS:-2}
# Leave NUM_POOLED_TOKENS unset to use the auto default (max(K*4, 32) = 32)
NUM_POOLED_TOKENS=${NUM_POOLED_TOKENS:-}

DATASETS=("mmvp" "mmstar" "mm_math" "math_vista" "math_vision" "hallusion" "scienceqa")

# ---- Common args shared by all groups ----
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

# ---- Helper: run one group sequentially on a given GPU ----
# Usage: run_group <gpu> <script> <out_dir> [extra args...]
run_group() {
    local gpu=$1
    local script=$2
    local out_dir=$3
    shift 3
    local extra_args=("$@")

    mkdir -p "${out_dir}"
    echo "=== [GPU ${gpu}] ${script}  →  ${out_dir} ==="

    for dataset in "${DATASETS[@]}"; do
        echo "[$(date +%T)] GPU ${gpu}  dataset=${dataset}"
        CUDA_VISIBLE_DEVICES=${gpu} python "${script}" \
            --dataset "${dataset}" \
            --output_dir "${out_dir}" \
            "${COMMON_ARGS[@]}" \
            "${extra_args[@]}" \
            > "${out_dir}/${dataset}.log" 2>&1
        echo "[$(date +%T)] Done    ${dataset}  $(tail -1 "${out_dir}/${dataset}.log")"
    done

    echo "=== [GPU ${gpu}] All done → ${out_dir} ==="
}

# Build optional --num_pooled_tokens arg for the fixed script
POOLED_ARG=()
if [ -n "${NUM_POOLED_TOKENS}" ]; then
    POOLED_ARG=(--num_pooled_tokens "${NUM_POOLED_TOKENS}")
fi

echo "========================================================"
echo " Fixed Visual Workspace Ablation"
echo "  K=${NUM_WORKSPACE_SLOTS}  r=${NUM_ROUTE_SLOTS}"
echo "  P=${NUM_POOLED_TOKENS:-auto}"
echo "  Model: ${MODEL}"
echo "========================================================"

# ---- Group A (GPU 0): LTPO-DMLR baseline, no workspace ----
run_group 0 main_vl_workspace_fixed.py \
    ./output/fixed_ablation_baseline \
    &

# ---- Group B (GPU 1): fixed workspace, inject_mode=add ----
run_group 1 main_vl_workspace_fixed.py \
    ./output/fixed_ablation_ws_add \
    --use_workspace \
    --num_workspace_slots "${NUM_WORKSPACE_SLOTS}" \
    --num_route_slots "${NUM_ROUTE_SLOTS}" \
    --workspace_inject_mode add \
    "${POOLED_ARG[@]}" \
    &

# ---- Group C (GPU 2): fixed workspace, inject_mode=prepend ----
run_group 2 main_vl_workspace_fixed.py \
    ./output/fixed_ablation_ws_prepend \
    --use_workspace \
    --num_workspace_slots "${NUM_WORKSPACE_SLOTS}" \
    --num_route_slots "${NUM_ROUTE_SLOTS}" \
    --workspace_inject_mode prepend \
    "${POOLED_ARG[@]}" \
    &

# ---- Group D (GPU 3): original workspace (add) for direct diff comparison ----
run_group 3 main_vl_workspace.py \
    ./output/fixed_ablation_original_add \
    --use_workspace \
    --num_workspace_slots "${NUM_WORKSPACE_SLOTS}" \
    --num_route_slots "${NUM_ROUTE_SLOTS}" \
    --workspace_inject_mode add \
    &

wait
echo "========================================================"
echo " Ablation complete. Results in ./output/fixed_ablation_*"
echo "  baseline       → fixed_ablation_baseline"
echo "  fixed + add    → fixed_ablation_ws_add"
echo "  fixed + prepend→ fixed_ablation_ws_prepend"
echo "  original + add → fixed_ablation_original_add"
echo "========================================================"
