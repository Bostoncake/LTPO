#!/bin/bash
# run_ltpo_vl_dmlr_direct_boxed_entropy_dev.sh
#
# LTPO sweep for the "direct \boxed{" variant under the
# first-generated-token ENTROPY objective (antithetic two-point
# estimator). For each dataset we fix the previously-found best
# token-init configuration (number of tokens + init mode), and sweep
# only the optimisation hyperparameters (max_num_steps, sigma, lr).
#
# Per-dataset fixed init (best from prior search):
#   math_vista_dev    : 2 tokens,  hidden-state init
#   math_vision_dev   : 4 tokens,  endoftext init
#   mm_math_dev       : 4 tokens,  hidden-state init
#   hallusion_dev     : 4 tokens,  hidden-state init
#   mmvp_dev          : 2 tokens,  endoftext init
#   mmstar_dev        : 4 tokens,  hidden-state init
#   scienceqa_dev     : 2 tokens,  endoftext init
#
# Sweeps (same ranges as run_ltpo_vl_dmlr_paired_grid_dev.sh, except
# num_thought_tokens is per-dataset rather than swept):
#   steps  ∈ {10, 15}
#   sigma  ∈ {5.0, 25.0}
#   lr     ∈ {1e-4, 5e-4, 1e-3}
# Fixed: sigma_decay=0.95, top_k=10, reward_type=entropy.
#
# Total: 2×2×3 = 12 configs × 7 datasets = 84 jobs.
#
# Usage:
#   bash scripts/run_ltpo_vl_dmlr_direct_boxed_entropy_dev.sh
#
# Overridable env vars:
#   N_GPUS    number of GPUs to use (default 8)
#   MODEL     path to model checkpoint

set -u

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=${N_GPUS:-8}
MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}

# ===========================================================================
# Per-dataset fixed init: "<num_thought_tokens> <init_mode>"
#   init_mode ∈ {endoftext, hidden}
# ===========================================================================
declare -A DATASET_TOKENS=(
    ["math_vista_dev"]=2
    ["math_vision_dev"]=4
    ["mm_math_dev"]=4
    ["hallusion_dev"]=4
    ["mmvp_dev"]=2
    ["mmstar_dev"]=4
    ["scienceqa_dev"]=2
)
declare -A DATASET_INIT=(
    ["math_vista_dev"]=hidden
    ["math_vision_dev"]=endoftext
    ["mm_math_dev"]=hidden
    ["hallusion_dev"]=hidden
    ["mmvp_dev"]=endoftext
    ["mmstar_dev"]=hidden
    ["scienceqa_dev"]=endoftext
)
DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

STEPS_LIST=(10 15)
SIGMA_LIST=(5.0 25.0)
SIGMA_DECAY=0.95
LR_LIST=(1e-4 5e-4 1e-3)
TOP_K=10

root_output=./output/ltpo_dmlr_direct_boxed/0514_best_token_init_entropy_param_search
mkdir -p "${root_output}"

# Build flat job list: "steps sigma lr dataset"
jobs=()
for steps in "${STEPS_LIST[@]}"; do
    for sigma in "${SIGMA_LIST[@]}"; do
        for lr in "${LR_LIST[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                jobs+=("${steps} ${sigma} ${lr} ${dataset}")
            done
        done
    done
done
total=${#jobs[@]}

echo "════════════════════════════════════════════════════════"
echo "LTPO-DMLR DIRECT-BOXED  ENTROPY (antithetic)  on dev sets"
echo "    steps  ∈ {${STEPS_LIST[*]}}"
echo "    sigma  ∈ {${SIGMA_LIST[*]}}"
echo "    lr     ∈ {${LR_LIST[*]}}"
echo "    sigma_decay=${SIGMA_DECAY}  top_k=${TOP_K}  reward_type=entropy  (fixed)"
echo "    Per-dataset (tokens, init):"
for d in "${DATASETS[@]}"; do
    echo "        ${d}: tokens=${DATASET_TOKENS[$d]}  init=${DATASET_INIT[$d]}"
done
echo "    Datasets: ${#DATASETS[@]}  Configs: $(( total / ${#DATASETS[@]} ))"
echo "    Total jobs: ${total}   GPUs: ${N_GPUS}"
echo "════════════════════════════════════════════════════════"

# ===========================================================================
# Dynamic GPU scheduling
# ===========================================================================
declare -a gpu_pids
for ((g=0; g<N_GPUS; g++)); do gpu_pids[$g]=-1; done

job_idx=0
while [ $job_idx -lt $total ]; do
    for ((g=0; g<N_GPUS; g++)); do
        [ $job_idx -ge $total ] && break
        pid=${gpu_pids[$g]}
        if [ "$pid" -eq -1 ] || ! kill -0 "$pid" 2>/dev/null; then
            read -r steps sigma lr dataset <<< "${jobs[$job_idx]}"

            tokens=${DATASET_TOKENS[$dataset]}
            init=${DATASET_INIT[$dataset]}

            init_flag=""
            init_tag="endoftext"
            if [ "$init" = "hidden" ]; then
                init_flag="--ltpo_thought_init_from_hidden"
                init_tag="hidden"
            fi

            tag="steps${steps}_sigma${sigma}_decay${SIGMA_DECAY}_lr${lr}_topk${TOP_K}_entropy"
            ds_tag="${dataset}_tokens${tokens}_init${init_tag}"
            out_dir="${root_output}/${tag}"
            mkdir -p "${out_dir}"
            log="${out_dir}/${ds_tag}.log"

            echo "[$(date +%T)] GPU ${g}  ←  ${tag}  |  ${ds_tag}"

            CUDA_VISIBLE_DEVICES=${g} python main_vl_dmlr_direct_boxed.py \
                --dataset           "${dataset}"   \
                --data_root         mllm_data      \
                --image_root        .              \
                --model_name_or_path "${MODEL}"    \
                --output_dir        "${out_dir}"   \
                --device            cuda           \
                --seed              42             \
                --max_new_tokens    2048           \
                --min_pixels        128            \
                --max_pixels        256            \
                --num_thought_tokens "${tokens}"   \
                --sigma             "${sigma}"     \
                --sigma_decay       "${SIGMA_DECAY}" \
                --lr                "${lr}"        \
                --max_num_steps     "${steps}"     \
                --top_k             "${TOP_K}"     \
                --reward_type       entropy        \
                ${init_flag}                       \
                --use_llm_verify                   \
                --verbose 1                        \
                > "${log}" 2>&1 &

            gpu_pids[$g]=$!
            job_idx=$((job_idx+1))
        fi
    done
    sleep 2
done

wait
echo ""
echo "════════════════════════════════════════════════════════"
echo "All ${total} jobs done. Results in ${root_output}"
echo "Results per config (sorted by mean accuracy across datasets):"
echo "════════════════════════════════════════════════════════"

# Summarise: for each config dir, average accuracy across all dataset result logs
for cfg_dir in "${root_output}"/steps*; do
    [ -d "$cfg_dir" ] || continue
    cfg=$(basename "$cfg_dir")
    total_acc=0
    count=0
    while IFS= read -r log; do
        acc=$(grep -oP 'accuracy=\K[0-9.]+' "$log" | tail -1)
        [ -n "$acc" ] && total_acc=$(awk "BEGIN{print $total_acc + $acc}") && count=$((count+1))
    done < <(find "$cfg_dir" -name "results.log")
    if [ "$count" -gt 0 ]; then
        mean=$(awk "BEGIN{printf \"%.4f\", $total_acc / $count}")
        echo "mean_acc=${mean}  n=${count}  | ${cfg}"
    fi
done | sort -t'=' -k2 -rn
