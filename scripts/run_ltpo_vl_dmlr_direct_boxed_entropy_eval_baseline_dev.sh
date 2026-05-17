#!/bin/bash
# run_ltpo_vl_dmlr_direct_boxed_entropy_eval_baseline_dev.sh
#
# Counterpart of run_ltpo_vl_dmlr_direct_boxed_entropy_baseline_dev.sh.
# Same per-dataset latent-token init scheme (num_thought_tokens + init
# mode) as the entropy LTPO sweep, but routed through the
# `--eval_baseline` branch of main_vl_dmlr_direct_boxed.py instead of
# through the LTPO path with `--max_num_steps 0`.
#
# How this differs from the LTPO-path baseline:
#   - Prompt placement: latent tokens sit AFTER the assistant role
#     marker and BEFORE the forced "\\boxed{" prefix (matching
#     --use_baseline_prompt LTPO), NOT inside the DMLR-style user-turn
#     instructions. (This is the --eval_baseline / 0514 placement.)
#   - Forward path:
#       * endoftext-init datasets → model.generate(**inputs) with
#         (input_ids, pixel_values, image_grid_thw, attention_mask) →
#         mRoPE ON.
#       * hidden-init datasets   → model.generate(inputs_embeds=...,
#         attention_mask=...) WITHOUT image_grid_thw → mRoPE OFF.
#     i.e. the same forward conditions as the 0514 baselines.
#
# Per-dataset fixed init (same as the entropy sweep & the LTPO-path
# baseline counterpart):
#   math_vista_dev    : 2 tokens,  hidden-state init
#   math_vision_dev   : 4 tokens,  endoftext init
#   mm_math_dev       : 4 tokens,  hidden-state init
#   hallusion_dev     : 4 tokens,  hidden-state init
#   mmvp_dev          : 2 tokens,  endoftext init
#   mmstar_dev        : 4 tokens,  hidden-state init
#   scienceqa_dev     : 2 tokens,  endoftext init
#
# Total: 1 config × 7 datasets = 7 jobs.
#
# Usage:
#   bash scripts/run_ltpo_vl_dmlr_direct_boxed_entropy_eval_baseline_dev.sh
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

root_output=./output/ltpo_dmlr_direct_boxed/0515_best_token_init_eval_baseline
mkdir -p "${root_output}"

jobs=()
for dataset in "${DATASETS[@]}"; do
    jobs+=("${dataset}")
done
total=${#jobs[@]}

echo "════════════════════════════════════════════════════════"
echo "LTPO-DMLR DIRECT-BOXED  EVAL-BASELINE (per-dataset init)  on dev sets"
echo "    Per-dataset (tokens, init):"
for d in "${DATASETS[@]}"; do
    echo "        ${d}: tokens=${DATASET_TOKENS[$d]}  init=${DATASET_INIT[$d]}"
done
echo "    Datasets: ${#DATASETS[@]}"
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
            dataset="${jobs[$job_idx]}"

            tokens=${DATASET_TOKENS[$dataset]}
            init=${DATASET_INIT[$dataset]}

            init_flag=""
            init_tag="endoftext"
            if [ "$init" = "hidden" ]; then
                init_flag="--baseline_thought_init_from_hidden"
                init_tag="hidden"
            fi

            tag="eval_baseline_perds_init"
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
                --eval_baseline                    \
                --baseline_with_thought_tokens     \
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
echo "Per-dataset accuracy:"
echo "════════════════════════════════════════════════════════"

for cfg_dir in "${root_output}"/eval_baseline_*; do
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
