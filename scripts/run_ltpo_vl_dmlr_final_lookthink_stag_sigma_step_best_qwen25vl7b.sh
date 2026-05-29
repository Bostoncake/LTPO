#!/bin/bash
# run_ltpo_vl_dmlr_final_lookthink_stag_sigma_step_best_qwen25vl7b.sh
#
# Launch one LTPO-DMLR FINAL lookthink-stag run per dataset on
# Qwen2.5-VL-7B, using the per-dataset BEST (sigma, max_num_steps)
# selected by scripts/find_best_grid.py from
#   output/ltpo_dmlr_final/0518_param_search_perds_init_lookthink_stag_sigma_step
#
# Frozen (tokens, init, lr, top_p, stag) carry over from the upstream
# stag param search (run_ltpo_vl_dmlr_final_lookthink_stag_sigma_step_search.sh).
# Other fixed knobs: reward_type=entropy_diff, compound_best_selection=diff,
# sigma_decay=0.95, top_k=10.
#
# Per-dataset best (steps, sigma) + frozen (tokens, init, lr, top_p, stag):
#                       steps  sigma   tokens  init       lr     top_p  stag
#   math_vista_dev      10     25.0    2       hidden     1e-3   0.5    2
#   math_vision_dev     15     5.0     4       endoftext  1e-3   0.9    5
#   mm_math_dev         10     5.0     4       hidden     1e-4   0.9    5
#   hallusion_dev       10     25.0    4       hidden     1e-3   0.5    2
#   mmvp_dev            10     5.0     2       endoftext  1e-3   0.9    2
#   mmstar_dev          10     5.0     4       hidden     1e-4   0.5    2
#   scienceqa_dev       15     25.0    2       endoftext  1e-3   0.5    5
#
# Usage:
#   bash scripts/run_ltpo_vl_dmlr_final_lookthink_stag_sigma_step_best_qwen25vl7b.sh
#
# Overridable env vars:
#   MODEL  path to the Qwen2.5-VL-7B checkpoint

set -u

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export MODEL_TYPE="${MODEL_TYPE:-qwen-max}"

MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}

REWARD_TYPE=entropy_diff
BEST_SELECTION=diff
SIGMA_DECAY=0.95
TOP_K=10

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
declare -A DATASET_LR=(
    ["math_vista_dev"]=1e-3
    ["math_vision_dev"]=1e-3
    ["mm_math_dev"]=1e-4
    ["hallusion_dev"]=1e-3
    ["mmvp_dev"]=1e-3
    ["mmstar_dev"]=1e-4
    ["scienceqa_dev"]=1e-3
)
declare -A DATASET_TOP_P=(
    ["math_vista_dev"]=0.5
    ["math_vision_dev"]=0.9
    ["mm_math_dev"]=0.9
    ["hallusion_dev"]=0.5
    ["mmvp_dev"]=0.9
    ["mmstar_dev"]=0.5
    ["scienceqa_dev"]=0.5
)
declare -A DATASET_STAG=(
    ["math_vista_dev"]=2
    ["math_vision_dev"]=5
    ["mm_math_dev"]=5
    ["hallusion_dev"]=2
    ["mmvp_dev"]=2
    ["mmstar_dev"]=2
    ["scienceqa_dev"]=5
)
# Per-dataset BEST (sigma, max_num_steps) from
# find_best_grid.py on the sigma_step grid search.
declare -A DATASET_SIGMA=(
    ["math_vista_dev"]=25.0
    ["math_vision_dev"]=5.0
    ["mm_math_dev"]=5.0
    ["hallusion_dev"]=25.0
    ["mmvp_dev"]=5.0
    ["mmstar_dev"]=5.0
    ["scienceqa_dev"]=25.0
)
declare -A DATASET_STEPS=(
    ["math_vista_dev"]=10
    ["math_vision_dev"]=15
    ["mm_math_dev"]=10
    ["hallusion_dev"]=10
    ["mmvp_dev"]=10
    ["mmstar_dev"]=10
    ["scienceqa_dev"]=15
)

DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

root_output=./output/ltpo_dmlr_final/0518_best_perds_init_lookthink_stag_sigma_step/qwen25vl7b
mkdir -p "${root_output}"

echo "========================================================"
echo "LTPO-DMLR FINAL lookthink-stag BEST per-dataset run"
echo "    model        : ${MODEL}"
echo "    reward_type  : ${REWARD_TYPE}  best_selection=${BEST_SELECTION}"
echo "    sigma_decay  : ${SIGMA_DECAY}  top_k=${TOP_K}"
echo "    Per-dataset (steps, sigma, tokens, init, lr, top_p, stag):"
for d in "${DATASETS[@]}"; do
    echo "        ${d}: steps=${DATASET_STEPS[$d]}  sigma=${DATASET_SIGMA[$d]}  tokens=${DATASET_TOKENS[$d]}  init=${DATASET_INIT[$d]}  lr=${DATASET_LR[$d]}  top_p=${DATASET_TOP_P[$d]}  stag=${DATASET_STAG[$d]}"
done
echo "    Output root  : ${root_output}"
echo "========================================================"

pids=()
for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    gpu="${i}"

    tokens=${DATASET_TOKENS[$dataset]}
    init=${DATASET_INIT[$dataset]}
    lr=${DATASET_LR[$dataset]}
    top_p=${DATASET_TOP_P[$dataset]}
    stag=${DATASET_STAG[$dataset]}
    sigma=${DATASET_SIGMA[$dataset]}
    steps=${DATASET_STEPS[$dataset]}

    init_flag=""
    embeds_flag=""
    init_tag="endoftext"
    if [ "$init" = "hidden" ]; then
        init_flag="--ltpo_thought_init_from_hidden"
        embeds_flag="--use_inputs_embeds"
        init_tag="hidden"
    fi

    out_dir="${root_output}/${dataset}"
    mkdir -p "${out_dir}"
    log="${out_dir}/run_tokens${tokens}_init${init_tag}_lr${lr}_topp${top_p}_stag${stag}_steps${steps}_sigma${sigma}.log"

    echo "[$(date +%T)] GPU ${gpu}  <-  ${dataset} (tokens=${tokens} init=${init_tag} lr=${lr} topp=${top_p} stag=${stag} steps=${steps} sigma=${sigma})"

    CUDA_VISIBLE_DEVICES=${gpu} python main_vl_dmlr_final.py \
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
        --reward_type       "${REWARD_TYPE}" \
        --compound_best_selection "${BEST_SELECTION}" \
        --enable_lookthink                 \
        --lookthink_top_p  "${top_p}"      \
        --lookthink_stagnation_steps "${stag}" \
        --use_llm_verify                   \
        --use_baseline_prompt              \
        --enable_baseline_fallback         \
        ${init_flag}                       \
        ${embeds_flag}                     \
        --verbose 1                        \
        > "${log}" 2>&1 &

    pids+=("$!")
done

for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
    echo "[$(date +%T)] Done: ${DATASETS[$i]}"
done

echo ""
echo "========================================================"
echo "All ${#DATASETS[@]} dataset runs finished. Results in ${root_output}"
echo "========================================================"

# Summarise accuracy per dataset
total_acc=0
count=0
for dataset in "${DATASETS[@]}"; do
    results_log="${root_output}/${dataset}/results.log"
    if [ -f "${results_log}" ]; then
        acc=$(grep -oP 'accuracy=\K[0-9.]+' "${results_log}" | tail -1)
        if [ -n "$acc" ]; then
            printf "  %-20s acc=%s\n" "${dataset}" "${acc}"
            total_acc=$(awk "BEGIN{print $total_acc + $acc}")
            count=$((count+1))
        fi
    fi
done
if [ "$count" -gt 0 ]; then
    mean=$(awk "BEGIN{printf \"%.4f\", $total_acc / $count}")
    echo "  ----------------------------------"
    echo "  mean_acc=${mean}  n=${count}"
fi
