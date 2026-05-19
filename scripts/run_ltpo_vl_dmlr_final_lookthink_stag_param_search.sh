#!/bin/bash
# run_ltpo_vl_dmlr_final_lookthink_stag_param_search.sh
#
# LOOK/THINK hyperparam sweep on the FINAL variant (main_vl_dmlr_final.py),
# mirroring run_ltpo_vl_dmlr_final_lookthink_param_search.sh but swapping the
# fixed reward-threshold gate for the stagnation-count gate
# (--lookthink_stagnation_steps): LOOK is triggered when the best reward has
# not been refreshed for N consecutive optimisation steps.
#
# Fixed settings:
#   reward_type               = entropy_diff
#   compound_best_selection   = diff
#   sigma                     = 25.0
#   steps                     = 10
#   sigma_decay               = 0.95
#   top_k                     = 10
#
# Swept settings:
#   lr                         in {1e-4, 1e-3}
#   lookthink_top_p            in {0.5, 0.9}
#   lookthink_stagnation_steps in {2, 5}
#
# Usage:
#   bash scripts/run_ltpo_vl_dmlr_final_lookthink_stag_param_search.sh
#
# Overridable env vars:
#   N_GPUS  number of GPUs to use (default 8)
#   MODEL   path to model checkpoint

set -u

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export MODEL_TYPE="${MODEL_TYPE:-qwen-max}"

N_GPUS=${N_GPUS:-8}
MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}

# ===========================================================================
# Per-dataset fixed init: "<num_thought_tokens> <init_mode>"
#   init_mode in {endoftext, hidden}
#   - endoftext: default embedding-table init for thought tokens
#                (no extra flag, default hook path)
#   - hidden:    --ltpo_thought_init_from_hidden + --use_inputs_embeds
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

REWARD_TYPE=entropy_diff
BEST_SELECTION=diff
STEPS=10
SIGMA=25.0
SIGMA_DECAY=0.95
TOP_K=10
LR_LIST=(1e-4 1e-3)
LOOKTHINK_TOP_P_LIST=(0.5 0.9)
LOOKTHINK_STAGNATION_STEPS_LIST=(2 5)

root_output=./output/ltpo_dmlr_final/0518_param_search_perds_init_lookthink_stag
mkdir -p "${root_output}"

jobs=()
# Keep the lookthink-specific knobs as the innermost dimensions so each
# dataset/config schedules side-by-side variants quickly.
for lr in "${LR_LIST[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for top_p in "${LOOKTHINK_TOP_P_LIST[@]}"; do
            for stag in "${LOOKTHINK_STAGNATION_STEPS_LIST[@]}"; do
                jobs+=("${lr} ${dataset} ${top_p} ${stag}")
            done
        done
    done
done

total=${#jobs[@]}

echo "========================================================"
echo "LTPO-DMLR FINAL LOOK/THINK (stagnation gate) param-search"
echo "    reward_type=${REWARD_TYPE}  best_selection=${BEST_SELECTION}"
echo "    lr                         in {${LR_LIST[*]}}"
echo "    lookthink_top_p            in {${LOOKTHINK_TOP_P_LIST[*]}}"
echo "    lookthink_stagnation_steps in {${LOOKTHINK_STAGNATION_STEPS_LIST[*]}}"
echo "    steps=${STEPS}  sigma=${SIGMA}  sigma_decay=${SIGMA_DECAY}  top_k=${TOP_K}"
echo "    Per-dataset (tokens, init):"
for d in "${DATASETS[@]}"; do
    echo "        ${d}: tokens=${DATASET_TOKENS[$d]}  init=${DATASET_INIT[$d]}"
done
echo "    Datasets: ${#DATASETS[@]}  Total jobs: ${total}  GPUs: ${N_GPUS}"
echo "========================================================"

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
            read -r lr dataset top_p stag <<< "${jobs[$job_idx]}"
            tokens=${DATASET_TOKENS[$dataset]}
            init=${DATASET_INIT[$dataset]}

            init_flag=""
            embeds_flag=""
            init_tag="endoftext"
            if [ "$init" = "hidden" ]; then
                init_flag="--ltpo_thought_init_from_hidden"
                embeds_flag="--use_inputs_embeds"
                init_tag="hidden"
            fi

            tag="lookthink_stag_reward${REWARD_TYPE}_best${BEST_SELECTION}_steps${STEPS}_sigma${SIGMA}_decay${SIGMA_DECAY}_lr${lr}_topk${TOP_K}_topp${top_p}_stag${stag}"
            out_dir="${root_output}/${tag}"
            mkdir -p "${out_dir}"
            log="${out_dir}/${dataset}_tokens${tokens}_init${init_tag}.log"

            echo "[$(date +%T)] GPU ${g}  <-  ${tag}  |  ${dataset} (tokens=${tokens} init=${init_tag})"

            CUDA_VISIBLE_DEVICES=${g} python main_vl_dmlr_final.py \
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
                --sigma             "${SIGMA}"     \
                --sigma_decay       "${SIGMA_DECAY}" \
                --lr                "${lr}"        \
                --max_num_steps     "${STEPS}"     \
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

            gpu_pids[$g]=$!
            job_idx=$((job_idx+1))
        fi
    done
    sleep 2
done

wait
echo ""
echo "========================================================"
echo "All ${total} jobs done. Results in ${root_output}"
echo "========================================================"

# Summarise mean accuracy across datasets per config dir
for cfg_dir in "${root_output}"/*; do
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
