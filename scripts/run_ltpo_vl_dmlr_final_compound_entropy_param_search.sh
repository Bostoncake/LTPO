#!/bin/bash
# run_ltpo_vl_dmlr_final_compound_entropy_param_search.sh
#
# LTPO-only hyperparam sweep on the FINAL variant (main_vl_dmlr_final.py),
# mirroring run_ltpo_vl_dmlr_final_param_search.sh except that:
#   - reward_type is one of the two new compound entropy objectives:
#         entropy_diff  → reward = r1 - r2                (no antithetic)
#         entropy_clip  → reward = (r1_pos - r1_neg)
#                                  + clip(r1_pos - r2_pos, min=0)
#       where r1 = first-output-token entropy under the normal forward
#       and r2 = first-output-token entropy when attention to image-token
#       positions is masked out. Both are minimisation objectives.
#   - the sweep additionally iterates over reward_type so the two new
#     losses are compared on the same grid.
#   - --reward_on_latent_tokens is NOT passed: the compound rewards are
#     only defined at the first-output-token position; passing the flag
#     would be silently ignored anyway, but we drop it for clarity.
#   - best-latent selection still uses r1 alone (handled inside
#     generate_vl), independent of the compound objective driving
#     optimisation.
#
# Per-dataset (num_thought_tokens, init_mode) is fixed and mirrors the
# init scheme from the reference param-search script.
#
# Usage:
#   bash scripts/run_ltpo_vl_dmlr_final_compound_entropy_param_search.sh
#
# Overridable env vars:
#   N_GPUS       number of GPUs to use (default 8)
#   MODEL        path to model checkpoint
#   REWARD_LIST  whitespace-separated subset of {entropy_diff entropy_clip}

set -u

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=${N_GPUS:-8}
MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}

# Reward types to sweep. Override REWARD_LIST to run a subset.
REWARD_LIST=${REWARD_LIST:-"entropy_diff entropy_clip"}
REWARD_TYPES=(${REWARD_LIST})
for r in "${REWARD_TYPES[@]}"; do
    case "${r}" in
        entropy_diff|entropy_clip) ;;
        *) echo "Unknown reward type '${r}' in REWARD_LIST. Use entropy_diff/entropy_clip." >&2; exit 1 ;;
    esac
done

# ===========================================================================
# Per-dataset fixed init: "<num_thought_tokens> <init_mode>"
#   init_mode ∈ {endoftext, hidden}
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

# LTPO sweep (tokens is per-dataset, NOT swept).
STEPS_LIST=(10 15)
SIGMA_LIST=(5.0 25.0)
SIGMA_DECAY=0.95
LR_LIST=(1e-4 5e-4 1e-3)
TOP_K=10

root_output=./output/ltpo_dmlr_final/0517_param_search_perds_init_compound_entropy
mkdir -p "${root_output}"

jobs=()
# Reward type is the INNERMOST loop dimension: for each (steps, sigma, lr,
# dataset) the two reward types are scheduled back-to-back, so they start
# nearly simultaneously on free GPUs and side-by-side comparisons surface
# as soon as the first config finishes.
for steps in "${STEPS_LIST[@]}"; do
    for sigma in "${SIGMA_LIST[@]}"; do
        for lr in "${LR_LIST[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                for reward in "${REWARD_TYPES[@]}"; do
                    jobs+=("${reward} ${steps} ${sigma} ${lr} ${dataset}")
                done
            done
        done
    done
done

total=${#jobs[@]}

echo "════════════════════════════════════════════════════════"
echo "LTPO-DMLR FINAL compound-entropy param-search on dev sets"
echo "    reward ∈ {${REWARD_TYPES[*]}}"
echo "    steps  ∈ {${STEPS_LIST[*]}}"
echo "    sigma  ∈ {${SIGMA_LIST[*]}}   lr    ∈ {${LR_LIST[*]}}"
echo "    sigma_decay=${SIGMA_DECAY}  top_k=${TOP_K}  (fixed)"
echo "    Per-dataset (tokens, init):"
for d in "${DATASETS[@]}"; do
    echo "        ${d}: tokens=${DATASET_TOKENS[$d]}  init=${DATASET_INIT[$d]}"
done
echo "    Datasets: ${#DATASETS[@]}  Configs per reward: $(( total / ${#DATASETS[@]} / ${#REWARD_TYPES[@]} ))"
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
            read -r reward steps sigma lr dataset <<< "${jobs[$job_idx]}"
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

            tag="reward${reward}_steps${steps}_sigma${sigma}_decay${SIGMA_DECAY}_lr${lr}_topk${TOP_K}"
            out_dir="${root_output}/${tag}"
            mkdir -p "${out_dir}"
            log="${out_dir}/${dataset}_tokens${tokens}_init${init_tag}.log"

            echo "[$(date +%T)] GPU ${g}  ←  ${tag}  |  ${dataset} (tokens=${tokens} init=${init_tag})"

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
                --sigma             "${sigma}"     \
                --sigma_decay       "${SIGMA_DECAY}" \
                --lr                "${lr}"        \
                --max_num_steps     "${steps}"     \
                --top_k             "${TOP_K}"     \
                --reward_type       "${reward}"    \
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
echo "════════════════════════════════════════════════════════"
echo "All ${total} jobs done. Results in ${root_output}"
echo "════════════════════════════════════════════════════════"

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
