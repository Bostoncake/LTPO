#!/bin/bash
# run_ltpo_vl_dmlr_direct_boxed_oracle_dev.sh
#
# Diagnostic LTPO sweep with the direct "\boxed{" assistant prefix.
# In this variant the model is forced to begin its output with
# "\boxed{" so it directly produces the final answer plus a closing
# "}" instead of visible CoT, and the LTPO confidence reward is scored
# at the prediction of the FIRST generated token (the answer content)
# rather than the thought-token span.
#
# Usage:
#   bash scripts/run_ltpo_vl_dmlr_direct_boxed_oracle_dev.sh           # MODE=ltpo (default)
#   bash scripts/run_ltpo_vl_dmlr_direct_boxed_oracle_dev.sh ltpo
#   bash scripts/run_ltpo_vl_dmlr_direct_boxed_oracle_dev.sh baseline
#
# Overridable env vars:
#   N_GPUS         number of GPUs to use (default 8)
#   MODEL          path to model checkpoint
#   REWARD_TYPE    LTPO optimisation objective: "confidence" (default) or
#                  "entropy". When set to "entropy", the reward is the
#                  information entropy of the top-k tokens at the first-
#                  generated-token position, instead of the top-k log-prob
#                  confidence.

set -u

MODE=${1:-ltpo}

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=${N_GPUS:-8}
MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}
REWARD_TYPE=${REWARD_TYPE:-confidence}

case "${REWARD_TYPE}" in
    confidence|entropy) ;;
    *) echo "Unknown REWARD_TYPE='${REWARD_TYPE}'. Use 'confidence' or 'entropy'." >&2; exit 1 ;;
esac

DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

# ===========================================================================
# Build job list depending on MODE
# ===========================================================================

if [ "$MODE" = "baseline" ]; then
    # Baseline: still uses the forced "\boxed{" assistant prefix so the
    # eval matches the conditions LTPO optimises under.  No hyperparam
    # sweep — one run per dataset.
    root_output=./output/ltpo_dmlr_direct_boxed/0515_baseline_thought_tokens2
    mkdir -p "${root_output}"

    jobs=()
    for dataset in "${DATASETS[@]}"; do
        jobs+=("${dataset}")
    done

elif [ "$MODE" = "ltpo" ]; then
    # LTPO grid: mirrors run_ltpo_vl_dmlr_paired_grid_dev.sh.
    TOKENS_LIST=(2 4)
    STEPS_LIST=(10 15)
    SIGMA_LIST=(5.0 25.0)
    SIGMA_DECAY=0.95
    LR_LIST=(1e-4 5e-4 1e-3)
    TOP_K=10

    root_output=./output/ltpo_dmlr_direct_boxed/0514_baseline_prompt_confidence_fallback
    mkdir -p "${root_output}"

    jobs=()
    for tokens in "${TOKENS_LIST[@]}"; do
        for steps in "${STEPS_LIST[@]}"; do
            for sigma in "${SIGMA_LIST[@]}"; do
                for lr in "${LR_LIST[@]}"; do
                    for dataset in "${DATASETS[@]}"; do
                        jobs+=("${tokens} ${steps} ${sigma} ${lr} ${dataset}")
                    done
                done
            done
        done
    done

else
    echo "Unknown MODE '${MODE}'. Use 'baseline' or 'ltpo'." >&2
    exit 1
fi

total=${#jobs[@]}

echo "════════════════════════════════════════════════════════"
echo "LTPO-DMLR DIRECT-BOXED (${MODE}) on dev sets"
if [ "$MODE" = "ltpo" ]; then
    echo "    tokens ∈ {${TOKENS_LIST[*]}}  steps ∈ {${STEPS_LIST[*]}}"
    echo "    sigma  ∈ {${SIGMA_LIST[*]}}   lr    ∈ {${LR_LIST[*]}}"
    echo "    sigma_decay=${SIGMA_DECAY}  top_k=${TOP_K}  (fixed)"
    echo "    reward_type=${REWARD_TYPE}"
    echo "    Datasets: ${#DATASETS[@]}  Configs: $(( total / ${#DATASETS[@]} ))"
fi
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

            if [ "$MODE" = "baseline" ]; then
                dataset="${jobs[$job_idx]}"
                tag="baseline_boxed"
                out_dir="${root_output}/${tag}"
                mkdir -p "${out_dir}"
                log="${out_dir}/${dataset}.log"

                echo "[$(date +%T)] GPU ${g}  ←  ${tag}  |  ${dataset}"

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
                    --eval_baseline                    \
                    --use_llm_verify                   \
                    --baseline_with_thought_tokens     \
                    --num_thought_tokens 2             \
                    --verbose 1                        \
                    > "${log}" 2>&1 &

            else
                read -r tokens steps sigma lr dataset <<< "${jobs[$job_idx]}"

                tag="tokens${tokens}_steps${steps}_sigma${sigma}_decay${SIGMA_DECAY}_lr${lr}_topk${TOP_K}_reward${REWARD_TYPE}"
                out_dir="${root_output}/${tag}"
                mkdir -p "${out_dir}"
                log="${out_dir}/${dataset}.log"

                echo "[$(date +%T)] GPU ${g}  ←  ${tag}  |  ${dataset}"

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
                    --reward_type       "${REWARD_TYPE}" \
                    --use_llm_verify                   \
                    --use_baseline_prompt              \
                    --enable_baseline_fallback         \
                    --verbose 1                        \
                    > "${log}" 2>&1 &
            fi

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
