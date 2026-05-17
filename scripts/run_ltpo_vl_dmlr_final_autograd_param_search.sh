#!/bin/bash
# run_ltpo_vl_dmlr_final_autograd_param_search.sh
#
# Autograd (--use_auto_grad) hyperparam sweep on the FINAL LTPO variant
# (main_vl_dmlr_final.py). Mirrors the per-dataset init scheme of
# run_ltpo_vl_dmlr_final_param_search.sh but:
#   - drops --sigma / --sigma_decay from the sweep (autograd path does
#     NOT inject ε; the step is a pure AdamW backprop update)
#   - sweeps lr × steps × {reward objective, reward position}
#
# Sweep axes:
#   lr     ∈ {1e-2, 1e-3, 1e-4}
#   steps  ∈ {3, 5}
#   mode   ∈ {
#       confidence_latent   : --reward_type confidence + --reward_on_latent_tokens
#       confidence_output   : --reward_type confidence  (reward at first-generated-token)
#       entropy_output      : --reward_type entropy     (reward at first-generated-token)
#   }
#
# Per-dataset (num_thought_tokens, init_mode) is fixed and identical to
# the reference noise-sampling sweep:
#   math_vista_dev    : 2 tokens, hidden   (--use_inputs_embeds)
#   math_vision_dev   : 4 tokens, endoftext
#   mm_math_dev       : 4 tokens, hidden   (--use_inputs_embeds)
#   hallusion_dev     : 4 tokens, hidden   (--use_inputs_embeds)
#   mmvp_dev          : 2 tokens, endoftext
#   mmstar_dev        : 4 tokens, hidden   (--use_inputs_embeds)
#   scienceqa_dev     : 2 tokens, endoftext
#
# All cases place the latent thought tokens in the baseline prompt
# position (--use_baseline_prompt) and keep the no-thought-token
# baseline fallback (--enable_baseline_fallback), matching the
# reference recipe.
#
# Total jobs = 3 (lr) × 2 (steps) × 3 (mode) × 7 (datasets) = 126
#
# Usage:
#   bash scripts/run_ltpo_vl_dmlr_final_autograd_param_search.sh
#
# Overridable env vars:
#   N_GPUS  number of GPUs (default 8)
#   MODEL   model checkpoint path

set -u

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=${N_GPUS:-8}
MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}

# ===========================================================================
# Per-dataset fixed init (tokens, init_mode)
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

# ===========================================================================
# Autograd sweep axes
# ===========================================================================
STEPS_LIST=(3 5)
LR_LIST=(1e-2 1e-3 1e-4)
TOP_K=10

# Three (reward_type, reward_position) modes.
# Each entry is "<mode_tag> <reward_type> <on_latent_flag>".
# on_latent_flag is "1" → pass --reward_on_latent_tokens, "0" → omit it.
MODES=(
    "confidence_latent confidence 1"
    "confidence_output confidence 0"
    "entropy_output    entropy    0"
)

root_output=./output/ltpo_dmlr_final/0517_autograd_param_search
mkdir -p "${root_output}"

jobs=()
for mode_spec in "${MODES[@]}"; do
    read -r mode_tag reward_type on_latent <<< "${mode_spec}"
    for steps in "${STEPS_LIST[@]}"; do
        for lr in "${LR_LIST[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                jobs+=("${mode_tag}|${reward_type}|${on_latent}|${steps}|${lr}|${dataset}")
            done
        done
    done
done

total=${#jobs[@]}
n_configs=$(( total / ${#DATASETS[@]} ))

echo "════════════════════════════════════════════════════════"
echo "LTPO-DMLR FINAL autograd param-search (per-dataset init)"
echo "    lr     ∈ {${LR_LIST[*]}}"
echo "    steps  ∈ {${STEPS_LIST[*]}}"
echo "    modes  : confidence_latent | confidence_output | entropy_output"
echo "    top_k  = ${TOP_K} (fixed)   AdamW: wd=1e-8 eps=1e-5"
echo "    Per-dataset (tokens, init):"
for d in "${DATASETS[@]}"; do
    echo "        ${d}: tokens=${DATASET_TOKENS[$d]}  init=${DATASET_INIT[$d]}"
done
echo "    Datasets: ${#DATASETS[@]}   Configs: ${n_configs}"
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
            IFS='|' read -r mode_tag reward_type on_latent steps lr dataset <<< "${jobs[$job_idx]}"
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

            latent_flag=""
            if [ "${on_latent}" = "1" ]; then
                latent_flag="--reward_on_latent_tokens"
            fi

            tag="autograd_${mode_tag}_steps${steps}_lr${lr}_topk${TOP_K}"
            out_dir="${root_output}/${tag}"
            mkdir -p "${out_dir}"
            log="${out_dir}/${dataset}_tokens${tokens}_init${init_tag}.log"

            echo "[$(date +%T)] GPU ${g}  ←  ${tag}  |  ${dataset} (tokens=${tokens} init=${init_tag})"

            CUDA_VISIBLE_DEVICES=${g} python main_vl_dmlr_final.py \
                --dataset            "${dataset}"     \
                --data_root          mllm_data        \
                --image_root         .                \
                --model_name_or_path "${MODEL}"       \
                --output_dir         "${out_dir}"     \
                --device             cuda             \
                --seed               42               \
                --max_new_tokens     2048             \
                --min_pixels         128              \
                --max_pixels         256              \
                --num_thought_tokens "${tokens}"      \
                --lr                 "${lr}"          \
                --max_num_steps      "${steps}"       \
                --top_k              "${TOP_K}"       \
                --reward_type        "${reward_type}" \
                --use_auto_grad                       \
                --use_llm_verify                      \
                --use_baseline_prompt                 \
                --enable_baseline_fallback            \
                ${init_flag}                          \
                ${embeds_flag}                        \
                ${latent_flag}                        \
                --verbose 1                           \
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

# Per-config mean accuracy across datasets, sorted desc
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
