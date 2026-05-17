#!/bin/bash
# run_ltpo_vl_dmlr_final_param_search.sh
#
# LTPO-only hyperparam sweep on the FINAL variant (main_vl_dmlr_final.py).
# Per-dataset (num_thought_tokens, init_mode) is fixed and mirrors the
# init scheme from run_ltpo_vl_dmlr_direct_boxed_entropy_eval_baseline_dev.sh.
# Sweeps steps / sigma / lr only. All methods place latent thought tokens
# in the baseline prompt position (--use_baseline_prompt) so the prompt
# aligns with the reference eval-baseline scheme.
#
# Per-dataset fixed init (from the reference eval-baseline run):
#   math_vista_dev    : 2 tokens,  hidden init   (--use_inputs_embeds)
#   math_vision_dev   : 4 tokens,  endoftext init
#   mm_math_dev       : 4 tokens,  hidden init   (--use_inputs_embeds)
#   hallusion_dev     : 4 tokens,  hidden init   (--use_inputs_embeds)
#   mmvp_dev          : 2 tokens,  endoftext init
#   mmstar_dev        : 4 tokens,  hidden init   (--use_inputs_embeds)
#   scienceqa_dev     : 2 tokens,  endoftext init
#
# Forward path:
#   - endoftext-init datasets → hook path (default):
#       model.generate(**inputs) + ThoughtEmbedInjector splicing the
#       thought-token rows. mRoPE / image-token scatter / rope_deltas
#       are preserved.
#   - hidden-init datasets    → legacy embeds path (--use_inputs_embeds):
#       pre-merge visual tokens into inputs_embeds, write the LTPO
#       latents into the thought-token rows in-place, then
#       model.generate(inputs_embeds=..., attention_mask=...).
#
# Usage:
#   bash scripts/run_ltpo_vl_dmlr_final_param_search.sh
#
# Overridable env vars:
#   N_GPUS       number of GPUs to use (default 8)
#   MODEL        path to model checkpoint
#   REWARD_TYPE  LTPO objective ("confidence" or "entropy", default entropy)

set -u

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=${N_GPUS:-8}
MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}
REWARD_TYPE=${REWARD_TYPE:-entropy}

case "${REWARD_TYPE}" in
    confidence|entropy) ;;
    *) echo "Unknown REWARD_TYPE='${REWARD_TYPE}'. Use 'confidence' or 'entropy'." >&2; exit 1 ;;
esac

# ===========================================================================
# Per-dataset fixed init: "<num_thought_tokens> <init_mode>"
#   init_mode ∈ {endoftext, hidden}
#   - endoftext: default embedding-table init for thought tokens
#                (no extra flag, default hook path)
#   - hidden:    --ltpo_thought_init_from_hidden + --use_inputs_embeds
#                so the thought rows are initialised from the pre-thought
#                hidden state and the rest of LTPO runs through the
#                legacy embeds path.
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

root_output=./output/ltpo_dmlr_final/0516_param_search_perds_init_${REWARD_TYPE}_on_latent
mkdir -p "${root_output}"

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
echo "LTPO-DMLR FINAL  param-search (per-dataset init)  on dev sets"
echo "    steps  ∈ {${STEPS_LIST[*]}}"
echo "    sigma  ∈ {${SIGMA_LIST[*]}}   lr    ∈ {${LR_LIST[*]}}"
echo "    sigma_decay=${SIGMA_DECAY}  top_k=${TOP_K}  (fixed)"
echo "    reward_type=${REWARD_TYPE}"
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
            embeds_flag=""
            init_tag="endoftext"
            if [ "$init" = "hidden" ]; then
                init_flag="--ltpo_thought_init_from_hidden"
                embeds_flag="--use_inputs_embeds"
                init_tag="hidden"
            fi

            tag="steps${steps}_sigma${sigma}_decay${SIGMA_DECAY}_lr${lr}_topk${TOP_K}_reward${REWARD_TYPE}"
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
                --reward_type       "${REWARD_TYPE}" \
                --use_llm_verify                   \
                --use_baseline_prompt              \
                --enable_baseline_fallback         \
                ${init_flag}                       \
                ${embeds_flag}                     \
                --reward_on_latent_tokens            \
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
