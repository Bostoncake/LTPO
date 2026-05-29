#!/bin/bash
# run_ltpo_vl_dmlr_final_lookthink_stag_sigma_step_best_qwen25vl7b_vis.sh
#
# Same per-dataset BEST configuration as
#   scripts/run_ltpo_vl_dmlr_final_lookthink_stag_sigma_step_best_qwen25vl7b.sh
# but launched against main_vl_dmlr_final_vis.py so that every time the
# lookthink gate opens and the top-p attention pool succeeds, the script
# saves a visualization PNG showing:
#   - the source image,
#   - the per-image-token normalised first-layer attention as a heatmap,
#   - the top-p selected image tokens highlighted with green boxes,
#   - the per-selected-token pooling weight (pre-renorm + renorm).
#
# One PNG per image per look-trigger:
#   ${VIS_ROOT}/<dataset>/sample_<idx>/look_<k>_step_<step>.png
#
# Use VIS_MAX_SAMPLES to cap how many examples are visualized per
# dataset (default 20). Set to -1 to visualize every example.
#
# Trace generation:
#   After every optimisation step, the script can ALSO run a free-form
#   generation with the current latent thought embeddings (no \boxed{
#   prefix) to expose the model's evolving reasoning path. Each traced
#   sample writes one dedicated log file:
#     ${TRACE_ROOT}/<dataset>/sample_<idx>/trace.log
#   Tracing is gated by TRACE_GEN_MAX_SAMPLES (default 5 per dataset to
#   keep wall time reasonable; each traced sample issues max_num_steps
#   extra generations). Set TRACE_GEN_MAX_SAMPLES=0 to disable tracing.
#
# Usage:
#   bash scripts/run_ltpo_vl_dmlr_final_lookthink_stag_sigma_step_best_qwen25vl7b_vis.sh
#
# Overridable env vars:
#   MODEL                     path to Qwen2.5-VL-7B checkpoint
#   VIS_ROOT                  root dir for per-dataset vis subdirs
#   VIS_MAX_SAMPLES           cap on samples to visualize per dataset (default 20)
#   TRACE_ROOT                root dir for per-dataset trace logs
#   TRACE_GEN_MAX_SAMPLES     cap on samples to trace per dataset (default 5; 0 disables)
#   TRACE_GEN_MAX_NEW_TOKENS  max_new_tokens for the per-step trace gen (default 512)

set -u

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export MODEL_TYPE="${MODEL_TYPE:-qwen-max}"

MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}
VIS_MAX_SAMPLES=${VIS_MAX_SAMPLES:-50}
TRACE_GEN_MAX_SAMPLES=${TRACE_GEN_MAX_SAMPLES:-10}
TRACE_GEN_MAX_NEW_TOKENS=${TRACE_GEN_MAX_NEW_TOKENS:-512}

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

# DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")
DATASETS=("mmvp_dev")

root_output=./output/ltpo_dmlr_final/0525_best_perds_init_lookthink_stag_sigma_step_vis_full_data/qwen25vl7b
VIS_ROOT=${VIS_ROOT:-${root_output}/_vis}
TRACE_ROOT=${TRACE_ROOT:-${root_output}/_trace}
mkdir -p "${root_output}" "${VIS_ROOT}" "${TRACE_ROOT}"

echo "========================================================"
echo "LTPO-DMLR FINAL lookthink-stag BEST per-dataset (VIS)"
echo "    model            : ${MODEL}"
echo "    reward_type      : ${REWARD_TYPE}  best_selection=${BEST_SELECTION}"
echo "    sigma_decay      : ${SIGMA_DECAY}  top_k=${TOP_K}"
echo "    Output root      : ${root_output}"
echo "    Vis root         : ${VIS_ROOT}"
echo "    Vis max samples  : ${VIS_MAX_SAMPLES}  (per dataset)"
echo "    Trace root       : ${TRACE_ROOT}"
echo "    Trace max samples: ${TRACE_GEN_MAX_SAMPLES}  (per dataset; 0 disables)"
echo "    Trace gen tokens : ${TRACE_GEN_MAX_NEW_TOKENS}"
echo "    Per-dataset (steps, sigma, tokens, init, lr, top_p, stag):"
for d in "${DATASETS[@]}"; do
    echo "        ${d}: steps=${DATASET_STEPS[$d]}  sigma=${DATASET_SIGMA[$d]}  tokens=${DATASET_TOKENS[$d]}  init=${DATASET_INIT[$d]}  lr=${DATASET_LR[$d]}  top_p=${DATASET_TOP_P[$d]}  stag=${DATASET_STAG[$d]}"
done
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
    vis_dir="${VIS_ROOT}/${dataset}"
    trace_dir="${TRACE_ROOT}/${dataset}"
    mkdir -p "${out_dir}" "${vis_dir}" "${trace_dir}"
    log="${out_dir}/run_tokens${tokens}_init${init_tag}_lr${lr}_topp${top_p}_stag${stag}_steps${steps}_sigma${sigma}.log"

    echo "[$(date +%T)] GPU ${gpu}  <-  ${dataset} (tokens=${tokens} init=${init_tag} lr=${lr} topp=${top_p} stag=${stag} steps=${steps} sigma=${sigma})"
    echo "    vis   -> ${vis_dir}"
    echo "    trace -> ${trace_dir}"

    CUDA_VISIBLE_DEVICES=${gpu} python main_vl_dmlr_final_vis.py \
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
        --vis_lookthink_dir "${vis_dir}"   \
        --vis_max_samples   "${VIS_MAX_SAMPLES}" \
        --trace_gen_dir     "${trace_dir}" \
        --trace_gen_max_samples     "${TRACE_GEN_MAX_SAMPLES}" \
        --trace_gen_max_new_tokens  "${TRACE_GEN_MAX_NEW_TOKENS}" \
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
echo "All ${#DATASETS[@]} dataset runs finished."
echo "  metrics -> ${root_output}"
echo "  vis     -> ${VIS_ROOT}"
echo "  trace   -> ${TRACE_ROOT}"
echo "========================================================"

# Summarise accuracy + visualisation/trace counts per dataset
total_acc=0
count=0
for dataset in "${DATASETS[@]}"; do
    results_log="${root_output}/${dataset}/results.log"
    n_vis=$(find "${VIS_ROOT}/${dataset}" -name 'look_*.png' 2>/dev/null | wc -l)
    n_trace=$(find "${TRACE_ROOT}/${dataset}" -name 'trace.log' 2>/dev/null | wc -l)
    if [ -f "${results_log}" ]; then
        acc=$(grep -oP 'accuracy=\K[0-9.]+' "${results_log}" | tail -1)
        if [ -n "$acc" ]; then
            printf "  %-20s acc=%s  vis_png=%s  trace_logs=%s\n" \
                "${dataset}" "${acc}" "${n_vis}" "${n_trace}"
            total_acc=$(awk "BEGIN{print $total_acc + $acc}")
            count=$((count+1))
        fi
    else
        printf "  %-20s acc=N/A   vis_png=%s  trace_logs=%s\n" \
            "${dataset}" "${n_vis}" "${n_trace}"
    fi
done
if [ "$count" -gt 0 ]; then
    mean=$(awk "BEGIN{printf \"%.4f\", $total_acc / $count}")
    echo "  ----------------------------------"
    echo "  mean_acc=${mean}  n=${count}"
fi
