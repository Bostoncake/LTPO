#!/bin/bash
# Run ICoT on the full MLLM datasets with per-sample efficiency profiling.
#
# Defaults target Qwen2.5-VL-7B-Instruct and full datasets (not *_dev).
# Set GPUS="0 1 2 3 4 5 6" to control device assignment.

set -u

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate ltpo
elif [ -f "/home/xiongyizhe/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/xiongyizhe/miniconda3/etc/profile.d/conda.sh"
    conda activate ltpo
fi

MODEL_PATH="${MODEL_PATH:-/home/xiongyizhe/hqs/storage/models/Qwen2.5-VL-7B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/icot_profile_full}"
DATA_ROOT="${DATA_ROOT:-mllm_data}"
IMAGE_ROOT="${IMAGE_ROOT:-.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MIN_PIXELS="${MIN_PIXELS:-128}"
MAX_PIXELS="${MAX_PIXELS:-256}"
NUM_SELECTED_PATCHES="${NUM_SELECTED_PATCHES:-16}"
MAX_SUB_IMGS="${MAX_SUB_IMGS:-3}"
PROFILE_FLOPS_EVERY="${PROFILE_FLOPS_EVERY:-1}"
USE_LLM_VERIFY="${USE_LLM_VERIFY:-1}"
VERBOSE="${VERBOSE:-1}"
GPUS_STR="${GPUS:-0 1 2 3 4 5 6}"

if [ -n "${HUGGING_FACE_TOKEN:-}" ]; then
    export HUGGING_FACE_TOKEN
fi
if [ "${USE_LLM_VERIFY}" = "1" ]; then
    : "${OPENAI_API_KEY:?Set OPENAI_API_KEY or run with USE_LLM_VERIFY=0 for efficiency-only smoke tests.}"
    export OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
    export MODEL_TYPE="${MODEL_TYPE:-qwen-max}"
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Model path not found: ${MODEL_PATH}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

DATASETS=("mmvp" "mmstar" "mm_math" "math_vista" "math_vision" "hallusion" "scienceqa")
read -r -a GPUS_ARR <<< "${GPUS_STR}"
if [ "${#GPUS_ARR[@]}" -eq 0 ]; then
    echo "No GPUs specified. Set GPUS=\"0 1 ...\"." >&2
    exit 1
fi

pids=()
for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    gpu="${GPUS_ARR[$((i % ${#GPUS_ARR[@]}))]}"
    log_path="${OUTPUT_ROOT}/qwen25vl7b_${dataset}_profile.log"
    echo "[$(date +%T)] Launching ${dataset} on GPU ${gpu} -> ${log_path}"

    extra_args=()
    if [ "${USE_LLM_VERIFY}" = "1" ]; then
        extra_args+=(--use_llm_verify)
    fi

    CUDA_VISIBLE_DEVICES="${gpu}" python main_vl_icot.py \
        --dataset "${dataset}" \
        --data_root "${DATA_ROOT}" \
        --image_root "${IMAGE_ROOT}" \
        --model_name_or_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_ROOT}" \
        --device cuda \
        --seed 42 \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --min_pixels "${MIN_PIXELS}" \
        --max_pixels "${MAX_PIXELS}" \
        --num_selected_patches "${NUM_SELECTED_PATCHES}" \
        --max_sub_imgs "${MAX_SUB_IMGS}" \
        --profile_efficiency \
        --profile_flops \
        --profile_flops_every "${PROFILE_FLOPS_EVERY}" \
        --resume \
        --verbose "${VERBOSE}" \
        "${extra_args[@]}" \
        > "${log_path}" 2>&1 &
    pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[$(date +%T)] Done: ${DATASETS[$i]}"
    else
        echo "[$(date +%T)] FAILED: ${DATASETS[$i]}" >&2
        failed=1
    fi
done

python scripts/summarize_icot_efficiency.py --output_root "${OUTPUT_ROOT}" || failed=1

if [ "${failed}" -ne 0 ]; then
    echo "One or more ICoT profile jobs failed. Check logs under ${OUTPUT_ROOT}." >&2
    exit 1
fi

echo "[$(date +%T)] All full-dataset ICoT profile jobs finished. Results in ${OUTPUT_ROOT}"
