#!/bin/bash
# run_icot_dev.sh — evaluate the ICoT baseline on the 7 dev splits for ONE model.
# Usage: bash scripts/run_icot_dev.sh <model_path> <suffix>

set -u

source /export/home/lanliwei.1/abcxyz/env/miniconda3/bin/activate
conda activate ltpo

MODEL_PATH="${1:-/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/Other_baseline/models/Qwen2.5-VL-7B-Instruct}"
SUFFIX="${2:-}"

:
"${HUGGING_FACE_TOKEN:?Set HUGGING_FACE_TOKEN before running this script}"
:
"${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script}"
export HUGGING_FACE_TOKEN OPENAI_API_KEY
export OPENAI_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export MODEL_TYPE="qwen-max"

output=./output/icot
mkdir -p "${output}"

DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

pids=()
for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    gpu="${i}"
    LOG="${output}/${SUFFIX}_${dataset}.log"
    echo "[$(date +%T)] Launching ${SUFFIX} ${dataset} on GPU ${gpu} -> ${LOG}"
    CUDA_VISIBLE_DEVICES=${gpu} python main_vl_icot.py \
        --dataset "${dataset}" \
        --data_root mllm_data \
        --image_root . \
        --model_name_or_path "${MODEL_PATH}" \
        --output_dir "${output}" \
        --device cuda \
        --seed 42 \
        --max_new_tokens 512 \
        --min_pixels 128 \
        --max_pixels 256 \
        --num_selected_patches 16 \
        --max_sub_imgs 3 \
        --use_llm_verify \
        --verbose 1 \
        > "${LOG}" 2>&1 &
    pids+=("$!")
done

for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
    echo "[$(date +%T)] Done: ${SUFFIX} ${DATASETS[$i]}"
done

echo "=== ICoT ${SUFFIX} all 7 dev sets finished. Results in ${output} ==="
