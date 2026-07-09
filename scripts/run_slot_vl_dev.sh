#!/bin/bash
# run_slot_vl_dev.sh — SLOT baseline on the 7 dev splits.
# Iterates the 4 models in the requested order; for each model, fans out
# the 7 datasets across GPUs 1..7 (GPU 0 reserved for the user's own job).

cd "$(dirname "$0")/.."

source /export/home/lanliwei.1/abcxyz/env/miniconda3/bin/activate
conda activate ltpo
set -u

:
"${HUGGING_FACE_TOKEN:?Set HUGGING_FACE_TOKEN before running this script}"
:
"${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script}"
export HUGGING_FACE_TOKEN OPENAI_API_KEY
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

MODELS_ROOT=/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/Other_baseline/models
MODELS=(
    "${MODELS_ROOT}/Qwen2.5-VL-7B-Instruct"
    "${MODELS_ROOT}/Qwen3-VL-8B-Instruct"
    "${MODELS_ROOT}/Qwen3-VL-4B-Instruct"
    "${MODELS_ROOT}/Qwen2.5-VL-3B-Instruct"
)

DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")
GPUS=(1 2 3 4 5 6 7)

OUTPUT_ROOT=./output/slot
mkdir -p "${OUTPUT_ROOT}"

for MODEL in "${MODELS[@]}"; do
    mname=$(basename "${MODEL}")
    echo "############################################################"
    echo "[$(date +%F\ %T)] === MODEL: ${mname} ==="
    echo "############################################################"

    pids=()
    for i in "${!DATASETS[@]}"; do
        ds="${DATASETS[$i]}"
        gpu="${GPUS[$i]}"
        logf="${OUTPUT_ROOT}/${mname}__${ds}.log"
        echo "[$(date +%T)] -> launch ${ds} on GPU ${gpu} -> ${logf}"
        CUDA_VISIBLE_DEVICES=${gpu} python main_vl_slot.py \
            --dataset "${ds}" \
            --data_root mllm_data \
            --image_root . \
            --model_name_or_path "${MODEL}" \
            --output_dir "${OUTPUT_ROOT}" \
            --device cuda \
            --seed 42 \
            --max_new_tokens 2048 \
            --min_pixels 128 \
            --max_pixels 256 \
            --slot_times 3 \
            --slot_lr 0.01 \
            --use_llm_verify \
            --verbose 1 \
            > "${logf}" 2>&1 &
        pids+=("$!")
    done

    for k in "${!pids[@]}"; do
        wait "${pids[$k]}"
        echo "[$(date +%T)] done ${DATASETS[$k]} on GPU ${GPUS[$k]}"
    done
    echo "[$(date +%F\ %T)] === FINISHED MODEL: ${mname} ==="
done

echo "[$(date +%F\ %T)] All models / datasets finished."
