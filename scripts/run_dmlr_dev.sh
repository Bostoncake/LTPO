#!/bin/bash
# run_dmlr_dev.sh — DMLR method on 300-sample dev splits.
#
# Requires dev splits to exist; generate them first with:
#   python mllm_data/create_dev_splits.py

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}
GPU=${GPU:-0}

output=./output/dev/dmlr
mkdir -p "${output}"

for dataset in "mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev"; do
    echo "[$(date +%T)] ${dataset} on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=${GPU} python main_vl_dmlr.py \
        --dataset "${dataset}" \
        --data_root mllm_data \
        --image_root . \
        --model_name_or_path "${MODEL}" \
        --output_dir "${output}" \
        --device cuda \
        --seed 42 \
        --max_new_tokens 2048 \
        --min_pixels 128 \
        --max_pixels 256 \
        --num_thought_tokens 2 \
        --sigma 25.0 \
        --sigma_decay 0.95 \
        --lr 0.01 \
        --max_num_steps 15 \
        --top_k 10 \
        --use_llm_verify \
        --verbose 1 \
        > "${output}/${dataset}.log" 2>&1
    echo "[$(date +%T)] Done: ${dataset}"
done

echo "=== All dev sets finished. Results in ${output} ==="
