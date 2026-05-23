#!/bin/bash

export HUGGING_FACE_TOKEN="${HUGGING_FACE_TOKEN:?set HUGGING_FACE_TOKEN before running}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY before running}"
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

output=./output/dmlr_aligned_dev_v8_prompt_baseline_8B
mkdir -p ${output}
gpu=0
# "mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev"
# "mmstar_dev" "math_vista_dev" "hallusion_dev"
for dataset in "mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev"; do
    # model=/export/home/lanliwei.1/abcxyz/storage/models/Qwen2.5-VL-3B-Instruct
    model=/export/home/lanliwei.1/abcxyz/storage/models/Qwen3-VL-8B-Instruct

    CUDA_VISIBLE_DEVICES=$gpu python main_vl_dmlr_v8.py \
        --dataset $dataset \
        --data_root mllm_data \
        --image_root . \
        --model_name_or_path $model \
        --output_dir ${output} \
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
        --eval_baseline \
        --verbose 1 > ${output}/${dataset}.log 2>&1 &
    gpu=$((gpu+1))
    # if [ $gpu -eq 5 ]; then
    #     gpu=6
    # fi
done
wait
