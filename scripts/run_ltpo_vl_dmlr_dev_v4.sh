#!/bin/bash
# run_ltpo_vl_dmlr_dev_v4.sh — v4 dataset-specific prompts with strategy
# hints and restored bridge text.
#
# Prompt changes (v4 vs v3):
#   SYSTEM_PROMPT is still dataset-specific via get_system_prompt(), but now
#   each includes a brief task-strategy hint BEFORE the baseline sentence:
#     - hallusion   → "Look carefully at the image details before deciding. Please reason step by step, ..."
#     - mmvp        → "Pay close attention to visual details in the image. ..."
#     - mmstar      → "Examine the image carefully and consider each option. ..."
#     - scienceqa   → "Apply relevant scientific knowledge to the question. ..."
#     - mm_math     → "Identify key information from the figure for your calculations. ..."
#     - math_vista  → "Interpret the visual information precisely before solving. ..."
#     - math_vision → "Analyze the geometric or mathematical figure carefully. ..."
#     - default     → "Please reason step by step, and put your final answer within \boxed{}."
#
#   input_content restores the short bridge sentence between question and
#   thought tokens (v3 removed it and performance dropped):
#     "{prompt}\nThe following tokens represent your internal thinking space.\n{tokens}"
#
# Everything else (model loading, RL loop, answer extraction, verification)
# is identical to the v1–v3 DMLR pipeline.

export HUGGING_FACE_TOKEN="***REDACTED_HF_TOKEN***"
export OPENAI_API_KEY="***REDACTED_OPENAI_KEY***"
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

output=./output/dmlr_aligned_dev_v4_prompt
mkdir -p ${output}
gpu=0
# "mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev"
for dataset in "mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev"; do
    model=/export/home/lanliwei.1/abcxyz/storage/models/Qwen2.5-VL-3B-Instruct

    CUDA_VISIBLE_DEVICES=$gpu python main_vl_dmlr_v4.py \
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
        --verbose 1 > ${output}/${dataset}.log 2>&1 &
    gpu=$((gpu+1))
done
wait
