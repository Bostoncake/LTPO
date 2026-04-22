#!/bin/bash
# run_ltpo_vl_dmlr_dev_v3.sh — v3 dataset-specific minimal prompts.
#
# Prompt changes (v3 vs v1/v2):
#   SYSTEM_PROMPT is now dataset-specific via get_system_prompt():
#     - hallusion           → "...put your final answer (Yes or No) within \boxed{}." 
#     - mmvp/mmstar/scienceqa → "...put your final answer (the option letter) within \boxed{}." 
#     - mm_math/math_vista/math_vision → "...put your final answer within \boxed{}." 
#   input_content: just "{question}\n{thought_tokens}" — no "PROBLEM:" prefix,
#     no "INTERNAL THINKING SPACE" paragraph, no extra instructions.
#     The data prompts already contain their own \boxed{}/choice instructions.
#
# Everything else (model loading, RL loop, answer extraction, verification)
# is identical to the v1/v2 DMLR pipeline.

export HUGGING_FACE_TOKEN="***REDACTED_HF_TOKEN***"
export OPENAI_API_KEY="***REDACTED_OPENAI_KEY***"
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

output=./output/dmlr_aligned_dev_v3_prompt
mkdir -p ${output}
gpu=0
# "mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev"
for dataset in "mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev"; do
    model=/export/home/lanliwei.1/abcxyz/storage/models/Qwen2.5-VL-3B-Instruct

    CUDA_VISIBLE_DEVICES=$gpu python main_vl_dmlr_v3.py \
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
