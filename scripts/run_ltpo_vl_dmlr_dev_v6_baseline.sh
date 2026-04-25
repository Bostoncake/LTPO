#!/bin/bash
# run_ltpo_vl_dmlr_dev_v6_baseline.sh — v6 baseline evaluation.
#
# Baseline mode: ALL datasets use the unified v2 system_prompt:
#   "Please reason step by step, and MUST put your final answer within \boxed{}."
# No thought tokens, no bridge text — just the raw question as user content.
#
# This is the same system_prompt that v2 used, applied uniformly across
# all datasets for a fair baseline comparison against the v6 LTPO run.
#
# Generation configs and verification mirror DMLR/script/run.sh.

export HUGGING_FACE_TOKEN="***REDACTED_HF_TOKEN***"
export OPENAI_API_KEY="***REDACTED_OPENAI_KEY***"
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

output=./output/dmlr_aligned_dev_v6_baseline
mkdir -p ${output}
gpu=0
# "mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev"
for dataset in "mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev"; do
    model=/export/home/lanliwei.1/abcxyz/storage/models/Qwen2.5-VL-3B-Instruct

    CUDA_VISIBLE_DEVICES=$gpu python main_vl_dmlr_v6.py \
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
done
wait
