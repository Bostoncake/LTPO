#!/bin/bash
# run_ltpo_vl_dmlr.sh — LTPO VL evaluation with DMLR-compatible pipeline.
#
# Generation configs, prompts, and verification mirror DMLR/script/run.sh.
# The LTPO code framework (pre-merged visual tokens, single-process) is kept.

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

output=./output/ltpo_dmlr_aligned_params_dev_no_optimize_cmp
mkdir -p ${output}
model_dir="/WillDevExt/xiongyizhe/models"

# for model in "Qwen2.5-VL-3B-Instruct" "Qwen3-VL-4B-Instruct" "Qwen3-VL-8B-Instruct"; do
for model in "Qwen2.5-VL-3B-Instruct"; do
    gpu=0
    for dataset in "mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev"; do

        CUDA_VISIBLE_DEVICES=$gpu python main_vl_dmlr.py \
            --dataset $dataset \
            --data_root mllm_data \
            --image_root . \
            --model_name_or_path $model_dir/$model \
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
done