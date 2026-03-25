#!/bin/bash
# run_ltpo_vl_dmlr.sh — LTPO VL evaluation with DMLR-compatible pipeline.
#
# Generation configs, prompts, and verification mirror DMLR/script/run.sh.
# The LTPO code framework (pre-merged visual tokens, single-process) is kept.

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

export CUDA_VISIBLE_DEVICES=0

for dataset in "scienceqa"; do
    model=/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct

    python main_vl_dmlr.py \
        --dataset $dataset \
        --data_root mllm_data \
        --image_root . \
        --model_name_or_path $model \
        --output_dir ./output \
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
        --verbose 1
done
