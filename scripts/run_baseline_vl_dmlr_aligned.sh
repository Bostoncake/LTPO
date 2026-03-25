#!/bin/bash
# run_baseline_vl_dmlr_aligned.sh
# Baseline (no LTPO) evaluation on MLLM benchmarks, aligned with DMLR.
#
# This runs standard VL inference (--eval_baseline) with the same
# system prompt, user prompt, dtype, sampling, and seed as DMLR.

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

export CUDA_VISIBLE_DEVICES=0

model=Qwen/Qwen2.5-VL-7B-Instruct
image_root=.
max_new_tokens=2048

for dataset in "mmstar" "mm_math" "math_vista" "math_vision" "hallusion" "scienceqa" "mmvp"; do
    output_dir=./output/dmlr_aligned_baseline/${dataset}
    echo "Running baseline ${dataset} ..."

    python main_vl.py \
        --dataset "${dataset}" \
        --data_root mllm_data \
        --image_root "${image_root}" \
        --model_name_or_path "${model}" \
        --output_dir "${output_dir}" \
        --device cuda \
        --seed 42 \
        --max_new_tokens ${max_new_tokens} \
        --eval_baseline \
        --verbose 0

    echo "Done ${dataset}"
done
