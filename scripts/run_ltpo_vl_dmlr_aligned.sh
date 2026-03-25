#!/bin/bash
# run_ltpo_vl_dmlr_aligned.sh
# LTPO evaluation on MLLM benchmarks with prompts/seed/sampling aligned to DMLR.
#
# Alignment summary:
#   - System prompt: DMLR's <think>/<answer> system prompt
#   - User prompt:   DMLR's vl_cot_prompt idx=0 (+ latent tokens)
#   - Data prompts:  LTPO-specific prefixes stripped at load time
#   - dtype:         float32 (same as DMLR)
#   - Sampling:      do_sample=False, num_beams=1 (greedy, same as DMLR)
#   - Seed:          42 (same as DMLR)

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

export CUDA_VISIBLE_DEVICES=0

model=Qwen/Qwen2.5-VL-7B-Instruct
image_root=.
max_new_tokens=2048

# LTPO optimization params
num_thought_tokens=4
lr=5e-2
sigma=5.0
sigma_decay=0.9
max_num_steps=10
topk=10

for dataset in "mmstar" "mm_math" "math_vista" "math_vision" "hallusion" "scienceqa" "mmvp"; do
    output_dir=./output/dmlr_aligned/${dataset}
    echo "Running ${dataset} ..."

    python main_vl.py \
        --dataset "${dataset}" \
        --data_root mllm_data \
        --image_root "${image_root}" \
        --model_name_or_path "${model}" \
        --output_dir "${output_dir}" \
        --device cuda \
        --seed 42 \
        --max_new_tokens ${max_new_tokens} \
        --max_num_steps ${max_num_steps} \
        --num_thought_tokens ${num_thought_tokens} \
        --sigma ${sigma} \
        --sigma_decay ${sigma_decay} \
        --lr ${lr} \
        --top_k ${topk} \
        --verbose 0

    echo "Done ${dataset}"
done
