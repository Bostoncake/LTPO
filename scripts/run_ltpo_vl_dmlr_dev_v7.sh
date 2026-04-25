#!/bin/bash
# run_ltpo_vl_dmlr_dev_v6.sh — v6 cherry-picked best prompts per dataset.
#
# v6 prompt design:
#   - system_prompt is dataset-specific, cherry-picked from v2/v3/v4
#     based on which performed best:
#       * MathVista     → v4 ("Interpret the visual information precisely...")
#       * MathVision    → v3 (plain "Please reason step by step...")
#       * MM-Math       → v2 ("Please reason step by step, and MUST...")
#       * HallusionBench→ v4 ("Look carefully at the image details...")
#       * MMVP          → v2 ("Please reason step by step, and MUST...")
#       * MMStar        → v4 ("Examine the image carefully...")
#       * ScienceQA     → v4 ("Apply relevant scientific knowledge...")
#   - prompt_instruction (user content) is also dataset-specific:
#       * v4 datasets: "{prompt}\nThe following tokens represent your
#         internal thinking space.\n{tokens}"
#       * v3 datasets (MathVision): "{prompt}\n{tokens}" (no bridge)
#       * v2 datasets (MM-Math, MMVP): "{prompt}\n\nThe following special
#         tokens represent YOUR INTERNAL THINKING SPACE where your
#         reasoning happens implicitly.\n{tokens}"
#
# Generation configs and verification mirror DMLR/script/run.sh.

export HUGGING_FACE_TOKEN="***REDACTED_HF_TOKEN***"
export OPENAI_API_KEY="***REDACTED_OPENAI_KEY***"
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

output=./output/dmlr_aligned_dev_v7_prompt
mkdir -p ${output}
gpu=0
# "mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev"
for dataset in "mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev"; do
    model=/export/home/lanliwei.1/abcxyz/storage/models/Qwen2.5-VL-3B-Instruct

    CUDA_VISIBLE_DEVICES=$gpu python main_vl_dmlr_v7.py \
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
