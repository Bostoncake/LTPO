#!/bin/bash
# Run VL LTPO evaluation on a multimodal dataset.
# Usage: bash scripts/run_vl_ltpo.sh

dataset="data/mm_math.json"
model="Qwen/Qwen2.5-VL-7B-Instruct"
max_new_tokens=2048
max_num_steps=20
num_thought_tokens=8
sigma=20.0
sigma_decay=0.95
lr=0.005
verbose=1

python vl_main.py \
    --dataset $dataset \
    --model_name_or_path $model \
    --output_dir ./output \
    --device cuda \
    --max_new_tokens $max_new_tokens \
    --max_num_steps $max_num_steps \
    --num_thought_tokens $num_thought_tokens \
    --sigma $sigma \
    --sigma_decay $sigma_decay \
    --lr $lr \
    --verbose $verbose \
    --start_data_idx 0 \
    --end_data_idx 100
