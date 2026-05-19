#!/bin/bash
# run_ltpo_vl_dmlr_final_visual_dev.sh — evaluate the FINAL+DMLR-visual variant
# on the 300-sample dev splits of the seven MLLM datasets.
#
# The DMLR visual-injection defaults follow /home/xiongyizhe/research/DMLR/script/run.sh.

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}

output=./output/dmlr_final_visual
mkdir -p "${output}"

DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

pids=()
for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    gpu="${i}"
    echo "[$(date +%T)] Launching ${dataset} on GPU ${gpu}..."
    CUDA_VISIBLE_DEVICES=${gpu} python main_vl_dmlr_final_visual.py \
        --dataset "${dataset}" \
        --data_root mllm_data \
        --image_root . \
        --model_name_or_path "${MODEL}" \
        --output_dir "${output}" \
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
        --num_selected_patches 16 \
        --initial_patch_count 1 \
        --patch_increment 1 \
        --visual_insert_stride 1 \
        --visual_injection_start_step 0 \
        --visual_injection_interval 1 \
        --use_llm_verify \
        --verbose 1 \
        > "${output}/${dataset}.log" 2>&1 &
    pids+=("$!")
done

for i in "${!pids[@]}"; do
    wait "${pids[$i]}"
    echo "[$(date +%T)] Done: ${DATASETS[$i]}"
done

echo "=== All dev sets finished. Results in ${output} ==="
