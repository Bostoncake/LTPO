#!/bin/bash
# run_ltpo_vl_dmlr_3models.sh — LTPO VL evaluation on 3 models with dynamic GPU pool.
#
# Runs all (model, dataset) pairs across N_GPUS GPUs so no GPU ever sits idle:
# as soon as a GPU finishes a job it immediately picks up the next one.

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=8
MODEL_DIR=/WillDevExt/xiongyizhe/models
MODELS=(
    "Qwen2.5-VL-7B-Instruct"
    "Qwen3-VL-4B-Instruct"
    "Qwen3-VL-8B-Instruct"
)
DATASETS=("mmvp" "mmstar" "mm_math" "math_vista" "math_vision" "hallusion" "scienceqa")

# Build flat job list: all (model, dataset) pairs — 3 × 7 = 21 jobs
jobs=()
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        jobs+=("$model $dataset")
    done
done

output=/home/xiongyizhe/research/LTPO/output/dmlr_vanilla

# Dynamic GPU pool: poll every 2 s; dispatch next job to any free GPU
declare -a gpu_pids
for ((g=0; g<N_GPUS; g++)); do gpu_pids[$g]=-1; done

job_idx=0
total=${#jobs[@]}

while [ $job_idx -lt $total ]; do
    for ((g=0; g<N_GPUS; g++)); do
        [ $job_idx -ge $total ] && break
        pid=${gpu_pids[$g]}
        if [ $pid -eq -1 ] || ! kill -0 "$pid" 2>/dev/null; then
            read -r model dataset <<< "${jobs[$job_idx]}"
            echo "[$(date +%T)] GPU $g  ←  $model / $dataset"
            CUDA_VISIBLE_DEVICES=$g python main_vl_dmlr.py \
                --dataset "$dataset" \
                --data_root mllm_data \
                --image_root . \
                --model_name_or_path "${MODEL_DIR}/${model}" \
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
                --use_llm_verify \
                --eval_baseline \
                --verbose 1 > "${output}/${model}_${dataset}.log" 2>&1 &
            gpu_pids[$g]=$!
            job_idx=$((job_idx+1))
        fi
    done
    sleep 2
done

wait
echo "All $total jobs done."
