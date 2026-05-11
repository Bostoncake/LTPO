#!/bin/bash
# run_ltpo_vl_original_vanilla_dev.sh — Vanilla baseline via original LTPO
# pipeline (main_vl.py --eval_baseline) on dev splits for Qwen2.5-VL-3B
# and Qwen3-VL models.

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

MODEL_DIR=/WillDevExt/xiongyizhe/models
MODELS=(
    "Qwen2.5-VL-3B-Instruct"
    "Qwen3-VL-4B-Instruct"
    "Qwen3-VL-8B-Instruct"
)
DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

output=./output/ltpo_original_vanilla_dev
mkdir -p "${output}"

# Build flat job list: model × dataset = 3 × 7 = 21 jobs
jobs=()
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        jobs+=("$model $dataset")
    done
done

total=${#jobs[@]}
N_GPUS=8
echo "════════════════════════════════════════════════"
echo "Vanilla baseline (main_vl.py --eval_baseline) on dev splits"
echo "  Total jobs: $total   GPUs: $N_GPUS"
echo "════════════════════════════════════════════════"

# Dynamic GPU pool
declare -a gpu_pids
for ((g=0; g<N_GPUS; g++)); do gpu_pids[$g]=-1; done

job_idx=0
while [ $job_idx -lt $total ]; do
    for ((g=0; g<N_GPUS; g++)); do
        [ $job_idx -ge $total ] && break
        pid=${gpu_pids[$g]}
        if [ $pid -eq -1 ] || ! kill -0 "$pid" 2>/dev/null; then
            read -r model dataset <<< "${jobs[$job_idx]}"

            exp_output="${output}/${model}/${dataset}"
            mkdir -p "${exp_output}"

            echo "[$(date +%T)] GPU $g  ←  $model / $dataset"

            CUDA_VISIBLE_DEVICES=$g python main_vl.py \
                --dataset "$dataset" \
                --data_root mllm_data \
                --image_root . \
                --model_name_or_path "${MODEL_DIR}/${model}" \
                --output_dir "${exp_output}" \
                --device cuda \
                --seed 42 \
                --max_new_tokens 2048 \
                --num_thought_tokens 4 \
                --sigma 5.0 \
                --sigma_decay 0.9 \
                --lr 5e-2 \
                --max_num_steps 10 \
                --top_k 10 \
                --eval_baseline \
                --verbose 0 \
                > "${exp_output}/baseline.stdout" 2>&1 &

            gpu_pids[$g]=$!
            job_idx=$((job_idx+1))
        fi
    done
    sleep 2
done

wait
echo ""
echo "════════════════════════════════════════════════"
echo "All $total jobs done. Results:"
echo "════════════════════════════════════════════════"
for model in "${MODELS[@]}"; do
    echo ""
    echo "── ${model} ──"
    find "${output}/${model}" -name "results.log" | while read -r log; do
        last=$(tail -1 "$log")
        echo "$last  | $(dirname "$log" | sed "s|${output}/${model}/||")"
    done | sort -t'=' -k4 -rn
done
