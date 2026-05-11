#!/bin/bash
# run_ltpo_vl_original_dev.sh — Original LTPO (main_vl.py) on dev splits
# for Qwen2.5-VL-3B and Qwen3-VL models.
#
# Uses the top-5 hyperparameter configs from grid search on Qwen2.5-VL-7B.
# Dynamic GPU pool so no GPU sits idle.

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=8
MODEL_DIR=/WillDevExt/xiongyizhe/models
MODELS=(
    "Qwen2.5-VL-3B-Instruct"
    "Qwen3-VL-4B-Instruct"
    "Qwen3-VL-8B-Instruct"
)
DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

# Top-5 configs from grid search (tokens lr sigma sigma_decay steps topk)
configs=(
    "4  5e-2  5.0  0.9   10  10"   # rank-1: Cluster-A, high-lr/high-sigma, decay=0.9
    "4  1e-2  0.1  0.95  10  10"   # rank-2: Cluster-A, low-lr/low-sigma,   decay=0.95
    "2  5e-3  5.0  0.95  15  10"   # rank-3: Cluster-B, high-lr/high-sigma, decay=0.95
    "2  1e-2  0.1  0.95  15  10"   # rank-4: Cluster-B, low-lr/low-sigma,   decay=0.95
    "2  1e-2  25.0 0.95  15  10"   # original DMLR configs
)

output=./output/ltpo_original_dev
mkdir -p "${output}"

# Build flat job list: model × dataset × config
jobs=()
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for cfg in "${configs[@]}"; do
            jobs+=("$model $dataset $cfg")
        done
    done
done

total=${#jobs[@]}
echo "════════════════════════════════════════════════"
echo "Original LTPO (main_vl.py) on dev splits"
echo "  Models    : ${#MODELS[@]}"
echo "  Datasets  : ${#DATASETS[@]}"
echo "  Configs   : ${#configs[@]}"
echo "  Total jobs: $total"
echo "  GPUs      : $N_GPUS"
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
            read -r model dataset tokens lr sigma decay steps topk <<< "${jobs[$job_idx]}"

            tag="tokens${tokens}_lr${lr}_sigma${sigma}_decay${decay}_steps${steps}_topk${topk}"
            exp_output="${output}/${model}/${dataset}"
            mkdir -p "${exp_output}"

            echo "[$(date +%T)] GPU $g  ←  $model / $dataset / $tag"

            CUDA_VISIBLE_DEVICES=$g python main_vl.py \
                --dataset "$dataset" \
                --data_root mllm_data \
                --image_root . \
                --model_name_or_path "${MODEL_DIR}/${model}" \
                --output_dir "${exp_output}" \
                --device cuda \
                --seed 42 \
                --max_new_tokens 2048 \
                --num_thought_tokens ${tokens} \
                --sigma ${sigma} \
                --sigma_decay ${decay} \
                --lr ${lr} \
                --max_num_steps ${steps} \
                --top_k ${topk} \
                --verbose 0 \
                > "${exp_output}/${tag}.stdout" 2>&1 &

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
