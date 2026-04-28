#!/bin/bash

export HUGGING_FACE_TOKEN="***REDACTED_HF_TOKEN***"
export OPENAI_API_KEY="***REDACTED_OPENAI_KEY***"
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

output=./output/dmlr_aligned_dev_v7_short_prompt_baseline
mkdir -p "${output}"

N_GPUS=${N_GPUS:-8}
MODELS=(
    "/export/home/lanliwei.1/abcxyz/storage/models/Qwen3-VL-4B-Instruct"
    "/export/home/lanliwei.1/abcxyz/storage/models/Qwen3-VL-8B-Instruct"
)
DATASETS=(
    "mmvp_dev"
    "mmstar_dev"
    "mm_math_dev"
    "math_vista_dev"
    "math_vision_dev"
    "hallusion_dev"
    "scienceqa_dev"
)

jobs=()
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        jobs+=("${model} ${dataset}")
    done
done

declare -a gpu_pids
for ((gpu=0; gpu<N_GPUS; gpu++)); do
    gpu_pids[$gpu]=-1
done

job_idx=0
total=${#jobs[@]}

echo "=== LTPO VL DMLR dev v7 two-model run ==="
echo "    GPUs: ${N_GPUS}"
echo "    Output: ${output}"
echo "    Total jobs: ${total}"

while [ "${job_idx}" -lt "${total}" ]; do
    for ((gpu=0; gpu<N_GPUS; gpu++)); do
        [ "${job_idx}" -ge "${total}" ] && break

        pid=${gpu_pids[$gpu]}
        if [ "${pid}" -eq -1 ] || ! kill -0 "${pid}" 2>/dev/null; then
            read -r model dataset <<< "${jobs[$job_idx]}"
            model_name=$(basename "${model}")
            log="${output}/${model_name}_${dataset}.log"

            echo "[$(date +%T)] GPU ${gpu} <- ${model_name} / ${dataset}"

            CUDA_VISIBLE_DEVICES=${gpu} python main_vl_dmlr_v7.py \
                --dataset "${dataset}" \
                --data_root mllm_data \
                --image_root . \
                --model_name_or_path "${model}" \
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
                --verbose 1 > "${log}" 2>&1 &

            gpu_pids[$gpu]=$!
            job_idx=$((job_idx+1))
        fi
    done
    sleep 2
done

wait
echo "=== All ${total} jobs done. Results in ${output} ==="
