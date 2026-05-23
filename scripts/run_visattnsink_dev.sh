#!/bin/bash
# run_visattnsink_dev.sh — sweep 4 models × 7 datasets with VisAttnSink alone.
#
# Models are processed sequentially in the user's requested order:
#   qwen2.5-VL-7B -> qwen3-VL-8B -> qwen3-VL-4B -> qwen2.5-VL-3B
# Within each model, datasets run in parallel across the available GPUs.

set -u

# Do NOT hardcode secrets in scripts. Read from environment instead.
# Example: export HUGGING_FACE_TOKEN="hf_..." && export OPENAI_API_KEY="sk-..."
export HUGGING_FACE_TOKEN="${HUGGING_FACE_TOKEN:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

MODELS_DIR=${MODELS_DIR:-/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/Other_baseline/models}
MODELS=(
    "Qwen2.5-VL-7B-Instruct"
    "Qwen3-VL-8B-Instruct"
    "Qwen3-VL-4B-Instruct"
    "Qwen2.5-VL-3B-Instruct"
)

DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
N_GPUS=${#GPU_LIST[@]}

root_output=${ROOT_OUTPUT:-./output/visattnsink_dev}
mkdir -p "${root_output}"

for model in "${MODELS[@]}"; do
    model_path="${MODELS_DIR}/${model}"
    out_dir="${root_output}/${model}"
    mkdir -p "${out_dir}"

    echo "════════════════════════════════════════════════════════"
    echo "Model: ${model}"
    echo "Output: ${out_dir}"
    echo "GPUs: ${GPUS}  (${N_GPUS} workers in parallel)"
    echo "════════════════════════════════════════════════════════"

    declare -a gpu_pids
    for ((g=0; g<N_GPUS; g++)); do gpu_pids[$g]=-1; done

    job_idx=0
    total=${#DATASETS[@]}
    while [ $job_idx -lt $total ]; do
        for ((g=0; g<N_GPUS; g++)); do
            [ $job_idx -ge $total ] && break
            pid=${gpu_pids[$g]}
            if [ "$pid" -eq -1 ] || ! kill -0 "$pid" 2>/dev/null; then
                dataset="${DATASETS[$job_idx]}"
                gpu="${GPU_LIST[$g]}"
                log="${out_dir}/${dataset}.log"
                echo "[$(date +%T)] GPU ${gpu} <- ${dataset}"

                CUDA_VISIBLE_DEVICES=${gpu} python main_vl_visattnsink.py \
                    --dataset            "${dataset}"       \
                    --data_root          mllm_data          \
                    --image_root         .                  \
                    --model_name_or_path "${model_path}"    \
                    --output_dir         "${out_dir}"       \
                    --device             cuda               \
                    --seed               42                 \
                    --max_new_tokens     2048               \
                    --min_pixels         128                \
                    --max_pixels         256                \
                    --vas_tau            20.0               \
                    --vas_rho            0.5                \
                    --vas_summ           0.2                \
                    --vas_p              0.6                \
                    --vas_except_last_layer 1               \
                    --use_llm_verify                        \
                    --verbose            1                  \
                    > "${log}" 2>&1 &
                gpu_pids[$g]=$!
                job_idx=$((job_idx+1))
            fi
        done
        sleep 2
    done

    wait
    echo "[$(date +%T)] Model ${model} done."
done

echo ""
echo "════════════════════════════════════════════════════════"
echo "All models finished. Results in ${root_output}"
echo "Per-(model,dataset) accuracy:"
echo "════════════════════════════════════════════════════════"
for model_dir in "${root_output}"/*; do
    [ -d "$model_dir" ] || continue
    for cfg in "$model_dir"/*; do
        [ -d "$cfg" ] || continue
        log="$cfg/results.log"
        [ -f "$log" ] || continue
        acc=$(grep -oP 'accuracy=\K[0-9.]+' "$log" | tail -1)
        echo "$(basename "$model_dir")/$(basename "$cfg"): acc=${acc}"
    done
done
