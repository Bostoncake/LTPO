#!/bin/bash
# run_ltpo_vl_dmlr_final_baseline_3models.sh
#
# LTPO baseline run on the FINAL variant (main_vl_dmlr_final.py) across the
# three target backbones — Qwen2.5-VL-3B, Qwen3-VL-4B, Qwen3-VL-8B — using a
# single fixed hyperparameter set per experiment.
#
# Configuration:
#   - Direct-boxed LTPO variant (main_vl_dmlr_final.py).
#   - --use_baseline_prompt    : raw question stays in the user turn, latent
#                                thought tokens sit in the assistant turn
#                                right before "\boxed{".
#   - --reward_on_latent_tokens: confidence/entropy reward averaged at every
#                                latent thought-token position instead of the
#                                first-generated-token position.
#   - One fixed hyperparam group (matches the visual_dev defaults):
#       tokens=2, sigma=25.0, sigma_decay=0.95, lr=0.01, steps=15, top_k=10.
#   - Default reward_type=confidence (FINAL's default), no lookthink.
#
# Usage:
#   bash scripts/run_ltpo_vl_dmlr_final_baseline_3models.sh
#
# Overridable env vars:
#   N_GPUS    number of GPUs to use (default 8)
#   MODEL_DIR root directory holding the three model checkpoints

set -u

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export MODEL_TYPE="${MODEL_TYPE:-qwen-max}"

N_GPUS=${N_GPUS:-8}
MODEL_DIR=${MODEL_DIR:-/WillDevExt/xiongyizhe/models}
MODELS=(
    "Qwen2.5-VL-3B-Instruct"
    "Qwen3-VL-4B-Instruct"
    "Qwen3-VL-8B-Instruct"
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

NUM_TOKENS=2
SIGMA=25.0
SIGMA_DECAY=0.95
LR=0.01
STEPS=15
TOP_K=10

root_output=./output/ltpo_dmlr_final/baseline_3models
mkdir -p "${root_output}"

jobs=()
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        jobs+=("${model} ${dataset}")
    done
done
total=${#jobs[@]}

echo "========================================================"
echo "LTPO-DMLR FINAL baseline run (3 models, fixed hparams)"
echo "    use_baseline_prompt=ON  reward_on_latent_tokens=ON"
echo "    tokens=${NUM_TOKENS}  sigma=${SIGMA}  sigma_decay=${SIGMA_DECAY}"
echo "    lr=${LR}  steps=${STEPS}  top_k=${TOP_K}"
echo "    Models  : ${MODELS[*]}"
echo "    Datasets: ${DATASETS[*]}"
echo "    Total jobs: ${total}  GPUs: ${N_GPUS}"
echo "========================================================"

declare -a gpu_pids
for ((g=0; g<N_GPUS; g++)); do gpu_pids[$g]=-1; done

job_idx=0
while [ $job_idx -lt $total ]; do
    for ((g=0; g<N_GPUS; g++)); do
        [ $job_idx -ge $total ] && break
        pid=${gpu_pids[$g]}
        if [ "$pid" -eq -1 ] || ! kill -0 "$pid" 2>/dev/null; then
            read -r model dataset <<< "${jobs[$job_idx]}"
            out_dir="${root_output}/${model}"
            mkdir -p "${out_dir}"
            log="${out_dir}/${dataset}.log"

            echo "[$(date +%T)] GPU ${g}  <-  ${model} / ${dataset}"

            CUDA_VISIBLE_DEVICES=${g} python main_vl_dmlr_final.py \
                --dataset            "${dataset}"      \
                --data_root          mllm_data         \
                --image_root         .                 \
                --model_name_or_path "${MODEL_DIR}/${model}" \
                --output_dir         "${out_dir}"      \
                --device             cuda              \
                --seed               42                \
                --max_new_tokens     2048              \
                --min_pixels         128               \
                --max_pixels         256               \
                --num_thought_tokens "${NUM_TOKENS}"   \
                --sigma              "${SIGMA}"        \
                --sigma_decay        "${SIGMA_DECAY}"  \
                --lr                 "${LR}"           \
                --max_num_steps      "${STEPS}"        \
                --top_k              "${TOP_K}"        \
                --use_baseline_prompt                  \
                --reward_on_latent_tokens              \
                --use_llm_verify                       \
                --verbose 1                            \
                > "${log}" 2>&1 &

            gpu_pids[$g]=$!
            job_idx=$((job_idx+1))
        fi
    done
    sleep 2
done

wait
echo ""
echo "========================================================"
echo "All ${total} jobs done. Results in ${root_output}"
echo "========================================================"

# Summarise mean accuracy across datasets per model dir
for model_dir in "${root_output}"/*; do
    [ -d "$model_dir" ] || continue
    model=$(basename "$model_dir")
    total_acc=0
    count=0
    while IFS= read -r results_log; do
        acc=$(grep -oP 'accuracy=\K[0-9.]+' "$results_log" | tail -1)
        [ -n "$acc" ] && total_acc=$(awk "BEGIN{print $total_acc + $acc}") && count=$((count+1))
    done < <(find "$model_dir" -name "results.log")
    if [ "$count" -gt 0 ]; then
        mean=$(awk "BEGIN{printf \"%.4f\", $total_acc / $count}")
        echo "mean_acc=${mean}  n=${count}  | ${model}"
    fi
done | sort -t'=' -k2 -rn
