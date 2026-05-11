#!/bin/bash
# run_ltpo_vl_dmlr_contrastive_grid_dev.sh
# Contrastive LTPO (r1 - r2) hyperparameter sweep on all 7 dev sets.
#
# Reward:  r1 - r2
#   r1 = confidence with full visual attention (standard LTPO).
#   r2 = confidence with image-token attention masked out.
# Optimising r1 - r2 widens the reasoning gap between "with image" and
# "without image", pushing latent thoughts to rely on the visual evidence.
#
# Sweeps: num_thought_tokens x max_num_steps x sigma x lr
# Fixed:  sigma_decay=0.95, top_k=10
#
# Usage:
#   N_GPUS=8 bash scripts/run_ltpo_vl_dmlr_contrastive_grid_dev.sh

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=${N_GPUS:-8}
MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}

# TOKENS_LIST=(2 4)
# STEPS_LIST=(10 15)
# SIGMA_LIST=(5.0 25.0)
# SIGMA_DECAY=0.95
# LR_LIST=(5e-3 1e-2 5e-2)
TOKENS_LIST=(4)
STEPS_LIST=(15)
SIGMA_LIST=(5.0)
SIGMA_DECAY=0.95
LR_LIST=(5e-3 1e-2 5e-2)
TOP_K=10

# Gap-bonus switch: USE_GAP_BONUS=1 enables reward = r1 + lambda*clip(r1-r2, 0).
# When USE_GAP_BONUS=0 (default) the reward stays as r1 - r2 and LAMBDA_LIST is ignored.
USE_GAP_BONUS=${USE_GAP_BONUS:-1}
LAMBDA_LIST=(1.0 10.0)

DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

root_output=./output/ltpo_dmlr_contrastive_grid_dev/0425_contrast_clipping_qwen2.5_7B
mkdir -p "${root_output}"

if [ "${USE_GAP_BONUS}" -eq 1 ]; then
    LAMBDAS=("${LAMBDA_LIST[@]}")
else
    LAMBDAS=("0")   # placeholder; not used when USE_GAP_BONUS=0
fi

jobs=()
for tokens in "${TOKENS_LIST[@]}"; do
    for steps in "${STEPS_LIST[@]}"; do
        for sigma in "${SIGMA_LIST[@]}"; do
            for lr in "${LR_LIST[@]}"; do
                for lam in "${LAMBDAS[@]}"; do
                    for dataset in "${DATASETS[@]}"; do
                        jobs+=("${tokens} ${steps} ${sigma} ${lr} ${lam} ${dataset}")
                    done
                done
            done
        done
    done
done
total=${#jobs[@]}

if [ "${USE_GAP_BONUS}" -eq 1 ]; then
    obj_str="r1 + lambda*clip(r1-r2, 0)   lambda in {${LAMBDA_LIST[*]}}"
else
    obj_str="r1 - r2"
fi
echo "========================================================"
echo "LTPO-DMLR CONTRASTIVE Grid Search on dev sets"
echo "    objective: ${obj_str}"
echo "    tokens in {${TOKENS_LIST[*]}}  steps in {${STEPS_LIST[*]}}"
echo "    sigma  in {${SIGMA_LIST[*]}}   lr    in {${LR_LIST[*]}}"
echo "    sigma_decay=${SIGMA_DECAY}  top_k=${TOP_K}  (fixed)"
echo "    Datasets: ${#DATASETS[@]}  Configs: $(( total / ${#DATASETS[@]} ))"
echo "    Total jobs: ${total}   GPUs: ${N_GPUS}"
echo "========================================================"

declare -a gpu_pids
for ((g=0; g<N_GPUS; g++)); do gpu_pids[$g]=-1; done

job_idx=0
while [ $job_idx -lt $total ]; do
    for ((g=0; g<N_GPUS; g++)); do
        [ $job_idx -ge $total ] && break
        pid=${gpu_pids[$g]}
        if [ "$pid" -eq -1 ] || ! kill -0 "$pid" 2>/dev/null; then
            read -r tokens steps sigma lr lam dataset <<< "${jobs[$job_idx]}"

            if [ "${USE_GAP_BONUS}" -eq 1 ]; then
                reward_tag="gapbonus${lam}"
                gap_args=(--use_gap_bonus --gap_lambda "${lam}")
            else
                reward_tag="contrastive"
                gap_args=()
            fi

            tag="tokens${tokens}_steps${steps}_sigma${sigma}_decay${SIGMA_DECAY}_lr${lr}_topk${TOP_K}_${reward_tag}"
            out_dir="${root_output}/${tag}"
            mkdir -p "${out_dir}"
            log="${out_dir}/${dataset}.log"

            echo "[$(date +%T)] GPU ${g}  <-  ${tag}  |  ${dataset}"

            CUDA_VISIBLE_DEVICES=${g} python main_vl_dmlr_contrastive.py \
                --dataset           "${dataset}"   \
                --data_root         mllm_data      \
                --image_root        .              \
                --model_name_or_path "${MODEL}"    \
                --output_dir        "${out_dir}"   \
                --device            cuda           \
                --seed              42             \
                --max_new_tokens    2048           \
                --min_pixels        128            \
                --max_pixels        256            \
                --num_thought_tokens "${tokens}"   \
                --sigma             "${sigma}"     \
                --sigma_decay       "${SIGMA_DECAY}" \
                --lr                "${lr}"        \
                --max_num_steps     "${steps}"     \
                --top_k             "${TOP_K}"     \
                --use_llm_verify                   \
                --verbose 1                        \
                "${gap_args[@]}"                   \
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
echo "Results per config (sorted by mean accuracy across datasets):"
echo "========================================================"

for cfg_dir in "${root_output}"/tokens*; do
    [ -d "$cfg_dir" ] || continue
    cfg=$(basename "$cfg_dir")
    total_acc=0
    count=0
    while IFS= read -r log; do
        acc=$(grep -oP 'accuracy=\K[0-9.]+' "$log" | tail -1)
        [ -n "$acc" ] && total_acc=$(awk "BEGIN{print $total_acc + $acc}") && count=$((count+1))
    done < <(find "$cfg_dir" -name "results.log")
    if [ "$count" -gt 0 ]; then
        mean=$(awk "BEGIN{printf \"%.4f\", $total_acc / $count}")
        echo "mean_acc=${mean}  n=${count}  | ${cfg}"
    fi
done | sort -t'=' -k2 -rn
