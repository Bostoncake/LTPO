#!/bin/bash
# run_ltpo_vl_dmlr_lookthink_grid_dev.sh
# LOOK/THINK grid: fix lookthink params, sweep LTPO params on the 7 dev sets.
#
# This script mirrors run_ltpo_vl_dmlr_optclip_bestr1_grid_dev.sh:
#   opt_reward  = clip   (optimise on r1 + lambda * max(r1 - r2, 0))
#   best_reward = r1     (pick best latent by r1 alone)
#   per-dataset gap_lambda inherited from the contrast-clipping grid.
#
# What's new: --enable_lookthink with fixed look_threshold and look_alpha.
# Same LTPO search space as the reference script.
#
# Sweeps: num_thought_tokens x max_num_steps x sigma x lr
# Fixed:  sigma_decay=0.95, top_k=10, look_threshold=0.0, look_alpha=0.1
#
# Usage:
#   N_GPUS=8 bash scripts/run_ltpo_vl_dmlr_lookthink_grid_dev.sh

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=${N_GPUS:-8}
MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}

TOKENS_LIST=(2 4)
STEPS_LIST=(10 15)
SIGMA_LIST=(5.0 25.0)
SIGMA_DECAY=0.95
LR_LIST=(5e-3 1e-2 5e-2)
TOP_K=10

# Fixed LOOK/THINK params for this stage of the search.
LOOK_THRESHOLD=0.01
LOOK_ALPHA=0.1

# Per-dataset best gap_lambda from the contrast-clipping grid (Qwen2.5-VL-7B).
declare -A GAP_LAMBDA
GAP_LAMBDA[math_vista_dev]=10.0
GAP_LAMBDA[math_vision_dev]=10.0
GAP_LAMBDA[mm_math_dev]=1.0
GAP_LAMBDA[hallusion_dev]=1.0
GAP_LAMBDA[mmvp_dev]=1.0
GAP_LAMBDA[mmstar_dev]=10.0
GAP_LAMBDA[scienceqa_dev]=10.0

DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

root_output=./output/ltpo_dmlr_contrastive_grid_dev/0502_lookthink_optclip_bestr1
mkdir -p "${root_output}"

jobs=()
for tokens in "${TOKENS_LIST[@]}"; do
    for steps in "${STEPS_LIST[@]}"; do
        for sigma in "${SIGMA_LIST[@]}"; do
            for lr in "${LR_LIST[@]}"; do
                for dataset in "${DATASETS[@]}"; do
                    jobs+=("${tokens} ${steps} ${sigma} ${lr} ${dataset}")
                done
            done
        done
    done
done
total=${#jobs[@]}

echo "========================================================"
echo "LTPO-DMLR LOOK/THINK Grid Search on dev sets"
echo "    objective: opt = r1 + lambda*clip(r1-r2, 0) ; best = r1"
echo "    lookthink: threshold=${LOOK_THRESHOLD}  alpha=${LOOK_ALPHA}  (fixed)"
echo "    per-dataset gap_lambda (from prior contrast-clipping grid):"
for d in "${DATASETS[@]}"; do
    echo "        ${d}: lambda=${GAP_LAMBDA[$d]}"
done
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
            read -r tokens steps sigma lr dataset <<< "${jobs[$job_idx]}"
            lam=${GAP_LAMBDA[$dataset]}

            tag="tokens${tokens}_steps${steps}_sigma${sigma}_decay${SIGMA_DECAY}_lr${lr}_topk${TOP_K}_optclip_bestr1_lt${LOOK_THRESHOLD}_la${LOOK_ALPHA}"
            out_dir="${root_output}/${tag}"
            mkdir -p "${out_dir}"
            log="${out_dir}/${dataset}.log"

            echo "[$(date +%T)] GPU ${g}  <-  ${tag}  |  ${dataset}  (lambda=${lam})"

            CUDA_VISIBLE_DEVICES=${g} python main_vl_dmlr_contrastive_lookthink.py \
                --dataset            "${dataset}"   \
                --data_root          mllm_data      \
                --image_root         .              \
                --model_name_or_path "${MODEL}"     \
                --output_dir         "${out_dir}"   \
                --device             cuda           \
                --seed               42             \
                --max_new_tokens     2048           \
                --min_pixels         128            \
                --max_pixels         256            \
                --num_thought_tokens "${tokens}"    \
                --sigma              "${sigma}"     \
                --sigma_decay        "${SIGMA_DECAY}" \
                --lr                 "${lr}"        \
                --max_num_steps      "${steps}"     \
                --top_k              "${TOP_K}"     \
                --gap_lambda         "${lam}"       \
                --decouple_reward                   \
                --opt_reward         clip           \
                --best_reward        r1             \
                --look_threshold     "${LOOK_THRESHOLD}" \
                --look_alpha         "${LOOK_ALPHA}" \
                --use_llm_verify                    \
                --verbose 1                         \
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
