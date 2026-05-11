#!/bin/bash
# run_ltpo_vl_dpo_grid_dev.sh
# Hyperparameter sweep for the Contrastive Latent-DPO test-time optimisation
# (main_vl_dpo.py).  In this sweep the DPO-related parameters are FIXED and we
# only vary the LTPO-side parameters: num_thought_tokens x max_num_steps x
# sigma x lr.  Use this to find the best LTPO regime under DPO updates before
# tuning DPO-specific knobs.
#
# Sweeps: num_thought_tokens x max_num_steps x sigma x lr
# Fixed (LTPO):  sigma_decay=0.95, top_k=10
# Fixed (DPO):   tt_opt_method=contrastive_dpo,
#                dpo_num_candidates=4, dpo_beta=0.1, dpo_alpha=1.0,
#                lambda_mask=1.0, dpo_margin=0.0, dpo_topk_pairs=0,
#                use_soft_dpo=True, mask_ratio=0.3, mask_fill=zero
#
# Usage:
#   N_GPUS=8 bash scripts/run_ltpo_vl_dpo_grid_dev.sh

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=${N_GPUS:-8}
MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}

# ---- LTPO grid (swept) ----
TOKENS_LIST=(2 4)
STEPS_LIST=(10 15)
SIGMA_LIST=(5.0 25.0)
SIGMA_DECAY=0.95
LR_LIST=(5e-3 1e-2 5e-2)
TOP_K=10

# ---- DPO config (fixed for this sweep) ----
TT_OPT_METHOD=contrastive_dpo
DPO_NUM_CANDIDATES=4
DPO_BETA=0.1
DPO_ALPHA=1.0
LAMBDA_MASK=1.0
DPO_MARGIN=0.0
DPO_TOPK_PAIRS=0
USE_SOFT_DPO=True
MASK_RATIO=1.0
MASK_FILL=zero

DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

root_output=./output/ltpo_dmlr_dpo_grid_dev/0430_dpo_fixed_ltpo_sweep
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
echo "Contrastive Latent-DPO LTPO-side Grid Search on dev sets"
echo "    DPO config (fixed):"
echo "        method=${TT_OPT_METHOD}  B=${DPO_NUM_CANDIDATES}"
echo "        beta=${DPO_BETA}  alpha=${DPO_ALPHA}  lambda_mask=${LAMBDA_MASK}"
echo "        margin=${DPO_MARGIN}  topk_pairs=${DPO_TOPK_PAIRS}"
echo "        use_soft_dpo=${USE_SOFT_DPO}  mask_ratio=${MASK_RATIO}  mask_fill=${MASK_FILL}"
echo "    LTPO sweep:"
echo "        tokens in {${TOKENS_LIST[*]}}  steps in {${STEPS_LIST[*]}}"
echo "        sigma  in {${SIGMA_LIST[*]}}   lr    in {${LR_LIST[*]}}"
echo "        sigma_decay=${SIGMA_DECAY}  top_k=${TOP_K}  (fixed)"
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

            tag="tokens${tokens}_steps${steps}_sigma${sigma}_decay${SIGMA_DECAY}_lr${lr}_topk${TOP_K}_dpoB${DPO_NUM_CANDIDATES}_beta${DPO_BETA}_mask${MASK_RATIO}${MASK_FILL}"
            out_dir="${root_output}/${tag}"
            mkdir -p "${out_dir}"
            log="${out_dir}/${dataset}.log"

            echo "[$(date +%T)] GPU ${g}  <-  ${tag}  |  ${dataset}"

            CUDA_VISIBLE_DEVICES=${g} python main_vl_dpo.py \
                --dataset             "${dataset}"           \
                --data_root           mllm_data              \
                --image_root          .                      \
                --model_name_or_path  "${MODEL}"             \
                --output_dir          "${out_dir}"           \
                --device              cuda                   \
                --seed                42                     \
                --max_new_tokens      2048                   \
                --min_pixels          128                    \
                --max_pixels          256                    \
                --num_thought_tokens  "${tokens}"            \
                --sigma               "${sigma}"             \
                --sigma_decay         "${SIGMA_DECAY}"       \
                --lr                  "${lr}"                \
                --max_num_steps       "${steps}"             \
                --top_k               "${TOP_K}"             \
                --tt_opt_method       "${TT_OPT_METHOD}"     \
                --dpo_num_candidates  "${DPO_NUM_CANDIDATES}" \
                --dpo_beta            "${DPO_BETA}"          \
                --dpo_alpha           "${DPO_ALPHA}"         \
                --lambda_mask         "${LAMBDA_MASK}"       \
                --dpo_margin          "${DPO_MARGIN}"        \
                --dpo_topk_pairs      "${DPO_TOPK_PAIRS}"    \
                --use_soft_dpo        "${USE_SOFT_DPO}"      \
                --mask_ratio          "${MASK_RATIO}"        \
                --mask_fill           "${MASK_FILL}"         \
                --dpo_update_mode     "explicit"             \
                --best_select_metric  "r1"                   \
                --use_llm_verify                             \
                --verbose 1                                  \
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
