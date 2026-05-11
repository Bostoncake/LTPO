#!/bin/bash
# run_ltpo_vl_dpo_perds_grid_dev.sh
#
# Sweep DPO-specific hyperparameters for the Contrastive Latent-DPO method,
# with LTPO-side parameters FIXED PER-DATASET to the best values from the
# contrast-optclip-bestr1 grid (see
#   scripts/find_best_grid.py output/ltpo_dmlr_contrastive_grid_dev/0426_contrast_optclip_bestr1
# ).
#
# Layout produced (compatible with scripts/find_best_grid.py):
#   <root>/<dpo_tag>/<model>-<dataset>-tokens..-...dmlr/results.log
# i.e. each <dpo_tag> directory contains all 7 datasets, but each dataset
# uses its own per-dataset best LTPO fingerprint internally.
#
# Sweeps: dpo_beta x lambda_mask x mask_ratio   (12 configs)
# Fixed (DPO):  num_candidates=4, alpha=1.0, margin=0.0, topk_pairs=0,
#               use_soft_dpo=True, mask_fill=zero,
#               update_mode=explicit, best_select_metric=r1
# Fixed (LTPO): per-dataset best from 0426_contrast_optclip_bestr1
#               sigma_decay=0.95, top_k=10
#
# Usage:
#   N_GPUS=8 bash scripts/run_ltpo_vl_dpo_perds_grid_dev.sh

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=${N_GPUS:-8}
MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}

# ---- LTPO best per dataset (tokens steps sigma lr) ----
declare -A LTPO_HP
LTPO_HP[math_vista_dev]="4 10 5.0 5e-2"
LTPO_HP[math_vision_dev]="2 15 25.0 1e-2"
LTPO_HP[mm_math_dev]="4 10 5.0 5e-2"
LTPO_HP[hallusion_dev]="4 10 25.0 1e-2"
LTPO_HP[mmvp_dev]="4 15 5.0 5e-3"
LTPO_HP[mmstar_dev]="4 15 25.0 1e-2"
LTPO_HP[scienceqa_dev]="4 15 5.0 1e-2"

SIGMA_DECAY=0.95
TOP_K=10

# ---- DPO grid (swept) ----
BETA_LIST=(0.1 0.5 1.0)
LAMBDA_MASK_LIST=(0.5 1.0)

# ---- DPO config (fixed) ----
TT_OPT_METHOD=contrastive_dpo
DPO_NUM_CANDIDATES=4
DPO_ALPHA=1.0
DPO_MARGIN=0.0
DPO_TOPK_PAIRS=0
USE_SOFT_DPO=True
MASK_FILL=attn_zero
MASK_RATIO_LIST=1.0
DPO_UPDATE_MODE=explicit
BEST_SELECT_METRIC=r1

DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

root_output=./output/ltpo_dmlr_dpo_grid_dev/0501_dpo_perds_sweep
mkdir -p "${root_output}"

# Build job list: dataset is OUTER so all DPO configs share LTPO_HP[dataset]
jobs=()
for beta in "${BETA_LIST[@]}"; do
    for lam in "${LAMBDA_MASK_LIST[@]}"; do
        for mr in "${MASK_RATIO_LIST[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                jobs+=("${beta} ${lam} ${mr} ${dataset}")
            done
        done
    done
done
total=${#jobs[@]}

echo "========================================================"
echo "Contrastive Latent-DPO  Per-dataset LTPO  +  DPO grid"
echo "    LTPO (per-dataset best, fixed):"
for d in "${DATASETS[@]}"; do
    echo "        ${d}: tokens steps sigma lr = ${LTPO_HP[$d]}"
done
echo "        sigma_decay=${SIGMA_DECAY}  top_k=${TOP_K}  (fixed)"
echo "    DPO sweep:"
echo "        beta in {${BETA_LIST[*]}}"
echo "        lambda_mask in {${LAMBDA_MASK_LIST[*]}}"
echo "        mask_ratio in {${MASK_RATIO_LIST[*]}}"
echo "    DPO fixed: B=${DPO_NUM_CANDIDATES}  alpha=${DPO_ALPHA}  margin=${DPO_MARGIN}"
echo "        topk_pairs=${DPO_TOPK_PAIRS}  soft=${USE_SOFT_DPO}  fill=${MASK_FILL}"
echo "        update_mode=${DPO_UPDATE_MODE}  best_select=${BEST_SELECT_METRIC}"
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
            read -r beta lam mr dataset <<< "${jobs[$job_idx]}"
            read -r tokens steps sigma lr <<< "${LTPO_HP[$dataset]}"

            # DPO-only tag (same set of tags shared across all datasets)
            tag="beta${beta}_lambdaMask${lam}_maskRatio${mr}_B${DPO_NUM_CANDIDATES}_${MASK_FILL}_${DPO_UPDATE_MODE}_sel${BEST_SELECT_METRIC}"
            out_dir="${root_output}/${tag}"
            mkdir -p "${out_dir}"
            log="${out_dir}/${dataset}.log"

            echo "[$(date +%T)] GPU ${g}  <-  ${tag}  |  ${dataset}  (LTPO: ${LTPO_HP[$dataset]})"

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
                --dpo_beta            "${beta}"              \
                --dpo_alpha           "${DPO_ALPHA}"         \
                --lambda_mask         "${lam}"               \
                --dpo_margin          "${DPO_MARGIN}"        \
                --dpo_topk_pairs      "${DPO_TOPK_PAIRS}"    \
                --use_soft_dpo        "${USE_SOFT_DPO}"      \
                --mask_ratio          "${mr}"                \
                --mask_fill           "${MASK_FILL}"         \
                --dpo_update_mode     "${DPO_UPDATE_MODE}"   \
                --best_select_metric  "${BEST_SELECT_METRIC}" \
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
echo "Tip: analyse with"
echo "    python scripts/find_best_grid.py ${root_output}"
echo "========================================================"

for cfg_dir in "${root_output}"/beta*; do
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
