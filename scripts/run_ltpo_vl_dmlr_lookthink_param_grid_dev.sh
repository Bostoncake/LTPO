#!/bin/bash
# run_ltpo_vl_dmlr_lookthink_param_grid_dev.sh
# Stage-2 LOOK/THINK grid: per-dataset best LTPO params from stage-1, then
# sweep (look_threshold, look_alpha) on the 7 dev sets.
#
# Source of per-dataset LTPO params (highest acc per dataset):
#   python scripts/find_best_grid.py \
#       output/ltpo_dmlr_contrastive_grid_dev/0502_lookthink_optclip_bestr1
# (cross-dataset summary, top-1 per dataset).
#
# Reward setup (unchanged from stage-1):
#   opt_reward  = clip   (optimise on r1 + lambda * max(r1 - r2, 0))
#   best_reward = r1     (pick best latent by r1 alone)
#   per-dataset gap_lambda inherited from the contrast-clipping grid.
#
# Sweeps: look_threshold x look_alpha
# Fixed (per dataset): tokens, steps, sigma, lr  (from stage-1 best)
# Fixed (global):      sigma_decay=0.95, top_k=10
#
# Usage:
#   N_GPUS=8 bash scripts/run_ltpo_vl_dmlr_lookthink_param_grid_dev.sh

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=${N_GPUS:-8}
MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}

# ---- Globally-fixed LTPO params ----
SIGMA_DECAY=0.95
TOP_K=10

# ---- Per-dataset best LTPO params (from stage-1 cross-dataset summary) ----
declare -A TOKENS_PD STEPS_PD SIGMA_PD LR_PD

# math_vista_dev   : tokens2  steps10  sigma25.0  lr5e-2
TOKENS_PD[math_vista_dev]=2;   STEPS_PD[math_vista_dev]=10;  SIGMA_PD[math_vista_dev]=25.0;  LR_PD[math_vista_dev]=5e-2
# math_vision_dev  : tokens4  steps15  sigma5.0   lr1e-2
TOKENS_PD[math_vision_dev]=4;  STEPS_PD[math_vision_dev]=15; SIGMA_PD[math_vision_dev]=5.0;  LR_PD[math_vision_dev]=1e-2
# mm_math_dev      : tokens4  steps10  sigma25.0  lr5e-2
TOKENS_PD[mm_math_dev]=4;      STEPS_PD[mm_math_dev]=10;     SIGMA_PD[mm_math_dev]=25.0;     LR_PD[mm_math_dev]=5e-2
# hallusion_dev    : tokens4  steps10  sigma25.0  lr5e-2
TOKENS_PD[hallusion_dev]=4;    STEPS_PD[hallusion_dev]=10;   SIGMA_PD[hallusion_dev]=25.0;   LR_PD[hallusion_dev]=5e-2
# mmvp_dev         : tokens4  steps15  sigma5.0   lr5e-3
TOKENS_PD[mmvp_dev]=4;         STEPS_PD[mmvp_dev]=15;        SIGMA_PD[mmvp_dev]=5.0;         LR_PD[mmvp_dev]=5e-3
# mmstar_dev       : tokens2  steps10  sigma5.0   lr5e-3
TOKENS_PD[mmstar_dev]=2;       STEPS_PD[mmstar_dev]=10;      SIGMA_PD[mmstar_dev]=5.0;       LR_PD[mmstar_dev]=5e-3
# scienceqa_dev    : tokens4  steps15  sigma5.0   lr1e-2
TOKENS_PD[scienceqa_dev]=4;    STEPS_PD[scienceqa_dev]=15;   SIGMA_PD[scienceqa_dev]=5.0;    LR_PD[scienceqa_dev]=1e-2

# ---- LOOK/THINK sweep space ----
# Threshold on delta = r1 - r2; below this we LOOK, otherwise we THINK.
# Spans a slightly negative value (almost-never-LOOK), the stage-1 default
# 0.01, and a few larger values that trigger LOOK more aggressively.
LOOK_THRESHOLD_LIST=(0.01 0.05 0.1)
# Visual injection strength.
LOOK_ALPHA_LIST=(0.05 0.1 0.2 0.5)

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

root_output=./output/ltpo_dmlr_contrastive_grid_dev/0505_lookthink_param_grid
mkdir -p "${root_output}"

jobs=()
for lt in "${LOOK_THRESHOLD_LIST[@]}"; do
    for la in "${LOOK_ALPHA_LIST[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            jobs+=("${lt} ${la} ${dataset}")
        done
    done
done
total=${#jobs[@]}

echo "========================================================"
echo "LTPO-DMLR LOOK/THINK Param Grid Search on dev sets"
echo "    objective: opt = r1 + lambda*clip(r1-r2, 0) ; best = r1"
echo "    fixed (global): decay=${SIGMA_DECAY}  topk=${TOP_K}"
echo "    per-dataset best LTPO params (from stage-1 top-1):"
for d in "${DATASETS[@]}"; do
    echo "        ${d}: tokens=${TOKENS_PD[$d]} steps=${STEPS_PD[$d]} sigma=${SIGMA_PD[$d]} lr=${LR_PD[$d]} lambda=${GAP_LAMBDA[$d]}"
done
echo "    look_threshold in {${LOOK_THRESHOLD_LIST[*]}}"
echo "    look_alpha     in {${LOOK_ALPHA_LIST[*]}}"
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
            read -r lt la dataset <<< "${jobs[$job_idx]}"
            lam=${GAP_LAMBDA[$dataset]}
            tokens=${TOKENS_PD[$dataset]}
            steps=${STEPS_PD[$dataset]}
            sigma=${SIGMA_PD[$dataset]}
            lr=${LR_PD[$dataset]}

            # Per-dataset LTPO params -> separate config dir per (lt, la).
            # The output_dir already encodes the dataset (dataset name appears
            # under output_dir as a sub-directory written by main_*.py), so
            # collisions between datasets within the same (lt, la) tag are
            # avoided by their differing per-dataset hparams in the dataset log.
            tag="lt${lt}_la${la}"
            out_dir="${root_output}/${tag}/${dataset}_tokens${tokens}_steps${steps}_sigma${sigma}_lr${lr}"
            mkdir -p "${out_dir}"
            log="${out_dir}/${dataset}.log"

            echo "[$(date +%T)] GPU ${g}  <-  ${tag}  |  ${dataset}  (tokens=${tokens} steps=${steps} sigma=${sigma} lr=${lr} lambda=${lam})"

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
                --look_threshold     "${lt}"        \
                --look_alpha         "${la}"        \
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
echo "Results per (lt, la) config (mean accuracy across datasets):"
echo "========================================================"

for cfg_dir in "${root_output}"/lt*; do
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
