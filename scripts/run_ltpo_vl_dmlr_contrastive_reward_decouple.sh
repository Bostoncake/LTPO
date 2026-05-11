#!/bin/bash
# run_ltpo_vl_dmlr_contrastive_reward_decouple.sh
# Grid over DECOUPLED (opt_reward, best_reward) pairs for the contrastive LTPO.
#
#   opt_reward  ∈ {r1, diff, clip}   (reward used to optimise latent thoughts)
#   best_reward ∈ {r1, diff, clip}   (reward used to pick best latent across steps)
#     r1   : confidence with full visual attention
#     diff : r1 - r2
#     clip : r1 + gap_lambda * max(r1 - r2, 0)
#
# LTPO hyperparameters are fixed to the per-dataset best values from
# scripts/run_workspace_fixed_dev_grid.sh (sigma_decay=0.95, top_k=10 fixed).
#
# Total: 3 × 3 × 7 = 63 jobs.
#
# Usage:
#   N_GPUS=8 bash scripts/run_ltpo_vl_dmlr_contrastive_reward_decouple.sh

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=${N_GPUS:-8}
MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}
GAP_LAMBDA=${GAP_LAMBDA:-1.0}

# ── Per-dataset best LTPO hyperparameters (same as workspace_fixed_dev_grid) ──
declare -A LTPO_TOKENS LTPO_STEPS LTPO_SIGMA LTPO_LR

LTPO_TOKENS[mmvp_dev]=4;        LTPO_STEPS[mmvp_dev]=15;  LTPO_SIGMA[mmvp_dev]=5.0;   LTPO_LR[mmvp_dev]=0.005
LTPO_TOKENS[mmstar_dev]=4;      LTPO_STEPS[mmstar_dev]=15; LTPO_SIGMA[mmstar_dev]=25.0; LTPO_LR[mmstar_dev]=0.01
LTPO_TOKENS[mm_math_dev]=4;     LTPO_STEPS[mm_math_dev]=10; LTPO_SIGMA[mm_math_dev]=5.0;  LTPO_LR[mm_math_dev]=0.05
LTPO_TOKENS[math_vista_dev]=2;  LTPO_STEPS[math_vista_dev]=15; LTPO_SIGMA[math_vista_dev]=5.0;  LTPO_LR[math_vista_dev]=0.05
LTPO_TOKENS[math_vision_dev]=4; LTPO_STEPS[math_vision_dev]=10; LTPO_SIGMA[math_vision_dev]=5.0;  LTPO_LR[math_vision_dev]=0.005
LTPO_TOKENS[hallusion_dev]=4;   LTPO_STEPS[hallusion_dev]=15; LTPO_SIGMA[hallusion_dev]=25.0; LTPO_LR[hallusion_dev]=0.005
LTPO_TOKENS[scienceqa_dev]=2;   LTPO_STEPS[scienceqa_dev]=10; LTPO_SIGMA[scienceqa_dev]=5.0;   LTPO_LR[scienceqa_dev]=0.05

OPT_REWARDS=(r1 diff clip)
BEST_REWARDS=(r1 diff clip)
DATASETS=(mmvp_dev mmstar_dev mm_math_dev math_vista_dev math_vision_dev hallusion_dev scienceqa_dev)

root_output=./output/ltpo_dmlr_contrastive_reward_decouple
mkdir -p "${root_output}"

jobs=()
for opt_r in "${OPT_REWARDS[@]}"; do
    for best_r in "${BEST_REWARDS[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            jobs+=("${opt_r} ${best_r} ${dataset}")
        done
    done
done
total=${#jobs[@]}

echo "════════════════════════════════════════════════════════════════"
echo "LTPO-DMLR contrastive: DECOUPLED (opt_reward, best_reward) grid"
echo "    opt_reward  ∈ {${OPT_REWARDS[*]}}"
echo "    best_reward ∈ {${BEST_REWARDS[*]}}"
echo "    gap_lambda  = ${GAP_LAMBDA}  (only affects the 'clip' variant)"
echo "    LTPO HPs: per-dataset best (sigma_decay=0.95, top_k=10 fixed)"
echo "    Datasets: ${#DATASETS[@]}   Reward pairs: 9"
echo "    Total jobs: ${total}   GPUs: ${N_GPUS}"
echo "════════════════════════════════════════════════════════════════"

declare -a gpu_pids
for ((g=0; g<N_GPUS; g++)); do gpu_pids[$g]=-1; done

job_idx=0
while [ $job_idx -lt $total ]; do
    for ((g=0; g<N_GPUS; g++)); do
        [ $job_idx -ge $total ] && break
        pid=${gpu_pids[$g]}
        if [ "$pid" -eq -1 ] || ! kill -0 "$pid" 2>/dev/null; then
            read -r opt_r best_r dataset <<< "${jobs[$job_idx]}"

            tokens=${LTPO_TOKENS[$dataset]}
            steps=${LTPO_STEPS[$dataset]}
            sigma=${LTPO_SIGMA[$dataset]}
            lr=${LTPO_LR[$dataset]}

            tag="opt_${opt_r}__best_${best_r}"
            out_dir="${root_output}/${tag}"
            mkdir -p "${out_dir}"
            log="${out_dir}/${dataset}.log"

            echo "[$(date +%T)] GPU ${g}  ←  ${tag}  |  ${dataset}  (tokens=${tokens} steps=${steps} sigma=${sigma} lr=${lr})"

            CUDA_VISIBLE_DEVICES=${g} python main_vl_dmlr_contrastive.py \
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
                --sigma_decay        0.95           \
                --lr                 "${lr}"        \
                --max_num_steps      "${steps}"     \
                --top_k              10             \
                --gap_lambda         "${GAP_LAMBDA}" \
                --decouple_reward                   \
                --opt_reward         "${opt_r}"     \
                --best_reward        "${best_r}"    \
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
echo "════════════════════════════════════════════════════════════════"
echo "All ${total} jobs done. Results in ${root_output}"
echo "════════════════════════════════════════════════════════════════"

for cfg_dir in "${root_output}"/opt_*; do
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
