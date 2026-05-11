#!/bin/bash
# run_workspace_mask_reward_dev_grid.sh — Mask-Reward Workspace sweep on 7 dev sets.
#
# For each image, during LTPO optimisation we do NOT interact with image slots.
# After optimisation finishes, every image slot is scored by masking its raw
# image tokens and observing the reward drop; the most-important slot is
# added (summed) to every latent thought token before the final generation
# pass.  See visual_workspace_mask_reward.py / ltpo_vl_workspace_mask_reward.py.
#
# LTPO hyperparameters are fixed to the per-dataset best values found by
# scripts/find_best_ltpo_dmlr.py (sigma_decay=0.95, top_k=10 across all).
#
# Mask-Reward grid:
#   K (num_workspace_slots) ∈ {8, 16}
#   inject_scale            ∈ {1.0, 0.5}
#
# Total: 2 × 2 = 4 configs × 7 datasets = 28 jobs.
#
# Usage:
#   N_GPUS=8 bash scripts/run_workspace_mask_reward_dev_grid.sh
#
# Overridable env vars:
#   N_GPUS    number of GPUs to use (default 8)
#   MODEL     path to model checkpoint

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=${N_GPUS:-8}
MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}

# ── Per-dataset best LTPO hyperparameters (from find_best_ltpo_dmlr.py) ──────
declare -A LTPO_TOKENS LTPO_STEPS LTPO_SIGMA LTPO_LR

LTPO_TOKENS[mmvp_dev]=4;        LTPO_STEPS[mmvp_dev]=15;  LTPO_SIGMA[mmvp_dev]=5.0;   LTPO_LR[mmvp_dev]=0.005
LTPO_TOKENS[mmstar_dev]=4;      LTPO_STEPS[mmstar_dev]=15; LTPO_SIGMA[mmstar_dev]=25.0; LTPO_LR[mmstar_dev]=0.01
LTPO_TOKENS[mm_math_dev]=4;     LTPO_STEPS[mm_math_dev]=10; LTPO_SIGMA[mm_math_dev]=5.0;  LTPO_LR[mm_math_dev]=0.05
LTPO_TOKENS[math_vista_dev]=2;  LTPO_STEPS[math_vista_dev]=15; LTPO_SIGMA[math_vista_dev]=5.0;  LTPO_LR[math_vista_dev]=0.05
LTPO_TOKENS[math_vision_dev]=4; LTPO_STEPS[math_vision_dev]=10; LTPO_SIGMA[math_vision_dev]=5.0;  LTPO_LR[math_vision_dev]=0.005
LTPO_TOKENS[hallusion_dev]=4;   LTPO_STEPS[hallusion_dev]=15; LTPO_SIGMA[hallusion_dev]=25.0; LTPO_LR[hallusion_dev]=0.005
LTPO_TOKENS[scienceqa_dev]=2;   LTPO_STEPS[scienceqa_dev]=10; LTPO_SIGMA[scienceqa_dev]=5.0;   LTPO_LR[scienceqa_dev]=0.05

# ── Mask-Reward grid ─────────────────────────────────────────────────────────
# K_VALUES=(8 16)
# INJECT_SCALES=(1.0 0.5)
K_VALUES=(4)
INJECT_SCALES=(0.0)
DATASETS=(mmvp_dev mmstar_dev mm_math_dev math_vista_dev math_vision_dev hallusion_dev scienceqa_dev)

root_output=./output/workspace_mask_reward_dev_grid
mkdir -p "${root_output}"

# Build flat job list: "K inject_scale dataset"
jobs=()
for K in "${K_VALUES[@]}"; do
    for s in "${INJECT_SCALES[@]}"; do
        for dataset in "${DATASETS[@]}"; do
            jobs+=("${K} ${s} ${dataset}")
        done
    done
done
total=${#jobs[@]}

echo "════════════════════════════════════════════════════════════════"
echo "Mask-Reward Workspace Grid Search on dev sets"
echo "    K ∈ {${K_VALUES[*]}}   inject_scale ∈ {${INJECT_SCALES[*]}}"
echo "    LTPO HPs: per-dataset best (sigma_decay=0.95, top_k=10 fixed)"
echo "    Datasets: ${#DATASETS[@]}   Configs: $(( total / ${#DATASETS[@]} ))"
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
            read -r K s dataset <<< "${jobs[$job_idx]}"

            tokens=${LTPO_TOKENS[$dataset]}
            steps=${LTPO_STEPS[$dataset]}
            sigma=${LTPO_SIGMA[$dataset]}
            lr=${LTPO_LR[$dataset]}

            out_dir="${root_output}/K${K}_s${s}"
            mkdir -p "${out_dir}"
            log="${root_output}/K${K}_s${s}_${dataset}.log"

            echo "[$(date +%T)] GPU ${g}  ←  K=${K} inject_scale=${s}  |  ${dataset}  (tokens=${tokens} steps=${steps} sigma=${sigma} lr=${lr})"

            CUDA_VISIBLE_DEVICES=${g} python main_vl_workspace_mask_reward.py \
                --dataset                    "${dataset}"    \
                --data_root                  mllm_data       \
                --image_root                 .               \
                --model_name_or_path         "${MODEL}"      \
                --output_dir                 "${out_dir}"    \
                --device                     cuda            \
                --seed                       42              \
                --max_new_tokens             2048            \
                --min_pixels                 128             \
                --max_pixels                 256             \
                --num_thought_tokens         "${tokens}"     \
                --sigma                      "${sigma}"      \
                --sigma_decay                0.95            \
                --lr                         "${lr}"         \
                --max_num_steps              "${steps}"      \
                --top_k                      10              \
                --use_mask_reward_workspace                  \
                --num_workspace_slots        "${K}"          \
                --inject_scale               "${s}"          \
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
echo "════════════════════════════════════════════════════════════════"
echo "All ${total} jobs done. Results in ${root_output}"
echo "════════════════════════════════════════════════════════════════"
