#!/bin/bash
# run_workspace_fixed_dev_grid.sh — Workspace hyperparameter grid search on all 7 dev sets.
#
# LTPO hyperparameters are fixed to the per-dataset best values found by
# scripts/find_best_ltpo_dmlr.py (sigma_decay=0.95, top_k=10 across all).
#
# Workspace grid:
#   K (num_workspace_slots) ∈ {8, 16}
#   r (num_route_slots)     ∈ {2, 4}
#   inject_mode             ∈ {prepend, add}
#
# Total: 2×2×2 = 8 configs × 7 datasets = 56 jobs.
# Est. wall time (8 GPUs, ~5 min/job): ≈ 35 min.
#
# Usage:
#   N_GPUS=8 bash scripts/run_workspace_fixed_dev_grid.sh
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
# sigma_decay=0.95 and top_k=10 are the same for all datasets.
declare -A LTPO_TOKENS LTPO_STEPS LTPO_SIGMA LTPO_LR

LTPO_TOKENS[mmvp_dev]=4;        LTPO_STEPS[mmvp_dev]=15;  LTPO_SIGMA[mmvp_dev]=5.0;   LTPO_LR[mmvp_dev]=0.005
LTPO_TOKENS[mmstar_dev]=4;      LTPO_STEPS[mmstar_dev]=15; LTPO_SIGMA[mmstar_dev]=25.0; LTPO_LR[mmstar_dev]=0.01
LTPO_TOKENS[mm_math_dev]=4;     LTPO_STEPS[mm_math_dev]=10; LTPO_SIGMA[mm_math_dev]=5.0;  LTPO_LR[mm_math_dev]=0.05
LTPO_TOKENS[math_vista_dev]=2;  LTPO_STEPS[math_vista_dev]=15; LTPO_SIGMA[math_vista_dev]=5.0;  LTPO_LR[math_vista_dev]=0.05
LTPO_TOKENS[math_vision_dev]=4; LTPO_STEPS[math_vision_dev]=10; LTPO_SIGMA[math_vision_dev]=5.0;  LTPO_LR[math_vision_dev]=0.005
LTPO_TOKENS[hallusion_dev]=4;   LTPO_STEPS[hallusion_dev]=15; LTPO_SIGMA[hallusion_dev]=25.0; LTPO_LR[hallusion_dev]=0.005
LTPO_TOKENS[scienceqa_dev]=2;   LTPO_STEPS[scienceqa_dev]=10; LTPO_SIGMA[scienceqa_dev]=5.0;   LTPO_LR[scienceqa_dev]=0.05

# ── Workspace grid ────────────────────────────────────────────────────────────
K_VALUES=(8 16)
R_VALUES=(2 4)
INJECT_MODES=(prepend add)
DATASETS=(mmvp_dev mmstar_dev mm_math_dev math_vista_dev math_vision_dev hallusion_dev scienceqa_dev)

root_output=./output/workspace_fixed_dev_grid
mkdir -p "${root_output}"

# Build flat job list: "K r inject_mode dataset"
jobs=()
for K in "${K_VALUES[@]}"; do
    for r in "${R_VALUES[@]}"; do
        for inject_mode in "${INJECT_MODES[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                jobs+=("${K} ${r} ${inject_mode} ${dataset}")
            done
        done
    done
done
total=${#jobs[@]}

echo "════════════════════════════════════════════════════════════════"
echo "Workspace Fixed Grid Search on dev sets"
echo "    K ∈ {${K_VALUES[*]}}  r ∈ {${R_VALUES[*]}}  mode ∈ {${INJECT_MODES[*]}}"
echo "    LTPO HPs: per-dataset best (sigma_decay=0.95, top_k=10 fixed)"
echo "    Datasets: ${#DATASETS[@]}   Workspace configs: $(( total / ${#DATASETS[@]} ))"
echo "    Total jobs: ${total}   GPUs: ${N_GPUS}"
echo "    Est. wall time: $(( (total + N_GPUS - 1) / N_GPUS * 5 )) min"
echo "════════════════════════════════════════════════════════════════"

# ── Dynamic GPU scheduling ────────────────────────────────────────────────────
declare -a gpu_pids
for ((g=0; g<N_GPUS; g++)); do gpu_pids[$g]=-1; done

job_idx=0
while [ $job_idx -lt $total ]; do
    for ((g=0; g<N_GPUS; g++)); do
        [ $job_idx -ge $total ] && break
        pid=${gpu_pids[$g]}
        if [ "$pid" -eq -1 ] || ! kill -0 "$pid" 2>/dev/null; then
            read -r K r inject_mode dataset <<< "${jobs[$job_idx]}"

            tokens=${LTPO_TOKENS[$dataset]}
            steps=${LTPO_STEPS[$dataset]}
            sigma=${LTPO_SIGMA[$dataset]}
            lr=${LTPO_LR[$dataset]}

            out_dir="${root_output}/K${K}_r${r}_${inject_mode}"
            mkdir -p "${out_dir}"
            log="${root_output}/K${K}_r${r}_${inject_mode}_${dataset}.log"

            echo "[$(date +%T)] GPU ${g}  ←  K=${K} r=${r} mode=${inject_mode}  |  ${dataset}  (tokens=${tokens} steps=${steps} sigma=${sigma} lr=${lr})"

            CUDA_VISIBLE_DEVICES=${g} python main_vl_workspace_fixed.py \
                --dataset             "${dataset}"    \
                --data_root           mllm_data       \
                --image_root          .               \
                --model_name_or_path  "${MODEL}"      \
                --output_dir          "${out_dir}"    \
                --device              cuda            \
                --seed                42              \
                --max_new_tokens      2048            \
                --min_pixels          128             \
                --max_pixels          256             \
                --num_thought_tokens  "${tokens}"     \
                --sigma               "${sigma}"      \
                --sigma_decay         0.95            \
                --lr                  "${lr}"         \
                --max_num_steps       "${steps}"      \
                --top_k               10              \
                --use_workspace                       \
                --num_workspace_slots "${K}"          \
                --num_route_slots     "${r}"          \
                --workspace_inject_mode "${inject_mode}" \
                --use_llm_verify                      \
                --verbose 1                           \
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
