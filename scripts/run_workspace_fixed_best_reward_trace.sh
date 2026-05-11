#!/bin/bash
# run_workspace_fixed_best_reward_trace.sh
#
# For every dev dataset, run main_vl_workspace_fixed.py with the per-dataset
# BEST visual-workspace hyperparameters (LTPO + workspace) identified by
# `python scripts/find_best_ltpo_dmlr.py --mode workspace \
#         --root output/workspace_fixed_dev_grid_avgroute`
# and turn on --save_reward_trace so the full per-step LTPO reward history
# (plus best-reward step index) is stored in logistics.pt for every example.
#
# Per-dataset best configs (route_mode=per_token, sigma_decay=0.95, top_k=10):
#   dataset          K   r  inject   tokens  steps  sigma   lr
#   mmvp_dev         8   4  prepend  4       15     5.0     0.005
#   mmstar_dev       16  4  prepend  4       15     25.0    0.01
#   mm_math_dev      16  4  prepend  4       10     5.0     0.05
#   math_vista_dev   8   4  add      2       15     5.0     0.05
#   math_vision_dev  16  4  prepend  4       10     5.0     0.005
#   hallusion_dev    8   4  prepend  4       15     25.0    0.005
#   scienceqa_dev    16  2  add      2       10     5.0     0.05
#
# Usage:
#   N_GPUS=8 bash scripts/run_workspace_fixed_best_reward_trace.sh
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

# ── Per-dataset best visual-workspace hyperparameters ────────────────────────
declare -A LTPO_TOKENS LTPO_STEPS LTPO_SIGMA LTPO_LR WS_K WS_R WS_INJECT

#                      tokens  steps  sigma   lr        K   r   inject
LTPO_TOKENS[mmvp_dev]=4;        LTPO_STEPS[mmvp_dev]=15;        LTPO_SIGMA[mmvp_dev]=5.0;        LTPO_LR[mmvp_dev]=0.005
WS_K[mmvp_dev]=8;               WS_R[mmvp_dev]=4;               WS_INJECT[mmvp_dev]=prepend

LTPO_TOKENS[mmstar_dev]=4;      LTPO_STEPS[mmstar_dev]=15;      LTPO_SIGMA[mmstar_dev]=25.0;     LTPO_LR[mmstar_dev]=0.01
WS_K[mmstar_dev]=16;            WS_R[mmstar_dev]=4;             WS_INJECT[mmstar_dev]=prepend

LTPO_TOKENS[mm_math_dev]=4;     LTPO_STEPS[mm_math_dev]=10;     LTPO_SIGMA[mm_math_dev]=5.0;     LTPO_LR[mm_math_dev]=0.05
WS_K[mm_math_dev]=16;           WS_R[mm_math_dev]=4;            WS_INJECT[mm_math_dev]=prepend

LTPO_TOKENS[math_vista_dev]=2;  LTPO_STEPS[math_vista_dev]=15;  LTPO_SIGMA[math_vista_dev]=5.0;  LTPO_LR[math_vista_dev]=0.05
WS_K[math_vista_dev]=8;         WS_R[math_vista_dev]=4;         WS_INJECT[math_vista_dev]=add

LTPO_TOKENS[math_vision_dev]=4; LTPO_STEPS[math_vision_dev]=10; LTPO_SIGMA[math_vision_dev]=5.0; LTPO_LR[math_vision_dev]=0.005
WS_K[math_vision_dev]=16;       WS_R[math_vision_dev]=4;        WS_INJECT[math_vision_dev]=prepend

LTPO_TOKENS[hallusion_dev]=4;   LTPO_STEPS[hallusion_dev]=15;   LTPO_SIGMA[hallusion_dev]=25.0;  LTPO_LR[hallusion_dev]=0.005
WS_K[hallusion_dev]=8;          WS_R[hallusion_dev]=4;          WS_INJECT[hallusion_dev]=prepend

LTPO_TOKENS[scienceqa_dev]=2;   LTPO_STEPS[scienceqa_dev]=10;   LTPO_SIGMA[scienceqa_dev]=5.0;   LTPO_LR[scienceqa_dev]=0.05
WS_K[scienceqa_dev]=16;         WS_R[scienceqa_dev]=2;          WS_INJECT[scienceqa_dev]=add

DATASETS=(mmvp_dev mmstar_dev mm_math_dev math_vista_dev math_vision_dev hallusion_dev scienceqa_dev)

root_output=./output/workspace_fixed_best_reward_trace
mkdir -p "${root_output}"

total=${#DATASETS[@]}

echo "════════════════════════════════════════════════════════════════"
echo "Workspace-Fixed BEST configs on dev sets  (reward trace ON)"
echo "    Datasets : ${total}   GPUs : ${N_GPUS}"
echo "    Output   : ${root_output}"
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
            dataset=${DATASETS[$job_idx]}

            tokens=${LTPO_TOKENS[$dataset]}
            steps=${LTPO_STEPS[$dataset]}
            sigma=${LTPO_SIGMA[$dataset]}
            lr=${LTPO_LR[$dataset]}
            K=${WS_K[$dataset]}
            r=${WS_R[$dataset]}
            inject_mode=${WS_INJECT[$dataset]}

            out_dir="${root_output}/${dataset}"
            mkdir -p "${out_dir}"
            log="${root_output}/${dataset}.log"

            echo "[$(date +%T)] GPU ${g}  ←  ${dataset}  (K=${K} r=${r} inject=${inject_mode}  tokens=${tokens} steps=${steps} sigma=${sigma} lr=${lr})"

            CUDA_VISIBLE_DEVICES=${g} python main_vl_workspace_fixed.py \
                --dataset               "${dataset}"    \
                --data_root             mllm_data       \
                --image_root            .               \
                --model_name_or_path    "${MODEL}"      \
                --output_dir            "${out_dir}"    \
                --device                cuda            \
                --seed                  42              \
                --max_new_tokens        2048            \
                --min_pixels            128             \
                --max_pixels            256             \
                --num_thought_tokens    "${tokens}"     \
                --sigma                 "${sigma}"      \
                --sigma_decay           0.95            \
                --lr                    "${lr}"         \
                --max_num_steps         "${steps}"      \
                --top_k                 10              \
                --use_workspace                         \
                --num_workspace_slots   "${K}"          \
                --num_route_slots       "${r}"          \
                --workspace_inject_mode "${inject_mode}" \
                --workspace_route_mode  per_token        \
                --use_llm_verify                        \
                --save_reward_trace                     \
                --verbose 1                             \
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
