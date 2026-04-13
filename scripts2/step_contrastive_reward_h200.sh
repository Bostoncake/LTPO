#!/bin/bash
# ============================================================
# Contrastive Visual Reward 评测: Qwen2.5-VL-3B × 7 个 dev 数据集
# 运行环境: H200 (无网络)
#
# 用法:
#   N_GPUS=7 bash scripts2/step_contrastive_reward_h200.sh
#   MODEL_DIR=/path/to/models N_GPUS=4 bash scripts2/step_contrastive_reward_h200.sh
# ============================================================
set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

# ---- 配置区 (按需修改) ----
N_GPUS=${N_GPUS:-7}
MODEL_DIR=${MODEL_DIR:-/inspire/hdd/project/qproject-fundationmodel/xiashijie-240108120112/qsh/models}
MODEL="Qwen2.5-VL-3B-Instruct"

DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

# Workspace + 对比奖励超参
NUM_WORKSPACE_SLOTS=8
NUM_ROUTE_SLOTS=2
INJECT_MODE=add
ROUTE_MODE=avg

# LTPO 超参
TOKENS=2
SIGMA=25.0
SIGMA_DECAY=0.95
LR=0.01
STEPS=15
TOP_K=10

output=./output/contrastive_reward_dev
mkdir -p "${output}"

total=${#DATASETS[@]}

echo "════════════════════════════════════════════════════════"
echo "Contrastive Visual Reward 评测 (H200, 离线)"
echo "    模型: ${MODEL_DIR}/${MODEL}"
echo "    Workspace: K=${NUM_WORKSPACE_SLOTS}  r=${NUM_ROUTE_SLOTS}  inject=${INJECT_MODE}  route=${ROUTE_MODE}"
echo "    LTPO: tokens=${TOKENS}  sigma=${SIGMA}  lr=${LR}  steps=${STEPS}"
echo "    数据集: ${total} 个 dev 集"
echo "    GPU 数: ${N_GPUS}"
echo "    输出: ${output}"
echo "════════════════════════════════════════════════════════"

# 动态 GPU 池调度
declare -a gpu_pids
for ((g=0; g<N_GPUS; g++)); do gpu_pids[$g]=-1; done

job_idx=0
while [ $job_idx -lt $total ]; do
    for ((g=0; g<N_GPUS; g++)); do
        [ $job_idx -ge $total ] && break
        pid=${gpu_pids[$g]}
        if [ "$pid" -eq -1 ] || ! kill -0 "$pid" 2>/dev/null; then
            dataset="${DATASETS[$job_idx]}"

            echo "[$(date +%T)] GPU ${g}  <-  ${dataset}"

            CUDA_VISIBLE_DEVICES=$g python main_vl_workspace_contrastive.py \
                --dataset           "${dataset}"                            \
                --data_root         mllm_data                               \
                --image_root        mllm_data                               \
                --model_name_or_path "${MODEL_DIR}/${MODEL}"                \
                --output_dir        "${output}"                             \
                --device            cuda                                    \
                --seed              42                                      \
                --max_new_tokens    2048                                    \
                --min_pixels        128                                     \
                --max_pixels        256                                     \
                --num_thought_tokens "${TOKENS}"                            \
                --sigma             "${SIGMA}"                              \
                --sigma_decay       "${SIGMA_DECAY}"                        \
                --lr                "${LR}"                                 \
                --max_num_steps     "${STEPS}"                              \
                --top_k             "${TOP_K}"                              \
                --use_workspace                                             \
                --contrastive_visual_reward                                 \
                --num_workspace_slots "${NUM_WORKSPACE_SLOTS}"              \
                --num_route_slots   "${NUM_ROUTE_SLOTS}"                    \
                --workspace_inject_mode "${INJECT_MODE}"                    \
                --workspace_route_mode  "${ROUTE_MODE}"                     \
                --verbose 1                                                 \
                > "${output}/${dataset}.log" 2>&1 &

            gpu_pids[$g]=$!
            job_idx=$((job_idx+1))
        fi
    done
    sleep 2
done

wait
echo ""
echo "════════════════════════════════════════════════════════"
echo "全部 ${total} 个 job 完成。结果在 ${output}/"
echo ""
echo "各数据集准确率:"
echo "════════════════════════════════════════════════════════"
for dataset in "${DATASETS[@]}"; do
    result_dirs=("${output}"/${MODEL}*${dataset}*)
    for rd in "${result_dirs[@]}"; do
        [ -f "$rd/results.log" ] || continue
        acc=$(grep -oP 'accuracy=\K[0-9.]+' "$rd/results.log" | tail -1)
        [ -n "$acc" ] && echo "  ${dataset}: ${acc}"
    done
done
