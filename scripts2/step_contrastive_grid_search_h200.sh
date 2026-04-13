#!/bin/bash
# ============================================================
# Contrastive Visual Reward 网格搜索 (v2 组合奖励)
# 运行环境: H200 (无网络)
#
# 搜索空间 (第一层):
#   beta (contrastive_weight) ∈ {0.0, 0.1, 0.3, 0.5, 1.0}
#   inject_mode              ∈ {add, prepend}
#   r (num_route_slots)      ∈ {2, 4}
#   共 5×2×2 = 20 种配置 × 7 数据集 = 140 个 job
#
# 固定参数:
#   tokens=2  steps=15  sigma=25.0  lr=0.01  K=8  sigma_decay=0.95  top_k=10
#
# 用法:
#   N_GPUS=7 bash scripts2/step_contrastive_grid_search_h200.sh
#   MODEL_DIR=/path/to/models N_GPUS=4 bash scripts2/step_contrastive_grid_search_h200.sh
# ============================================================
set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

# ---- 配置区 ----
N_GPUS=${N_GPUS:-7}
MODEL_DIR=${MODEL_DIR:-/inspire/hdd/project/qproject-fundationmodel/xiashijie-240108120112/qsh/models}
MODEL="Qwen2.5-VL-3B-Instruct"

DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

# 搜索空间
BETA_LIST=(0.0 0.1 0.3 0.5 1.0)
INJECT_LIST=(add prepend)
R_LIST=(2 4)

# 固定参数
K=8
ROUTE_MODE=avg
TOKENS=2
SIGMA=25.0
SIGMA_DECAY=0.95
LR=0.01
STEPS=15
TOP_K=10

root_output=./output/contrastive_grid_dev
mkdir -p "${root_output}"

# 构建 job 列表
declare -a JOBS
for beta in "${BETA_LIST[@]}"; do
    for inject in "${INJECT_LIST[@]}"; do
        for r in "${R_LIST[@]}"; do
            for dataset in "${DATASETS[@]}"; do
                JOBS+=("${beta} ${inject} ${r} ${dataset}")
            done
        done
    done
done
total=${#JOBS[@]}
n_configs=$(( total / ${#DATASETS[@]} ))

echo "════════════════════════════════════════════════════════"
echo "Contrastive Visual Reward 网格搜索 (H200)"
echo "    模型: ${MODEL_DIR}/${MODEL}"
echo "    beta    ∈ {${BETA_LIST[*]}}"
echo "    inject  ∈ {${INJECT_LIST[*]}}"
echo "    r       ∈ {${R_LIST[*]}}"
echo "    固定: K=${K}  tokens=${TOKENS}  sigma=${SIGMA}  lr=${LR}  steps=${STEPS}"
echo "    数据集: ${#DATASETS[@]} 个 dev 集"
echo "    配置数: ${n_configs}   Job 总数: ${total}"
echo "    GPU 数: ${N_GPUS}"
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
            read -r beta inject r dataset <<< "${JOBS[$job_idx]}"

            tag="beta${beta}_${inject}_r${r}"
            out_dir="${root_output}/${tag}"
            mkdir -p "${out_dir}"
            log="${out_dir}/${dataset}.log"

            echo "[$(date +%T)] GPU ${g}  <-  ${tag}  |  ${dataset}"

            CUDA_VISIBLE_DEVICES=$g python main_vl_workspace_contrastive_v2.py \
                --dataset           "${dataset}"                            \
                --data_root         mllm_data                               \
                --image_root        mllm_data                               \
                --model_name_or_path "${MODEL_DIR}/${MODEL}"                \
                --output_dir        "${out_dir}"                            \
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
                --contrastive_weight "${beta}"                              \
                --num_workspace_slots "${K}"                                \
                --num_route_slots   "${r}"                                  \
                --workspace_inject_mode "${inject}"                         \
                --workspace_route_mode  "${ROUTE_MODE}"                     \
                --verbose 1                                                 \
                > "${log}" 2>&1 &

            gpu_pids[$g]=$!
            job_idx=$((job_idx+1))
        fi
    done
    sleep 2
done

wait
echo ""
echo "════════════════════════════════════════════════════════"
echo "全部 ${total} 个 job 完成。结果在 ${root_output}/"
echo ""
echo "Rule-based 各配置平均准确率 (降序排列):"
echo "════════════════════════════════════════════════════════"

for cfg_dir in "${root_output}"/beta*; do
    [ -d "$cfg_dir" ] || continue
    cfg=$(basename "$cfg_dir")
    total_acc=0
    count=0
    while IFS= read -r log; do
        acc=$(grep -oP 'accuracy=\K[0-9.]+' "$log" | tail -1)
        [ -n "$acc" ] && total_acc=$(awk "BEGIN{print $total_acc + $acc}") && count=$((count+1))
    done < <(find "$cfg_dir" -name "results.log" -path "*/results.log")
    if [ "$count" -gt 0 ]; then
        mean=$(awk "BEGIN{printf \"%.4f\", $total_acc / $count}")
        echo "mean_acc=${mean}  n=${count}  | ${cfg}"
    fi
done | sort -t'=' -k2 -rn
