#!/bin/bash
# ============================================================
# Step 3b-3models: 3 个模型 × 超参数网格搜索 (dev 集)
# 运行环境: H200 (无网络)
#
# 搜索空间:
#   models ∈ {Qwen2.5-VL-3B, Qwen3-VL-4B, Qwen3-VL-8B}
#   tokens ∈ {2, 4}   steps ∈ {10, 15}
#   sigma  ∈ {5.0, 25.0}   lr ∈ {5e-3, 1e-2, 5e-2}
#   共 3 模型 × 24 配置 × 7 数据集 = 504 个 job
#
# 用法:
#   N_GPUS=8 bash scripts2/step3b_grid_search_3models_h200.sh
#   MODEL_DIR=/path/to/models N_GPUS=4 bash scripts2/step3b_grid_search_3models_h200.sh
# ============================================================
set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

# ---- 配置区 (按需修改) ----
N_GPUS=${N_GPUS:-8}
MODEL_DIR=${MODEL_DIR:-/inspire/hdd/project/qproject-fundationmodel/xiashijie-240108120112/qsh/models}
# MODELS=(
#     "Qwen2.5-VL-3B-Instruct"
#     "Qwen3-VL-4B-Instruct"
#     "Qwen3-VL-8B-Instruct"
# )
MODELS=(
    "Qwen3-VL-4B-Instruct"
    "Qwen3-VL-8B-Instruct"
)

# TOKENS_LIST=(2 4)
# STEPS_LIST=(10 15)
# SIGMA_LIST=(5.0 25.0)
# SIGMA_DECAY=0.95
# LR_LIST=(5e-3 1e-2 5e-2)
# TOP_K=10

TOKENS_LIST=(1 2)
STEPS_LIST=(1 3)
SIGMA_LIST=(10.0 20.0)
SIGMA_DECAY=0.95
LR_LIST=(1e-4 5e-4 1e-3)
TOP_K=10

DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

root_output=./output/ltpo_dmlr_grid_dev_v4
mkdir -p "${root_output}"

# 构建 job 列表: model × tokens × steps × sigma × lr × dataset
jobs=()
for model in "${MODELS[@]}"; do
    for tokens in "${TOKENS_LIST[@]}"; do
        for steps in "${STEPS_LIST[@]}"; do
            for sigma in "${SIGMA_LIST[@]}"; do
                for lr in "${LR_LIST[@]}"; do
                    for dataset in "${DATASETS[@]}"; do
                        jobs+=("${model} ${tokens} ${steps} ${sigma} ${lr} ${dataset}")
                    done
                done
            done
        done
    done
done
total=${#jobs[@]}

echo "════════════════════════════════════════════════════════"
echo "Step 3b-3models: LTPO 网格搜索 (H200, 离线)"
echo "    模型目录: ${MODEL_DIR}"
echo "    模型: ${MODELS[*]}"
echo "    tokens ∈ {${TOKENS_LIST[*]}}  steps ∈ {${STEPS_LIST[*]}}"
echo "    sigma  ∈ {${SIGMA_LIST[*]}}   lr    ∈ {${LR_LIST[*]}}"
echo "    sigma_decay=${SIGMA_DECAY}  top_k=${TOP_K}  (固定)"
echo "    数据集: ${#DATASETS[@]} 个 dev 集 (各 300 样本)"
echo "    配置数/模型: $(( total / ${#MODELS[@]} / ${#DATASETS[@]} ))"
echo "    Job 总数: ${total}   GPU 数: ${N_GPUS}"
echo "    预估时间: $(( (total + N_GPUS - 1) / N_GPUS * 5 )) 分钟"
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
            read -r model tokens steps sigma lr dataset <<< "${jobs[$job_idx]}"

            tag="${model}/tokens${tokens}_steps${steps}_sigma${sigma}_decay${SIGMA_DECAY}_lr${lr}_topk${TOP_K}"
            out_dir="${root_output}/${tag}"
            mkdir -p "${out_dir}"
            log="${out_dir}/${dataset}.log"

            echo "[$(date +%T)] GPU ${g}  <-  ${model} | ${tag##*/} | ${dataset}"

            CUDA_VISIBLE_DEVICES=${g} python main_vl_dmlr.py \
                --dataset           "${dataset}"              \
                --data_root         mllm_data                 \
                --image_root        mllm_data                 \
                --model_name_or_path "${MODEL_DIR}/${model}"  \
                --output_dir        "${out_dir}"              \
                --device            cuda                      \
                --seed              42                        \
                --max_new_tokens    2048                      \
                --min_pixels        128                       \
                --max_pixels        256                       \
                --num_thought_tokens "${tokens}"              \
                --sigma             "${sigma}"                \
                --sigma_decay       "${SIGMA_DECAY}"          \
                --lr                "${lr}"                   \
                --max_num_steps     "${steps}"                \
                --top_k             "${TOP_K}"                \
                --verbose 1                                   \
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
echo "Rule-based 各模型各配置平均准确率 (降序排列):"
echo "════════════════════════════════════════════════════════"

for model in "${MODELS[@]}"; do
    echo ""
    echo "--- ${model} ---"
    for cfg_dir in "${root_output}/${model}"/tokens*; do
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
done
