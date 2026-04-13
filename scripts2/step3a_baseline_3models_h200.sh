#!/bin/bash
# ============================================================
# Step 3a-3models: 3 个模型 × 7 个数据集 Baseline 评测
# 运行环境: H200 (无网络)
#
# 对应原始脚本: scripts/run_ltpo_vl_dmlr_3models.sh
# 改动:
#   - 去掉 --use_llm_verify
#   - --image_root 改为 mllm_data
#   - 模型路径和输出路径可配置
#
# 用法:
#   N_GPUS=8 bash scripts2/step3a_baseline_3models_h200.sh
#   MODEL_DIR=/path/to/models N_GPUS=4 bash scripts2/step3a_baseline_3models_h200.sh
# ============================================================
set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

# ---- 配置区 (按需修改) ----
N_GPUS=${N_GPUS:-8}
MODEL_DIR=${MODEL_DIR:-/inspire/hdd/project/qproject-fundationmodel/xiashijie-240108120112/qsh/models}
MODELS=(
    "Qwen2.5-VL-3B-Instruct"
    "Qwen3-VL-4B-Instruct"
    "Qwen3-VL-8B-Instruct"
)
# DATASETS=("mmvp" "mmstar" "mm_math" "math_vista" "math_vision" "hallusion" "scienceqa")
DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

output=./output/dmlr_vanilla_dev
mkdir -p "${output}"

# 构建 job 列表: 3 × 7 = 21 个 job
jobs=()
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        jobs+=("$model $dataset")
    done
done
total=${#jobs[@]}

echo "════════════════════════════════════════════════════════"
echo "Step 3a-3models: Baseline 评测 (H200, 离线)"
echo "    模型目录: ${MODEL_DIR}"
echo "    模型: ${MODELS[*]}"
echo "    数据集: ${#DATASETS[@]} 个"
echo "    Job 总数: ${total}   GPU 数: ${N_GPUS}"
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
            read -r model dataset <<< "${jobs[$job_idx]}"

            echo "[$(date +%T)] GPU ${g}  <-  ${model} / ${dataset}"

            CUDA_VISIBLE_DEVICES=$g python main_vl_dmlr.py \
                --dataset "$dataset" \
                --data_root mllm_data \
                --image_root mllm_data \
                --model_name_or_path "${MODEL_DIR}/${model}" \
                --output_dir "${output}" \
                --device cuda \
                --seed 42 \
                --max_new_tokens 2048 \
                --min_pixels 128 \
                --max_pixels 256 \
                --num_thought_tokens 2 \
                --sigma 25.0 \
                --sigma_decay 0.95 \
                --lr 0.01 \
                --max_num_steps 15 \
                --top_k 10 \
                --eval_baseline \
                --verbose 1 > "${output}/${model}_${dataset}.log" 2>&1 &

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
echo "════════════════════════════════════════════════════════"
