#!/bin/bash
# ============================================================
# Step 3a: Baseline 评测 (无 LTPO 优化)
# 运行环境: H200 (无网络)
#
# 对应原始脚本: scripts/run_ltpo_vl_dmlr.sh
# 改动: 去掉 --use_llm_verify，使用 rule-based 判断
# ============================================================
set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

# ---- 配置区 (按需修改) ----
MODEL=${MODEL:-/inspire/hdd/project/qproject-fundationmodel/xiashijie-240108120112/qsh/models}
N_GPUS=${N_GPUS:-7}    # 可用 GPU 数量，默认 7（每个数据集一张卡）

output=./output/dmlr_vanilla
mkdir -p "${output}"

DATASETS=("mmvp" "mmstar" "mm_math" "math_vista" "math_vision" "hallusion" "scienceqa")

echo "========================================"
echo "Step 3a: Baseline 评测 (H200, 离线)"
echo "  模型: ${MODEL}"
echo "  GPU 数: ${N_GPUS}"
echo "  数据集: ${DATASETS[*]}"
echo "  输出: ${output}"
echo "========================================"

gpu=0
for dataset in "${DATASETS[@]}"; do
    echo "[$(date +%T)] GPU ${gpu} <- ${dataset}"

    CUDA_VISIBLE_DEVICES=$gpu python main_vl_dmlr.py \
        --dataset "$dataset" \
        --data_root mllm_data \
        --image_root mllm_data \
        --model_name_or_path "$MODEL" \
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
        --verbose 1 > "${output}/${dataset}.log" 2>&1 &

    gpu=$((gpu + 1))

    # 如果 GPU 用完了，等待当前一批完成再继续
    if [ "$gpu" -ge "$N_GPUS" ]; then
        echo "  等待当前 ${N_GPUS} 个任务完成..."
        wait
        gpu=0
    fi
done

wait
echo ""
echo "Baseline 评测全部完成。结果在 ${output}/"
