#!/bin/bash

export HUGGING_FACE_TOKEN="***REDACTED_HF_TOKEN***"
export OPENAI_API_KEY="***REDACTED_OPENAI_KEY***"
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

MODEL_3B=/export/home/lanliwei.1/abcxyz/storage/models/Qwen2.5-VL-3B-Instruct
MODEL_4B=/export/home/lanliwei.1/abcxyz/storage/models/Qwen3-VL-4B-Instruct

OUT_BASE=./output/dmlr_aligned_full_v7_prompt_baseline
OUT_V7=./output/dmlr_aligned_full_v7_prompt
OUT_V8=./output/dmlr_aligned_dev_v8_prompt_baseline

mkdir -p "$OUT_BASE" "$OUT_V7" "$OUT_V8"

COMMON_ARGS=(
    --data_root mllm_data
    --image_root .
    --device cuda
    --seed 42
    --max_new_tokens 2048
    --min_pixels 128
    --max_pixels 256
    --num_thought_tokens 2
    --sigma 25.0
    --sigma_decay 0.95
    --lr 0.01
    --max_num_steps 15
    --top_k 10
    --use_llm_verify
    --verbose 1
)

# 每张卡串行跑自己的任务列：v7_baseline → v7_full → v8（可选）
# 参数：run_lane <gpu> <v7_dataset> [v8_dataset]
run_lane() {
    local gpu=$1
    local ds_v7=$2
    local ds_v8=${3:-}

    echo "[$(date '+%H:%M:%S')] GPU $gpu | v7_baseline/$ds_v7  start"
    CUDA_VISIBLE_DEVICES=$gpu python main_vl_dmlr_v7.py \
        --dataset $ds_v7 \
        --model_name_or_path $MODEL_3B \
        --output_dir ${OUT_BASE} \
        "${COMMON_ARGS[@]}" \
        --eval_baseline \
        > ${OUT_BASE}/${ds_v7}.log 2>&1
    echo "[$(date '+%H:%M:%S')] GPU $gpu | v7_baseline/$ds_v7  done"

    echo "[$(date '+%H:%M:%S')] GPU $gpu | v7_full/$ds_v7      start"
    CUDA_VISIBLE_DEVICES=$gpu python main_vl_dmlr_v7.py \
        --dataset $ds_v7 \
        --model_name_or_path $MODEL_3B \
        --output_dir ${OUT_V7} \
        "${COMMON_ARGS[@]}" \
        > ${OUT_V7}/${ds_v7}.log 2>&1
    echo "[$(date '+%H:%M:%S')] GPU $gpu | v7_full/$ds_v7      done"

    if [[ -n "$ds_v8" ]]; then
        echo "[$(date '+%H:%M:%S')] GPU $gpu | v8/$ds_v8          start"
        CUDA_VISIBLE_DEVICES=$gpu python main_vl_dmlr_v8.py \
            --dataset $ds_v8 \
            --model_name_or_path $MODEL_4B \
            --output_dir ${OUT_V8} \
            "${COMMON_ARGS[@]}" \
            --eval_baseline \
            > ${OUT_V8}/${ds_v8}.log 2>&1
        echo "[$(date '+%H:%M:%S')] GPU $gpu | v8/$ds_v8          done"
    fi
}

# GPU 0-6 各自并行，卡内串行
# mm_math 在 v8 里没有对应的 dev dataset，GPU 2 只跑 2 个任务
run_lane 0  mmvp        mmvp_dev        &
run_lane 1  mmstar      mmstar_dev      &
run_lane 2  mm_math                     &
run_lane 3  math_vista  math_vista_dev  &
run_lane 4  math_vision math_vision_dev &
run_lane 5  hallusion   hallusion_dev   &
run_lane 6  scienceqa   scienceqa_dev   &

wait
echo "[$(date '+%H:%M:%S')] All done."
