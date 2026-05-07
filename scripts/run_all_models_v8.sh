#!/bin/bash

export HUGGING_FACE_TOKEN="***REDACTED_HF_TOKEN***"
export OPENAI_API_KEY="***REDACTED_OPENAI_KEY***"
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

MODEL_4B=/export/home/lanliwei.1/abcxyz/storage/models/Qwen3-VL-4B-Instruct
MODEL_8B=/export/home/lanliwei.1/abcxyz/storage/models/Qwen3-VL-8B-Instruct
MODEL_3B=/export/home/lanliwei.1/abcxyz/storage/models/Qwen2.5-VL-3B-Instruct

OUT_BASELINE=./output/dmlr_aligned_full_v8_prompt_baseline
OUT_FULL=./output/dmlr_aligned_full_v8_prompt

mkdir -p "$OUT_BASELINE" "$OUT_FULL"

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

# 每张卡串行跑自己的任务列：3-4B baseline → 3-4B full → 3-8B baseline → 3-8B full → 2.5-3B baseline → 2.5-3B full
run_lane() {
    local gpu=$1
    local ds=$2

    echo "[$(date '+%H:%M:%S')] GPU $gpu | 3-4B baseline/$ds  start"
    CUDA_VISIBLE_DEVICES=$gpu python main_vl_dmlr_v8.py \
        --dataset $ds \
        --model_name_or_path $MODEL_4B \
        --output_dir ${OUT_BASELINE} \
        "${COMMON_ARGS[@]}" \
        --eval_baseline \
        > ${OUT_BASELINE}/${ds}_4b.log 2>&1
    echo "[$(date '+%H:%M:%S')] GPU $gpu | 3-4B baseline/$ds  done"

    echo "[$(date '+%H:%M:%S')] GPU $gpu | 3-4B full/$ds      start"
    CUDA_VISIBLE_DEVICES=$gpu python main_vl_dmlr_v8.py \
        --dataset $ds \
        --model_name_or_path $MODEL_4B \
        --output_dir ${OUT_FULL} \
        "${COMMON_ARGS[@]}" \
        > ${OUT_FULL}/${ds}_4b.log 2>&1
    echo "[$(date '+%H:%M:%S')] GPU $gpu | 3-4B full/$ds      done"

    echo "[$(date '+%H:%M:%S')] GPU $gpu | 3-8B baseline/$ds  start"
    CUDA_VISIBLE_DEVICES=$gpu python main_vl_dmlr_v8.py \
        --dataset $ds \
        --model_name_or_path $MODEL_8B \
        --output_dir ${OUT_BASELINE} \
        "${COMMON_ARGS[@]}" \
        --eval_baseline \
        > ${OUT_BASELINE}/${ds}_8b.log 2>&1
    echo "[$(date '+%H:%M:%S')] GPU $gpu | 3-8B baseline/$ds  done"

    echo "[$(date '+%H:%M:%S')] GPU $gpu | 3-8B full/$ds      start"
    CUDA_VISIBLE_DEVICES=$gpu python main_vl_dmlr_v8.py \
        --dataset $ds \
        --model_name_or_path $MODEL_8B \
        --output_dir ${OUT_FULL} \
        "${COMMON_ARGS[@]}" \
        > ${OUT_FULL}/${ds}_8b.log 2>&1
    echo "[$(date '+%H:%M:%S')] GPU $gpu | 3-8B full/$ds      done"

    echo "[$(date '+%H:%M:%S')] GPU $gpu | 2.5-3B baseline/$ds  start"
    CUDA_VISIBLE_DEVICES=$gpu python main_vl_dmlr_v8.py \
        --dataset $ds \
        --model_name_or_path $MODEL_3B \
        --output_dir ${OUT_BASELINE} \
        "${COMMON_ARGS[@]}" \
        --eval_baseline \
        > ${OUT_BASELINE}/${ds}_3b.log 2>&1
    echo "[$(date '+%H:%M:%S')] GPU $gpu | 2.5-3B baseline/$ds  done"

    echo "[$(date '+%H:%M:%S')] GPU $gpu | 2.5-3B full/$ds      start"
    CUDA_VISIBLE_DEVICES=$gpu python main_vl_dmlr_v8.py \
        --dataset $ds \
        --model_name_or_path $MODEL_3B \
        --output_dir ${OUT_FULL} \
        "${COMMON_ARGS[@]}" \
        > ${OUT_FULL}/${ds}_3b.log 2>&1
    echo "[$(date '+%H:%M:%S')] GPU $gpu | 2.5-3B full/$ds      done"
}

# GPU 0-6 各自并行，卡内串行
run_lane 0  mmvp        &
run_lane 1  mmstar      &
run_lane 2  mm_math     &
run_lane 3  math_vista  &
run_lane 4  math_vision &
run_lane 5  hallusion   &
run_lane 6  scienceqa   &

wait
echo "[$(date '+%H:%M:%S')] All done."
