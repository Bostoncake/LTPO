#!/bin/bash
# run_slot_resume.sh — resume Qwen3-VL-4B (math_vision, mm_math) and run
# Qwen2.5-VL-3B from logistics checkpoints. Uses setsid so child jobs are
# in their own session (immune to parent-shell SIGHUP, which orphaned the
# previous launcher's tail-end jobs).

cd "$(dirname "$0")/.."

source /export/home/lanliwei.1/abcxyz/env/miniconda3/bin/activate
conda activate ltpo

:
"${HUGGING_FACE_TOKEN:?Set HUGGING_FACE_TOKEN before running this script}"
:
"${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script}"
export HUGGING_FACE_TOKEN OPENAI_API_KEY
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

MODELS_ROOT=/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/Other_baseline/models
QWEN3_4B="${MODELS_ROOT}/Qwen3-VL-4B-Instruct"
QWEN25_3B="${MODELS_ROOT}/Qwen2.5-VL-3B-Instruct"
OUTPUT_ROOT=./output/slot

# Helper: launch one (model, dataset) job on a GPU in its own session.
launch_one() {
    local model="$1" ds="$2" gpu="$3"
    local mname=$(basename "${model}")
    local logf="${OUTPUT_ROOT}/${mname}__${ds}.log"
    echo "[$(date +%T)] -> GPU ${gpu}  ${mname}  ${ds}  -> ${logf}"
    CUDA_VISIBLE_DEVICES=${gpu} setsid python main_vl_slot.py \
        --dataset "${ds}" \
        --data_root mllm_data \
        --image_root . \
        --model_name_or_path "${model}" \
        --output_dir "${OUTPUT_ROOT}" \
        --device cuda \
        --seed 42 \
        --max_new_tokens 2048 \
        --min_pixels 128 \
        --max_pixels 256 \
        --slot_times 3 \
        --slot_lr 0.01 \
        --use_llm_verify \
        --resume \
        --verbose 1 \
        >> "${logf}" 2>&1 &
    echo "$!"
}

PIDS=()
NAMES=()
# Phase A: Qwen3-VL-4B leftovers on GPUs 0,1 (the slow ones).
PIDS+=( "$(launch_one "${QWEN3_4B}" math_vision_dev 0)" ); NAMES+=("q3_4b__math_vision")
PIDS+=( "$(launch_one "${QWEN3_4B}" mm_math_dev      1)" ); NAMES+=("q3_4b__mm_math")
# Phase A also: Qwen2.5-VL-3B on GPUs 2..7 (6 datasets).
PIDS+=( "$(launch_one "${QWEN25_3B}" mmvp_dev        2)" ); NAMES+=("q25_3b__mmvp")
PIDS+=( "$(launch_one "${QWEN25_3B}" mmstar_dev      3)" ); NAMES+=("q25_3b__mmstar")
PIDS+=( "$(launch_one "${QWEN25_3B}" mm_math_dev     4)" ); NAMES+=("q25_3b__mm_math")
PIDS+=( "$(launch_one "${QWEN25_3B}" math_vista_dev  5)" ); NAMES+=("q25_3b__math_vista")
PIDS+=( "$(launch_one "${QWEN25_3B}" hallusion_dev   6)" ); NAMES+=("q25_3b__hallusion")
PIDS+=( "$(launch_one "${QWEN25_3B}" scienceqa_dev   7)" ); NAMES+=("q25_3b__scienceqa")

for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || echo "[$(date +%T)] WARN ${NAMES[$i]} exit non-zero"
    echo "[$(date +%T)] phaseA done: ${NAMES[$i]}"
done

# Phase B: launch the missing Qwen2.5-VL-3B math_vision on whichever GPU is
# still free (try GPU 0 since Qwen3-4B math_vision probably finished by now).
echo "[$(date +%T)] === phase B: Qwen2.5-VL-3B math_vision_dev ==="
PID_B=$(launch_one "${QWEN25_3B}" math_vision_dev 0)
wait "${PID_B}" || echo "[$(date +%T)] WARN q25_3b__math_vision exit non-zero"
echo "[$(date +%T)] phase B done"

echo "[$(date +%F\ %T)] === ALL DONE ==="
