#!/bin/bash
# run_ltpo_vl_dmlr_log_topk.sh — Single LTPO run with top-k logging.
#
# Runs ONE LTPO configuration on ONE dataset and writes the top-k tokens
# that the confidence reward is rewarding (the tokens occupying the highest
# predicted-probability positions at each thought-token position) into the
# log file at every RL step.
#
# Configuration / dataset are deliberately fixed (no grid). Override the
# environment variables below if you want different values.
#
# Usage:
#   bash scripts/run_ltpo_vl_dmlr_log_topk.sh
#
# Overridable env vars:
#   GPU            CUDA device index (default 0)
#   MODEL          path to model checkpoint
#   DATASET        dataset name (default mmvp_dev)
#   NUM_EXAMPLES   number of examples to run (default 5; set to 999 for full)

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

GPU=${GPU:-0}
MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}
DATASET=${DATASET:-mmvp_dev}
NUM_EXAMPLES=${NUM_EXAMPLES:-5}

# Fixed single config (representative point inside the standard LTPO grid)
TOKENS=4
STEPS=15
SIGMA=5.0
SIGMA_DECAY=0.95
LR=5e-3
TOP_K=10

root_output=./output/ltpo_dmlr_log_topk
tag="tokens${TOKENS}_steps${STEPS}_sigma${SIGMA}_decay${SIGMA_DECAY}_lr${LR}_topk${TOP_K}"
out_dir="${root_output}/${tag}"
mkdir -p "${out_dir}"
log="${out_dir}/${DATASET}.log"

echo "════════════════════════════════════════════════════════"
echo "LTPO-DMLR single run with top-k token logging"
echo "    Dataset: ${DATASET}  (examples 0..${NUM_EXAMPLES})"
echo "    Config:  ${tag}"
echo "    Log:     ${log}"
echo "════════════════════════════════════════════════════════"

CUDA_VISIBLE_DEVICES=${GPU} python main_vl_dmlr.py \
    --dataset            "${DATASET}"    \
    --data_root          mllm_data       \
    --image_root         .               \
    --model_name_or_path "${MODEL}"      \
    --output_dir         "${out_dir}"    \
    --device             cuda            \
    --seed               42              \
    --max_new_tokens     2048            \
    --min_pixels         128             \
    --max_pixels         256             \
    --num_thought_tokens "${TOKENS}"     \
    --sigma              "${SIGMA}"      \
    --sigma_decay        "${SIGMA_DECAY}" \
    --lr                 "${LR}"         \
    --max_num_steps      "${STEPS}"      \
    --top_k              "${TOP_K}"      \
    --end_data_idx       "${NUM_EXAMPLES}" \
    --use_llm_verify                     \
    --log_topk_tokens                    \
    --verbose 1                          \
    > "${log}" 2>&1

echo "Done. Log written to ${log}"
echo ""
echo "Search the log for the top-k tokens with:"
echo "    grep -n 'top-' '${log}'"
echo "    grep -n -A \$(($((TOKENS+1)) + 1)) 'top-' '${log}' | head -80"
