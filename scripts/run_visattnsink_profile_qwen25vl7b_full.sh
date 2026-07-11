#!/usr/bin/env bash
# Run VisAttnSink on the full MLLM splits with lightweight inference profiling.
#
# Default target:
#   /home/xiongyizhe/hqs/storage/models/Qwen2.5-VL-7B-Instruct
#
# Useful overrides:
#   GPUS=0,1,2,3
#   ROOT_OUTPUT=./output/visattnsink_profile_qwen25vl7b_full
#   END_DATA_IDX=1              # smoke test only
#   USE_LLM_VERIFY=0            # rule-based judge

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export HUGGING_FACE_TOKEN="${HUGGING_FACE_TOKEN:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export MODEL_TYPE="${MODEL_TYPE:-qwen-max}"

MODEL_PATH="${MODEL_PATH:-/home/xiongyizhe/hqs/storage/models/Qwen2.5-VL-7B-Instruct}"
MODEL_NAME="$(basename "${MODEL_PATH}")"
DATASETS=(${DATASETS:-mmvp mmstar mm_math math_vista math_vision hallusion scienceqa})

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
N_GPUS=${#GPU_LIST[@]}

ROOT_OUTPUT="${ROOT_OUTPUT:-./output/visattnsink_profile_qwen25vl7b_full}"
mkdir -p "${ROOT_OUTPUT}/${MODEL_NAME}"

CONDA_ENV="${CONDA_ENV:-ltpo}"
if [ -n "${PYTHON_BIN:-}" ]; then
    PY_CMD=("${PYTHON_BIN}")
elif command -v conda >/dev/null 2>&1; then
    PY_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" python)
else
    PY_CMD=(python)
fi

VERIFY_ARGS=()
if [ "${USE_LLM_VERIFY:-1}" != "0" ]; then
    VERIFY_ARGS+=(--use_llm_verify)
fi

RANGE_ARGS=()
if [ -n "${START_DATA_IDX:-}" ]; then
    RANGE_ARGS+=(--start_data_idx "${START_DATA_IDX}")
fi
if [ -n "${END_DATA_IDX:-}" ]; then
    RANGE_ARGS+=(--end_data_idx "${END_DATA_IDX}")
fi

COMMON_ARGS=(
    --data_root mllm_data
    --image_root .
    --model_name_or_path "${MODEL_PATH}"
    --output_dir "${ROOT_OUTPUT}/${MODEL_NAME}"
    --device cuda
    --seed "${SEED:-42}"
    --max_new_tokens "${MAX_NEW_TOKENS:-2048}"
    --min_pixels "${MIN_PIXELS:-128}"
    --max_pixels "${MAX_PIXELS:-256}"
    --vas_tau "${VAS_TAU:-20.0}"
    --vas_rho "${VAS_RHO:-0.5}"
    --vas_summ "${VAS_SUMM:-0.2}"
    --vas_p "${VAS_P:-0.6}"
    --vas_except_last_layer "${VAS_EXCEPT_LAST_LAYER:-1}"
    --profile_inference
    --profile_sync_cuda
    --verbose "${VERBOSE:-1}"
)

if [ "${RESUME:-0}" != "0" ]; then
    COMMON_ARGS+=(--resume)
fi

echo "============================================================"
echo "VisAttnSink full profiling"
echo "Model:       ${MODEL_PATH}"
echo "Datasets:    ${DATASETS[*]}"
echo "Output:      ${ROOT_OUTPUT}/${MODEL_NAME}"
echo "GPUs:        ${GPUS} (${N_GPUS} workers)"
echo "Python:      ${PY_CMD[*]}"
echo "LLM judge:   ${USE_LLM_VERIFY:-1}  MODEL_TYPE=${MODEL_TYPE}"
echo "Range args:  ${RANGE_ARGS[*]:-(full dataset)}"
echo "============================================================"

failures=0
job_idx=0
total=${#DATASETS[@]}

while [ ${job_idx} -lt ${total} ]; do
    declare -a batch_pids=()
    declare -a batch_logs=()
    declare -a batch_gpus=()
    for ((g=0; g<N_GPUS; g++)); do
        [ ${job_idx} -ge ${total} ] && break
        dataset="${DATASETS[$job_idx]}"
        gpu="${GPU_LIST[$g]}"
        log="${ROOT_OUTPUT}/${MODEL_NAME}/${dataset}.log"
        echo "[$(date +%F' '%T)] GPU ${gpu} <- ${dataset}"

        CUDA_VISIBLE_DEVICES="${gpu}" "${PY_CMD[@]}" main_vl_visattnsink.py \
            --dataset "${dataset}" \
            "${COMMON_ARGS[@]}" \
            "${VERIFY_ARGS[@]}" \
            "${RANGE_ARGS[@]}" \
            > "${log}" 2>&1 &
        batch_pids+=($!)
        batch_logs+=("${log}")
        batch_gpus+=("${gpu}")
        job_idx=$((job_idx + 1))
    done
    for ((j=0; j<${#batch_pids[@]}; j++)); do
        if ! wait "${batch_pids[$j]}"; then
            echo "[ERROR] Job on GPU ${batch_gpus[$j]} failed. Log: ${batch_logs[$j]}"
            failures=$((failures + 1))
        fi
    done
done

echo ""
echo "============================================================"
echo "Per-dataset results"
echo "============================================================"
find "${ROOT_OUTPUT}/${MODEL_NAME}" -name results.log -print | sort | while read -r result_log; do
    cfg_dir="$(dirname "${result_log}")"
    acc="$(grep -oP 'accuracy=\K[0-9.]+' "${result_log}" | tail -1 || true)"
    summary="${cfg_dir}/profile_summary.json"
    if [ -f "${summary}" ]; then
        "${PY_CMD[@]}" - "${summary}" "$(basename "${cfg_dir}")" "${acc}" <<'PY'
import json
import sys

summary_path, name, acc = sys.argv[1:4]
with open(summary_path) as f:
    s = json.load(f)
print(
    f"{name}: acc={acc or 'NA'} "
    f"examples={s.get('num_profiled_examples', 0)} "
    f"total_flops={s.get('total_flops_sum', 0.0):.4e} "
    f"mean_flops={s.get('total_flops_mean', 0.0):.4e} "
    f"gen_wall={s.get('generate_wall_s_sum', 0.0):.2f}s "
    f"tok/s={s.get('tokens_per_second_mean', 0.0):.3f}"
)
PY
    else
        echo "$(basename "${cfg_dir}"): acc=${acc:-NA} profile_summary=missing"
    fi
done

if [ ${failures} -ne 0 ]; then
    echo "[ERROR] ${failures} job(s) failed."
    exit 1
fi

echo "Done. Outputs are under ${ROOT_OUTPUT}/${MODEL_NAME}"
