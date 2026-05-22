#!/bin/bash
# run_ltpo_vl_dmlr_direct_boxed_baseline_thtok_sweep_dev_qwen25vl3b.sh
#
# Direct-boxed BASELINE thought-token-init sweep on dev sets for Qwen2.5-VL-3B.
# Tests --baseline_thought_init_from_mean across 1-5 latent thought tokens:
#   (1) 1 thought token,  mean-of-text-embeddings init  -> tag: thtok1_mean
#   (2) 2 thought tokens, mean-of-text-embeddings init  -> tag: thtok2_mean
#   (3) 3 thought tokens, mean-of-text-embeddings init  -> tag: thtok3_mean
#   (4) 4 thought tokens, mean-of-text-embeddings init  -> tag: thtok4_mean
#   (5) 5 thought tokens, mean-of-text-embeddings init  -> tag: thtok5_mean
#
# Mirrors the baseline branch of
# run_ltpo_vl_dmlr_direct_boxed_oracle_dev.sh (same datasets, same
# inference args, same per-GPU pid-tracking scheduler). All scenarios are
# launched in a single pool so no GPU is left idle.
#
# Usage:
#   bash scripts/run_ltpo_vl_dmlr_direct_boxed_baseline_thtok_sweep_dev_qwen25vl3b.sh qwen25vl3b
#
# Overridable env vars:
#   N_GPUS         number of GPUs to use (default 8)
#   MODEL          override model path (otherwise inferred from MODEL_KEY)
#   ROOT_OUTPUT    override output root (otherwise inferred from MODEL_KEY)

set -u

if [ "$#" -lt 1 ]; then
    echo "Usage: bash $0 {qwen25vl3b}" >&2
    exit 1
fi

MODEL_KEY="$1"

case "${MODEL_KEY}" in
    qwen25vl3b)
        DEFAULT_MODEL=/WillDevExt/xiongyizhe/models/Qwen2.5-VL-3B-Instruct
        MODEL_TAG=qwen25vl3b
        ;;
    *)
        echo "Unknown MODEL_KEY '${MODEL_KEY}'. Use qwen25vl3b." >&2
        exit 1
        ;;
esac

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=${N_GPUS:-8}
MODEL=${MODEL:-${DEFAULT_MODEL}}
ROOT_OUTPUT=${ROOT_OUTPUT:-./output/ltpo_dmlr_direct_boxed/0521_baseline_thtok_sweep_${MODEL_TAG}}

DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

# Each scenario encodes: tag | num_thought_tokens | with_thought_tokens(0/1) | init_from_hidden(0/1) | init_from_mean(0/1)
SCENARIOS=(
    "thtok1_mean|1|1|0|1"
    "thtok2_mean|2|1|0|1"
    "thtok3_mean|3|1|0|1"
    "thtok4_mean|4|1|0|1"
    "thtok5_mean|5|1|0|1"
)

mkdir -p "${ROOT_OUTPUT}"

# Build flat job list: (scenario_idx, dataset)
jobs=()
for s_idx in "${!SCENARIOS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        jobs+=("${s_idx} ${dataset}")
    done
done
total=${#jobs[@]}

echo "════════════════════════════════════════════════════════"
echo "DIRECT-BOXED BASELINE thought-token-init sweep on dev sets"
echo "    Model key : ${MODEL_KEY}"
echo "    Model path: ${MODEL}"
echo "    Output    : ${ROOT_OUTPUT}"
echo "    Scenarios : ${#SCENARIOS[@]}  Datasets: ${#DATASETS[@]}"
echo "    Total jobs: ${total}   GPUs: ${N_GPUS}"
echo "════════════════════════════════════════════════════════"

declare -a gpu_pids
for ((g=0; g<N_GPUS; g++)); do gpu_pids[$g]=-1; done

job_idx=0
while [ $job_idx -lt $total ]; do
    for ((g=0; g<N_GPUS; g++)); do
        [ $job_idx -ge $total ] && break
        pid=${gpu_pids[$g]}
        if [ "$pid" -eq -1 ] || ! kill -0 "$pid" 2>/dev/null; then
            read -r s_idx dataset <<< "${jobs[$job_idx]}"
            IFS='|' read -r tag num_tokens with_thtok init_hidden init_mean <<< "${SCENARIOS[$s_idx]}"

            out_dir="${ROOT_OUTPUT}/${tag}"
            mkdir -p "${out_dir}"
            log="${out_dir}/${dataset}.log"

            echo "[$(date +%T)] GPU ${g}  ←  ${tag}  |  ${dataset}"

            extra_args=()
            if [ "${with_thtok}" = "1" ]; then
                extra_args+=(--baseline_with_thought_tokens)
                extra_args+=(--num_thought_tokens "${num_tokens}")
                if [ "${init_hidden}" = "1" ]; then
                    extra_args+=(--baseline_thought_init_from_hidden)
                fi
                if [ "${init_mean}" = "1" ]; then
                    extra_args+=(--baseline_thought_init_from_mean)
                fi
            else
                # Still pass num_thought_tokens for argparse default consistency
                extra_args+=(--num_thought_tokens 2)
            fi

            CUDA_VISIBLE_DEVICES=${g} python main_vl_dmlr_direct_boxed.py \
                --dataset           "${dataset}"   \
                --data_root         mllm_data      \
                --image_root        .              \
                --model_name_or_path "${MODEL}"    \
                --output_dir        "${out_dir}"   \
                --device            cuda           \
                --seed              42             \
                --max_new_tokens    2048           \
                --min_pixels        128            \
                --max_pixels        256            \
                --eval_baseline                    \
                --use_llm_verify                   \
                "${extra_args[@]}"                 \
                --verbose 1                        \
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
echo "All ${total} jobs done. Results in ${ROOT_OUTPUT}"
echo "Per-scenario mean accuracy:"
echo "════════════════════════════════════════════════════════"

for cfg_dir in "${ROOT_OUTPUT}"/*; do
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
