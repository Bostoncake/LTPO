#!/bin/bash
# run_ltpo_vl_dmlr_final_baseline_dev.sh
#
# No-latent-tokens baseline whose forward path is aligned with the
# FIXED LTPO implementation (ltpo_vl_dmlr_final / main_vl_dmlr_final),
# in which image_grid_thw is preserved alongside the pre-merged
# inputs_embeds so Qwen2.5-VL keeps mRoPE position_ids.
#
# Why this script is enough:
#   With the fix, the LTPO path runs model.generate with
#   (inputs_embeds, image_grid_thw, attention_mask) — i.e. mRoPE on.
#   The --eval_baseline branch (no thought tokens, no init flag) runs
#   model.generate(**inputs) with (input_ids, pixel_values,
#   image_grid_thw, attention_mask). These two paths hit the same
#   model.forward (one pre-merges visuals externally, the other lets the
#   model merge internally), so the no-latent baseline produced here is
#   apples-to-apples with the fixed LTPO forward.
#
# Total: 7 datasets × 1 config = 7 jobs.
#
# Usage:
#   bash scripts/run_ltpo_vl_dmlr_final_baseline_dev.sh
#
# Overridable env vars:
#   N_GPUS    number of GPUs to use (default 8)
#   MODEL     path to model checkpoint

set -u

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

N_GPUS=${N_GPUS:-8}
MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}

DATASETS=("mmvp_dev" "mmstar_dev" "mm_math_dev" "math_vista_dev" "math_vision_dev" "hallusion_dev" "scienceqa_dev")

root_output=./output/ltpo_dmlr_final/0515_baseline_no_latent_ltpo_aligned
mkdir -p "${root_output}"

jobs=()
for dataset in "${DATASETS[@]}"; do
    jobs+=("${dataset}")
done
total=${#jobs[@]}

echo "════════════════════════════════════════════════════════"
echo "LTPO-DMLR FINAL  no-latent baseline  (mRoPE on)"
echo "    Datasets: ${#DATASETS[@]}   Total jobs: ${total}   GPUs: ${N_GPUS}"
echo "════════════════════════════════════════════════════════"

declare -a gpu_pids
for ((g=0; g<N_GPUS; g++)); do gpu_pids[$g]=-1; done

job_idx=0
while [ $job_idx -lt $total ]; do
    for ((g=0; g<N_GPUS; g++)); do
        [ $job_idx -ge $total ] && break
        pid=${gpu_pids[$g]}
        if [ "$pid" -eq -1 ] || ! kill -0 "$pid" 2>/dev/null; then
            dataset="${jobs[$job_idx]}"

            tag="baseline_no_latent"
            out_dir="${root_output}/${tag}"
            mkdir -p "${out_dir}"
            log="${out_dir}/${dataset}.log"

            echo "[$(date +%T)] GPU ${g}  ←  ${tag}  |  ${dataset}"

            CUDA_VISIBLE_DEVICES=${g} python main_vl_dmlr_final.py \
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
echo "All ${total} jobs done. Results in ${root_output}"
echo "Per-dataset accuracy:"
echo "════════════════════════════════════════════════════════"

for cfg_dir in "${root_output}"/baseline_*; do
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
