#!/bin/bash
# wait_and_launch_visattnsink.sh — poll for GPU availability, then launch
# the VisAttnSink sweep. We need each worker to have ≥ ~30 GB free for
# the 7-8B Qwen-VL models running with eager attention; smaller models need
# less but we use one threshold to keep things simple.
#
# Strategy:
#   - Every CHECK_INTERVAL seconds, ask nvidia-smi for free memory per GPU.
#   - As soon as at least MIN_GPUS GPUs each have ≥ THRESH_MB free, build a
#     GPUS list of those device ids and invoke run_visattnsink_dev.sh.
#   - Otherwise log a one-line "still waiting" message.

set -u
cd "$(dirname "$0")/.."

CHECK_INTERVAL=${CHECK_INTERVAL:-120}
MIN_GPUS=${MIN_GPUS:-4}
THRESH_MB=${THRESH_MB:-30000}
LOG=${LOG:-./output/visattnsink_dev/launcher.log}
mkdir -p "$(dirname "$LOG")"

echo "[$(date +%F\ %T)] launcher started; need >=${MIN_GPUS} GPUs with >=${THRESH_MB} MiB free" >> "$LOG"

while true; do
    free_list=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    selected=()
    while IFS=',' read -r idx free; do
        idx=$(echo "$idx" | tr -d ' ')
        free=$(echo "$free" | tr -d ' ')
        if [ "$free" -ge "$THRESH_MB" ]; then
            selected+=("$idx")
        fi
    done <<< "$free_list"

    if [ "${#selected[@]}" -ge "$MIN_GPUS" ]; then
        gpus=$(IFS=','; echo "${selected[*]}")
        echo "[$(date +%F\ %T)] launching with GPUS=${gpus}" >> "$LOG"
        export GPUS="$gpus"
        bash scripts/run_visattnsink_dev.sh >> "$LOG" 2>&1
        echo "[$(date +%F\ %T)] sweep finished." >> "$LOG"
        exit 0
    fi

    echo "[$(date +%F\ %T)] still waiting; eligible GPUs=${#selected[@]}/$MIN_GPUS" >> "$LOG"
    sleep "$CHECK_INTERVAL"
done
