#!/bin/bash
# run_icot_all_models.sh — run ICoT baseline sequentially over 4 models.
# Order requested by user: qwen2.5-7B → qwen3-8B → qwen3-4B → qwen2.5-3B.

set -u
cd /export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO

source /export/home/lanliwei.1/abcxyz/env/miniconda3/bin/activate
conda activate ltpo

MODELS=(
    "/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/Other_baseline/models/Qwen2.5-VL-7B-Instruct"
    "/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/Other_baseline/models/Qwen3-VL-8B-Instruct"
    "/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/Other_baseline/models/Qwen3-VL-4B-Instruct"
    "/export/home/lanliwei.1/abcxyz/projects/latent-reasoning/Other_baseline/models/Qwen2.5-VL-3B-Instruct"
)
TAGS=(
    "qwen25vl7b"
    "qwen3vl8b"
    "qwen3vl4b"
    "qwen25vl3b"
)

for i in "${!MODELS[@]}"; do
    echo "================================================================"
    echo "[$(date +%T)] Starting model ${TAGS[$i]} -> ${MODELS[$i]}"
    echo "================================================================"
    bash scripts/run_icot_dev.sh "${MODELS[$i]}" "${TAGS[$i]}"
    echo "[$(date +%T)] Finished model ${TAGS[$i]}"
done

echo "[$(date +%T)] All 4 models finished."
