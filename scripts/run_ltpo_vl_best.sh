export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

# ── Fixed settings ─────────────────────────────────────────────────────────────
model=/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct
image_root=.
max_new_tokens=2048
base_output=./output/0321_best_all_dsets
num_gpus=7

mkdir -p "${base_output}"

# ── 5 representative configs derived from top-10 grid-search analysis ──────────
#
# Top-10 reveals two structural clusters plus one fixed consensus:
#   Cluster A (tokens=4, steps=10): higher token budget, fewer steps → ranks 1-2
#   Cluster B (tokens=2, steps=15): lower token budget, more steps   → ranks 3-5
#   Within each cluster the key axis is (lr, sigma):
#     high-lr / high-sigma  (lr≈5e-2, sigma=5.0)
#     low-lr  / low-sigma   (lr≈1e-2, sigma=0.1)
#   Consensus fixed values: topk=10 (in all top-4), sigma_decay=0.95 (8/10 top)
#   Rank-1 exception: sigma_decay=0.9 with high-lr/high-sigma → keep as config-5
#
# 5 configs × 6 datasets = 30 jobs  →  4 batches of 8 GPUs  (~60 min)
#
# Format: "tokens lr sigma sigma_decay steps topk"
configs=(
  "4  5e-2  5.0  0.9   10  10"   # rank-1: Cluster-A, high-lr/high-sigma, decay=0.9
  "4  1e-2  0.1  0.95  10  10"   # rank-2: Cluster-A, low-lr/low-sigma,   decay=0.95
  "2  5e-3  5.0  0.95  15  10"   # rank-3: Cluster-B, high-lr/high-sigma, decay=0.95
  "2  1e-2  0.1  0.95  15  10"   # rank-4: Cluster-B, low-lr/low-sigma,   decay=0.95
  "4  5e-2  5.0  0.95  10  10"   # rank-1 variant: test sensitivity of decay (0.9→0.95)
)

datasets=("mmstar" "mm_math" "math_vista" "math_vision" "hallusion" "scienceqa")

# ── Build job list (dataset × config) ─────────────────────────────────────────
jobs=()
for dataset in "${datasets[@]}"; do
  for cfg in "${configs[@]}"; do
    jobs+=("$dataset $cfg")
  done
done

total=${#jobs[@]}
echo "════════════════════════════════════════════════"
echo "LTPO-VL: 5-config subset on 6 datasets"
echo "  Configs   : ${#configs[@]}"
echo "  Datasets  : ${#datasets[@]}"
echo "  Total jobs: $total"
echo "  GPUs      : $num_gpus  (IDs 1–7)"
echo "  Est. time : $(( (total + num_gpus - 1) / num_gpus * 15 )) min"
echo "════════════════════════════════════════════════"

# ── Run in batches of num_gpus ─────────────────────────────────────────────────
batch_pids=()
gpu_slot=1

for job in "${jobs[@]}"; do
  read -r dataset tokens lr sigma decay steps topk <<< "$job"

  exp_output="${base_output}/${dataset}"
  mkdir -p "${exp_output}"

  tag="tokens${tokens}_lr${lr}_sigma${sigma}_decay${decay}_steps${steps}_topk${topk}"
  echo "[GPU ${gpu_slot}] dataset=${dataset}  ${tag}"

  CUDA_VISIBLE_DEVICES=${gpu_slot} python main_vl.py \
    --dataset            "${dataset}"          \
    --data_root          mllm_data             \
    --image_root         "${image_root}"       \
    --model_name_or_path "${model}"            \
    --output_dir         "${exp_output}"       \
    --device             cuda                  \
    --max_new_tokens     ${max_new_tokens}     \
    --max_num_steps      ${steps}              \
    --num_thought_tokens ${tokens}             \
    --sigma              ${sigma}              \
    --sigma_decay        ${decay}              \
    --lr                 ${lr}                 \
    --top_k              ${topk}              \
    --verbose            0                     \
    > "${exp_output}/${tag}.stdout" 2>&1 &

  batch_pids+=($!)
  gpu_slot=$(( gpu_slot + 1 ))

  if [ ${#batch_pids[@]} -eq $num_gpus ]; then
    echo "  → waiting for batch of $num_gpus …"
    for pid in "${batch_pids[@]}"; do wait "$pid"; done
    batch_pids=()
    gpu_slot=1
    echo "  → batch done"
  fi
done

if [ ${#batch_pids[@]} -gt 0 ]; then
  echo "  → waiting for final batch of ${#batch_pids[@]} …"
  for pid in "${batch_pids[@]}"; do wait "$pid"; done
fi

# ── Summarise results per dataset ─────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "Results per dataset (sorted by accuracy):"
echo "════════════════════════════════════════════════"
for dataset in "${datasets[@]}"; do
  echo ""
  echo "── ${dataset} ──"
  find "${base_output}/${dataset}" -name "results.log" | while read -r log; do
    last=$(tail -1 "$log")
    echo "$last  | $(dirname "$log")"
  done | sort -t'=' -k4 -rn
done
