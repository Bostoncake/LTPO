export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MODEL_TYPE=qwen-max

# ── Fixed settings ────────────────────────────────────────────────────────────
dataset="hallusion"
model=/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct
image_root=.
max_new_tokens=2048
base_output=./output/gridsearch_hallusion
num_gpus=8

mkdir -p ${base_output}

# ── Hyperparameter grid ───────────────────────────────────────────────────────
# Ranges adapted from Table 6 (Qwen-2.5-7B Instruct row) for a VL task.
#   thought_tokens : {2, 4, 6}           (3)
#   steps          : {10, 15}            (2)
#   sigma          : {0.1, 1, 5}         (3)
#   sigma_decay    : {0.9, 0.95}         (2)
#   lr             : {1e-3,5e-3,1e-2,5e-2} (4)
#   top_k          : {5, 10}             (2)
# Total: 3×2×3×2×4×2 = 288 experiments
# Wall time: 288/8 GPUs × 15 min = 540 min ≈ 9 h  (within 15 h budget)

thought_tokens_list=(2 4 6)
steps_list=(10 15)
sigma_list=(0.1 1 5)
sigma_decay_list=(0.95)
lr_list=(1e-3 1e-2)
topk_list=(10)

# ── Build job list ────────────────────────────────────────────────────────────
jobs=()
for tokens in "${thought_tokens_list[@]}"; do
  for steps in "${steps_list[@]}"; do
    for sigma in "${sigma_list[@]}"; do
      for decay in "${sigma_decay_list[@]}"; do
        for lr in "${lr_list[@]}"; do
          for topk in "${topk_list[@]}"; do
            jobs+=("$tokens $steps $sigma $decay $lr $topk")
          done
        done
      done
    done
  done
done

total=${#jobs[@]}
echo "Total experiments: $total"
echo "Parallel GPUs    : $num_gpus"
echo "Est. wall time   : $(( (total + num_gpus - 1) / num_gpus * 15 )) min"
echo "────────────────────────────────────────────────"

# ── Run in batches of num_gpus ────────────────────────────────────────────────
batch_pids=()
gpu_slot=0

for job in "${jobs[@]}"; do
  read -r tokens steps sigma decay lr topk <<< "$job"

  # Each experiment writes into its own leaf directory so steps/topk
  # collisions are avoided (main_vl.py auto-appends tokens/lr/sigma/decay).
  exp_output="${base_output}"

  echo "[GPU ${gpu_slot}] tokens=${tokens} steps=${steps} sigma=${sigma} decay=${decay} lr=${lr} topk=${topk}"

  CUDA_VISIBLE_DEVICES=${gpu_slot} python main_vl.py \
    --dataset        "$dataset"         \
    --data_root      mllm_data          \
    --image_root     "$image_root"      \
    --model_name_or_path "$model"       \
    --output_dir     "$exp_output"      \
    --device         cuda               \
    --max_new_tokens $max_new_tokens    \
    --max_num_steps  $steps             \
    --num_thought_tokens $tokens        \
    --sigma          $sigma             \
    --sigma_decay    $decay             \
    --lr             $lr                \
    --top_k          $topk              \
    --verbose        0                  \
    > "${exp_output}/tokens${tokens}_lr${lr}_sigma${sigma}_decay${decay}_steps${steps}_topk${topk}.stdout" 2>&1 &

  batch_pids+=($!)
  gpu_slot=$(( gpu_slot + 1 ))

  # When all GPUs are busy, wait for the whole batch before continuing.
  if [ ${#batch_pids[@]} -eq $num_gpus ]; then
    echo "  → waiting for batch of $num_gpus …"
    for pid in "${batch_pids[@]}"; do wait "$pid"; done
    batch_pids=()
    gpu_slot=0
    echo "  → batch done"
  fi
done

# Wait for any leftover jobs in the final partial batch
if [ ${#batch_pids[@]} -gt 0 ]; then
  echo "  → waiting for final batch of ${#batch_pids[@]} …"
  for pid in "${batch_pids[@]}"; do wait "$pid"; done
fi

# ── Summarise results ─────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "Grid search complete. Results (sorted by accuracy):"
echo "════════════════════════════════════════════════"

# Collect every results.log, prepend its parent dir, then sort by accuracy field
find "$base_output" -name "results.log" | while read -r log; do
  dir=$(dirname "$log")
  # Extract last accuracy line (append-mode logs may have multiple entries)
  last=$(tail -1 "$log")
  echo "$last  | $dir"
done | sort -t'=' -k4 -rn
