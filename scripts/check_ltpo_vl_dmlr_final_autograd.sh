#!/bin/bash
# check_ltpo_vl_dmlr_final_autograd.sh
#
# Smoke-test that the FINAL LTPO variant's --use_auto_grad branch in
# ltpo_vl_dmlr_final.generate_vl actually backprops through the model
# and the optimiser actually updates the latent thought tokens, for
# BOTH init modes:
#
#   - endoftext init (default, hook path)        : tested on mmvp_dev
#       latent thought embeddings come from the embedding table; the
#       forward-pre-hook ThoughtEmbedInjector splices them into the
#       internal inputs_embeds during the prefill call.
#
#   - hidden init    (--use_inputs_embeds path)  : tested on math_vista_dev
#       visual tokens are pre-merged into inputs_embeds, latent rows
#       are re-initialised from the last-layer hidden state before the
#       thought block, and the model is driven from inputs_embeds.
#
# Diagnostics (emitted by generate_vl when verbose>=2 in the
# use_auto_grad branch) appear as `[autograd-debug] ...` lines:
#
#   [autograd-debug] init (...): shape=..., requires_grad=True, ||param||=...
#   [autograd-debug] step i: reward=..., reward.requires_grad=..., reward.grad_fn=...
#   [autograd-debug] step i: ||grad||=..., max|grad|=..., grad_all_zero=...
#   [autograd-debug] step i: ||Δparam||=..., ||param||: prev -> new
#
# PASS criteria (for each case):
#   1. reward.grad_fn is NOT 'None'
#   2. thought_hidden_states.grad is NOT None and grad_all_zero is False
#   3. ||Δparam|| > 0 on at least one step (param actually moved)
#
# FAIL signals:
#   - 'reward.grad_fn=None'        → graph is broken before reward
#   - 'thought_hidden_states.grad IS NONE' or 'grad_all_zero=True'
#                                  → graph is broken between reward and the parameter
#   - '||Δparam||=0.000000e+00' on every step
#                                  → grad reached optimiser but step is a no-op
#
# Usage:
#   bash scripts/check_ltpo_vl_dmlr_final_autograd.sh
#
# Overridable env vars:
#   MODEL  path to model checkpoint
#   GPU    CUDA device index (default 0)

set -u

export HUGGING_FACE_TOKEN=<YOUR_HF_TOKEN>
export OPENAI_API_KEY=<YOUR_OPENAI_KEY>

MODEL=${MODEL:-/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct}
GPU=${GPU:-0}

OUT_ROOT=./output/ltpo_dmlr_final_autograd_check
mkdir -p "${OUT_ROOT}"

# Minimal-cost knobs:
#   --start_data_idx 0 / --end_data_idx 2  → only 2 examples per dataset
#   --max_num_steps 3                      → 3 backprop steps is enough to
#                                            see Δparam move
#   --max_new_tokens 64                    → cap generation length so each
#                                            example finishes fast
#   --verbose 2                            → turn on [autograd-debug] lines
#
# Reward is computed at the first-generated-token position (default),
# NOT at latent positions — i.e. --reward_on_latent_tokens is OMITTED,
# matching the "reward on the output token" scheme the autograd run is
# meant to validate.
COMMON_ARGS=(
    --model_name_or_path "${MODEL}"
    --data_root           mllm_data
    --image_root          .
    --device              cuda
    --seed                42
    --max_new_tokens      64
    --min_pixels          128
    --max_pixels          256
    --sigma               5.0
    --sigma_decay         0.95
    --lr                  1e-3
    --max_num_steps       3
    --top_k               10
    --reward_type         confidence
    --use_baseline_prompt
    --use_auto_grad
    --start_data_idx      0
    --end_data_idx        2
    --verbose             2
    --disable_save_logistics
)

run_case () {
    local tag=$1
    local dataset=$2
    local tokens=$3
    local init=$4

    local init_flag=""
    local embeds_flag=""
    if [ "${init}" = "hidden" ]; then
        init_flag="--ltpo_thought_init_from_hidden"
        embeds_flag="--use_inputs_embeds"
    fi

    local out_dir="${OUT_ROOT}/${tag}"
    mkdir -p "${out_dir}"
    local log="${out_dir}/run.log"

    echo "════════════════════════════════════════════════════════"
    echo "[$(date +%T)] case=${tag}  dataset=${dataset}  tokens=${tokens}  init=${init}"
    echo "    log: ${log}"
    echo "════════════════════════════════════════════════════════"

    CUDA_VISIBLE_DEVICES=${GPU} python main_vl_dmlr_final.py \
        --dataset            "${dataset}" \
        --output_dir         "${out_dir}" \
        --num_thought_tokens "${tokens}" \
        ${init_flag} \
        ${embeds_flag} \
        "${COMMON_ARGS[@]}" \
        > "${log}" 2>&1
    local rc=$?

    echo "    exit code: ${rc}"
    echo "    --- [autograd-debug] lines ---"
    grep -F "[autograd-debug]" "${log}" || echo "    (none found — check verbose/--use_auto_grad)"
    echo

    # PASS/FAIL summary for this case
    local pass=1
    if ! grep -F "[autograd-debug]" "${log}" >/dev/null; then
        echo "    RESULT: FAIL — no autograd-debug lines emitted"
        return
    fi
    if grep -Fq "reward.grad_fn=None" "${log}"; then
        echo "    FAIL signal: reward.grad_fn=None  (graph broken before reward)"
        pass=0
    fi
    if grep -Fq "thought_hidden_states.grad IS NONE" "${log}"; then
        echo "    FAIL signal: param.grad is None  (graph broken before parameter)"
        pass=0
    fi
    if grep -Fq "grad_all_zero=True" "${log}"; then
        echo "    FAIL signal: grad_all_zero=True  (graph reached param but zeroed)"
        pass=0
    fi
    # All Δparam lines being exactly 0 means optimiser did nothing.
    local nonzero_delta
    nonzero_delta=$(grep -oE '\|\|Δparam\|\|=[0-9.eE+-]+' "${log}" \
                    | awk -F'=' '{print $2}' \
                    | awk '$1+0 > 0' | wc -l)
    if [ "${nonzero_delta}" -eq 0 ]; then
        echo "    FAIL signal: no step produced ||Δparam|| > 0"
        pass=0
    fi

    if [ "${pass}" -eq 1 ]; then
        echo "    RESULT: PASS — autograd updates the latent tokens"
    else
        echo "    RESULT: FAIL — see signals above"
    fi
    echo
}

run_case endoftext_mmvp_dev    mmvp_dev        2 endoftext
run_case hidden_math_vista_dev math_vista_dev  2 hidden

echo "════════════════════════════════════════════════════════"
echo "Autograd update verification finished."
echo "Logs under ${OUT_ROOT}/<case>/run.log"
echo "════════════════════════════════════════════════════════"
