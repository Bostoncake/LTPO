"""
Debug script to verify whether latent_thought_tokens actually take effect
during the LTPO v4 pipeline.

Checks:
  1. Thought tokens are correctly located in the tokenised input.
  2. The RL optimisation loop actually modifies the thought embeddings.
  3. The optimised embeddings change the model's generated output.

Usage:
    CUDA_VISIBLE_DEVICES=0 python debug_thought_tokens.py
"""

import os
import json
import torch
import numpy as np
import random
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from ltpo_vl_dmlr_v4 import (
    build_inputs_vl,
    get_system_prompt,
    _thought_token_ids,
    _thought_token_str,
    generate_vl,
)
from ltpo import get_confidence
from reward import RewardModel

# ---------- config ----------
MODEL_PATH = "/export/home/lanliwei.1/abcxyz/storage/models/Qwen2.5-VL-3B-Instruct"
DATASET_JSON = "mllm_data/mmvp_dev.json"
IMAGE_ROOT = "."
DEVICE = "cuda"
NUM_THOUGHT_TOKENS = 2
NUM_SAMPLES = 2          # only run 2 samples for quick debugging
MAX_RL_STEPS = 5         # fewer steps for speed
SEED = 42

# ---------- setup ----------
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

print("=" * 70)
print("Loading model …")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    attn_implementation="eager",
).to(DEVICE)
model.eval()

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    padding_side="left",
    min_pixels=128 * 28 * 28,
    max_pixels=256 * 28 * 28,
)
tokenizer = processor.tokenizer

reward_model = RewardModel(
    model=model,
    tokenizer=tokenizer,
    num_thought_tokens=NUM_THOUGHT_TOKENS,
)

# ---------- load data ----------
with open(DATASET_JSON) as f:
    data = json.load(f)

print(f"Loaded {len(data)} examples; using first {NUM_SAMPLES} for debug.\n")


def load_image(path):
    full = path if os.path.isabs(path) else os.path.join(IMAGE_ROOT, path)
    if not os.path.exists(full):
        print(f"  [WARN] image not found: {full}")
        return None
    return Image.open(full).convert("RGB")


# ================================================================
# Main debug loop
# ================================================================
for sample_idx in range(NUM_SAMPLES):
    ex = data[sample_idx]
    question = ex["prompt"]
    true_answer = ex["solution"]
    image = load_image(ex["image_path"])

    print("=" * 70)
    print(f"[Sample {sample_idx}]")
    print(f"  Question : {question[:120]}…")
    print(f"  Answer   : {true_answer}")
    print()

    data_name = "mmvp_dev"
    model_name = MODEL_PATH

    # ------------------------------------------------------------------
    # CHECK 1: Thought-token string & IDs
    # ------------------------------------------------------------------
    thought_str = _thought_token_str(model_name, NUM_THOUGHT_TOKENS)
    thought_ids = _thought_token_ids(tokenizer, model_name, NUM_THOUGHT_TOKENS)
    print(f"  [CHECK 1] Thought token string : {repr(thought_str)}")
    print(f"  [CHECK 1] Thought token IDs    : {thought_ids}")

    # ------------------------------------------------------------------
    # CHECK 2: build_inputs_vl — thought_idx correctness
    # ------------------------------------------------------------------
    inputs, thought_idx = build_inputs_vl(
        processor=processor,
        model=model,
        image=image,
        num_thought_tokens=NUM_THOUGHT_TOKENS,
        prompt=question,
        data_name=data_name,
        model_name=model_name,
    )

    print(f"  [CHECK 2] thought_idx          : {thought_idx}")
    print(f"  [CHECK 2] inputs_embeds shape  : {inputs['inputs_embeds'].shape}")
    print(f"  [CHECK 2] attention_mask shape  : {inputs['attention_mask'].shape}")

    # Verify the identified region is not all zeros
    region = inputs["inputs_embeds"][0, thought_idx[0]:thought_idx[1]]
    print(f"  [CHECK 2] thought region norm  : {region.norm().item():.6f}")
    print(f"  [CHECK 2] thought region mean  : {region.mean().item():.6f}")
    print(f"  [CHECK 2] thought region std   : {region.std().item():.6f}")
    print()

    # ------------------------------------------------------------------
    # CHECK 3: RL loop — does optimisation change the embeddings?
    # ------------------------------------------------------------------
    inputs_embeds = inputs["inputs_embeds"]
    original_thought = inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone()
    thought_hidden = original_thought.clone()

    sigma = 25.0
    lr = 0.01
    sigma_decay = 0.95

    print(f"  [CHECK 3] Running {MAX_RL_STEPS} RL steps …")
    rewards = []
    embedding_diffs = []

    for step in range(MAX_RL_STEPS):
        epsilon = torch.normal(mean=0.0, std=sigma, size=thought_hidden.shape).to(DEVICE)
        cand = thought_hidden + epsilon

        with torch.no_grad():
            reward = get_confidence(
                model=model,
                inputs=inputs,
                thought_idx=thought_idx,
                thought_hidden_states=cand,
                k=10,
            )
        rewards.append(float(reward))

        grad_ascent = lr * reward * epsilon / sigma ** 2
        thought_hidden = thought_hidden + grad_ascent
        sigma *= sigma_decay

        diff = (thought_hidden - original_thought).norm().item()
        embedding_diffs.append(diff)

        print(f"    step {step}: reward={float(reward):.6f}  "
              f"embed_diff_from_init={diff:.6f}  sigma={sigma:.4f}")

    print()
    print(f"  [CHECK 3] Rewards over steps   : {[f'{r:.4f}' for r in rewards]}")
    print(f"  [CHECK 3] Embed diff from init  : {[f'{d:.4f}' for d in embedding_diffs]}")
    print(f"  [CHECK 3] Final diff norm       : {embedding_diffs[-1]:.6f}")
    embeds_changed = embedding_diffs[-1] > 1e-6
    print(f"  [CHECK 3] Embeddings changed?   : {'YES ✓' if embeds_changed else 'NO ✗'}")
    print()

    # ------------------------------------------------------------------
    # CHECK 4: Does the changed embedding affect generation?
    # Compare output with original (un-optimised) vs optimised embeddings.
    # ------------------------------------------------------------------
    print(f"  [CHECK 4] Generating with ORIGINAL thought embeddings …")
    inputs_embeds_orig = inputs["inputs_embeds"].clone()
    inputs_embeds_orig[0, thought_idx[0]:thought_idx[1]] = original_thought
    inputs_orig = {
        "inputs_embeds": inputs_embeds_orig,
        "attention_mask": inputs["attention_mask"],
    }
    with torch.no_grad():
        out_orig = model.generate(
            **inputs_orig, max_new_tokens=256, do_sample=False,
            temperature=0.0, top_p=None, num_beams=1,
        )
    text_orig = tokenizer.decode(out_orig[0], skip_special_tokens=True)

    print(f"  [CHECK 4] Generating with OPTIMISED thought embeddings …")
    inputs_embeds_opt = inputs["inputs_embeds"].clone()
    inputs_embeds_opt[0, thought_idx[0]:thought_idx[1]] = thought_hidden
    inputs_opt = {
        "inputs_embeds": inputs_embeds_opt,
        "attention_mask": inputs["attention_mask"],
    }
    with torch.no_grad():
        out_opt = model.generate(
            **inputs_opt, max_new_tokens=256, do_sample=False,
            temperature=0.0, top_p=None, num_beams=1,
        )
    text_opt = tokenizer.decode(out_opt[0], skip_special_tokens=True)

    # Also generate with RANDOM embeddings as a sanity check
    print(f"  [CHECK 4] Generating with RANDOM thought embeddings …")
    random_thought = torch.randn_like(original_thought) * original_thought.std()
    inputs_embeds_rnd = inputs["inputs_embeds"].clone()
    inputs_embeds_rnd[0, thought_idx[0]:thought_idx[1]] = random_thought
    inputs_rnd = {
        "inputs_embeds": inputs_embeds_rnd,
        "attention_mask": inputs["attention_mask"],
    }
    with torch.no_grad():
        out_rnd = model.generate(
            **inputs_rnd, max_new_tokens=256, do_sample=False,
            temperature=0.0, top_p=None, num_beams=1,
        )
    text_rnd = tokenizer.decode(out_rnd[0], skip_special_tokens=True)

    print()
    print(f"  --- ORIGINAL output (first 300 chars) ---")
    print(f"  {text_orig[:300]}")
    print(f"  --- OPTIMISED output (first 300 chars) ---")
    print(f"  {text_opt[:300]}")
    print(f"  --- RANDOM output (first 300 chars) ---")
    print(f"  {text_rnd[:300]}")
    print()

    outputs_differ = text_orig != text_opt
    random_differs = text_orig != text_rnd
    print(f"  [CHECK 4] Original vs Optimised differ? : {'YES ✓' if outputs_differ else 'NO — same output ✗'}")
    print(f"  [CHECK 4] Original vs Random differ?    : {'YES ✓' if random_differs else 'NO — same output ✗'}")

    # ------------------------------------------------------------------
    # CHECK 5: Confidence reward values — do they actually vary?
    # ------------------------------------------------------------------
    print()
    print(f"  [CHECK 5] Confidence with original embeddings …")
    with torch.no_grad():
        conf_orig = get_confidence(
            model=model, inputs=inputs, thought_idx=thought_idx,
            thought_hidden_states=original_thought, k=10,
        )
    print(f"    confidence (original)  = {float(conf_orig):.6f}")

    with torch.no_grad():
        conf_opt = get_confidence(
            model=model, inputs=inputs, thought_idx=thought_idx,
            thought_hidden_states=thought_hidden, k=10,
        )
    print(f"    confidence (optimised) = {float(conf_opt):.6f}")

    with torch.no_grad():
        conf_rnd = get_confidence(
            model=model, inputs=inputs, thought_idx=thought_idx,
            thought_hidden_states=random_thought, k=10,
        )
    print(f"    confidence (random)    = {float(conf_rnd):.6f}")

    conf_varies = not (float(conf_orig) == float(conf_opt) == float(conf_rnd))
    print(f"  [CHECK 5] Confidence varies across embeddings? : {'YES ✓' if conf_varies else 'NO ✗'}")

    print()
    print("=" * 70)

    # clean up GPU memory
    del inputs, inputs_embeds, inputs_orig, inputs_opt, inputs_rnd
    del out_orig, out_opt, out_rnd
    torch.cuda.empty_cache()

print("\n\n>>> DEBUG SUMMARY <<<")
print("If CHECK 3 shows embeddings changed and CHECK 4 shows different outputs,")
print("then latent_thought_tokens ARE taking effect during the LTPO pipeline.")
print("If outputs are identical regardless of embeddings, the thought tokens")
print("are NOT influencing generation (they may be ignored by the model).")
