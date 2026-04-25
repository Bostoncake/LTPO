"""
Debug script v2: test with more aggressive parameters to determine the
threshold at which thought token changes actually affect generation.

Tests multiple configurations:
  - num_thought_tokens: 2, 10, 50
  - RL steps: 15, 50
  - Also: direct replacement with large-magnitude embeddings
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
    _thought_token_ids,
    _thought_token_str,
)
from ltpo import get_confidence

MODEL_PATH = "/export/home/lanliwei.1/abcxyz/storage/models/Qwen2.5-VL-3B-Instruct"
DATASET_JSON = "mllm_data/mmvp_dev.json"
IMAGE_ROOT = "."
DEVICE = "cuda"
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

print("Loading model …")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float32,
    trust_remote_code=True, attn_implementation="eager",
).to(DEVICE)
model.eval()

processor = AutoProcessor.from_pretrained(
    MODEL_PATH, padding_side="left",
    min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28,
)
tokenizer = processor.tokenizer

with open(DATASET_JSON) as f:
    data = json.load(f)

ex = data[0]
question = ex["prompt"]
image_path = ex["image_path"]
full_path = image_path if os.path.isabs(image_path) else os.path.join(IMAGE_ROOT, image_path)
image = Image.open(full_path).convert("RGB")

data_name = "mmvp_dev"
model_name = MODEL_PATH


def generate_text(inputs_dict, max_tokens=256):
    with torch.no_grad():
        out = model.generate(
            **inputs_dict, max_new_tokens=max_tokens, do_sample=False,
            temperature=0.0, top_p=None, num_beams=1,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ==========================================================================
# TEST A: With default 2 tokens, run many more RL steps (50)
# ==========================================================================
print("\n" + "=" * 70)
print("TEST A: 2 thought tokens, 50 RL steps, sigma=25")
print("=" * 70)

inputs, thought_idx = build_inputs_vl(
    processor=processor, model=model, image=image,
    num_thought_tokens=2, prompt=question,
    data_name=data_name, model_name=model_name,
)

original_thought = inputs["inputs_embeds"][0, thought_idx[0]:thought_idx[1]].clone()
thought_hidden = original_thought.clone()
sigma = 25.0
lr = 0.01
best_reward = -999.0
best_thought = thought_hidden.clone()

for step in range(50):
    epsilon = torch.normal(mean=0.0, std=sigma, size=thought_hidden.shape).to(DEVICE)
    cand = thought_hidden + epsilon
    with torch.no_grad():
        reward = get_confidence(model=model, inputs=inputs, thought_idx=thought_idx,
                                thought_hidden_states=cand, k=10)
    grad = lr * reward * epsilon / sigma ** 2
    thought_hidden = thought_hidden + grad
    sigma *= 0.95
    if float(reward) > best_reward:
        best_reward = float(reward)
        best_thought = thought_hidden.clone()
    if step % 10 == 0:
        diff = (thought_hidden - original_thought).norm().item()
        print(f"  step {step:3d}: reward={float(reward):.4f}  diff={diff:.4f}  sigma={sigma:.4f}")

diff_final = (best_thought - original_thought).norm().item()
print(f"  Best reward: {best_reward:.4f}, final diff from init: {diff_final:.4f}")

# Generate with original vs best
ie_orig = inputs["inputs_embeds"].clone()
ie_orig[0, thought_idx[0]:thought_idx[1]] = original_thought
text_orig = generate_text({"inputs_embeds": ie_orig, "attention_mask": inputs["attention_mask"]})

ie_best = inputs["inputs_embeds"].clone()
ie_best[0, thought_idx[0]:thought_idx[1]] = best_thought
text_best = generate_text({"inputs_embeds": ie_best, "attention_mask": inputs["attention_mask"]})

print(f"\n  ORIGINAL (200ch): {text_orig[:200]}")
print(f"  BEST     (200ch): {text_best[:200]}")
print(f"  Differ? {'YES ✓' if text_orig != text_best else 'NO ✗'}")

del inputs
torch.cuda.empty_cache()

# ==========================================================================
# TEST B: Scale up the diff manually — multiply by 10x, 100x
# ==========================================================================
print("\n" + "=" * 70)
print("TEST B: Manually scale optimised diff by 10x and 100x")
print("=" * 70)

inputs2, tidx2 = build_inputs_vl(
    processor=processor, model=model, image=image,
    num_thought_tokens=2, prompt=question,
    data_name=data_name, model_name=model_name,
)
orig2 = inputs2["inputs_embeds"][0, tidx2[0]:tidx2[1]].clone()
delta = best_thought - original_thought  # reuse from TEST A

for scale in [1.0, 10.0, 100.0, 1000.0]:
    scaled = orig2 + delta * scale
    ie = inputs2["inputs_embeds"].clone()
    ie[0, tidx2[0]:tidx2[1]] = scaled
    text = generate_text({"inputs_embeds": ie, "attention_mask": inputs2["attention_mask"]})
    diff_norm = (scaled - orig2).norm().item()
    print(f"\n  scale={scale:7.1f}  diff_norm={diff_norm:.4f}")
    print(f"  Output (200ch): {text[:200]}")
    print(f"  Differ from orig? {'YES ✓' if text != text_orig else 'NO ✗'}")

del inputs2
torch.cuda.empty_cache()

# ==========================================================================
# TEST C: More thought tokens (10)
# ==========================================================================
print("\n" + "=" * 70)
print("TEST C: 10 thought tokens, 30 RL steps")
print("=" * 70)

inputs3, tidx3 = build_inputs_vl(
    processor=processor, model=model, image=image,
    num_thought_tokens=10, prompt=question,
    data_name=data_name, model_name=model_name,
)
orig3 = inputs3["inputs_embeds"][0, tidx3[0]:tidx3[1]].clone()
th3 = orig3.clone()
sigma3 = 25.0

for step in range(30):
    epsilon = torch.normal(mean=0.0, std=sigma3, size=th3.shape).to(DEVICE)
    cand = th3 + epsilon
    with torch.no_grad():
        reward = get_confidence(model=model, inputs=inputs3, thought_idx=tidx3,
                                thought_hidden_states=cand, k=10)
    grad = 0.01 * reward * epsilon / sigma3 ** 2
    th3 = th3 + grad
    sigma3 *= 0.95
    if step % 5 == 0:
        diff = (th3 - orig3).norm().item()
        print(f"  step {step:3d}: reward={float(reward):.4f}  diff={diff:.4f}")

ie_orig3 = inputs3["inputs_embeds"].clone()
ie_orig3[0, tidx3[0]:tidx3[1]] = orig3
text_orig3 = generate_text({"inputs_embeds": ie_orig3, "attention_mask": inputs3["attention_mask"]})

ie_opt3 = inputs3["inputs_embeds"].clone()
ie_opt3[0, tidx3[0]:tidx3[1]] = th3
text_opt3 = generate_text({"inputs_embeds": ie_opt3, "attention_mask": inputs3["attention_mask"]})

print(f"\n  ORIGINAL (200ch): {text_orig3[:200]}")
print(f"  OPTIMISED(200ch): {text_opt3[:200]}")
print(f"  Differ? {'YES ✓' if text_orig3 != text_opt3 else 'NO ✗'}")

# Also try with zeros in thought region
ie_zero = inputs3["inputs_embeds"].clone()
ie_zero[0, tidx3[0]:tidx3[1]] = torch.zeros_like(orig3)
text_zero = generate_text({"inputs_embeds": ie_zero, "attention_mask": inputs3["attention_mask"]})
print(f"  ZEROED   (200ch): {text_zero[:200]}")
print(f"  Zeroed differ from orig? {'YES ✓' if text_zero != text_orig3 else 'NO ✗'}")

del inputs3
torch.cuda.empty_cache()

print("\n>>> DONE <<<")
