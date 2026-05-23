"""calibrate_outlier_dims.py

Find the persistent outlier hidden-dim indices for each Qwen-VL model.

VisAttnSink's DimProspector flags "sink tokens" by checking the RMS-norm
of hidden states at a small set of *fixed* outlier dimensions (DIM_SINK).
For LLaMA-2-7B these are dims 2533 and 1415. Qwen has no published table,
so we calibrate them here: feed a few image+text examples through the
text decoder, gather hidden_states from a mid layer, RMS-normalise, and
rank dimensions by their median absolute value across all token positions.
The top-K dims are the persistent "register" / outlier dims.

Usage:
    python calibrate_outlier_dims.py \
        --model_path /path/to/Qwen2.5-VL-7B-Instruct \
        --dataset mmstar_dev --n_samples 8 --topk 2 --layer 12

Prints the top-K outlier dim indices.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_vl import get_mllm_dataset
from ltpo_vl_dmlr import SYSTEM_PROMPT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset", default="mmstar_dev")
    p.add_argument("--data_root", default="mllm_data")
    p.add_argument("--image_root", default=".")
    p.add_argument("--n_samples", type=int, default=8)
    p.add_argument("--layer", type=int, default=-1,
                   help="Which decoder layer's input to inspect. -1 = middle layer.")
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--min_pixels", type=int, default=128)
    p.add_argument("--max_pixels", type=int, default=256)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        padding_side="left",
        min_pixels=args.min_pixels * 28 * 28,
        max_pixels=args.max_pixels * 28 * 28,
    )

    # Locate text decoder layers.
    text_cfg = getattr(model.config, "text_config", model.config)
    n_layers = text_cfg.num_hidden_layers
    layer_idx = args.layer if args.layer >= 0 else n_layers // 2

    text_model = None
    for name in ("model", "language_model", "thinker"):
        cand = getattr(model, name, None)
        if cand is None:
            continue
        # Walk one level for nested wrappers.
        if hasattr(cand, "language_model"):
            cand = cand.language_model
        if hasattr(cand, "model"):
            cand = cand.model
        if hasattr(cand, "layers"):
            text_model = cand
            break
    if text_model is None:
        # Fallback: walk the whole model for a `.layers` ModuleList.
        for _, m in model.named_modules():
            if hasattr(m, "layers") and isinstance(m.layers, torch.nn.ModuleList):
                if len(m.layers) == n_layers:
                    text_model = m
                    break
    assert text_model is not None and hasattr(text_model, "layers"), \
        "could not find text decoder layers"
    layers = text_model.layers
    print(f"Found {len(layers)} decoder layers; sampling layer {layer_idx}")

    captured: dict = {}
    def hook(module, inputs):
        hs = inputs[0] if isinstance(inputs, tuple) and len(inputs) > 0 else inputs
        if isinstance(hs, torch.Tensor):
            captured["hs"] = hs.detach()
    handle = layers[layer_idx].register_forward_pre_hook(hook)

    dataset = get_mllm_dataset(args.dataset, data_root=args.data_root)

    median_abs = None  # accumulator over samples: median |rms| per dim
    count = 0
    for i in range(min(args.n_samples, len(dataset))):
        ex = dataset[i]
        ipath = ex["image_path"]
        if not ipath:
            continue
        path = ipath if os.path.isabs(ipath) else os.path.join(args.image_root, ipath)
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert("RGB")
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": ex["question"]},
            ]},
        ]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[img], return_tensors="pt").to(args.device)

        with torch.inference_mode():
            model(**inputs)

        hs = captured["hs"].to(torch.float32)  # [1, seq, dim]
        var = hs.pow(2).mean(-1, keepdim=True)
        rms = (hs * torch.rsqrt(var + 1e-6)).abs()  # [1, seq, dim]
        # Median across token positions: gives the persistent magnitude per dim.
        med = rms[0].median(dim=0).values  # [dim]
        median_abs = med if median_abs is None else (median_abs + med)
        count += 1

    handle.remove()
    median_abs = median_abs / max(count, 1)

    top_vals, top_idx = median_abs.topk(args.topk)
    print(f"\nModel: {args.model_path}")
    print(f"Layer {layer_idx} / {n_layers}, samples used: {count}")
    print(f"Hidden dim: {median_abs.numel()}")
    print(f"Top-{args.topk} outlier dims (by median |rmsnorm| over tokens):")
    for v, idx in zip(top_vals.tolist(), top_idx.tolist()):
        print(f"  dim={idx:5d}   median|rms|={v:.3f}")
    # Also report next 5 for context.
    next_vals, next_idx = median_abs.topk(args.topk + 5)
    print("Next dims (for sanity):")
    for v, idx in zip(next_vals.tolist()[args.topk:], next_idx.tolist()[args.topk:]):
        print(f"  dim={idx:5d}   median|rms|={v:.3f}")

    # Emit a python literal for direct copy.
    print(f"\nDIM_SINK_ENTRY: {os.path.basename(args.model_path.rstrip('/'))!r}: {top_idx.tolist()},")


if __name__ == "__main__":
    main()
