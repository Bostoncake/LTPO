"""
compare_generate_logits.py — Compare next-token logits between two ways of
feeding a multimodal prompt (with inserted thought tokens) into the model:

  A) input_ids + pixel_values path:
        outputs = model(**inputs)        # inputs has input_ids/pixel_values/...
  B) merged inputs_embeds path:
        inputs_embeds = _merge_visual_tokens(model, input_ids, inputs, ...)
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=...)

Pre-processing mirrors the baseline branch of main_vl_dmlr_direct_boxed.py
when called with --eval_baseline --baseline_with_thought_tokens
--num_thought_tokens 2: the user turn is the raw question, and the
assistant turn becomes "<thought_tokens>\n\\boxed{".

Usage example:

    CUDA_VISIBLE_DEVICES=0 python compare_generate_logits.py \
        --dataset      mmvp_dev \
        --data_idx     0 \
        --model_name_or_path /WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct \
        --num_thought_tokens 2

Both paths are exercised through model.generate(..., output_scores=True,
return_dict_in_generate=True) — so Approach A is byte-identical with the
generate call used by main_vl_dmlr_direct_boxed.py:561-568 (which is what
the baseline branch actually runs). The per-step next-token logits are
captured and compared.
"""

import argparse
import os

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

from data_vl import get_mllm_dataset
from ltpo_vl_dmlr_direct_boxed import SYSTEM_PROMPT, ASSISTANT_BOXED_PREFIX
from ltpo_vl_dmlr import (
    _thought_token_str,
    _thought_token_ids,
    _find_thought_token_start,
    _merge_visual_tokens,
)


SUPPORTED_DATASETS = [
    "mmvp_dev",
    "mmstar_dev",
    "mm_math_dev",
    "math_vista_dev",
    "math_vision_dev",
    "hallusion_dev",
    "scienceqa_dev",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=SUPPORTED_DATASETS,
        help="One of the 7 dev sets.",
    )
    p.add_argument("--data_idx", type=int, required=True, help="Index inside the dev set.")
    p.add_argument(
        "--model_name_or_path",
        type=str,
        default="/WillDevExt/xiongyizhe/models/Qwen2.5-VL-7B-Instruct",
    )
    p.add_argument("--data_root", type=str, default="mllm_data")
    p.add_argument("--image_root", type=str, default=".")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--min_pixels", type=int, default=128)
    p.add_argument("--max_pixels", type=int, default=256)
    p.add_argument("--num_thought_tokens", type=int, default=4)
    p.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-k tokens to print for the next-token distribution at the last position.",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=4,
        help=(
            "How many generation steps to run under each path. The per-step "
            "next-token logits are captured via output_scores=True and "
            "compared. Step 0 is the first generated token."
        ),
    )
    return p.parse_args()


def load_image(image_path: str, image_root: str):
    if not image_path:
        return None
    path = image_path if os.path.isabs(image_path) else os.path.join(image_root, image_path)
    if not os.path.exists(path):
        print(f"[WARN] Image not found at {path}")
        return None
    return Image.open(path).convert("RGB")


def build_baseline_inputs(processor, image, question, model_name, num_thought_tokens, device):
    """Mirror the --eval_baseline --baseline_with_thought_tokens branch."""
    assistant_suffix = (
        _thought_token_str(model_name, num_thought_tokens)
        + "\n"
        + ASSISTANT_BOXED_PREFIX
    )

    if image is not None:
        if "qwen" in model_name.lower():
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            text = text + assistant_suffix
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        else:
            text = f"<image>\n{question}" + assistant_suffix
            inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        text = text + assistant_suffix
        inputs = processor(text=[text], return_tensors="pt").to(device)

    return inputs, text


def summarise_logit_diff(logits_a: torch.Tensor, logits_b: torch.Tensor) -> dict:
    """Per-position summary stats. Both tensors should have shape (seq_len, vocab_size)."""
    assert logits_a.shape == logits_b.shape, (logits_a.shape, logits_b.shape)
    diff = (logits_a.float() - logits_b.float())
    abs_diff = diff.abs()

    a_flat = logits_a.float().flatten()
    b_flat = logits_b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=0).item()

    log_probs_a = torch.log_softmax(logits_a.float(), dim=-1)
    log_probs_b = torch.log_softmax(logits_b.float(), dim=-1)
    probs_a = log_probs_a.exp()
    # Symmetric KL = KL(a||b) + KL(b||a), averaged across positions.
    kl_ab = (probs_a * (log_probs_a - log_probs_b)).sum(dim=-1)
    probs_b = log_probs_b.exp()
    kl_ba = (probs_b * (log_probs_b - log_probs_a)).sum(dim=-1)

    argmax_a = logits_a.argmax(dim=-1)
    argmax_b = logits_b.argmax(dim=-1)
    argmax_match = (argmax_a == argmax_b)

    return {
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "cosine_similarity": cos,
        "kl_ab_mean": kl_ab.mean().item(),
        "kl_ba_mean": kl_ba.mean().item(),
        "kl_ab_max": kl_ab.max().item(),
        "kl_ba_max": kl_ba.max().item(),
        "argmax_match_ratio": argmax_match.float().mean().item(),
        "argmax_match_count": int(argmax_match.sum().item()),
        "num_positions": int(logits_a.shape[0]),
    }


def print_topk(logits_row: torch.Tensor, tokenizer, k: int, title: str):
    probs = torch.softmax(logits_row.float(), dim=-1)
    topk = torch.topk(probs, k=k)
    print(f"  {title}:")
    for rank, (p, tid) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
        tok_str = tokenizer.decode([tid])
        print(f"    [{rank:2d}] id={tid:6d}  p={p:.6f}  text={tok_str!r}")


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.model_name_or_path} ...")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",
        token=os.environ.get("HUGGING_FACE_TOKEN"),
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        padding_side="left",
        min_pixels=args.min_pixels * 28 * 28,
        max_pixels=args.max_pixels * 28 * 28,
        token=os.environ.get("HUGGING_FACE_TOKEN"),
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    dataset = get_mllm_dataset(args.dataset, data_root=args.data_root)
    if not (0 <= args.data_idx < len(dataset)):
        raise IndexError(
            f"data_idx={args.data_idx} out of range for '{args.dataset}' (len={len(dataset)})."
        )

    example = dataset[args.data_idx]
    question = example["question"]
    true_answer = example["answer"]
    image = load_image(example["image_path"], args.image_root)

    print("─" * 80)
    print(f"Dataset:        {args.dataset}")
    print(f"Data index:     {args.data_idx}  (size={len(dataset)})")
    print(f"Image path:     {example['image_path']}")
    print(f"Question:       {question[:200]}{'…' if len(question) > 200 else ''}")
    print(f"True answer:    {true_answer}")
    print(f"Thought tokens: {args.num_thought_tokens}")
    print("─" * 80)

    # ---- Build baseline-style inputs ----
    inputs, text = build_baseline_inputs(
        processor=processor,
        image=image,
        question=question,
        model_name=args.model_name_or_path,
        num_thought_tokens=args.num_thought_tokens,
        device=device,
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get(
        "attention_mask", torch.ones(input_ids.shape, device=device)
    )
    seq_len = input_ids.shape[1]
    print(f"Input seq_len: {seq_len}")

    # Locate thought-token positions for context.
    thought_ids = _thought_token_ids(
        tokenizer, args.model_name_or_path, args.num_thought_tokens
    )
    thought_start = _find_thought_token_start(input_ids[0].tolist(), thought_ids)
    thought_end = thought_start + args.num_thought_tokens
    print(f"Thought token range: [{thought_start}, {thought_end})  ids={thought_ids}")

    # Pre-merge visual tokens for Approach B.
    with torch.no_grad():
        inputs_embeds = _merge_visual_tokens(
            model, input_ids, inputs, args.model_name_or_path
        )
    print(f"Merged inputs_embeds shape: {tuple(inputs_embeds.shape)}")

    # ============================================================
    # Approach A: model.generate(**inputs, ...)
    # Mirrors main_vl_dmlr_direct_boxed.py:561-568 exactly. We add
    # output_scores=True / return_dict_in_generate=True so we can
    # inspect the per-step next-token logits Generate actually uses.
    # ============================================================
    print(
        f"\n>>> Approach A: model.generate(**inputs, max_new_tokens={args.max_new_tokens}, "
        f"output_scores=True)"
    )
    with torch.no_grad():
        gen_a = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=None,
            num_beams=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
    # gen_a.scores: tuple of length n_steps, each (batch=1, vocab).
    scores_a = torch.stack(list(gen_a.scores), dim=0)[:, 0, :]  # (steps, vocab)
    # generate(**inputs) returns the full sequence (prompt + new tokens).
    new_ids_a = gen_a.sequences[0, seq_len:].tolist()
    print(f"  steps captured: {scores_a.shape[0]}")
    print(f"  new ids:        {new_ids_a}")
    print(f"  decoded:        {tokenizer.decode(new_ids_a, skip_special_tokens=False)!r}")

    # ============================================================
    # Approach B: model.generate(inputs_embeds=..., attention_mask=...)
    # ============================================================
    print(
        f"\n>>> Approach B: model.generate(inputs_embeds=..., "
        f"max_new_tokens={args.max_new_tokens}, output_scores=True)"
    )
    with torch.no_grad():
        gen_b = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=None,
            num_beams=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
    scores_b = torch.stack(list(gen_b.scores), dim=0)[:, 0, :]  # (steps, vocab)
    # generate(inputs_embeds=...) returns only the new tokens.
    new_ids_b = gen_b.sequences[0].tolist()
    print(f"  steps captured: {scores_b.shape[0]}")
    print(f"  new ids:        {new_ids_b}")
    print(f"  decoded:        {tokenizer.decode(new_ids_b, skip_special_tokens=False)!r}")

    # Truncate to a common step count for the diff summary.
    n_steps = min(scores_a.shape[0], scores_b.shape[0])
    scores_a = scores_a[:n_steps]
    scores_b = scores_b[:n_steps]

    # ============================================================
    # Per-step logit comparison.
    # Step 0 is the distribution that picks the FIRST generated token.
    # ============================================================
    print("\n" + "=" * 80)
    print(f"LOGIT COMPARISON ACROSS GENERATED STEPS  (n_steps={n_steps})")
    print("=" * 80)
    overall_stats = summarise_logit_diff(scores_a, scores_b)
    for k, v in overall_stats.items():
        if isinstance(v, float):
            print(f"  {k:24s}: {v:.6e}")
        else:
            print(f"  {k:24s}: {v}")

    print("\n  Per-step breakdown:")
    for s in range(n_steps):
        argmax_a_s = int(scores_a[s].argmax().item())
        argmax_b_s = int(scores_b[s].argmax().item())
        diff_max = (scores_a[s].float() - scores_b[s].float()).abs().max().item()
        match = "MATCH" if argmax_a_s == argmax_b_s else "DIFF "
        print(
            f"    step {s}:  argmax(A)={argmax_a_s:6d} ({tokenizer.decode([argmax_a_s])!r})  "
            f"argmax(B)={argmax_b_s:6d} ({tokenizer.decode([argmax_b_s])!r})  "
            f"max|Δ|={diff_max:.4e}  [{match}]"
        )

    # ============================================================
    # Step-0 top-k drill-down — the first generated token.
    # This is the position the user reported as differing between
    # the two approaches.
    # ============================================================
    print("\n" + "=" * 80)
    print("STEP-0 TOP-K  (distribution that picks the first generated token)")
    print("=" * 80)
    print_topk(scores_a[0], tokenizer, args.topk, "Approach A top-k")
    print_topk(scores_b[0], tokenizer, args.topk, "Approach B top-k")

    step0_stats = summarise_logit_diff(scores_a[0:1], scores_b[0:1])
    print("\n  Step-0 diff:")
    for k, v in step0_stats.items():
        if isinstance(v, float):
            print(f"    {k:22s}: {v:.6e}")
        else:
            print(f"    {k:22s}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
