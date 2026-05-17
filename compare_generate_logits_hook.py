"""
compare_generate_logits_hook.py — Three-way comparison of generate() on a
multimodal prompt that has latent thought tokens inserted in the assistant
turn:

  A) ids_pixels :  model.generate(**inputs)                     [reference]
  B) embeds     :  model.generate(inputs_embeds=merged, attention_mask=...)
  C) hook       :  model.generate(**inputs) + forward pre-hook on
                   model.language_model that overwrites the thought-token
                   rows of `inputs_embeds` after the official multimodal
                   prep is done.

Why this exists
---------------
Approach B is what the current LTPO path uses so it can plug optimised
latent embeddings into the thought-token slots. But generate(inputs_embeds=...)
loses access to `input_ids` and `image_grid_thw`, which Qwen2.5-VL needs
to compute M-RoPE position_ids and `rope_deltas`. The fallback in
`get_rope_index` collapses to a 1D RoPE, so the prefill positions are
wrong and the first generated token diverges from Approach A.

Approach C keeps the official `generate(**inputs)` path intact (M-RoPE,
image-token scatter, rope_deltas cache, attention mask, etc.) and only
hooks in to overwrite the thought-token rows of `inputs_embeds`
immediately before the inner language_model decoder runs. For the
identity overwrite used here (i.e. writing the SAME default `<|endoftext|>`
embedding the input_ids path would produce), Approach C should be
byte-equivalent to Approach A.

Pre-processing mirrors the baseline branch of main_vl_dmlr_direct_boxed.py
(--eval_baseline --baseline_with_thought_tokens --num_thought_tokens 2):
the user turn carries the raw question and the assistant turn becomes
"<thought_tokens>\\n\\boxed{".

Outputs
-------
- Per-step next-token logits captured via output_scores=True for all
  three paths, with pairwise diff stats and step-0 top-k drill-down.
- End-to-end wall-clock timing for each path (warmup + N timed runs),
  with output_scores disabled so the timings reflect plain generation.
"""

import argparse
import os
import time

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
        "--dataset", type=str, required=True, choices=SUPPORTED_DATASETS,
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
    p.add_argument("--num_thought_tokens", type=int, default=2)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=4,
        help="Generation steps to capture for the logit comparison.",
    )
    p.add_argument(
        "--time_max_new_tokens",
        type=int,
        default=64,
        help="Generation steps for the timing benchmark (longer runs hide hook overhead better).",
    )
    p.add_argument(
        "--time_runs",
        type=int,
        default=3,
        help="Number of timed runs per approach after one warmup run.",
    )
    p.add_argument(
        "--skip_timing",
        action="store_true",
        help="Skip the speed benchmark and just run the logit comparison.",
    )
    return p.parse_args()


def load_image(image_path, image_root):
    if not image_path:
        return None
    path = image_path if os.path.isabs(image_path) else os.path.join(image_root, image_path)
    if not os.path.exists(path):
        print(f"[WARN] Image not found at {path}")
        return None
    return Image.open(path).convert("RGB")


def build_baseline_inputs(processor, image, question, model_name, num_thought_tokens, device):
    """Same prompt construction as main_vl_dmlr_direct_boxed.py's baseline branch."""
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
    """Aggregate stats. Tensors must have shape (n_positions, vocab)."""
    assert logits_a.shape == logits_b.shape, (logits_a.shape, logits_b.shape)
    diff = logits_a.float() - logits_b.float()
    abs_diff = diff.abs()

    cos = torch.nn.functional.cosine_similarity(
        logits_a.float().flatten(), logits_b.float().flatten(), dim=0
    ).item()

    log_probs_a = torch.log_softmax(logits_a.float(), dim=-1)
    log_probs_b = torch.log_softmax(logits_b.float(), dim=-1)
    probs_a = log_probs_a.exp()
    probs_b = log_probs_b.exp()
    kl_ab = (probs_a * (log_probs_a - log_probs_b)).sum(dim=-1)
    kl_ba = (probs_b * (log_probs_b - log_probs_a)).sum(dim=-1)

    argmax_a = logits_a.argmax(dim=-1)
    argmax_b = logits_b.argmax(dim=-1)
    match = (argmax_a == argmax_b)

    return {
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "cosine_similarity": cos,
        "kl_ab_mean": kl_ab.mean().item(),
        "kl_ba_mean": kl_ba.mean().item(),
        "kl_ab_max": kl_ab.max().item(),
        "kl_ba_max": kl_ba.max().item(),
        "argmax_match_ratio": match.float().mean().item(),
        "argmax_match_count": int(match.sum().item()),
        "num_positions": int(logits_a.shape[0]),
    }


def print_topk(logits_row: torch.Tensor, tokenizer, k: int, title: str):
    probs = torch.softmax(logits_row.float(), dim=-1)
    topk = torch.topk(probs, k=k)
    print(f"  {title}:")
    for rank, (prob, tid) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
        print(f"    [{rank:2d}] id={tid:6d}  p={prob:.6f}  text={tokenizer.decode([tid])!r}")


# ---------------------------------------------------------------------------
# The hook: a forward pre-hook on the inner language_model that overwrites
# only the thought-token rows of `inputs_embeds`. Fires for every forward
# call (prefill + each decode step), but only modifies the prefill call
# (seq_len > 1).
# ---------------------------------------------------------------------------

class ThoughtEmbedInjector:
    def __init__(self, thought_start: int, thought_end: int, thought_embeds: torch.Tensor):
        """
        thought_embeds: (num_thought_tokens, d_model). The hook writes these
        rows into inputs_embeds[:, thought_start:thought_end] whenever a
        multi-token (prefill) forward pass is intercepted.
        """
        self.thought_start = thought_start
        self.thought_end = thought_end
        self.thought_embeds = thought_embeds
        self.prefill_hits = 0
        self.decode_hits = 0

    def __call__(self, module, args, kwargs):
        inputs_embeds = kwargs.get("inputs_embeds", None)
        if inputs_embeds is None or inputs_embeds.shape[1] <= 1:
            # Decode-step (single new token) — leave it alone.
            self.decode_hits += 1 if inputs_embeds is not None else 0
            return args, kwargs

        # Prefill — splice the thought rows in.
        modified = inputs_embeds.clone()
        te = self.thought_embeds
        if te.dim() == 2:
            te = te.unsqueeze(0)  # (1, num_thought, d)
        modified[:, self.thought_start:self.thought_end] = te.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        kwargs["inputs_embeds"] = modified
        self.prefill_hits += 1
        return args, kwargs


# ---------------------------------------------------------------------------
# Per-approach generate wrappers. Each returns (sequences, scores_or_None).
# `scores` is a stacked (n_steps, vocab) tensor when output_scores=True;
# otherwise None.
# ---------------------------------------------------------------------------

def gen_ids_pixels(model, inputs, max_new_tokens, want_scores):
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=None,
            num_beams=1,
            output_scores=want_scores,
            return_dict_in_generate=want_scores,
        )
    if want_scores:
        return out.sequences, torch.stack(list(out.scores), dim=0)[:, 0, :]
    return out, None


def gen_embeds(model, inputs_embeds, attention_mask, max_new_tokens, want_scores):
    with torch.no_grad():
        out = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=None,
            num_beams=1,
            output_scores=want_scores,
            return_dict_in_generate=want_scores,
        )
    if want_scores:
        return out.sequences, torch.stack(list(out.scores), dim=0)[:, 0, :]
    return out, None


def gen_hook(model, inputs, injector, max_new_tokens, want_scores):
    handle = model.language_model.register_forward_pre_hook(injector, with_kwargs=True)
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=None,
                num_beams=1,
                output_scores=want_scores,
                return_dict_in_generate=want_scores,
            )
    finally:
        handle.remove()
    if want_scores:
        return out.sequences, torch.stack(list(out.scores), dim=0)[:, 0, :]
    return out, None


def time_fn(fn, n_runs: int, n_warmup: int = 1):
    """Run fn (zero args) n_warmup + n_runs times, return list of n_runs durations (s)."""
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


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

    inputs, _ = build_baseline_inputs(
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

    thought_ids = _thought_token_ids(
        tokenizer, args.model_name_or_path, args.num_thought_tokens
    )
    thought_start = _find_thought_token_start(input_ids[0].tolist(), thought_ids)
    thought_end = thought_start + args.num_thought_tokens
    print(f"Thought token range: [{thought_start}, {thought_end})  ids={thought_ids}")

    # Pre-merged inputs_embeds: used by Approach B AND to source the default
    # thought-token embeddings for the identity-hook check in Approach C.
    with torch.no_grad():
        inputs_embeds = _merge_visual_tokens(
            model, input_ids, inputs, args.model_name_or_path
        )
    print(f"Merged inputs_embeds shape: {tuple(inputs_embeds.shape)}")

    # For the comparison we use IDENTITY overwrite values — the same
    # default <|endoftext|> embeddings the input_ids path would produce.
    # In real LTPO use, replace `thought_embeds` with optimised latents.
    identity_thought_embeds = inputs_embeds[0, thought_start:thought_end].clone()
    injector_identity = ThoughtEmbedInjector(
        thought_start, thought_end, identity_thought_embeds
    )

    # ========================================================================
    # Logit comparison (output_scores=True)
    # ========================================================================
    N = args.max_new_tokens

    print(f"\n>>> Approach A (ids_pixels): generate(**inputs), output_scores=True, "
          f"max_new_tokens={N}")
    seq_a, scores_a = gen_ids_pixels(model, inputs, N, want_scores=True)
    new_a = seq_a[0, seq_len:].tolist()
    print(f"  steps captured: {scores_a.shape[0]}")
    print(f"  new ids:        {new_a}")
    print(f"  decoded:        {tokenizer.decode(new_a, skip_special_tokens=False)!r}")

    print(f"\n>>> Approach B (embeds): generate(inputs_embeds=...), output_scores=True, "
          f"max_new_tokens={N}")
    seq_b, scores_b = gen_embeds(model, inputs_embeds, attention_mask, N, want_scores=True)
    new_b = seq_b[0].tolist()  # generate(inputs_embeds=...) returns only new tokens
    print(f"  steps captured: {scores_b.shape[0]}")
    print(f"  new ids:        {new_b}")
    print(f"  decoded:        {tokenizer.decode(new_b, skip_special_tokens=False)!r}")

    print(f"\n>>> Approach C (hook): generate(**inputs)+pre-hook on language_model, "
          f"output_scores=True, max_new_tokens={N}")
    # Reset hit counters for a clean reading.
    injector_identity.prefill_hits = 0
    injector_identity.decode_hits = 0
    seq_c, scores_c = gen_hook(model, inputs, injector_identity, N, want_scores=True)
    new_c = seq_c[0, seq_len:].tolist()
    print(f"  steps captured: {scores_c.shape[0]}")
    print(f"  new ids:        {new_c}")
    print(f"  decoded:        {tokenizer.decode(new_c, skip_special_tokens=False)!r}")
    print(f"  hook fired:     prefill={injector_identity.prefill_hits}, "
          f"decode={injector_identity.decode_hits}")

    n_steps = min(scores_a.shape[0], scores_b.shape[0], scores_c.shape[0])
    scores_a = scores_a[:n_steps]
    scores_b = scores_b[:n_steps]
    scores_c = scores_c[:n_steps]

    def pairwise(label, x, y):
        print(f"\n--- {label} ---")
        stats = summarise_logit_diff(x, y)
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k:24s}: {v:.6e}")
            else:
                print(f"  {k:24s}: {v}")

    print("\n" + "=" * 80)
    print(f"PAIRWISE LOGIT COMPARISON  (n_steps={n_steps})")
    print("=" * 80)
    pairwise("A (ids_pixels)  vs  B (embeds)", scores_a, scores_b)
    pairwise("A (ids_pixels)  vs  C (hook)",   scores_a, scores_c)
    pairwise("B (embeds)      vs  C (hook)",   scores_b, scores_c)

    # Per-step argmax table for quick visual sanity check.
    print("\n  Per-step argmax (A / B / C):")
    for s in range(n_steps):
        am_a = int(scores_a[s].argmax().item())
        am_b = int(scores_b[s].argmax().item())
        am_c = int(scores_c[s].argmax().item())
        ac_match = "MATCH" if am_a == am_c else "DIFF "
        ab_match = "MATCH" if am_a == am_b else "DIFF "
        print(
            f"    step {s}:  A={am_a:6d} ({tokenizer.decode([am_a])!r:>6s})  "
            f"B={am_b:6d} ({tokenizer.decode([am_b])!r:>6s}) [A↔B {ab_match}]  "
            f"C={am_c:6d} ({tokenizer.decode([am_c])!r:>6s}) [A↔C {ac_match}]"
        )

    print("\n" + "=" * 80)
    print("STEP-0 TOP-K  (first generated token, by approach)")
    print("=" * 80)
    print_topk(scores_a[0], tokenizer, args.topk, "Approach A (ids_pixels)")
    print_topk(scores_b[0], tokenizer, args.topk, "Approach B (embeds)")
    print_topk(scores_c[0], tokenizer, args.topk, "Approach C (hook)")

    # ========================================================================
    # Speed benchmark (output_scores disabled for clean timing)
    # ========================================================================
    if args.skip_timing:
        print("\n[skip_timing] Done.")
        return

    Nt = args.time_max_new_tokens
    R = args.time_runs
    print("\n" + "=" * 80)
    print(f"SPEED BENCHMARK  (max_new_tokens={Nt}, runs={R}, warmup=1)")
    print("=" * 80)

    # Re-create an injector with fresh counters for timing.
    injector_for_timing = ThoughtEmbedInjector(
        thought_start, thought_end, identity_thought_embeds
    )

    print("Timing Approach A (ids_pixels) ...")
    t_a = time_fn(lambda: gen_ids_pixels(model, inputs, Nt, want_scores=False), n_runs=R)
    print("Timing Approach B (embeds) ...")
    t_b = time_fn(
        lambda: gen_embeds(model, inputs_embeds, attention_mask, Nt, want_scores=False),
        n_runs=R,
    )
    print("Timing Approach C (hook) ...")
    t_c = time_fn(
        lambda: gen_hook(model, inputs, injector_for_timing, Nt, want_scores=False),
        n_runs=R,
    )

    def fmt(times):
        mean = sum(times) / len(times)
        return f"mean={mean:.3f}s  min={min(times):.3f}s  max={max(times):.3f}s  runs={times}"

    print(f"  A (ids_pixels):  {fmt(t_a)}")
    print(f"  B (embeds):      {fmt(t_b)}")
    print(f"  C (hook):        {fmt(t_c)}")

    mean_a = sum(t_a) / len(t_a)
    mean_c = sum(t_c) / len(t_c)
    overhead_ms = (mean_c - mean_a) * 1000
    pct = (mean_c / mean_a - 1.0) * 100
    print(
        f"\n  Hook overhead vs A: Δ={overhead_ms:+.1f} ms  "
        f"({pct:+.2f}% per generate call, includes one prefill clone+scatter)"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
