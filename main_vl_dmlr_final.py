"""
main_vl_dmlr_final.py — Evaluate LTPO direct-boxed FINAL variant.

Identical to main_vl_dmlr_direct_boxed.py except imports come from
ltpo_vl_dmlr_final, which keeps the official input_ids + pixel_values
path through every model/generate call and uses a forward pre-hook
(``ThoughtEmbedInjector``) to inject LTPO-controlled latents into the
thought-token rows of the internal inputs_embeds. This preserves
Qwen2.5-VL's M-RoPE, image-token scatter, rope_deltas cache and
KV-cache handling — all of which break when generation is driven
from ``inputs_embeds`` alone.

The same hook approach is used by the ``--eval_baseline
--baseline_with_thought_tokens`` branch when one of the re-init flags
(``--baseline_thought_init_from_hidden`` /
``--baseline_thought_init_from_mean``) is set.
"""

import argparse
import os
import random
import re

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from openai import OpenAI

from data_vl import get_mllm_dataset
from ltpo_vl_dmlr_final import (
    generate_vl,
    SYSTEM_PROMPT,
    ASSISTANT_BOXED_PREFIX,
    ThoughtEmbedInjector,
)
from ltpo_vl_dmlr import (
    _thought_token_str,
    _thought_token_ids,
    _find_thought_token_start,
    _merge_visual_tokens,
)


huggingface_token = os.environ.get('HUGGING_FACE_TOKEN')


# ---------------------------------------------------------------------------
# Answer extraction — DMLR-compatible (unchanged)
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str:
    if not text:
        return ""

    try:
        low = text.lower()
        start = low.find("<answer>")
        end = low.find("</answer>")
        if start != -1 and end != -1 and end > start:
            ans = text[start + len("<answer>"):end].strip()
            ans = ans.strip('$')
            ans = re.sub(r'\\displaystyle\s*', '', ans)
            ans = re.sub(r'\s+', ' ', ans).strip()
            if ans:
                return ans

        boxed_contents = []
        for m in re.finditer(r'\\boxed\s*\{', text):
            open_brace_pos = text.find('{', m.end() - 1)
            if open_brace_pos == -1:
                continue
            depth = 0
            i = open_brace_pos
            while i < len(text):
                ch = text[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        boxed = text[open_brace_pos + 1:i]
                        boxed = boxed.strip().strip('$')
                        boxed = re.sub(r'\\displaystyle\s*', '', boxed)
                        b = boxed.strip()
                        if b.startswith(r'\text{') and b.endswith('}'):
                            b = b[len(r'\text{'):-1].strip()
                        boxed = re.sub(r'\s+', ' ', b).strip()
                        if boxed:
                            boxed_contents.append(boxed)
                        break
                i += 1
        if boxed_contents:
            return boxed_contents[-1]

    except Exception:
        pass

    return text.strip()


# ---------------------------------------------------------------------------
# Verification — DMLR-compatible (unchanged)
# ---------------------------------------------------------------------------

_llm_client: OpenAI | None = None
_llm_model: str | None = None


def _get_llm_client() -> OpenAI:
    global _llm_client, _llm_model
    if _llm_client is not None:
        return _llm_client
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_API_BASE_URL")
    _llm_model = os.environ.get("MODEL_TYPE", "gpt-4o-2024-08-06")
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    _llm_client = OpenAI(**kwargs)
    return _llm_client


def verify_solution_equivalence(solution: str, ground_truth: str) -> bool:
    if not solution or not ground_truth:
        return False

    class EquivalenceResult(BaseModel):
        equivalent: bool

    client = _get_llm_client()
    model = _llm_model or "gpt-4o-2024-08-06"
    try:
        resp = client.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Compare the following two answers and decide if they express the same final result."
                        f"Return a json object with field 'equivalent' set to true if they are the same, false otherwise."
                        f"Note that for multiple-choice questions, prividing the correct option is counted correct."
                        f"Candidate answer: {solution}\n\n"
                        f"Ground truth: {ground_truth}\n\n"
                    ),
                }
            ],
            response_format=EquivalenceResult,
            temperature=0,
        )
        parsed: EquivalenceResult = resp.choices[0].message.parsed
        return bool(parsed.equivalent)
    except Exception as e:
        print(f"[verify_solution_equivalence ERROR] {e}")
        return False


def judge_answer_rule(predicted: str, ground_truth: str) -> bool:
    p = str(predicted).strip()
    g = str(ground_truth).strip()
    if p == g:
        return True
    if p.upper() == g.upper():
        return True
    if g and g in p:
        return True
    return False


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LTPO direct-boxed variant on MLLM benchmarks"
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="mllm_data")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_data_idx", type=int, default=0)
    parser.add_argument("--end_data_idx", type=int, default=99999)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--min_pixels", type=int, default=128)
    parser.add_argument("--max_pixels", type=int, default=256)

    parser.add_argument("--num_thought_tokens", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=25.0)
    parser.add_argument("--sigma_decay", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_num_steps", type=int, default=15)

    parser.add_argument("--reward_threshold", type=float, default=-1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--disable_conf_reward", action="store_true")
    parser.add_argument("--disable_best_reward", action="store_true")
    parser.add_argument(
        "--reward_type",
        type=str,
        default="confidence",
        choices=["confidence", "entropy", "entropy_diff", "entropy_clip"],
        help=(
            "Optimisation objective for the internal LTPO reward at the "
            "first-generated-token position. 'confidence' (default) uses "
            "the negative mean log of the top-k probs; 'entropy' uses the "
            "information entropy -sum(p_i*log p_i) over the top-k probs. "
            "'entropy_diff' is the compound objective r1-r2 (single-sample "
            "NES descent); 'entropy_clip' is r1 + clip(r1-r2, min=0) where "
            "r1 = first-output-token entropy under the normal forward and "
            "r2 = first-output-token entropy when attention to image-token "
            "positions is masked out. Compound objectives are minimised; "
            "compound best-latent selection is controlled by "
            "--compound_best_selection. The compound rewards are always "
            "scored at the first-output-token position, so "
            "--reward_on_latent_tokens is silently ignored for them."
        ),
    )
    parser.add_argument(
        "--compound_best_selection",
        type=str,
        default="r1_pos",
        choices=["r1_pos", "diff", "post_update_entropy"],
        help=(
            "Best-latent selection criterion for compound entropy rewards "
            "only. 'r1_pos' ranks by normal-forward entropy at the +eps "
            "probe; 'diff' ranks by r1_pos-r2_pos at the +eps probe; "
            "'post_update_entropy' performs one extra normal forward after "
            "the update and ranks by the entropy of the exact latent that "
            "will be saved. Ignored for confidence/plain entropy rewards."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt_suffix", type=str, default="")
    parser.add_argument("--use_auto_grad", action="store_true")
    parser.add_argument("--eval_baseline", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--disable_save_logistics", action="store_true")
    parser.add_argument("--log_topk_tokens", action="store_true")
    parser.add_argument(
        "--use_baseline_prompt",
        action="store_true",
        help=(
            "If set, the LTPO path uses the eval-baseline user prompt "
            "(just the raw question) and places the latent thought tokens "
            "in the assistant turn, immediately after the assistant role "
            "marker and before '\\boxed{' (with a single newline between "
            "the thought tokens and '\\boxed{'). Otherwise the default "
            "direct-boxed LTPO prompt is used (thought tokens embedded in "
            "the user turn, surrounded by DMLR-style instructions)."
        ),
    )
    parser.add_argument(
        "--baseline_with_thought_tokens",
        action="store_true",
        help=(
            "Only meaningful with --eval_baseline. If set, the baseline "
            "inserts --num_thought_tokens latent thought tokens between "
            "the assistant role marker and the forced '\\boxed{' prefix "
            "(separated by a single newline), matching the placement "
            "used by --use_baseline_prompt LTPO. No optimisation is "
            "performed — the model just runs inference with the default "
            "embeddings of those tokens."
        ),
    )
    parser.add_argument(
        "--baseline_thought_init_from_hidden",
        action="store_true",
        help=(
            "Only meaningful together with --eval_baseline and "
            "--baseline_with_thought_tokens. If set, the latent thought "
            "tokens are initialised from the last-layer hidden state "
            "(pre lm_head) of the position immediately before them — "
            "i.e. the last token of the assistant role marker — instead "
            "of from the default '<|endoftext|>' (or model-specific) "
            "token embedding. No optimisation is performed; the model "
            "just runs a single forward generation pass with the "
            "re-initialised embeddings."
        ),
    )
    parser.add_argument(
        "--baseline_thought_init_from_mean",
        action="store_true",
        help=(
            "Only meaningful together with --eval_baseline and "
            "--baseline_with_thought_tokens. If set, the latent thought "
            "tokens are initialised from the mean of the model's input "
            "token-embedding table (averaged across the vocabulary). "
            "Mutually exclusive with --baseline_thought_init_from_hidden. "
            "No optimisation is performed."
        ),
    )
    parser.add_argument(
        "--ltpo_thought_init_from_hidden",
        action="store_true",
        help=(
            "Re-initialise the latent thought-token embeddings in the "
            "LTPO path from the last-layer hidden state (pre lm_head) of "
            "the position immediately before the thought block, mirroring "
            "the eval-baseline '--baseline_thought_init_from_hidden' "
            "init. Without this flag, the thought tokens start from the "
            "default '<|endoftext|>' (or model-specific) token embedding."
        ),
    )
    parser.add_argument(
        "--reward_on_latent_tokens",
        action="store_true",
        help=(
            "If set, the LTPO reward (confidence/entropy) is computed at "
            "every latent thought-token position and averaged, instead of "
            "at the single first-generated-token position. The objective "
            "scale (confidence/entropy) is unchanged. Ignored on the "
            "no-thought-token baseline-fallback forward (it still scores "
            "the first-generated-token position)."
        ),
    )
    parser.add_argument(
        "--enable_baseline_fallback",
        action="store_true",
        help=(
            "If set, compute the first-generated-token reward of the "
            "no-thought-token baseline inputs (vanilla eval-baseline "
            "prompt + forced '\\boxed{' prefix) once per sample and let "
            "it compete in the best-step selection. If no LTPO step "
            "beats this baseline reward, generation falls back to the "
            "no-thought-token baseline inputs. Only applies to the "
            "first-token reward path (confidence/entropy); ignored when "
            "--disable_conf_reward is set."
        ),
    )

    parser.add_argument("--use_llm_verify", action="store_true")
    parser.add_argument(
        "--use_inputs_embeds",
        action="store_true",
        help=(
            "If set, fall back to the legacy embeds-based path: pre-merge "
            "visual tokens into inputs_embeds, write LTPO latents into the "
            "thought-token rows in-place, and call "
            "``model.generate(inputs_embeds=...)``. The default (flag not "
            "set) uses the new hook path — official "
            "``model.generate(**inputs)`` plus a forward pre-hook on "
            "model.language_model — which preserves Qwen2.5-VL's M-RoPE, "
            "image-token scatter, rope_deltas cache and KV-cache logic."
        ),
    )
    parser.add_argument(
        "--persist_latent_tokens",
        action="store_true",
        help=(
            "If set, the LTPO latent thought tokens are NOT re-initialised "
            "for each sample. The first sample uses the normal init (or "
            "--ltpo_thought_init_from_hidden if also set); every subsequent "
            "sample starts LTPO from the best-optimised thought-token "
            "embeddings of the previous sample. Only applies to the LTPO "
            "path (ignored under --eval_baseline). The default (flag not "
            "set) re-initialises latent tokens for every sample (existing "
            "behaviour)."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_image(image_path: str, image_root: str) -> Image.Image | None:
    if not image_path:
        return None
    path = image_path if os.path.isabs(image_path) else os.path.join(image_root, image_path)
    if not os.path.exists(path):
        print(f"[WARNING] Image not found: {path}")
        return None
    return Image.open(path).convert('RGB')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if args.baseline_thought_init_from_hidden and args.baseline_thought_init_from_mean:
        raise ValueError(
            "--baseline_thought_init_from_hidden and "
            "--baseline_thought_init_from_mean are mutually exclusive."
        )

    if args.seed:
        set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",
        token=huggingface_token,
    )
    model.to(device)
    model.eval()

    processor_kwargs = {
        "padding_side": "left",
        "min_pixels": args.min_pixels * 28 * 28,
        "max_pixels": args.max_pixels * 28 * 28,
    }
    if huggingface_token:
        processor_kwargs["token"] = huggingface_token
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, **processor_kwargs)

    from reward import RewardModel
    reward_model = RewardModel(
        model=model,
        tokenizer=processor.tokenizer if hasattr(processor, 'tokenizer') else processor,
        num_thought_tokens=args.num_thought_tokens,
    )

    dataset = get_mllm_dataset(args.dataset, data_root=args.data_root)
    if args.verbose:
        print(f"Loaded {len(dataset)} examples from '{args.dataset}'")
        print(f"Example[0]: {dataset[0]['question'][:120]}...")

    model_name = args.model_name_or_path.split("/")[-1]
    data_name = args.dataset.split("/")[-1]
    if args.disable_conf_reward:
        reward_suffix = ""
    elif args.reward_type == "entropy":
        reward_suffix = "-entropy"
    elif args.reward_type == "entropy_diff":
        reward_suffix = "-entdiff"
    elif args.reward_type == "entropy_clip":
        reward_suffix = "-entclip"
    else:
        reward_suffix = "-conf"

    if args.eval_baseline:
        output_suffix = "-" + args.ckpt_suffix if args.ckpt_suffix else ""
        if args.baseline_with_thought_tokens:
            if args.baseline_thought_init_from_hidden:
                init_tag = "-hinit"
            elif args.baseline_thought_init_from_mean:
                init_tag = "-minit"
            else:
                init_tag = ""
            baseline_thtok_suffix = f"-thtok{args.num_thought_tokens}{init_tag}"
        else:
            baseline_thtok_suffix = ""
        output_dir = (
            f"{args.output_dir}/{model_name}-{data_name}"
            f"-max_tokens{args.max_new_tokens}-boxed"
            + baseline_thtok_suffix + output_suffix
        )
    else:
        prompt_suffix = "-baseprompt" if args.use_baseline_prompt else ""
        fallback_suffix = "-bfallback" if args.enable_baseline_fallback else ""
        init_suffix = "-hinit" if args.ltpo_thought_init_from_hidden else ""
        persist_suffix = "-persist" if args.persist_latent_tokens else ""
        output_dir = (
            f"{args.output_dir}/{model_name}-{data_name}"
            f"-tokens{args.num_thought_tokens}-lr{args.lr}"
            f"-sigma{args.sigma}-sigdecay{args.sigma_decay}"
            f"-steps{args.max_num_steps}-topk{args.top_k}" + reward_suffix
            + "-boxed" + prompt_suffix + fallback_suffix + init_suffix
            + persist_suffix
        )

    start_data_idx = max(0, args.start_data_idx)
    end_data_idx = min(args.end_data_idx, len(dataset))

    total, correct = 0, 0
    entries = []

    if args.resume and not args.disable_save_logistics:
        logistics_path = f"{output_dir}/logistics.pt"
        if os.path.exists(logistics_path):
            print(f"Resuming from {output_dir}")
            logistics = torch.load(logistics_path)
            start_data_idx = logistics["start_idx"]
            correct = logistics["correct"]
            total = logistics["total"]
            entries = logistics["entries"]

    print(f"Evaluating '{args.dataset}' [{start_data_idx}, {end_data_idx})...")

    # Cross-sample carry-over for --persist_latent_tokens. Stays None for the
    # first sample (so the normal init runs); afterwards holds the previous
    # sample's best-optimised thought-token embeddings.
    persistent_thought_embeds = None

    for i in tqdm(range(start_data_idx, end_data_idx)):
        example = dataset[i]
        question = example['question']
        true_answer = example['answer']

        if true_answer is None:
            continue

        image = load_image(example['image_path'], args.image_root)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if args.verbose:
            print(f"\n[{i}] Q: {question[:100]}...")
            print(f"[{i}] True answer: {true_answer}")

        best_reward, best_reward_step, stop_reason = None, None, None

        if args.eval_baseline:
            # ---- Baseline still uses the forced "\\boxed{" assistant prefix ----
            # Optionally insert latent thought tokens between the assistant
            # role marker and "\\boxed{" (same placement as LTPO under
            # --use_baseline_prompt) — inference only, no optimisation.
            if args.baseline_with_thought_tokens:
                assistant_suffix = (
                    _thought_token_str(args.model_name_or_path, args.num_thought_tokens)
                    + "\n" + ASSISTANT_BOXED_PREFIX
                )
            else:
                assistant_suffix = ASSISTANT_BOXED_PREFIX

            if image is not None:
                if 'qwen' in args.model_name_or_path.lower():
                    messages = [
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'image', 'image': image},
                                {'type': 'text', 'text': question},
                            ],
                        },
                    ]
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    text = text + assistant_suffix
                    inputs = processor(
                        text=[text], images=[image], return_tensors='pt'
                    ).to(device)
                else:
                    text = f"<image>\n{question}" + assistant_suffix
                    inputs = processor(
                        images=image, text=text, return_tensors='pt'
                    ).to(device)
            else:
                messages = [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': question},
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                text = text + assistant_suffix
                inputs = processor(text=[text], return_tensors='pt').to(device)

            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

            reinit_thought = args.baseline_with_thought_tokens and (
                args.baseline_thought_init_from_hidden
                or args.baseline_thought_init_from_mean
            )

            if reinit_thought:
                # Re-initialise the thought-token embeddings to either the
                # last-layer hidden state (pre lm_head) of the position
                # immediately before the thought block, or the mean of
                # the model's input token-embedding table. Then run a
                # single generate pass (no optimisation). The actual
                # write/forward differs between the legacy embeds path
                # (--use_inputs_embeds) and the default hook path.
                input_ids = inputs['input_ids']
                thought_ids = _thought_token_ids(
                    tokenizer, args.model_name_or_path, args.num_thought_tokens
                )
                thought_start = _find_thought_token_start(
                    input_ids[0].tolist(), thought_ids
                )
                if args.use_inputs_embeds:
                    with torch.no_grad():
                        inputs_embeds = _merge_visual_tokens(
                            model, input_ids, inputs, args.model_name_or_path
                        )
                        attn = inputs.get(
                            'attention_mask',
                            torch.ones(inputs_embeds.shape[:2], device=device),
                        )
                        if args.baseline_thought_init_from_hidden:
                            fwd = model(
                                inputs_embeds=inputs_embeds,
                                attention_mask=attn,
                                output_hidden_states=True,
                                return_dict=True,
                            )
                            init_vec = fwd.hidden_states[-1][0, thought_start - 1]
                        else:
                            embed_weight = model.get_input_embeddings().weight
                            init_vec = embed_weight.mean(dim=0).to(
                                dtype=inputs_embeds.dtype, device=inputs_embeds.device
                            )
                        inputs_embeds[0, thought_start:thought_start + args.num_thought_tokens] = init_vec
                        raw_outputs = model.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attn,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            temperature=0.0,
                            top_p=None,
                            num_beams=1,
                        )
                    # generate(inputs_embeds=...) returns only new tokens;
                    # prepend the boxed prefix so extract_answer can still
                    # recover "\boxed{...}".
                    output = ASSISTANT_BOXED_PREFIX + tokenizer.decode(
                        raw_outputs[0], skip_special_tokens=True
                    )
                else:
                    with torch.no_grad():
                        if args.baseline_thought_init_from_hidden:
                            fwd = model(
                                **inputs, output_hidden_states=True, return_dict=True
                            )
                            init_vec = fwd.hidden_states[-1][0, thought_start - 1].clone()
                            del fwd
                            torch.cuda.empty_cache()
                        else:
                            embed_weight = model.get_input_embeddings().weight
                            init_vec = embed_weight.mean(dim=0).detach().clone()

                        thought_embeds = init_vec.unsqueeze(0).expand(
                            args.num_thought_tokens, -1
                        ).contiguous()
                        injector = ThoughtEmbedInjector(
                            thought_start,
                            thought_start + args.num_thought_tokens,
                            thought_embeds,
                        )
                        handle = model.language_model.register_forward_pre_hook(
                            injector, with_kwargs=True
                        )
                        try:
                            raw_outputs = model.generate(
                                **inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                temperature=0.0,
                                top_p=None,
                                num_beams=1,
                            )
                        finally:
                            handle.remove()
                    # generate(**inputs) returns the full sequence; the prompt
                    # already contains the forced "\\boxed{" prefix so
                    # extract_answer can still recover the answer.
                    output = tokenizer.decode(raw_outputs[0], skip_special_tokens=True)
            else:
                with torch.no_grad():
                    raw_outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=0.0,
                        top_p=None,
                        num_beams=1,
                    )
                output = tokenizer.decode(raw_outputs[0], skip_special_tokens=True)

        else:
            gen_result = generate_vl(
                processor=processor,
                model=model,
                reward_model=reward_model,
                image=image,
                question=question,
                num_thought_tokens=args.num_thought_tokens,
                max_rl_steps=args.max_num_steps,
                max_new_tokens=args.max_new_tokens,
                reward_threshold=args.reward_threshold,
                lr=args.lr,
                sigma=args.sigma,
                sigma_decay=args.sigma_decay,
                use_auto_grad=args.use_auto_grad,
                disable_conf_reward=args.disable_conf_reward,
                disable_best_reward=args.disable_best_reward,
                data_name=args.dataset,
                model_name=args.model_name_or_path,
                verbose=args.verbose,
                top_k=args.top_k,
                log_topk_tokens=args.log_topk_tokens,
                reward_type=args.reward_type,
                compound_best_selection=args.compound_best_selection,
                use_baseline_prompt=args.use_baseline_prompt,
                enable_baseline_fallback=args.enable_baseline_fallback,
                thought_init_from_hidden=args.ltpo_thought_init_from_hidden,
                use_inputs_embeds=args.use_inputs_embeds,
                initial_thought_embeds_override=persistent_thought_embeds,
                return_best_thought_embeds=args.persist_latent_tokens,
                reward_on_latent_tokens=args.reward_on_latent_tokens,
            )
            if args.persist_latent_tokens:
                output, best_reward, best_reward_step, stop_reason, best_thought_embeds = gen_result
                persistent_thought_embeds = best_thought_embeds
            else:
                output, best_reward, best_reward_step, stop_reason = gen_result

        answer = extract_answer(output)

        if args.use_llm_verify:
            is_correct = verify_solution_equivalence(answer, true_answer)
        else:
            is_correct = judge_answer_rule(answer, true_answer)

        correct += is_correct
        total += 1

        if args.verbose:
            if args.verbose > 1:
                print(f"[{i}] LLM response:\n{output}")
            print(f"[{i}] Extracted: {answer}  |  True: {true_answer}  |  Correct: {is_correct}")
            print(f"[{i}] Best reward: {best_reward}, step: {best_reward_step}, stop: {stop_reason}")

        if not args.disable_save_logistics:
            entries.append(dict(
                data_idx=i,
                question=question,
                response=output,
                answer=answer,
                true_answer=true_answer,
                is_correct=is_correct,
                best_reward=best_reward,
                best_reward_step=best_reward_step,
                stop_reason=stop_reason,
            ))
            torch.save({
                "start_idx": i + 1,
                "total": total,
                "correct": correct,
                "entries": entries,
            }, f"{output_dir}/logistics.pt")

        print(f"Running accuracy: {correct}/{total} = {correct / total:.4f}")

    if total > 0:
        print(f"\n>>> Final: correct={correct}, total={total}, accuracy={correct / total:.4f}")
    print(f">>> Correct indices: {[e['data_idx'] for e in entries if e['is_correct']]}")

    with open(f"{output_dir}/results.log", "a") as f:
        f.write(
            f"Data Idx with Correct Answer: "
            f"{[entry['data_idx'] for entry in entries if entry['is_correct']]}\n"
        )
        f.write(f"correct={correct}, total={total}, accuracy={correct / total:.4f}\n")


if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        print(f"-- {arg}: {getattr(args, arg)}")
    main(args)
