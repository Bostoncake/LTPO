"""
main_vl_workspace_mask_reward.py
================================
Evaluate LTPO + Mask-Reward Visual Workspace on VL benchmarks.

Key difference vs. main_vl_workspace_fixed.py
---------------------------------------------
During LTPO optimisation, latent thought tokens do NOT interact with image
slots at all (pure LTPO).  After optimisation converges, every image slot
is evaluated by masking the raw image tokens that belong to it and measuring
the drop in reward; the slot whose removal hurts reward the most is added
(summed) to every latent thought token before the final generation pass.

See visual_workspace_mask_reward.py / ltpo_vl_workspace_mask_reward.py for
details.

CLI additions (vs. main_vl_dmlr.py):
  --use_mask_reward_workspace    enable the mask-reward workspace (default off)
  --num_workspace_slots K        target slot count (default 8)
  --inject_scale s               scale applied to the injected slot (default 1.0)
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
from ltpo_vl_dmlr import SYSTEM_PROMPT
from ltpo_vl_workspace_mask_reward import generate_vl_mask_reward
from visual_workspace_mask_reward import MaskRewardWorkspaceConfig


huggingface_token = os.environ.get('HUGGING_FACE_TOKEN')


# ---------------------------------------------------------------------------
# Answer extraction (identical to main_vl_dmlr.py)
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
# Verification (identical to main_vl_dmlr.py)
# ---------------------------------------------------------------------------

_llm_client = None
_llm_model = None


def _get_llm_client():
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
            messages=[{
                "role": "user",
                "content": (
                    f"Compare the following two answers and decide if they express the same final result."
                    f"Return a json object with field 'equivalent' set to true if they are the same, false otherwise."
                    f"Note that for multiple-choice questions, providing the correct option is counted correct."
                    f"Candidate answer: {solution}\n\nGround truth: {ground_truth}\n\n"
                ),
            }],
            response_format=EquivalenceResult,
            temperature=0,
        )
        return bool(resp.choices[0].message.parsed.equivalent)
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
        description="Evaluate LTPO + Mask-Reward Visual Workspace on MLLM benchmarks"
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

    # LTPO hyperparameters (DMLR defaults)
    parser.add_argument("--num_thought_tokens", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=25.0)
    parser.add_argument("--sigma_decay", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_num_steps", type=int, default=15)

    # Reward
    parser.add_argument("--reward_threshold", type=float, default=-1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--disable_conf_reward", action="store_true")
    parser.add_argument("--disable_best_reward", action="store_true")

    # Mask-Reward Workspace
    parser.add_argument(
        "--use_mask_reward_workspace", action="store_true",
        help="Enable post-hoc mask-reward slot injection (default: off)",
    )
    parser.add_argument(
        "--num_workspace_slots", type=int, default=8,
        help="K: target number of disjoint image slots to evaluate",
    )
    parser.add_argument(
        "--inject_scale", type=float, default=1.0,
        help=(
            "Scale factor applied to the selected slot embedding when adding "
            "it to every latent thought token. 1.0 = raw sum."
        ),
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt_suffix", type=str, default="")
    parser.add_argument("--use_auto_grad", action="store_true")
    parser.add_argument("--eval_baseline", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--disable_save_logistics", action="store_true")
    parser.add_argument("--use_llm_verify", action="store_true")
    parser.add_argument(
        "--save_reward_trace", action="store_true",
        help="Persist per-step LTPO rewards and selected slot index per example.",
    )

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_image(image_path: str, image_root: str):
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

    ws_config = MaskRewardWorkspaceConfig(
        enabled=args.use_mask_reward_workspace,
        num_workspace_slots=args.num_workspace_slots,
        inject_scale=args.inject_scale,
    )
    if ws_config.enabled:
        print(
            f"[Mask-Reward Workspace] K={ws_config.num_workspace_slots}  "
            f"inject_scale={ws_config.inject_scale}"
        )

    dataset = get_mllm_dataset(args.dataset, data_root=args.data_root)
    if args.verbose:
        print(f"Loaded {len(dataset)} examples from '{args.dataset}'")

    model_name = args.model_name_or_path.split("/")[-1]
    data_name = args.dataset.split("/")[-1]
    conf_suffix = "" if args.disable_conf_reward else "-conf"
    mr_suffix = (
        f"-mr{ws_config.num_workspace_slots}"
        f"-s{ws_config.inject_scale:g}"
        if ws_config.enabled else ""
    )

    if args.eval_baseline:
        output_suffix = "-" + args.ckpt_suffix if args.ckpt_suffix else ""
        output_dir = (
            f"{args.output_dir}/{model_name}-{data_name}"
            f"-max_tokens{args.max_new_tokens}" + output_suffix
        )
    else:
        output_dir = (
            f"{args.output_dir}/{model_name}-{data_name}"
            f"-tokens{args.num_thought_tokens}-lr{args.lr}"
            f"-sigma{args.sigma}-sigdecay{args.sigma_decay}"
            f"-steps{args.max_num_steps}-topk{args.top_k}"
            + conf_suffix + mr_suffix + "-maskreward"
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
        step_rewards = None
        selected_slot_idx = -1

        if args.eval_baseline:
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
                    inputs = processor(
                        text=[text], images=[image], return_tensors='pt'
                    ).to(device)
                else:
                    text = f"<image>\n{question}"
                    inputs = processor(images=image, text=text, return_tensors='pt').to(device)
            else:
                messages = [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': question},
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(text=[text], return_tensors='pt').to(device)

            with torch.no_grad():
                raw_outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=None,
                    num_beams=1,
                )
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            output = tokenizer.decode(raw_outputs[0], skip_special_tokens=True)

        else:
            output, best_reward, best_reward_step, stop_reason, step_rewards, selected_slot_idx = (
                generate_vl_mask_reward(
                    processor=processor,
                    model=model,
                    reward_model=reward_model,
                    image=image,
                    question=question,
                    ws_config=ws_config,
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
                )
            )

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
            print(
                f"[{i}] Best reward: {best_reward}, step: {best_reward_step}, "
                f"stop: {stop_reason}, selected_slot: {selected_slot_idx}"
            )

        if not args.disable_save_logistics:
            entry = dict(
                data_idx=i,
                question=question,
                response=output,
                answer=answer,
                true_answer=true_answer,
                is_correct=is_correct,
                best_reward=best_reward,
                best_reward_step=best_reward_step,
                stop_reason=stop_reason,
                selected_slot_idx=selected_slot_idx,
            )
            if args.save_reward_trace:
                entry["step_rewards"] = step_rewards
            entries.append(entry)
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
