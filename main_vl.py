"""
main_vl.py – Evaluate LTPO on Vision-Language (MLLM) benchmarks.

Usage example:
    python main_vl.py \
        --dataset math_vista \
        --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
        --output_dir ./output \
        --num_thought_tokens 10

Required env vars:
    HUGGING_FACE_TOKEN      – HuggingFace access token
    OPENAI_API_KEY          – Key for the LLM judge
    OPENAI_API_BASE_URL     – Judge API base URL
    MODEL_TYPE              – Judge model name  (e.g. qwen3-vl-plus)
"""

import argparse
import os
import random

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

from data_vl import get_mllm_dataset
from extract_judge_answer.utils_vl import (
    extract_answer_vl,
    extract_true_answer_vl,
    judge_answer_vl,
)
from ltpo_vl import generate_vl
from reward import RewardModel


huggingface_token = os.environ.get('HUGGING_FACE_TOKEN')


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LTPO on MLLM benchmarks")

    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name, e.g. math_vista, mmstar, hallusion")
    parser.add_argument("--data_root", type=str, default="mllm_data",
                        help="Directory containing the MLLM JSON files")
    parser.add_argument("--image_root", type=str, default="",
                        help="Root directory prepended to relative image_path values")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_data_idx", type=int, default=0)
    parser.add_argument("--end_data_idx", type=int, default=99999)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")

    # Optimisation
    parser.add_argument("--num_thought_tokens", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--sigma_decay", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--max_num_steps", type=int, default=10)

    # Reward
    parser.add_argument("--reward_threshold", type=float, default=-1)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--disable_conf_reward", action="store_true")
    parser.add_argument("--disable_best_reward", action="store_true")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt_suffix", type=str, default="")
    parser.add_argument("--use_auto_grad", action="store_true")
    parser.add_argument("--eval_baseline", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--disable_save_logistics", action="store_true")

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
    """Load a PIL image, resolving relative paths against image_root."""
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

    # ---- Load model & processor ----
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        token=huggingface_token,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        token=huggingface_token,
    )

    # ---- Load reward model (uses the same base LLM for confidence reward) ----
    reward_model = RewardModel(
        model=model,
        tokenizer=processor.tokenizer if hasattr(processor, 'tokenizer') else processor,
        num_thought_tokens=args.num_thought_tokens,
    )

    # ---- Load dataset ----
    dataset = get_mllm_dataset(args.dataset, data_root=args.data_root)
    if args.verbose:
        print(f"Loaded {len(dataset)} examples from '{args.dataset}'")
        print(f"Example[0]: {dataset[0]['question'][:120]}...")

    model_name = args.model_name_or_path.split("/")[-1]
    data_name = args.dataset.split("/")[-1]
    conf_suffix = "" if args.disable_conf_reward else "-conf"

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
            f"-steps{args.max_num_steps}-topk{args.top_k}" + conf_suffix
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
        true_answer = extract_true_answer_vl(example['answer'], name=args.dataset)

        if true_answer is None:
            continue

        image = load_image(example['image_path'], args.image_root)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if args.verbose:
            print(f"\n[{i}] Q: {question[:100]}...")
            print(f"[{i}] True answer: {true_answer}")

        best_reward, best_reward_step = None, None

        if args.eval_baseline:
            # ---- Baseline: standard VL inference ----
            if image is not None:
                if 'qwen' in args.model_name_or_path.lower():
                    messages = [{
                        'role': 'user',
                        'content': [
                            {'type': 'image', 'image': image},
                            {'type': 'text', 'text': question},
                        ],
                    }]
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = processor(
                        text=[text], images=[image], return_tensors='pt'
                    ).to(device)
                else:
                    text = f"<image>\n{question}"
                    inputs = processor(
                        images=image, text=text, return_tensors='pt'
                    ).to(device)
            else:
                messages = [{'role': 'user', 'content': question}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(text=[text], return_tensors='pt').to(device)

            with torch.no_grad():
                raw_outputs = model.generate(**inputs, **dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=None,
                    num_beams=1,
                ))
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            output = tokenizer.decode(raw_outputs[0], skip_special_tokens=True)

        else:
            # ---- LTPO optimised generation ----
            output, best_reward, best_reward_step = generate_vl(
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
            )

        # ---- Extract & judge answer ----
        # print(f"[Output]: {output}")
        answer = extract_answer_vl(output, data_name=args.dataset)
        is_correct = judge_answer_vl(
            output=output,
            label=true_answer,
            data_name=args.dataset,
            question=question,
        )
        correct += is_correct
        total += 1

        if args.verbose:
            if args.verbose > 1:
                print(f"[{i}] LLM response:\n{output}")
            print(f"[{i}] Extracted: {answer}  |  True: {true_answer}  |  Correct: {is_correct}")
            print(f"[{i}] Best reward: {best_reward}, step: {best_reward_step}")

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
            ))
            torch.save({
                "start_idx": i + 1,
                "total": total,
                "correct": correct,
                "entries": entries,
            }, f"{output_dir}/logistics.pt")

        print(f"Running accuracy: {correct}/{total} = {correct / total:.4f}")

    # ---- Final summary ----
    if total > 0:
        print(f"\n>>> Final: correct={correct}, total={total}, accuracy={correct / total:.4f}")
    print(f">>> Correct indices: {[e['data_idx'] for e in entries if e['is_correct']]}")

    with open(f"{output_dir}/results.log", "a") as f:
        f.write(f"Data Idx with Correct Answer: {[entry['data_idx'] for entry in entries if entry['is_correct']]}\n")
        f.write(f"correct={correct}, total={total}, accuracy={correct / total:.4f}\n")


if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        print(f"-- {arg}: {getattr(args, arg)}")
    main(args)
