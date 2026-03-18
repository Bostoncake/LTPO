"""
Entry point for Vision-Language LTPO evaluation.

This mirrors ``main.py`` (text-only LTPO) but loads a VL model and dataset,
and calls the multimodal generation loop.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from multimodal import (
    get_vl_dataset,
    generate_vl,
    VLRewardModel,
    extract_answer,
    extract_true_answer,
    judge_answer,
    args_to_dict,
)


huggingface_token = os.environ.get("HUGGING_FACE_TOKEN", None)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a VL model with LTPO")
    # data
    p.add_argument("--dataset", type=str, default="data/mm_math.json", help="Path to dataset JSON file")
    p.add_argument("--start_data_idx", type=int, default=0)
    p.add_argument("--end_data_idx", type=int, default=100)

    # model
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28, help="Min image resolution")
    p.add_argument("--max_pixels", type=int, default=1280 * 28 * 28, help="Max image resolution")

    # generation
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--solver_prompt_idx", type=int, default=0)

    # LTPO optimisation
    p.add_argument("--num_thought_tokens", type=int, default=8)
    p.add_argument("--sigma", type=float, default=20.0)
    p.add_argument("--sigma_decay", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--max_num_steps", type=int, default=20)

    # reward
    p.add_argument("--reward_threshold", type=float, default=-1)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--disable_conf_reward", action="store_true")
    p.add_argument("--disable_best_reward", action="store_true")

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ckpt_suffix", type=str, default="")
    p.add_argument("--use_auto_grad", action="store_true")
    p.add_argument("--eval_baseline", action="store_true", help="Evaluate without LTPO (baseline)")
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--disable_save_logistics", action="store_true")
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    if args.seed:
        set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load VL model & processor ---
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        token=huggingface_token,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        token=huggingface_token,
        attn_implementation="eager",
    )
    model.eval()

    # --- Reward model ---
    reward_model = VLRewardModel(
        model=model,
        processor=processor,
        num_thought_tokens=args.num_thought_tokens,
    )

    # --- Dataset ---
    dataset = get_vl_dataset(
        args.dataset,
        processor=processor,
        prompt_idx=args.solver_prompt_idx,
        start_idx=args.start_data_idx,
        end_idx=args.end_data_idx,
    )
    if args.verbose:
        print(f"Dataset loaded: {len(dataset)} examples")

    # --- Output dir ---
    model_name = args.model_name_or_path.split("/")[-1]
    data_name = os.path.splitext(os.path.basename(args.dataset))[0]
    conf_suffix = "" if args.disable_conf_reward else "-conf"
    if args.eval_baseline:
        suffix = f"-{args.ckpt_suffix}" if args.ckpt_suffix else ""
        output_dir = f"{args.output_dir}/{model_name}-{data_name}-max_tokens{args.max_new_tokens}-prompt{args.solver_prompt_idx}{suffix}"
    else:
        output_dir = (
            f"{args.output_dir}/{model_name}-{data_name}"
            f"-tokens{args.num_thought_tokens}-lr{args.lr}"
            f"-sigma{args.sigma}-sigdecay{args.sigma_decay}{conf_suffix}"
        )
    os.makedirs(output_dir, exist_ok=True)

    # --- Resume ---
    total, correct = 0, 0
    entries = []
    start_idx = 0
    if args.resume and not args.disable_save_logistics:
        logistics_path = f"{output_dir}/logistics.pt"
        if os.path.exists(logistics_path):
            logistics = torch.load(logistics_path)
            start_idx = logistics["start_idx"]
            correct = logistics["correct"]
            total = logistics["total"]
            entries = logistics["entries"]
            print(f"Resumed from {logistics_path} (idx={start_idx})")

    end_idx = min(args.end_data_idx, len(dataset))
    print(f"Evaluating {args.dataset} from {start_idx} to {end_idx} ...")

    for i in tqdm(range(start_idx, end_idx)):
        example = dataset[i]
        question = example["question"]
        image = example.get("image", None)
        true_answer = extract_true_answer(example["answer"], name=args.dataset)
        if true_answer is None:
            continue

        if args.verbose:
            print(f"[{i}] Q: {question[:120]}...")
            print(f"[{i}] True answer: {true_answer}")

        if args.eval_baseline:
            # Baseline: generate without LTPO
            messages = example.get("messages", [{"role": "user", "content": question}])
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if image is not None:
                from PIL import Image as PILImage
                if isinstance(image, str) and os.path.exists(image):
                    image_obj = PILImage.open(image).convert("RGB")
                else:
                    image_obj = image
                inputs = processor(text=[text], images=[image_obj], return_tensors="pt", padding=True).to(device)
            else:
                inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
            outputs = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
            )
            output = processor.decode(outputs[0], skip_special_tokens=True)
            best_reward, best_reward_step = None, None
        else:
            output, best_reward, best_reward_step, stop_reason = generate_vl(
                processor=processor,
                model=model,
                reward_model=reward_model,
                question=question,
                image=image,
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

        answer = extract_answer(output)
        is_correct = False
        if answer is not None:
            is_correct = judge_answer(output, true_answer, data_name=args.dataset)
            correct += is_correct

        if args.verbose:
            if args.verbose > 1:
                print(f"[{i}] Response:\n{output}")
            print(f"[{i}] Extracted: {answer} | True: {true_answer} | Correct: {is_correct}")

        if not args.disable_save_logistics:
            entries.append(dict(
                data_idx=i,
                question=question,
                response=output,
                answer=answer,
                is_correct=is_correct,
                best_reward=best_reward,
                best_reward_step=best_reward_step,
            ))

        total += 1

        if not args.disable_save_logistics:
            torch.save(
                {"start_idx": i + 1, "total": total, "correct": correct, "entries": entries},
                f"{output_dir}/logistics.pt",
            )
        print(f"  correct={correct}, total={total}, acc={correct / total:.4f}")

    print(f"\n>>> Final: correct={correct}, total={total}, accuracy={correct / total:.4f}")


if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        print(f"-- {arg}: {getattr(args, arg)}")
    main(args)
