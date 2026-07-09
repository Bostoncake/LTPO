"""
main_vl_slot.py — Driver for the SLOT baseline on LTPO's MLLM dev splits.

Mirrors main_vl_dmlr_final_visual.py; replaces the LTPO+DMLR-visual
generate with slot_vl.generate_vl.
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
from slot_vl import generate_vl


huggingface_token = os.environ.get("HUGGING_FACE_TOKEN")


def extract_answer(text: str) -> str:
    if not text:
        return ""
    try:
        low = text.lower()
        start = low.find("<answer>")
        end = low.find("</answer>")
        if start != -1 and end != -1 and end > start:
            ans = text[start + len("<answer>"):end].strip()
            ans = ans.strip("$")
            ans = re.sub(r"\\displaystyle\s*", "", ans)
            ans = re.sub(r"\s+", " ", ans).strip()
            if ans:
                return ans

        boxed_contents = []
        for m in re.finditer(r"\\boxed\s*\{", text):
            open_brace_pos = text.find("{", m.end() - 1)
            if open_brace_pos == -1:
                continue
            depth = 0
            i = open_brace_pos
            while i < len(text):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        boxed = text[open_brace_pos + 1:i]
                        boxed = boxed.strip().strip("$")
                        boxed = re.sub(r"\\displaystyle\s*", "", boxed)
                        b = boxed.strip()
                        if b.startswith(r"\text{") and b.endswith("}"):
                            b = b[len(r"\text{"):-1].strip()
                        boxed = re.sub(r"\s+", " ", b).strip()
                        if boxed:
                            boxed_contents.append(boxed)
                        break
                i += 1
        if boxed_contents:
            return boxed_contents[-1]
    except Exception:
        pass
    return text.strip()


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
            messages=[{
                "role": "user",
                "content": (
                    f"Compare the following two answers and decide if they express the same final result."
                    f"Return a json object with field 'equivalent' set to true if they are the same, false otherwise."
                    f"Note that for multiple-choice questions, prividing the correct option is counted correct."
                    f"Candidate answer: {solution}\n\n"
                    f"Ground truth: {ground_truth}\n\n"
                ),
            }],
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SLOT baseline on MLLM benchmarks")
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

    # SLOT knobs (defaults match Other_baseline/SLOT/run.sh: times=3, lr=0.01).
    parser.add_argument("--slot_times", type=int, default=3)
    parser.add_argument("--slot_lr", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt_suffix", type=str, default="")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--disable_save_logistics", action="store_true")
    parser.add_argument("--use_llm_verify", action="store_true")

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_image(image_path: str, image_root: str):
    if not image_path:
        return None
    path = image_path if os.path.isabs(image_path) else os.path.join(image_root, image_path)
    if not os.path.exists(path):
        print(f"[WARNING] Image not found: {path}")
        return None
    return Image.open(path).convert("RGB")


def main(args):
    if args.seed:
        set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
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

    dataset = get_mllm_dataset(args.dataset, data_root=args.data_root)
    if args.verbose:
        print(f"Loaded {len(dataset)} examples from '{args.dataset}'")
        print(f"Example[0]: {dataset[0]['question'][:120]}...")

    model_name = args.model_name_or_path.split("/")[-1]
    data_name = args.dataset.split("/")[-1]
    output_suffix = "-" + args.ckpt_suffix if args.ckpt_suffix else ""
    output_dir = (
        f"{args.output_dir}/{model_name}-{data_name}"
        f"-slot-times{args.slot_times}-lr{args.slot_lr}" + output_suffix
    )
    os.makedirs(output_dir, exist_ok=True)

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
        question = example["question"]
        true_answer = example["answer"]
        if true_answer is None:
            continue
        image = load_image(example["image_path"], args.image_root)

        if args.verbose:
            print(f"\n[{i}] Q: {question[:100]}...")
            print(f"[{i}] True answer: {true_answer}")

        output, final_loss, last_step, stop_reason = generate_vl(
            processor=processor,
            model=model,
            image=image,
            question=question,
            max_new_tokens=args.max_new_tokens,
            slot_times=args.slot_times,
            slot_lr=args.slot_lr,
            model_name=args.model_name_or_path,
            verbose=args.verbose,
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
            print(f"[{i}] SLOT final_loss: {final_loss}, stop: {stop_reason}")

        if not args.disable_save_logistics:
            entries.append(dict(
                data_idx=i,
                question=question,
                response=output,
                answer=answer,
                true_answer=true_answer,
                is_correct=is_correct,
                final_loss=final_loss,
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
