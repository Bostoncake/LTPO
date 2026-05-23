"""
main_vl_visattnsink.py — evaluate **VisAttnSink alone** on LTPO's dev
splits, using LTPO's data loading and prompt format.

This mirrors `main_vl_dmlr_final.py --eval_baseline --use_baseline_prompt`
(forced `\boxed{` assistant prefix, LTPO SYSTEM_PROMPT, no latent thought
tokens, no NES loop) but additionally installs VisAttnSink's per-layer
attention redistribution.

CLI mirrors `main_vl_dmlr_final_visual.py` for the args that matter and
adds the VisAttnSink hyper-params (tau/rho/summ/p/except_last_layer).
"""

from __future__ import annotations

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
from visattnsink_core import DIM_SINK, LogicEngine, MetadataStation
from visattnsink_patch import install_visattnsink


huggingface_token = os.environ.get("HUGGING_FACE_TOKEN")

ASSISTANT_BOXED_PREFIX = "\\boxed{"


# ---------------------------------------------------------------------------
# Answer extraction + LLM verification (copied from main_vl_dmlr_final.py).
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


_llm_client = None
_llm_model = None


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
                        "Compare the following two answers and decide if they express the same final result."
                        "Return a json object with field 'equivalent' set to true if they are the same, false otherwise."
                        "Note that for multiple-choice questions, prividing the correct option is counted correct."
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
    if p == g or p.upper() == g.upper():
        return True
    if g and g in p:
        return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="VisAttnSink baseline on MLLM dev splits")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="mllm_data")
    parser.add_argument("--image_root", type=str, default=".")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_data_idx", type=int, default=0)
    parser.add_argument("--end_data_idx", type=int, default=99999)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--min_pixels", type=int, default=128)
    parser.add_argument("--max_pixels", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--use_llm_verify", action="store_true")
    parser.add_argument("--ckpt_suffix", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable_save_logistics", action="store_true")

    # VisAttnSink hyper-params (from Other_baseline/VisAttnSink/A_exps/lv1.5_7b.yml).
    parser.add_argument("--vas_tau", type=float, default=20.0)
    parser.add_argument("--vas_rho", type=float, default=0.5)
    parser.add_argument("--vas_summ", type=float, default=0.2)
    parser.add_argument("--vas_p", type=float, default=0.6)
    parser.add_argument("--vas_except_last_layer", type=int, default=1)
    parser.add_argument("--disable_vas", action="store_true",
                        help="Run the same pipeline but with VisAttnSink disabled (sanity check).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _locate_image_span(input_ids: torch.Tensor, model) -> tuple[int, int]:
    img_tok_id = getattr(model.config, "image_token_id", None)
    if img_tok_id is None:
        img_tok_id = getattr(model.config, "image_token_index", None)
    if img_tok_id is None:
        return 0, 0
    pos = (input_ids[0] == img_tok_id).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        return 0, 0
    return int(pos[0].item()), int(pos[-1].item()) + 1


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
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    if not args.disable_vas:
        model_key = args.model_name_or_path.rstrip("/").split("/")[-1]
        dim_sink = DIM_SINK.get(model_key)
        if dim_sink is None:
            raise RuntimeError(
                f"No calibrated DIM_SINK entry for '{model_key}'. "
                f"Run calibrate_outlier_dims.py first or add to visattnsink_core.DIM_SINK."
            )
        print(f"[VisAttnSink] Using outlier dims for {model_key}: {dim_sink}")
        LogicEngine.activate(
            tau=args.vas_tau,
            rho=args.vas_rho,
            summ=args.vas_summ,
            p=args.vas_p,
            except_last_layer=bool(args.vas_except_last_layer),
            dim_sink=dim_sink,
        )
        install_visattnsink(model)
    else:
        LogicEngine.enabled = False

    # Read model config -> head/layer counts for MetadataStation.
    text_cfg = getattr(model.config, "text_config", model.config)
    num_layers = getattr(text_cfg, "num_hidden_layers", None)
    num_heads = getattr(text_cfg, "num_attention_heads", None)
    if num_layers is None or num_heads is None:
        raise RuntimeError("Could not derive num_hidden_layers/num_attention_heads from model.config")

    dataset = get_mllm_dataset(args.dataset, data_root=args.data_root)
    if args.verbose:
        print(f"Loaded {len(dataset)} examples from '{args.dataset}'")
        print(f"Example[0]: {dataset[0]['question'][:120]}...")

    model_name = args.model_name_or_path.rstrip("/").split("/")[-1]
    data_name = args.dataset.split("/")[-1]
    suffix = f"-{args.ckpt_suffix}" if args.ckpt_suffix else ""
    output_dir = (
        f"{args.output_dir}/{model_name}-{data_name}"
        f"-visattnsink-tau{args.vas_tau}-rho{args.vas_rho}"
        f"-summ{args.vas_summ}-p{args.vas_p}{suffix}"
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

        if image is not None:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text = text + ASSISTANT_BOXED_PREFIX
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            text = text + ASSISTANT_BOXED_PREFIX
            inputs = processor(text=[text], return_tensors="pt").to(device)

        img_start, img_end = _locate_image_span(inputs["input_ids"], model)
        MetadataStation.activate(
            image_start=img_start,
            image_end=img_end,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
        )
        LogicEngine.clear()

        try:
            with torch.inference_mode():
                raw_outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=None,
                    num_beams=1,
                )

            new_tokens = raw_outputs[0, inputs["input_ids"].shape[1]:]
            response = ASSISTANT_BOXED_PREFIX + tokenizer.decode(new_tokens, skip_special_tokens=True)
        except Exception as e:
            print(f"[{i}] generation error: {e}")
            response = ""
        finally:
            MetadataStation.deactivate()
            LogicEngine.clear()

        answer = extract_answer(response)

        if args.use_llm_verify:
            is_correct = verify_solution_equivalence(answer, true_answer)
        else:
            is_correct = judge_answer_rule(answer, true_answer)

        correct += int(is_correct)
        total += 1

        if args.verbose:
            print(f"[{i}] Q: {question[:80]}...")
            print(f"[{i}] Extracted: {answer}  |  True: {true_answer}  |  Correct: {is_correct}")
            print(f"Running accuracy: {correct}/{total} = {correct / total:.4f}")

        if not args.disable_save_logistics:
            entries.append(dict(
                data_idx=i,
                question=question,
                response=response,
                answer=answer,
                true_answer=true_answer,
                is_correct=is_correct,
            ))
            torch.save({
                "start_idx": i + 1,
                "total": total,
                "correct": correct,
                "entries": entries,
            }, f"{output_dir}/logistics.pt")

        del raw_outputs
        torch.cuda.empty_cache()

    if total > 0:
        print(f"\n>>> Final: correct={correct}, total={total}, accuracy={correct / total:.4f}")
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
