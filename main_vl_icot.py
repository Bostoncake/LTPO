"""
main_vl_icot.py — evaluate the ICoT baseline on MLLM dev sets.

Mirrors main_vl_dmlr_final_visual.py for data loading, answer extraction
and LLM-judge equivalence, but the per-sample generation calls
`icot_vl.generate_icot` instead of the LTPO RL loop.
"""

import argparse
import json
import os
import random
import re
import time

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from openai import OpenAI

from data_vl import get_mllm_dataset
from icot_vl import generate_icot


huggingface_token = os.environ.get("HUGGING_FACE_TOKEN")


# ---------------------------------------------------------------------------
# Answer extraction — copied verbatim from main_vl_dmlr_final_visual.
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


def _cuda_synchronize(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _sum_profiler_flops(prof) -> int:
    total = 0
    for event in prof.key_averages():
        total += int(getattr(event, "flops", 0) or 0)
    return total


def _sum_profiler_time_ms(prof, attr: str) -> float:
    total = 0.0
    for event in prof.key_averages():
        value = getattr(event, attr, None)
        if value is None and attr == "cuda_time_total":
            value = getattr(event, "device_time_total", 0.0)
        total += float(value or 0.0)
    return total / 1000.0


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    return str(obj)


def _make_efficiency_summary(records: list[dict]) -> dict:
    if not records:
        return {"num_samples": 0}

    def total(key: str) -> float:
        return float(sum(float(r.get(key, 0.0) or 0.0) for r in records))

    def avg(key: str) -> float:
        return total(key) / max(1, len(records))

    profiled = [r for r in records if r.get("profiled_flops")]
    total_profiled_flops = float(sum(float(r.get("profiler_estimated_flops", 0.0) or 0.0) for r in profiled))
    mean_profiled_flops = total_profiled_flops / max(1, len(profiled))
    estimated_total_flops = (
        total_profiled_flops
        if len(profiled) == len(records)
        else mean_profiled_flops * len(records)
    )

    total_wall = total("generation_wall_time_s")
    total_generated = total("generated_tokens")
    total_forward = total("forward_passes")
    return {
        "num_samples": len(records),
        "num_profiled_flop_samples": len(profiled),
        "flops_profiled_every_sample": len(profiled) == len(records),
        "total_generation_wall_time_s": total_wall,
        "avg_generation_wall_time_s": avg("generation_wall_time_s"),
        "total_judge_time_s": total("judge_time_s"),
        "avg_judge_time_s": avg("judge_time_s"),
        "total_generated_tokens": int(total_generated),
        "avg_generated_tokens": avg("generated_tokens"),
        "generation_tokens_per_second": total_generated / max(total_wall, 1e-12),
        "total_forward_passes": int(total_forward),
        "avg_forward_passes": avg("forward_passes"),
        "forward_passes_per_second": total_forward / max(total_wall, 1e-12),
        "total_prefill_forward_passes": int(total("prefill_forward_passes")),
        "total_decode_forward_passes": int(total("decode_forward_passes")),
        "total_subimage_forward_passes": int(total("subimage_forward_passes")),
        "total_attention_forward_passes": int(total("attention_forward_passes")),
        "total_subimages_inserted": int(total("num_sub_imgs")),
        "avg_subimages_inserted": avg("num_sub_imgs"),
        "total_subimage_tokens_inserted": int(total("subimage_tokens_inserted")),
        "avg_prompt_tokens": avg("prompt_tokens"),
        "avg_image_tokens": avg("image_tokens"),
        "avg_max_context_tokens": avg("max_context_tokens"),
        "total_model_forward_input_tokens": int(total("model_forward_input_tokens")),
        "total_profiled_flops": total_profiled_flops,
        "mean_profiled_flops_per_sample": mean_profiled_flops,
        "estimated_total_flops": estimated_total_flops,
        "estimated_total_tflops": estimated_total_flops / 1e12,
        "flops_note": (
            "PyTorch profiler with_flops=True reports an operator-level estimate, "
            "mainly for matmul/addmm/conv-like ops. Attention softmax, cache "
            "bookkeeping, sampling, Python overhead and some fused kernels may be "
            "missing, so treat FLOPs as a conservative comparable estimate."
        ),
    }


def _write_efficiency_summary(output_dir: str, records: list[dict]):
    summary = _make_efficiency_summary(records)
    json_path = os.path.join(output_dir, "efficiency_summary.json")
    md_path = os.path.join(output_dir, "efficiency_summary.md")
    with open(json_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)

    lines = [
        "# ICoT Efficiency Summary",
        "",
        f"- Samples: {summary.get('num_samples', 0)}",
        f"- Profiled FLOP samples: {summary.get('num_profiled_flop_samples', 0)}",
        f"- Total generation wall time: {summary.get('total_generation_wall_time_s', 0.0):.4f} s",
        f"- Average generation wall time: {summary.get('avg_generation_wall_time_s', 0.0):.4f} s/sample",
        f"- Total generated tokens: {summary.get('total_generated_tokens', 0)}",
        f"- Generation throughput: {summary.get('generation_tokens_per_second', 0.0):.4f} tokens/s",
        f"- Total forward passes: {summary.get('total_forward_passes', 0)}",
        f"- Average forward passes: {summary.get('avg_forward_passes', 0.0):.4f}/sample",
        f"- Total inserted sub-images: {summary.get('total_subimages_inserted', 0)}",
        f"- Estimated total FLOPs: {summary.get('estimated_total_flops', 0.0):.6e}",
        f"- Estimated total TFLOPs: {summary.get('estimated_total_tflops', 0.0):.4f}",
        "",
        "FLOPs note: " + summary.get("flops_note", ""),
        "",
    ]
    with open(md_path, "w") as f:
        f.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ICoT baseline on MLLM benchmarks")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="mllm_data")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_data_idx", type=int, default=0)
    parser.add_argument("--end_data_idx", type=int, default=99999)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--min_pixels", type=int, default=128)
    parser.add_argument("--max_pixels", type=int, default=256)

    # ICoT knobs
    parser.add_argument("--num_selected_patches", type=int, default=16)
    parser.add_argument("--max_sub_imgs", type=int, default=3)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt_suffix", type=str, default="")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--disable_save_logistics", action="store_true")
    parser.add_argument("--use_llm_verify", action="store_true")
    parser.add_argument("--profile_efficiency", action="store_true")
    parser.add_argument("--profile_flops", action="store_true")
    parser.add_argument(
        "--profile_flops_every",
        type=int,
        default=1,
        help="Profile FLOPs every N evaluated samples when --profile_flops is set.",
    )
    parser.add_argument(
        "--profile_output_name",
        type=str,
        default="efficiency_profile.jsonl",
        help="Per-sample JSONL efficiency output inside the run directory.",
    )

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
        attn_implementation="eager",
        token=huggingface_token or None,
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
        f"-patches{args.num_selected_patches}-subimgs{args.max_sub_imgs}"
        f"-icot" + output_suffix
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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    args.profile_efficiency = args.profile_efficiency or args.profile_flops
    profile_jsonl_path = os.path.join(output_dir, args.profile_output_name)
    if args.profile_efficiency and (not args.resume or not entries):
        with open(profile_jsonl_path, "w") as f:
            pass
    efficiency_records = [
        e["efficiency"] for e in entries
        if isinstance(e, dict) and isinstance(e.get("efficiency"), dict)
    ]

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

        try:
            profile_this = (
                args.profile_flops
                and args.profile_flops_every > 0
                and ((total % args.profile_flops_every) == 0)
            )
            gen_kwargs = dict(
                processor=processor,
                model=model,
                image=image,
                question=question,
                model_name=args.model_name_or_path,
                max_new_tokens=args.max_new_tokens,
                num_selected_patches=args.num_selected_patches,
                max_sub_imgs=args.max_sub_imgs,
                verbose=args.verbose > 1,
                return_stats=args.profile_efficiency,
            )
            profiler_estimated_flops = None
            profiler_cpu_time_ms = None
            profiler_cuda_time_ms = None
            _cuda_synchronize(device)
            gen_t0 = time.perf_counter()
            if profile_this:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if str(device).startswith("cuda") and torch.cuda.is_available():
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(
                    activities=activities,
                    with_flops=True,
                    record_shapes=False,
                    profile_memory=False,
                ) as prof:
                    gen_result = generate_icot(**gen_kwargs)
                _cuda_synchronize(device)
                gen_t1 = time.perf_counter()
                profiler_estimated_flops = _sum_profiler_flops(prof)
                profiler_cpu_time_ms = _sum_profiler_time_ms(prof, "cpu_time_total")
                profiler_cuda_time_ms = _sum_profiler_time_ms(prof, "cuda_time_total")
            else:
                gen_result = generate_icot(**gen_kwargs)
                _cuda_synchronize(device)
                gen_t1 = time.perf_counter()

            if args.profile_efficiency:
                response, num_sub_imgs, stop_reason, gen_stats = gen_result
            else:
                response, num_sub_imgs, stop_reason = gen_result
                gen_stats = {}

            efficiency = dict(gen_stats)
            if args.profile_efficiency:
                generation_wall_time_s = float(gen_t1 - gen_t0)
                efficiency.update(dict(
                    data_idx=i,
                    dataset=args.dataset,
                    profiled_flops=bool(profile_this),
                    generation_wall_time_s=generation_wall_time_s,
                    profiler_estimated_flops=profiler_estimated_flops,
                    profiler_cpu_time_total_ms=profiler_cpu_time_ms,
                    profiler_cuda_time_total_ms=profiler_cuda_time_ms,
                    generated_tokens_per_second=(
                        float(gen_stats.get("generated_tokens", 0))
                        / max(generation_wall_time_s, 1e-12)
                    ),
                    forward_passes_per_second=(
                        float(gen_stats.get("forward_passes", 0))
                        / max(generation_wall_time_s, 1e-12)
                    ),
                ))
        except Exception as e:
            print(f"[{i}] ERROR: {e}")
            import traceback
            traceback.print_exc()
            response, num_sub_imgs, stop_reason = "", 0, f"error:{e}"
            efficiency = dict(
                data_idx=i,
                dataset=args.dataset,
                error=str(e),
                profiled_flops=False,
            ) if args.profile_efficiency else {}

        answer = extract_answer(response)

        judge_t0 = time.perf_counter()
        if args.use_llm_verify:
            is_correct = verify_solution_equivalence(answer, true_answer)
        else:
            is_correct = judge_answer_rule(answer, true_answer)
        judge_t1 = time.perf_counter()
        if args.profile_efficiency:
            efficiency["judge_time_s"] = float(judge_t1 - judge_t0)
            efficiency_records.append(efficiency)
            with open(profile_jsonl_path, "a") as f:
                f.write(json.dumps(efficiency, ensure_ascii=False, default=_json_default) + "\n")

        correct += is_correct
        total += 1

        if args.verbose:
            if args.verbose > 1:
                print(f"[{i}] LLM response:\n{response}")
            print(
                f"[{i}] Extracted: {answer}  |  True: {true_answer}  |  "
                f"Correct: {is_correct}  |  subimgs={num_sub_imgs}  stop={stop_reason}"
            )

        if not args.disable_save_logistics:
            entries.append(dict(
                data_idx=i,
                question=question,
                response=response,
                answer=answer,
                true_answer=true_answer,
                is_correct=is_correct,
                num_sub_imgs=num_sub_imgs,
                stop_reason=stop_reason,
                efficiency=efficiency if args.profile_efficiency else None,
            ))
            torch.save({
                "start_idx": i + 1,
                "total": total,
                "correct": correct,
                "entries": entries,
            }, f"{output_dir}/logistics.pt")

        print(f"Running accuracy: {correct}/{total} = {correct / max(1, total):.4f}")

    if total > 0:
        print(f"\n>>> Final: correct={correct}, total={total}, accuracy={correct / total:.4f}")
    print(f">>> Correct indices: {[e['data_idx'] for e in entries if e['is_correct']]}")

    with open(f"{output_dir}/results.log", "a") as f:
        f.write(
            f"Data Idx with Correct Answer: "
            f"{[entry['data_idx'] for entry in entries if entry['is_correct']]}\n"
        )
        f.write(f"correct={correct}, total={total}, accuracy={correct / max(1, total):.4f}\n")
        if args.profile_efficiency:
            f.write(f"efficiency_profile={profile_jsonl_path}\n")
            f.write(f"efficiency_summary={output_dir}/efficiency_summary.json\n")

    if args.profile_efficiency:
        _write_efficiency_summary(output_dir, efficiency_records)


if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        print(f"-- {arg}: {getattr(args, arg)}")
    main(args)
