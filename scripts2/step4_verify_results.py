"""
Step 4: 在有网络的机器 (4090) 上，读取 H200 推理结果并用 LLM 验证。

用法:
    export OPENAI_API_KEY=<YOUR_OPENAI_KEY>
    export OPENAI_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    export MODEL_TYPE=qwen-max

    # 验证 baseline 结果
    python scripts2/step4_verify_results.py ./output/dmlr_vanilla

    # 验证网格搜索结果
    python scripts2/step4_verify_results.py ./output/ltpo_dmlr_grid_dev

    # 只验证网格搜索中某个配置
    python scripts2/step4_verify_results.py ./output/ltpo_dmlr_grid_dev/tokens2_steps15_sigma25.0_decay0.95_lr1e-2_topk10
"""

import os
import sys

import torch
from pydantic import BaseModel
from openai import OpenAI
from tqdm import tqdm


# ---------------------------------------------------------------------------
# LLM 验证 (与 main_vl_dmlr.py 中逻辑一致)
# ---------------------------------------------------------------------------

MODEL_TYPE = os.environ.get("MODEL_TYPE", "qwen-max")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    base_url=os.environ.get("OPENAI_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)


class EquivalenceResult(BaseModel):
    equivalent: bool


def verify_solution_equivalence(solution: str, ground_truth: str) -> bool:
    """LLM-based structured verification (matches DMLR's verify_solution_equivalence)."""
    if not solution or not ground_truth:
        return False
    try:
        resp = client.chat.completions.parse(
            model=MODEL_TYPE,
            messages=[{
                "role": "user",
                "content": (
                    f"Compare the following two answers and decide if they express the same final result."
                    f"Return a json object with field 'equivalent' set to true if they are the same, false otherwise."
                    f"Note that for multiple-choice questions, providing the correct option is counted correct."
                    f"Candidate answer: {solution}\n\n"
                    f"Ground truth: {ground_truth}\n\n"
                ),
            }],
            response_format=EquivalenceResult,
            temperature=0,
        )
        return bool(resp.choices[0].message.parsed.equivalent)
    except Exception as e:
        print(f"  [verify ERROR] {e}")
        return False


# ---------------------------------------------------------------------------
# 查找并验证所有 logistics.pt
# ---------------------------------------------------------------------------

def find_logistics_files(root: str) -> list[str]:
    """递归查找所有 logistics.pt 文件。"""
    results = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f == "logistics.pt":
                results.append(os.path.join(dirpath, f))
    results.sort()
    return results


def verify_one_logistics(logistics_path: str) -> dict:
    """验证一个 logistics.pt 文件，返回统计信息。"""
    data = torch.load(logistics_path, weights_only=False)
    entries = data.get("entries", [])
    if not entries:
        return {"path": logistics_path, "total": 0, "correct": 0, "accuracy": 0.0}

    correct = 0
    total = 0
    for entry in tqdm(entries, desc=os.path.basename(os.path.dirname(logistics_path)), leave=False):
        answer = entry.get("answer", "")
        true_answer = entry.get("true_answer", "")
        is_correct = verify_solution_equivalence(answer, true_answer)
        entry["is_correct_llm"] = is_correct
        correct += is_correct
        total += 1

    accuracy = correct / total if total > 0 else 0.0

    # 保存带 LLM 验证的结果
    data["correct_llm"] = correct
    data["total_llm"] = total
    data["accuracy_llm"] = accuracy
    data["entries"] = entries

    verified_path = logistics_path.replace("logistics.pt", "logistics_llm_verified.pt")
    torch.save(data, verified_path)

    # 追加到 results.log
    results_log = os.path.join(os.path.dirname(logistics_path), "results.log")
    with open(results_log, "a") as f:
        f.write(f"\n[LLM Verified] correct={correct}, total={total}, accuracy={accuracy:.4f}\n")
        f.write(f"[LLM Verified] Correct indices: {[e['data_idx'] for e in entries if e.get('is_correct_llm')]}\n")

    return {"path": logistics_path, "total": total, "correct": correct, "accuracy": accuracy}


def main():
    if len(sys.argv) < 2:
        print("用法: python step4_verify_results.py <output_dir>")
        print("")
        print("示例:")
        print("  python scripts2/step4_verify_results.py ./output/dmlr_vanilla")
        print("  python scripts2/step4_verify_results.py ./output/ltpo_dmlr_grid_dev")
        sys.exit(1)

    output_dir = sys.argv[1]
    if not os.path.isdir(output_dir):
        print(f"[ERROR] 目录不存在: {output_dir}")
        sys.exit(1)

    # 检查环境变量
    if not os.environ.get("OPENAI_API_KEY"):
        print("[ERROR] 请设置 OPENAI_API_KEY 环境变量")
        sys.exit(1)

    logistics_files = find_logistics_files(output_dir)
    if not logistics_files:
        print(f"[ERROR] 在 {output_dir} 下未找到 logistics.pt 文件")
        sys.exit(1)

    print(f"找到 {len(logistics_files)} 个 logistics.pt 文件")
    print(f"API: {os.environ.get('OPENAI_API_BASE_URL', 'default')}")
    print(f"Model: {MODEL_TYPE}")
    print("")

    all_results = []
    for lf in logistics_files:
        rel_path = os.path.relpath(lf, output_dir)
        print(f"验证: {rel_path}")
        result = verify_one_logistics(lf)
        print(f"  -> {result['correct']}/{result['total']} = {result['accuracy']:.4f}")
        all_results.append(result)

    # 汇总 — 按模型分组
    print("")
    print("=" * 60)
    print("汇总 (按模型分组)")
    print("=" * 60)

    from collections import defaultdict
    by_model = defaultdict(list)
    for r in all_results:
        rel = os.path.relpath(r["path"], output_dir)
        # 尝试从路径中提取模型名 (如 Qwen2.5-VL-3B-Instruct/tokens.../...)
        parts = rel.split(os.sep)
        model_name = "unknown"
        for p in parts:
            if "qwen" in p.lower() or "llama" in p.lower() or "mistral" in p.lower():
                model_name = p
                break
        by_model[model_name].append(r)

    for model_name in sorted(by_model.keys()):
        results = by_model[model_name]
        print(f"\n--- {model_name} ---")
        for r in sorted(results, key=lambda x: -x["accuracy"]):
            rel = os.path.relpath(r["path"], output_dir)
            print(f"  {r['accuracy']:.4f}  ({r['correct']}/{r['total']})  {rel}")

    # 保存汇总文件
    summary_path = os.path.join(output_dir, "llm_verify_summary.txt")
    with open(summary_path, "w") as f:
        f.write("LLM Verification Summary\n")
        f.write(f"API: {os.environ.get('OPENAI_API_BASE_URL', 'default')}\n")
        f.write(f"Model: {MODEL_TYPE}\n\n")
        for model_name in sorted(by_model.keys()):
            results = by_model[model_name]
            f.write(f"\n=== {model_name} ===\n")
            for r in sorted(results, key=lambda x: -x["accuracy"]):
                rel = os.path.relpath(r["path"], output_dir)
                f.write(f"{r['accuracy']:.4f}  ({r['correct']}/{r['total']})  {rel}\n")
    print(f"\n汇总已保存到: {summary_path}")


if __name__ == "__main__":
    main()
