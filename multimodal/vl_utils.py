"""
Utility helpers for VL LTPO — answer extraction and judging.
"""
import re

import torch


def extract_answer(text):
    """
    Extract the final answer from a model response.

    Priority: <answer>...</answer> > last \\boxed{...} > raw text.
    """
    if not text:
        return ""

    # 1) <answer>...</answer>
    low = text.lower()
    start = low.find("<answer>")
    end = low.find("</answer>")
    if start != -1 and end != -1 and end > start:
        ans = text[start + len("<answer>"):end].strip().strip("$")
        ans = re.sub(r"\\displaystyle\s*", "", ans)
        ans = re.sub(r"\s+", " ", ans).strip()
        if ans:
            return ans

    # 2) Last balanced \\boxed{...}
    boxed_contents = []
    for m in re.finditer(r"\\boxed\s*\{", text):
        open_pos = text.find("{", m.end() - 1)
        if open_pos == -1:
            continue
        depth, i = 0, open_pos
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    boxed = text[open_pos + 1:i].strip().strip("$")
                    boxed = re.sub(r"\\displaystyle\s*", "", boxed)
                    boxed = re.sub(r"\s+", " ", boxed).strip()
                    if boxed:
                        boxed_contents.append(boxed)
                    break
            i += 1
    if boxed_contents:
        return boxed_contents[-1]

    return text.strip()


def extract_true_answer(text, name=""):
    """Return ground-truth answer as-is (JSON datasets are pre-formatted)."""
    return text


def judge_answer(response, label, data_name="", extract=True, prompt_idx=0):
    """
    Judge whether the extracted answer matches the ground truth.

    Uses exact match, case-insensitive match, and substring containment.
    """
    if extract:
        response = extract_answer(response)
    r = str(response).strip()
    l = str(label).strip()
    if r == l:
        return True
    if r.upper() == l.upper():
        return True
    if l and l in r:
        return True
    return False


def args_to_dict(args):
    """Convert an ``argparse.Namespace`` to a JSON-serialisable dict."""
    result = {}
    for key, value in vars(args).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            result[key] = value
        elif isinstance(value, torch.Tensor):
            result[key] = value.item() if value.numel() == 1 else value.tolist()
        else:
            try:
                result[key] = str(value)
            except Exception:
                result[key] = None
    return result
