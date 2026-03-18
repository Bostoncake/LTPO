"""
Data loading for Vision-Language LTPO.

Loads datasets from JSON files following the DMLR data format:
  [{"prompt": "...", "solution": "...", "image_path": "...", "idx": 0}, ...]
"""
import json
import os
from multiprocessing import Pool, cpu_count

from datasets import Dataset

from .vl_prompts import vl_cot_prompt


def _process_item(item, index):
    """Normalise a single JSON record into the standard column schema."""
    return {
        "question": item.get("prompt", ""),
        "answer": item.get("solution", ""),
        "idx": item.get("idx", index),
        "image_path": item.get("image_path", ""),
        "image": item.get("image_path", ""),
    }


def load_json_dataset(json_path, num_workers=None, start_idx=None, end_idx=None):
    """
    Load a dataset from a JSON file.

    Args:
        json_path: Path to JSON file.
        num_workers: Parallel workers (default: min(cpu_count, 16)).
        start_idx: Optional inclusive start index.
        end_idx: Optional exclusive end index.

    Returns:
        A HuggingFace ``Dataset``.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"JSON must contain a list, got {type(data)}")

    total = len(data)
    base_index = 0
    if start_idx is not None and end_idx is not None:
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, total)
        base_index = start_idx
        data = data[start_idx:end_idx]

    if num_workers is None:
        num_workers = min(cpu_count(), 16)

    if num_workers > 1 and len(data) > 100:
        with Pool(processes=num_workers) as pool:
            processed = pool.starmap(
                _process_item, [(item, base_index + i) for i, item in enumerate(data)]
            )
    else:
        processed = [_process_item(item, base_index + i) for i, item in enumerate(data)]

    return Dataset.from_list(processed)


def get_vl_dataset(data_name_or_path, processor, prompt_idx=0, start_idx=None, end_idx=None):
    """
    High-level loader for VL datasets stored as JSON.

    Args:
        data_name_or_path: Path to a ``.json`` file.
        processor: VL processor (used for compatibility, not applied eagerly).
        prompt_idx: Prompt variant index (0/1/2).
        start_idx / end_idx: Optional subset bounds.

    Returns:
        A ``Dataset`` with lazy per-example transforms that add ``messages``
        and ``image`` columns on access.
    """
    if not (data_name_or_path.endswith(".json") and os.path.exists(data_name_or_path)):
        raise ValueError(
            f"Expected an existing JSON file, got: {data_name_or_path}"
        )

    dataset = load_json_dataset(data_name_or_path, start_idx=start_idx, end_idx=end_idx)

    def _add_messages(example):
        q = example["question"]
        q_prompted = vl_cot_prompt(q, prompt_idx=prompt_idx)

        image_data = example.get("image", None)
        example["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q_prompted},
                ],
            }
        ]
        example["image"] = image_data
        return example

    dataset.set_transform(_add_messages)
    return dataset
