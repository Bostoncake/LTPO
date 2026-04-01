"""
Data loader for MLLM (Vision-Language) datasets stored in mllm_data/.
"""
import json
import os

from datasets import Dataset

MLLM_DATASET_NAMES = ['hallusion', 'math_vision', 'math_vista', 'mm_math', 'mmstar', 'mmvp', 'scienceqa']


def is_mllm_dataset(data_name: str) -> bool:
    return any(n in data_name.lower() for n in MLLM_DATASET_NAMES)


def get_mllm_dataset(data_name: str, data_root: str = 'mllm_data') -> Dataset:
    """
    Load an MLLM dataset from mllm_data/<name>.json.

    Each item in the JSON has keys: prompt, solution, image_path, idx
    (mm_math and mmvp additionally have a 'completion' key, which is ignored).

    Returns a Dataset with columns:
        question    (str)  – the text prompt / question
        answer      (str)  – ground-truth solution string
        image_path  (str)  – path to the image file
    """
    dataset_key = None
    for name in MLLM_DATASET_NAMES:
        if name in data_name.lower():
            dataset_key = name
            break
    if dataset_key is None:
        raise ValueError(
            f"Unsupported MLLM dataset: {data_name}. "
            f"Supported: {MLLM_DATASET_NAMES}"
        )

    json_path = os.path.join(data_root, f"{data_name}.json")
    with open(json_path, 'r') as f:
        data = json.load(f)

    questions, answers, image_paths = [], [], []
    for item in data:
        questions.append(item['prompt'])
        answers.append(item['solution'])
        image_paths.append(item['image_path'])

    return Dataset.from_dict({
        'question': questions,
        'answer': answers,
        'image_path': image_paths,
    })
