"""
Create 300-sample dev splits for each benchmark.

Randomly samples 300 items from each dataset JSON with a fixed seed (42)
for reproducibility. MMVP already has exactly 300 samples so it is copied
as-is. Run once from the repo root:

    python mllm_data/create_dev_splits.py
"""
import json
import os
import random

DATASETS = ['hallusion', 'math_vision', 'math_vista', 'mm_math', 'mmstar', 'mmvp', 'scienceqa']
DEV_SIZE = 300
SEED = 42
DATA_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    rng = random.Random(SEED)
    for name in DATASETS:
        src = os.path.join(DATA_ROOT, f"{name}.json")
        dst = os.path.join(DATA_ROOT, f"{name}_dev.json")

        with open(src, 'r') as f:
            data = json.load(f)

        if len(data) <= DEV_SIZE:
            dev = data
            print(f"{name}: {len(data)} samples (already <= {DEV_SIZE}, using all)")
        else:
            dev = rng.sample(data, DEV_SIZE)
            print(f"{name}: sampled {DEV_SIZE}/{len(data)} items -> {dst}")

        with open(dst, 'w') as f:
            json.dump(dev, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
