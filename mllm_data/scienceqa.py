import os
import json
from datasets import load_dataset

def download_mmstar_images():
    dataset = load_dataset("lmms-lab/ScienceQA-IMG", split="test")

    out_dir = os.path.join(os.path.dirname(__file__), "dataset", "scienceqa", "images", "test")
    os.makedirs(out_dir, exist_ok=True)

    with open("/home/xiongyizhe/research/LTPO/mllm_data/scienceqa.json", "r") as f:
        local_cont = json.loads(f.read())
    for idx, item in enumerate(dataset):
        cur_item_local_cont = local_cont[idx]
        filename = os.path.basename(cur_item_local_cont["image_path"])
        out_path = os.path.join(out_dir, filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not os.path.exists(out_path):
            q = item["question"]
            p = cur_item_local_cont["prompt"]
            if q in p:
                item["image"].save(out_path)
            else:
                print(f"WARNING: idx={idx}, online question: {q}, local prompt: {p}")
                import pdb; pdb.set_trace()
        else:
            raise NotImplementedError

    print(f"Saved {len(dataset)} images to {out_dir}")

if __name__ == "__main__":
    download_mmstar_images()