import os
from datasets import load_dataset

def download_mmstar_images():
    dataset = load_dataset("lmms-lab/HallusionBench", split="image")

    out_dir = os.path.join(os.path.dirname(__file__), "dataset", "hallusion_bench", "hallusion_bench")
    os.makedirs(out_dir, exist_ok=True)

    for item in dataset:
        filename = item["filename"].lstrip("./")  # e.g. "VS/chart/11_1.png"
        out_path = os.path.join(out_dir, filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if not os.path.exists(out_path):
            item["image"].save(out_path)

    print(f"Saved {len(dataset)} images to {out_dir}")

if __name__ == "__main__":
    download_mmstar_images()
