"""
Huggingface dataset: Lin-Chen/MMStar
{
"source": "MMBench",
"split": "val",
"image_path": "images/0.jpg"
}
put the images in this directory, like: dataset/mmstar/images/val/0.jpg
"""

import os
from datasets import load_dataset

def download_mmstar_images():
    dataset = load_dataset("Lin-Chen/MMStar", split="val")

    out_dir = os.path.join(os.path.dirname(__file__), "dataset", "mmstar", "images", "val")
    os.makedirs(out_dir, exist_ok=True)

    for item in dataset:
        image_path = item["meta_info"]["image_path"]  # e.g. "images/0.jpg"
        filename = os.path.basename(image_path)  # e.g. "0.jpg"
        out_path = os.path.join(out_dir, filename)
        if not os.path.exists(out_path):
            item["image"].save(out_path)

    print(f"Saved {len(dataset)} images to {out_dir}")

if __name__ == "__main__":
    download_mmstar_images()
