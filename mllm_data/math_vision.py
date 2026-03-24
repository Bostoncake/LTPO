import os
from datasets import load_dataset

def download_mmstar_images():
    dataset = load_dataset("MathLLMs/MathVision", split="test")

    out_dir = os.path.join(os.path.dirname(__file__), "dataset", "math_vision", "dataset", "math_vision", "images")
    os.makedirs(out_dir, exist_ok=True)

    for item in dataset:
        image_path = item["image"]  # e.g. "images/0.jpg"
        filename = os.path.splitext(os.path.basename(image_path))[0] + ".png"
        out_path = os.path.join(out_dir, filename)
        if not os.path.exists(out_path):
            item["decoded_image"].save(out_path, format="PNG")

    print(f"Saved {len(dataset)} images to {out_dir}")

if __name__ == "__main__":
    download_mmstar_images()
