#!/bin/bash
# ============================================================
# Step 2: 下载所有数据集图像
# 运行环境: 4090 (有网络)
#
# 下载完成后，在项目根目录创建 dataset/ 符号链接指向 mllm_data/dataset/
# 使得 --image_root . 能正确找到图像。
# ============================================================
set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

echo "========================================"
echo "Step 2: 下载数据集图像"
echo "项目根目录: $PROJ_ROOT"
echo "========================================"

# ------------------------------------------------------------------
# 2a. Python 下载脚本（从 HuggingFace datasets 库下载）
# ------------------------------------------------------------------

echo ""
echo "[1/7] hallusion (HuggingFace datasets)"
python mllm_data/hallusion.py

echo ""
echo "[2/7] mmstar (HuggingFace datasets)"
python mllm_data/mmstar.py

echo ""
echo "[3/7] math_vision (HuggingFace datasets)"
python mllm_data/math_vision.py

echo ""
echo "[4/7] scienceqa (HuggingFace datasets)"
# scienceqa.py 中有硬编码路径，用修复版本
python scripts2/download_scienceqa_fixed.py

# ------------------------------------------------------------------
# 2b. Git clone 下载（需要 git lfs）
# ------------------------------------------------------------------

echo ""
echo "[5/7] mmvp (git clone)"
if [ ! -d "mllm_data/dataset/mmvp" ]; then
    cd /tmp
    git clone https://huggingface.co/datasets/MMVP/MMVP mmvp_download || true
    mkdir -p "$PROJ_ROOT/mllm_data/dataset/mmvp"
    if [ -d "mmvp_download/MMVP Images" ]; then
        cp -r "mmvp_download/MMVP Images" "$PROJ_ROOT/mllm_data/dataset/mmvp/MMVP Images"
    elif [ -d "mmvp_download/MMVP images" ]; then
        cp -r "mmvp_download/MMVP images" "$PROJ_ROOT/mllm_data/dataset/mmvp/MMVP Images"
    fi
    rm -rf mmvp_download
    cd "$PROJ_ROOT"
    echo "mmvp 下载完成"
else
    echo "mmvp 已存在，跳过"
fi

echo ""
echo "[6/7] math_vista (git clone + unzip)"
if [ ! -d "mllm_data/dataset/math_vista" ]; then
    cd /tmp
    git clone https://huggingface.co/datasets/AI4Math/MathVista math_vista_download || true
    mkdir -p "$PROJ_ROOT/mllm_data/dataset/math_vista"
    if [ -f "math_vista_download/images.zip" ]; then
        cd math_vista_download
        unzip -q images.zip
        cp -r images "$PROJ_ROOT/mllm_data/dataset/math_vista/images"
        cd /tmp
    fi
    rm -rf math_vista_download
    cd "$PROJ_ROOT"
    echo "math_vista 下载完成"
else
    echo "math_vista 已存在，跳过"
fi

echo ""
echo "[7/7] mm_math (git clone + unzip)"
if [ ! -d "mllm_data/dataset/mm_math" ]; then
    cd /tmp
    git clone https://huggingface.co/datasets/THU-KEG/MM_Math mm_math_download || true
    mkdir -p "$PROJ_ROOT/mllm_data/dataset/mm_math/images/MM_Math"
    if [ -d "mm_math_download/MM_Math" ]; then
        cd mm_math_download/MM_Math
        if [ -f "MM_Math.zip" ]; then
            unzip -q MM_Math.zip
        fi
        cp -r MM_Math "$PROJ_ROOT/mllm_data/dataset/mm_math/images/MM_Math/MM_Math"
        cd /tmp
    fi
    rm -rf mm_math_download
    cd "$PROJ_ROOT"
    echo "mm_math 下载完成"
else
    echo "mm_math 已存在，跳过"
fi

# ------------------------------------------------------------------
# 2c. 验证
# ------------------------------------------------------------------
# 运行时 --image_root mllm_data，所以实际路径为 mllm_data/<image_path>

echo ""
echo "========================================"
echo "验证图像文件"
echo "========================================"

python3 -c "
import json, os

IMAGE_ROOT = 'mllm_data'
datasets = ['mmvp', 'mmstar', 'mm_math', 'math_vista', 'math_vision', 'hallusion', 'scienceqa']
for ds in datasets:
    data = json.load(open(f'mllm_data/{ds}.json'))
    total = len(data)
    found = sum(1 for item in data if os.path.exists(os.path.join(IMAGE_ROOT, item['image_path'])))
    status = 'OK' if found == total else 'MISSING'
    print(f'  {ds:15s}: {found}/{total} images found  [{status}]')
"

echo ""
echo "数据下载完成。如果有 MISSING，请检查对应数据集的下载脚本。"
