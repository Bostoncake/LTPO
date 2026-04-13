#!/bin/bash
# ============================================================
# Step 1: 安装 Python 依赖
# 运行环境: 4090 (有网络)
# ============================================================
set -e

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

echo "========================================"
echo "Step 1: 安装 Python 依赖"
echo "项目根目录: $PROJ_ROOT"
echo "========================================"

echo "[1/3] pip install -r requirements.txt"
pip install -r requirements.txt

echo "[2/3] 安装 latex2sympy"
cd extract_judge_answer/latex2sympy
pip install -e .
cd "$PROJ_ROOT"

echo "[3/3] 安装 math-verify, word2number"
pip install math-verify
pip install word2number

echo ""
echo "依赖安装完成。"
