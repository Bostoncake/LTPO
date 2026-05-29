#!/usr/bin/env python
"""
Re-plot a top-p lookthink visualization from a (possibly hand-edited)
``*_attn_full_grid.txt`` sidecar produced by ``main_vl_dmlr_final_vis.py``.

The full grid in the txt file may have been modified, so on load this
script:
  1. Reads the matrix with np.loadtxt (skipping ``#`` header / footer).
  2. Clips negatives to 0 and re-normalises the grid to sum to 1.
  3. Re-runs the top-p selection used by the live pipeline
     (mirrors ``_pool_top_p_visual_tokens`` in ``ltpo_vl_dmlr_final_vis.py``):
     smallest set of cells whose cumulative descending weights >= top_p.
  4. Renders the same style as ``_save_lookthink_vis`` after the recent
     edits: image background, red heatmap of *selected-only*
     renormalised weights, green boxes around the selected cells, no
     numeric labels, no title.

Usage:
    python scripts/replot_lookthink_from_grid.py \\
        path/to/look_KK_step_SS_attn_full_grid.txt \\
        [--image path/to/source.png]                # optional
        [--top_p 0.9]                                # default: from header
        [--out path/to/out.png]                      # default: <txt>_replot.png

If ``--image`` is omitted, the script tries to auto-load the sample
image via ``data_vl.get_mllm_dataset(dataset)[sample_idx]`` using the
layout that ``main_vl_dmlr_final_vis.py`` writes:
  .../_vis/<dataset>/sample_<NNNNN>/look_KK_step_SS_attn_full_grid.txt
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


# Mirror main_vl_dmlr_final_vis.py: Qwen2.5-VL uses patch_size=14.
_VIS_PATCH_PX = 14


_HDR = {
    "grid": re.compile(r"#\s*grid_shape:\s*(\d+)\s*x\s*(\d+)"),
    "thw":  re.compile(r"image_grid_thw\(T,H,W\):\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)"),
    "merge": re.compile(r"spatial_merge_size:\s*(\d+)"),
    "top_p": re.compile(r"#\s*top_p:\s*([0-9.eE+-]+)"),
}


def _parse_header(txt_path: str) -> dict:
    meta = {
        "grid_h": None, "grid_w": None,
        "T_p": 1, "H_p": None, "W_p": None,
        "merge": 2, "top_p": None,
    }
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                # Header is a single contiguous block; stop at first data row.
                break
            m = _HDR["grid"].search(line)
            if m:
                meta["grid_h"] = int(m.group(1))
                meta["grid_w"] = int(m.group(2))
            m = _HDR["thw"].search(line)
            if m:
                meta["T_p"] = int(m.group(1))
                meta["H_p"] = int(m.group(2))
                meta["W_p"] = int(m.group(3))
            m = _HDR["merge"].search(line)
            if m:
                meta["merge"] = int(m.group(1))
            m = _HDR["top_p"].search(line)
            if m:
                meta["top_p"] = float(m.group(1))
    return meta


def _load_grid(txt_path: str, grid_h: int | None, grid_w: int | None) -> np.ndarray:
    grid = np.loadtxt(txt_path, dtype=np.float32, comments="#")
    if grid.ndim == 1:
        if grid_h is None or grid_w is None:
            raise ValueError(
                "Grid loaded as 1D and header has no grid_shape; "
                "cannot infer (grid_h, grid_w). Restore the header line "
                "'# grid_shape: H x W' or reshape the matrix by hand."
            )
        if grid.size != grid_h * grid_w:
            raise ValueError(
                f"1D grid size {grid.size} != grid_h*grid_w "
                f"({grid_h}*{grid_w}={grid_h*grid_w})."
            )
        grid = grid.reshape(grid_h, grid_w)
    if grid_h is not None and grid_w is not None and grid.shape != (grid_h, grid_w):
        if grid.size == grid_h * grid_w:
            grid = grid.reshape(grid_h, grid_w)
        else:
            raise ValueError(
                f"Loaded grid shape {grid.shape} does not match header "
                f"({grid_h}, {grid_w})."
            )
    return grid


def _topp_select(weights_flat: np.ndarray, top_p: float) -> np.ndarray:
    """Mirror of ``_pool_top_p_visual_tokens``. Returns the kept flat
    indices in attention-descending order."""
    n = int(weights_flat.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    p = float(max(0.0, min(1.0, top_p)))
    order = np.argsort(-weights_flat, kind="stable")
    sorted_w = weights_flat[order]
    if p <= 0.0:
        keep_n = 1
    elif p >= 1.0:
        keep_n = n
    else:
        cum = np.cumsum(sorted_w)
        keep_n = int((cum < p).sum()) + 1
    return order[:keep_n]


def _auto_find_image(
    txt_path: str,
    dataset_override: str | None,
    image_root_override: str | None,
) -> Image.Image | None:
    """
    Best-effort auto-load: parses sample_<NNNNN> from the parent dir and
    dataset from the parent-of-parent dir, then loads via data_vl.
    Returns None on failure; caller must then pass --image explicitly.
    """
    sample_dir = os.path.dirname(os.path.abspath(txt_path))
    m = re.match(r"sample_(\d+)$", os.path.basename(sample_dir))
    if not m:
        return None
    sample_idx = int(m.group(1))
    dataset = dataset_override or os.path.basename(os.path.dirname(sample_dir))

    # Try to import LTPO data loader (relative to repo root).
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from data_vl import get_mllm_dataset
    except Exception as e:
        print(f"[replot] cannot import data_vl ({e}); pass --image explicitly.",
              file=sys.stderr)
        return None

    data_root = os.path.join(repo_root, "mllm_data")
    try:
        ds = get_mllm_dataset(dataset, data_root=data_root)
        example = ds[sample_idx]
    except Exception as e:
        print(f"[replot] dataset load failed ({dataset}[{sample_idx}]): {e}",
              file=sys.stderr)
        return None

    ipath = example.get("image_path")
    if not ipath:
        return None
    # Try a few common image-root resolutions.
    candidates = []
    if image_root_override:
        candidates.append(os.path.join(image_root_override, ipath))
    candidates.append(ipath)
    candidates.append(os.path.join(repo_root, ipath))
    candidates.append(os.path.join(repo_root, "mllm_data", ipath))
    for full in candidates:
        if full and os.path.isabs(full) and os.path.exists(full):
            return Image.open(full)
        if full and os.path.exists(full):
            return Image.open(full)
    print(f"[replot] could not locate image file {ipath} (tried {candidates}).",
          file=sys.stderr)
    return None


def replot(
    txt_path: str,
    image_path: str | None,
    out_path: str | None,
    top_p_override: float | None,
    dataset_override: str | None,
    image_root_override: str | None,
) -> str:
    meta = _parse_header(txt_path)
    grid = _load_grid(txt_path, meta["grid_h"], meta["grid_w"])
    if meta["grid_h"] is None or meta["grid_w"] is None:
        meta["grid_h"], meta["grid_w"] = int(grid.shape[0]), int(grid.shape[1])
    grid_h = int(meta["grid_h"])
    grid_w = int(meta["grid_w"])

    # Re-normalise the (possibly modified) grid so non-negative cells
    # sum to 1 over the frame.
    grid = grid.astype(np.float32)
    grid = np.clip(grid, a_min=0.0, a_max=None)
    total = float(grid.sum())
    if total > 0.0:
        grid = grid / total

    flat = grid.reshape(-1)
    top_p = top_p_override if top_p_override is not None else meta["top_p"]
    if top_p is None:
        top_p = 0.9

    keep_indices = _topp_select(flat, float(top_p))
    selected_flat = flat[keep_indices]
    sel_sum = float(selected_flat.sum())
    sel_norm = (
        selected_flat / sel_sum if sel_sum > 0 else selected_flat
    )

    selected_grid = np.zeros_like(grid)
    for w, idx in zip(sel_norm, keep_indices):
        r, c = divmod(int(idx), grid_w)
        selected_grid[r, c] = float(w)

    # Load source image.
    if image_path:
        img = Image.open(image_path)
    else:
        img = _auto_find_image(txt_path, dataset_override, image_root_override)
    if img is None:
        raise RuntimeError(
            "Could not load source image. Pass --image <path> explicitly "
            "(or set --dataset / --image_root)."
        )

    H_p = int(meta["H_p"]) if meta["H_p"] is not None else grid_h * int(meta["merge"])
    W_p = int(meta["W_p"]) if meta["W_p"] is not None else grid_w * int(meta["merge"])
    merge = int(meta["merge"])
    img_w_px = W_p * _VIS_PATCH_PX
    img_h_px = H_p * _VIS_PATCH_PX
    img_rgb = img.convert("RGB").resize((img_w_px, img_h_px))
    cell_px = merge * _VIS_PATCH_PX

    fig_w_in = max(4.0, img_w_px / 100.0)
    fig_h_in = max(4.0, img_h_px / 100.0)
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=120)
    ax.imshow(img_rgb)

    vmax = float(selected_grid.max()) if selected_grid.size > 0 else 0.0
    if vmax <= 0.0:
        vmax = 1e-8
    heat = np.zeros((img_h_px, img_w_px), dtype=np.float32)
    for r in range(grid_h):
        for c in range(grid_w):
            heat[r * cell_px:(r + 1) * cell_px,
                 c * cell_px:(c + 1) * cell_px] = selected_grid[r, c]
    ax.imshow(heat, cmap="Reds", alpha=0.45, vmin=0.0, vmax=vmax)

    for idx in keep_indices:
        r, c = divmod(int(idx), grid_w)
        rect = mpatches.Rectangle(
            (c * cell_px, r * cell_px), cell_px, cell_px,
            linewidth=1.6, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)
    ax.axis("off")

    if out_path is None:
        out_path = os.path.splitext(txt_path)[0] + "_replot.png"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

    print(
        f"[replot] wrote {out_path}\n"
        f"         top_p={float(top_p):.4f}  grid={grid_h}x{grid_w}  "
        f"n_selected={len(keep_indices)} / {grid_h*grid_w}  "
        f"mass_of_selected_in_renormalised_grid={sel_sum:.6f}"
    )
    return out_path


def parse_args():
    ap = argparse.ArgumentParser(
        description=("Re-plot a top-p lookthink visualization from a "
                     "(possibly hand-edited) attn_full_grid.txt sidecar."),
    )
    ap.add_argument(
        "txt_path",
        type=str,
        help="path to a *_attn_full_grid.txt produced by main_vl_dmlr_final_vis.py",
    )
    ap.add_argument(
        "--image",
        type=str,
        default=None,
        help=("Path to the source image. If omitted, the script tries to "
              "auto-load it via data_vl using the sample idx parsed from "
              "the parent dir name (sample_<NNNNN>) and the dataset name "
              "from the parent-of-parent dir."),
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path. Defaults to <txt>_replot.png alongside the input.",
    )
    ap.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Override the top_p stored in the txt header.",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(auto-load mode) dataset name override.",
    )
    ap.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="(auto-load mode) image_root override.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    replot(
        txt_path=args.txt_path,
        image_path=args.image,
        out_path=args.out,
        top_p_override=args.top_p,
        dataset_override=args.dataset,
        image_root_override=args.image_root,
    )


if __name__ == "__main__":
    main()
