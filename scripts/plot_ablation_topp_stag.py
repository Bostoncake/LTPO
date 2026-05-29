"""Plot two ablation figures:
  (1) Performance vs. top-p (the p value used in nucleus sampling).
  (2) Performance vs. stagnation step N.

Fill in the x-values and y-values in the marked sections below, then run:
    python scripts/plot_ablation_topp_stag.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt

# ============================================================
# Global font size
# ============================================================
FONT_SIZE = 18
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
})

# ============================================================
# Output configuration
# ============================================================
OUT_DIR = Path("output/ablation_plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Name of the metric you are plotting on the y-axis (e.g. "Accuracy (%)").
METRIC_NAME = "Accuracy (%)"  # TODO: change to your metric

# ============================================================
# (1) Effect of top-p
# ============================================================
# # TODO: fill in the p values you swept over.
# TOPP_X = [
#     0.1, 0.3, 0.5, 0.7, 0.9,
# ]
# # TODO: fill in the corresponding metric values (same length as TOPP_X).
# TOPP_Y = [
#     56.3, 57.0, 58.3, 58.3, 58.0,
# ]

# # ============================================================
# # (2) Effect of stagnation step N
# # ============================================================
# # TODO: fill in the N values you swept over.
# STAG_X = [
#     1, 2, 3, 4, 5,
# ]
# # TODO: fill in the corresponding metric values (same length as STAG_X).
# STAG_Y = [
#     57.7, 58.0, 57.0, 56.3, 55.0,
# ]

# TODO: fill in the p values you swept over.
TOPP_X = [
    0.1, 0.3, 0.5, 0.7, 0.9,
]
# TODO: fill in the corresponding metric values (same length as TOPP_X).
TOPP_Y = [
    21.3, 21.3, 22.0, 23.0, 23.3,
]

# ============================================================
# (2) Effect of stagnation step N
# ============================================================
# TODO: fill in the N values you swept over.
STAG_X = [
    1, 2, 3, 4, 5,
]
# TODO: fill in the corresponding metric values (same length as STAG_X).
STAG_Y = [
    22.0, 22.0, 21.3, 23.0, 23.3,
]


def _plot(
    xs: list,
    ys: list,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    if len(xs) == 0 or len(ys) == 0:
        print(f"[skip] {out_path.name}: x or y list is empty — fill in values first.")
        return
    if len(xs) != len(ys):
        raise ValueError(
            f"x ({len(xs)}) and y ({len(ys)}) must have the same length for {out_path.name}"
        )

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(xs, ys, marker="o", linewidth=2.0, color="#1f77b4")
    # ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] {out_path}")


def main() -> None:
    dataset = "MathVision"
    _plot(
        TOPP_X,
        TOPP_Y,
        title=f"{dataset} Performance",
        xlabel=r"top-$p$",
        ylabel=METRIC_NAME,
        out_path=OUT_DIR / f"ablation_topp_{dataset}.pdf",
    )
    _plot(
        STAG_X,
        STAG_Y,
        title=f"{dataset} Performance",
        xlabel=r"Stagnation step $N$",
        ylabel=METRIC_NAME,
        out_path=OUT_DIR / f"ablation_stag_step_{dataset}.pdf",
    )


if __name__ == "__main__":
    main()
