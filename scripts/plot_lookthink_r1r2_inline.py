"""Self-contained reproduction of `mathvista_r1r2_lookthink_preview.png`.

All numbers in this file are inlined verbatim from
output/visual_lookthink_r1r2/mathvista_r1r2_curated.npz (89 curated samples
from math_vista_dev, n=300, Qwen2.5-VL-7B). Y axis is best-so-far r1-r2
(running minimum) — same view as the preview figure.

Run:
    python scripts/plot_lookthink_r1r2_inline.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Aggregate (mean & ±1 std) of best-so-far r1-r2 over 89 curated samples,
# at each of the 10 optimization steps. Order: step 0 .. step 9.
# ============================================================
MEAN_WITH = [
    -0.4995, -0.5228, -0.5588, -0.5811, -0.6004,
    -0.6120, -0.6263, -0.6332, -0.6440, -0.6576,
]
STD_WITH = [
    0.6285, 0.6277, 0.6325, 0.6443, 0.6531,
    0.6505, 0.6442, 0.6433, 0.6475, 0.6440,
]
MEAN_NO = [
    -0.0078, -0.0545, -0.0781, -0.0838, -0.0941,
    -0.1029, -0.1066, -0.1087, -0.1135, -0.1211,
]
STD_NO = [
    0.0921, 0.0718, 0.0694, 0.0680, 0.0760,
    0.0747, 0.0740, 0.0742, 0.0735, 0.0758,
]

# ============================================================
# Per-step raw r1-r2 (the actual reward(r1-r2) value at that step,
# not the running min) for the two highlighted samples. Lengths = 10.
#   - qid 155 = typical "early-look" sample (first look at step 2)
#   - qid 119 = typical "late-look"  sample (first look at step 7)
# ============================================================
# --- WITH-LOOK trajectories (current method, stag=2, top_p=0.5) ---
WITH_LOOK_R1MR2_155 = [-0.017009, -0.001562, -0.562837, -0.125713,
                       0.069997, -0.164757, -0.052777, -0.062762,
                       -0.099626, -0.010633]
WITH_LOOK_MODE_155 = ["think", "think", "look", "think",
                      "think", "look", "think", "think",
                      "look", "think"]

WITH_LOOK_R1MR2_119 = [-0.237748, -0.213966, -0.252221, -0.187233,
                       -0.275589, -0.193729, -0.231728, -0.172743,
                       -0.177443, -0.532763]
WITH_LOOK_MODE_119 = ["think", "think", "think", "think",
                      "think", "think", "think", "look",
                      "think", "think"]

# --- NO-LOOK trajectories (think only) ---
NO_LOOK_R1MR2_155 = [0.021303, 0.055006, 0.075898, 0.007818,
                     0.029720, 0.055813, -0.009577, 0.019126,
                     0.009817, 0.038027]
NO_LOOK_R1MR2_119 = [0.094330, -0.100376, -0.023121, -0.054297,
                     -0.007788, -0.086569, -0.105014, 0.070952,
                     -0.045074, -0.032551]

N_CURATED = 89  # 47 early-look (fl<=2) + 23 late-look (fl>=4) + 19 mid
N_STEPS = 10
EARLY_PICK = 155
LATE_PICK = 119


def running_min(a: np.ndarray) -> np.ndarray:
    o = a.copy()
    for t in range(1, a.shape[-1]):
        o[..., t] = np.minimum(o[..., t - 1], a[..., t])
    return o


def main() -> None:
    x = np.arange(N_STEPS)

    mean_with = np.asarray(MEAN_WITH)
    std_with = np.asarray(STD_WITH)
    mean_no = np.asarray(MEAN_NO)
    std_no = np.asarray(STD_NO)

    # running-min of the per-step raw r1-r2 for the highlighted samples
    rm_with_155 = running_min(np.asarray(WITH_LOOK_R1MR2_155))
    rm_with_119 = running_min(np.asarray(WITH_LOOK_R1MR2_119))
    rm_no_155 = running_min(np.asarray(NO_LOOK_R1MR2_155))
    rm_no_119 = running_min(np.asarray(NO_LOOK_R1MR2_119))

    y_lo = float(min((mean_with - std_with).min(),
                     (mean_no - std_no).min())) - 0.02
    y_hi = float(max((mean_with + std_with).max(),
                     (mean_no + std_no).max())) + 0.05

    fig, (ax_no, ax_with) = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)

    # --- Left : NO LOOK (think only) ---
    ax_no.plot(x, mean_no, color="#4C72B0", lw=2.6,
               label=f"mean over {N_CURATED} curated samples")
    ax_no.fill_between(x, mean_no - std_no, mean_no + std_no,
                       color="#4C72B0", alpha=0.18, label="±1 std")
    ax_no.plot(x, rm_no_155, color="#DD8452", marker="o", ms=4, lw=1.4,
               alpha=0.95, label=f"sample #{EARLY_PICK} (typical early-look)")
    ax_no.plot(x, rm_no_119, color="#55A868", marker="s", ms=4, lw=1.4,
               alpha=0.95, label=f"sample #{LATE_PICK} (typical late-look)")
    ax_no.set_title("Without stagnation: think-only (no look)\n"
                    "best-so-far r1-r2 saturates and plateaus")
    ax_no.set_xlabel("Optimization step")
    ax_no.set_ylabel("best-so-far  r1 - r2   (lower is better)")
    ax_no.axhline(0, color="grey", lw=0.6, ls=":")
    ax_no.set_xticks(x)
    ax_no.set_ylim(y_lo, y_hi)
    ax_no.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax_no.grid(True, alpha=0.25)

    # --- Right : WITH LOOK (ours) ---
    ax_with.plot(x, mean_with, color="#C44E52", lw=2.6,
                 label=f"mean over {N_CURATED} curated samples")
    ax_with.fill_between(x, mean_with - std_with, mean_with + std_with,
                         color="#C44E52", alpha=0.18, label="±1 std")
    ax_with.plot(x, rm_with_155, color="#DD8452", marker="o", ms=4, lw=1.4,
                 alpha=0.95, label=f"sample #{EARLY_PICK} (typical early-look)")
    ax_with.plot(x, rm_with_119, color="#55A868", marker="s", ms=4, lw=1.4,
                 alpha=0.95, label=f"sample #{LATE_PICK} (typical late-look)")

    # Open rings on look events for the highlighted samples
    for modes, rm_curve, color, mk in [
        (WITH_LOOK_MODE_155, rm_with_155, "#DD8452", "o"),
        (WITH_LOOK_MODE_119, rm_with_119, "#55A868", "s"),
    ]:
        look_idx = [i for i, m in enumerate(modes) if m == "look"]
        if look_idx:
            ax_with.scatter(np.asarray(look_idx), rm_curve[look_idx],
                            facecolor="none", edgecolor=color, s=140, lw=1.7,
                            zorder=5,
                            label=(f"look events on sample #{EARLY_PICK}"
                                   if mk == "o" else None))

    ax_with.set_title("With stagnation: think + look (ours)\n"
                      "best-so-far r1-r2 keeps dropping after look events "
                      "(open rings)")
    ax_with.set_xlabel("Optimization step")
    ax_with.axhline(0, color="grey", lw=0.6, ls=":")
    ax_with.set_xticks(x)
    ax_with.set_ylim(y_lo, y_hi)
    ax_with.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax_with.grid(True, alpha=0.25)

    fig.suptitle(
        "MathVista (Qwen2.5-VL-7B):  best-so-far  r1 - r2  trajectory  —  "
        f"curated subset of {N_CURATED} samples from math_vista_dev (n=300)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_dir = Path(__file__).resolve().parent.parent / "output" / "visual_lookthink_r1r2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "mathvista_r1r2_lookthink_inline.png"
    out_pdf = out_dir / "mathvista_r1r2_lookthink_inline.pdf"
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    print("Saved:", out_png)
    print("Saved:", out_pdf)


if __name__ == "__main__":
    main()
