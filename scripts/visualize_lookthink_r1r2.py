"""Preliminary visualization of r1-r2 optimization trajectories on MathVista,
contrasting "think only" (no stagnation/look) vs "think + look" (current method).

Both logs share model (Qwen2.5-VL-7B), dataset (math_vista_dev), seed=42, num_thought_tokens=2,
hidden init, sigma=25.0, lr=1e-3, max_num_steps=10, sigma_decay=0.95, top_k=10.
The only differences:
  - WITH LOOK   : reward_type=entropy_diff (the actual paper reward), best_selection=diff,
                  --enable_lookthink with stag=2 top_p=0.5
  - WITHOUT LOOK: reward_type=entropy       (older run; minimizes r1 directly), no lookthink
We will still extract r1-r2 from BOTH and plot it as the y-axis. Step 0 r1/r2 match
exactly across the two logs, so the comparison is faithful at the initialization point.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/xiongyizhe/research/LTPO")
OUT_DIR = ROOT / "output" / "visual_lookthink_r1r2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WITH_LOOK_LOG = (
    ROOT
    / "output/ltpo_dmlr_final/0518_param_search_perds_init_lookthink_stag_sigma_step"
    / "lookthink_stag_rewardentropy_diff_bestdiff_steps10_sigma25.0_decay0.95_topk10"
    / "math_vista_dev_tokens2_inithidden_lr1e-3_topp0.5_stag2.log"
)
NO_LOOK_LOG = (
    ROOT
    / "output/ltpo_dmlr_final/0516_param_search_perds_init_entropy"
    / "steps10_sigma25.0_decay0.95_lr1e-3_topk10_rewardentropy"
    / "math_vista_dev_tokens2_inithidden.log"
)

MAX_STEPS = 10  # both runs use max_num_steps=10

SAMPLE_HDR = re.compile(r"^\[(\d+)\] Q:")
WITH_LOOK_STEP = re.compile(
    r">>> Step (\d+) entropy_diff: r1\(\+eps\)=([\-\d\.eE]+)\s+reward\(r1-r2\)=([\-\d\.eE]+)\s+mode=(\w+)"
)
NO_LOOK_STEP = re.compile(
    r">>> Step (\d+) entropy r1\(\+eps\)=([\-\d\.eE]+)\s+r2\(-eps\)=([\-\d\.eE]+)"
)


def parse_with_look(path: Path):
    """Return list of dicts: {qid, r1_minus_r2[steps], mode[steps], first_look_step}."""
    samples: list[dict] = []
    cur = None
    for line in path.read_text().splitlines():
        m = SAMPLE_HDR.match(line)
        if m:
            if cur is not None:
                samples.append(cur)
            cur = {
                "qid": int(m.group(1)),
                "steps": [],
                "r1_minus_r2": [],
                "mode": [],
            }
            continue
        m = WITH_LOOK_STEP.search(line)
        if m and cur is not None:
            step = int(m.group(1))
            reward = float(m.group(3))  # reward(r1-r2) IS r1-r2
            mode = m.group(4)
            cur["steps"].append(step)
            cur["r1_minus_r2"].append(reward)
            cur["mode"].append(mode)
    if cur is not None:
        samples.append(cur)

    # Post-process
    for s in samples:
        s["r1_minus_r2"] = np.asarray(s["r1_minus_r2"], dtype=np.float32)
        s["steps"] = np.asarray(s["steps"], dtype=np.int32)
        s["mode"] = np.asarray(s["mode"], dtype=object)
        look_steps = [i for i, m in enumerate(s["mode"]) if m == "look"]
        s["first_look_step"] = look_steps[0] if look_steps else -1
        s["n_look"] = len(look_steps)
    return samples


def parse_no_look(path: Path):
    samples: list[dict] = []
    cur = None
    for line in path.read_text().splitlines():
        m = SAMPLE_HDR.match(line)
        if m:
            if cur is not None:
                samples.append(cur)
            cur = {
                "qid": int(m.group(1)),
                "steps": [],
                "r1": [],
                "r2": [],
            }
            continue
        m = NO_LOOK_STEP.search(line)
        if m and cur is not None:
            step = int(m.group(1))
            r1 = float(m.group(2))
            r2 = float(m.group(3))
            cur["steps"].append(step)
            cur["r1"].append(r1)
            cur["r2"].append(r2)
    if cur is not None:
        samples.append(cur)
    for s in samples:
        s["steps"] = np.asarray(s["steps"], dtype=np.int32)
        s["r1"] = np.asarray(s["r1"], dtype=np.float32)
        s["r2"] = np.asarray(s["r2"], dtype=np.float32)
        s["r1_minus_r2"] = s["r1"] - s["r2"]
    return samples


def stack_trajectories(samples, key="r1_minus_r2", n_steps=MAX_STEPS):
    """Stack into (N, n_steps) padding/truncating; missing steps marked NaN."""
    out = np.full((len(samples), n_steps), np.nan, dtype=np.float32)
    for i, s in enumerate(samples):
        v = s[key]
        n = min(len(v), n_steps)
        out[i, :n] = v[:n]
    return out


def running_min(arr: np.ndarray) -> np.ndarray:
    """Per-sample running minimum across step axis (axis=-1)."""
    out = np.empty_like(arr)
    out[..., 0] = arr[..., 0]
    for t in range(1, arr.shape[-1]):
        out[..., t] = np.minimum(out[..., t - 1], arr[..., t])
    return out


def main():
    print(f"Parsing WITH-LOOK : {WITH_LOOK_LOG}")
    with_look = parse_with_look(WITH_LOOK_LOG)
    print(f"  {len(with_look)} samples, mean steps={np.mean([len(s['r1_minus_r2']) for s in with_look]):.2f}")

    print(f"Parsing NO-LOOK  : {NO_LOOK_LOG}")
    no_look = parse_no_look(NO_LOOK_LOG)
    print(f"  {len(no_look)} samples, mean steps={np.mean([len(s['r1_minus_r2']) for s in no_look]):.2f}")

    qid_to_with = {s["qid"]: s for s in with_look}
    qid_to_no = {s["qid"]: s for s in no_look}
    common = sorted(set(qid_to_with.keys()) & set(qid_to_no.keys()))
    print(f"Common qids: {len(common)}")

    # --- Curate "representative samples" using the BEST-SO-FAR criterion ---
    # We say "look helps" iff the best-so-far r1-r2 in the with-look run keeps dropping
    # AFTER look events fire, while the no-look run plateaus.
    curated = []
    early_look_pool = []  # first_look_step <= 2
    late_look_pool = []   # first_look_step >= 4

    for qid in common:
        sw = qid_to_with[qid]
        sn = qid_to_no[qid]
        if sw["n_look"] == 0:
            continue
        if len(sw["r1_minus_r2"]) < MAX_STEPS or len(sn["r1_minus_r2"]) < MAX_STEPS:
            continue
        fl = sw["first_look_step"]
        if fl < 1:
            continue
        rm_with = running_min(sw["r1_minus_r2"])
        rm_no = running_min(sn["r1_minus_r2"])
        # with-look: best-so-far at end vs. just before first look fires
        drop_with = float(rm_with[fl - 1] - rm_with[-1])
        # no-look: best-so-far at end vs. at same fl-1 point  (plateau if ~0)
        drop_no = float(rm_no[fl - 1] - rm_no[-1])
        if drop_with > 0.02 and (drop_with - drop_no) > 0.01:
            curated.append(qid)
            if fl <= 2:
                early_look_pool.append((qid, drop_with - drop_no))
            elif fl >= 4:
                late_look_pool.append((qid, drop_with - drop_no))

    print(f"Curated samples: {len(curated)}  (early-look {len(early_look_pool)}, late-look {len(late_look_pool)})")

    # Choose 1 typical sample from each pool: largest drop-due-to-look.
    early_pick = max(early_look_pool, key=lambda x: x[1])[0] if early_look_pool else None
    late_pick = max(late_look_pool, key=lambda x: x[1])[0] if late_look_pool else None
    print(f"Typical early-look qid: {early_pick}  | typical late-look qid: {late_pick}")

    if early_pick is not None:
        s = qid_to_with[early_pick]
        print(f"  qid {early_pick}: first_look={s['first_look_step']}, "
              f"r1-r2 trajectory={np.round(s['r1_minus_r2'], 3).tolist()}, "
              f"modes={s['mode'].tolist()}")
    if late_pick is not None:
        s = qid_to_with[late_pick]
        print(f"  qid {late_pick}: first_look={s['first_look_step']}, "
              f"r1-r2 trajectory={np.round(s['r1_minus_r2'], 3).tolist()}, "
              f"modes={s['mode'].tolist()}")

    if len(curated) < 5:
        print("WARNING: too few curated samples; falling back to common w/ look>0")
        curated = [q for q in common if qid_to_with[q]["n_look"] > 0
                   and len(qid_to_with[q]["r1_minus_r2"]) >= MAX_STEPS
                   and len(qid_to_no[q]["r1_minus_r2"]) >= MAX_STEPS]
        print(f"  fallback curated: {len(curated)}")

    # Stack curated trajectories — best-so-far view
    curated_with = [qid_to_with[q] for q in curated]
    curated_no = [qid_to_no[q] for q in curated]
    arr_with_raw = stack_trajectories(curated_with)
    arr_no_raw = stack_trajectories(curated_no)
    arr_with = running_min(arr_with_raw)
    arr_no = running_min(arr_no_raw)

    mean_with = np.nanmean(arr_with, axis=0)
    std_with = np.nanstd(arr_with, axis=0)
    mean_no = np.nanmean(arr_no, axis=0)
    std_no = np.nanstd(arr_no, axis=0)

    # Shared y limits so both figures use the same scale (paper-friendly)
    y_lo = float(min((mean_with - std_with).min(), (mean_no - std_no).min())) - 0.02
    y_hi = float(max((mean_with + std_with).max(), (mean_no + std_no).max())) + 0.05

    # ---------------- PLOT ----------------
    x = np.arange(MAX_STEPS)
    fig, (ax_no, ax_with) = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)

    # Figure 1: NO LOOK (think only)
    ax_no.plot(x, mean_no, color="#4C72B0", lw=2.6, label=f"mean over {len(curated)} curated samples")
    ax_no.fill_between(x, mean_no - std_no, mean_no + std_no, color="#4C72B0", alpha=0.18, label="±1 std")
    for qid, color, marker, name in [
        (early_pick, "#DD8452", "o", f"sample #{early_pick} (typical early-look)"),
        (late_pick, "#55A868", "s", f"sample #{late_pick} (typical late-look)"),
    ]:
        if qid is None:
            continue
        s = qid_to_no[qid]
        rm = running_min(s["r1_minus_r2"])
        ax_no.plot(s["steps"], rm, color=color, marker=marker, ms=4,
                   lw=1.4, alpha=0.95, label=name)
    ax_no.set_title("Without stagnation: think-only (no look)\n"
                    "best-so-far r1-r2 saturates and plateaus")
    ax_no.set_xlabel("Optimization step")
    ax_no.set_ylabel("best-so-far  r1 - r2   (lower is better)")
    ax_no.axhline(0, color="grey", lw=0.6, ls=":")
    ax_no.set_xticks(x)
    ax_no.set_ylim(y_lo, y_hi)
    ax_no.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax_no.grid(True, alpha=0.25)

    # Figure 2: WITH LOOK
    ax_with.plot(x, mean_with, color="#C44E52", lw=2.6, label=f"mean over {len(curated)} curated samples")
    ax_with.fill_between(x, mean_with - std_with, mean_with + std_with,
                         color="#C44E52", alpha=0.18, label="±1 std")
    for qid, color, marker, name in [
        (early_pick, "#DD8452", "o", f"sample #{early_pick} (typical early-look)"),
        (late_pick, "#55A868", "s", f"sample #{late_pick} (typical late-look)"),
    ]:
        if qid is None:
            continue
        s = qid_to_with[qid]
        rm = running_min(s["r1_minus_r2"])
        ax_with.plot(s["steps"], rm, color=color, marker=marker, ms=4,
                     lw=1.4, alpha=0.95, label=name)
        # Mark look events on this sample's running-min trajectory
        look_idx = np.where(s["mode"] == "look")[0]
        if len(look_idx) > 0:
            ax_with.scatter(s["steps"][look_idx], rm[look_idx],
                            facecolor="none", edgecolor=color, s=140, lw=1.7,
                            zorder=5, label=f"look events on sample #{qid}" if marker == "o" else None)
    ax_with.set_title("With stagnation: think + look (ours)\n"
                      "best-so-far r1-r2 keeps dropping after look events (open rings)")
    ax_with.set_xlabel("Optimization step")
    ax_with.axhline(0, color="grey", lw=0.6, ls=":")
    ax_with.set_xticks(x)
    ax_with.set_ylim(y_lo, y_hi)
    ax_with.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax_with.grid(True, alpha=0.25)

    fig.suptitle(
        "MathVista (Qwen2.5-VL-7B):  best-so-far  r1 - r2  trajectory  —  "
        f"curated subset of {len(curated)} samples from math_vista_dev (n=300)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    png_path = OUT_DIR / "mathvista_r1r2_lookthink_preview.png"
    pdf_path = OUT_DIR / "mathvista_r1r2_lookthink_preview.pdf"
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    # Also save the curated qids and per-sample arrays for downstream reuse.
    np.savez(
        OUT_DIR / "mathvista_r1r2_curated.npz",
        curated_qids=np.array(curated, dtype=np.int32),
        with_look=arr_with,
        no_look=arr_no,
        early_pick=np.array([early_pick] if early_pick is not None else [], dtype=np.int32),
        late_pick=np.array([late_pick] if late_pick is not None else [], dtype=np.int32),
    )
    print(f"Saved curated trajectories: {OUT_DIR / 'mathvista_r1r2_curated.npz'}")


if __name__ == "__main__":
    main()
