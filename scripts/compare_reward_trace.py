#!/usr/bin/env python3
"""
compare_reward_trace.py — Compare LTPO reward traces with / without workspace.

Summarises, per dataset:

  * Mean best_reward (the size of the best reward)
  * Mean best_reward_step (where the best reward appears in the ES loop)

for two experiment roots:

  --ws-root   (default: output/workspace_fixed_best_reward_trace)
              LTPO + visual workspace, best per-dataset configs.
  --no-ws-root (default: output/workspace_fixed_best_ltpo_only_reward_trace)
              LTPO only (workspace disabled), same per-dataset LTPO HPs.

and then reports the effect of turning the workspace on, in two flavours:

  (a) dataset-level delta
      mean_ws(dataset) - mean_no_ws(dataset)
      averaged across datasets.

  (b) per-sample delta
      paired by data_idx inside each dataset:
         per_sample_delta[i] = ws[i].best_reward - no_ws[i].best_reward
      the mean is taken across all matched samples (first within each
      dataset, then averaged across datasets → treats every sample as one
      observation).

Usage
-----
    python scripts/compare_reward_trace.py
    python scripts/compare_reward_trace.py --ws-root PATH --no-ws-root PATH
"""
import argparse
import os
import sys
from collections import defaultdict
from statistics import mean

import torch


DATASETS = [
    "mmvp_dev", "mmstar_dev", "mm_math_dev",
    "math_vista_dev", "math_vision_dev", "hallusion_dev", "scienceqa_dev",
]


# ── IO ────────────────────────────────────────────────────────────────────────

def find_logistics(root: str, dataset: str) -> str | None:
    """
    Find the logistics.pt under {root}/{dataset}/<single-run-dir>/logistics.pt.
    Returns None if the dataset dir does not exist or no logistics file is found.
    """
    ds_dir = os.path.join(root, dataset)
    if not os.path.isdir(ds_dir):
        return None
    for sub in sorted(os.listdir(ds_dir)):
        p = os.path.join(ds_dir, sub, "logistics.pt")
        if os.path.isfile(p):
            return p
    return None


def load_entries(path: str) -> list[dict]:
    data = torch.load(path, weights_only=False)
    return data.get("entries", [])


def summarise(entries: list[dict]) -> dict:
    rewards = [e["best_reward"] for e in entries if e.get("best_reward") is not None]
    steps = [e["best_reward_step"] for e in entries
             if e.get("best_reward_step") is not None]
    return {
        "n": len(entries),
        "n_reward": len(rewards),
        "n_step": len(steps),
        "mean_reward": mean(rewards) if rewards else float("nan"),
        "mean_step": mean(steps) if steps else float("nan"),
    }


def index_by_data_idx(entries: list[dict]) -> dict[int, dict]:
    return {e["data_idx"]: e for e in entries if "data_idx" in e}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ws-root", default="output/workspace_fixed_best_reward_trace",
                   help="Root dir for LTPO + workspace results.")
    p.add_argument("--no-ws-root",
                   default="output/workspace_fixed_best_ltpo_only_reward_trace",
                   help="Root dir for LTPO-only results.")
    p.add_argument("--datasets", nargs="+", default=DATASETS,
                   help="Datasets to include (default: all seven dev sets).")
    return p.parse_args()


def main():
    args = parse_args()

    for root in (args.ws_root, args.no_ws_root):
        if not os.path.isdir(root):
            sys.exit(f"ERROR: directory not found: {root}")

    rows = []                                    # per-dataset aggregates
    per_sample_reward_deltas: list[float] = []   # flat across all datasets
    per_sample_step_deltas:   list[float] = []
    per_dataset_sample_mean:  dict[str, dict] = {}

    for ds in args.datasets:
        ws_path   = find_logistics(args.ws_root,    ds)
        nows_path = find_logistics(args.no_ws_root, ds)
        if ws_path is None or nows_path is None:
            print(f"[WARN] skipping {ds}: "
                  f"{'ws missing ' if ws_path is None else ''}"
                  f"{'no-ws missing' if nows_path is None else ''}")
            continue

        ws_entries   = load_entries(ws_path)
        nows_entries = load_entries(nows_path)

        ws_sum   = summarise(ws_entries)
        nows_sum = summarise(nows_entries)

        # Paired, per-sample deltas (matched by data_idx)
        ws_map   = index_by_data_idx(ws_entries)
        nows_map = index_by_data_idx(nows_entries)
        common = sorted(set(ws_map) & set(nows_map))

        r_deltas, s_deltas = [], []
        for idx in common:
            a, b = ws_map[idx], nows_map[idx]
            if (a.get("best_reward") is not None
                    and b.get("best_reward") is not None):
                r_deltas.append(a["best_reward"] - b["best_reward"])
            if (a.get("best_reward_step") is not None
                    and b.get("best_reward_step") is not None):
                s_deltas.append(a["best_reward_step"] - b["best_reward_step"])

        per_sample_reward_deltas.extend(r_deltas)
        per_sample_step_deltas.extend(s_deltas)
        per_dataset_sample_mean[ds] = {
            "n_common": len(common),
            "mean_reward_delta": mean(r_deltas) if r_deltas else float("nan"),
            "mean_step_delta":   mean(s_deltas) if s_deltas else float("nan"),
        }

        rows.append({
            "dataset":      ds,
            "n_ws":         ws_sum["n"],
            "n_no_ws":      nows_sum["n"],
            "ws_reward":    ws_sum["mean_reward"],
            "nows_reward":  nows_sum["mean_reward"],
            "ws_step":      ws_sum["mean_step"],
            "nows_step":    nows_sum["mean_step"],
        })

    if not rows:
        sys.exit("No datasets produced comparable results.")

    W = 96
    print("=" * W)
    print("Reward-trace comparison  (workspace on vs LTPO-only)")
    print(f"  ws root    : {args.ws_root}")
    print(f"  no-ws root : {args.no_ws_root}")
    print("=" * W)

    # ── Per-dataset table ──
    print(f"\n{'dataset':<18} {'n(ws/no)':>10}  "
          f"{'reward_ws':>10} {'reward_no':>10} {'Δreward':>9}  "
          f"{'step_ws':>8} {'step_no':>8} {'Δstep':>7}")
    print("-" * W)
    for r in rows:
        n_str = f"{r['n_ws']}/{r['n_no_ws']}"
        d_r = r["ws_reward"] - r["nows_reward"]
        d_s = r["ws_step"]   - r["nows_step"]
        print(f"{r['dataset']:<18} {n_str:>10}  "
              f"{r['ws_reward']:>10.4f} {r['nows_reward']:>10.4f} {d_r:>+9.4f}  "
              f"{r['ws_step']:>8.3f} {r['nows_step']:>8.3f} {d_s:>+7.3f}")

    # ── Dataset-level average delta (average of per-dataset means) ──
    dataset_reward_deltas = [r["ws_reward"] - r["nows_reward"] for r in rows]
    dataset_step_deltas   = [r["ws_step"]   - r["nows_step"]   for r in rows]

    print("\n" + "=" * W)
    print("Aggregate: DATASET-LEVEL means "
          "(average of per-dataset (mean_ws - mean_no_ws))")
    print("=" * W)
    print(f"  mean Δbest_reward      : {mean(dataset_reward_deltas):+.4f}  "
          f"(over {len(dataset_reward_deltas)} datasets)")
    print(f"  mean Δbest_reward_step : {mean(dataset_step_deltas):+.4f}  "
          f"(over {len(dataset_step_deltas)} datasets)")

    # ── Per-sample delta (pair by data_idx inside each dataset, then flatten) ──
    print("\n" + "=" * W)
    print("Aggregate: PER-SAMPLE means "
          "(paired by data_idx inside each dataset, then averaged)")
    print("=" * W)
    print(f"\n{'dataset':<18} {'n_common':>9}  "
          f"{'mean_Δreward':>13} {'mean_Δstep':>12}")
    print("-" * 60)
    for ds, d in per_dataset_sample_mean.items():
        print(f"{ds:<18} {d['n_common']:>9}  "
              f"{d['mean_reward_delta']:>+13.4f} {d['mean_step_delta']:>+12.3f}")

    if per_sample_reward_deltas:
        print(f"\n  mean Δbest_reward  (flat over samples) : "
              f"{mean(per_sample_reward_deltas):+.4f}  "
              f"(over {len(per_sample_reward_deltas)} samples)")
    if per_sample_step_deltas:
        print(f"  mean Δbest_step    (flat over samples) : "
              f"{mean(per_sample_step_deltas):+.4f}  "
              f"(over {len(per_sample_step_deltas)} samples)")

    # Macro vs micro at a glance
    if rows and per_sample_reward_deltas:
        print("\nNote: the dataset-level mean treats each dataset as one obs;")
        print("      the per-sample mean treats each example as one obs.")


if __name__ == "__main__":
    main()
