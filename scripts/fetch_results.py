#!/usr/bin/env python3
"""
fetch_results.py — Collect benchmark results from a directory tree.

For each distinct experiment (grouped by model/method across all datasets),
prints one space-separated line:

  <METHOD_NAME> <MATHVISTA> <MATHVISION> <MM-MATH> <HALLUSION> <MMVP> <MMSTAR> <SCIENCEQA>

Accuracy values are formatted as percentages (e.g. 0.2100 → 21.00).
Missing datasets for a method are shown as "-".

Usage:
    python scripts/fetch_results.py <directory>
    python scripts/fetch_results.py output/dmlr_aligned_dev
    python scripts/fetch_results.py output/ltpo_dmlr_grid_dev
"""

import argparse
import os
import re
import sys

# ── Dataset configuration ──────────────────────────────────────────────────────

# Order must match the required output columns
DATASET_COLS = [
    "math_vista_dev",   # MATHVISTA
    "math_vision_dev",  # MATHVISION
    "mm_math_dev",      # MM-MATH
    "hallusion_dev",    # HALLUSION
    "mmvp_dev",         # MMVP
    "mmstar_dev",       # MMSTAR
    "scienceqa_dev",    # SCIENCEQA
]

HEADER = "METHOD MATHVISTA MATHVISION MM-MATH HALLUSION MMVP MMSTAR SCIENCEQA"

# Sorted longest-first so that partial matches don't shadow longer dataset names
_DATASETS_BY_LEN = sorted(DATASET_COLS, key=len, reverse=True)

# Accuracy line in results.log:  correct=42, total=300, accuracy=0.1400
ACC_RE = re.compile(r"correct=(\d+),\s*total=(\d+),\s*accuracy=([0-9.]+)")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_results_log(path: str):
    """Return accuracy float from the last matching line, or None."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return None
    matches = ACC_RE.findall(text)
    if not matches:
        return None
    _correct, _total, acc = matches[-1]
    return float(acc)


def find_dataset_in_name(name: str):
    """Return the first dataset name found in `name`, or None."""
    for ds in _DATASETS_BY_LEN:
        if ds in name:
            return ds
    return None


def strip_dataset_from_name(name: str, dataset: str) -> str:
    """Remove `dataset` from `name` and clean up stray separators."""
    result = name.replace(dataset, "")
    # Collapse multiple consecutive separators (- or _) into one
    result = re.sub(r'[-_]{2,}', lambda m: m.group(0)[0], result)
    # Strip leading/trailing separators
    result = result.strip("-_")
    return result


def method_key_from_path(rel_path: str, dataset: str) -> str:
    """
    Derive a method key from the relative path to the results.log directory.

    The dataset name is stripped from every path component so that the key is
    identical for all datasets belonging to the same experiment.  Components
    that become empty after stripping (e.g. a directory that *is* the dataset
    name) are dropped entirely.
    """
    parts = rel_path.replace("\\", "/").split("/")
    cleaned = []
    for part in parts:
        c = strip_dataset_from_name(part, dataset)
        if c:
            cleaned.append(c)
    return "/".join(cleaned) if cleaned else "default"


# ── Main logic ────────────────────────────────────────────────────────────────

def collect(root: str, group_by_exp_type: bool = False) -> dict:
    """
    Walk `root` and return:
        { method_key: { dataset: accuracy_float } }

    When `group_by_exp_type` is True, assumes the tree is laid out as
    `<root>/<exp_type>/<per_dataset_experiment>/results.log` (where per-dataset
    experiments within the same `<exp_type>` may use different hyperparameters)
    and groups rows by the top-level `<exp_type>` directory.
    """
    data: dict[str, dict[str, float]] = {}

    for dirpath, _dirs, files in os.walk(root):
        if "results.log" not in files:
            continue

        rel = os.path.relpath(dirpath, root)
        # rel is "." if results.log is directly in root — skip that edge case
        if rel == ".":
            continue

        # Find which dataset this directory corresponds to
        dataset = find_dataset_in_name(rel)
        if dataset is None:
            # Skip directories we can't identify
            continue

        acc = parse_results_log(os.path.join(dirpath, "results.log"))
        if acc is None:
            continue

        if group_by_exp_type:
            parts = rel.replace("\\", "/").split("/")
            if len(parts) < 2:
                # results.log sits directly under an <exp_type> dir — no
                # per-dataset subdirectory, can't apply the grouping cleanly
                continue
            key = parts[0]
        else:
            key = method_key_from_path(rel, dataset)
        data.setdefault(key, {})[dataset] = acc

    return data


def format_acc(val) -> str:
    if val is None:
        return "-"
    return f"{val * 100:.2f}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("directory", help="Root directory to search for results")
    parser.add_argument(
        "--header", action="store_true",
        help="Print column header as first line",
    )
    parser.add_argument(
        "--sort", choices=["name", "mean"], default="name",
        help="Sort rows by method name (default) or mean accuracy",
    )
    parser.add_argument(
        "--group-by-exp-type", action="store_true",
        help=(
            "Group results by top-level <exp_type> subdirectory. "
            "Use when the input is laid out as "
            "<root>/<exp_type>/<per_dataset_experiment>/results.log and each "
            "per-dataset experiment within an <exp_type> may use different "
            "hyperparameters (e.g. output/workspace_mask_reward_dev_grid)."
        ),
    )
    args = parser.parse_args()

    root = args.directory
    if not os.path.isdir(root):
        sys.exit(f"ERROR: directory not found: {root}")

    data = collect(root, group_by_exp_type=args.group_by_exp_type)
    if not data:
        sys.exit(
            f"No completed results found under '{root}'.\n"
            "Make sure results.log files exist in the expected subdirectories."
        )

    if args.sort == "mean":
        def sort_key(item):
            accs = [v for v in item[1].values() if v is not None]
            return -(sum(accs) / len(accs)) if accs else 0.0
    else:
        sort_key = lambda item: item[0].lower()

    rows = sorted(data.items(), key=sort_key)

    if args.header:
        print(HEADER)

    for method_key, ds_accs in rows:
        cols = [format_acc(ds_accs.get(ds)) for ds in DATASET_COLS]
        print(method_key, *cols)


if __name__ == "__main__":
    main()
