#!/usr/bin/env python3
"""
find_best_grid.py — Summarise grid-search results with dynamic hparam parsing.

Expects a three-level layout:
    <root>/<search>/<experiment>/results.log

Hyperparameters are extracted automatically from the <search> directory name,
which must be a sequence of "<key><value>" tokens joined by "_", e.g.
    tokens2_steps15_sigma25.0_decay0.95_lr5e-3_topk10_contrastive
Tokens with no value (e.g. "contrastive") are kept as flags.

The dataset is detected by matching a known name inside each <experiment> dir.

Usage (from repo root):
    python scripts/find_best_grid.py <root> [--top N] [--model MODEL]

Example:
    python scripts/find_best_grid.py output/ltpo_dmlr_contrastive_grid_dev/0423_first_search
"""
import argparse
import os
import re
import sys
from collections import defaultdict

DATASETS = [
    "math_vista_dev", "math_vision_dev", "mm_math_dev", "hallusion_dev",
    "mmvp_dev", "mmstar_dev", "scienceqa_dev",
]
DATASET_HEADERS = [
    "MathVista", "MathVision", "MM-Math", "HallusionBench",
    "MMVP", "MMStar", "ScienceQA",
]

ACC_RE = re.compile(r"correct=(\d+),\s*total=(\d+),\s*accuracy=([0-9.]+)")
TOK_RE = re.compile(r"^([A-Za-z]+)(.*)$")


def parse_results_log(path):
    try:
        with open(path) as f:
            matches = ACC_RE.findall(f.read())
    except OSError:
        return None
    if not matches:
        return None
    c, t, a = matches[-1]
    return int(c), int(t), float(a)


def parse_search_name(name):
    """Parse '<key><val>_<key><val>_<flag>_…' into an ordered dict.

    Tokens with an empty value part are stored as flags (value=True).
    """
    hp = {}
    for tok in name.split("_"):
        m = TOK_RE.match(tok)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        hp[key] = val if val else True
    return hp


def detect_dataset(exp_name):
    for d in DATASETS:
        if d in exp_name:
            return d
    return None


def fmt_hp(hp):
    return "  ".join(k if v is True else f"{k}={v}" for k, v in hp.items())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="Root dir containing <search>/<experiment> subdirs")
    p.add_argument("--top", type=int, default=3,
                   help="Top-N configs per dataset (default: %(default)s)")
    p.add_argument("--model", default=None,
                   help="Filter by substring in experiment dir name (case-insensitive)")
    args = p.parse_args()

    root = args.root.rstrip("/")
    if not os.path.isdir(root):
        sys.exit(f"ERROR: directory not found: {root}")

    results = defaultdict(list)           # dataset -> [entry, …]
    configs = defaultdict(dict)           # search  -> {dataset: acc}
    search_hp = {}                        # search  -> hparams
    n_found = 0

    for search in sorted(os.listdir(root)):
        search_path = os.path.join(root, search)
        if not os.path.isdir(search_path):
            continue
        hp = parse_search_name(search)
        search_hp[search] = hp
        for exp in sorted(os.listdir(search_path)):
            log_path = os.path.join(search_path, exp, "results.log")
            if not os.path.isfile(log_path):
                continue
            if args.model and args.model.lower() not in exp.lower():
                continue
            dataset = detect_dataset(exp)
            if dataset is None:
                continue
            parsed = parse_results_log(log_path)
            if parsed is None:
                continue
            correct, total, acc = parsed
            results[dataset].append(dict(acc=acc, correct=correct, total=total,
                                         dir_name=exp, search=search, hparams=hp))
            configs[search][dataset] = acc
            n_found += 1

    if not results:
        sys.exit(f"No completed results found under '{root}'.")

    W = 80
    print("=" * W)
    print(f"Grid Search Summary  —  {root}"
          + (f"  [model filter: {args.model}]" if args.model else ""))
    print(f"  Completed runs : {n_found}")
    print("=" * W)

    best_per_dataset = {}
    for dataset in DATASETS:
        if dataset not in results:
            continue
        ranked = sorted(results[dataset], key=lambda x: x["acc"], reverse=True)
        best = ranked[0]
        best_per_dataset[dataset] = best
        print(f"\n── {dataset} {'─' * max(1, W - 4 - len(dataset))}")
        print(f"  Best    {best['acc']:.4f}  ({best['correct']}/{best['total']})")
        print(f"  Search  {best['search']}")
        print(f"  HP      {fmt_hp(best['hparams'])}")

        n = min(args.top, len(ranked))
        if n > 1:
            print(f"\n  Top-{n}:")
            for rank, r in enumerate(ranked[:n], 1):
                ct = f"{r['correct']}/{r['total']}"
                print(f"    {rank}. acc={r['acc']:.4f}  ({ct:<11}) {r['search']}")

    # Cross-dataset summary
    print(f"\n{'=' * W}")
    print("Cross-dataset Summary  (best accuracy per dataset)")
    print(f"{'=' * W}")
    accs = []
    for dataset in DATASETS:
        if dataset not in best_per_dataset:
            continue
        b = best_per_dataset[dataset]
        accs.append(b["acc"])
        print(f"  {dataset:<22} {b['acc']:>7.4f}   {b['search']}")
    if accs:
        print(f"\n  Mean best-acc over {len(accs)} datasets : {sum(accs)/len(accs):.4f}")

    # Config ranking by mean accuracy
    n_ds = len(results)
    complete, partial = {}, {}
    for search, ds_acc in configs.items():
        (complete if len(ds_acc) == n_ds else partial)[search] = ds_acc

    def mean(d):
        return sum(d.values()) / len(d)

    if complete:
        ranked = sorted(complete.items(), key=lambda x: mean(x[1]), reverse=True)
        print(f"\n{'=' * W}")
        print(f"Config Ranking by Mean Accuracy  (all {n_ds} datasets complete)")
        print(f"{'=' * W}")
        print(f"  {'Rank':<5} {'Mean Acc':>9}  Search")
        for rank, (search, ds_acc) in enumerate(ranked, 1):
            print(f"  {rank:<5} {mean(ds_acc):>9.4f}  {search}")

    if partial:
        ranked_partial = sorted(partial.items(), key=lambda x: mean(x[1]), reverse=True)
        print(f"\n  Partial configs ({len(partial)} with < {n_ds} datasets):")
        for search, ds_acc in ranked_partial:
            print(f"    mean={mean(ds_acc):.4f}  n={len(ds_acc)}  {search}")

    # Per-config results table — one row per config, columns in fixed order.
    # Accuracies printed as percentages (×100) for easy copy-paste.
    print(f"\n{'=' * W}")
    print("Per-config Results Table  (tab-separated, one row per config)")
    print(f"{'=' * W}")
    print("Config\t" + "\t".join(DATASET_HEADERS))
    for search, ds_acc in sorted(configs.items(), key=lambda x: mean(x[1]), reverse=True):
        cells = [f"{ds_acc[d]*100:.2f}" if d in ds_acc else "-" for d in DATASETS]
        print(search + "\t" + "\t".join(cells))

    # Final row: per-dataset best across all searched configs.
    best_cells = [f"{max(r['acc'] for r in results[d])*100:.2f}" if d in results else "-"
                  for d in DATASETS]
    print("Best (per-dataset max)\t" + "\t".join(best_cells))

    print()


if __name__ == "__main__":
    main()
