#!/usr/bin/env python3
"""
find_best_ltpo_dmlr.py — Summarise grid-search results from run_ltpo_vl_dmlr_grid_dev.sh.

Recursively finds every results.log under the root output dir.  Each file lives
in a directory whose name encodes the dataset and all hyperparameters
(written by main_vl_dmlr.py), so no stdout log parsing is needed.

Usage (from repo root):
    python scripts/find_best_ltpo_dmlr.py [--root output/ltpo_dmlr_grid_dev] [--top N]
"""
import argparse
import os
import re
import sys
from collections import defaultdict

# ── Patterns ──────────────────────────────────────────────────────────────────

DATASETS = [
    "mmvp_dev", "mmstar_dev", "mm_math_dev",
    "math_vista_dev", "math_vision_dev", "hallusion_dev", "scienceqa_dev",
]

# Directory name written by main_vl_dmlr.py (non-baseline path):
#   {model_name}-{data_name}-tokens{t}-lr{lr}-sigma{s}-sigdecay{sd}-steps{n}-topk{k}[-conf]-dmlr
_DS_ALT = "|".join(re.escape(d) for d in DATASETS)
DIR_RE = re.compile(
    rf"(?P<model>.+?)-(?P<dataset>{_DS_ALT})"
    r"-tokens(?P<tokens>[^-]+)"
    r"-lr(?P<lr>[^-]+)"
    r"-sigma(?P<sigma>[^-]+)"
    r"-sigdecay(?P<decay>[^-]+)"
    r"-steps(?P<steps>[^-]+)"
    r"-topk(?P<topk>[^-]+)"
    r"(?:-conf)?-dmlr$"
)

# Last accuracy line in results.log:  correct=42, total=300, accuracy=0.1400
ACC_RE = re.compile(r"correct=(\d+),\s*total=(\d+),\s*accuracy=([0-9.]+)")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_results_log(path: str):
    """Return (correct, total, accuracy) from the last matching line, or None."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return None
    matches = ACC_RE.findall(text)
    if not matches:
        return None
    correct, total, acc = matches[-1]
    return int(correct), int(total), float(acc)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="output/ltpo_dmlr_grid_dev",
                   help="Root output directory of the grid search (default: %(default)s)")
    p.add_argument("--top", type=int, default=3,
                   help="Number of top configs to show per dataset (default: %(default)s)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    root = args.root

    if not os.path.isdir(root):
        sys.exit(f"ERROR: directory not found: {root}")

    # results[dataset] = list of {acc, correct, total, dir_name, hparams}
    results = defaultdict(list)
    n_found = n_unmatched = 0

    for dirpath, _dirs, files in os.walk(root):
        if "results.log" not in files:
            continue
        dir_name = os.path.basename(dirpath)
        m = DIR_RE.match(dir_name)
        if not m:
            n_unmatched += 1
            continue

        parsed = parse_results_log(os.path.join(dirpath, "results.log"))
        if parsed is None:
            continue

        correct, total, acc = parsed
        dataset = m.group("dataset")
        hparams = {k: m.group(k) for k in ("model", "tokens", "lr", "sigma", "decay", "steps", "topk")}
        results[dataset].append(dict(acc=acc, correct=correct, total=total,
                                     dir_name=dir_name, hparams=hparams))
        n_found += 1

    if not results:
        sys.exit(
            f"No completed results found under '{root}'.\n"
            "Make sure the jobs have finished and results.log files exist."
        )

    W = 74
    print("=" * W)
    print(f"LTPO-DMLR Grid Search  —  {root}")
    print(f"  Completed runs : {n_found}"
          + (f"   |   Unrecognised dirs : {n_unmatched}" if n_unmatched else ""))
    print("=" * W)

    best_per_dataset = {}   # for cross-dataset summary

    for dataset in DATASETS:
        if dataset not in results:
            continue
        ranked = sorted(results[dataset], key=lambda x: x["acc"], reverse=True)
        best = ranked[0]
        best_per_dataset[dataset] = best
        hp = best["hparams"]

        print(f"\n── {dataset} {'─' * max(1, W - 4 - len(dataset))}")
        print(f"  Best  {best['acc']:.4f}  ({best['correct']}/{best['total']})")
        print(f"  Dir   {best['dir_name']}")
        print(f"  HP    tokens={hp['tokens']}  steps={hp['steps']}  "
              f"sigma={hp['sigma']}  decay={hp['decay']}  "
              f"lr={hp['lr']}  topk={hp['topk']}")

        top_n = min(args.top, len(ranked))
        if top_n > 1:
            print(f"\n  Top-{top_n}:")
            print(f"  {'Rank':<5} {'Acc':>7}  {'C/T':<11}  tokens  steps  sigma    lr       decay")
            print(f"  {'----':<5} {'---':>7}  {'---':<11}  {'------':<7} {'-----':<6} {'--------':<8} {'-------':<8} {'-----'}")
            for rank, r in enumerate(ranked[:top_n], 1):
                h = r["hparams"]
                ct = f"{r['correct']}/{r['total']}"
                print(f"  {rank:<5} {r['acc']:>7.4f}  {ct:<11}  "
                      f"{h['tokens']:<7} {h['steps']:<6} {h['sigma']:<8} {h['lr']:<8} {h['decay']}")

    # ── Cross-dataset summary ──────────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print("Cross-dataset Summary  (best accuracy per dataset)")
    print(f"{'=' * W}")
    print(f"  {'Dataset':<22} {'Acc':>7}  tokens  steps  sigma    lr       decay")
    print(f"  {'-------':<22} {'---':>7}  {'------':<7} {'-----':<6} {'--------':<8} {'-------':<8} {'-----'}")

    accs = []
    for dataset in DATASETS:
        if dataset not in best_per_dataset:
            continue
        b = best_per_dataset[dataset]
        h = b["hparams"]
        accs.append(b["acc"])
        print(f"  {dataset:<22} {b['acc']:>7.4f}  "
              f"{h['tokens']:<7} {h['steps']:<6} {h['sigma']:<8} {h['lr']:<8} {h['decay']}")

    if accs:
        print(f"\n  Mean best-accuracy over {len(accs)} datasets : {sum(accs)/len(accs):.4f}")

    # ── Config ranking by mean accuracy (across ALL datasets) ─────────────────
    config_accs: dict[str, list] = defaultdict(list)
    for dataset, entries in results.items():
        for r in entries:
            # Key by (tokens, steps, sigma, decay, lr, topk) — ignore model/dataset
            hp = r["hparams"]
            key = (f"tokens={hp['tokens']} steps={hp['steps']} sigma={hp['sigma']} "
                   f"decay={hp['decay']} lr={hp['lr']} topk={hp['topk']}")
            config_accs[key].append(r["acc"])

    n_ds = len(results)
    complete = {k: v for k, v in config_accs.items() if len(v) == n_ds}
    partial  = {k: v for k, v in config_accs.items() if len(v) <  n_ds}

    if complete:
        ranked_cfgs = sorted(complete.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)
        print(f"\n{'=' * W}")
        print(f"Config Ranking by Mean Accuracy  (all {n_ds} datasets complete)")
        print(f"{'=' * W}")
        print(f"  {'Rank':<5} {'Mean Acc':>9}  Config")
        print(f"  {'----':<5} {'--------':>9}  {'------'}")
        for rank, (cfg, vs) in enumerate(ranked_cfgs, 1):
            mean = sum(vs) / len(vs)
            print(f"  {rank:<5} {mean:>9.4f}  {cfg}")

    if partial:
        ranked_partial = sorted(partial.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)
        print(f"\n  Partial configs ({len(partial)} with < {n_ds} datasets):")
        for cfg, vs in ranked_partial:
            print(f"    mean={sum(vs)/len(vs):.4f}  n={len(vs)}  {cfg}")

    print()


if __name__ == "__main__":
    main()
# python scripts/find_best_ltpo_dmlr.py --root output/ltpo_dmlr_grid_dev --top 5