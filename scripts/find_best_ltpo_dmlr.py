#!/usr/bin/env python3
"""
find_best_ltpo_dmlr.py — Summarise grid-search results.

Supports two experiment types via --mode:

  dmlr      (default) Results from run_ltpo_vl_dmlr_grid_dev.sh.
            Directory pattern: {model}-{dataset}-tokens{t}-lr{lr}-sigma{s}
                               -sigdecay{sd}-steps{n}-topk{k}[-conf]-dmlr
            Default root: output/ltpo_dmlr_grid_dev

  workspace Results from run_workspace_fixed_dev_grid.sh.
            Directory pattern: {model}-{dataset}-tokens{t}-lr{lr}-sigma{s}
                               -sigdecay{sd}-steps{n}-topk{k}[-conf]
                               -ws{K}p{P}r{r}-{inject_mode}-fixed-workspace
            Default root: output/workspace_fixed_dev_grid

Usage (from repo root):
    python scripts/find_best_ltpo_dmlr.py [--root DIR] [--top N] [--mode dmlr|workspace] [--model MODEL]
"""
import argparse
import os
import re
import sys
from collections import defaultdict

# ── Patterns ──────────────────────────────────────────────────────────────────

DATASETS = [
    "math_vista_dev", "math_vision_dev", "mm_math_dev", "hallusion_dev", 
    "mmvp_dev", "mmstar_dev", "scienceqa_dev",
]

_DS_ALT = "|".join(re.escape(d) for d in DATASETS)

# DMLR directory pattern (non-baseline):
#   {model}-{dataset}-tokens{t}-lr{lr}-sigma{s}-sigdecay{sd}-steps{n}-topk{k}[-conf]-dmlr
DMLR_DIR_RE = re.compile(
    rf"(?P<model>.+?)-(?P<dataset>{_DS_ALT})"
    r"-tokens(?P<tokens>[^-]+)"
    r"-lr(?P<lr>[^-]+)"
    r"-sigma(?P<sigma>[^-]+)"
    r"-sigdecay(?P<decay>[^-]+)"
    r"-steps(?P<steps>[^-]+)"
    r"-topk(?P<topk>[^-]+)"
    r"(?:-conf)?-dmlr$"
)

# Workspace directory pattern:
#   {model}-{dataset}-tokens{t}-lr{lr}-sigma{s}-sigdecay{sd}-steps{n}-topk{k}
#   [-conf]-ws{K}p{P}r{r}-{inject_mode}-fixed-workspace
WORKSPACE_DIR_RE = re.compile(
    rf"(?P<model>.+?)-(?P<dataset>{_DS_ALT})"
    r"-tokens(?P<tokens>[^-]+)"
    r"-lr(?P<lr>[^-]+)"
    r"-sigma(?P<sigma>[^-]+)"
    r"-sigdecay(?P<decay>[^-]+)"
    r"-steps(?P<steps>[^-]+)"
    r"-topk(?P<topk>[^-]+)"
    r"(?:-conf)?"
    r"-ws(?P<K>\d+)p(?P<P>\d+)r(?P<r>\d+)"
    r"-(?P<inject_mode>[^-]+)"
    r"-per_token"
    r"-fixed-warmup7-workspace$"
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["dmlr", "workspace"], default="dmlr",
                   help="Experiment type to summarise (default: dmlr)")
    p.add_argument("--root", default=None,
                   help="Root output directory (default depends on --mode)")
    p.add_argument("--top", type=int, default=3,
                   help="Number of top configs to show per dataset (default: %(default)s)")
    p.add_argument("--model", default=None,
                   help="Filter results to dirs whose model name contains this substring "
                        "(case-insensitive). E.g. --model Qwen2.5-VL-3B")
    return p.parse_args()


# ── DMLR mode ─────────────────────────────────────────────────────────────────

def run_dmlr(root: str, top_n: int, model_filter: str | None = None):
    results = defaultdict(list)
    n_found = n_unmatched = 0

    for dirpath, _dirs, files in os.walk(root):
        if "results.log" not in files:
            continue
        dir_name = os.path.basename(dirpath)
        m = DMLR_DIR_RE.match(dir_name)
        if not m:
            n_unmatched += 1
            continue

        if model_filter and model_filter.lower() not in m.group("model").lower():
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
    print(f"LTPO-DMLR Grid Search  —  {root}"
          + (f"  [model filter: {model_filter}]" if model_filter else ""))
    print(f"  Completed runs : {n_found}"
          + (f"   |   Unrecognised dirs : {n_unmatched}" if n_unmatched else ""))
    print("=" * W)

    best_per_dataset = {}

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

        n = min(top_n, len(ranked))
        if n > 1:
            print(f"\n  Top-{n}:")
            print(f"  {'Rank':<5} {'Acc':>7}  {'C/T':<11}  tokens  steps  sigma    lr       decay")
            print(f"  {'----':<5} {'---':>7}  {'---':<11}  {'------':<7} {'-----':<6} {'--------':<8} {'-------':<8} {'-----'}")
            for rank, r in enumerate(ranked[:n], 1):
                h = r["hparams"]
                ct = f"{r['correct']}/{r['total']}"
                print(f"  {rank:<5} {r['acc']:>7.4f}  {ct:<11}  "
                      f"{h['tokens']:<7} {h['steps']:<6} {h['sigma']:<8} {h['lr']:<8} {h['decay']}")

    # Cross-dataset summary
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

    # Config ranking by mean accuracy across all datasets
    config_accs: dict[str, list] = defaultdict(list)
    for dataset, entries in results.items():
        for r in entries:
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


# ── Workspace mode ─────────────────────────────────────────────────────────────

def run_workspace(root: str, top_n: int, model_filter: str | None = None):
    results = defaultdict(list)
    n_found = n_unmatched = 0

    for dirpath, _dirs, files in os.walk(root):
        if "results.log" not in files:
            continue
        dir_name = os.path.basename(dirpath)
        m = WORKSPACE_DIR_RE.match(dir_name)
        if not m:
            n_unmatched += 1
            continue

        if model_filter and model_filter.lower() not in m.group("model").lower():
            continue

        parsed = parse_results_log(os.path.join(dirpath, "results.log"))
        if parsed is None:
            continue

        correct, total, acc = parsed
        dataset = m.group("dataset")
        hparams = {k: m.group(k) for k in
                   ("model", "tokens", "lr", "sigma", "decay", "steps", "topk",
                    "K", "P", "r", "inject_mode")}
        results[dataset].append(dict(acc=acc, correct=correct, total=total,
                                     dir_name=dir_name, hparams=hparams))
        n_found += 1

    if not results:
        sys.exit(
            f"No completed results found under '{root}'.\n"
            "Make sure the jobs have finished and results.log files exist."
        )

    W = 80
    print("=" * W)
    print(f"Workspace-Fixed Grid Search  —  {root}"
          + (f"  [model filter: {model_filter}]" if model_filter else ""))
    print(f"  Completed runs : {n_found}"
          + (f"   |   Unrecognised dirs : {n_unmatched}" if n_unmatched else ""))
    print("=" * W)

    best_per_dataset = {}

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
        print(f"  WS    K={hp['K']}  P={hp['P']}  r={hp['r']}  mode={hp['inject_mode']}")
        print(f"  LTPO  tokens={hp['tokens']}  steps={hp['steps']}  "
              f"sigma={hp['sigma']}  decay={hp['decay']}  lr={hp['lr']}")

        n = min(top_n, len(ranked))
        if n > 1:
            print(f"\n  Top-{n}:")
            print(f"  {'Rank':<5} {'Acc':>7}  {'C/T':<11}  K    r    mode       tokens  steps")
            print(f"  {'----':<5} {'---':>7}  {'---':<11}  {'--':<5} {'--':<5} {'----------':<11} {'------':<7} {'-----'}")
            for rank, entry in enumerate(ranked[:n], 1):
                h = entry["hparams"]
                ct = f"{entry['correct']}/{entry['total']}"
                print(f"  {rank:<5} {entry['acc']:>7.4f}  {ct:<11}  "
                      f"{h['K']:<5} {h['r']:<5} {h['inject_mode']:<11} "
                      f"{h['tokens']:<7} {h['steps']}")

    # Cross-dataset summary
    print(f"\n{'=' * W}")
    print("Cross-dataset Summary  (best accuracy per dataset)")
    print(f"{'=' * W}")
    print(f"  {'Dataset':<22} {'Acc':>7}  K    r    mode       tokens  steps  sigma")
    print(f"  {'-------':<22} {'---':>7}  {'--':<5} {'--':<5} {'----------':<11} {'------':<7} {'-----':<6} {'-----'}")

    accs = []
    for dataset in DATASETS:
        if dataset not in best_per_dataset:
            continue
        b = best_per_dataset[dataset]
        h = b["hparams"]
        accs.append(b["acc"])
        print(f"  {dataset:<22} {b['acc']:>7.4f}  "
              f"{h['K']:<5} {h['r']:<5} {h['inject_mode']:<11} "
              f"{h['tokens']:<7} {h['steps']:<6} {h['sigma']}")

    if accs:
        print(f"\n  Mean best-accuracy over {len(accs)} datasets : {sum(accs)/len(accs):.4f}")

    # Config ranking by mean accuracy — key on workspace HPs only (K, P, r, inject_mode)
    # since LTPO HPs are fixed per dataset in this experiment.
    config_accs: dict[str, list] = defaultdict(list)
    for dataset, entries in results.items():
        for entry in entries:
            hp = entry["hparams"]
            key = (f"K={hp['K']} P={hp['P']} r={hp['r']} mode={hp['inject_mode']}")
            config_accs[key].append(entry["acc"])

    n_ds = len(results)
    complete = {k: v for k, v in config_accs.items() if len(v) == n_ds}
    partial  = {k: v for k, v in config_accs.items() if len(v) <  n_ds}

    if complete:
        ranked_cfgs = sorted(complete.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)
        print(f"\n{'=' * W}")
        print(f"Workspace Config Ranking by Mean Accuracy  (all {n_ds} datasets complete)")
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

    print(" ".join(f"{acc*100:.2f}" for acc in accs))


# ── Main ──────────────────────────────────────────────────────────────────────

DEFAULT_ROOTS = {
    "dmlr":      "output/ltpo_dmlr_grid_dev",
    "workspace": "output/workspace_fixed_dev_grid",
}

def main():
    args = parse_args()
    root = args.root or DEFAULT_ROOTS[args.mode]

    if not os.path.isdir(root):
        sys.exit(f"ERROR: directory not found: {root}")

    if args.mode == "dmlr":
        run_dmlr(root, args.top, args.model)
    else:
        run_workspace(root, args.top, args.model)


if __name__ == "__main__":
    main()

# Examples:
# python scripts/find_best_ltpo_dmlr.py --root output/ltpo_dmlr_grid_dev --top 5
# python scripts/find_best_ltpo_dmlr.py --root output/ltpo_dmlr_grid_dev --model Qwen2.5-VL-3B
# python scripts/find_best_ltpo_dmlr.py --mode workspace --top 5
# python scripts/find_best_ltpo_dmlr.py --mode workspace --root output/workspace_fixed_dev_grid --top 3
# python scripts/find_best_ltpo_dmlr.py --mode workspace --model Qwen2.5-VL-7B
