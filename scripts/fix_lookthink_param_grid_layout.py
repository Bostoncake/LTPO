#!/usr/bin/env python3
"""
fix_lookthink_param_grid_layout.py — Flatten 0505-style param-grid output dirs
into the 3-level layout expected by scripts/find_best_grid.py.

Source layout (one extra level — produced by run_ltpo_vl_dmlr_lookthink_param_grid_dev.sh):

    <src>/<lt_la>/<dataset>_<hp_subset>/<Qwen_exp_dir>/results.log
                               ^^^^^^^^^^ extra middle level

Target layout (matches 0502_lookthink_optclip_bestr1, accepted by find_best_grid.py):

    <dst>/<lt_la>/<Qwen_exp_dir>/results.log

Per-dataset training logs (e.g. ``hallusion_dev.log``) are also linked to the top
of each ``<lt_la>`` dir so the layout is fully analogous to the reference.

By default a sibling directory ``<src>_fixed`` is created using symlinks, so the
source tree is left untouched. Pass ``--inplace`` to mutate the source dir
(moves the inner exp dirs up one level and removes the empty middle dirs).

Usage:
    python scripts/fix_lookthink_param_grid_layout.py \\
        output/ltpo_dmlr_contrastive_grid_dev/0505_lookthink_param_grid

Then:
    python scripts/find_best_grid.py \\
        output/ltpo_dmlr_contrastive_grid_dev/0505_lookthink_param_grid_fixed
"""
import argparse
import os
import shutil
import sys


def collect_entries(src):
    """Yield (lt_la, leaf_name, leaf_abspath) for every file/dir under
    <src>/<lt_la>/<dataset_hp>/."""
    for lt_la in sorted(os.listdir(src)):
        lt_la_path = os.path.join(src, lt_la)
        if not os.path.isdir(lt_la_path):
            continue
        for sub in sorted(os.listdir(lt_la_path)):
            sub_path = os.path.join(lt_la_path, sub)
            if not os.path.isdir(sub_path):
                continue
            for leaf in sorted(os.listdir(sub_path)):
                yield lt_la, leaf, os.path.join(sub_path, leaf)


def link_or_copy(src_path, dst_path, mode):
    if os.path.lexists(dst_path):
        if os.path.islink(dst_path):
            os.unlink(dst_path)
        elif os.path.isdir(dst_path):
            shutil.rmtree(dst_path)
        else:
            os.remove(dst_path)
    if mode == "symlink":
        os.symlink(os.path.abspath(src_path), dst_path)
    elif mode == "copy":
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
    elif mode == "move":
        shutil.move(src_path, dst_path)
    else:
        raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("src", help="Source root (e.g. .../0505_lookthink_param_grid)")
    ap.add_argument("--dst", default=None,
                    help="Destination root (default: <src>_fixed)")
    ap.add_argument("--copy", action="store_true",
                    help="Copy entries instead of symlinking")
    ap.add_argument("--inplace", action="store_true",
                    help="Mutate <src> in place (move entries up, drop middle dirs). "
                         "Mutually exclusive with --copy/--dst.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = os.path.abspath(args.src.rstrip("/"))
    if not os.path.isdir(src):
        sys.exit(f"ERROR: source not found: {src}")

    if args.inplace and (args.copy or args.dst):
        sys.exit("ERROR: --inplace cannot be combined with --copy or --dst.")

    if args.inplace:
        dst = src
        mode = "move"
    else:
        dst = os.path.abspath(args.dst) if args.dst else src + "_fixed"
        mode = "copy" if args.copy else "symlink"

    n = 0
    skipped_dupes = 0
    seen = set()  # (lt_la, leaf) — within a single lt_la, each leaf must be unique
    plan = []

    for lt_la, leaf, leaf_path in collect_entries(src):
        key = (lt_la, leaf)
        if key in seen:
            skipped_dupes += 1
            continue
        seen.add(key)
        plan.append((lt_la, leaf, leaf_path))

    if not plan:
        sys.exit(f"No entries found under {src} matching expected layout.")

    middle_dirs = set()
    for lt_la, leaf, leaf_path in plan:
        tgt_dir = os.path.join(dst, lt_la)
        tgt_path = os.path.join(tgt_dir, leaf)
        middle_dirs.add(os.path.dirname(leaf_path))
        if args.dry_run:
            print(f"{leaf_path}\n  -> {tgt_path}")
            continue
        os.makedirs(tgt_dir, exist_ok=True)
        link_or_copy(leaf_path, tgt_path, mode)
        n += 1

    if args.inplace and not args.dry_run:
        # Now remove the (presumably empty) <dataset_hp> middle dirs.
        for d in sorted(middle_dirs, reverse=True):
            try:
                os.rmdir(d)
            except OSError as e:
                print(f"WARN: could not remove {d}: {e}", file=sys.stderr)

    print(f"\nProcessed {n} entries (mode={mode}) -> {dst}")
    if skipped_dupes:
        print(f"Skipped {skipped_dupes} duplicate leaf names within the same lt_la dir.")
    if not args.dry_run:
        rel = os.path.relpath(dst)
        print(f"\nNow run:\n    python scripts/find_best_grid.py {rel}")


if __name__ == "__main__":
    main()
