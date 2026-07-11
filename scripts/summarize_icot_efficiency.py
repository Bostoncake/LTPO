import argparse
import json
import os
from glob import glob


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def total(records, key):
    return float(sum(float(r.get(key, 0.0) or 0.0) for r in records))


def main():
    parser = argparse.ArgumentParser(description="Summarise ICoT efficiency JSONL files.")
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    paths = sorted(glob(os.path.join(args.output_root, "*", "efficiency_profile.jsonl")))
    records = []
    by_dataset = {}
    for path in paths:
        dataset_records = list(read_jsonl(path))
        if not dataset_records:
            continue
        dataset = dataset_records[0].get("dataset") or os.path.basename(os.path.dirname(path))
        by_dataset[dataset] = dataset_records
        records.extend(dataset_records)

    profiled = [r for r in records if r.get("profiled_flops")]
    total_profiled_flops = total(profiled, "profiler_estimated_flops")
    mean_profiled_flops = total_profiled_flops / max(1, len(profiled))
    estimated_total_flops = (
        total_profiled_flops if len(profiled) == len(records)
        else mean_profiled_flops * len(records)
    )

    summary = {
        "output_root": args.output_root,
        "num_datasets": len(by_dataset),
        "num_samples": len(records),
        "num_profiled_flop_samples": len(profiled),
        "total_generation_wall_time_s": total(records, "generation_wall_time_s"),
        "total_judge_time_s": total(records, "judge_time_s"),
        "total_generated_tokens": int(total(records, "generated_tokens")),
        "total_forward_passes": int(total(records, "forward_passes")),
        "total_prefill_forward_passes": int(total(records, "prefill_forward_passes")),
        "total_decode_forward_passes": int(total(records, "decode_forward_passes")),
        "total_subimage_forward_passes": int(total(records, "subimage_forward_passes")),
        "total_attention_forward_passes": int(total(records, "attention_forward_passes")),
        "total_subimages_inserted": int(total(records, "num_sub_imgs")),
        "total_subimage_tokens_inserted": int(total(records, "subimage_tokens_inserted")),
        "total_profiled_flops": total_profiled_flops,
        "mean_profiled_flops_per_sample": mean_profiled_flops,
        "estimated_total_flops": estimated_total_flops,
        "estimated_total_tflops": estimated_total_flops / 1e12,
        "datasets": {},
    }
    wall = summary["total_generation_wall_time_s"]
    summary["generation_tokens_per_second"] = summary["total_generated_tokens"] / max(wall, 1e-12)
    summary["forward_passes_per_second"] = summary["total_forward_passes"] / max(wall, 1e-12)

    for dataset, ds_records in by_dataset.items():
        ds_profiled = [r for r in ds_records if r.get("profiled_flops")]
        ds_profiled_flops = total(ds_profiled, "profiler_estimated_flops")
        ds_mean_flops = ds_profiled_flops / max(1, len(ds_profiled))
        ds_est_flops = (
            ds_profiled_flops if len(ds_profiled) == len(ds_records)
            else ds_mean_flops * len(ds_records)
        )
        ds_wall = total(ds_records, "generation_wall_time_s")
        summary["datasets"][dataset] = {
            "num_samples": len(ds_records),
            "num_profiled_flop_samples": len(ds_profiled),
            "total_generation_wall_time_s": ds_wall,
            "avg_generation_wall_time_s": ds_wall / max(1, len(ds_records)),
            "total_generated_tokens": int(total(ds_records, "generated_tokens")),
            "total_forward_passes": int(total(ds_records, "forward_passes")),
            "total_subimages_inserted": int(total(ds_records, "num_sub_imgs")),
            "estimated_total_flops": ds_est_flops,
            "estimated_total_tflops": ds_est_flops / 1e12,
        }

    json_path = os.path.join(args.output_root, "efficiency_summary_all.json")
    md_path = os.path.join(args.output_root, "efficiency_summary_all.md")
    with open(json_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        "# ICoT Full-Run Efficiency Summary",
        "",
        f"- Output root: `{args.output_root}`",
        f"- Datasets: {summary['num_datasets']}",
        f"- Samples: {summary['num_samples']}",
        f"- Profiled FLOP samples: {summary['num_profiled_flop_samples']}",
        f"- Total generation wall time: {summary['total_generation_wall_time_s']:.4f} s",
        f"- Total generated tokens: {summary['total_generated_tokens']}",
        f"- Generation throughput: {summary['generation_tokens_per_second']:.4f} tokens/s",
        f"- Total forward passes: {summary['total_forward_passes']}",
        f"- Forward pass throughput: {summary['forward_passes_per_second']:.4f} forwards/s",
        f"- Total inserted sub-images: {summary['total_subimages_inserted']}",
        f"- Estimated total FLOPs: {summary['estimated_total_flops']:.6e}",
        f"- Estimated total TFLOPs: {summary['estimated_total_tflops']:.4f}",
        "",
        "| Dataset | Samples | Wall time (s) | Fwd passes | Sub-images | Est. TFLOPs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, ds in sorted(summary["datasets"].items()):
        lines.append(
            f"| {dataset} | {ds['num_samples']} | "
            f"{ds['total_generation_wall_time_s']:.4f} | "
            f"{ds['total_forward_passes']} | "
            f"{ds['total_subimages_inserted']} | "
            f"{ds['estimated_total_tflops']:.4f} |"
        )
    lines.extend([
        "",
        "FLOPs are PyTorch profiler estimates (`with_flops=True`). They are useful",
        "for comparing runs but may omit non-matmul operators, fused kernels, cache",
        "bookkeeping and Python overhead.",
        "",
    ])
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
