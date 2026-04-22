import os
import re
import argparse


BENCHMARKS = [
    ("MathVista", "math_vista_dev"),
    ("MathVision", "math_vision_dev"),
    ("MM-Math", "mm_math_dev"),
    ("HallusionBench", "hallusion_dev"),
    ("MMVP", "mmvp_dev"),
    ("MMStar", "mmstar_dev"),
    ("ScienceQA", "scienceqa_dev"),
]


def parse_results_log(path):
    """Parse a results.log file and return (correct, total, accuracy) or None."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        text = f.read()
    m = re.search(r"correct=(\d+),\s*total=(\d+),\s*accuracy=([\d.]+)", text)
    if m:
        return int(m.group(1)), int(m.group(2)), float(m.group(3))
    return None


def collect(output_dir):
    # discover all experiment subdirectories
    subdirs = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    ]

    # group by config (strip benchmark name part)
    configs = {}
    for d in subdirs:
        for _, bench_key in BENCHMARKS:
            tag = f"-{bench_key}-"
            if tag in d:
                prefix = d.split(tag)[0]
                suffix = d.split(tag)[1]
                config_name = f"{prefix}-{suffix}"
                configs.setdefault(config_name, {})[bench_key] = os.path.join(
                    output_dir, d, "results.log"
                )
                break

    return configs


def to_markdown(configs):
    bench_header = "| " + " | ".join(b[0] for b in BENCHMARKS) + " |"
    bench_sep = "|" + "|".join("---" for _ in BENCHMARKS) + "|"
    lines = []

    for config_name in sorted(configs.keys()):
        lines.append(f"**{config_name}**\n")
        lines.append(bench_header)
        lines.append(bench_sep)
        bench_paths = configs[config_name]
        row = []
        for _, bench_key in BENCHMARKS:
            path = bench_paths.get(bench_key)
            if path:
                result = parse_results_log(path)
                if result:
                    row.append(f"{result[2]*100:.2f}")
                else:
                    row.append("-")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Collect benchmark results into markdown table")
    parser.add_argument(
        "output_dir",
        help="Path to the output directory containing experiment subdirs",
    )
    args = parser.parse_args()

    configs = collect(args.output_dir)
    if not configs:
        print("No results found.")
        return

    md = to_markdown(configs)
    print(md)

    # save to output/log/<dirname>.md
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "output", "log")
    os.makedirs(log_dir, exist_ok=True)
    dirname = os.path.basename(os.path.normpath(args.output_dir))
    save_path = os.path.join(log_dir, f"{dirname}.md")
    with open(save_path, "w") as f:
        f.write(md + "\n")
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
