"""
main_vl_dmlr_paired.py – Evaluate LTPO paired-noise per-token variant on
VL benchmarks with the DMLR-compatible pipeline.

This is a minimal wrapper around main_vl_dmlr.py: it reuses all the args,
answer extraction, verification, and main loop, but swaps in the paired
per-token `generate_vl` from ltpo_vl_dmlr_paired.

Usage:
    python main_vl_dmlr_paired.py \\
        --dataset scienceqa \\
        --data_root mllm_data \\
        --model_name_or_path /path/to/Qwen2.5-VL-7B-Instruct \\
        --output_dir ./output \\
        --use_llm_verify
"""

import main_vl_dmlr
import ltpo_vl_dmlr_paired


# Swap the generate_vl referenced inside main_vl_dmlr.main() with the
# paired-noise per-token implementation.
main_vl_dmlr.generate_vl = ltpo_vl_dmlr_paired.generate_vl


if __name__ == "__main__":
    args = main_vl_dmlr.parse_args()
    for arg in vars(args):
        print(f"-- {arg}: {getattr(args, arg)}")
    main_vl_dmlr.main(args)
