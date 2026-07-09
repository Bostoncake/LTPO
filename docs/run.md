# run

## quick start

nohup bash ./scripts/run_all_models_v8.sh &> ./output/log/run_all_models_v8.log &
nohup bash ./scripts/run_ltpo_vl_dmlr_dev_v8.sh &> ./output/log/dmlr_prompt8.log &
nohup bash ./scripts/run_ltpo_vl_dmlr_dev_v8_baseline.sh &> ./output/log/dmlr_prompt8_baseline.log &
nohup bash ./scripts/run_ltpo_vl_dmlr_dev_v8_8B_baseline.sh &> ./output/log/dmlr_prompt8_8B_baseline.log &
nohup bash ./scripts/run_ltpo_vl_dmlr_dev_v8_8B.sh &> ./output/log/dmlr_prompt8_8B.log &
nohup bash ./scripts/run_ltpo_vl_dmlr_dev_v9_8B_baseline.sh &> ./output/log/dmlr_prompt9_8B_baseline.log &

nohup bash ./scripts/run_all_models_v8_copy.sh &> ./output/log/run_all_models_v8_hall.log &

python collect_results.py output/dmlr_aligned_dev_v8_prompt
python collect_results.py output/dmlr_aligned_dev_v8_prompt_baseline
python collect_results.py output/dmlr_aligned_dev_v8_prompt_baseline_8B
python collect_results.py output/dmlr_aligned_dev_v8_prompt_baseline
python collect_results.py output/dmlr_aligned_full_v7_prompt
python collect_results.py output/dmlr_aligned_full_v7_prompt_baseline
python collect_results.py output/dmlr_aligned_dev_v8_prompt_baseline_91
python collect_results.py output/dmlr_aligned_dev_v8_prompt_baseline_91_sample
python collect_results.py output/dmlr_aligned_dev_v8_prompt_baseline_v10_sample
python collect_results.py output/dmlr_aligned_dev_v8_8B_baseline
python collect_results.py output/dmlr_aligned_dev_v8_8B
python collect_results.py output/dmlr_aligned_dev_v9_8B_baseline


nohup bash ./scripts/run_ltpo_vl_dmlr_dev.sh &> ./output/log/dmlr_original_prompt.log &
nohup bash ./scripts/run_ltpo_vl_dmlr_dev_v2.sh &> ./output/log/dmlr_prompt2_test.log &
nohup bash ./scripts/run_ltpo_vl_dmlr_dev_v3.sh &> ./output/log/dmlr_prompt3.log &
nohup bash ./scripts/run_ltpo_vl_dmlr_dev_v4.sh &> ./output/log/dmlr_prompt4.log &
nohup bash ./scripts/run_ltpo_vl_dmlr_dev_v5.sh &> ./output/log/dmlr_prompt5.log &
nohup bash ./scripts/run_ltpo_vl_dmlr_dev_v6.sh &> ./output/log/dmlr_prompt6.log &
nohup bash ./scripts/run_ltpo_vl_dmlr_dev_v7.sh &> ./output/log/dmlr_prompt7_3_4B.log &
nohup bash ./scripts/run_ltpo_vl_dmlr_dev_v7_qwen3_short.sh &> ./output/log/dmlr_prompt7_qwen3_short.log &
nohup bash ./scripts/run_ltpo_vl_dmlr_dev_v7_qwen3_short_baseline.sh &> ./output/log/dmlr_prompt7_qwen3_short_baseline.log &

nohup bash ./scripts/run_ltpo_vl_dmlr_dev_v8.sh &> ./output/log/dmlr_prompt8_baseline.log &
nohup bash ./scripts/run_ltpo_vl_dmlr_dev_v8.sh &> ./output/log/dmlr_prompt8.log &

nohup bash ./scripts/run_ltpo_vl_dmlr_dev_baseline.sh &> ./output/log/dmlr_original_prompt_baseline.log &
nohup bash ./scripts/run_ltpo_vl_dmlr_dev_v2_baseline.sh &> ./output/log/dmlr_baseline2_test.log &




