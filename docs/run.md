# run the scripts2

按照 First，Second，Third，来表示当时跑代码先后的顺序。
用scripts2的原因是，H200联不了网，所以先H200跑，然后用4090跑step4_verify_results.py。

## First

对应表格栏目：
> Qwen-2.5-VL-3B:
> Vanilla	全量 qwen-max 复现（LTPO codebase, align DMLR settings, mllm branch）:

```bash
bash scripts2/step3a_baseline_h200.sh
python scripts2/step4_verify_results.py ./output/dmlr_vanilla
```

## Second

对应表格栏目：
> 3个模型的：
> Vanilla	dev-set qwen-max 复现（LTPO codebase, align DMLR settings, mllm branch）:

```bash
bash scripts2/step3a_baseline_3models_h200.sh
python scripts2/step4_verify_results.py ./output/dmlr_vanilla_dev
```

## Third

对应表格栏目：
> 3个模型的调参：
> LTPO	dev_set	qwen-max 调参

```bash
bash scripts2/step3b_grid_search_3models_h200.sh
python scripts2/step4_verify_results.py ./output/ltpo_dmlr_grid_dev
```

这个输出整理出来两个栏目，分别是：
> LTPO	dev_set	 qwen-max	调参（每个bench取最大值）
> LTPO	dev_set	qwen-max	调参（最优参数值tokens2_steps15_sigma25.0_decay0.95_1r5e-3_topk10）

## Fourth

对应表格栏目：
> Qwen-2.5-VL-3B:
> Contrastive_reward	dev_set	qwen-max

> 优化训练的时候，每次做两次前向，带着视觉信息的和不带视觉信息的，reward设计成带视觉信息的confidence（现在的reward） 减 不带视觉信息的confidence。
> reward = confidence(with_visual) - confidence(without_visual)

```bash
bash scripts2/step_contrastive_reward_h200.sh
python scripts2/step4_verify_results.py ./output/contrastive_reward_dev
```

## Fifth

对应表格栏目（还没更新）：
> Qwen-2.5-VL-3B:
> Contrastive_reward	dev_set	qwen-max 调参

> 考虑这样：reward=a*原来的reward+b*现在的对比reward，a=1，调b
> reward = conf(with_visual) + beta * (conf(with_visual) - conf(without_visual))

```bash
bash scripts2/step_contrastive_grid_search_h200.sh
python scripts2/step4_verify_results.py ./output/contrastive_reward_dev
```
