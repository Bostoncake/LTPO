# Visual Workspace for LTPO-on-MLLM

## 概述

在现有 **LTPO-on-MLLM**（`ltpo_vl_dmlr.py`）的基础上，新增了一个最小版 **visual workspace + routing** 机制。

核心思路：在每个优化 step 中，根据当前 latent think tokens 的状态从静态 workspace slots 中检索最相关的视觉证据（evidence），将其注入到 forward 过程，以引导 confidence reward 优化。Reward 计算逻辑、ES/Adam 更新策略、backbone 冻结策略均与原 LTPO 完全一致，无任何改动。

---

## 新增文件

| 文件 | 作用 |
|------|------|
| [`visual_workspace.py`](visual_workspace.py) | `WorkspaceConfig` 配置类；`VisualWorkspaceBuilder` 构建 workspace slots；`WorkspaceRouter` 计算路由权重 |
| [`ltpo_vl_workspace.py`](ltpo_vl_workspace.py) | `build_inputs_vl_workspace`：在 `build_inputs_vl` 基础上新增 image mask 提取、workspace 构建、prepend 序列扩展；`generate_vl_workspace`：含 workspace 注入的优化主循环 |
| [`main_vl_workspace.py`](main_vl_workspace.py) | 评测入口，复用 `main_vl_dmlr.py` 的所有辅助函数，新增 workspace 相关 CLI 参数 |

**原有文件零改动：** `ltpo.py` / `ltpo_vl.py` / `ltpo_vl_dmlr.py` / `main_vl_dmlr.py` 均未修改。

---

## 架构说明

### VisualWorkspaceBuilder

```
输入: inputs_embeds (1, seq_len, d)  +  image_mask (1, seq_len)  +  question_state (T, d)
输出: workspace_slots  shape = (K, d)
```

**算法（question-guided top-K selection）：**

1. 从 `inputs_embeds` 中取出所有 image token 的 embedding（由 image_mask 筛选，在 `_merge_visual_tokens` 之后提取）
2. 用 `question_state`（初始 thought token embeddings）的 mean 作为 query
3. 对所有 image token 做点积打分，取 top-K
4. 若 image token 数量不足 K，tile 补全
5. 结果 `.detach()`，全程无梯度

每个样本构建一次，全优化步骤静态复用。

### WorkspaceRouter

```
输入: thought_hidden_states (T, d)  +  workspace_slots (K, d)
输出: alpha (r,)  +  selected_indices (r,)  +  selected_slots (r, d)
```

**算法：**

1. Mean-pool thought tokens → query vector (d,)
2. 与所有 K 个 slots 做点积 → scores (K,)
3. Top-r 选择 → softmax 权重 alpha (r,)
4. 返回对应的 slots

全程在 `torch.no_grad()` 下运行，routing 路径无梯度。

### 注入方式（workspace_inject_mode）

| 模式 | 行为 | 序列长度变化 |
|------|------|------------|
| `"add"` | `effective_thought = thought_cand + evidence_mean.detach()` | 无变化（默认） |
| `"prepend"` | 在 thought tokens 前插入 r 个 evidence token positions，`thought_idx` 右移 r | +r |

`"add"` 模式：evidence mean（alpha 加权的 selected slots 均值）直接叠加到 thought candidates，不改变序列布局，对 auto_grad 模式完全兼容。

`"prepend"` 模式：在 `build_inputs_vl_workspace` 阶段预分配 r 个占位 token，每个 step 写入当前路由结果，`get_confidence` 的 forward 自然看到 evidence。

### 优化主循环（per-instance）

```
build_inputs_vl_workspace()
  ├── 构建 workspace_slots  [一次]
  └── 扩展 inputs_embeds（prepend 模式）

for step in range(max_rl_steps):
    epsilon ~ N(0, sigma²)
    thought_cand = thought + epsilon

    # Workspace routing
    alpha, indices, selected_slots = router.route(thought_cand, workspace_slots)
    effective_thought = inject(thought_cand, selected_slots, alpha)   # add or prepend

    # Reward（不变）
    reward = get_confidence(model, inputs, thought_idx, effective_thought)

    # ES / Adam 更新（不变）
    thought += lr * reward * epsilon / sigma²

sigma *= sigma_decay
```

---

## 新增 Config 字段

| CLI 参数 | `WorkspaceConfig` 字段 | 类型 | 默认值 | 说明 |
|---------|----------------------|------|--------|------|
| `--use_workspace` | `enabled` | bool | `False` | 主开关；关闭时行为与 `main_vl_dmlr.py` 完全相同 |
| `--num_workspace_slots` | `num_workspace_slots` | int | `8` | K：workspace 容量（每样本构建的槽数） |
| `--num_route_slots` | `num_route_slots` | int | `2` | r：每个 step 选中并注入的槽数量 |
| `--workspace_inject_mode` | `workspace_inject_mode` | str | `"add"` | 注入方式：`"add"` 或 `"prepend"` |

---

## 工程约束

- **backbone 全冻结**：visual encoder、language model 均不训练
- **workspace slots 不训练**：`VisualWorkspaceBuilder.build()` 和 `WorkspaceRouter.route()` 全程 `no_grad + detach`
- **只有 latent think tokens 参与优化**：与原 LTPO 完全一致
- **无外部工具依赖**：不引入 detector / crop / box predictor
- **reward 不变**：沿用 `get_confidence`（confidence/margin 风格）
- **现有 benchmark/eval 路径保留**：`--eval_baseline` 仍然生效

---

## 运行命令

### 最小运行（单数据集，workspace 开启）

```bash
python main_vl_workspace.py \
    --dataset scienceqa \
    --data_root mllm_data \
    --model_name_or_path /path/to/Qwen2.5-VL-7B-Instruct \
    --output_dir ./output/workspace \
    --use_workspace \
    --num_workspace_slots 8 \
    --num_route_slots 2 \
    --workspace_inject_mode add
```

### Baseline（不启用 workspace，与 main_vl_dmlr.py 等价）

```bash
python main_vl_workspace.py \
    --dataset scienceqa \
    --data_root mllm_data \
    --model_name_or_path /path/to/Qwen2.5-VL-7B-Instruct \
    --output_dir ./output/workspace
# （不加 --use_workspace）
```

### Smoke test（单样本，3 个优化步骤）

```bash
python main_vl_workspace.py \
    --dataset scienceqa \
    --data_root mllm_data \
    --model_name_or_path /path/to/Qwen2.5-VL-7B-Instruct \
    --output_dir /tmp/ws_smoke \
    --start_data_idx 0 --end_data_idx 1 \
    --max_num_steps 3 \
    --use_workspace \
    --num_workspace_slots 4 \
    --num_route_slots 2 \
    --verbose 2
```

### 纯 Python 单元 smoke test（无需 GPU / 模型）

```python
import torch
from visual_workspace import WorkspaceConfig, VisualWorkspaceBuilder, WorkspaceRouter

embeds = torch.randn(1, 50, 1024)
mask   = torch.zeros(1, 50, dtype=torch.bool)
mask[0, 5:30] = True          # 25 image tokens
thought = torch.randn(2, 1024)

slots = VisualWorkspaceBuilder(8).build(embeds, mask, thought)
alpha, idx, sel = WorkspaceRouter(2).route(thought, slots)

assert slots.shape == (8, 1024)
assert sel.shape   == (2, 1024)
print("OK")
```

---

## 启动脚本

| 脚本 | 用途 |
|------|------|
| [`scripts/run_workspace_single.sh`](scripts/run_workspace_single.sh) | 单 GPU，遍历全部数据集，workspace 开启 |
| [`scripts/run_workspace_ablation.sh`](scripts/run_workspace_ablation.sh) | 消融：baseline / add / prepend 三组并行，用于对比 |
| [`scripts/run_workspace_3models.sh`](scripts/run_workspace_3models.sh) | 多 GPU 动态调度，3 模型 × 全数据集，workspace 开启 |
