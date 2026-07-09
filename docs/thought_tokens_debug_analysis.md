# Latent Thought Tokens 调试分析报告

> 日期: 2026-04-25
> 目标: 验证 `ltpo_vl_dmlr_v4.py` 中 `latent_thought_tokens` 在实际运行时是否真正影响模型的生成输出。

---

## 1. 背景

LTPO 的核心机制是在 prompt 中插入若干特殊的 "thought tokens"（对于 Qwen 模型就是 `<|endoftext|>` 重复 N 次），然后在 embedding 层面上通过 RL 优化这些 token 的 hidden states，期望优化后的 embeddings 能引导模型生成更好的回答。

**关键问题**: 这些 thought tokens 在实际推理过程中真的被模型"读取"并影响输出了吗？还是模型完全忽略了它们？

## 2. 调试方法

编写了两个调试脚本 (`debug_thought_tokens.py`, `debug_thought_tokens_v2.py`)，对以下 5 个维度进行检查：

| 检查项 | 做法 |
|--------|------|
| **CHECK 1**: token 定位 | 打印 thought token 的字符串表示和 token ID，确认它们被正确编码 |
| **CHECK 2**: 索引正确性 | 调用 `build_inputs_vl` 后检查 `thought_idx`、embeddings shape、thought region 的统计量（norm/mean/std） |
| **CHECK 3**: RL 优化有效性 | 执行 RL 循环，逐步记录 reward 值和 embedding 与初始值的 L2 距离 |
| **CHECK 4**: 生成影响 | 分别用**原始**、**优化后**、**随机**三种 embeddings 生成输出，对比文本是否不同 |
| **CHECK 5**: confidence 灵敏度 | 对三种 embeddings 计算 confidence reward，验证 reward 函数是否对 embedding 变化敏感 |

实验配置:
- 模型: `Qwen2.5-VL-3B-Instruct`
- 数据集: `mmvp_dev` (前 2 个样本)
- 设备: 单张 NVIDIA H200

## 3. 实验一: 默认参数 (2 tokens, 5 RL 步)

对应脚本默认运行配置 (`--num_thought_tokens 2 --max_num_steps 15`，调试中只跑了 5 步加速)。

### 3.1 结果

**CHECK 1 & 2 — token 定位 ✅**

```
thought token string : '<|endoftext|><|endoftext|>'
thought token IDs    : [151643, 151643]
thought_idx          : [236, 238]
inputs_embeds shape  : torch.Size([1, 243, 2048])
thought region norm  : 1.531441
thought region std   : 0.023931
```

thought tokens 被正确插入到 token 序列中，位置 236-237（共 2 个 token），且 embeddings 非零。

**CHECK 3 — RL 优化 ✅**

```
step 0: reward=9.3893  embed_diff_from_init=0.2394  sigma=23.7500
step 1: reward=9.2029  embed_diff_from_init=0.3425  sigma=22.5625
step 2: reward=9.0206  embed_diff_from_init=0.4259  sigma=21.4344
step 3: reward=9.3761  embed_diff_from_init=0.5067  sigma=20.3627
step 4: reward=9.1479  embed_diff_from_init=0.5853  sigma=19.3445
```

- reward 在 9.0~9.4 之间波动
- embedding 与初始值的 L2 距离逐步增长到 ~0.59

**CHECK 4 — 生成对比 ❌**

| 配置 | 输出片段 |
|------|---------|
| 原始 embeddings | "...The butterfly in the **image** is clearly visible with its wings spread out..." |
| 优化后 embeddings | "...The butterfly in the **image** is clearly visible with its wings spread out..." |
| 随机 embeddings | "...The butterfly is a **Monarch butterfly**..." |

- **原始 vs 优化: 完全相同** — 5 步 RL 的变化量不足以影响输出
- **原始 vs 随机: 不同** — 证明模型确实在读取 thought token 位置的 embeddings

**CHECK 5 — confidence 灵敏度 ✅**

```
confidence (original)  = 15.5069
confidence (optimised) = 15.3695
confidence (random)    = 12.5367
```

reward 函数对 embedding 变化是敏感的，三种设置给出了不同的值。

### 3.2 小结

默认参数下（2 tokens, 5 步），RL 优化**确实在修改 embeddings**，但**修改幅度不够大**（diff ~0.59），不足以改变生成输出。模型生成过程中 thought tokens 的位置**确实被模型读取**（随机 embeddings 能改变输出），但 RL 优化产生的微小扰动被后续 243 个 token 的上下文稀释了。

---

## 4. 实验二: 增大参数

为了确认"变化量不够大"是根本原因，进行了三组对照实验。

### 4.1 TEST A: 2 tokens, 50 RL 步

```
step   0: reward= 9.3893  diff= 0.2394
step  10: reward= 9.0479  diff= 1.0333
step  20: reward=10.1221  diff= 2.1884
step  30: reward=11.5844  diff= 4.0089
step  40: reward=11.6549  diff= 6.9494
Best reward: 12.5596, final diff from init: 2.6263
```

| 配置 | 输出片段 |
|------|---------|
| 原始 | "...The butterfly in the **image** is clearly visible..." |
| 优化 (50步) | "...The **image** shows a butterfly..." |

**输出不同了 ✅** — 50 步后 diff 达到 2.63，足以影响生成。

### 4.2 TEST B: 手动放大 diff 倍数

把 50 步优化得到的 delta 向量乘以不同的倍数：

| 放大倍数 | diff norm | 与原始输出不同？ |
|---------|-----------|-----------------|
| 1x | 2.63 | ✅ YES |
| 10x | 26.26 | ✅ YES |
| 100x | 262.63 | ✅ YES |
| 1000x | 2626.29 | ✅ YES |

所有放大后的 embeddings 都产生了不同于原始的输出。说明模型对 thought token 位置的 embedding 值是敏感的。

### 4.3 TEST C: 10 tokens, 30 RL 步

```
step   0: reward=5.4858  diff=0.3141
step  15: reward=6.1301  diff=2.0559
step  25: reward=7.7432  diff=4.3467
```

| 配置 | 输出片段 |
|------|---------|
| 原始 | "...The butterfly in the **image** is clearly visible..." |
| 优化 (30步) | "...The **image** shows a butterfly..." |
| 清零 | "...The butterfly is **clearly** visible with its wings..." |

三者输出均不同 ✅。清零 thought tokens 也导致输出变化，进一步确认模型在读取这些位置。

---

## 5. 结论

### latent_thought_tokens 是否起作用？

**是的，但在当前默认配置下效果可能不稳定。**

具体来说：

1. **机制本身是有效的**:
   - thought tokens 被正确插入到 token 序列中
   - `build_inputs_vl` 正确定位了 thought token 在 embeddings 中的位置
   - 模型确实在 forward pass 中读取了这些位置的 embedding 值（不同的 embedding → 不同的输出）
   - RL 优化循环确实在更新 embeddings，reward 确实在变化

2. **问题在于变化量的阈值**:
   - 从实验数据来看，embedding 变化的 L2 norm 需要达到 **~2.0 以上**才能稳定地影响生成输出
   - 默认配置 (2 tokens, 15 步, sigma=25, sigma_decay=0.95) 在 15 步后的 diff 大约在 1.0-1.5 之间，处于**临界区域** — 部分样本可能受影响，部分不会
   - 5 步时 diff 仅 ~0.6，几乎不可能影响输出

3. **sigma_decay 衰减过快是一个因素**:
   - `sigma_decay=0.95` 使得 sigma 从 25.0 衰减到:
     - 第 5 步: 19.3
     - 第 15 步: 11.6
     - 第 30 步: 5.3
     - 第 50 步: 1.8
   - 后期扰动幅度很小，grad ascent 更新量也随之缩小

### 建议的改进方向

| 方向 | 具体做法 | 预期效果 |
|------|---------|---------|
| 增加 thought tokens 数量 | `--num_thought_tokens 10` 或更多 | 更多可优化参数，更大的总影响面 |
| 增加 RL 步数 | `--max_num_steps 30~50` | 更充分地优化 embeddings |
| 减缓 sigma 衰减 | `--sigma_decay 0.98` 或 `0.99` | 后期仍有足够的探索幅度 |
| 增大学习率 | `--lr 0.05` | 每步更大的更新幅度 |

---

## 附录: 复现方法

```bash
cd /export/home/lanliwei.1/abcxyz/projects/latent-reasoning/LTPO

# 实验一: 默认参数快速检查
CUDA_VISIBLE_DEVICES=0 python debug_thought_tokens.py

# 实验二: 深度对照实验
CUDA_VISIBLE_DEVICES=0 python debug_thought_tokens_v2.py
```

调试脚本位于项目根目录:
- `debug_thought_tokens.py` — 基础 5 项检查
- `debug_thought_tokens_v2.py` — 变化量阈值探索 (50 步、放大倍数、10 tokens、清零对照)
