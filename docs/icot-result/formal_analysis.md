# ICoT 异常结果的形式化描述

## 1. 问题定义

设一个 VLM 在输入图像 token 序列 $V=(v_1,\dots,v_n)$ 和文本上下文 $X$ 下生成答案：

$$
p_\theta(y_t \mid X, V, y_{<t}; \Pi)
$$

其中 $\Pi$ 是位置编码方案。对 Qwen2/Qwen2.5-VL 这类模型，$\Pi$ 不是普通一维位置，而是对视觉 token 使用三维 M-RoPE：

$$
\Pi(v_i) = (\pi^T_i,\pi^H_i,\pi^W_i)
$$

并且这些位置由 `input_ids` 中的 `<|vision_start|> <|image_pad|> ... <|vision_end|>` 结构和 `image_grid_thw=(T,H,W)` 共同决定。

ICoT/ADS 在生成过程中引入一个视觉思维操作：

$$
S_t = \operatorname{TopK}_{i \in [1,n]} A_t(i)
$$

其中 $A_t(i)$ 是当前文本状态对原图视觉 token $v_i$ 的注意力分数。然后把选中的视觉 patch 重新插入到生成序列中：

$$
C_{t+1} = [C_t;\ \langle vs\rangle,\ v_{i_1},\dots,v_{i_k},\ \langle ve\rangle]
$$

该操作成立的隐含前提是：

$$
\Pi_{\text{insert}}(v_{i_j}) \approx \Pi_{\text{train}}(v_{i_j})
$$

即插入后的视觉 token 仍被模型识别为合法视觉 token，而不是普通文本位置上的任意 embedding。

## 2. 官方 Qwen2-VL ICoT 满足的条件

官方 Qwen2-VL ICoT 在插入 $S_t$ 时近似满足这个条件。代码路径：

`/home/xiongyizhe/hqs/projects/rebuttal/H200-91/ICoT/qwen2_vl/icot_qwen_model.py`

关键步骤：

1. `157-159`：构造 `[151652] + [151655] * 16 + [151653]`，即 `<|vision_start|> + image_pad + <|vision_end|>` 的视觉 token 外壳。
2. `161-164`：手写 `interleaved_position_ids`，形状为 $3 \times 18$，分别对应 temporal/height/width 位置。
3. `165-168`：调整 `rope_deltas` 和 `cache_position`，让后续文本 token 的位置继续对齐。

可形式化为：

$$
\Pi_{\text{official}}(S_t)
= \operatorname{Grid3D}(4 \times 4) + \operatorname{offset}(C_t)
$$

并且后续 token 的位置 offset 被同步修正。

## 3. 当前 LTPO 迁移版实际满足的条件

本仓库 `icot_vl.py` 的 prefill 是正确的：

$$
\Pi_{\text{prefill}}(V)=\operatorname{M\text{-}RoPE}(input\_ids, image\_grid\_thw)
$$

对应 `icot_vl.py:149-158`。

但是 decode 和插入子图时，代码变为：

$$
\Pi_{\text{insert}}(S_t)
= \operatorname{Seq1D}(cur\_len,\dots,cur\_len+17)
$$

对应：

- `icot_vl.py:219-227`：单步 decode 只传 `inputs_embeds` 和 `cache_position`；
- `icot_vl.py:248-260`：子图也只传拼好的 `inputs_embeds=sub`；
- `icot_vl.py:255-256`：子图位置是连续一维 `torch.arange(cur_len, cur_len + sub.size(1))`；
- 没有为插入子图构造新的 `input_ids`、`image_grid_thw` 或 3D `position_ids`。

因此：

$$
\Pi_{\text{insert}}(S_t) \ne \Pi_{\text{official}}(S_t)
$$

并且偏差不是一个小的常数平移，而是维度结构不同：官方是三维局部视觉网格，当前实现是三维上重复的一维顺序位置。

## 4. 为什么这会造成 Qwen2.5-VL-7B 失效

Qwen2.5-VL 的 `get_rope_index` 明确分两支：

$$
\Pi =
\begin{cases}
\operatorname{M\text{-}RoPE}(input\_ids, image\_grid\_thw), & input\_ids \ne \varnothing \land image\_grid\_thw \ne \varnothing \\
\operatorname{RoPE}_{1D}(attention\_mask), & \text{otherwise}
\end{cases}
$$

代码依据：

- `modeling_qwen2_5_vl.py:956-1015`：函数说明视觉部分需要 temporal/height/width 位置；
- `modeling_qwen2_5_vl.py:1022-1119`：有 `input_ids` 和 `image_grid_thw` 时计算 3D M-RoPE；
- `modeling_qwen2_5_vl.py:1120-1138`：否则退化到一维位置。

当前 ICoT 插入阶段缺少 `input_ids` 和 `image_grid_thw`，所以插入的 $S_t$ 在 Qwen2.5-VL 看来不是训练分布中的图像块序列，而是“文本时间线上出现的一段视觉 embedding”。这会改变下一步生成分布：

$$
D_{\mathrm{KL}}\left(
p_\theta(y_{t+1}\mid C_t,S_t;\Pi_{\text{official}})
\parallel
p_\theta(y_{t+1}\mid C_t,S_t;\Pi_{\text{current}})
\right) \gg 0
$$

当这种分布漂移发生在推理中段，并且最多重复 3 次时，原本应提供视觉证据的 $S_t$ 会变成噪声或错误提示：

$$
\Delta =
\operatorname{Acc}(\text{ICoT-current})
- \operatorname{Acc}(\text{vanilla/base})
< 0
$$

这和观察一致：Qwen2.5-VL-7B 在 SLOT 或普通 baseline 中不是弱模型，但在当前 ICoT 迁移版中显著下降。

## 5. 不是“测错”的三个论据

### 论据 A：异常具有模型-方法交互性

如果是评测脚本或 judge 全局错误，Qwen3-VL-4B/8B 不应同时得到合理结果。实际 Qwen3-VL-4B/8B 的 ICoT 结果明显高于 Qwen2.5-VL-7B，说明评测流程没有整体失效。

### 论据 B：Qwen2.5-VL-7B 本身能力正常

`docs/MY_RUN_SLOT.md` 中 Qwen2.5-VL-7B + SLOT 在同一 dev 评测设置下表现正常，例如：

| Dataset | Qwen2.5-VL-7B + SLOT |
| --- | ---: |
| mmvp_dev | 69.67 |
| mmstar_dev | 56.67 |
| math_vista_dev | 57.00 |
| hallusion_dev | 67.33 |
| scienceqa_dev | 59.00 |

因此低分不能归因于 Qwen2.5-VL-7B 基座能力。

### 论据 C：已有工作指出 ICoT/ADS 的天然失效模式

原 ICoT 论文提出 ADS：用注意力图选择图像区域并插入视觉 rationale，它的目标是把文本 CoT 扩展为图文交错 CoT。见 Gao et al. 2025： https://arxiv.org/abs/2411.19488

后续 DaP-ICoT 明确指出早期 ICoT 存在两个问题：

1. static visual thought positioning：固定位置插入视觉信息，导致推理模式僵硬、冗余；
2. broken visual thought representation：选择离散、不连续的视觉 token，语义不连贯，可能损害理解并遗漏关键信息。

见 Liu et al. 2026： https://arxiv.org/abs/2603.21754

我们的实现正好同时满足这两个风险条件：

$$
S_t = \operatorname{TopK}(A_t)
$$

选择的是离散 patch，而触发策略是 newline 间隔和固定 token 间隔：

$$
\operatorname{Insert}(S_t) \iff
linebreaks \ge last+2 \land step \ge 60 \land step-last\_step \ge 80
$$

这不是根据“当前是否真的需要视觉信息”动态决定，也不保证 top-k patch 构成语义完整区域。

## 6. Qwen3-VL 为何更鲁棒

Qwen3-VL 的技术报告声称其原生支持长 interleaved multimodal context，并引入 enhanced interleaved-MRoPE 和 DeepStack。见： https://arxiv.org/abs/2511.21631

在当前 Transformers 实现里，Qwen3-VL forward 也有额外的视觉路径：

- `modeling_qwen3_vl.py:1137-1143`：获取 image embeddings；
- `modeling_qwen3_vl.py:1153-1175`：构建 `visual_pos_masks` 和 `deepstack_visual_embeds`；
- `modeling_qwen3_vl.py:1223-1231`：把这些视觉信号传入 language model。

这说明 Qwen3-VL 在架构和训练上更强调 interleaved multimodal context。当前迁移版虽然仍不严格等价，但 Qwen3-VL 对这种扰动更鲁棒，所以结果没有像 Qwen2.5-VL-7B 那样崩。

## 7. Rebuttal 可用的形式化表述

可以把异常描述成一个“视觉思维 token 合法性条件”：

$$
\mathcal{C}_{valid}(S_t) =
\left[
S_t \text{ has valid visual boundary tokens}
\right]
\land
\left[
\Pi(S_t)=\operatorname{Grid3D}(S_t)+offset
\right]
\land
\left[
rope\_deltas \text{ is consistent after insertion}
\right]
$$

官方 Qwen2-VL ICoT 近似满足：

$$
\mathcal{C}_{valid}^{official}(S_t) = 1
$$

当前跨模型迁移版满足第一项，但不满足后两项：

$$
\mathcal{C}_{valid}^{current}(S_t)
= 1 \land 0 \land 0 = 0
$$

因此，ICoT 在 Qwen2.5-VL-7B 上的低分可以被解释为：

$$
\text{ICoT failure}
=
\text{ADS discontinuity}
 \oplus
\text{invalid M-RoPE placement}
 \oplus
\text{fixed insertion schedule}
$$

其中最主要、最可由代码证明的项是 invalid M-RoPE placement。

## 8. 建议写法

英文：

> The anomalously low Qwen2.5-VL-7B result is best understood as an architecture-specific failure mode of transferring ADS-style ICoT, rather than a base-model or evaluator failure. In Qwen2/Qwen2.5-VL, inserted visual-thought tokens must be accompanied by a valid multimodal RoPE layout and consistent `rope_deltas`. The official Qwen2-VL ICoT implementation explicitly constructs a 3D position layout for the inserted visual tokens and adjusts the cache positions. Our portable implementation inserts selected visual embeddings with sequential cache positions so that it can run across Qwen2.5-VL and Qwen3-VL without patching Transformers. This approximation makes the inserted visual thoughts out-of-distribution for Qwen2.5-VL, which can turn visual-thought insertion from a helpful evidence retrieval step into a perturbation. This is consistent with follow-up ICoT work that identifies static visual-thought insertion and broken/discontinuous visual-token representations as failure modes of prior ICoT methods.

中文：

> Qwen2.5-VL-7B 的 ICoT 结果异常低，可以形式化为视觉思维 token 的合法性条件被破坏：插入的视觉 token 虽然有视觉边界 token 和 patch embedding，但缺少 Qwen2.5-VL 所需的 3D M-RoPE 子图位置与一致的 `rope_deltas` 修正。官方 Qwen2-VL ICoT 为插入子图显式构造三维位置并调整缓存；我们的跨模型实现为了兼容 Qwen2.5-VL/Qwen3-VL，采用连续 cache position 插入 embedding，导致 Qwen2.5-VL 看到分布外 visual thoughts。结合后续工作指出的 fixed insertion 和 broken visual-token representation 问题，这一低分更应解释为 ICoT/ADS 在 Qwen2.5-VL 架构迁移下的失效模式，而不是简单评测错误。
