# ICoT 在 Qwen2.5-VL-7B 上低分的简明分析

## 结论

这组结果不建议表述为“测错了”。更合理的解释是：我们在 LTPO 中复现的是一个跨架构迁移版 ICoT/ADS，而不是官方 Qwen2-VL 实现的逐行等价版本；这个迁移版在 Qwen2.5-VL-7B 上触发了一个很明确的失效模式：运行中插入的视觉 token 没有按 Qwen2.5-VL 训练时需要的 3D M-RoPE 视觉位置来编码，导致插入视觉信息反而扰乱后续推理。

一句话版：

> ICoT 的 ADS 机制依赖“把被注意到的图像 patch 作为新的视觉思维插回语言序列”。在 Qwen2/Qwen2.5-VL 这类 M-RoPE 模型中，视觉 token 不只是 embedding，还依赖正确的 temporal-height-width 三维位置。我们的可移植实现为了兼容 Qwen2.5-VL/Qwen3-VL，使用了顺序 cache position 插入子图，没有完整复刻官方 Qwen2-VL ICoT 中对子图 token 的 3D 位置和 `rope_deltas` 调整。因此 Qwen2.5-VL-7B 的低分更像跨架构 ICoT 适配失效，而不是模型本身能力差或数据评测错误。

## 结果现象

`docs/MY_RUN_ICoT.md` 里 Qwen2.5-VL-7B 的 ICoT 结果明显异常：

| Dataset | Qwen2.5-VL-7B ICoT | Qwen2.5-VL-3B ICoT | Qwen3-VL-4B ICoT | Qwen3-VL-8B ICoT |
| --- | ---: | ---: | ---: | ---: |
| math_vista_dev | 11.00 | 27.67 | 51.00 | 56.67 |
| math_vision_dev | 4.00 | 12.33 | 17.00 | 22.00 |
| mm_math_dev | 2.67 | 12.00 | 25.67 | 28.33 |
| hallusion_dev | 32.00 | 52.67 | 66.00 | 70.67 |
| mmvp_dev | 50.00 | 46.00 | 76.67 | 75.00 |
| mmstar_dev | 14.67 | 44.00 | 54.33 | 56.33 |
| scienceqa_dev | 18.33 | 32.33 | 55.67 | 57.33 |

这个模式有两个重要含义：

1. 不是 Qwen2.5-VL-7B 本身不行。`docs/MY_RUN_SLOT.md` 中 Qwen2.5-VL-7B + SLOT 在同一组 dev 集上能达到 mmstar 56.67、math_vista 57.00、hallusion 67.33、scienceqa 59.00 等正常水平。
2. 也不是 ICoT 一定不行。Qwen3-VL-4B/8B 的 ICoT 结果基本在合理区间，说明同一评测脚本并没有全局崩掉。

因此异常集中在“Qwen2.5-VL-7B + 当前 ICoT 迁移实现”这个组合上。

## 代码层面的主要原因

### 1. 官方 Qwen2-VL ICoT 不是只插 embedding

原仓库 `/home/xiongyizhe/hqs/projects/rebuttal/H200-91/ICoT/qwen2_vl/icot_qwen_model.py` 在插入 16 个视觉 patch 时，同时做了三件事：

- 选出 top-16 patch embedding；
- 拼接 `<|vision_start|> + 16 * <|image_pad|> + <|vision_end|>`；
- 给这 18 个 token 构造一个手写的 3D M-RoPE 位置网格，并调整 `rope_deltas` 和 `cache_position`。

关键位置：`icot_qwen_model.py:157-168`。其中 `interleaved_position_ids` 是 3 行，对应 temporal/height/width 三个维度；`self.rope_deltas += -12` 也说明官方实现知道插入子图会改变后续位置对齐。

### 2. LTPO 迁移版为了兼容，改成了顺序 1D cache 插入

本仓库 `icot_vl.py` 的实现先用官方路径做 prefill：

- `icot_vl.py:149-158`：`input_ids + pixel_values + image_grid_thw`，这一步位置是正常的。

但进入逐 token decoding 之后，后续 token 和插入子图都走 `inputs_embeds`：

- `icot_vl.py:219-227`：普通解码 token 用 `inputs_embeds=next_embed`；
- `icot_vl.py:248-260`：子图 token 直接拼成 embedding 后用 `inputs_embeds=sub`；
- `icot_vl.py:255-256`：子图的 `cache_position` 是简单连续的一维位置；
- 没有为子图重建 `input_ids`/`image_grid_thw`，也没有复刻原实现的 3D `interleaved_position_ids` 和 `rope_deltas` 修正。

这和官方 Qwen2-VL ICoT 的位置处理不等价。

### 3. Qwen2.5-VL 明确依赖图像 token 的 3D M-RoPE

当前环境里的 `transformers` Qwen2.5-VL 实现写得很清楚：`get_rope_index` 会根据 `input_ids + image_grid_thw` 给视觉部分计算 temporal/height/width 三维位置；如果拿不到这些信息，就退化到普通的顺序 RoPE。

关键位置：

- `/home/xiongyizhe/miniconda3/envs/ltpo/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:956-1015`：函数说明和返回的 3D position ids；
- 同文件 `1022-1119`：只有 `input_ids` 与 `image_grid_thw` 存在时才计算视觉 3D 位置；
- 同文件 `1120-1138`：否则走普通 1D fallback。

本仓库自己的 LTPO 最终版也已经绕过了类似问题：`ltpo_vl_dmlr_final.py` 顶部注释说明，直接 `generate(inputs_embeds=...)` 会破坏 Qwen2.5-VL 的 M-RoPE、image-token scatter、`rope_deltas` 和 KV-cache 逻辑，所以改成 hook 官方 `generate(**inputs)` 路径。当前 `icot_vl.py` 为了运行中插入视觉 token，又重新落入了这个风险区。

## 为什么 Qwen3 结果反而还可以

这不矛盾。Qwen3-VL 的模型结构和训练目标更偏向长上下文、强 interleaved multimodal 输入。Qwen3-VL 技术报告称其支持 256K interleaved multimodal context，并引入 enhanced interleaved-MRoPE 和 DeepStack；Transformers 的 Qwen3-VL forward 也会传 `visual_pos_masks` 和 `deepstack_visual_embeds` 到语言模型。也就是说，Qwen3-VL 对“中途又出现一些视觉相关 token”的分布外扰动可能更鲁棒。

但这不能反推 Qwen2.5-VL-7B 应该同样有效。Qwen2.5-VL 的视觉 token 位置更依赖正确的 M-RoPE 组织方式；当前迁移版在 Qwen2.5-VL-7B 上低分，符合“视觉思维 token 位置编码错误导致干扰”的机制预期。

## 还有两个次要因素

### 1. ADS 本身选择的是离散 top-k patch，可能不是语义连续区域

原 ICoT/ADS 用注意力选 top-k image token。后续 DaP-ICoT 工作已经指出，已有 ICoT 方法存在两个限制：固定位置插入视觉信息会造成静态、低效的推理模式；离散视觉 token 会形成不连续、语义不连贯的 visual thought，可能损害理解并遗漏关键信息。

这正好对应我们的实现：每次 newline 触发，固定插入 16 个按注意力 top-k 选出的 patch，而不是一个语义完整区域。

### 2. prompt/解码/评测与官方 ICoT 不同

官方 Qwen2-VL ICoT 主要在 M3CoT/ScienceQA/LLaVA-W 等设置下跑，脚本里是 `do_sample=True, temperature=0.8, top_p=0.9, max_new_tokens=128`。我们这里是 7 个 dev 集、LTPO 的 boxed/system prompt、greedy token-by-token decoding、Qwen-Max LLM judge。这个迁移是合理 baseline，但不是官方 ICoT 实验的严格同条件复现。

`docs/MY_RUN_ICoT.md` 还记录过第一次 Qwen2.5-VL-7B 会在开头模板之后马上触发插入，导致 mmstar/scienceqa/math 几乎崩到约 2%；后来加了 `step >= 60` 和间隔 80 token 才恢复部分正常。这也说明 Qwen2.5-VL-7B 对插入时机和格式非常敏感。

## 对审稿人的建议表述

可以这样写：

> We agree that the Qwen2.5-VL-7B ICoT result is unusually low. Our analysis suggests that this is not caused by a failure of the base model or the evaluator, but by an architecture-specific limitation of directly transferring ADS-style ICoT to Qwen2.5-VL. In Qwen2/Qwen2.5-VL, visual tokens are not ordinary sequential tokens: they require multimodal RoPE positions derived from the temporal-height-width image grid. The official Qwen2-VL ICoT implementation explicitly constructs a 3D position layout and adjusts `rope_deltas` when inserting visual-thought tokens. Our portable implementation, designed to run across Qwen2.5-VL and Qwen3-VL without patching Transformers, inserts selected visual embeddings through the cache with sequential positions. This creates out-of-distribution visual-thought tokens for Qwen2.5-VL, and can make the inserted visual thoughts harmful rather than helpful. This interpretation is also consistent with recent follow-up work on ICoT, which identifies static visual-thought positioning and broken/discontinuous visual-token representations as failure modes of prior ICoT-style methods.

中文可简化为：

> Qwen2.5-VL-7B 的 ICoT 低分不是说明主模型能力差，而是 ADS-style ICoT 对 Qwen2.5-VL 的位置编码和视觉 token 表示要求非常敏感。官方 Qwen2-VL ICoT 插入子图时手动构造 3D M-RoPE 位置并修正 `rope_deltas`；我们的跨模型实现为了兼容 Qwen2.5/Qwen3，采用连续 cache position 插入视觉 embedding，导致 Qwen2.5-VL-7B 看到的视觉思维 token 分布外。后续 ICoT 工作也指出固定插入和离散 top-k 视觉 token 会产生破碎、不连贯的 visual thoughts，因此该异常可以解释为 ICoT 机制在该架构迁移下的失效，而非简单实验错误。

## 可引用资料

- ICoT 原论文：Gao et al., Interleaved-Modal Chain-of-Thought, CVPR 2025 / arXiv 2411.19488. https://arxiv.org/abs/2411.19488
- ICoT 官方仓库：说明 ADS/ICoT 是 plug-and-play，并包含 Qwen2-VL patch 实现。https://github.com/jungao1106/ICoT
- Qwen2.5-VL Transformers 文档：Qwen2.5-VL 使用 upgraded MRoPE 和 3D position IDs。https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl
- Qwen2-VL 技术报告：Qwen2-VL 引入 dynamic resolution 和 M-RoPE 来融合多模态信息。https://arxiv.org/html/2409.12191v2
- Qwen3-VL 技术报告：Qwen3-VL 支持 interleaved multimodal context，并引入 enhanced interleaved-MRoPE 和 DeepStack。https://arxiv.org/abs/2511.21631
- DaP-ICoT 后续工作：指出已有 ICoT 存在 static visual thought positioning 和 broken visual thought representation。https://arxiv.org/abs/2603.21754
