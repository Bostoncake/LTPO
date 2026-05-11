你需要基于当前 codebase 中已有的 main_vl_dmlr.py 和 ltpo_vl_dmlr.py，实现一个最小改动版本的 “Contrastive Latent-DPO” 测试时间 latent token 优化方法，用于替代原 LTPO 中基于 REINFORCE 的 latent representation update。

重要要求：
1. 不要直接修改原始 main_vl_dmlr.py 和 ltpo_vl_dmlr.py。
2. 请先将 main_vl_dmlr.py 复制为 main_vl_dpo.py，将 ltpo_vl_dmlr.py 复制为 ltpo_vl_dpo.py。
3. 后续所有代码修改都只在 main_vl_dpo.py 和 ltpo_vl_dpo.py 中完成。
4. 原始 main_vl_dmlr.py 和 ltpo_vl_dmlr.py 必须保持完全不变。
5. 修改应尽可能小，优先复用原有 LTPO 的函数、参数、模型加载、latent token 插入、reward 计算和生成流程。
6. 不要重构整个工程，不要引入复杂新模块。
7. 代码尽可能简洁，只保留必要的最少部分注释。

请先阅读 main_vl_dmlr.py 和 ltpo_vl_dmlr.py，理解当前 LTPO 的：
- latent token 插入方式；
- latent candidate 采样方式；
- reward / confidence score 计算方式；
- REINFORCE latent update；
- 最终 answer generation 流程。

然后在新文件中完成如下功能。

核心算法目标：
当前 LTPO 大致是：

- 维护 latent tokens H；
- 每一步采样 perturbation：
  A_i = H + eps_i；
- 对每个 A_i 计算 reward；
- 用 REINFORCE 根据 reward 更新 H。

现在需要在 ltpo_vl_dpo.py 中新增或替换为 DPO-style latent preference update：

- 对同一个输入，构造 full visual input 和 masked visual input；
- 对每个 sampled latent candidate A_i，分别计算：
  r1_i = score(full_image, prompt, A_i)
  r2_i = score(masked_image, prompt, A_i)
- 定义 contrastive score：
  g_i = r1_i - lambda_mask * r2_i
  默认 g_i = r1_i - r2_i
- 用 g_i 构造 latent candidate 之间的 preference pair；
- 用 DPO-style loss/update 优化 H，而不是使用 REINFORCE。

具体实现要求：

1. 文件复制与导入关系

   - 复制 main_vl_dmlr.py -> main_vl_dpo.py。
   - 复制 ltpo_vl_dmlr.py -> ltpo_vl_dpo.py。
   - 在 main_vl_dpo.py 中，将原来 import ltpo 的地方改为 import ltpo_dpo，或从 ltpo_dpo import 对应函数。
   - 不要影响原 main_vl_dmlr.py 调用原 ltpo_vl_dmlr.py 的路径。

2. 保留原 LTPO 的 latent candidate 采样方式：

   A_i = H + eps_i, eps_i ~ N(0, sigma^2 I)

   尽量直接复用原 ltpo_vl_dmlr.py 中已有的采样逻辑。

3. 增加 masked visual input 的构造接口。

   如果 codebase 已经有 image token / visual feature mask 逻辑，请复用。
   如果没有，请在 ltpo_vl_dpo.py 中实现一个最小 fallback：

   - 默认支持传入 mask_indices 或 mask_ratio；
   - 将被 mask 的视觉 tokens 替换为 0 或 visual feature mean；
   - full_image 分支保持不变；
   - masked_image 分支只用于 reward / preference 构造，不用于最终生成。

4. 对每个 latent candidate A_i，计算两个 score：

   r1_i = confidence_score(model, full_visual, prompt, A_i)
   r2_i = confidence_score(model, masked_visual, prompt, A_i)

   score 尽量复用原 LTPO reward 函数。
   注意：
   - 必须统一成“越大越好”。
   - 如果原 reward 是 cost / negative confidence 形式，请在这里取负号。
   - 不要重新写一套复杂 reward，除非原 reward 无法复用。

5. 在 main_vl_dpo.py 中新增参数：

   --tt_opt_method，默认 contrastive_dpo，可选 ltpo_reinforce / contrastive_dpo
   --dpo_num_candidates，默认 4
   --dpo_beta，默认 0.1 或 0.5
   --dpo_alpha，默认 1.0
   --lambda_mask，默认 1.0
   --dpo_margin，默认 0.0
   --dpo_topk_pairs，默认 None 或 0
   --use_soft_dpo，默认 True
   --mask_ratio，默认 0.3
   --mask_fill，默认 zero，可选 zero / mean

   要求：
   - main_vl_dpo.py 的默认路径运行 contrastive_dpo；
   - 如设置 --tt_opt_method ltpo_reinforce，则在新文件中仍可调用原 LTPO-style REINFORCE 逻辑，便于对照；
   - 不要改变原 main_vl_dmlr.py 的参数行为。

   关于 --dpo_num_candidates参数的重要说明：
   该参数表示每个 test-time optimization step 中采样多少个 latent candidates。在 contrastive_dpo 模式下，每一步必须采样多个 latent candidates，注意：
   - DPO update 需要 pairwise preference，因此 B 必须 >= 2；
   - 不要沿用原 LTPO 中单个 perturbation 的 REINFORCE 更新方式；
   - 原 LTPO 的单样本采样逻辑可以复用，但需要扩展为一次采样 B 个 candidates。

6. 定义 contrastive score：

   g_i = r1_i - lambda_mask * r2_i

   其中 r1_i 和 r2_i 只用于构造 preference，应该 stop-gradient。

7. 构造 preference pairs：

   - 若 use_soft_dpo=True：
     对所有 i != j 或 top/bottom candidates 构造 pair；
     soft target:
       p_ij = sigmoid(dpo_alpha * (g_i - g_j))

   - 若 use_soft_dpo=False：
     只保留：
       g_i > g_j + dpo_margin
     target p_ij = 1

   - 可过滤：
       |g_i - g_j| <= dpo_margin

   - 若 dpo_topk_pairs > 0：
     优先只用 top-k g_i 作为 preferred candidates、bottom-k g_i 作为 rejected candidates 构造 pair，减少计算量。

   - 若 pair 为空：
     不要报错；
     可跳过本步更新，或退化为选择 g_i 最大的 candidate 作为当前 H。

8. 实现 latent-policy DPO。

   当前 policy 是 Gaussian latent policy：

   pi_H(A) = N(A; H, sigma^2 I)

   reference policy 固定为初始 latent tokens：

   H_ref = H_init

   pi_ref(A) = N(A; H_ref, sigma^2 I)

   对 pair (i, j)，DPO logit 为：

   z_ij = beta * [
     log pi_H(A_i) - log pi_ref(A_i)
     - log pi_H(A_j) + log pi_ref(A_j)
   ]

   其中：

   log pi_H(A) = - ||A - H||^2 / (2 * sigma^2)

   log pi_ref(A) = - ||A - H_ref||^2 / (2 * sigma^2)

   soft DPO loss：

   L = - mean[
     p_ij * log_sigmoid(z_ij)
     + (1 - p_ij) * log_sigmoid(-z_ij)
   ]

   hard DPO loss：

   L = - mean[log_sigmoid(z_ij)]

9. 更新方式：

   - 推荐用 autograd 只对 H 求梯度；
   - model 参数必须 requires_grad=False；
   - A_i、g_i、r1_i、r2_i 用于构造 preference 时应 stop-gradient；
   - DPO loss 只通过 log pi_H(A_i) 中的 H 反传；
   - 每一步更新 H 后，可保留原 LTPO 的 norm clipping / latent clipping / sigma schedule，如果已有的话；
   - 保证 tensor device / dtype 与原实现一致。

10. 最终答案生成：

   - 只使用优化后的 H；
   - 只使用 full visual input；
   - masked visual input 只用于测试时优化阶段；
   - masked visual input 不参与最终 decode。

11. 兼容性与安全检查：

   请在修改后做基本静态检查：

   - 确认原 main_vl_dmlr.py 和 ltpo_vl_dmlr.py 未被修改；
   - 确认 main_vl_dpo.py 能正确解析新增参数；
   - 确认 main_vl_dpo.py 调用的是 ltpo_vl_dpo.py；
   - 确认 ltpo_vl_dpo.py 中原 LTPO-style 路径仍可运行；
   - 确认 contrastive_dpo 路径中只有 latent tokens H 被优化，model 权重不会更新；
   - 确认 pair 为空时不会崩溃；
   - 确认最终生成只使用 full visual input。

请直接完成代码修改，并在最后简要说明：

- 新复制了哪些文件；
- 修改了哪些新文件；
- 原始 main_vl_dmlr.py 和 ltpo_vl_dmlr.py 是否保持不变；
- 新增了哪些参数；
- contrastive_dpo 的运行入口是什么；
- 和原 LTPO REINFORCE 更新相比，核心差异在哪里。