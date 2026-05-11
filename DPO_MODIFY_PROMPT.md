请继续基于当前已经生成的 main_vl_dpo.py 和 ltpo_vl_dpo.py 修改代码，但仍然保持最小改动。原始 main_vl_dmlr.py 和 ltpo_vl_dmlr.py 仍然不要修改。

这次需要补两个功能：

1. 增加 “LTPO-style explicit DPO update”：
   当前 contrastive_dpo 的 H 更新似乎统一使用了 autograd backward。
   我希望新增一种不依赖 backward 的显式更新方式，保持和原 LTPO 类似的范式：
   - 显式计算 DPO 更新方向；
   - 在 torch.no_grad() 下直接更新 H；
   - 不通过 loss.backward() 更新 H。

2. 增加 “更新 loss 与 best selection 解耦”：
   DPO 更新仍然可以用 contrastive score g = r1 - lambda_mask * r2 构造 preference pairs；
   但是每一步更新后，选择 best latent candidate 用于下一步或最终生成时，需要支持两种选择标准：
   - 根据 g 最大选择；
   - 根据 r1 最大选择。

请按下面要求实现。

========================
一、新增参数
========================

请在 main_vl_dpo.py 中新增以下参数：

--dpo_update_mode，默认 explicit，可选 explicit / backward

含义：
- explicit：使用 LTPO-style 显式公式更新 H，不调用 loss.backward()
- backward：保留当前已有的 autograd DPO loss 更新方式，作为对照

--best_select_metric，默认 g，可选 g / r1

含义：
- g：选择 contrastive score 最大的 candidate 作为 best
- r1：选择 full visual confidence score 最大的 candidate 作为 best

要求：
- 默认使用 --dpo_update_mode explicit
- 默认使用 --best_select_metric g
- 不要破坏已有 contrastive_dpo 入口
- 不要修改原始 main_vl_dmlr.py 和 ltpo_vl_dmlr.py

========================
二、显式 DPO 更新公式
========================

当前 latent policy 是：

pi_H(A) = N(A; H, sigma^2 I)

reference policy 是：

pi_ref(A) = N(A; H_ref, sigma^2 I)

其中 H_ref 是初始 latent tokens，固定不更新。

对任意 pair (i, j)，定义：

z_ij = beta * [
  log pi_H(A_i) - log pi_ref(A_i)
  - log pi_H(A_j) + log pi_ref(A_j)
]

在相同 sigma 下，有：

log pi_H(A) = - ||A - H||^2 / (2 * sigma^2)

并且：

grad_H [log pi_H(A_i) - log pi_H(A_j)]
= (A_i - A_j) / sigma^2

因此 soft DPO 的显式更新方向为：

delta_H = mean_{(i,j)} [
  weight_ij * (p_ij - sigmoid(z_ij)) * beta * (A_i - A_j) / sigma^2
]

其中：

p_ij = sigmoid(dpo_alpha * (g_i - g_j))

如果 use_soft_dpo=False，则：

p_ij = 1

最终在 no_grad 下更新：

H = H + lr * delta_H

注意：
- 这里是 gradient descent on loss 等价得到的 ascent update direction；
- 不要再对这个 explicit update 调用 backward；
- A_i、A_j、g_i、r1_i、r2_i 都应该 detach 后用于构造 pair 和 delta_H；
- z_ij 可以用 detach 的 A_i/A_j、当前 H、H_ref 显式计算；
- H 是唯一被更新的变量；
- model 参数必须保持冻结。

========================
三、显式 z_ij 计算
========================

请在 ltpo_vl_dpo.py 中实现一个简洁函数，例如：

compute_gaussian_logprob(A, H, sigma)

返回：

- ((A - H) ** 2).sum(dim=latent_dims) / (2 * sigma ** 2)

注意：
- 实际代码中需要保留 batch / candidate 维度；
- latent_dims 应覆盖 latent token length 和 hidden dim；
- 返回每个 candidate 的 logprob 标量；
- 不要把不同 candidates 混在一起。

然后计算：

logp_i = log pi_H(A_i)
logp_ref_i = log pi_ref(A_i)

z_ij = beta * [
  logp_i - logp_ref_i - logp_j + logp_ref_j
]

如果当前代码中 H 的 shape 和 A_i 的 shape 有额外 batch 维度，请保证 broadcasting 正确。

========================
四、pair 构造和 explicit update
========================

请复用当前已有的 pair 构造逻辑，但确保支持 explicit update。

推荐实现方式：

1. 每个 optimization step 采样 B=dpo_num_candidates 个 candidates：
   A_all: shape roughly [B, ...latent_shape...]

2. 对每个 A_i 计算：
   r1_i
   r2_i
   g_i = r1_i - lambda_mask * r2_i

3. 构造 pairs：
   pair_i_indices
   pair_j_indices

4. 对每个 pair 取：
   A_i = A_all[pair_i_indices]
   A_j = A_all[pair_j_indices]
   g_i = g_all[pair_i_indices]
   g_j = g_all[pair_j_indices]

5. 计算：
   p_ij = sigmoid(dpo_alpha * (g_i - g_j))   # soft
   或 p_ij = ones_like(...)                  # hard

6. 计算：
   z_ij = beta * [
     logpi_H_i - logpi_ref_i
     - logpi_H_j + logpi_ref_j
   ]

7. 计算显式更新：

   coeff_ij = weight_ij * (p_ij - sigmoid(z_ij)) * beta / (sigma ** 2)

   delta_H = mean over pairs of:
     coeff_ij * (A_i - A_j)

   注意：
   - coeff_ij 需要 reshape / view 成可以 broadcast 到 A_i - A_j 的形状；
   - delta_H 的 shape 必须和 H 一致；
   - 如果 H 有 batch 维度，确保 delta_H 和 H 对齐。

8. no_grad 更新：

   with torch.no_grad():
       H.add_(lr * delta_H)

   如果原 LTPO 有 latent norm clipping / perturbation clipping / sigma schedule，请继续复用。

9. 如果 pair 为空：
   - 不要报错；
   - 可以跳过 explicit update；
   - 但仍然执行 best selection；
   - 打印或记录一个简短 warning 即可，不要大量输出。

========================
五、保留 backward 版本
========================

不要删除当前已有的 backward DPO loss 版本。

请通过参数控制：

if args.dpo_update_mode == "explicit":
    使用显式 delta_H 更新
elif args.dpo_update_mode == "backward":
    使用当前已有 loss.backward() 更新

这样我可以比较：
- LTPO REINFORCE
- DPO backward
- DPO explicit

========================
六、更新 loss 与 best selection 解耦
========================

请增加 best selection 逻辑，确保它和 DPO update loss 解耦。

DPO update 的 pair 构造仍然默认基于：

g_i = r1_i - lambda_mask * r2_i

但是每一步选择 best candidate 时，支持：

if best_select_metric == "g":
    best_idx = argmax(g_all)
elif best_select_metric == "r1":
    best_idx = argmax(r1_all)

这个 best_idx 可以用于：
- 更新后将 H 设置为或靠近 best candidate，如果当前代码已有类似 best-candidate selection；
- 或用于最终在多个 candidates 中选最优 latent；
- 请沿用当前代码中已有的 best selection 使用方式，不要重新设计复杂流程。

关键要求：
- best_select_metric 只影响 best candidate selection；
- 不影响 DPO preference pair 构造；
- 不影响 DPO loss / delta_H 的定义；
- r1_all 和 g_all 都必须 detach 后用于 selection。

请在代码中保持语义清晰：
- update_score / preference_score 使用 g；
- selection_score 根据 best_select_metric 选择 g 或 r1。

========================
七、简洁性要求
========================

请保持代码简洁：
- 不要大规模重构；
- 不要引入新文件；
- 不要写长篇注释；
- 只保留必要注释，例如 “explicit DPO update, LTPO-style”；
- 尽量复用现有函数；
- 不要改变原始 main_vl_dmlr.py 和 ltpo_vl_dmlr.py。

========================
八、修改后检查
========================

修改完成后，请检查：

1. main_vl_dpo.py 中新增了：
   --dpo_update_mode
   --best_select_metric

2. contrastive_dpo 默认使用：
   --dpo_update_mode explicit
   --best_select_metric g

3. explicit 模式下：
   - 不调用 loss.backward()
   - 不创建 optimizer 来更新 H
   - 只在 no_grad 下用 H.add_(lr * delta_H) 更新 H

4. backward 模式下：
   - 保留当前已有实现

5. best_select_metric == r1 时：
   - DPO 更新仍然用 g 构造 pairs
   - 只有 best candidate selection 改成 argmax(r1_all)

6. 原始 main_vl_dmlr.py 和 ltpo_vl_dmlr.py 未被修改。

最后请简要说明：
- 修改了哪些文件；
- 新增了哪些参数；
- explicit DPO update 的实现位置；
- best_select_metric 如何影响 selection；
- 如何运行 explicit DPO 和 backward DPO 两个版本。