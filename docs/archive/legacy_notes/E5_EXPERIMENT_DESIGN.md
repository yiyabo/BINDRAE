# BINDRAE E5 实验设计文档（Pocket χ 定向增权）

> 状态：拟执行  
> 目的：在**不改架构**、**单变量**、**可解释**的前提下，直接检验当前 Stage-1 的核心工作假说：
>
> **当前瓶颈不是模型不会学，而是训练目标没有足够明确地对准 pocket prior，导致 pocket sidechain signal 被更全局、更稠密的几何项稀释。**

---

## 1. 背景与实验动机

当前事实源：`docs/METHOD_REVIEW_BRIEF.md`

截至当前，我们已经有如下结论：

1. **E2 已经证明了 selection misalignment**
   - 只用 `val_total_loss` 选模会掩盖 pocket 目标上的改善；
   - 切换到 `pocket_chi1_acc` 后，best checkpoint 发生了变化；
   - 当前 E2 best：`pocket_chi1_acc = 0.3657`。

2. **E3（`w_fape: 1.0 -> 0.5`）截至当前仍未超过 E2**
   - 这说明“继续全局削弱 FAPE”至少**不是已被证明有效**的方向；
   - 同时也说明，如果确实存在 objective misalignment，那么问题不一定能靠“继续全局砍 FAPE”来解决。

3. **当前代码中的 loss 结构确实存在 pocket-vs-global 不对称**
   - `chi` supervision 使用 `w_res_warmed`，即经过 pocket warmup 的 residue weight；
   - `fape` 则主要按 `node_mask` 覆盖有效残基，等价于更接近**全局几何项**；
   - 因此，当前 Stage-1 更像在训练一个“带 pocket 偏置的局部几何恢复器”，而不是一个“显式优化 pocket χ prior 的模型”。

基于以上背景，E5 的核心目的不是继续探测“FAPE 要不要再低一些”，而是更直接地回答：

> **如果我们显式增强 pocket χ1 目标，Stage-1 的主指标会不会上升？**

---

## 2. E5 的核心设计原则

E5 必须满足以下四条原则：

1. **单变量**：相对当前 baseline（建议以 E2 为主对照）只改一个关键变量；
2. **不改架构**：不碰 trunk / IPA / ligand encoder / head 结构；
3. **可解释**：若结果变好或变差，都能明确归因于“pocket χ 目标是否被更直接强调”；
4. **与当前主指标一致**：实验成败以 `pocket_chi1_acc` 为首要判断标准。

---

## 3. 推荐 baseline 与对照关系

### 3.1 主 baseline

建议将 **E2** 作为 E5 的主 baseline：

- 原因：
  - E2 是当前最佳已验证结果；
  - E2 已经修正了 selection misalignment；
  - 其 best `pocket_chi1_acc = 0.3657` 是当前最清晰的对照基线。

### 3.2 为什么不直接沿 E3 继续叠加

在 E3 还未明确优于 E2 之前，不建议把 E5 叠加在 E3 上。原因：

- 这样会同时混入两个效应：
  1. 全局 FAPE 降权
  2. pocket χ 增权
- 一旦结果变动，归因会被污染；
- E5 作为“目标对齐验证实验”，最需要的是**干净**。

因此建议：

> **E5 以 E2 配置为底座，只新增一个显式 pocket χ1 定向项。**

---

## 4. E5 的推荐实现方式

### 4.1 不推荐的粗实现

不建议直接：

- 把 `w_chi` 从 `1.0` 提到 `1.5` / `2.0`

原因：

- 这会同时放大全部 χ1–χ4、全部残基的 torsion 监督；
- 测试到的是“整体 torsion 更重要了”，而不是“pocket χ1 是否被更明确优化”。

### 4.2 推荐的精细实现

建议新增一个**显式 auxiliary term**：

\[
L_{total} = w_{fape}L_{fape} + w_{chi}L_{chi} + w_{clash}L_{clash} + \lambda_{pchi1} L_{pocket\_chi1}
\]

其中：

- `L_pocket_chi1` 只作用在：
  - χ1 有定义的残基；
  - `w_res_warmed > 0` 的 pocket-weighted 区域；
  - 最好使用 **soft pocket weighting** 而不是硬阈值；
- 该项只针对 **χ1**，不扩展到 χ2–χ4；
- 目的是把训练目标更直接地拉向当前主指标。

### 4.3 推荐的 `L_pocket_chi1` 形式

建议优先使用与现有 `torsion_sincos_loss` 同风格的连续项，而不是单独引入分类损失。

可行形式：

\[
L_{pocket\_chi1} = \frac{\sum_i w_i^{pocket} m_i^{chi1} \, \ell(\hat{\chi}_{1,i}, \chi_{1,i})}{\sum_i w_i^{pocket} m_i^{chi1} + \epsilon}
\]

其中：

- `w_i^{pocket}`：建议直接复用当前 `w_res_warmed`
- `m_i^{chi1}`：χ1 定义掩码
- `ℓ`：与当前 χ sin/cos regression 相同的 wrap-aware 连续 loss

这样做的优点：

1. 与现有 loss family 一致；
2. 不引入新的目标分布语义；
3. 直接测“把 pocket χ1 拉得更紧”是否有增益。

---

## 5. 推荐超参数策略

### 5.1 第一版（最保守）

建议第一轮只做一个保守值：

- `λ_pchi1 = 0.2`

理由：

- 当前已有 `L_chi` 在全 χ 上生效；
- 新项只是一个定向强化，不应一上来就主导总 loss；
- 先用小权重看主指标是否有方向性改善。

### 5.2 第二版（若第一版有方向性提升）

若第一版在以下指标上出现正向趋势：

- `pocket_chi1_acc`
- `pocket_chi1_cMAE`
- `pocket_contact_recovery`

且 guardrails 未明显恶化，则可再做：

- `λ_pchi1 = 0.4`

### 5.3 不建议第一轮就尝试的设置

- `λ_pchi1 >= 0.8`
- 同时叠加 FAPE warmup
- 同时修改 pocket 定义 / metric / scheduler

这些都会降低实验可解释性。

---

## 6. 训练配置建议

### 6.1 除新增项外，其他保持与 E2 一致

建议保持以下项不变：

- `selection_metric = pocket_chi1_acc`
- `w_fape = 1.0`
- `w_chi = 1.0`
- `w_clash = 0.1`
- scheduler / warmup / batch / model_size / patience 全部不变

### 6.2 为什么 selection metric 仍保留 `pocket_chi1_acc`

因为 E5 的首要问题就是：

> **显式 pocket χ1 目标能否把主目标拉起来。**

如果此时又同时切选模逻辑，会丢失可解释性。

---

## 7. E5 的成功 / 失败判定标准

### 7.1 主成功标准

相对 E2：

- `best pocket_chi1_acc` **明确超过 0.3657**

### 7.2 次成功标准

以下指标中至少 2 项改善，且 guardrails 不明显变差：

- `pocket_chi1_cMAE` 下降
- `pocket_chi1+2_acc` 上升
- `pocket_contact_recovery` 上升
- `chi1_acc` 不下降或小幅上升

### 7.3 失败标准

以下任一情况可视为 E5 失败：

1. `best pocket_chi1_acc` 未超过 E2，且 secondary metrics 无改善；
2. `pocket_chi1_acc` 有小幅提升，但伴随：
   - `val_fape` 明显恶化；
   - `pocket_clash%` 明显恶化；
   - contact 指标无改善；
3. 曲线表现出明显的不稳定或 severe overfit。

---

## 8. 需要同步补上的评估项（E5 不应单独跑）

E5 不能只看一个 `pocket_chi1_acc`。  
因此建议在 E5 启动前，至少补上以下评估：

1. `pocket_chi1_cMAE` / median `|Δχ1|`
2. `pocket_chi1+2_acc`
3. `pocket_contact_recovery`
4. `pocket_clash%`
5. 更可靠的 `pocket_irmsd`（若保留）

这些将在另一份实现方案文档中展开。

---

## 9. 建议的实验命名与记录规范

建议命名：

- `E5a`: `lambda_pchi1 = 0.2`
- `E5b`: `lambda_pchi1 = 0.4`（仅在 E5a 有正向信号时）

每个 run 必须记录：

- `selection_metric`
- `w_fape, w_chi, w_clash`
- `lambda_pchi1`
- secondary metrics 全量 epoch 曲线
- best epoch 对应完整 metric snapshot

---

## 10. 结论

E5 的本质不是“再调一个 loss 权重”，而是：

> **用一个最小、最干净、最可解释的实验，直接验证当前最重要的工作假说：Stage-1 之所以还不够强，是因为训练目标没有足够明确地对准 pocket χ prior。**

如果 E5 有效，我们将得到：

- 一个更可信的优化方向；
- 一个更合理的 Stage-1 prior 叙事；
- 更坚实的 Stage-2 进入条件。

如果 E5 无效，也同样有价值：

- 它将帮助我们排除“只是目标没对准”这个解释；
- 从而把下一步注意力转向 pocket 定义、数据分桶、或结构建模本身。
