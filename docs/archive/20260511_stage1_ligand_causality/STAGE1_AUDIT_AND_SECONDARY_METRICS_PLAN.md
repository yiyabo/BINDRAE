# Stage-1 Secondary Metrics + Dual-Mask Audit 实现方案

> 状态：待实现  
> 目的：在不改变当前 Stage-1 主体架构的前提下，补足**评估闭环**，避免继续只靠 `pocket_chi1_acc` 一个指标做重大方法决策。

---

## 1. 为什么现在必须补这一层

当前 Stage-1 已经证明了两件事：

1. **只看 `val_total_loss` 会错过 pocket 目标上的最佳点**；
2. **`pocket_chi1_acc` 是比 total loss 更合适的主指标**。

但当前仍存在两个明显问题：

### 问题 A：主指标过单一

当前 `pocket_chi1_acc` 只有一个 hard-threshold hit-rate 语义：

- 只看 χ1；
- 只看 `|Δχ1| < 20°` 是否命中；
- 不看误差分布；
- 不看 χ2；
- 不看 ligand-facing contact 是否恢复对；
- 不看是否以更糟糕的局部几何为代价换来的改善。

### 问题 B：当前 pocket 定义可能太粗

当前 pocket mask 实质上来自：

- `w_res = sigmoid(d(Cα_apo, ligand))`
- 验证中使用 `w_res > 0.5`

这是一种**训练友好的近邻代理定义**，但不一定是最适合最终评价“ligand-relevant pocket prior”的定义。

因此，在继续推进 E5 之前，我们需要先补足：

1. **secondary metrics**：更完整地看 prior 强度；
2. **dual-mask audit**：判断当前 pocket mask 是否正在稀释、误导或低估真正的 pocket signal。

---

## 2. 实施范围与设计原则

### 2.1 本方案要实现什么

本方案包含两类内容：

#### A. 新增 secondary metrics
- `pocket_chi1_cMAE` / `pocket_chi1_median_abs_deg`
- `pocket_chi1+2_acc`
- `pocket_contact_recovery`
- `pocket_clash_pct`
- 更可靠的 `pocket_irmsd`（若保留）

#### B. dual-mask audit
使用至少两种不同语义的 pocket subset 进行并行报告：

1. **apo-distance mask**（当前主 mask）
2. **holo-contact / holo-distance mask**（更贴近真实结合态）
3. 可选：**ligand-facing subset**（更严格的 contact-relevant 残基）

### 2.2 设计原则

1. **不改训练语义，只先补评估**
2. **尽量复用当前 batch 字段，不引入新的数据依赖**
3. **先支持最终评估和验证日志，再考虑训练期全量启用**
4. **slow metrics 与 fast metrics 分层管理**

---

## 3. 当前代码与可用信息盘点

### 3.1 当前 Stage1Batch 已具备的字段

当前 `Stage1Batch` 已包含：

- `Ca_apo`, `Ca_holo`
- `N_apo/C_apo`, `N_holo/C_holo`
- `lig_points`
- `w_res`
- `chi_holo`, `chi_mask`
- `atom14_holo`, `atom14_holo_mask`（可选）
- `node_mask`

这意味着：

### 已可直接构造

#### 当前主 mask（已有）
- `apo_distance_mask = (w_res > 0.5)`

#### Holo-based mask（可直接算）
- 用 `Ca_holo` 与 `lig_points` 重新跑一遍当前 `compute_pocket_weights()`
- 得到 `w_res_holo`
- 再阈值化：`holo_distance_mask = (w_res_holo > 0.5)`

#### 更严格的 ligand-facing 子集（可新增函数）
- 直接基于 `atom14_holo` 与 `lig_points` 计算“任一侧链重原子到 ligand 最近距离 < d_contact”的残基
- 这能更接近真正的 binding-relevant sidechain subset

因此：

> **不需要修改数据格式，也不需要额外预处理，就能先把 dual-mask audit 做起来。**

---

## 4. Secondary metrics 设计

---

### 4.1 `pocket_chi1_cMAE`

#### 目的
弥补 hard-threshold hit-rate 的不足，提供连续误差视角。

#### 定义

在指定 mask 上：

- 取 χ1 有定义的残基；
- 计算 wrap 后角度差；
- 取绝对值并转成 degree；
- 汇报：
  - mean absolute error
  - median absolute error

#### 推荐报告字段
- `pocket_chi1_mae_deg`
- `pocket_chi1_median_deg`

---

### 4.2 `pocket_chi1+2_acc`

#### 目的
避免只看 χ1 导致对真正关键侧链恢复程度判断不完整。

#### 定义

在指定 mask 上，仅对：

- χ1 和 χ2 都定义的残基

若同时满足：

- `|Δχ1| < 20°`
- `|Δχ2| < 20°`

则记为命中。

#### 推荐报告字段
- `pocket_chi12_acc`

---

### 4.3 `pocket_contact_recovery`

#### 目的
把评估从“角度是否对”推进到“与 ligand 的局部相互作用是否恢复对”。

#### 第一版建议（简单、可实现）

对每个残基 i：

- 用预测结构（优先 `atom14_pos`）计算该残基任一有效原子到 ligand 的最小距离 `d_pred(i)`；
- 用真值 holo（优先 `atom14_holo`）计算 `d_true(i)`；
- 定义 contact：
  - `contact_pred(i) = [d_pred(i) < d_contact]`
  - `contact_true(i) = [d_true(i) < d_contact]`

其中 `d_contact` 推荐先用：

- **4.5 Å**

然后在残基层面计算：

- precision
- recall
- F1
- IoU / Jaccard

#### 推荐报告字段
- `pocket_contact_precision`
- `pocket_contact_recall`
- `pocket_contact_f1`
- `pocket_contact_iou`

> 推荐优先保留 `F1` 或 `IoU` 作为摘要指标。

---

### 4.4 `pocket_clash_pct`

#### 目的
作为 guardrail，防止 pocket χ 指标提升是以更差的局部原子几何为代价换来的。

#### 建议

当前 trainer 里的 `clash_pct` 计算是默认关闭的，而且当前默认日志值不可靠。  
建议：

- 保留全结构 clash 作为已有指标；
- 新增一个只在 mask 子集上的局部 clash 报告：
  - 只统计 pocket residues 的有效原子与其他 pocket atoms 的 clash

#### 推荐报告字段
- `pocket_clash_pct`

---

### 4.5 更可靠的 `pocket_irmsd`

#### 当前问题

当前 `validate()` 中：

- `compute_slow_metrics = False`
- 因而日志中的 `pocket_irmsd = 0.0` 不是可信评估结果

#### 建议

将其改为：

- fast 验证默认不算
- final eval / explicit `--compute_slow_metrics` 时启用

同时修正 aggregation 逻辑，确保：

- 是按样本平均，不是只靠 batch 数兜底

#### 推荐报告字段
- `pocket_irmsd_ca`
- 若后续扩展 atom14 版本，可再加：`pocket_irmsd_atom14`

---

## 5. Dual-mask audit 设计

### 5.1 为什么需要 dual-mask

当前唯一 pocket mask：

- `apo_distance_mask = (w_res > 0.5)`

这在训练语义上合理，但在评估上可能不够“binding-relevant”。  
因此 dual-mask audit 的核心不是替换当前训练定义，而是回答：

> **当前主指标到底是在低估模型、误导模型，还是恰好足够？**

---

### 5.2 建议同时报告的 3 个 subset

#### Mask A：`apo_distance_mask`（当前主 mask）

定义：
- 直接使用 `w_res > 0.5`

作用：
- 保持与当前历史结果的可比性；
- 作为当前主线标准不能取消。

#### Mask B：`holo_distance_mask`

定义：
- 用 `Ca_holo` 与 `lig_points` 重算 `compute_pocket_weights()`
- 再阈值化：`w_res_holo > 0.5`

作用：
- 检查训练/评估分布偏移的影响；
- 观察 apo-based mask 是否系统性偏宽或偏窄。

#### Mask C：`ligand_facing_mask`（推荐新增）

定义（第一版建议）：
- 对每个残基，使用真值 holo 的 sidechain atom14（排除 backbone N/CA/C/O 可选）
- 若该残基任一有效侧链原子到 ligand 最近距离 < `d_contact`（建议 4.5 Å）
- 则记为 ligand-facing / true contact-relevant residue

作用：
- 提供最贴近“真正 pocket interaction 子集”的评估；
- 帮助判断当前 `apo_distance_mask` 是否把过多无关残基混进主指标。

---

### 5.3 审计时应同时比较哪些量

对每个 subset（A/B/C）都报告：

- `chi1_acc`
- `chi1_mae_deg`
- `chi12_acc`
- `contact_recovery`
- `clash_pct`
- 可选：`pocket_irmsd`

然后重点看以下差异：

#### 情况 1：Mask C 明显高于 Mask A
说明：
- 当前 apo-distance pocket mask 太宽；
- 当前主指标可能被无关残基稀释；
- Stage-1 实际比现在看起来更接近“有用 prior”。

#### 情况 2：Mask C 也很低
说明：
- 问题不是 mask 噪声；
- Stage-1 的 pocket-relevant sidechain prior 确实还弱；
- 更支持继续做 E5 / 目标对齐实验。

#### 情况 3：Mask B 与 Mask A 差异大
说明：
- apo-based pocket definition 与真实 holo pocket 偏差显著；
- 后续应认真评估是否要引入更细的 contact-aware weighting 或联合定义。

---

## 6. 推荐实现位置

### 6.1 `utils/metrics.py`

推荐新增函数：

1. `compute_chi_angle_errors(...)`
   - 返回 wrap 后绝对误差数组（度）

2. `compute_chi_mae(...)`
   - mean / median 版本

3. `compute_chi12_accuracy(...)`
   - χ1+χ2 joint hit

4. `compute_residue_contact_mask(...)`
   - 输入 residue atom coords + ligand coords
   - 返回 residue-level contact bool

5. `compute_contact_recovery(...)`
   - 计算 precision / recall / f1 / iou

6. `compute_pocket_clash_percentage(...)`
   - 仅在 mask 限制下统计 clash

### 6.2 `src/stage1/training/trainer.py`

推荐改动：

1. 在 `validate()` 中新增 fast metrics 分支
2. 把 `compute_slow_metrics` 改成可配置项，而不是硬编码 `False`
3. 新增 dual-mask audit 的 subset 计算与日志输出
4. 修正 `pocket_irmsd` / `clash_pct` aggregation

### 6.3 `src/stage1/datasets/dataset_stage1.py`

当前**不建议**先改 batch 结构。

原因：
- 现有字段已足够支撑 audit；
- 先用现有字段做零成本评估，价值最高；
- 后续若 audit 显示 contact-aware mask 非常关键，再考虑是否缓存更多字段。

---

## 7. 运行策略建议

### 7.1 Phase A：先做 evaluation-only audit

推荐先不动训练，只对：

- E2 best checkpoint
- E3 latest / best checkpoint

跑一次完整 secondary metrics + dual-mask audit。

目的：

- 用最小成本判断：
  - 当前主指标是否被 mask 稀释；
  - E2 与 E3 的真实差异在哪。

### 7.2 Phase B：再决定是否进入 E5

如果 audit 结果显示：

- Mask C 上也明显不强；
- 而且 E3 没赢 E2；

则 E5 的优先级进一步坐实。

如果 audit 结果显示：

- Mask C 比 Mask A 高很多；

则说明当前主问题有一部分来自评估定义，E5 仍可做，但文档叙事和封板标准要更谨慎。

---

## 8. 日志与交付建议

建议最终输出为两个层级：

### 8.1 验证期常规日志（轻量）
- `chi1_acc`
- `pocket_chi1_acc`
- `pocket_chi1_mae_deg`
- `pocket_chi12_acc`

### 8.2 审计报告（重）
- E2 vs E3
- Mask A vs B vs C
- 全量 contact / clash / iRMSD / χ error distribution

建议保存为：

- `logs/stage1/<run>/audit_metrics.json`
- `logs/stage1/<run>/audit_summary.md`

---

## 9. 结论

当前最该做的不是继续盲调 loss，也不是先大改模型，而是：

> **把 Stage-1 的评估体系从“单指标驱动”升级为“主指标 + 次指标 + subset 审计”的闭环。**

只有这样，我们才能回答下面这些真正关键的问题：

1. 当前 `0.3657` 到底是真的弱，还是被粗口袋定义低估？
2. E3 为什么没赢 E2？
3. E5 如果提升了，提升的是“真实 pocket prior”，还是只是某个单指标？

这份方案的价值在于：

- **不依赖改架构**
- **不依赖新数据格式**
- **可以立刻落地**
- **能直接降低后续实验解释风险**
