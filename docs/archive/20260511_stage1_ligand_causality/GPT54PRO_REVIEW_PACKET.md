# BINDRAE 发给 GPT-5.4-pro 的方法审阅包

> 用途：这是一份**给外部高强度方法 reviewer 使用的入口文档**。它不是完整方法文档，而是一个审阅导航页，告诉 reviewer：
> 1) 哪些文件必须读；
> 2) 哪个文件是当前事实源；
> 3) 你最希望他回答什么；
> 4) 当前项目的真实状态是什么。

---

## 0. 先读这个：当前审阅任务的本质

请不要把这次审阅理解成“BINDRAE 两阶段大方法是否成立”的泛泛讨论。  
当前真正需要审查的是：

> **Stage-1 这个 ligand-conditioned pocket prior 目前到底够不够强，当前实验瓶颈是什么，下一步最优先的优化动作是什么，以及 Stage-1 是否已经足以支撑 Stage-2。**

也就是说，当前核心不是：

- “Stage-2 bridge flow 数学形式对不对？”
- “最终论文蓝图是否宏大完整？”

而是：

- **Stage-1 当前定义是否合理？**
- **当前指标是否偏低？**
- **当前训练目标是否真的对准了 pocket prior？**
- **下一步最值得做的实验是什么？**

---

## 1. 阅读顺序（非常重要）

请按以下顺序阅读文档：

### 1）`docs/STAGE1_CURRENT_DILEMMA_SUMMARY_FOR_GPT54PRO.md`
这是**当前最新的增量困境总结**。  
如果你时间有限，请先读它。它集中描述了：

- E2 / E5 / E6 的最新结果；
- 当前 objective / supervision 路线到底推进到了哪里；
- 我们当前真正卡住的问题是什么；
- 为什么我们开始接近“该讨论架构”的边界。

### 2）`docs/METHOD_REVIEW_BRIEF.md`
这是**当前事实源（source of truth）**。  
如果其他文档与它冲突，请**以它为准**。

它包含：

- 当前项目真实状态
- Stage-1 当前定位
- 已做实验与最新结果
- 关键指标定义
- 当前最想让 reviewer 判断的问题

### 3）`docs/STAGE1_SUMMARY.md`
这是 **legacy 背景文档**，主要帮助理解早期 Stage-1 脉络。  
注意：它描述的是较早期的 **CASF holo-only Stage-1**，不是当前主线实现。

### 4）`docs/stage1_optimize.md`
这是**历史优化设想文档**。  
其中有很多调研与建议，但**不代表当前实验优先级**。

### 5）`docs/理论/Stage2理论与指导.md`
这是 **Stage-2 的理论蓝图与最终方法方向**。  
它的作用是帮助你理解：为什么我们如此在意 Stage-1 prior 的强度。

---

## 2. 当前真实状态（请务必基于这个状态审阅）

### 2.1 当前主线

- **主线：Stage-1 封板前优化**
- **非主线：Stage-2 理论已较完整，但当前不进入主实验推进**

### 2.2 当前对 Stage-1 的真实定义

我们当前不把 Stage-1 视为：

- 通用 holo generator
- 通用 apo→holo induced-fit solver
- 全局蛋白构象重建器

而是更明确地把它定义为：

> **ligand-conditioned, pocket-sensitive structural prior**

### 2.2.1 最新 objective / supervision 试验结论（重要）

在最近一轮实验里，我们依次做了：

- E5a：`lambda_pchi1 = 0.2`
- E5b：`lambda_pchi1 = 0.4`
- E6a：`ligand_facing` hard-mask + delayed/ramp
- E6b：从 E6 共享前缀分叉出的 **slow-ramp** 分支

当前结论是：

- E5a 有帮助但偏弱；
- E5b 更差；
- E6a 早期有希望但没站住；
- **E6b 是当前 objective / supervision 路线中最好的版本，但仍然只是接近 E2，并未稳定超越 E2。**

这意味着：

> 当前问题已经不太像“再调一个系数就会解决”，而更接近“是否应该开始转向架构层面”的边界判断。

更具体地说：

- 它应该对 binding pocket / local sidechain geometry 敏感；
- 它的目标是为 Stage-2 提供：
  - endpoint prior
  - 后半段路径 prior
  - FK / FAPE / clash 等 geometry toolkit

### 2.3 当前最关键的问题

当前最关键的问题不是：

- 训练是否稳定（目前稳定）
- 是否能跑通（目前能跑通）

而是：

> **Stage-1 目前学到的 pocket prior 还不够强。**

当前最关键指标是：

- `pocket_chi1_acc`

当前最好结果：

- **E2 best `pocket_chi1_acc = 0.3657`**

当前 E3（`w_fape: 1.0 -> 0.5`）阶段性状态：

- **截至当前仍未超过 E2**
- 当前 E3 最好约为 **0.3579**（阶段性结果）

因此我们当前的判断是：

> Stage-1 已经有信号、有可用性，但还未达到“足够强、足以放心支撑 Stage-2”的程度。

---

## 3. 审阅时请特别注意的边界条件

### 3.1 不要误用 legacy Stage-1 结果

早期/legacy Stage-1（如 CASF holo-only）曾经达到更高的 χ1 数字，但它的 setting 更容易：

- holo backbone 输入
- holo reconstruction 任务
- 不是真正的 apo-conditioned pocket prior

因此：

> legacy Stage-1 的高分不能直接说明当前 Stage-1 已经足够强。

### 3.2 不要把当前项目误解为“立刻推进 Stage-2”

当前真正的问题不是 Stage-2 理论是否写得漂亮，而是：

> **Stage-1 是否已经强到可以作为 Stage-2 的可靠 prior。**

### 3.3 不要把 “total val loss 更低” 误判为“prior 更强”

我们已经通过 E2 验证过：

- 使用 `val_total_loss` 选模会掩盖 pocket 目标的改善；
- `pocket_chi1_acc` 更适合当前 Stage-1 的真实用途。

---

## 4. 我们最希望你回答的问题

请围绕以下问题进行高强度审查：

### Q1. 两阶段方法分解是否合理？

也就是：

- Stage-1 = pocket prior
- Stage-2 = continuous apo→holo bridge

这样的任务切分是否合理？

### Q2. Stage-1 当前定位是否正确？

也就是：

- 把 Stage-1 定义为 `ligand-conditioned pocket structural prior`
- 而不是 generic holo decoder

这样的收缩是否正确？

### Q3. `pocket_chi1_acc` 是否适合作为当前主指标？

如果是：
- 为什么？

如果不够：
- 还应该补哪些 secondary metrics？

### Q4. 当前 best `pocket_chi1_acc ≈ 0.3657` 是什么水平？

请你结合文献与 setting 难度谨慎判断：

- 很低 / 偏低 / 可接受 / 不错 / 很强

注意区分：

- fixed-backbone sidechain packing
- predicted backbone repacking
- apo→holo / induced-fit / ligand-conditioned pocket prior

这些 setting 难度不同，请不要直接混比。

### Q5. 当前最可信的瓶颈是什么？

我们当前怀疑是：

- selection / objective misalignment
- global geometry terms 与 pocket χ 目标之间存在竞争

请判断：

- 这个诊断是否成立？
- 如果成立，最可疑机制是什么？
- 如果不成立，真正的瓶颈更可能是什么？

### Q6. 下一步实验优先级怎么排？

当前候选包括：

- E4：FAPE warmup
- E5：pocket χ 定向增权
- metric 体系补充
- pocket mask / pocket 定义审计
- 其他中等改动

请给出一个明确优先级，而不是模糊建议。

### Q7. 项目叙事是否应该调整？

你是否建议我们把方法叙事进一步明确成：

- Stage-1 = pocket prior module
- Stage-2 = apo→holo bridge / path model

如果要改，请给出更准确的摘要写法。

---

## 5. 你输出时希望采用的格式

请按以下结构回答：

### A. Executive verdict
- 5–10 条 bullet，直接给总判断

### B. What is fundamentally right
- 当前方法里真正正确、值得保留的部分

### C. What is fundamentally weak
- 当前最限制项目成功的核心弱点（按严重程度排序）

### D. Metric audit
- 评估 `pocket_chi1_acc` 是否合理
- 当前 `0.3657` 是什么水平
- 你建议的 Stage-1 封板阈值如何定义

### E. Bottleneck diagnosis
- 当前最可信的瓶颈是什么
- 哪些是次要问题
- 哪些是伪问题

### F. Optimization priority
- 请给一个明确的优先级表，例如：

| Priority | Action | Why | Expected upside | Risk | Do now? |
|----------|--------|-----|-----------------|------|---------|

### G. Narrative / documentation revision
- 你建议如何修改项目叙事与文档组织

### H. Final recommendation
- 用一句话说：下一步最该做什么

---

## 6. 额外要求

请在回答中注意：

1. 区分：
   - **高置信结论**
   - **合理推测但证据不足的结论**

2. 如果引用文献或 benchmark，请区分：
   - **直接可比**
   - **弱可比**

3. 如果你建议大改模型结构，请说明为什么当前证据已经足够支撑大改。否则请优先给出：
   - 单变量
   - 易解释
   - 快速证伪
   - 不大改架构
   的实验建议。

4. 请不要给泛泛表扬。我需要的是：
   - 方法审核
   - 问题诊断
   - 优先级设计
   - 叙事校正

---

## 7. 当前我自己的初步判断（供你挑战，而不是要求你同意）

当前我自己的倾向是：

- Stage-1 方向没错，但 prior 还不够强；
- `pocket_chi1_acc` 是主指标，但不是唯一指标；
- 目前最值得怀疑的是目标函数对齐问题；
- 下一步优先级大概率应是：
  1. 明确 pocket χ 定向增权
  2. 补 secondary metrics
  3. 再考虑 FAPE warmup
- 在 Stage-1 没达到更强 prior 之前，不应推进 Stage-2 主实验。

如果你不同意，请直接反驳，并给出替代方案。

---

## 8. 最后一句话

> 请把这次审阅当成一次“是否允许 Stage-1 进入下一阶段”的 gate review，而不是一次泛泛的方法讨论。
