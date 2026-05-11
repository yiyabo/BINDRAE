# Stage-1 当前困境总结（供 GPT-5.4-pro 直接审阅）

> 用途：这是一份**面向当前阶段的增量总结**。它不重复整个方法文档，而是把最近一轮 E5 / E6 试验链路、当前困境、以及我们最想让 reviewer 判断的问题压缩到一页内。

---

## 0. 一句话摘要

当前 Stage-1 的真实状态是：

> **objective / supervision 路线已经被推进到相当靠前的位置，并且方向上是有效的；但截至 E6b，我们仍然只能“接近 E2”，而没有稳定打穿 E2 平台。**

因此，当前最关键的问题已经不再是“再调一个 loss 系数会不会更好”，而是：

> **当前瓶颈到底仍然属于 supervision / curriculum，还是已经开始转向架构层面的 ligand-conditioned information routing / sidechain specialization 问题。**

---

## 1. 当前请 reviewer 重点理解的事实

### 1.1 当前主目标

当前主目标仍然是：

- **先把 Stage-1 封板**；
- Stage-1 当前定位为：
  - **ligand-conditioned pocket structural prior**；
- 当前不推进 Stage-2 主实验。

### 1.2 当前核心指标

当前主指标仍是：

- `pocket_chi1_acc`

含义：

- 在 pocket 残基（当前主口径为 apo-distance pocket）里；
- 只看 χ1 有定义的残基；
- 若 `|Δχ1| < 20°` 记为命中；
- 最后求命中比例。

### 1.3 当前验证 / 审计口径

我们现在已经完成：

- secondary metrics
- dual-mask audit
- checkpoint-level audit script

因此当前判断不再只依赖单一旧日志，而是基于更新后的审计口径。

---

## 2. 当前最重要的实验结果

### 2.1 E2（当前锚点）

- 历史 best `pocket_chi1_acc ≈ 0.3657`
- 重新审计后，best checkpoint 的 `pocket_chi1_acc ≈ 0.3683`

当前我们把它视为：

> **Stage-1 objective / supervision 路线需要打穿的实际平台锚点。**

### 2.2 E3（全局降低 FAPE 权重）

- `w_fape: 1.0 -> 0.5`
- 没有超过 E2
- 结论：

> **继续全局降低 FAPE 权重不是主要方向。**

### 2.3 E5a（直接加入 pocket χ1 auxiliary loss，`lambda_pchi1=0.2`）

- best `pocket_chi1_acc ≈ 0.3604`
- 有正向信号，但幅度有限

结论：

> **说明“显式对准 pocket χ1 目标”是方向正确的，但当前实现偏温和。**

### 2.4 E5b（直接把 `lambda_pchi1` 加到 0.4）

- 早期表现明显差于 E5a
- 没有形成有效改进

结论：

> **不是简单把辅助项加得更重就能解决问题；更强并不等于更好。**

### 2.5 E6a（`ligand_facing` hard-mask + delayed/ramp）

- best `pocket_chi1_acc ≈ 0.3619`
- 早期有希望，但后续没有站住，平台回落
- 已中止

结论：

> **说明“supervision localization + curriculum”方向比 E5 更对，但原始 E6a 时序仍不够理想。**

### 2.6 E6b（shared-prefix branch；slow-ramp 版本）

设置：

- 同样 `ligand_facing` hard-mask
- 同样 `lambda_pchi1 = 0.2`
- 只把 `pchi1_ramp_steps` 从 `4000 -> 8000`

结果：

- best `pocket_chi1_acc ≈ 0.3658`
- 这是当前 objective / supervision 路线中最好的新版本
- 但它仍然：
  - **没有稳定超过 E2 的 ~0.3683**
  - 更像是**接近 E2 平台**，而不是实质性打穿平台

---

## 3. 当前最真实的困境是什么

### 3.1 已经可以确认“方向上有效”

当前 evidence 已足够说明：

- 简单改 `w_fape` 不够；
- 直接加重 `pchi1` 会过强；
- 更精准的 `ligand_facing` hard-mask + 更合理的时序控制是有价值的；
- E6b 说明 **curriculum 比单纯加大 λ 更关键**。

### 3.2 但仍然没有完成“平台突破”

即使是当前最好版本 E6b：

- 也只是 best 到了 `~0.3658`
- 仍然没有稳定越过 E2 的 `~0.3683`

因此当前困境不是：

- “完全没信号”
- “训练不稳定”
- “loss 改了也没有任何响应”

而是：

> **我们已经能让模型朝正确方向移动，但这个移动幅度不足以可靠打穿现有平台。**

### 3.3 我们当前最怀疑的瓶颈

当前我们最怀疑的，不再是单纯的 `λ` 调得不对，而是：

1. **supervision support set 仍然不够精确 / 不够高效**；
2. **auxiliary signal 的 curriculum 虽然改善了，但仍不足以改写最终平台**；
3. **模型本身对 ligand-conditioned sidechain decision 的信息路由能力可能不够强**。

也就是说，当前正在逼近这样一个判断：

> **瓶颈可能开始从 objective 设计，转向 architecture / representation / routing。**

---

## 4. 关于“继续多跑 10–20 轮会不会突然更好”的判断

这是当前我们一个非常实际的问题。

当前主观判断是：

> **有可能再变好一点，但不太可能“好很多”。**

理由：

- E6b 的最好点出现在相对较早的位置（best 在 epoch 5）；
- 后续更多是在略低的平台附近波动；
- 这更像“已经接近当前配置上限”，而不是“还处于持续抬升的平台建立期”；
- 因此继续多跑 10–20 轮，可能再得到一些小收益，但**不太像会把结果显著改写到稳定超过 E2，更不太像能推到我们之前讨论的 ~0.373+ 级别。**

换句话说：

> **继续训练的边际收益大概率有限，不应把主要希望建立在“尾盘惊喜”上。**

这不是统计显著性结论，而是当前基于曲线形态的研发判断。

---

## 5. 当前我们最希望 reviewer 判断的问题

请 reviewer 重点回答以下问题：

### Q1. 当前是不是已经基本耗尽了 objective / supervision 路线？

也就是：

- 在已经做过 E3 / E5a / E5b / E6a / E6b 的前提下；
- 当前 best 仍只是接近 E2 而未稳定超过；

这是否已经足以说明：

> **继续微调 loss / mask / schedule 的边际价值很低。**

### Q2. 如果是，该优先怀疑哪类架构问题？

例如：

- ligand → residue 的条件信息路由不足？
- 侧链局部定向能力不足？
- backbone/global prior 与 sidechain refinement 共用主干导致冲突？

### Q3. 如果 reviewer 认为还没到动架构的时候，那么 objective 路线最后还值得试什么？

请给：

- **最后 1–2 个最值得试的动作**；
- 而不是泛泛地说“再调调参数看看”。

### Q4. 如果 reviewer 认为现在该动架构，那么最小的下一步架构改动应该是什么？

我们当前不想推翻整个系统，而是想知道：

> **最小、最可解释、最值得先做的 architecture delta 是什么。**

---

## 6. 当前内部倾向（供 reviewer 参考，但不要被它绑住）

我们当前内部倾向是：

- E6b 是 objective / supervision 路线里最好的版本；
- 但 E6b 仍未稳定超越 E2；
- 因此：

> **当前已经很接近“该开始讨论架构改动”的边界。**

但我们还希望外部 reviewer 帮我们判断：

- 这是否已经足够构成 architecture escalation 的证据；
- 以及如果要动架构，最应该动哪一层。

---

## 7. 我们希望 reviewer 输出时尽量明确的结论

请尽量避免模糊表述，例如：

- “方向是好的，可以再试试”
- “可能还需要更多实验”

我们更希望得到的是：

1. **继续 objective 路线** / **转向 architecture** 的明确判断；
2. 若继续 objective，最后只保留哪 1–2 个动作；
3. 若转向 architecture，优先做哪个最小改动；
4. 如何定义 Stage-1 当前这一阶段的“封板阈值”。
