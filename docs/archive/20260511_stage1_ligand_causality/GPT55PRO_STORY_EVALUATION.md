# 请 GPT-5.5 Pro 评估：BINDRAE 的 Paper Story 和竞争格局

## 背景

我们在做一个叫 BINDRAE 的项目：给定 apo 蛋白结构 + 配体 3D pose → 预测 holo 侧链构象（induced-fit side-chain prediction）。两阶段架构：
- Stage-1：预测 ligand-causal χ1 rotamer posterior（哪些残基因为配体需要改变 rotamer）
- Stage-2：Torsion Flow Model，用 Stage-1 的 posterior 作为 soft guidance 生成侧链构象

## 我们搜到的最新竞品（2024-2025）

### 1. ApoDock / ApoPack (JCTC 2025)
- MPNN-based conditional side-chain packing
- 输入：protein backbone + **2D ligand information**（不是 3D pose）
- 三阶段：ApoPack(侧链预测) → 传统 docking(pose sampling) → ApoScore(重排序)
- 发表在 JCTC（中等偏上期刊）

### 2. PackDock / PackPocket (Nature 子刊级别, 2024-2025)
- Diffusion model 采样 binding pocket 侧链构象
- 可以 with or without ligand
- 结合 conformation selection + induced fit
- 实际应用中发现了纳摩尔级化合物
- 发表级别较高

### 3. DiffBindFR (Chemical Science 2024)
- SE(3) equivariant diffusion，full-atom flexible docking
- 同时预测 ligand pose + protein side-chain
- 显式考虑蛋白侧链构象

### 4. LigandMPNN (Nature Methods 2025)
- David Baker 组
- 蛋白序列设计 + 侧链 packing，考虑配体/核酸/金属
- 自回归预测 chi1→chi2→chi3→chi4
- 混合 circular normal distribution
- 发在 Nature Methods

### 5. ApolloDiff (OpenReview/会议)
- Apo→holo conditional diffusion
- 不需要知道配体（只用 apo + sequence）
- 74% 的情况下生成的构象比 apo 更接近 holo

### 6. SIGMADOCK (arXiv 2025)
- Fragment-based SE(3) Riemannian diffusion for docking
- 提到可以扩展到 flexible side-chain（作为 fragments）

### 7. Ligand-Transformer (Nature Communications 2025)
- 序列级别预测 conformational population shift
- 预测配体结合引起的构象群体变化

## 我们的差异化 / 潜在 Novelty

相比上述方法，BINDRAE 的独特之处：

1. **显式 ligand causality 验证**：我们不只是训练一个 conditioned model，而是通过 ligand-ablation 诊断严格证明预测确实是"因为配体"而不是"backbone prior"。据我所知，上述方法都没有做这种因果性验证。

2. **Geometry-bypass scorer**：用 FK 生成候选 rotamer 的实际原子坐标，直接计算与配体的几何兼容性（RBF 距离、clash）。这比 ApoPack 的纯 message passing 或 PackDock 的 diffusion 更加 physically interpretable。

3. **Likelihood-ratio guidance**：Stage-2 不消费 absolute posterior，而是消费 `log p(r|L) - log p(r|∅)`（有配体 vs 无配体的 log-prob 差）。这保证了：如果 Stage-1 不确定，Stage-2 不会被错误 prior 伤害。

4. **两阶段解耦**：Stage-1 专注于"哪些残基需要变、变到哪"（discrete posterior），Stage-2 专注于"怎么生成连续构象"（torsion flow）。比 end-to-end diffusion 更可解释、更可调试。

## 我想让你评估的问题

1. **竞争格局判断**：ApoDock(JCTC)、PackDock(高影响力)、LigandMPNN(Nature Methods) 已经占据了这个领域。我们还有空间吗？我们的 novelty 够不够？

2. **Story 方向建议**：以下哪个 story 最有竞争力？
   - A) "Ligand-causal posterior + likelihood-ratio guidance"（方法论 novelty）
   - B) "Geometry-bypass scorer 证明显式几何特征是 ligand causality 的必要条件"（empirical insight）
   - C) "两阶段 induced-fit prediction 框架，Stage-1 posterior 可解释、可调试"（系统设计）
   - D) 其他你觉得更好的角度？

3. **目标期刊/会议建议**：考虑到竞品已经发了 JCTC/Nature Methods/Chemical Science，我们应该瞄准什么级别？需要达到什么效果？

4. **关键实验建议**：为了让 paper 有说服力，除了 pocket RMSD 和 χ1 accuracy，还需要哪些实验？（比如 cross-docking、virtual screening enrichment、case study 等）

5. **风险评估**：如果我们的 geometry scorer 实验成功（val_chi1_rotamer_acc > 0.55 + ligand causality confirmed），你觉得这个工作的完成度和发表可能性如何？

## 约束
- 计算资源：4×A100 40GB，可以跑多轮实验
- 数据：~65K apo-holo triplets（PDB 来源）
- 时间：希望 2-3 个月内完成实验 + 写作
- 团队：主要是我一个人在做（AI 辅助开发）
