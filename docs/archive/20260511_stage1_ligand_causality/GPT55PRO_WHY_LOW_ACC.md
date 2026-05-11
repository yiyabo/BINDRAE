# 咨询 GPT-5.5 Pro：为什么我们的 χ1 rotamer accuracy 这么低？

## 问题

我们的 ligand-conditioned χ1 rotamer 3-bin 分类（g-/t/g+）在验证集上只能达到 **0.53**，但业界 native backbone side-chain packing 的 χ1 accuracy 普遍在 **85-90%**（SCWRL4 86%, OPUS-Rota 88-90%, AttnPacker 90%+, AlphaFold2 89%）。

即使考虑到我们的任务更难（apo backbone → 预测 holo side-chain，而不是 native backbone repacking），3-bin 粗分类的准确率也不应该只有 53%。我们怀疑有根本性的问题。

## 当前实验数据

### M2（纯 geometry scorer，无 s_geo）
```
epoch 21: 0.445
epoch 22: 0.458
epoch 23: 0.463
epoch 24: 0.469
epoch 25: 0.489
epoch 26: 0.496
epoch 27: 0.490
epoch 28: 0.496
epoch 29: 0.510
epoch 30: 0.516
epoch 31: 0.529
epoch 32: 0.531
```

### M3（geometry + s_geo 32维投影）
```
epoch 21: 0.464
epoch 22: 0.474
epoch 23: 0.473
epoch 24: 0.482
epoch 25: 0.509
epoch 26: 0.529
```
M3 学习效率更高（6 epoch 达到 M2 12 epoch 的水平），还在涨。

### 之前的 s_geo-only scorer
- 天花板 ~0.508，无 ligand causality

## 当前 Geometry Scorer 架构

```python
class GeometryCandidateScorer:
    """
    输入特征 (47维，M3模式下79维):
    - aatype one-hot (21)
    - apo phi/psi sin/cos (4)
    - apo chi1 sin/cos (2)
    - candidate chi1 sin/cos (2)  # 固定 bin center: g-=-60°, t=180°, g+=60°
    - RBF distance features (16)  # 候选 sidechain atoms 到 ligand atoms 的距离
    - clash indicator (1)
    - min sidechain-ligand distance (1)
    - [M3] s_geo projection (32)  # IPA 输出的压缩表示
    
    网络: Linear(input, 64) → GELU → Dropout → Linear(64, 64) → GELU → Dropout → Linear(64, 1)
    输出: [B, N, 3] logits (每个残基3个候选的分数)
    """
```

### FK 候选几何生成
- 对每个残基，用 3 个 chi1 bin center 通过 Forward Kinematics 生成 idealized sidechain atom14 坐标
- 只看 chi1-dependent atoms（rigid group >= 4）
- 计算这些原子到 ligand atoms 的 pairwise 距离
- RBF 编码后 mean-pool 得到 16 维特征

### 训练设置
- Backbone 冻结（ESM-2 → Adapter → LigandConditioner → EdgeEmbed → FlashIPA 8层）
- 只训练 geometry scorer (~7K params) + ligand conditioner (~94K params)
- Loss: 3-class CE with w_res pocket weighting
- 数据: ~65K apo-holo triplets, ~2.8K validation
- 4×A100, batch_size=2 per GPU
- Resume from epoch 21 of a previous ligand conditioner unfreeze run

## 我的疑问

### 1. 为什么 3-bin 分类只有 53%？

业界 native backbone χ1 accuracy 85-90%。即使 apo→holo 更难，3-bin 分类（每 bin 120°宽）应该比精确角度预测（±20° tolerance）更容易。一个简单的 Dunbrack library baseline（只看 residue type + φ/ψ）应该就能到 70%+ 吧？

**可能的原因：**
- 我们的 3-bin 定义有问题？（g-: <-60°, t: -60°~60°, g+: >60°）
- 数据质量问题？（apo/holo 对齐、chi 角度计算）
- 训练集太小？（65K）
- 模型容量太小？（7K params 的 scorer）
- 全局 CE loss 被非 pocket 残基主导？
- 从 epoch 21 resume 导致 LR/optimizer 状态不干净？

### 2. 业界的 χ1 accuracy 是怎么算的？

SCWRL4 报告的 86% 是用 ±40° tolerance 算的，我们的 3-bin 分类（每 bin 120°）理论上应该更宽松。但我们的 metric 是 argmax(3-class logits) vs true bin，这和 ±40° tolerance 不完全等价。

**请帮我理清：**
- 我们的 3-bin accuracy 和业界的 χ1 accuracy 是否可比？
- 如果不可比，我们应该用什么 metric 来和 SCWRL4/AttnPacker 对比？

### 3. 合理的 baseline 应该是多少？

请帮我估算以下 baseline 在我们的 3-bin 分类任务上应该达到的准确率：
- Random: 33.3%
- Apo carryover（直接用 apo 的 chi1）: ?%
- Dunbrack library（residue type + φ/ψ）: ?%
- Native backbone repacking（SCWRL4 级别）: ?%

### 4. 架构建议

如果 53% 确实太低，你觉得瓶颈最可能在哪？
- Scorer 容量（7K params 太小？）
- 特征不够（缺少邻居残基信息、溶剂可及性等？）
- Loss 设计（全局 CE 不够？）
- 数据/标签问题？
- 还是 apo→holo 任务本身就比 native repacking 难很多，53% 其实合理？

### 5. 如果要冲 70%+，最关键的改动是什么？

考虑到我们有 22 天 deadline（ISCBAI26），什么改动性价比最高？

## 约束
- Backbone 已训好，不想从头重训
- 可以 unfreeze ligand conditioner + 最后几层 IPA
- 4×A100 40GB
- 65K 训练样本
