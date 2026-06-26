# RAEv2 应用于蛋白质构象预测的可行性分析

**日期**: 2026-06-12
**状态**: 深入调研完成，提出适配方案
**决策**: 核心思想可借鉴，但需要重大架构适配

---

## 1. RAEv2 核心架构解析

### 1.1 RAE (Representation Autoencoder) 基础

**核心设计**:
- **Encoder**: 冻结的预训练视觉编码器 (DINOv2, SigLIP2, MAE 等)
- **Decoder**: 可训练的 ViT-MAE 风格解码器
- **Latent Space**: 编码器的表征输出（高维、语义丰富）
- **训练**:
  - Stage 1: 重建（训练 decoder）
  - Stage 2: 在 latent space 上训练扩散模型

**关键特性**:
- 冻结编码器提供语义丰富、结构一致的表征
- Latent space 是高维的（384-1024维），不是压缩的低维空间
- 重建质量高，生成质量好

### 1.2 RAEv2 改进

**核心创新**:
1. **广义表征**: 输出 = 最后 K 层编码器特征的加权和（K=1 恢复原始 RAE）
2. **10x 更快收敛**: 相比之前 baseline
3. **REPA (Representation Prediction)**: 在 RAE latent space 中的 x-prediction，用于自引导
4. **维度相关噪声调度**: 针对高维 latent space 的特殊设计

**关键洞察**:
- REPA head 是轻量 MLP，预测能力天然弱于完整模型，起到 AutoGuidance 的作用
- 通过 reformulating output head 为 x-prediction，REPA head 本身可用于 internal guidance
- 消除了对单独弱模型 (AutoGuidance) 或额外前向传播 (CFG) 的需求

---

## 2. 蛋白质特异性自编码器相关工作

### 2.1 ProteinAE (2025)
- **设计**: 扩散自编码器，将 E(3) 坐标映射到连续 latent space
- **架构**: 非等变 Diffusion Transformer + bottleneck
- **训练**: 单一 flow matching 目标，端到端训练
- **结果**: SOTA 重建质量，latent diffusion 绕过显式等变性需求

### 2.2 Latent Diffusion for Protein Structure (Fu et al., 2024)
- **设计**: 等变蛋白质自编码器 + 等变扩散模型
- **架构**: 编码器将蛋白质嵌入 latent space，扩散模型学习 latent 分布
- **结果**: 有效生成新颖、可设计的蛋白质骨架结构

### 2.3 Ophiuchus (2023)
- **设计**: SO(3)-等变粗粒化模型
- **架构**: 层次化卷积粗化 + latent diffusion
- **创新**: 首个直接生成全原子蛋白质结构的生成模型

### 2.4 Baker Lab VAE (2024)
- **设计**: VAE 用于蛋白质构象集合生成
- **应用**: K-Ras 蛋白质集合生成
- **方法**: 在 VAE latent space 中采样，用 RoseTTAFold 生成结构

### 2.5 LD-FPG (2025)
- **设计**: 全原子蛋白质构象的 latent diffusion
- **架构**: 图嵌入上的 latent diffusion
- **目标**: 捕获复杂动态转变中的构象景观

---

## 3. 核心问题：RAEv2 能否解决我们的问题？

### 3.1 我们遇到的问题

**根本限制**: Chi1 prediction loss 主导整个训练，任何辅助信号都被压制

**具体表现**:
- 8+ 种方法全部失败，同一模式
- 辅助 loss 信号增长，但实际 lift 保持 ~0.001
- prediction_changed_rate = 0.0（模型从不偏离 apo carryover）

**根本原因**: 模型直接从 apo 预测 holo，没有强制学习"配体导致了什么变化"这个因果关系

### 3.2 RAEv2 的设计目标 vs 我们的需求

| 维度 | RAEv2 | 我们的需求 |
|------|-------|-----------|
| **任务类型** | 无条件生成/重建 | 条件因果预测 |
| **输入** | 单一图像 | (apo_structure, ligand) |
| **输出** | 重建图像 | holo_structure |
| **学习目标** | 重建质量 | 学习配体→构象变化的因果关系 |
| **对称性** | 无特殊要求 | SE(3) 等变性必需 |

**关键差异**: RAEv2 学习"图像是什么"，我们需要学习"配体导致了什么变化"

### 3.3 直接应用 RAEv2 的问题

**如果直接套用 RAEv2**:
```
Encoder: holo_structure → latent_z (frozen)
Decoder: latent_z → holo_structure
```

**问题**:
1. **没有配体条件化**: 模型不知道配体信息
2. **没有因果关系**: 只是重建，不学习"为什么变化"
3. **推理时无法使用**: 没有 holo 作为输入，无法得到 latent_z

**结论**: RAEv2 as-is 无法解决我们的问题

---

## 4. 适配方案：Change-Prediction RAE

### 4.1 核心思想

**关键洞察**: 不预测 holo 本身，而是预测"apo→holo 的变化"在 latent space 中的表示

**架构设计**:
```
Frozen Encoder: protein_structure → latent_z (pretrained, frozen)
  - Encode both apo and holo to get z_apo and z_holo
  - delta_z = z_holo - z_apo (encoding the CHANGE)

Ligand Predictor: (apo_structure, ligand) → predicted_delta_z
  - This is what we train
  - Loss: ||predicted_delta_z - delta_z||^2

Decoder: (apo_structure, predicted_delta_z) → holo_structure
  - z_holo_pred = z_apo + predicted_delta_z
  - holo_pred = Decoder(z_holo_pred)
```

### 4.2 为什么这可能有效

**1. 避免 chi1 dominance**:
- Loss 在 latent space 上，不是直接在 chi angles 上
- Latent space 是高维语义空间，可能更容易学习变化模式

**2. 强制学习配体因果关系**:
- 最小化 loss 的唯一方式是从配体预测 delta_z
- 模型必须学习"配体→构象变化"的映射

**3. 利用预训练编码器**:
- 冻结编码器提供好的 latent space
- Latent space 可能已经编码了构象相关信息

### 4.3 为什么这可能仍然失败

**1. Latent space 可能不捕获变化**:
- 如果编码器不以线性/可加方式编码构象变化，delta_z 可能无意义
- 需要验证: z_holo - z_apo 是否有意义？

**2. SE(3) 等变性**:
- Latent space 和 decoder 必须尊重 SE(3) 对称性
- 需要仔细设计以保持等变性

**3. Decoder 质量**:
- 如果 decoder 不能从 latent 良好重建，整个方法失败
- 需要高质量的蛋白质结构 autoencoder

**4. 数据稀缺性**:
- apo/holo 配对数据有限
- 可能不足以学习复杂的 ligand→delta_z 映射

---

## 5. 具体实现方案

### 5.1 方案 A: 基于现有架构的最小改动

**使用现有组件**:
- **Encoder**: ESM-2 + IPA (冻结)
  - 输入: apo 或 holo 结构
  - 输出: per-residue latent representation [B, N, c_s]
- **Ligand Predictor**: 现有的 LigandConditioner + MLP
  - 输入: (apo_structure, ligand)
  - 输出: predicted_delta_z [B, N, c_s]
- **Decoder**: 现有的 TorsionHead + FK
  - 输入: z_holo_pred = z_apo + predicted_delta_z
  - 输出: holo structure

**训练目标**:
```python
# Encode apo and holo
z_apo = encoder(apo_structure)  # frozen
z_holo = encoder(holo_structure)  # frozen
delta_z_true = z_holo - z_apo

# Predict delta_z from ligand
delta_z_pred = ligand_predictor(apo_structure, ligand)

# Loss in latent space
loss_latent = ||delta_z_pred - delta_z_true||^2

# Decode to structure
z_holo_pred = z_apo + delta_z_pred
holo_pred = decoder(z_holo_pred)

# Reconstruction loss (optional, for regularization)
loss_recon = FAPE(holo_pred, holo_true) + chi_loss(holo_pred, holo_true)

# Total loss
loss = loss_latent + lambda_recon * loss_recon
```

**优点**:
- 最小改动，快速验证
- 利用现有高质量组件
- 可以直接测试是否有效

**缺点**:
- 现有 encoder 可能不是为 latent space 变化预测设计的
- 可能仍然遇到 chi1 dominance（如果 recon loss 太强）

### 5.2 方案 B: 专用蛋白质结构 Autoencoder

**训练专用 autoencoder**:
- **Encoder**: 类似 ProteinAE 的架构
  - 输入: 蛋白质结构 (backbone frames + chi angles)
  - 输出: latent_z [B, N, d_latent]
  - 训练: 重建目标
- **Decoder**: 从 latent_z 重建结构
  - 输出: backbone frames + chi angles
  - 使用 FK 得到全原子坐标

**然后训练 ligand predictor**:
- 输入: (apo_structure, ligand)
- 输出: predicted_delta_z
- Loss: ||predicted_delta_z - (encode(holo) - encode(apo))||^2

**优点**:
- Latent space 专门为蛋白质结构设计
- 可以更好地捕获构象变化
- 可以设计 SE(3) 等变性

**缺点**:
- 需要从头训练 autoencoder（大量工作）
- 不确定 latent space 是否适合变化预测
- 更长的开发周期

### 5.3 方案 C: 两阶段训练（推荐）

**Stage 1a: 训练蛋白质结构 autoencoder**
- 使用 apo/holo 配对数据
- 训练目标: 重建质量
- 验证: 重建误差 < 阈值

**Stage 1b: 冻结 encoder，训练 ligand predictor**
- 冻结 encoder
- 训练 ligand→delta_z predictor
- Loss: latent space 变化预测
- 验证: ligand lift metrics

**Stage 1c: 端到端微调（可选）**
- 解冻所有组件
- 小学习率微调
- 验证: 整体性能

**优点**:
- 分阶段验证，每阶段可独立评估
- 可以先验证 autoencoder 质量
- 可以诊断哪个阶段失败

**缺点**:
- 更复杂的训练流程
- 需要更多计算资源

---

## 6. 关键实验设计

### 6.1 验证 Latent Space 是否捕获变化

**实验 1: Delta-z 可视化**
- 计算 delta_z = encode(holo) - encode(apo) 对所有样本
- PCA/t-SNE 可视化
- 检查: 相似配体是否产生相似的 delta_z？

**实验 2: Delta-z 预测基线**
- 训练简单 MLP: (apo, ligand) → delta_z
- 检查: 能否比随机基线更好？
- 如果不行，说明 latent space 不适合这个任务

### 6.2 验证 Ligand Predictor 是否学习因果关系

**实验 3: Ligand Lift 测试**
- 使用现有诊断框架
- 比较: correct ligand vs no ligand vs scrambled ligand
- 期望: correct ligand 显著优于其他

**实验 4: Prediction Change Rate**
- 检查: prediction_changed_rate > 0?
- 如果仍然 = 0，说明方法失败

### 6.3 端到端验证

**实验 5: 完整流程测试**
- 从 apo + ligand 预测 holo
- 评估: chi1 accuracy, FAPE, contact lift
- 与现有方法比较

---

## 7. 风险评估与决策建议

### 7.1 成功概率评估

| 方案 | 成功概率 | 工作量 | 风险 |
|------|---------|--------|------|
| **方案 A (最小改动)** | 30% | 1-2 周 | 现有 encoder 可能不适合 |
| **方案 B (专用 AE)** | 40% | 4-6 周 | Latent space 质量不确定 |
| **方案 C (两阶段)** | 50% | 3-4 周 | 更复杂但可诊断 |

### 7.2 关键风险

**1. Latent space 不适合变化预测**
- 概率: 高
- 缓解: 先做实验 1-2 验证

**2. SE(3) 等变性难以保持**
- 概率: 中
- 缓解: 使用现有 SE(3)-equivariant 组件

**3. 数据不足以学习复杂映射**
- 概率: 中
- 缓解: 数据增强，预训练

**4. 仍然遇到 chi1 dominance**
- 概率: 中
- 缓解: 只用 latent loss，不用 chi loss

### 7.3 决策建议

**推荐路径**: 方案 C (两阶段训练)

**理由**:
1. **分阶段验证**: 可以早期发现失败
2. **利用现有工作**: 可以复用 ProteinAE 等现有代码
3. **科学价值**: 即使失败，也是有价值的 negative result

**具体步骤**:
1. **Week 1**: 实现方案 A (最小改动)，快速验证
   - 如果有效 → 继续优化
   - 如果无效 → 进入方案 C

2. **Week 2-3**: 方案 C Stage 1a (训练 autoencoder)
   - 验证重建质量
   - 验证 delta_z 是否有意义

3. **Week 4**: 方案 C Stage 1b (训练 ligand predictor)
   - 验证 ligand lift
   - 验证 prediction change rate

4. **Week 5**: 端到端测试 + 论文撰写

---

## 8. 与 Oracle 咨询的关键问题

**核心问题**:
给定 RAEv2 的核心思想（frozen encoder + trained decoder + high-dimensional latent space），以及我们的 change-prediction 适配方案，这是否能从根本上解决 chi1 dominance 问题？

**具体问题**:
1. Latent space change prediction 是否能避免 chi1 dominance？
2. 如何设计 latent space 使其对构象变化敏感？
3. SE(3) 等变性如何在 latent space 中保持？
4. 是否有更好的架构设计我们没考虑到？

---

## 9. 结论

**RAEv2 as-is 无法解决我们的问题**，因为它是为无条件生成/重建设计的。

**但核心思想可以借鉴**:
- Frozen encoder 提供好的 latent space
- High-dimensional latent space 可能比直接预测 chi angles 更好
- Change prediction in latent space 可能避免 chi1 dominance

**推荐方案**: Change-Prediction RAE (方案 C)
- 预测 latent space 中的变化，而不是直接预测结构
- 强制学习 ligand→delta_z 映射
- 分阶段验证，早期发现失败

**下一步**:
1. 快速实现方案 A 验证可行性
2. 如果有效，继续优化
3. 如果无效，考虑放弃这个方向或尝试其他方法
