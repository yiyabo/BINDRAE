# BINDRAE Stage-1 工作总结与设计反思

## 1. 任务定义与整体理念

### 1.1 Stage-1 的任务

- 数据形态：使用 CASF-2016 复合物，仅包含 **holo + ligand**（蛋白为结合态结构，配体为真实结合位姿）。
- 任务定义：
  - 输入：
    - 序列的 ESM-2 表征（冻结 encoder）
    - holo 主链骨架坐标 (N, Cα, C)
    - 配体 token（3D 坐标 + 20 维类型/拓扑特征）
  - 输出：
    - 7 个 torsion 角（phi / psi / omega / chi1–4）的分布
    - χ1 rotamer 分类（g⁻ / t / g⁺）
    - 通过 OpenFold 风格 FK 重建得到的全原子 atom14 坐标
  - 监督目标：在 **配体条件化** 下重构原始 holo 结构。

本质上，Stage-1 是一个 **ligand-conditioned protein conformation autoencoder / decoder**：
- 冻结 ESM-2 作为表征编码器。
- 训练一个几何 decoder（Adapter + LigandConditioner + IPA + TorsionHead + FK）在给定 ligand pose 条件下重建 holo 构象。

### 1.2 与传统 APO+ligand→HOLO 的区别与取舍

传统“诱导契合”网络往往直接做 **apo + ligand → holo**：模型同时学习“从 apo 出发的构象变化”和“终态 holo”的信息。

本项目的 Stage-1 刻意选择 **HOLO+ligand → HOLO 重构**，核心考虑：

- **数据可得性与稳定性**：
  - 现有数据集只有 holo 复合物，缺乏稳定的大规模 apo–holo 配对。
  - 强行做 apo→holo 会把数据噪声和 alignment 问题都压到网络上，训练难度和不确定性都很高。

- **与 RAE 两阶段范式对齐**：
  - RAE 中：Stage-1 训练一个高保真 **表示自编码器 / decoder**，Stage-2 在 latent 空间上做 diffusion / transport。
  - 对应到 BINDRAE：
    - Stage-1 专注于在 ligand 条件下学好“holo 构象分布的 decoder / 先验”；
    - Stage-2 再负责在构象/潜空间中学习 **apo→holo 的连续路径（flow / bridge）**。

因此：
- Stage-1 **不直接学习 apo→holo 的位移场**，而是学习“给定 ligand 时 holo 应该长什么样”。
- 真正的 “apo→holo 动力学 / 连续路径” 将放在 Stage-2 中解决，这一点与 RAE 的“Stage-1 重建, Stage-2 生成/transport”理念是一致的。


## 2. 模型架构与关键设计选择

### 2.1 数据与 IPABatch

在 `dataset_ipa.py` 中定义的 `IPABatch` 核心字段：

- 蛋白：`esm`, `N`, `Ca`, `C`, `node_mask`；
- 配体：`lig_points [B, M, 3]`, `lig_types [B, M, 20]`, `lig_mask`；
- 扭转角 GT：`torsion_angles [B, N, 7]`, `torsion_mask`；
- 口袋权重：`w_res [B, N]`（用于 pocket 加权、iRMSD 等）。

这里已经明确：训练数据是 **holo backbone + holo ligand** 的单构象样本，没有 apo 形态参与。

### 2.2 Stage-1 网络结构

`Stage1Model` 的主干流程：

1. **ESM Adapter**
   - 冻结 ESM-2 提供 per-residue 表征，Adapter 将 1280 维映射到内部通道 `c_s=384`。

2. **EdgeEmbedder**
   - 基于 Cα 坐标构造 pairwise edge 特征 `z_f1`, `z_f2`，为 FlashIPA 提供几何上下文。

3. **LigandConditioner（多层配体条件化）**
   - `LigandTokenEmbedding`：将配体坐标 (3D) 与 20 维类型/拓扑特征拼接，经 MLP 嵌入为 `d_lig` 维 token。
   - `ProteinLigandCrossAttention`：蛋白特征 ← 配体 token 的 cross-attn。
   - `FiLMModulation`：将配体信息以 FiLM 方式调制回蛋白特征，配合 gate/warmup 控制定量影响。
   - 在 Adapter 后立即注入一次，并在 FlashIPA 内部每层几何更新后再次调用（多层 ligand conditioning）。

4. **FlashIPA 几何分支**
   - 在配体条件化后的蛋白节点特征和 edge 特征上运行多层 IPAs，输出更新的 `s_geo` 和 `rigids_updated`。

5. **TorsionHead + Chi1RotamerHead**
   - `TorsionHead`：基于 `s_geo` 预测 7 个 torsion 的 (sin, cos) 分布。
   - `Chi1RotamerHead`：基于 `s_geo` 预测 χ1 的离散 rotamer 类别，作为辅助监督。

6. **OpenFoldFK**
   - 使用 `pred_torsions` + `rigids_updated` + `aatype` 进行 kinematics 展开，得到 atom14 坐标与掩码，用于 FAPE / distance / clash 计算。

这一架构使得：
- 配体空间信息（坐标 + 类型）在 **特征域和几何域** 都多次注入；
- 最终的 torsion 和全原子坐标都已经是 **ligand-aware** 的结果。

### 2.3 配体特征与条件化的演进

- **Phase 1：增加 χ1 rotamer 辅助头**
  - 通过离散 rotamer 监督明确 χ1 的模态结构，提升 χ1 精度与收敛稳定性。

- **Phase 2：多层 LigandConditioner**
  - 从“单次条件化”演进到在 IPA 各层间嵌入 ligand cross-attn，有效提升 pocket χ1 与整体几何对配体的敏感性。

- **Phase 3：20 维 ligand 类型/拓扑特征**
  - 从最早的 12 维 one-hot（元素/芳香/电荷）扩展到 20 维：
    - 是否在环中、环大小 bucket；
    - 重原子度数 bucket；
    - 是否有杂原子邻居等局部拓扑特征。
  - 在 `ligand_utils.py`、`LigandConditioner`、`dataset_ipa.py` 中打通，增强了配体空间信息的表达能力。


## 3. 损失函数与评价指标

### 3.1 训练损失设计

在 `Stage1Trainer.compute_loss` 中组合了以下损失：

- **Torsion loss (`w_torsion`)**
  - 对 7 个 torsion 角进行分布回归，采用加权角度差损失。

- **χ1 rotamer loss (`w_rotamer`)**
  - 将真实 χ1 映射到 3 个 rotamer bin，训练分类头，增强 χ1 多模态结构的学习。

- **Distance loss (`w_dist`)**
  - 基于 FK 重建的 Cα 与真实 Cα 的加权 L2 距离。

- **FAPE loss (`w_fape`)**
  - 使用 N/CA/C 三原子构造的帧进行 FAPE 计算，作为 backbone 局部刚性与几何一致性的主损失之一。

- **Clash penalty (`w_clash`)**
  - 基于 atom14 全部原子对，惩罚过近的非键合原子对（阈值约 2.2 Å），控制全原子 clash 水平。

此外还应用：
- **Pocket 权重 warmup (`w_res` + kappa)**：从 0.1 逐渐过渡到真实 `w_res`，稳定 early training；
- **Ligand gate warmup**（在 LigandConditioner 内部实现）：逐步打开配体条件化的影响，避免早期训练被 ligand 噪声主导。

### 3.2 验证指标与日志

在 `validate` 中记录了：

- Val total loss 及各分项（torsion / rotamer / dist / fape / clash）。
- **χ1 角度命中率**（基于角度差门限）。
- **χ1 rotamer 分类准确率**。
- **FAPE**（val 平均）。
- **Pocket iRMSD**（基于 Cα，对 `w_res>0.5` 的口袋残基做 Kabsch 对齐后的 RMSD）。
- **Clash%**（每个结构中，基于 atom14 计算的 clash 原子对比例）。

这些指标同时打印在训练日志的单行 summary 中，方便对比不同 epoch 的表现。


## 4. 主要实验结果（当前 best checkpoint）

基于一轮完整训练（batch_size=4, lr≈5e-4 等配置），早停在约 60+ epoch 时：

- **最佳 Val Loss**：约 **0.278**（出现在 epoch 58，一致于日志）。
- **χ1 角度命中率**：约 **75–76%**。
- **χ1 rotamer 准确率**：约 **75%**。
- **FAPE（主链）**：约 **0.055 Å** 级别（内部标度下，远优于设定阈值）。
- **Pocket iRMSD（Cα）**：约 **0.01 Å**（在当前数据与实现下已非常小）。
- **Clash%**：约 **9.4%**，在当前 clash 定义与权重下稳定在该水平。

整体训练曲线表现为：
- Train loss 持续缓慢下降，未见爆炸或严重过拟合迹象；
- Val loss 和 χ1 在 50+ epoch 后进入平稳区间，围绕最优值作小幅震荡，最终由 early stopping 触发结束训练；
- 口袋相关指标（pocket iRMSD、pocket χ1）表现优于全局平均，表明 `w_res` 与多层 ligand conditioning 确实把容量集中到了 binding region。


## 5. χ1 误差分析与启发

我们基于专门的分析脚本，对 val 集逐残基 χ1 进行了统计，得到以下结构性结论。

### 5.1 全局误差分布

在 val 集上（N≈9540 个有 χ1 的残基）：

- mean |Δχ1| ≈ **27°**，median ≈ **12°**；
- p90 ≈ **84°**，p95 ≈ **109°**；
- 误差直方图：
  - [0,20°): **65.4%**
  - [20,40°): 13.9%
  - [40,60°): 5.7%
  - [60,90°): 6.3%
  - [90,180°]: 8.7%

粗略地按 `|Δχ1|<30°` 估计 χ1 命中率，可得到 ≈ 0.72–0.73，与验证日志中 ≈75% 的 χ1 命中率一致。

整体而言：
- 大部分 χ1 误差集中在 20° 以内；
- 仍然存在一个长尾（约 15% >60°，8–9% >90°），对应彻底选错 rotamer 的残基。

### 5.2 口袋 vs 非口袋

按 `w_res>0.5` 定义 pocket：

- **Pocket 残基（N=218）：**
  - mean ≈ **18°**，median ≈ **9°**；
  - [0,20°): **79.4%**；
  - 高误差比例显著低于全局。

- **Non-pocket 残基（N=9322）：**
  - 统计量基本与全局一致：mean≈27°，[0,20°)≈65%。

进一步看高误差点的 pocket 占比：

- 全局 `frac(|Δχ1|>60°) ≈ 0.150`，其中 **仅约 1.1%** 在 pocket 内；
- `frac(|Δχ1|>90°) ≈ 0.087`，其中 **仅约 1.2%** 在 pocket 内。

**结论：**
- 当前模型的绝大部分 χ1 大错误来自 **非口袋 / 背景残基**；
- 口袋残基在 χ1 上明显更好，模型的表达能力优先投入到了与 binding 直接相关的区域，这是符合目标的。

### 5.3 按氨基酸类型的难度结构（全局）

按 `frac(|Δχ1|>60°)` 从大到小排序，全局最难的一些 aa：

- **S (0.224), N (0.213), E (0.211), K (0.194), Q (0.189), V (0.184)**；
- 随后是 D/R/M/T/H/C/W/I/L/F/Y/P 等，整体高误差比例逐渐下降。

这些最难的 aa 多为：
- 小极性/带电残基（S/N/E/K/Q/D），多位于表面，构象多模态；
- 部分 β-branched 或短疏水（V），也容易受局部噪声或 packing 差异影响；
- 它们的高误差点绝大多数 **不在口袋**（`pocket_frac_high≈0` 或极小）。

相对容易的 aa 包括：
- Pro 和多数芳香/疏水残基（P/F/Y/W/L/I），mean 误差 <~ 22°，p90 也相对较小，高误差比例在 0.1 以下甚至更低。

### 5.4 口袋内按氨基酸类型

只看 `is_pocket=True` 的子集（N=218）：

- 样本量较小，但趋势上：
  - 绝大多数 aa 在口袋内的 `frac(|Δχ1|>60°)` 非常低（很多为 0）；
  - 芳香（F/Y/W）、酸碱（D/E/H/K）、小极性（S/T/Q）在口袋内的 χ1 表现整体良好；
  - 少量 L/R/I/M/N 等在 pocket 内有略高的高误差比例，但受样本量影响较大。

综合口袋与全局视角：
- **最难的 S/N/E/K/Q/V 等，在口袋内的表现并不糟糕，其全局高误差主要来自非口袋背景残基。**
- 口袋中真正“重要的”侧链（芳香、酸碱、电荷配对位点）已经处于一个比较健康的误差水平。

### 5.5 对 loss/采样策略的启发

从 χ1 误差结构出发，我们得到几条对后续设计的指导性结论：

1. **不必为了压低全局 S/E/K/Q/V 的误差而大幅重构损失函数。**
   - 这些 aa 的大部分高误差点在非口袋，对 ligand binding 的直接影响有限；
   - 过度追求全局 χ1“完美”，可能会在本来物理多模态的位置强行拟合单一 rotamer，甚至牺牲其他指标。

2. **如需“针对性加强”，优先考虑 pocket∩特定 aa 的轻量 re-weighting。**
   - 可作为后续实验方向（暂未实现）：
     - 在 torsion / rotamer loss 中对 `(is_pocket=True) & (aa ∈ {L,R,I,M,W,N,...})` 增加一个小的权重因子（例如 ×1.1–1.2），
       在不破坏整体平衡的前提下弱化少数 pocket 内高误差模式。

3. **为 Stage-2 提供 error-aware 的监督过滤或权重。**
   - `chi1_errors_val.npz` 中记录了 per-residue 的误差信息，后续 Stage-2 可以：
     - 对“口袋内高误差 χ1”点降低监督强度或权重；
     - 在构象路径训练中区分高置信度 / 低置信度的 torsion 监督，避免被 Stage-1 的长尾误差误导。


## 6. 与 RAE 与 Stage-2 的衔接思考

### 6.1 Stage-1 作为 ligand-conditioned decoder / prior

对比 RAE 的两阶段设计：

- RAE：Stage-1 训练一个高保真 decoder（图像重建），Stage-2 在 latent 空间做 diffusion / transport。
- BINDRAE：
  - Stage-1：
    - 在 ESM-2 表征 + ligand tokens 的条件下，重建 holo 构象；
    - 实现了一个“配体条件化的蛋白构象 decoder / 先验”。
  - Stage-2（规划中）：
    - 在 torsion / rigid / 表征空间中，学习从 apo 分布到 holo 分布的连续路径（flow / score-based bridge 等）。

因此当前的 Stage-1 设计，与 RAE 的理念是对齐的：
- **Stage-1 不负责“从噪声或 apo 出发”，而负责“给一个好的条件表示，我能 decode 出合理的 holo 构象”。**
- Stage-2 则在这个 decoder / 先验之上，完成真正的 apo→holo transport。

### 6.2 伪-apo 扰动 denoising 的设想

在此基础上，我们提出了一个中长期的 Stage-1 扩展方向（Phase 3b）：

- 在 Stage-1 中加入 **小幅度“denoising autoencoder” 风格训练**：
  - 对 holo torsion / backbone 人为加小噪声（例如 χ1/χ2 略偏、backbone 轻微扰动），视作“近 apo”状态；
  - 在 ligand 条件下训练模型把该“noised holo” 拉回原始 holo。

- 这样可以让 Stage-1 在局部上显式学习一点“**从邻近构象 → holo** 的向量场”；
  - Stage-2 在全局构象空间中则相当于 **把这些局部向量场串联起来，形成长程 apo→holo 轨迹**；
  - 整体依然保持“Stage-1 高保真重建 + Stage-2 transport”的解耦精神，只是在 Stage-1 上增加了局部动力学 flavor。

该方向目前仅作为规划记录，尚未实施。


## 7. 小结与后续工作

### 7.1 当前 Stage-1 的状态

- 任务形态：HOLO + ligand → HOLO 重构，明确承担“配体条件化 decoder / 先验”的角色，而非直接 apo→holo 网络。
- 架构：
  - 冻结 ESM-2 表征；
  - Adapter + EdgeEmbedder + FlashIPA + 多层 LigandConditioner + TorsionHead + Chi1RotamerHead + FK；
  - 配体信息通过 20 维拓扑类型 + 多层 cross-attn/FiLM 强调 pocket 几何。
- 性能：
  - χ1 命中率 ≈75–76%，rotamer 准确率 ≈75%；
  - FAPE 和 pocket iRMSD 均处于十分理想的范围；
  - clash% ≈9.4%，有改进空间但已稳定；
  - 口袋残基在 χ1 上显著优于全局，绝大多数大误差 χ1 来自非口袋背景残基。

从“为 Stage-2 提供一个可靠的 ligand-conditioned holo 先验”的角度，
**当前 Stage-1 已经达到一个可以放心“封板”、将主要精力转向 Stage-2 设计的状态。**

### 7.2 后续重点方向

结合 TODO 与分析结果，Stage-1 相关的后续重点可以是：

- 在不打破现有平衡的前提下，探索 **口袋∩特定 aa 的轻量 loss 权重 re-weighting**；
- 进一步将 `chi1_errors_val.npz` 融入 Stage-2 的监督设计中，把高/低置信度监督显式区分开；
- 中长期考虑引入 Stage-1 伪-apo denoising 训练，增强局部“towards holo”的向量场信息，为 apo→holo 连续路径学习提供更平滑的局部先验。

整体而言，Stage-1 的架构与任务选择已经和后续 Stage-2/RAE 风格两阶段 pipeline 形成了清晰分工：
- **Stage-1：学好“在 ligand 条件下 holo 的样子”。**
- **Stage-2：学好“如何从 apo 走到这个 holo manifold 上”。**

本文件作为当前 Stage-1 阶段性工作的总结与设计反思，为后续 Stage-2 设计与论文撰写提供参考基线。
