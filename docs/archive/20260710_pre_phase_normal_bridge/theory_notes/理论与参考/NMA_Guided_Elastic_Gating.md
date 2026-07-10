# NMA-Guided Elastic Gating：物理感知的自适应门控

> **Physics-Informed Adaptive Gating for Large-Scale Conformational Changes**

---

## 1. 问题背景

### 1.1 数据分布：95% vs 5%

根据 AHoJ 数据库的统计：

| 类型 | 比例 | 骨架 RMSD | 典型特征 |
|------|------|-----------|----------|
| **小变构** | ~95% | < 5 Å | 侧链旋转、Loop 微调、局部重塑 |
| **大变构** | ~5% | > 5 Å | 铰链开合、结合后折叠、结构域重排 |

### 1.2 当前 BINDRAE 设计的适用性

Stage-2 的 Pocket Gate 机制：

\[
g_i(t) = \sigma\bigl(\mathrm{MLP}([h_i^{\mathrm{enc}}; t_{\mathrm{emb}}; w_{\mathrm{res},i}])\bigr)
\]

其中：

- \(w_{\mathrm{res},i}\)：基于配体距离的残基权重（离口袋近 → 权重大 → 允许变动）
- **核心假设**："背景板（远离口袋的区域）基本不动"

**对于 95% 的小变构**：完美适用

- 变构集中在口袋区域
- 远离口袋的区域确实不动

**对于 5% 的大变构**：假设失效

- 铰链区可能远离口袋，但需要大幅度运动
- "背景板"本身发生了重组

### 1.3 设计目标

为 5% 的 edge case 重写整个架构 → **得不偿失**

需要一个满足以下条件的解决方案：

- 轻量级（Lightweight）
- 即插即用（Plug-and-play）
- 物理可解释（Physics-informed）

---

## 2. 理论基础：简正模分析（NMA）

### 2.1 弹性网络模型（ENM）

将蛋白质建模为由弹簧连接的质点网络。对于 \(N\) 个 Cα 原子，定义势能函数：

\[
V = \frac{\gamma}{2} \sum_{i<j} C_{ij} \bigl(r_{ij} - r_{ij}^0\bigr)^2
\]

其中：

- \(\gamma\)：弹簧常数（通常取 1 kcal/mol/Å²）
- \(C_{ij}\)：接触矩阵，当 \(r_{ij}^0 < r_c\)（截断距离，通常 15 Å）时为 1，否则为 0
- \(r_{ij}\)：原子 i 和 j 之间的距离

### 2.2 Hessian 矩阵与简正模

在平衡位置附近，构建 \(3N \times 3N\) 的 Hessian 矩阵 \(\mathbf{H}\)：

\[
H_{\alpha\beta}^{ij} = \frac{\partial^2 V}{\partial x_i^\alpha \partial x_j^\beta}
\]

对 \(\mathbf{H}\) 进行特征值分解：

\[
\mathbf{H} \mathbf{u}_k = \lambda_k \mathbf{u}_k
\]

其中：

- \(\lambda_k\)：第 k 个特征值（振动频率的平方）
- \(\mathbf{u}_k\)：第 k 个特征向量（简正模方向）

**关键性质**：

- 前 6 个模态对应平动和转动（\(\lambda = 0\)），应跳过
- **低频模态**（\(\lambda\) 小）对应大尺度集体运动（如铰链开合）
- **高频模态**（\(\lambda\) 大）对应局部振动

### 2.3 物理意义

**核心文献结论**：蛋白质的大尺度变构（RMSD > 3Å），通常沿着其**低频简正模**的方向发生。

**类比**：就像吉他弦有固定的振动模式，蛋白质骨架也有其"最省力"的运动模式（如铰链开合）。这些模式不需要深度学习去"猜"，用简单的弹性网络模型几秒钟就能算出来。

---

## 3. 方法：NMA-Guided Elastic Gating

### 3.1 核心思想

利用 NMA 预计算出蛋白质的"运动倾向"，作为 Gate 的额外输入。

**物理直觉**：

- \(w_{\mathrm{res},i}\) 告诉模型："你离口袋多近"（**空间邻近性**）
- \(M_i^{\mathrm{nma}}\) 告诉模型："物理学说你多容易动"（**力学易变性**）

### 3.2 NMA 位移特征

对每个 Apo 结构，计算前 \(K\) 个低频模态（通常 \(K = 3 \sim 5\)）。对于残基 \(i\)，定义其 NMA 位移幅值：

\[
M_i^{(k)} = \|\mathbf{u}_k^{(i)}\| = \sqrt{(u_k^{x,i})^2 + (u_k^{y,i})^2 + (u_k^{z,i})^2}
\]

其中 \(\mathbf{u}_k^{(i)}\) 是残基 \(i\) 在第 \(k\) 个模态下的位移向量。

**聚合方案**：可将多模态位移聚合为单一分数：

\[
M_i^{\mathrm{nma}} = \sum_{k=1}^{K} w_k \cdot M_i^{(k)}
\]

其中 \(w_k\) 可以是均匀权重或反比于模态频率的权重。

### 3.3 增强版门控机制

**原始 Pocket Gate**：

\[
g_i(t) = \sigma\bigl(\mathrm{MLP}([h_i; t_{\mathrm{emb}}; w_{\mathrm{res},i}])\bigr)
\]

**增强版 Pocket Gate**：

\[
g_i(t) = \sigma\bigl(\mathrm{MLP}([h_i; t_{\mathrm{emb}}; w_{\mathrm{res},i}; M_i^{\mathrm{nma}}])\bigr)
\]

或者采用加性形式：

\[
g_i(t) = \sigma\bigl(\mathrm{MLP}_{\mathrm{base}}(h_i, t, w_{\mathrm{res},i}) + \alpha \cdot \mathrm{MLP}_{\mathrm{nma}}(M_i^{\mathrm{nma}})\bigr)
\]

**逻辑变化**：

- **常规情况**：\(M_i^{\mathrm{nma}}\) 很小且分布均匀，门控主要由口袋距离 \(w_{\mathrm{res}}\) 决定（保持原本优势）
- **大变构情况**：如果是一个 Hinge 运动，铰链区的 \(M_i^{\mathrm{nma}}\) 很大。网络会发现："虽然 \(w_{\mathrm{res}}\) 很小（离口袋远），但 \(M_i^{\mathrm{nma}}\) 很大（物理上该动）"，于是把 \(g_i(t)\) 打开

### 3.4 时间动态松弛（可选）

让 NMA 修正项只在轨迹早期起作用：

\[
g_i(t) = \sigma\bigl(\mathrm{MLP}_{\mathrm{base}}(\cdot) + \beta(t) \cdot \mathrm{MLP}_{\mathrm{nma}}(M_i^{\mathrm{nma}})\bigr)
\]

其中时间衰减函数：

\[
\beta(t) = \sigma\left(\frac{t_0 - t}{\tau}\right)
\]

- 在 \(t < t_0\)（早期）：\(\beta(t) \approx 1\)，允许沿 NMA 方向大幅度调整
- 在 \(t > t_0\)（后期）：\(\beta(t) \approx 0\)，关掉 NMA 权重，专注口袋精修

---

## 4. 方案优势

| 维度 | 评价 |
|------|------|
| **轻量级** | 仅增加 \(K\) 维输入特征（\(K \approx 3 \sim 5\)），不改变主体架构 |
| **物理可解释** | NMA 是经典生物物理方法，审稿人熟悉且认可 |
| **零风险** | 无大变构时，网络自动学习将 NMA 权重置为 0，退化回原版 |
| **计算成本** | 使用 ProDy 库，< 1 秒/结构，可离线预处理 |
| **论文亮点** | "Physics-Informed Deep Learning" 的高光点 |

---

## 5. 与 Stage-2 的整合

### 5.1 改动清单

| 组件 | 改动 |
|------|------|
| `Stage2Dataset` | 增加 `nma_features` 字段 |
| `Stage2Batch` | 增加 `nma_features: [B, N, K]` |
| `PocketGate` | 输入维度增加 \(K\) |
| 预处理 | 新增 NMA 特征预计算脚本 |

### 5.2 实现优先级

1. **Phase 1**：先不加 NMA，验证基础 Stage-2 管线
2. **Phase 2**：预计算 NMA 特征，作为消融实验
3. **Phase 3**：如果 RMSD > 5Å 的样本表现显著提升，则正式整合

---

## 6. 参考文献

1. **弹性网络模型**：Bahar, I., & Rader, A. J. (2005). Coarse-grained normal mode analysis in structural biology. *Current opinion in structural biology*.

2. **ProDy 库**：Bakan, A., Meireles, L. M., & Bahar, I. (2011). ProDy: protein dynamics inferred from theory and experiments. *Bioinformatics*.

3. **NMA 与变构**：Tama, F., & Sanejouand, Y. H. (2001). Conformational change of proteins arising from normal mode calculations. *Protein engineering*.

4. **AHoJ 数据库**：Škrlj, B., et al. (2022). AHoJ: Search and Retrieval of Apo and Holo Protein Structures.

---

## 7. 总结

> 与其让模型在巨大的参数空间里盲目搜索"哪里是背景，哪里是铰链"，不如直接把答案（NMA 特征）喂给它。

这是一个**高性价比的创新点**，完全符合"不改动主体架构"的诉求，同时为论文提供了 **Deep Learning + Biophysics** 结合的亮点。
