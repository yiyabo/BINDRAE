# BINDRAE Stage‑2 理论与实现指导

> 面向“最高标准”的设计，不做降级或兼容性妥协。
>
> 本文在 `Stage2.md` 的算法蓝图基础上，从**理论视角**和**实现规范**两个层面重新组织 Stage‑2 方案，目标是：
>
> - 在混合状态空间（SE(3) 骨架刚体 + torsion 角）中，构造一个**配体条件化、口袋门控**的 apo→holo 桥流；
> - 明确参考桥（reference bridge）、Flow Matching / Bridge Flow 的数学形式；
> - 系统性地把 Stage‑1 的几何先验（FK / FAPE / clash / pocket contact / Stage‑1 holo prior）提升到**整条路径**上作为正则；
> - 给出不降级的工程落地方向。

---

## 1. 设计目标与严格要求

### 1.1 任务目标

给定：

- **序列** `S`（冻 ESM‑2 表征），
- **apo 结构** `P_apo`（未结合或弱结合，含 backbone 与 torsion），
- **holo 结构** `P_holo`（真实配体结合构象），
- **配体** `L`（具有 bound pose），

在时间区间 \([0,1]\) 上构造一个**时间连续的随机过程** \(x(t)\)，满足：

- \(x(0)\) 分布在 apo 构象附近，\(x(1)\) 分布在 holo 构象附近；
- 过程在几何上 SE(3)‑equivariant，并主要在**口袋区域自由度**上发生显著变化；
- 给定配体和序列，该过程是一个**条件桥（conditional bridge）**，近似满足 Schrödinger bridge / bridge flow 的性质；
- 任意时间点 \(t\) 的状态 \(x(t)\) 经 Stage‑1 的 FK 解码后，几何上（FAPE / clash / pocket contact）均保持物理合理。

### 1.2 严格设计原则

1. **不做降级到“只看坐标”或“只看 torsion”**：
   - 最终形式在混合状态空间 \(\mathcal{M} = \mathcal{F} \times \Theta\) 上建模：
     - \(\mathcal{F} = \mathrm{SE(3)}^N\)：每个残基一个 N/Cα/C 刚体帧；
     - \(\Theta = (S^1)^K\)：所有定义 torsion 自由度（φ, ψ, ω, χ1–4）。
2. **显式建模向量场 \(v_\Theta(x,t)\)**：
   - 同时预测 \(d\,\text{rigids}/dt\) 与 \(d\theta/dt\)，
   - 不仅依赖状态，还**条件化于**：ESM、ligand tokens、pocket 权重、时间 t。
3. **采用带噪参考桥 + Conditional Flow Matching / Bridge Flow**：
   - 不简单用线性插值训练一个回归器；
   - 明确参考桥的解析形式，使用 PCFM / CFM 理论框架。
4. **路径级几何约束不降级**：
   - FAPE / clash / pocket contact / Stage‑1 prior 必须在路径上起作用，而不只是终点修饰。
5. **Stage‑1 作为固定 holo decoder/prior**：
   - Stage‑2 不推翻 Stage‑1，而是在其之上学习动态；
   - 终点和后半段路径均受 Stage‑1 holo 先验的软约束。

---

## 2. 状态空间与对称性

### 2.1 混合状态空间 \(\mathcal{M} = \mathcal{F} \times \Theta\)

对一个 N 残基蛋白：

- **刚体空间** \(\mathcal{F}\)：
  - 每个残基 i 有一个刚体帧 \(F_i = (R_i, t_i)\)，其中 \(R_i \in \mathrm{SO(3)}, \ t_i \in \mathbb{R}^3\)；
  - 整体骨架帧 \(F = (F_1,\dots,F_N) \in \mathrm{SE(3)}^N\)。

- **torsion 空间** \(\Theta\)：
  - 每个残基 i 有一个 7 维 torsion 向量：
    \[
    \theta_i = (\phi_i, \psi_i, \omega_i, \chi_{1,i}, \dots, \chi_{4,i}) \in (S^1)^7,
    \]
    实际上部分 χ_k 不一定定义，通过掩码控制；
  - 全体 torsion 记为 \(\theta = (\theta_1, \dots, \theta_N) \in (S^1)^K\)，K 为所有定义扭转角总数。

全状态：
\[
 x = (F, \theta) \in \mathcal{M} = \mathrm{SE(3)}^N \times (S^1)^K.
\]

### 2.2 SE(3)‑equivariance 与不变性

- **整体刚体变换不应影响路径的“形状”**：
  - 若对 apo/holo 及中间所有帧施加同一个全局刚体变换 \(G \in \mathrm{SE(3)}\)，
  - 则希望学到的向量场 \(v_\Theta\) 在坐标系变化下保持 equivariant。

- 在实现上：
  - torsion 分量天然 SE(3) 不变；
  - backbone rigid 部分可通过基于局部帧的 IPA / SE(3)‑equivariant 模块实现（参考 Stage‑1 的 FlashIPA）。

### 2.3 口袋权重与自由度聚焦

- 定义 \(w_{\mathrm{res},i} \in [0,1]\) 作为残基 i 的口袋权重（由 holo+ligand 预处理得到）；
- 训练与推理过程中：
  - 对高 \(w_{\mathrm{res}}\) 区域允许更大、更多样的速度场；
  - 对低 \(w_{\mathrm{res}}\) 区域施加更强的平滑/收缩正则，以避免无意义全局抖动。

---

## 3. 条件概率视角：apo→holo 桥流

### 3.1 “桥”问题的形式化

固定配体 \(L\) 与序列 \(S\) 时：

- 定义：
  - apo 端点分布 \(p_0(x) = p(x | \text{apo}, S, L)\)，
  - holo 端点分布 \(p_1(x) = p(x | \text{holo}, S, L)\)。

目标：

- 构造一个时间连续 Markov 过程 \((X_t)_{t\in[0,1]}\)，满足：
  - \(X_0 \sim p_0, \ X_1 \sim p_1\)（或尽可能接近）；
  - 在动力学上“简单”（例如漂移 + 小噪声），
  - 同时保持几何 / 生物物理约束。

这就是 Schrödinger bridge / bridge flow 问题的条件版本：

- 最小化一个 Kullback–Leibler 或能量泛函，
- 约束端点分布为 \(p_0, p_1\)。

### 3.2 Flow Matching / Bridge Flow 近似

直接求 SB 的解析解非常困难，因此采用 **Conditional Flow Matching (CFM) / Pairwise CFM / PCFM** 的范式：

1. 构造一个参考桥 \(X_t^{\text{ref}}\)：
   - 通常是简单插值 + Brownian 噪声的过程；
2. 构造对应解析速度场 \(u_t^{\text{ref}}(x)\)；
3. 通过最小化：
   \[
   \mathbb{E}_{x_0,x_1,t,\xi} \big[ \lVert v_\Theta(X_t^{\text{ref}}, t) - u_t^{\text{ref}}(X_t^{\text{ref}}) \rVert^2 \big]
   \]
   来学习一个“更简单、参数化”的向量场 \(v_\Theta\)，近似参考桥的漂移；
4. 在推理时，只积分 \(v_\Theta\) 即可。

> 实现层面上，参考桥 \(X_t^{\text{ref}}\) 本身是一个带噪的随机过程，而 \(v_\Theta\) 参数化的是其**期望漂移**：训练时在随机参考桥样本上做 Flow Matching，推理阶段默认解确定性 ODE \(\tfrac{dx}{dt} = v_\Theta(x,t)\)。如需显式随机路径，可在积分时叠加小噪声或扩展为 SDE 形式。

Stage‑2 中，我们在混合空间 \(\mathcal{M}\) 上构造参考桥，并在此基础上做 CFM。

---

## 4. 参考桥的构造（刚体 + torsion）

### 4.1 torsion 参考桥：周期空间上的 Brownian bridge

给定端点 torsion 向量 \(\theta_0, \theta_1 \in (S^1)^K\)：

1. 定义最短角差：
   \[
   \Delta\theta = \mathrm{wrap\_to\_\pi}(\theta_1 - \theta_0) \in (-\pi, \pi]^K.
   \]

2. 选取平滑插值函数 \(\gamma(t)\)、噪声尺度 \(\sigma(t)\)：
   - 示例：
     \[
     \gamma(t) = t, \quad \sigma(t) = \lambda \sqrt{t(1-t)},
     \]
     其中 \(\lambda\) 控制噪声强度；
   - 或使用 smoothstep：\(\gamma(t) = 3t^2 - 2t^3\)，在端点更平滑。

3. 定义参考桥：
   \[
   \theta_t^{\text{ref}} = \theta_0 + \gamma(t) \, \Delta\theta + \sigma(t) \, \xi, \quad \xi \sim \mathcal{N}(0, I_K),
   \]
   然后对每个分量再做一次 \(\mathrm{wrap\_to\_\pi}\) 映射回 \((-\pi, \pi]\)。

4. 解析速度（在欧氏近似下）：
   \[
   u_t^{\text{ref}}(\theta_t^{\text{ref}}) = \frac{d}{dt}\theta_t^{\text{ref}}
   = \gamma'(t)\,\Delta\theta + \sigma'(t)\,\xi.
   \]

> 注：在实现中，可以有两种层级：
> - **无噪声 PCFM**：\(\sigma(t)\equiv 0\)，目标更干净；
> - **带噪桥流**：\(\sigma(t)\neq0\)，理论上更接近 SB。本文默认以带噪形式为“最高标准”设定。

### 4.2 刚体 SE(3) 参考桥：帧间 geodesic + Brownian 扰动

对每个残基 i：

- apo 帧 \(F_{0,i} = (R_{0,i}, t_{0,i})\)，
- holo 帧 \(F_{1,i} = (R_{1,i}, t_{1,i})\)。

构造：

1. 旋转部分：
   - 定义相对旋转：\(R_{\Delta,i} = R_{1,i} R_{0,i}^\top\)；
   - 取对数映射 \(\Omega_i = \log(R_{\Delta,i}) \in \mathfrak{so}(3)\)（轴‑角向量）；
   - 参考轨迹：
     \[
     R_i^{\text{ref}}(t) = \exp(\gamma(t)\, \Omega_i) R_{0,i}.
     \]

2. 平移部分：
   - 直接线性插值：
     \[
     t_i^{\text{ref}}(t) = t_{0,i} + \gamma(t) (t_{1,i} - t_{0,i}).
     \]

3. 若考虑噪声，可在 Lie algebra 上加入 Brownian 扰动：
   \[
   R_i^{\text{ref}}(t) = \exp(\gamma(t)\, \Omega_i + \sigma_R(t)\,\xi_{R,i}) R_{0,i},
   \]
   平移类似加 \(\sigma_T(t)\,\xi_{T,i}\)。

4. 对刚体帧的参考速度可在 Lie algebra 中解析：
   - 旋转速度 \(\omega_i(t) \in \mathbb{R}^3\)：
     \[
     \omega_i^{\text{ref}}(t) = \gamma'(t)\, \Omega_i + \sigma_R'(t)\,\xi_{R,i},
     \]
   - 平移速度 \(v_i^{\text{ref}}(t) = \frac{d}{dt}t_i^{\text{ref}}(t)\)。

最终，参考桥状态：
\[
X_t^{\text{ref}} = (F^{\text{ref}}(t), \theta^{\text{ref}}(t)),
\]
参考速度 \(u_t^{\text{ref}} = (\dot F^{\text{ref}}(t), \dot\theta^{\text{ref}}(t))\)。

---

## 5. 向量场学习：Conditional Flow Matching / Bridge Flow

### 5.1 条件输入与向量场形式

在任意时间 t，我们要学习一个条件向量场：

\[
 v_\Theta(x,t \mid S, L, w_{\mathrm{res}})
 = \big( \dot F_\Theta(x,t), \dot\theta_\Theta(x,t) \big),
\]

其中条件包含：

- **序列特征**：冻结 ESM‑2 per‑res 表征，经 Adapter 投影到内部通道 \(c_s\)；
- **配体特征**：
  - ligand tokens：配体重原子 + 方向探针 + 20D 类型/拓扑特征，经 `LigandTokenEmbedding` 得到 \(L_{\text{tok}}\)；
- **口袋权重** \(w_{\mathrm{res}}\)：作为 scalar feature 拼入每个残基 token，并在 loss 中做加权；
- **时间** t：通过 sin/cos 或小 MLP 嵌入为 \(e_t \in \mathbb{R}^{d_t}\)。

模型总体结构：

1. 用 FK 将当前 \(x(t)\) 解码为 atom14 坐标；
2. 用 EdgeEmbedder 构造 residue‑residue pair 特征；
3. 用 ESM Adapter + LigandConditioner + FlashIPA 得到 ligand‑aware 的几何表征 \(h_i(t)\)；
4. 通过 pocket gate MLP 计算每个残基的门控 \(g_i(t)\in(0,1)\)；
5. 通过 torsion head & rigid head 输出 \(\dot\theta_\Theta, \dot F_\Theta\)，再乘以 \(g_i(t)\)。

> 在 Stage‑2 中，我们将 ESM Adapter + EdgeEmbedder + LigandConditioner + FlashIPA 视为一个几何 encoder（记为 Enc_trunk），其内部对刚体帧的更新仅作为 encoder 的辅助状态，不回写到全局状态 \(F(t)\)。全局 \(F(t)\) 的演化**只由**向量场 \(\dot F_\Theta(x,t)\) 控制，以避免“state rigids”与“encoder 内部 rigids”双重更新导致的不一致。这一点与 Stage‑1 文档中将 FlashIPA 视作几何主干、而非显式状态更新器的设定保持一致。

### 5.2 Flow Matching 损失（带口袋加权）

对参考桥样本 \(X_t^{\text{ref}}, u_t^{\text{ref}}\)：

\[
L_{\text{FM}}
= \mathbb{E}_{b,t,\xi} \left[\sum_{i,k}
   w_{\mathrm{res},i}^\alpha \, \mathrm{mask}_{i,k}
   \big\| v_\Theta^{(i,k)}(X_t^{\text{ref}}, t) - u_t^{\text{ref},(i,k)} \big\|^2
\right],
\]

- \(\mathrm{mask}_{i,k}\) 由 \(bb\_mask\) 与 \(chi\_mask\) 组合而来，
- \(w_{\mathrm{res},i}^\alpha\)（\(\alpha\approx1\sim2\)）突出 pocket 自由度的重要性；
- 刚体和平移分量同样使用 \(w_{\mathrm{res}}\) 加权。

这就是一个**条件 PCFM / CFM** 目标：

- 对给定配体/序列/口袋权重条件下，学习最优漂移 \(v_\Theta\)，近似参考桥。 

### 5.3 端点一致性（endpoint consistency）

为确保数值积分后确实从 apo 到 holo，再加入端点一致性项：

1. 从 \(x_0\) 出发，数值积分：
   \[
   \frac{dx(t)}{dt} = v_\Theta(x(t), t), \quad x(0) = x_0,
   \]
   得到 \(x_\Theta(1)\)。
2. 对比 \(x_\Theta(1)\) 与真实 \(x_1\)（或 Stage‑1 holo prior）：
   - torsion L2 / wrap‑angle loss；
   - backbone FAPE（使用 N/Cα/C 帧），对口袋加权；

\[
L_{\text{endpoint}} = \lambda_\theta \cdot \mathrm{Loss}_\theta(\theta_\Theta(1), \theta_1) + \lambda_{\text{FAPE}} \cdot \mathrm{FAPE}(x_\Theta(1), x_1).
\]

端点一致性无需每步都计算，可隔一定步数或在训练后期启用，以控制计算量。

---

## 6. 几何与生物物理路径正则

Flow Matching 只“对齐”参考桥的速度场，并不直接约束中间帧的物理合理性。利用 Stage‑1 的几何工具，我们在若干时间点 \(t_k\) 额外施加**路径正则**。仅作为轻量/预热方案时，可以在参考桥 \(X_t^{\text{ref}}\) 上短暂施加同样的几何项，但本文默认主方案是在 learned path 上约束。

### 6.1 FAPE / Cα 路径平滑（L_smooth）

对一组时间点 \(0 < t_1 < \dots < t_K < 1\)：

1. 解码 \(x(t_k)\) 为 atom14 坐标（或至少 backbone Cα）；
2. 使用 FAPE 或 Cα L2 约束相邻帧：

\[
L_{\text{smooth}} = \sum_{k=0}^{K-1}
 \mathbb{E}\big[ w_{\mathrm{res}} \cdot d_{\text{backbone}}(x(t_k), x(t_{k+1}))^2 \big],
\]

- 其中 \(d_{\text{backbone}}\) 可取 FAPE 或 Kabsch‑aligned Cα RMSD；
- 口袋权重用于强调 pocket 的连续性和平滑性。

### 6.2 路径 clash 正则（L_clash）

在每个 \(t_k\) 上：

- 使用与 Stage‑1 相同的随机子采样 clash loss（方案 A）：
  - 从 atom14 中随机选若干 chunk，计算非成键距离是否小于阈值（≈2.0–2.2 Å）；
  - 得到 clash 百分比或惩罚项。

\[
L_{\text{clash}} = \sum_{k} \mathbb{E}\big[ \mathrm{ClashPenalty}(x(t_k)) \big].
\]

这保证路径上不会出现大量严重自穿插，即便端点是合理的。

### 6.3 口袋接触软单调性（L_contact）

定义口袋残基集合：\(\mathcal{P} = \{ i : w_{\mathrm{res},i} > \tau \}\)。

1. 对每个时间点 t，定义一个 soft contact 分数：
   \[
   C(t) = \frac{1}{|\mathcal{P}|} \sum_{i \in \mathcal{P}} \mathrm{soft\_contact}(\mathrm{dist}(\text{res}_i, L; t)),
   \]
   其中 soft_contact 可用 logistic(dist) 或 RBF 转换（越接近配体越接近 1）。

2. 希望 \(t \to C(t)\) 随时间大致单调递增：

\[
L_{\text{contact}} = \sum_{k=0}^{K-1} \max\big(0, C(t_k) - C(t_{k+1}) - \varepsilon\big),
\]

- \(\varepsilon \ge 0\) 为允许的最大下降幅度（例如 0.05），只在接触强度下降超过 \(\varepsilon\) 时施加惩罚；
- 本质上是一个**口袋接触强度的软单调性约束**，避免路径出现“先塞进去再抽出来”的非物理行为。

这一项是 Stage‑2 的一个**重要 novel 设计点**，利用 ligand‑conditioned pocket 先验对整条路径的方向性加约束。

### 6.4 Stage‑1 holo prior 对齐（L_prior）

利用 Stage‑1 训练好的 holo decoder，将其作为 Stage‑2 在后半段的先验：

1. 对固定条件（apo backbone + ESM + 配体）运行 Stage‑1，得到其预测的 holo torsion 分布 \(\theta_{\text{stage1}}\)；
2. 对路径上较大的 t（例如 \(t > t_\text{mid}\)，如 0.5）：

\[
L_{\text{prior}} = \mathbb{E}_{t > t_\text{mid}} \big[ w_{\mathrm{res}} \cdot d_\theta(\theta(t), \theta_{\text{stage1}})^2 \big],
\]

- \(d_\theta\) 是 wrap‑aware 的角度差度量；
- 只在后半段施加，以免过早把路径拉向单一点，保留前半段多样性。

在训练阶段（存在真实 holo 端点）时，\(L_{\text{endpoint}}\) 与 \(L_{\text{FM}}\) 仍以 \(x_1 = x_{\text{holo}}\) 为主监督，\(L_{\text{prior}}\) 建议以较小权重使用，作为后半段轨迹的平滑先验；而在推理阶段（仅有 apo + ligand 时），Stage‑1 预测的 \(\theta_{\text{stage1}}\) 及由其构造的 pseudo‑holo 端点 \(x_1'\) 则充当终点条件，对应第 8.2 节中的实际应用场景。

这一项使 Stage‑2 在终点附近“落”到 Stage‑1 学到的 holo manifold 上，形成**高保真先验 + 连续路径**的组合。

---

## 7. 总损失与训练算法

### 7.1 总损失形式

综合上述组件：

$$
L = L_{\text{FM}} + \lambda_{\text{end}} L_{\text{endpoint}} + \lambda_{\text{geom}} \big( L_{\text{smooth}} + L_{\text{clash}} + L_{\text{contact}} + L_{\text{prior}} \big)
$$

- 所有 residue‑level 项都可以再乘以 \(w_{\mathrm{res}}\) 或其幂，以强化 pocket 区域；
- 所有 residue‑level 项可以使用组合权重 \(w_i = w_{\mathrm{res},i}^\alpha \cdot r_i\)，其中 \(r_i\) 来自 Stage‑1 χ1 offline 误差分析（例如依据《Stage‑1 工作总结与 χ1 误差分析》或 `chi1_error_analysis.py` / `chi1_error_posthoc.py` 统计得到的置信度），对高误差长尾残基略减权；
- 这样可以显式地把 Stage‑1 中对 χ1 长尾行为的认识注入 Stage‑2 的路径监督，使两阶段在权重设计上形成闭环。

### 7.2 单步训练过程（理论级别）

对一个 batch 的 apo–holo–ligand 三元组：

1. 样本转为 `Stage2Batch`：包含 \(\theta_0, \theta_1, F_0, F_1, ESM, L_{\text{tok}}, w_{\mathrm{res}}\) 等；
2. 采样时间 \(t \sim \mathcal{U}(0,1)\)，采样噪声 \(\xi\)；
3. 构造参考桥 \(X_t^{\text{ref}}, u_t^{\text{ref}}\)（torsion + SE(3)）；
4. 通过 `TorsionFlowNet` 计算 \(v_\Theta(X_t^{\text{ref}}, t)\)；
5. 计算 \(L_{\text{FM}}\)；
6. 在若干 \(t_k\)（可以与 t 相同或独立采样）上：
   - 从 \(x_0\) 或当前状态出发，积分到 \(t_k\) 得到 \(x_\Theta(t_k)\)；
   - 对 \(x_\Theta(t_k)\) 解码 FK，计算 \(L_{\text{smooth}}, L_{\text{clash}}, L_{\text{contact}}, L_{\text{prior}}\)；仅在训练早期可选在 \(X_t^{\text{ref}}\) 上近似这些几何项，作为 warmup。
7. （可选）周期性计算 \(L_{\text{endpoint}}\)；
8. 聚合为 \(L\) 并反向传播更新 \(v_\Theta\) 参数。

---

## 8. 推理与评估

### 8.1 已知 apo + holo + ligand（路径重建）

- 用与训练完全相同的条件构建 \(x_0, x_1\)；
- 只依赖学习到的向量场 \(v_\Theta\)：
  \[
    \frac{dx}{dt} = v_\Theta(x(t), t), \quad x(0) = x_0,
  \]
  使用 ODE 求解器或固定步长积分（Euler / Heun / RK4）；
- 在多个 t 处解码路径 \(x(t)\)，可视化 apo→holo 结构变化，并评估：
  - 与真实 holo 的终点误差（torsion/FAPE/pocket iRMSD）；
  - 路径上的 clash% 与 contact 单调性。

### 8.2 仅 apo + ligand（使用 Stage‑1 先验）

- 用 Stage‑1 在 (apo backbone + ESM + ligand) 条件下预测 holo‑like torsion \(\theta_{\text{stage1}}\)；
- 构造 pseudo‑holo 端点 \(x_1'\)：
  - 刚体可以沿用 apo backbone 或经少量几何 refinement；
- 在 \(x_0\) 与 \(x_1'\) 之间运行同一 Stage‑2 桥流，得到一条“依托 Stage‑1 先验”的 apo→holo 轨迹；
- 这是 Stage‑2 在真实应用中更常见的推理模式。

### 8.3 评估指标建议

- 终点质量：
  - torsion 角误差（特别是 χ1 命中率、rotamer 准确率）；
  - backbone/pocket FAPE；
  - pocket Cα iRMSD；
  - clash%；
- 路径质量：
  - 平均 clash%；
  - contact 曲线 \(C(t)\) 的形状与单调性违例比例；
  - 局部 FAPE / Cα RMSD 的平滑度；
- 统计分析：
  - 分 pocket / 非 pocket、按氨基酸类型统计路径误差和自由度参与度。

---

## 9. 与 Stage‑1 与相关工作的关系（总结）

- **Stage‑1**：
  - 冻结 ESM encoder；
  - 通过 Adapter + LigandConditioner + FlashIPA + TorsionHead + FK，学习“给定配体时 holo 应该长什么样”；
  - 提供高保真 ligand‑conditioned holo decoder / prior 和几何 loss 工具箱。

- **Stage‑2**（本指导文档）：
  - 在 \(\mathrm{SE(3)}^N \times (S^1)^K\) 上，
  - 用 Conditional Flow Matching / Bridge Flow 学习 apo→holo 的时间连续向量场，
  - 该向量场：
    - 明确条件化于 ligand + ESM + pocket 权重；
    - 通过 pocket‑gate 聚焦 binding region；
    - 在路径上满足 FAPE / clash / contact / prior 等生物物理约束。

- 与 RAE / ProteinZen / SBALIGN / DiSCO 等工作的关系：
  - 继承了“冻结 encoder + 解码器 + 潜空间/构象空间 flow”的两阶段精神；
  - 但：
    - 使用物理上更可解释的 torsion + SE(3) 混合状态空间；
    - 显式建模 ligand 条件；
    - 利用 pocket‑gated vector field 与 contact 单调性做领域先验约束；
    - 将一个高保真的 holo decoder（Stage‑1）直接融入 bridge flow 的路径能量。

> 补充说明：本文以显式状态 \(x=(F,\theta)\) 上的桥流为主线；在工程实现上，也可以在 Enc_trunk 输出的 per‑residue latent \(z_i\) 上构建一个简化的 latent flow，并复用相同的几何解码器与路径正则，作为对比或过渡方案，两者在整体框架上是一致的。

---

## 10. 实现层面的硬约束与建议

本指导文档假定：

- 不做“只 torsion”或“只坐标”的简化；
- 不弱化或移除几何/接触正则，只在计算量上做合理调度（如稀疏时间点计算 endpoint consistency）。

落地实现时建议：

1. **模块结构**：
   - `src/stage2/datasets/dataset_bridge.py`：构建 `Stage2Batch`；
   - `src/stage2/models/torsion_flow.py`：实现 `TorsionFlowNet`（含 pocket gate、torsion/rigid heads）；
   - `src/stage2/training/trainer.py`：
     - 参考桥构造（torsion + SE(3)）；
     - Flow Matching + 几何正则 + endpoint consistency；
     - 调用 Stage‑1 的 FK 和 loss 组件。

2. **数值稳定性**：
   - 时间积分优先使用 Heun / RK4 而非纯 Euler；
   - 控制步数 T（如 20–40）与 batch size 的平衡；
   - 对刚体旋转的更新使用指数/对数映射，参考 Stage‑1 / FlashIPA 中对 SO(3) 的实现：在旋转角接近 0 时使用安全近似（如泰勒展开或对角度做 clamp），避免数值不稳定和漂移出 SO(3)。

3. **资源与调度**：
   - 几何正则（特别是 clash）成本较高，路径上的时间点数量建议 \(K\approx3\text{–}5\)，并在每个 \(t_k\) 上使用与 Stage‑1 相同的随机 chunk 采样方案 A（例如每条样本约 512 个原子对）；
   - endpoint consistency 可隔若干 step 再启用，以节省 ODE 积分开销；
   - 噪声日程建议：实现上可先采用 \(\sigma(t)\equiv0\) 的 deterministic PCFM 作为默认配置，保证训练与数值积分稳定；在此基础上，如需更强多样性或更接近 Schrödinger bridge 设定，可逐步引入小幅 \(\sigma(t)\)。

4. **实验规划**：
   - 在完整混合空间与完整正则框架下，仍可通过“先低 epoch、小数据子集”探索合适的权重与步数，
   - 这属于**收敛与稳定性调参**，不属于理论上的“降级”或“简化”。

本文件与 `Stage2.md` 共同构成 Stage‑2 的“理论 + 算法 + 工程规范”三件套：

- `理论与参考.md`：总体动机与两阶段框架；
- `Stage2.md`：详细算法蓝图与伪代码；
- `Stage2理论与指导.md`（本文）：在最高标准前提下的理论抽象与实现约束说明。
