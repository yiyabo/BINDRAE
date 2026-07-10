# BINDRAE Stage‑2 理论与实现指导


> 本文在 `Stage2.md` 的算法蓝图基础上，从**理论视角**和**实现规范**两个层面重新组织 Stage‑2 方案，目标是：
>
> - 在混合状态空间（SE(3) 骨架刚体 + 侧链 χ torsion）中，构造一个**配体条件化、口袋门控**的 apo→holo 桥流；
> - 明确参考桥（reference bridge）、Flow Matching / Bridge Flow 的数学形式；
> - 系统性地把 Stage‑1 的几何先验（FK / FAPE / clash / pocket contact / Stage‑1 holo prior）提升到**整条路径**上作为正则；

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

> 任务边界（必须写清楚）：本文主线假设 **bound pose 已知**，定位为 *known‑pose induced‑fit / 构象路径生成*，不将 docking/pose 搜索作为主线贡献。建议在实验中报告 **pose 噪声敏感性曲线**（对 pose 注入 RMSD 扰动或使用 docked pose）以明确适用边界与鲁棒性。

### 1.2 严格设计原则

1. **主线采用“去冗余”的混合状态（不在全原子坐标上做 flow）**：
   - 主线显式状态在 \(\mathcal{M}=\mathcal{F}\times\mathcal{X}\) 上建模：
     - \(\mathcal{F}=\mathrm{SE(3)}^N\)：每残基一个 backbone 刚体帧（N/Cα/C frame）；
     - \(\mathcal{X}=(S^1)^{K_\chi}\)：侧链 torsion（χ1–χ4）集合（通过掩码决定哪些 χ_k 定义）。
   - **φ/ψ/ω 不作为可积分的显式状态变量**：若需要，可从当前 \(F(t)\) 派生得到（或仅在端点/评估中使用），以避免 \((F,\theta)\) 同时演化带来的冗余与不一致风险。
   - **官方 baseline/消融**：提供 torsion‑only（仅角变量）或 full‑torsion（φ/ψ/ω/χ）版本，用于稳定性对照与消融，但不作为主线默认。
2. **显式建模向量场 \(v_\Theta(x,t)\)**：
   - 同时预测 \(dF/dt\) 与 \(d\chi/dt\)，
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

### 2.1 混合状态空间 \(\mathcal{M} = \mathcal{F} \times \mathcal{X}\)

对一个 N 残基蛋白：

- **刚体空间** \(\mathcal{F}\)：
  - 每个残基 i 有一个刚体帧 \(F_i = (R_i, t_i)\)，其中 \(R_i \in \mathrm{SO(3)}, \ t_i \in \mathbb{R}^3\)；
  - 整体骨架帧 \(F = (F_1,\dots,F_N) \in \mathrm{SE(3)}^N\)。

- **侧链 torsion 空间** \(\mathcal{X}\)：
  - 每个残基 i 有一个最多 4 维的侧链 torsion 向量：
    \[
    \chi_i = (\chi_{1,i}, \dots, \chi_{4,i}) \in (S^1)^4,
    \]
    实际上部分 χ_k 不一定定义，通过掩码控制；
  - 全体侧链 torsion 记为 \(\chi = (\chi_1, \dots, \chi_N) \in (S^1)^{K_\chi}\)，其中 \(K_\chi\) 为所有定义 χ 自由度总数。
  - 主链 torsion（φ/ψ/ω）可作为数据字段/评估指标保留，但不作为显式状态随时间演化。

全状态：
\[
 x = (F, \chi) \in \mathcal{M} = \mathrm{SE(3)}^N \times (S^1)^{K_\chi}.
\]

### 2.2 SE(3)‑equivariance 与不变性

- **整体刚体变换不应影响路径的“形状”**：
  - 若对 apo/holo 及中间所有帧施加同一个全局刚体变换 \(G \in \mathrm{SE(3)}\)，
  - 则希望学到的向量场 \(v_\Theta\) 在坐标系变化下保持 equivariant。

- 在实现上：
  - torsion 分量天然 SE(3) 不变；
  - backbone rigid 部分可通过基于局部帧的 IPA / SE(3)‑equivariant 模块实现（参考 Stage‑1 的 FlashIPA）。

### 2.3 口袋权重与自由度聚焦

- 定义 \(w_{\mathrm{res},i} \in [0,1]\) 作为残基 i 的口袋权重（残基–配体距离 + 图膨胀 + RBF/Logistic soft weighting 等可复现规则生成）；
- **训练/推理一致性（推荐默认）**：统一用 **apo backbone + ligand pose（在 apo 坐标系）** 来计算 \(w_{\mathrm{res}}\)（训练阶段也如此），避免“训练用 holo 真值 pocket、推理只能用 apo pocket”导致的分布偏移。
  - 若训练数据同时有 holo，可额外记录 \(w_{\mathrm{res}}^{\text{holo}}\) 供分析/消融，但不作为默认输入；
  - 更稳健的可选项是使用并集权重：\(w_{\mathrm{res}}=\max(w_{\mathrm{res}}^{\text{apo}}, w_{\mathrm{res}}^{\text{holo}})\)；推理时可用 \(w_{\mathrm{res}}=\max(w_{\mathrm{res}}^{\text{apo}}, w_{\mathrm{res}}^{\text{stage1}})\)，其中 \(w_{\mathrm{res}}^{\text{stage1}}\) 由 Stage‑1 预测 pseudo‑holo 后再计算。
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
  - 在动力学上“简单”（漂移驱动的确定性过程），
  - 同时保持几何 / 生物物理约束。

这就是 Schrödinger bridge / bridge flow 问题的条件版本：

- 最小化一个 Kullback–Leibler 或能量泛函，
- 约束端点分布为 \(p_0, p_1\)。

### 3.2 Flow Matching / Bridge Flow 近似

直接求 SB 的解析解非常困难，因此采用 **Conditional Flow Matching (CFM) / Pairwise CFM / PCFM** 的范式：

1. 构造一个参考桥 \(X_t^{\text{ref}}\)：
   - 采用简单插值的 deterministic 过程；
2. 构造对应解析速度场 \(u_t^{\text{ref}}(x)\)；
3. 通过最小化：
   \[
   \mathbb{E}_{x_0,x_1,t} \big[ \lVert v_\Theta(X_t^{\text{ref}}, t) - u_t^{\text{ref}}(X_t^{\text{ref}}) \rVert^2 \big]
   \]
   来学习一个“更简单、参数化”的向量场 \(v_\Theta\)，近似参考桥的漂移；
4. 在推理时，只积分 \(v_\Theta\) 即可。

> 实现层面上，参考桥 \(X_t^{\text{ref}}\) 采用 deterministic 插值过程，训练与推理均在确定性 ODE \(\tfrac{dx}{dt} = v_\Theta(x,t)\) 上进行。

Stage‑2 中，我们在混合空间 \(\mathcal{M}\) 上构造参考桥，并在此基础上做 CFM。

---

## 4. 参考桥的构造（刚体 + torsion）

### 4.1 χ torsion 参考桥：周期空间上的 deterministic bridge

给定端点侧链 torsion 向量 \(\chi_0, \chi_1 \in (S^1)^{K_\chi}\)：

1. 定义最短角差：
   \[
   \Delta\chi = \mathrm{wrap\_to\_\pi}(\chi_1 - \chi_0) \in (-\pi, \pi]^{K_\chi}.
   \]

2. 选取平滑插值函数 \(\gamma(t)\)（默认 smoothstep）：
   \[
   \gamma(t) = 3t^2 - 2t^3, \qquad \gamma'(t)=6t-6t^2.
   \]

3. 定义参考桥：
   \[
   \chi_t^{\text{ref}} = \chi_0 + \gamma(t) \, \Delta\chi,
   \]
   然后对每个分量再做一次 \(\mathrm{wrap\_to\_\pi}\) 映射回 \((-\pi, \pi]\)。

4. 解析速度（在欧氏近似下）：
   \[
   u_t^{\text{ref}}(\chi_t^{\text{ref}}) = \frac{d}{dt}\chi_t^{\text{ref}}
   = \gamma'(t)\,\Delta\chi.
   \]

### 4.2 刚体 SE(3) 参考桥：帧间 geodesic

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

3. 对刚体帧的参考速度可在 Lie algebra 中解析：
   - 旋转速度 \(\omega_i(t) \in \mathbb{R}^3\)：
     \[
     \omega_i^{\text{ref}}(t) = \gamma'(t)\, \Omega_i,
     \]
   - 平移速度 \(v_i^{\text{ref}}(t) = \frac{d}{dt}t_i^{\text{ref}}(t)\)。

> 速度/更新约定（必须统一）：本文默认向量场输出的是 **body‑frame（右平凡化）** 的 twist \((\omega_i, v_i)\)，离散更新采用右乘：
> \[
> R_i(t+\Delta t) = R_i(t)\,\exp(\Delta t\,\widehat{\omega_i}), \quad
> t_i(t+\Delta t) = t_i(t) + R_i(t)\,(\Delta t\, v_i).
> \]
> 这与 Stage‑1 中 `Rigid.compose(delta_rigid)` 的“局部增量右乘”语义保持一致；若实现采用 world‑frame（左平凡化）速度，需同步修改速度定义与更新公式，避免团队实现不一致导致训练/推理行为偏差。

最终，参考桥状态：
\[
X_t^{\text{ref}} = (F^{\text{ref}}(t), \chi^{\text{ref}}(t)),
\]
参考速度 \(u_t^{\text{ref}} = (\dot F^{\text{ref}}(t), \dot\chi^{\text{ref}}(t))\)。

---

## 5. 向量场学习：Conditional Flow Matching / Bridge Flow

### 5.1 条件输入与向量场形式

在任意时间 t，我们要学习一个条件向量场：

\[
 v_\Theta(x,t \mid S, L, w_{\mathrm{res}})
 = \big( \dot F_\Theta(x,t), \dot\chi_\Theta(x,t) \big),
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
5. 通过 χ‑torsion head & rigid head 输出 \(\dot\chi_\Theta, \dot F_\Theta\)，再乘以 \(g_i(t)\)。

> 在 Stage‑2 中，我们将 ESM Adapter + EdgeEmbedder + LigandConditioner + FlashIPA 视为一个几何 encoder（记为 Enc_trunk），其内部对刚体帧的更新仅作为 encoder 的辅助状态，不回写到全局状态 \(F(t)\)。全局 \(F(t)\) 的演化**只由**向量场 \(\dot F_\Theta(x,t)\) 控制，以避免“state rigids”与“encoder 内部 rigids”双重更新导致的不一致。这一点与 Stage‑1 文档中将 FlashIPA 视作几何主干、而非显式状态更新器的设定保持一致。

**实现口径（主线不做硬冻结）**：不对残基集合做硬阈值 mask/冻结。网络对所有残基输出速度分量，门控 \(g_i(t)\) 对速度做连续缩放；并用 \(L_{\text{bg}}\) 对低 \(w\)（或 \(w^{\mathrm{eff}}\)）区域显式惩罚残余速度。该“软子空间约束”避免硬 mask 的边界伪影，并允许铰链/结构域发生协同刚体运动。

### 5.2 Flow Matching 损失（带口袋加权）

对参考桥样本 \(X_t^{\text{ref}}, u_t^{\text{ref}}\)：

\[
L_{\text{FM}}
= \mathbb{E}_{b,t,\xi} \left[\sum_{i,k}
   w_{\mathrm{res},i}^\alpha \, \mathrm{mask}_{i,k}
   \big\| v_\Theta^{(i,k)}(X_t^{\text{ref}}, t) - u_t^{\text{ref},(i,k)} \big\|^2
\right],
\]

- \(\mathrm{mask}_{i,k}\) 主线默认使用 \(chi\_mask\)（仅 χ 自由度）；full‑torsion baseline 可用 \(bb\_mask\) 与 \(chi\_mask\) 组合，
- \(w_{\mathrm{res},i}^\alpha\)（\(\alpha\approx1\sim2\)）突出 pocket 自由度的重要性；
- 刚体和平移分量同样使用 \(w_{\mathrm{res}}\) 加权。

这就是一个**条件 PCFM / CFM** 目标：

- 对给定配体/序列/口袋权重条件下，学习最优漂移 \(v_\Theta\)，近似参考桥。 

> 可选增强（NMA‑guided gating / 大变构）：为避免“gate 打开但 hinge 区域没梯度”，建议将“哪里该动”的信息同时注入 gate 与 loss 权重，形成闭环。做法之一是定义
> \[
> w_i^{\mathrm{eff}}=\max\Bigl(w_{\mathrm{res},i},\ \lambda\cdot \mathrm{norm}(M_i^{\mathrm{nma}})\Bigr),
> \]
> 并在 FM / endpoint / smoothness 等 residue‑level loss 中用 \(w_i^{\mathrm{eff}}\) 替换 \(w_{\mathrm{res},i}\)（或仅在“大变构 bucket”启用）。实现时也可用更平滑的混合替代硬 `max`，并叠加 time‑decay \(\beta(t)\) 让 NMA 主要影响早期。

### 5.3 端点一致性（endpoint consistency）

为确保数值积分后确实从 apo 到 holo，再加入端点一致性项：

1. 从 \(x_0\) 出发，数值积分：
   \[
   \frac{dx(t)}{dt} = v_\Theta(x(t), t), \quad x(0) = x_0,
   \]
   得到 \(x_\Theta(1)\)。
2. 对比 \(x_\Theta(1)\) 与真实 \(x_1\)（或 Stage‑1 holo prior）：
   - χ torsion L2 / wrap‑angle loss；
   - backbone FAPE（使用 N/Cα/C 帧），对口袋加权；

\[
L_{\text{endpoint}} = \lambda_\chi \cdot \mathrm{Loss}_\chi(\chi_\Theta(1), \chi_1) + \lambda_{\text{FAPE}} \cdot \mathrm{FAPE}(x_\Theta(1), x_1).
\]

端点一致性无需每步都计算，可隔一定步数或在训练后期启用，以控制计算量。

### 5.4 背景稳定（显式约束，推荐默认开启）

仅靠 \(w_{\mathrm{res}}^\alpha\) 的 pocket‑weighted FM loss，低 \(w_{\mathrm{res}}\) 区域往往训练信号很弱：这会与第 2.3 节“非口袋区域应更稳定/更收缩”的原则产生张力（容易出现远端漂移或学成“只动 pocket 的插值器”）。

因此建议显式加入一个“背景稳定”项，在参考桥采样点上直接约束向量场幅度：

\[
L_{\text{bg}}
= \mathbb{E}_{b,t,\xi}\left[\sum_i (1-w_i)^\beta
\left(
\|\dot t_i\|^2 + \|\dot \omega_i\|^2 + \sum_k \mathrm{mask}_{i,k}\,\|\dot\chi_{i,k}\|^2
\right)\right],
\]

- \(w_i\) 默认取 \(w_{\mathrm{res},i}\)（或启用 NMA 时取 \(w_i^{\mathrm{eff}}\)）；
- \(\beta\approx1\sim2\) 用于更强地抑制非口袋区域的速度场；
- 该项与 pocket gate 是互补关系：gate 负责“哪里允许动”，\(L_{\text{bg}}\) 负责“哪里必须别乱动”。

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

### 6.3 肽键几何护栏（L_pep，推荐默认开启）

仅靠 FAPE/clash/smoothness 仍可能出现“端点合理但中间肽键几何轻微漂移”。为保证链连接一致性，建议在路径上加入最小的肽键几何护栏（对连续残基对 \(i,i{+}1\)）：

- **C–N 键长**（必需）：
  \[
  d_i^{CN}(t)=\|\mathbf x_{C,i}(t)-\mathbf x_{N,i+1}(t)\|_2,\qquad
  L_{\text{pep\_bond}}=\sum_k\sum_i \big(d_i^{CN}(t_k)-d_0^{CN}\big)^2.
  \]
- **键角**（可选但推荐）：
  \[
  L_{\text{pep\_angle}}=\sum_k\sum_i
  \Big(
  \big(\angle(C\alpha_i,C_i,N_{i+1})-\theta_0^{C\alpha C N}\big)^2+
  \big(\angle(C_i,N_{i+1},C\alpha_{i+1})-\theta_0^{C N C\alpha}\big)^2
  \Big).
  \]

总项：
\[
L_{\text{pep}}=L_{\text{pep\_bond}}+\lambda_{\text{ang}}L_{\text{pep\_angle}},
\]
其中 \((d_0^{CN},\theta_0)\) 可取 Stage‑1/模板中的理想肽键几何常数。

### 6.4 口袋接触软单调性（L_contact）

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

### 6.5 Stage‑1 holo prior 对齐（L_prior）

利用 Stage‑1 训练好的 holo decoder，将其作为 Stage‑2 在后半段的先验：

1. 对固定条件（apo backbone + ESM + 配体）运行 Stage‑1，得到其预测的 holo‑like 侧链 torsion \(\chi_{\text{stage1}}\)（以及可选的 backbone frames）；
2. 对路径上较大的 t（例如 \(t > t_\text{mid}\)，如 0.5）：

\[
L_{\text{prior}} = \mathbb{E}_{t > t_\text{mid}} \big[ w_{\mathrm{res}} \cdot d_\chi(\chi(t), \chi_{\text{stage1}})^2 \big],
\]

- \(d_\chi\) 是 wrap‑aware 的角度差度量；
- 只在后半段施加，以免过早把路径拉向单一点，保留前半段多样性。

在训练阶段（存在真实 holo 端点）时，\(L_{\text{endpoint}}\) 与 \(L_{\text{FM}}\) 仍以 \(x_1 = x_{\text{holo}}\) 为主监督，\(L_{\text{prior}}\) 建议以较小权重使用，作为后半段轨迹的平滑先验；而在推理阶段（仅有 apo + ligand 时），Stage‑1 预测的 \(\chi_{\text{stage1}}\) 及由其构造的 pseudo‑holo 端点 \(x_1'\) 则充当终点条件，对应第 8.2 节中的实际应用场景。

> 若目标覆盖显著 backbone/domain motion，可额外在 \(t>t_\text{mid}\) 对 \(F(t)\) 加一个弱 prior（例如对齐到 Stage‑1 输出的 \(F_{\text{stage1}}\) 或其他 coarse endpoint prior），与本节的 \(L_{\text{prior}}\) 互补。

这一项使 Stage‑2 在终点附近“落”到 Stage‑1 学到的 holo manifold 上，形成**高保真先验 + 连续路径**的组合。

---

## 7. 总损失与训练算法

### 7.1 总损失形式

综合上述组件：

$$
L = L_{\text{FM}} + \lambda_{\text{end}} L_{\text{endpoint}} + \lambda_{\text{bg}} L_{\text{bg}} + \lambda_{\text{geom}} \big( L_{\text{smooth}} + L_{\text{clash}} + L_{\text{pep}} + L_{\text{contact}} + L_{\text{prior}} \big)
$$

- 所有 residue‑level 项都可以再乘以 \(w_{\mathrm{res}}\) 或其幂，以强化 pocket 区域；
- \(L_{\text{bg}}\) 推荐默认开启：用 \((1-w)^\beta\) 显式抑制非口袋区域的速度场/漂移，使“口袋聚焦”与“背景稳定”在损失层面闭环一致；
- 所有 residue‑level 项可以使用组合权重 \(w_i = w_{\mathrm{res},i}^\alpha \cdot r_i\)，其中 \(r_i\) 来自 Stage‑1 χ1 offline 误差分析（例如依据《Stage‑1 工作总结与 χ1 误差分析》或 `chi1_error_analysis.py` / `chi1_error_posthoc.py` 统计得到的置信度），对高误差长尾残基略减权；
- 这样可以显式地把 Stage‑1 中对 χ1 长尾行为的认识注入 Stage‑2 的路径监督，使两阶段在权重设计上形成闭环。

### 7.2 单步训练过程（理论级别）

对一个 batch 的 apo–holo–ligand 三元组：

1. 样本转为 `Stage2Batch`：包含 \(\chi_0, \chi_1, F_0, F_1, ESM, L_{\text{tok}}, w_{\mathrm{res}}\) 等（φ/ψ/ω 可作为数据字段保留，但不作为显式状态）；
2. 采样时间 \(t \sim \mathcal{U}(0,1)\)；
3. 构造参考桥 \(X_t^{\text{ref}}, u_t^{\text{ref}}\)（torsion + SE(3)）；
4. 通过 `TorsionFlowNet` 计算 \(v_\Theta(X_t^{\text{ref}}, t)\)；
5. 计算 \(L_{\text{FM}}\)；
6. 在若干 \(t_k\)（可以与 t 相同或独立采样）上：
   - 从 \(x_0\) 或当前状态出发，积分到 \(t_k\) 得到 \(x_\Theta(t_k)\)；
   - 对 \(x_\Theta(t_k)\) 解码 FK，计算 \(L_{\text{smooth}}, L_{\text{clash}}, L_{\text{pep}}, L_{\text{contact}}, L_{\text{prior}}\)；仅在训练早期可选在 \(X_t^{\text{ref}}\) 上近似这些几何项，作为 warmup。
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
  使用固定步长的 **Heun（二阶 RK）** 积分；
- 在多个 t 处解码路径 \(x(t)\)，可视化 apo→holo 结构变化，并评估：
  - 与真实 holo 的终点误差（torsion/FAPE/pocket iRMSD）；
  - 路径上的 clash% 与 contact 单调性。

### 8.2 仅 apo + ligand（使用 Stage‑1 先验）

- 首先用与训练一致的规则，从 **apo backbone + ligand pose（apo 坐标系）** 计算 \(w_{\mathrm{res}}\)（或其并集/Stage‑1 衍生版本，见第 2.3 节）。
- 用 Stage‑1 在 (apo backbone + ESM + ligand) 条件下预测 holo‑like 侧链 torsion \(\chi_{\text{stage1}}\)（若 Stage‑1 同时输出 backbone frames，可一并取用）。
- 构造 pseudo‑holo 端点 \(x_1'=(F_1', \chi_1')\) 的几种推荐方式：
  - **局部诱导契合（默认，适用于小变构）**：\(F_1' = F_0\)，\(\chi_1' = \chi_{\text{stage1}}\)。此时 Stage‑2 主要生成 χ/局部口袋自由度的连续路径；
  - **包含 backbone 变化（面向大变构）**：若 Stage‑1 能给出 \(F_{\text{stage1}}\)，可取 \(F_1' = F_{\text{stage1}}\)；或用轻量 refinement/粗粒度先验（例如 NMA/ENM 的低频模态）给出一个 \(F_1'\) 的可行初值；
  - **折中方案（更稳健）**：构造 \(F_1'\) 时让高 \(w^{\mathrm{eff}}\) 区域承担主要的帧变化，其余区域保持接近 \(F_0\)（并由 pocket gate + \(L_{\text{bg}}\) 自动收缩/稳定）。
- 在 \(x_0\) 与 \(x_1'\) 之间运行同一 Stage‑2 桥流，得到一条“依托 Stage‑1 先验”的 apo→holo 轨迹。

> 说明：若在推理中固定 \(F_1'=F_0\)，Stage‑2 将无法表达显著的 backbone/domain motion；因此若目标覆盖“大构象变化”，需要显式提供/生成一个非平凡的 \(F_1'\)（或将问题重新定位为“口袋局部构象通路”）。

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
- 鲁棒性（主线必做）：
  - **pose 噪声敏感性曲线**：对 ligand bound pose 注入不同 RMSD 的扰动（或使用 docked pose），报告终点与路径指标（clash/contact/smooth/endpoint）的退化曲线，以明确“known pose 主线”的适用边界。

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

> 补充说明：本文以显式状态 \(x=(F,\chi)\) 上的桥流为主线；在工程实现上，也可以在 Enc_trunk 输出的 per‑residue latent \(z_i\) 上构建一个简化的 latent flow，并复用相同的几何解码器与路径正则，作为对比或过渡方案，两者在整体框架上是一致的。

---

## 10. 实现层面的硬约束与建议

本指导文档假定：

- 主线显式状态采用“去冗余”的混合形式 \(x=(F,\chi)\)（backbone frames + sidechain χ）；
- 同时提供 torsion‑only / full‑torsion 版本作为官方 baseline 与消融对照（用于稳定性/调参/对比），但不作为主线默认；
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
   - 时间积分使用 **Heun（二阶 RK）**；
   - 控制步数 T（如 20–40）与 batch size 的平衡；
   - 对刚体旋转的更新使用指数/对数映射，参考 Stage‑1 / FlashIPA 中对 SO(3) 的实现：在旋转角接近 0 时使用安全近似（如泰勒展开或对角度做 clamp），避免数值不稳定和漂移出 SO(3)。

3. **资源与调度**：
   - 几何正则（特别是 clash）成本较高，路径上的时间点数量建议 \(K\approx3\text{–}5\)，并在每个 \(t_k\) 上使用与 Stage‑1 相同的随机 chunk 采样方案 A（例如每条样本约 512 个原子对）；
   - endpoint consistency 可隔若干 step 再启用，以节省 ODE 积分开销；
   - 噪声日程：当前实现采用 \(\sigma(t)\equiv0\) 的 deterministic PCFM，以保证训练与数值积分稳定。

4. **实验规划**：
   - 在完整混合空间与完整正则框架下，仍可通过“先低 epoch、小数据子集”探索合适的权重与步数，
   - 这属于**收敛与稳定性调参**，不属于理论上的“降级”或“简化”。

本文件与 `Stage2.md` 共同构成 Stage‑2 的“理论 + 算法 + 工程规范”三件套：

- `理论与参考.md`：总体动机与两阶段框架；
- `Stage2.md`：详细算法蓝图与伪代码；
- `Stage2理论与指导.md`（本文）：在最高标准前提下的理论抽象与实现约束说明。

---

## 附录A：方案评估与迭代依据（评审视角）

> 目的：用“问题定义 → 利弊 → 可行性 → 模块衔接 → 实验设计”的顺序，给 Stage‑2 方案一个更偏 **paper/落地** 的评估版本，作为后续迭代与消融的依据。
>
> 评估对象主要来自：
> - `Stage2.md`（算法蓝图）
> - `Stage2理论与指导.md`（理论抽象与实现规范）
> - `理论与参考/NMA_Guided_Elastic_Gating.md`（NMA‑guided gating 插件）

### A1) 我理解的整体 idea（Stage‑2 到底在做什么）

你这套不是单点技巧，而是一个**两阶段 + 路径生成 + 物理先验可插拔**的完整研究方案：

- **核心问题**：不是只生成 apo/holo 两个端点，而是在 apo→holo 结合过程中生成**连续构象通路**；并且避免“直接在坐标空间学动力学”带来的噪声/对称性/不可解释性问题。
- **Stage‑2 主干**：在 \([0,1]\) 上构造条件随机过程 \(x(t)\)，端点贴近 apo/holo；模型要求 **SE(3)‑equivariant**，变化主要聚焦口袋自由度；用 **参考桥 + Conditional Flow Matching / Bridge Flow** 学时间连续向量场，并将 Stage‑1 的几何护栏（FK/FAPE/clash/contact/prior）提升到**路径级**约束。
- **状态空间选择**：优先在“去冗余”的混合空间 \(x=(F,\chi)\in \mathrm{SE(3)}^N\times(S^1)^{K_\chi}\) 上建模（并将 torsion‑only / full‑torsion 作为稳定性/对比基线）。
- **NMA‑Guided Elastic Gating 插件**：补“远离口袋但物理易动”的铰链/结构域重排短板；用 NMA(ENM) 低频模态幅值特征 \(M_i^{\mathrm{nma}}\) 作为 gate 的额外证据，做到 plug‑and‑play。

### A2) 从专业角度：这套 idea 是什么水平？

整体属于**研究型、可写成 paper 的系统方案**（不是 brainstorm）。如果按“研究成熟度（学术+工程综合）”粗略打分（10 分满分）：

- **问题定义与动机：8.5/10**（连续路径 + 条件生成 + 口袋聚焦，动机硬）
- **理论框架完整度：8/10**（桥流/CFM + 混合状态空间 + 路径级正则，闭环明确）
- **创新点密度：7.5/10（偏务实组合）**
  - 口袋接触软单调性作为路径方向性先验
  - NMA‑guided gating 作为“物理可解释 + 低侵入”补丁
- **工程可落地性：7.5/10**（模块拆分、损失、训练调度建议已经接近可开工规格）
- **主要风险：中等偏高**（不是想法飘，而是任务本身重：数据/稳定性/大变构稀缺）

一句话：这是“严肃 proposal”的水平；最终上限主要取决于数据与实验能否把每个部件打穿，而不是再堆概念。

### A3) 方案优势（为什么这条路线对路）

- **状态空间设计合理**：\(\chi\) 天然 SE(3) 不变，\(F\) 用等变模块处理，能绕开很多“坐标噪声 + 群对称性”坑。
- **路径级护栏是关键**：FM/桥流只对齐漂移不保证中间帧物理合理；把 FAPE/clash/contact/prior 抬到路径上能避免“端点像、中间乱”。
- **contact 软单调性很像 contribution**：领域先验 → 可微正则 → 路径可解释性，容易写进论文主贡献。
- **Stage‑1 prior 的定位务实**：Stage‑2 不推翻 Stage‑1，把 Stage‑1 作为 holo manifold 的软约束（尤其后半段）能显著降低落点失败风险。
- **NMA 插件定位聪明**：补 5% 大变构短板而不改主架构，工程侵入小。

### A4) 主要问题与风险（以及如何让它“真提升”）

#### 风险 1：NMA 想打的 5% 大变构，可能被“口袋加权目标”抵消

Stage‑2 的 FM loss 与多项正则都围绕口袋加权（\(w_{\mathrm{res}}\)）展开。潜在矛盾是：

- NMA‑gate 的目标：让“离口袋远但应该动”的铰链残基也动起来；
- 但训练信号主要在口袋：模型可能学到“只把口袋调好即可”，铰链区域即使 gate 打开也缺少梯度指导。

**建议：把 NMA 从“只改 gate”升级为“gate + loss 权重闭环”。**最简单可行的做法之一：

\[
w_i^{\mathrm{eff}}=\max\Bigl(w_{\mathrm{res},i},\ \lambda\cdot \mathrm{norm}(M_i^{\mathrm{nma}})\Bigr)
\]

在 **FM loss / smoothness / endpoint consistency 的 backbone 项**中用 \(w_i^{\mathrm{eff}}\) 替代 \(w_{\mathrm{res},i}\)（或仅在“大变构 bucket”启用），让“哪里该动”的信息同时喂给 gate 与损失权重，形成学习闭环。

> 实现细节上也可用更平滑的混合（避免硬 `max`），或加 time‑decay \(\beta(t)\) 让 NMA 主要影响早期。

#### 风险 2：NMA 幅值只回答“哪里容易动”，不回答“往哪动”

幅值特征非常适合决定“别锁死哪些区域”，但方向仍要靠网络学。大变构样本少时，需要靠：

- 更合理的参考桥（rotation geodesic / torsion circular interpolation）；
- 在非口袋、NMA‑high 区域给足训练信号（见风险 1）；
- 或引入更强的结构先验（例如额外的 coarse backbone regularizer / domain motion 指标）。

#### 风险 3：参考桥与真实动力学错位 → 容易学成“漂亮插值器”

如果 reference bridge 太“线性插值味”，FM 会把模型拉向插值器；而路径正则再把它往物理方向掰，权重稍不对就会拧巴（不稳/不提升）。

> 对策：把 torsion 的 circular interpolation 与旋转的 SO(3) geodesic/SLERP 写清并实现，保证参考桥与几何正则一致。

#### 风险 4：系统复杂度高，训练稳定性会是硬仗

旋转 log/exp 数值、mask/缺失残基、atom14 与 clash 子采样等，都会在“桥流 + 路径正则”下被放大。

> 对策：把“稳定性优先”的最小闭环配置写成默认配置（Heun、K≈3–5 时间点正则、endpoint consistency 稀疏启用）。

#### 风险 5：应用假设“配体 bound pose 已知”限制场景

若论文/项目定位为“已知 pose 的结构解释/构象路径生成”，这不是硬伤；若想覆盖 docking 场景，需要额外讨论 pose 误差对 \(w_{\mathrm{res}}\)、contact 单调性等先验的影响。

### A5) 可行性判断（能做出来吗？值不值得做？）

- **能落地**：模块链条清晰（条件输入 → Enc_trunk → pocket gate → heads → FM loss → 路径正则 → 采样/积分）。
- **最大不确定性**：不是“能不能跑起来”，而是“能否在 5% 大变构上拿到明确增益且不伤 95% 常规样本”。

### A6) 模块衔接验收（按接口逐段检查）

- **Stage‑1 ↔ Stage‑2：衔接强度高**。Stage‑2 把 Stage‑1 当固定 decoder/prior，并把其几何 loss 抬到路径级，是标准且稳的两阶段闭环。
- **条件信息贯穿：衔接强度高**。ESM / ligand tokens / \(w_{\mathrm{res}}\) / t 在定义与结构上贯穿模型，清晰可实现。
- **NMA‑guided gating：接口强度高，但训练信号闭环目前中等**。仅 gate 注入不足以保证大变构改善；建议配合 \(w^{\mathrm{eff}}\) 或只在大变构子集启用。
- **latent 路线 vs 显式状态路线：可并存，但需明确主线**。建议以显式 \(x=(F,\chi)\) 为主线；latent flow 作为对比/过渡/消融分支，避免“同时开两条高速路”造成叙事分散。

### A7) 最能打动审稿人的实验组织方式（建议直接照此写实验计划）

1. **按样本难度分桶：小变构 vs 大变构（必须）**
   - 小变构：不降级（至少不明显变差）
   - 大变构：显著提升（backbone/domain motion 指标更关键）
2. **NMA‑gate ablation 要做“带闭环”的**
   - baseline：原 \(w_{\mathrm{res}}\) gate
   - +NMA 仅进 gate
   - +NMA 进 gate + loss 权重闭环（\(w^{\mathrm{eff}}\)）
   - +NMA + time‑decay \(\beta(t)\)
3. **contact 单调性要单独打出来**
   - 报告 \(C(t)\) 曲线平均形状
   - 单调性违例比例
   - 以及对终点质量/路径质量的影响（避免“好看但不更准”）

### A8) 工程落地前的“护栏自检”（避免把 Stage‑1 偏差放大到路径级）

Stage‑2 会把 Stage‑1 的 FK/FAPE/clash 等当作路径正则工具箱；落地前建议先明确并验收：

- Stage‑1 的 FK/atom14 mask/clash 计算对缺失原子与 padding 的处理是否严格一致（否则路径级 clash 可能被“幽灵原子”主导）。
- 旋转/rigid 更新的 clipping 与数值稳定性策略是否符合预期（否则桥流积分可能发散）。

> 这一节是“风险提示”，不是结论；具体以实现验收与小规模 sanity check 为准。
