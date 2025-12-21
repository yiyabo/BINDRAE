好，我直接给你一个「完整 Stage2 算法蓝图」，不是 MVP，而是**对你现有 Stage1 体系高度贴合、又有一定新意**的一整套东西。我们后面可以在这个基础上一点点 refine。

我会按这几个部分讲：

1. 我们到底选什么空间来学 apo→holo（显式坐标 vs torsion vs latent）
2. 数据准备：apo/holo 对怎么组织、特征怎么对齐
3. Stage2 模型的**状态表示 + 条件输入**
4. 具体的算法：**配体条件化的 Bridge Flow（SE(3)+χ）/ “桥流”**
5. 训练时的损失设计（包括路径上的几何正则、单调口袋接触等）
6. 推理 / 采样流程（如何从 apo 出发生成整条轨迹）
7. 和现有工作（RAE / ProteinZen / SBALIGN / DiSCO）的对比，说明咱们哪里不一样

---

## 0. 总体设计一句话版

我先用一句话概括我要给你的 Stage2：

> **在“backbone frames (SE(3)^N) + sidechain χ torsions + ESM + 配体”的条件空间里，用 *pairwise CFM/PCFM* 学一个从 apo → holo 的时间连续向量场（同时作用于 \(F(t)\) 与 \(\chi(t)\)），
> 向量场只在“口袋相关的自由度”上大幅非零，
> 并且在训练时用 Stage1 的 FK + 几何 loss 对整条路径做 FAPE / clash / 接触单调性正则。**

这个东西你可以理解成：

* 概念上靠近 ProteinZen 的“在潜空间 + SE(3) 上做 Flow Matching”
* 但我们：

  * 不在全原子坐标上做 flow，而是在 **per‑residue backbone frames（SE(3)^N）** 上做等变更新，同时只对侧链 **χ torsion（周期空间）** 建模（去冗余）；
  * 明确做的是 **配体条件化的 apo→holo 路径**，而不是无条件生成；
  * 把 Stage1 的 FK + clash/FAPE 这些几何约束“抬到整条路径上”，而不是只约束终点。

---

## 1. 为什么选「torsion 空间」来做 Stage2？

你在《理论与参考》里已经写了：解码头是「扭转角 → FK 重建」，本质上“自由度都在 torsion 里”。
这对 Stage2 是个巨大的优势：

* **SE(3) 自然不变**：角度本身不随整体转移变，省了很多 equivariant 负担；
* **维度有限**：每个残基最多 7 个 torsion，N≈300 时就是 2100 维左右，比直接在 3×N 原子坐标上好很多；
* 你已经有成熟的 `extract_torsions.py` 和 FK 模块，可以无缝沿用。

同时，RAE / ProteinZen / FoldFlow-2 这些工作都强调了“**在结构自由度主导的空间上做 Flow Matching，比在原始坐标上更稳**”：

* ProteinZen：骨架帧在 SE(3) 上做 flow，细节在 latent 里做 flow。
* FoldFlow-2：在 SE(3)-equivariant 空间里对 backbone 做 flow matching。

我们就走一个“**混合状态流（SE(3)+χ） + 几何解码器 (Stage1 FK)**”的路线：

* **主线**状态变量 \(x(t)=(F(t),\chi(t))\)：\(F\in\mathrm{SE(3)}^N\)，\(\chi\) 仅包含 χ1–χ4（φ/ψ/ω 不作为可积分显式状态，用于评估/辅助监督或从 \(F(t)\) 派生）；
* **官方 baseline/消融**：torsion‑only（仅角变量，包含 χ 或包含 full‑torsion φ/ψ/ω/χ）用于稳定性对照与消融；
* Stage1 提供：

  * FK 解码器；
  * 几何 loss（FAPE / distance / clash）和口袋权重 `w_res`；

---

## 2. 数据：如何构造 (apo, holo, ligand) 三元组

你现在 Stage1 用的是 CASF‑2016，只有 holo。Stage2 必须换一套数据源码（比如 APObind / scPDB / PDB 里手工配对），这里先给你设计“数据形态”，实现时你再去具体选库。

### 2.1 每个样本的数据结构

对每个 target（蛋白 + 配体）我们希望有：

* 一个 apo 结构 `P_apo`：未结合或弱结合构象；
* 一个 holo 结构 `P_holo`：你现在 Stage1 用的类似；
* 同一个配体 `L`（或者至少非常相似的配体），有一个参考 bound pose（通常来自 holo）。

我们为每对 (P_apo, P_holo, L) 做：

1. **序列对齐**

   * 确保两个结构能一一对应到相同残基序列；
   * 用你现在抽 torsion 时已有的“序列连续性检查”逻辑。

2. **torsion 抽取**

   * 用你现有 `extract_torsions.py`，分别得到：

     * `θ_apo ∈ R^{N×7}`（φ, ψ, ω, χ1–4）
     * `θ_holo ∈ R^{N×7}`
   * 同时记录 `bb_mask, chi_mask`（作为 Stage2 的有效性掩码）。

3. **backbone 帧（rigids）**

   * 从 apo/holo 的 N/Cα/C 构造每残基 backbone frame（例如 OpenFold/FlashIPA 风格的 `Rigid`）：
     * `F_apo ∈ SE(3)^N`
     * `F_holo ∈ SE(3)^N`

4. **配体表示**

   * 完全复用 Stage1 的 ligand 处理：

     * RemoveAllHs → 重原子坐标；
     * HBD/HBA/芳香/带电原子 → 方向探针；
     * 20D 类型/拓扑特征。

5. **口袋权重 w_res**

   * 用一个可复现的几何规则定义（残基–配体距离 + 图膨胀 + RBF soft weight 等）。
   * **推荐默认：训练/推理一致** ——统一基于 **apo backbone + ligand pose（在 apo 坐标系）** 计算 `w_res`；若训练数据同时有 holo，可额外记录 `w_res^holo` 做分析/消融。
   * 更稳健的可选项是并集权重：`w_res = max(w_res^apo, w_res^holo)`；推理时可替换为 `max(w_res^apo, w_res^stage1)`（Stage‑1 预测 pseudo‑holo 后再计算）。

6. **ESM 特征**

   * 一次性跑序列 → ESM‑2 per-res embeddings；缓存成 `esm.pt`（你已有）。

最终每个样本是：

```python
{
  "theta_apo":  [N, 7],
  "theta_holo": [N, 7],
  "bb_mask":    [N, 3],
  "chi_mask":   [N, 4],
  "rigids_apo": "Rigid[N] / SE(3)^N",
  "rigids_holo":"Rigid[N] / SE(3)^N",
  "esm_res":    [N, 1280],
  "lig_points": [M, 3],
  "lig_types":  [M, 20],
  "w_res":      [N],
  "aatype":     [N],
  ...
}
```

可以复用你现在 `IPABatch` 的很多代码路径。

---

## 3. Stage2 的状态与条件：我们到底在什么空间学 flow？

### 3.1 状态变量：torsion + mask

对任意时间 t 的状态，我们用：

* **主线显式状态**：
  * `F_t ∈ SE(3)^N`：每残基一个 backbone 刚体帧；
  * `χ_t ∈ R^{N×4}`：每残基 χ1–χ4（通过 `chi_mask` 决定哪些 χ_k 有效）；
  * χ 表示方式：用 `(sin, cos)` 展开，避免角度 wrap 问题（你 Stage1 已经这么做了）。
* **官方 baseline/消融**：也可用 torsion‑only `θ_t ∈ R^{N×7}`（φ/ψ/ω/χ1–χ4）作为角变量状态，便于做稳定性对照与消融。

mask：

* 主线默认仅使用 `chi_mask`；
* full‑torsion baseline 才使用 `bb_mask`（φ/ψ/ω）与 `chi_mask` 的组合；
* 训练时只对 mask=1 的自由度做 flow loss。

### 3.2 条件输入 cond

**cond 包含：**

1. `E_res = Adapter(ESM_res) ∈ R^{N×c_s}`：

   * 完全复用 Stage1 的 ESM Adapter（1280→384），并 **冻结**。

2. Ligand tokens：

   * `L_tok = LigandTokenEmbedding(lig_points, lig_types) ∈ R^{M×d_lig}`：
   * 完全复用 Stage1 的 ligand featurization 和 embedding。

3. Pocket weights：

   * `w_res ∈ [0,1]^N`，作为 scalar feature + loss 权重；

4. 时间 t：

   * 用一个小的 time embedding（如 sin/cos 或 MLP(t)→R^{d_t}），拼进 per-res token。

---

## 4. 核心算法：配体条件化 Bridge Flow（SE(3)+χ）

这部分就是我们真正的 **Stage2 算法**。我会先给出总体形式，再给细节。

### 4.1 Pairwise Flow Matching 背景

Flow Matching（FM / CFM / PCFM）现在非常成熟了：

* 给定起点分布 p₀（apo）和终点分布 p₁（holo），
* 你指定一个“桥”路径 x_t（比如线性插值 + 噪声），
* 然后学一个 velocity field u_ϕ(x,t) 来逼近真速度 v*(x,t) = d/dt x_t。

SBALIGN / DiSCO 则是在 SB 框架下，在蛋白/分子构象空间里学“带 prior 的桥”，本质上也是在学一个 time-dependent drift。

我们借鉴的是这类 **pairwise 条件 Flow Matching** 的技术路线，但：

* backbone 用 per‑residue SE(3) frames 建模（等变、可解释）；
* 侧链角变量默认只建模 χ（周期 manifold，去冗余）；
* 条件里塞的是 ESM + ligand + pocket；
* 并且在路径上加了你 Stage1 的几何 loss 作为 regularizer。

### 4.2 定义参考桥路径 χ_t / F_t（apo→holo）

对每个样本：

* 侧链 χ：用周期空间上的 wrap‑aware 插值/噪声桥（如下）；
* backbone frames \(F\)：用 SE(3) 上的 geodesic +（可选）Lie algebra Brownian 扰动（实现细节建议参考 `Stage2理论与指导.md`）。

下面先写 χ 的 deterministic path（先不加噪声）：

1. 先对角做最短差值（考虑 wrap）：

```python
Δχ = wrap_to_pi(χ_holo - χ_apo)  # 映射到 (-π, π]
```

2. 选择一个 scalar schedule γ(t)，比如简单的 γ(t)=t 或 smoothstep：

```python
γ(t) = t              # 简单线性
或
γ(t) = 3 t^2 - 2 t^3  # 在端点附近放缓
```

3. 定义桥路径：

```python
χ_t = χ_apo + γ(t) * Δχ  # 每个 χ 角线性插 / smooth 插
```

4. 解析真速度（target velocity）：

```python
v*(χ_t, t) = dχ_t/dt = γ'(t) * Δχ
# 若γ(t)=t，则 v* = Δχ（与 t 无关）
```

这里最大的好处：**v* 不依赖 χ_t，只依赖 Δχ**，所以 target 很干净；
在 PCFM 的框架里，这就是 textbook 的 pair-coupled velocity。

你可以后面再叠加一个小的 Gaussian 噪声项，把它推向 SB，那是后话。

### 4.3 Stage2 模型：Ligand-conditioned FlowNet（SE(3)+χ）

#### 4.3.1 输入组装

对每个时间 t 和样本：

1. **角度编码（χ‑only）**

   * `X_t = [sin χ_t, cos χ_t] ∈ R^{N×8}`；

2. **残基层 token 初始输入**

```python
h_i^0 = concat(
  Adapter(esm_i),     # [c_s]
  X_t[i],             # [8]
  w_res[i],           # [1]
  time_embed(t)       # [d_t]
)  # → R^{c_s+8+1+d_t}
```

3. **配体 token L_tok**

   * 来自 Stage1 LigandTokenEmbedding；

4. **图结构 / edge 特征（可选）**

   * 可以重用 Stage1 的 EdgeEmbedder 生成蛋白内 edge；
   * 对 FlowNet 来说不是必须，但有会更一致。

#### 4.3.2 网络结构（建议）

我们定义一个专用的 **FlowNet**（或沿用 `TorsionFlowNet` 命名，但主线包含 rigid head）：

* Backbone：

  * K 层 “蛋白–配体混合 Transformer / GNN”：

    * 每层先做 protein–ligand cross-attn（和 Stage1 LigandConditioner 类似），
    * 再做 protein–protein self-attn（或 IPA-lite，不用刚体更新）。

* 输出头：

  * **主线（χ‑only）**：对每个残基输出 χ 速度：`u_ϕ^χ(i) ∈ R^{4}`（由 `chi_mask` 决定哪些 χ_k 有效）；
  * 同时输出 backbone frame 的速度：`u_ϕ^F(i) = (ω_i, v_i) ∈ R^3 × R^3`（body‑frame twist，用于更新 \(F(t)\)）；
  * baseline/ablation 才输出 full‑torsion：`u_ϕ^\theta(i) ∈ R^{7}`（φ/ψ/ω/χ1..4）。

你可以想象这是“把 Stage1 的 LigandConditioner + 一部分 Transformer trunk 拿来当 encoder”，并在其顶部接两个 head：`dF/dt` 与 `dχ/dt`。注意：Enc_trunk 内部可能使用临时 rigids 做几何建模，但**不回写**显式状态 \(F(t)\)（state rigids 只由向量场积分更新），避免双重更新不一致。

#### 4.3.3 Flow Matching loss

对每个样本、时间 t、残基 i、角 k：

```python
L_flow = E_{(p,t)} [ Σ_{i,k} mask_{i,k} * w_res[i]^α * || u_ϕ(i,k; (F_t, χ_t), cond, t) - v*(i,k) ||^2 ]
```

* 主线 `mask_{i,k}` 只来自 `chi_mask`（full‑torsion baseline 才使用 bb_mask/chi_mask 组合）；
* `w_res[i]^α` 作为 pocket 加权（建议 α≈1 或 2），**加强口袋自由度的监督**；
* v*(i,k)=Δχ(i,k) * γ'(t) 是我们上面定义的真速度（χ 部分）。

---

## 5. 路径几何正则：把 Stage1 的 loss 抬到整条轨迹上

Flow Matching 只管“端点之间”的速度是否正确，**不保证中间状态物理上好看**。
SBALIGN / DiSCO 里就特别强调要加能量/几何约束来 regularize path。

你这里最大的武器是：**Stage1 的 FK + FAPE + distance + clash + 口袋权重**。

### 5.1 做什么？

在训练 Stage2 时，我们在若干个中间时间点 t_k（比如 3–5 个）上：

1. 用当前状态 \(x(t_k)=(F(t_k), \chi(t_k))\) 通过 **FK + aatype** 解码出 atom14 坐标（直接复用 Stage1 FK 模块；主链 torsion φ/ψ/ω 如需使用可从 \(F(t_k)\) 派生或在 baseline 中固定）；

2. 和对应的“目标几何”比：

   * 对于 backbone，可以用：

     * 两端都约束（apo & holo）：

       * 例如在 t=0.0 附近希望更接近 apo backbone，在 t=1.0 附近更接近 holo（用 FAPE / distance）；
       * 中间帧可以用 smoothed weight 在这两者之间插值。
   * 对于口袋，可以加：

     * **clash penalty**：沿用 Stage1 的随机采样方案 A（512 原子 chunk），保持路径上碰撞不过分。
     * **接触/距离软约束**：比如让 pocket 残基和 ligand 的最小距离随 t 大致单调减小（结合更紧）。

3. 整体几何正则：

```python
coords_tk = FK(F_tk, χ_tk, aatype)  # atom14 coords at time t_k
L_geom = Σ_k [
  λ_fape * FAPE_backbone(coords_tk, target_k) 
  + λ_clash * clash(coords_tk)
  + λ_cont * contact_loss(coords_tk, ligand, t_k)
]
```

这里 `target_k` 可以设计成：

* **端点附近**：靠近 apo/holo 的真实结构（用预计算的 FK θ_apo / θ_holo 对应坐标）；
* **中间**：只做 clash/接触，不强约束某个具体结构（你并不一定有真实中间状态）。

### 5.2 口袋接触的“软单调性”约束（Novel 点之一）

基于你已有的 pocket mask / w_res，我们可以设计一个挺自然的路径正则：

* 定义某个标量：

```python
C(t) = 平均_{i: w_res[i]>0.5}  soft_contact(probability of residue i contacting ligand at time t)
```

比如 soft_contact 用 logistic(dist_ij) 类似 trRosetta 的 contact logit。

* 我们希望：

  * 在 t 接近 0（apo）时，C(t) 低；
  * 在 t 接近 1（holo）时，C(t) 高；
  * 中间大致“非减”，但允许一点噪动。

可以实现一个“软单调性”loss：

```python
for k in range(K-1):
    L_mono += relu( C(t_k) - C(t_{k+1}) - ε )
```

ε 是一个小负容忍（比如 -0.02），允许轻微抖动。

**直觉**：

> 让路径在“口袋接触强度”这个 summary 上大致朝同一个方向走，
> 避免出现“先塞进去再拉出来”的鬼畜轨迹。

这点在文献里我没见有人针对 pocket contact 明说，你这里是可以写成一个小的 novelty 的。

---

## 6. 总体损失：Flow Matching + 几何正则

总的训练目标：

```python
L_total = L_flow 
        + λ_geom * L_geom
```

* `L_flow` 是 pairwise velocity 回归（是理论上的主任务）；
* `L_geom` 用 Stage1 的各类几何 loss（FAPE + clash + contact）和上面说的 C(t) 单调性作为路径正则；
* 口袋权重 w_res 出现在：

  * `L_flow` 里（让 pocket torsion 的速度更被重视）；
  * `L_geom` 里（pocket FAPE / contact / clash 权重更高）。

---

## 7. 推理 / 采样流程：从 apo 出发生成整条 holo 路径

给一个新样本（有 apo 结构 + 配体 + 序列）：

1. 数据准备：

   * 用同样的 `extract_torsions.py` 提出 `θ_apo`；
   * 用 holo 的 ligand pose（或者你的 docking pose）构造 ligand tokens；
   * 计算 `w_res`（推荐从 apo backbone + ligand pose 在 apo 坐标系下得到；必要时可用 Stage‑1 预测 pseudo‑holo 衍生 `w_res^stage1` 再取并集）。

2. 初始状态（主线）：

   * `F(0) = F_apo`；
   * `χ(0) = χ_apo`；
   * cond = {Adapter(ESM), Ligand tokens, w_res}。

3. 数值积分解 ODE：

```python
for step = 0..T-1:
    t = step / T
    ω, v_trans, v_chi = u_ϕ(F(t), χ(t), t, cond)  # SE(3)^N + [N,4]
    F(t+Δt) = update_rigid_right(F(t), ω, v_trans)
    χ(t+Δt) = wrap_to_pi(χ(t) + Δt * v_chi)
```

* 使用简单的 Euler 或 Heun / RK4；
* Δt = 1/T，比如 T=20~40。

4. 路径解码：

* 在若干 t（比如每一小步或每隔几步）用 FK 解码出 atom14；
* 你可以选择再过一遍 Stage1（用你现在的 torsion head + FK）做一个“末端 refinement”。

5. 输出：

* 全路径：`{F(t), χ(t), coords(t)}`，可视化成整条 apo→holo 动态；
* 终点 holo’：`(F(1), χ(1))` 以及解码后的坐标。

---

## 8. 和现有工作的关系 & 你这套算法的 Novelty 在哪

### 8.1 和 RAE

RAE 的思想：冻结大 encoder（DINO/SigLIP），训一个 decoder 重构，再在 latent 上做 DiT / FM。

* 你现在：

  * 冻结 ESM 当 encoder；
  * Stage1 已经是一个几何 decoder（holo 重构）；
  * Stage2 不再在“图像 latent”上做扩散，而是在 “**torsion latent**”上做 flow。
* 这跟 RAE 的 spirit 是对齐的，只不过 latent 的物理意义更强（扭转角）。

### 8.2 和 ProteinZen / FoldFlow‑2

ProteinZen：骨架 SE(3) + latent 上合并做 flow matching，生成 all‑atom。

* 相同点：

  * 用 FM 在结构自由度上做流；
  * 把复杂的全原子几何托付给单独的解码器（这里是 Stage1 FK + clash loss）。
* 不同点 / 你的亮点：

  * 他们通常是 **无条件生成 / design**，你是明确的 **apo→holo 条件路径**；
  * 状态空间选在 **torsion**，而不是 backbone frame + latent；
  * **加入配体条件**（ESM+ligand tokens+方向探针）和 pocket 权重，让流只在 pocket 相关自由度上大幅非零。

### 8.3 和 SBALIGN / DiSCO / DSB 系列

SBALIGN：在对齐的 apo/holo 结构对上，用 Schrödinger Bridge 建立连续轨迹，应用在蛋白构象变化 + docking 上。

* 他们主要在 **3D 坐标空间**上做桥，drift 是在 R^3N 里；
* 你这里在 **torsion (S¹)^K** 上做 flow，物理自由度更加精简；
* 你多了一个“**配体条件化 + 口袋单调 contact 正则**”这一层 domain 先验，这部分在现有 SB paper 里我没见有人系统做；
* 你还把 Stage1 已经训练好的 “ligand-aware holo decoder” 当成 path energy 的一部分（通过 clash / FAPE / contact loss）。

---

## 9. 这套 Stage2 方案，对你现在工程的“贴合度”和落地性

我保证这不是“空中楼阁”：

* 你**已经有**：

  * `extract_torsions.py` （可直接用于 apo/holo）；
  * ESM 缓存 pipeline；
  * ligand featurization（RemoveAllHs + 方向探针 + 20D 类型）；
  * pocket w_res 生成逻辑；
  * FK 模块 + clash / FAPE / distance loss；

* 你**只需要补**：

  1. 数据侧：一个 apo/holo dataset prepare 脚本（其实结构跟你现在 `dataprcess.md` 的 pipeline 很像，只是多了 apo 支路）。
  2. 新的 `Stage2Dataset`，返回 `rigids_apo/holo` + `torsion_apo/holo`（θ_apo/θ_holo，主线用 χ slice）+ ESM / ligand / w_res；
  3. `TorsionFlowNet` 模型实现（可以借 Stage1 的 LigandConditioner + Transformer block）；
  4. Flow Matching 的训练 loop + 几何 regularizer（调用你现有 loss 模块）。

---

如果你愿意，下一步我可以帮你把这个 Stage2 算法写成：

* 一份适合直接放进 `STAGE2_PIPELINE.md` 的方法说明；
* 再细一点，拆成伪代码 / 模块表：`Stage2Dataset`, `TorsionFlowNet`, `Trainer` 的函数签名、输入输出张量 shape 等。

这样你甚至可以按这个 blueprint 在 repo 里开 `src/stage2/` 开发，和 Stage1 的工程风格保持完全一致。

我先直接给你一版可以粘进 `STAGE2_PIPELINE.md` 的英文方法说明，然后在后半部分给出 **Stage2Dataset / TorsionFlowNet / Trainer** 的伪代码和张量 shape 规格。你可以按需要改名字/细节。

---

## 📄 一、`STAGE2_PIPELINE.md` 方法说明草稿

你可以把这一段几乎原样放到仓库里，风格上尽量跟 `BINDRAE Stage-1 Pipeline` 对齐。

---

### 1. High-level Overview

Stage‑2 in BINDRAE is designed as a **ligand‑conditioned, pocket‑gated bridge flow** that learns continuous **apo→holo** conformational pathways:

* **Task**
  Given:

  * An **apo** protein structure (unbound state).
  * A **holo** protein structure (ligand‑bound state).
  * The bound **ligand pose**.

  Learn a **time‑dependent vector field** on a hybrid state space (rigid frames + torsion angles) that transports the apo conformation to the holo conformation under ligand conditioning.

* **Inputs**

  * Frozen ESM‑2 per‑residue embeddings (same as Stage‑1).
  * Apo and holo backbone coordinates (N, Cα, C) and per‑residue torsions (φ, ψ, ω, χ1–χ4).
  * Ligand tokens: 3D coordinates of heavy atoms + **direction probes** + 20‑D type/topology features (element, aromaticity, ring/degree buckets, hetero‑neighbor counts, etc.).
  * Pocket weights `w_res` computed by a reproducible geometric rule (residue–ligand distance + graph expansion + RBF/Logistic soft weighting); recommended default is **apo backbone + ligand pose expressed in the apo frame** for train/inference consistency (optionally union with holo‑based weights when available).

* **Outputs**

  * A continuous trajectory (x(t)), (t \in [0,1]), of **hybrid states**:

    * Per‑residue rigid frames (SE(3)) for the backbone.
    * Per‑residue torsion angles (φ, ψ, ω, χ1–χ4).
  * Each (x(t)) can be decoded to full‑atom atom14 coordinates via the **same FK module and geometry losses** used in Stage‑1.

Conceptually, Stage‑1 learns **“what a holo conformation should look like given a ligand”**, while Stage‑2 learns **“how to continuously move from apo to that ligand‑conditioned holo manifold”** in a geometrically consistent way.

---

### 2. Data and Preprocessing

#### 2.1 Apo–holo–ligand triplets

Stage‑2 requires triplets where apo and holo structures can be aligned to the same sequence:

* **Apo structure** `P_apo`: unbound or weakly bound conformation.
* **Holo structure** `P_holo`: ligand‑bound conformation (similar source as Stage‑1 CASF/PDBbind complexes).
* **Ligand** `L`: bound pose from the holo complex.

For each triplet:

1. **Sequence alignment / mapping**

   * Ensure `P_apo` and `P_holo` have a consistent residue indexing after chain/sequence alignment.
2. **Torsion extraction**

   * Use the same `extract_torsions.py` pipeline as Stage‑1 to obtain torsions:

     * `torsion_apo[N,7]`, `torsion_holo[N,7]` and masks (`bb_mask`, `chi_mask`).
3. **Rigid frames**

   * From the aligned N/Cα/C coordinates of apo and holo, construct OpenFold‑style rigid frames per residue: `rigids_apo[N]`, `rigids_holo[N]`.
4. **Ligand featurization**

   * Reuse the Stage‑1 ligand pipeline:

     * RDKit `RemoveAllHs`.
     * Direction probes for HBD/HBA/aromatic/charged atoms.
     * 20‑D ligand type/topology feature vector per token.
5. **Pocket weights**

   * Compute `w_res[N]` using the same distance‑based + graph‑expansion + soft weighting used in Stage‑1, but recommended default is **apo + ligand pose (apo frame)** for train/inference consistency; when holo is available you may additionally compute `w_res^holo` for analysis or take a union.
6. **ESM features**

   * Cache ESM‑2 per‑residue embeddings once per sequence (shared by apo and holo).

#### 2.2 Stage‑2 batch structure

Each Stage‑2 training sample is packaged into a `Stage2Batch` (analogous to `IPABatch` in Stage‑1):

* `esm [B, N, d_esm]`
* `aatype [B, N]`
* `torsion_apo [B, N, 7]`, `torsion_holo [B, N, 7]`
* `bb_mask [B, N, 3]`, `chi_mask [B, N, 4]`
* `rigids_apo`, `rigids_holo` (OpenFold‑style Rigid objects or `[B, N, 4, 4]` SE(3) matrices)
* `lig_points [B, M, 3]`, `lig_types [B, M, 20]`, `lig_mask [B, M]`
* `w_res [B, N]`

---

### 3. State Representation

Stage‑2 operates on a **hybrid state** combining rigid backbone frames and sidechain torsion angles (χ):

* **Rigid frames**:
  For residue (i), a rotation (R_i(t) \in SO(3)) and translation (t_i(t) \in \mathbb{R}^3), representing an N/Cα/C frame (same convention as Stage‑1 FK).
* **Sidechain torsion angles (mainline)**:
  Per residue (\chi_i(t) = (\chi_1,\dots,\chi_4)), masked by `chi_mask`.
  Backbone torsions (\phi,\psi,\omega) are **not** treated as explicit time‑evolving state variables in the recommended mainline (they can be derived from `rigids(t)` or kept as auxiliary supervision / bookkeeping).
  Internally, torsions are represented as `(sin, cos)` pairs to avoid angle wrap issues, consistent with Stage‑1.
  (For baseline/ablation, you may also model the full 7‑tuple torsions.)

The full state at time (t) is:

[
x(t) = { \text{rigids}(t), \ \chi(t) }
]

With endpoints:

* (x(0)) from `(rigids_apo, chi_apo)` where `chi_apo` is the χ‑slice of `torsion_apo`
* (x(1)) from `(rigids_holo, chi_holo)` where `chi_holo` is the χ‑slice of `torsion_holo`

---

### 4. Model Architecture: Ligand‑Conditioned Hybrid Bridge Flow

Stage‑2 learns a **time‑dependent vector field**:

[
v_\Theta(x,t \mid \text{seq}, \text{lig}, w_{\text{res}})
= \left\{ \frac{d}{dt}\text{rigids}_{\text{state}}(t), \ \frac{d}{dt}\chi(t) \right\}
]

such that integrating this field from apo state at (t=0) yields the holo state at (t=1).

#### 4.1 Feature backbone (reusing Stage‑1 components)

For a given state (x(t)):

1. **Decode coordinates via FK**

   * **Mainline**: reconstruct atom14 coordinates from `rigids_state(t)` + `chi(t)` (+ `aatype`), where backbone torsions (φ/ψ/ω) are **not** explicit time‑evolving state variables (they can be derived from `rigids_state(t)` when needed, or kept fixed as a baseline).
   * **Baseline/ablation**: optionally use full torsions `θ(t)=(φ,ψ,ω,χ1..4)` and decode from `rigids_state(t)` + `θ(t)`.
2. **Edge features (EdgeEmbedder)**

   * Build residue–residue pair features from current Cα coordinates (RBF distances, etc.), using the Stage‑1 `EdgeEmbedder`.
3. **ESM Adapter**

   * Project frozen ESM per‑residue embeddings to the internal channel dimension `c_s` as in Stage‑1.
4. **LigandConditioner (multi‑layer)**

   * Embed ligand tokens (atoms + direction probes + 20‑D type) using `LigandTokenEmbedding`.
   * Apply protein–ligand cross‑attention + FiLM modulation to residue features **before and between** FlashIPA layers (same schedule as Stage‑1).
5. **FlashIPA stack**

   * Run a small stack (e.g. 3 layers) of FlashIPA to obtain **ligand‑aware geometric features** `h_i(t)`.
   * FlashIPA may internally update a set of encoder rigids (`rigids_enc`) for attention geometry, but these are **not written back** to the explicit state `rigids_state(t)` (single-source-of-truth for state evolution).

In other words, Stage‑2 reuses the Stage‑1 “geometry trunk” (ESM Adapter + EdgeEmbedder + LigandConditioner + FlashIPA) as an encoder of intermediate states along the apo→holo path.

#### 4.2 Pocket‑gated vector field

To focus motion on residues that are likely to move upon ligand binding, Stage‑2 uses **pocket weights** `w_res` to define a **soft gate** per residue:

[
g_i(t) = \sigma\big( \mathrm{MLP}([h_i(t), w_{\text{res},i}, \mathrm{time_embed}(t)]) \big)
]

* `g_i(t) ∈ (0,1)` indicates how much residue (i) is allowed to move at time (t).
* Pocket residues (high `w_res`) tend to have `g_i` closer to 1, non‑pocket residues closer to 0.

This gate scales the predicted velocities for both rigid frames and χ torsions, effectively making the vector field **pocket‑gated**.

#### 4.3 Velocity heads

Two heads are attached on top of `h_i(t)`:

1. **Torsion velocity head**

   * **Mainline (χ‑only)**:
     * Input: `h_i(t)`, encoded χ torsions `χ_i(t)` (sin/cos), `w_res[i]`, time embedding.
     * Output: angular velocities `dχ_i/dt ∈ ℝ⁴` (masked by `chi_mask`).
     * Loss is computed using wrap‑aware metrics (e.g. `1 − cos(Δχ)`), consistent with Stage‑1 torsion loss.
   * **Baseline/ablation (full torsion)**:
     * Output: `dθ_i/dt ∈ ℝ⁷` for (φ,ψ,ω,χ1..4).

2. **Rigid frame velocity head**

   * Output:

     * Translation velocity `dt_i/dt ∈ ℝ³`.
     * Rotation velocity in the Lie algebra `so(3)` (e.g. axis‑angle vector) for `dR_i/dt`.
   * This makes Stage‑2 similar in spirit to SE(3) flow models such as FoldFlow / ProteinZen, but now **conditioned on ligand and apo/holo endpoints**.

Both heads’ outputs are multiplied by `g_i(t)` (pocket gate) before being used in the Flow Matching loss.

---

### 5. Training Objectives

Stage‑2 is trained with a combination of **conditional flow matching** and **geometric/path regularizers**.

#### 5.1 Reference bridge and Conditional Flow Matching

For each apo–holo pair ((x_0, x_1)), a simple **reference bridge** is defined:

* **Sidechain χ torsions (mainline)**

  * Compute wrapped difference: `Δχ = wrap_to_pi(χ_holo − χ_apo)`.
  * Reference trajectory:
    [
    χ^{\text{ref}}_t = χ_0 + γ(t) Δχ + σ(t) ξ,\quad ξ \sim \mathcal{N}(0,I)
    ]
    where `γ(t)` is a smooth schedule (e.g. linear or smoothstep) and `σ(t)` a Brownian bridge‑style noise schedule vanishing at endpoints.
  * (Baseline/ablation) You may also define the bridge on full torsions `θ=(φ,ψ,ω,χ)`; in that case replace χ by θ throughout.
* **Rigid frames (SE(3))**

  * For each residue, define a geodesic on SE(3) from `rigids_apo` to `rigids_holo`, optionally adding small equivariant Brownian noise as in Schrödinger‑bridge‑style bridges.

The corresponding reference velocities (u^{\text{ref}}_t) (for torsions and frames) can be computed analytically from these interpolation formulas.

**Conditional Flow Matching loss**:

[
L_{\text{FM}} = \mathbb{E}*{b,t,ξ}
\left[
\sum*{i,k} w_{\text{res},i}^\alpha \cdot \mathrm{mask}*{i,k}
\left| v*\Theta(x^{\text{ref}}*t, t)*{{i,k}} - u^{\text{ref}}*t{}*{{i,k}} \right|^2
\right]
]

* `mask_{i,k}` from `bb_mask` / `chi_mask`.
* `w_res^α` emphasizes pocket torsions/backbone DOFs (α≈1–2).

This loss encourages the learned vector field to match the reference bridge in expectation.

#### 5.2 Endpoint consistency

To ensure that integrating the learned vector field from `x_0` indeed reaches `x_1`, an endpoint consistency term is added:

* Integrate the ODE
  ( \frac{dx(t)}{dt} = v_\Theta(x(t), t) )
  numerically from `t=0` to `t=1` starting at `x_0` to obtain `x_Θ(1)`.
* Apply:

  * **Mainline**: a χ‑level L2 / wrap‑aware loss on `(χ_Θ(1), χ_holo)` (masked by `chi_mask`).
  * (Baseline/ablation) a torsion‑level loss on full `(θ_Θ(1), θ_holo)` for `(φ,ψ,ω,χ)` if you choose to model full torsions.
  * A backbone FAPE loss between decoded coordinates from `x_Θ(1)` and true holo coordinates.

This term can be computed less frequently (e.g. every few steps) to control cost.

#### 5.3 Geometric and biophysical path regularization

For a set of intermediate times ({t_k}), states `x(t_k)` are decoded via FK to atom14 coordinates and regularized using Stage‑1’s geometric losses:

1. **FAPE smoothness**

   * FAPE between consecutive frames `(x(t_k), x(t_{k+1}))` to enforce local path smoothness.
2. **Clash penalty**

   * Apply the same **random‑chunk clash loss (scheme A)** as Stage‑1 to discourage steric clashes along the path.
3. **Pocket contact monotonicity**

   * Define a soft contact score `C(t)` between pocket residues (`w_res > 0.5`) and ligand;
   * Penalize large violations of approximate monotonic increase in `C(t)` as `t` approaches 1, encouraging physically interpretable “binding” paths.
4. **Stage‑1 prior alignment (late‑time)**

   * For time steps `t > t_mid` (e.g. >0.5), add a soft penalty that encourages `χ(t)` to approach the ligand‑conditioned holo‑like χ torsions predicted by the trained Stage‑1 decoder (acting as a fixed holo prior).
   * (Optional, for large backbone motion) also add a weak late‑time prior on `rigids_state(t)` towards a coarse endpoint (e.g. Stage‑1 predicted frames or an NMA/ENM‑based endpoint initialization).
5. **Background stability (recommended default)**

   * Add a velocity‑magnitude penalty on non‑pocket residues to suppress far‑field drift:
     \[
     L_{\text{bg}}=\sum_i (1-w_{\text{res},i})^\beta \left(\|\dot t_i\|^2+\|\dot\omega_i\|^2+\|\dot\chi_i\|^2\right).
     \]
   * This makes the “pocket‑focused motion” assumption explicit and matches the recommended mainline spec in `Stage2理论与指导.md`.

#### 5.4 Total loss

The total Stage‑2 loss is a weighted sum:

[
L = L_{\text{FM}} + \lambda_{\text{end}} L_{\text{endpoint}} + \lambda_{\text{bg}} L_{\text{bg}}

* \lambda_{\text{geom}}(L_{\text{smooth}} + L_{\text{clash}} + L_{\text{contact}} + L_{\text{prior}})
  ]

with `w_res` and χ1‑based confidence weights used whenever a residue‑level loss is computed.

---

### 6. Inference / Sampling

#### 6.1 Known apo + holo + ligand (path reconstruction / analysis)

Given `(P_apo, P_holo, L)`:

1. Build `x_0`, `x_1` as in the training pipeline.
2. Integrate the learned ODE:

[
\frac{dx}{dt} = v_\Theta(x(t), t \mid \text{cond})
]

from `t=0` to `t=1` (e.g. with an ODE solver or fixed‑step integrator).
3. Decode `x(t)` at discrete steps to atom14 structures via FK, obtaining a continuous apo→holo trajectory.

Multiple stochastic variants can be obtained by adding small noise to initial conditions or by using an SDE analog of the learned vector field.

#### 6.2 Apo + ligand only (using Stage‑1 holo prior)

In scenarios where only apo + ligand are available:

1. Use Stage‑1 as a **ligand‑conditioned holo prior**:

   * Input: apo backbone + ESM + ligand tokens.
   * Output: a plausible holo‑like sidechain torsion `χ_stage1` (and optionally a refined backbone frame field `rigids_stage1`).
2. Set `x_1` to `(rigids_apo + χ_stage1)` by default, or use `(rigids_stage1 + χ_stage1)` when available / when targeting large backbone motion.
3. Run Stage‑2 LC‑BridgeFlow between `x_0` (true apo) and this prior‑based `x_1` to obtain a plausible apo→holo path.

---

### 7. Role of Stage‑2 in the Overall BINDRAE System

Within BINDRAE:

* **Stage‑1** (ligand‑conditioned holo decoder) learns **what** holo conformations look like under ligand conditioning.
* **Stage‑2** (LC‑BridgeFlow) learns **how** to continuously transport apo conformations onto that holo manifold in a physically consistent way, with:

  * Hybrid SE(3)+torsion state space.
  * Ligand‑conditioned, pocket‑gated vector fields.
  * Geometry and contact‑aware path regularization.

---

## 🧩 二、伪代码 & 模块表（Stage2Dataset / TorsionFlowNet / Trainer）

下面是更工程化的草图，用的是 PyTorch 风格，主要目的是把 **接口和 shape** 讲清楚，方便你在 `src/stage2/` 里开工。

### 1. Stage2Dataset & Stage2Batch

```python
from dataclasses import dataclass
from torch.utils.data import Dataset
import torch

@dataclass
class Stage2Batch:
    # [B, N, d_esm]  frozen ESM-2 per-residue embeddings
    esm: torch.FloatTensor

    # [B, N]  residue type indices (0..20)
    aatype: torch.LongTensor

    # [B, N, 7]  apo/holo torsions in radians
    torsion_apo: torch.FloatTensor
    torsion_holo: torch.FloatTensor

    # [B, N, 3] backbone masks for φ, ψ, ω
    bb_mask: torch.BoolTensor
    # [B, N, 4] side-chain χ1–χ4 masks
    chi_mask: torch.BoolTensor

    # Rigid frames; could be custom OpenFold Rigid objects or [B, N, 4, 4] SE(3) matrices
    rigids_apo: object  # or torch.FloatTensor[B, N, 4, 4]
    rigids_holo: object

    # Ligand tokens
    # [B, M, 3] heavy atoms + direction probes coordinates
    lig_points: torch.FloatTensor
    # [B, M, 20] 20D type/topology features
    lig_types: torch.FloatTensor
    # [B, M]
    lig_mask: torch.BoolTensor

    # [B, N] soft pocket weights in [0,1]
    w_res: torch.FloatTensor

    # Optional metadata
    pdb_id: list[str] | None = None
    chain_id: list[str] | None = None
```

```python
class Stage2Dataset(Dataset):
    """
    Dataset of apo–holo–ligand triplets for Stage-2 bridge flow training.
    Each __getitem__ returns a Stage2Batch of batch size 1 (collate_fn stacks them).
    """
    def __init__(self, index_file: str, data_root: str):
        # index_file: JSON/CSV listing apo/holo IDs, ligand paths, etc.
        # data_root: root directory containing processed torsions/ESM/ligand/pocket files
        ...

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Stage2Batch:
        sample = self.samples[idx]

        # 1) load ESM per-residue embeddings
        esm = torch.load(sample.esm_path)["per_residue"]  # [N, d_esm]

        # 2) load torsions & masks for apo and holo
        tors_apo = np.load(sample.torsion_apo_path)  # phi, psi, omega, chi
        tors_holo = np.load(sample.torsion_holo_path)
        # pack into [N,7] tensors (phi, psi, omega, chi1..4)

        # 3) construct rigids_apo / rigids_holo from backbone N/Ca/C coords
        # using same conventions as Stage-1 FK / OpenFold rigids

        # 4) load ligand tokens: lig_points [M,3], lig_types [M,20]
        # via same ligand_utils used in Stage-1

        # 5) load pocket weights w_res [N]
        w_res = np.load(sample.w_res_path)

        # 6) aatype & masks (from torsion npz)
        aatype = tors_apo["aatype"]
        bb_mask = tors_apo["bb_mask"]    # [N,3]
        chi_mask = tors_apo["chi_mask"]  # [N,4]

        # wrap into Stage2Batch (add batch dim =1 , collate will stack)
        return Stage2Batch(
            esm=torch.from_numpy(esm).unsqueeze(0),          # [1,N,d_esm]
            aatype=torch.from_numpy(aatype).unsqueeze(0),    # [1,N]
            torsion_apo=torch.from_numpy(tors_apo["angles"]).unsqueeze(0),   # [1,N,7]
            torsion_holo=torch.from_numpy(tors_holo["angles"]).unsqueeze(0), # [1,N,7]
            bb_mask=torch.from_numpy(bb_mask).unsqueeze(0),  # [1,N,3]
            chi_mask=torch.from_numpy(chi_mask).unsqueeze(0),# [1,N,4]
            rigids_apo=rigids_apo,
            rigids_holo=rigids_holo,
            lig_points=torch.from_numpy(lig_points).unsqueeze(0),  # [1,M,3]
            lig_types=torch.from_numpy(lig_types).unsqueeze(0),    # [1,M,20]
            lig_mask=torch.from_numpy(lig_mask).unsqueeze(0),      # [1,M]
            w_res=torch.from_numpy(w_res).unsqueeze(0),            # [1,N]
            pdb_id=[sample.pdb_id],
            chain_id=[sample.chain_id],
        )
```

> collate_fn 只要把各字段在 batch 维度上 `torch.cat` 就行，Rigid 对象可以用自定义容器。

---

### 2. TorsionFlowNet（向量场网络）

TorsionFlowNet 表示刚才方法说明里的 `v_Θ(x,t|cond)`。这只是一个接口草图，你实现时可以直接复用 Stage‑1 的 Adapter / EdgeEmbedder / LigandConditioner / FlashIPA。

```python
class TorsionFlowNet(torch.nn.Module):
    """
    Ligand-conditioned, pocket-gated hybrid bridge flow:
    mainline predicts d(chi)/dt and d(rigids)/dt for a given state x(t).
    """

    def __init__(self,
                 d_esm: int = 1280,
                 c_s: int = 384,
                 d_lig_type: int = 20,
                 n_ipa_layers: int = 3):
        super().__init__()

        # 1) ESM adapter: [N, d_esm] -> [N, c_s]
        self.esm_adapter = ESMAdapter(d_esm=d_esm, c_s=c_s)

        # 2) Ligand token embedding: [M, 3+20] -> [M, d_lig]
        self.lig_embed = LigandTokenEmbedding(d_in=3 + d_lig_type,
                                              d_hidden=c_s)

        # 3) Edge embedder (same as Stage-1)
        self.edge_embedder = EdgeEmbedder(c_s=c_s, c_z=128)

        # 4) Ligand conditioner (multi-layer)
        self.ligand_conditioner = LigandConditioner(c_s=c_s,
                                                    d_lig=c_s)

        # 5) FlashIPA backbone (geometry trunk)
        self.ipa_stack = FlashIPABlockStack(
            c_s=c_s,
            c_z=128,
            n_layers=n_ipa_layers,
            # z_factor_rank=2, etc. as in Stage-1
        )

        # 6) Time embedding (e.g., sinusoidal or small MLP)
        self.time_embed = TimeEmbedding(d_time=64)

        # 7) Pocket gate MLP: [c_s + 1 + d_time] -> [1]
        self.gate_mlp = torch.nn.Sequential(
            torch.nn.Linear(c_s + 1 + 64, c_s),
            torch.nn.SiLU(),
            torch.nn.Linear(c_s, 1),
        )

        # 8) χ velocity head: [c_s + 4*2 + 1 + d_time] -> [4]
        self.chi_head = torch.nn.Sequential(
            torch.nn.Linear(c_s + 8 + 1 + 64, c_s),
            torch.nn.SiLU(),
            torch.nn.Linear(c_s, 4),
        )

        # 9) Rigid velocity head: [c_s + 1 + d_time] -> [3 + 3] (rot + trans)
        self.rigid_head = torch.nn.Sequential(
            torch.nn.Linear(c_s + 1 + 64, c_s),
            torch.nn.SiLU(),
            torch.nn.Linear(c_s, 6),  # 3 for ω (so(3)), 3 for translation
        )

    def forward(self,
                # state at time t
                chi: torch.FloatTensor,       # [B, N, 4] angles in radians (χ1..χ4, masked by chi_mask)
                rigids: object,               # Rigid[B, N] or [B, N, 4, 4]
                # static conditioning
                esm: torch.FloatTensor,       # [B, N, d_esm]
                aatype: torch.LongTensor,     # [B, N]
                lig_points: torch.FloatTensor,# [B, M, 3]
                lig_types: torch.FloatTensor, # [B, M, 20]
                lig_mask: torch.BoolTensor,   # [B, M]
                w_res: torch.FloatTensor,     # [B, N]
                # time
                t: torch.FloatTensor,         # [B] in [0,1]
                ) -> dict:
        """
        Returns:
            {
                "d_chi": [B, N, 4],         # χ angular velocities
                "d_rigid_rot": [B, N, 3],   # axis-angle velocities
                "d_rigid_trans": [B, N, 3], # translation velocities
                "gate": [B, N, 1]           # pocket gate (0..1)
            }
        """

        B, N, _ = chi.shape

        # 1) time embedding
        t_emb = self.time_embed(t)          # [B, d_time]
        t_emb = t_emb.unsqueeze(1).expand(B, N, -1)  # [B, N, d_time]

        # 2) ESM adapter
        s0 = self.esm_adapter(esm)          # [B, N, c_s]

        # 3) decode coords from (rigids_state, chi) via FK (OpenFold FK)
        # Build full torsions [B,N,7] = [phi,psi,omega (derived or fixed), chi1..4]
        torsion_full = assemble_torsion_full(
            rigids=rigids,
            chi=chi,
            # backbone torsions can be derived from rigids or taken from apo as a baseline
        )
        coords_atom14 = fk_from_torsion_and_rigid(
            rigids=rigids,
            torsion=torsion_full,
            aatype=aatype,
        )  # e.g. [B, N, 14, 3]

        # 4) Edge features from Cα coords
        ca_coords = coords_atom14[:, :, 1, :]   # assume index 1 is Cα
        z_f1, z_f2 = self.edge_embedder(s0, ca_coords)

        # 5) ligand embedding
        lig_feat = torch.cat([lig_points, lig_types], dim=-1)  # [B, M, 3+20]
        lig_tok = self.lig_embed(lig_feat)                     # [B, M, c_s]

        # 6) Ligand-conditioned features via FlashIPA stack
        s = s0
        rigids_enc = rigids  # encoder-only rigids; do NOT overwrite state rigids
        for ipa_layer in self.ipa_stack.layers:
            # ligand conditioning before each IPA
            s = self.ligand_conditioner(s, lig_tok, lig_mask)
            s, rigids_enc = ipa_layer(s, rigids_enc, z_f1, z_f2)

        h = s  # [B, N, c_s]

        # 7) pocket gate
        gate_input = torch.cat([h, w_res.unsqueeze(-1), t_emb], dim=-1)  # [B,N,c_s+1+d_time]
        gate = torch.sigmoid(self.gate_mlp(gate_input))                  # [B,N,1]

        # 8) χ velocity
        # encode χ as sin/cos
        sin_cos = torch.stack([torch.sin(chi), torch.cos(chi)], dim=-1)  # [B,N,4,2]
        sin_cos = sin_cos.view(B, N, 8)  # flatten last two dims

        chi_input = torch.cat([h, sin_cos, w_res.unsqueeze(-1), t_emb], dim=-1)
        d_chi = self.chi_head(chi_input)                    # [B,N,4]
        d_chi = d_chi * gate.squeeze(-1).unsqueeze(-1)      # pocket-gated

        # 9) rigid velocity
        rigid_input = torch.cat([h, w_res.unsqueeze(-1), t_emb], dim=-1)
        rigid_vel = self.rigid_head(rigid_input)           # [B,N,6]
        d_rot, d_trans = rigid_vel[..., :3], rigid_vel[..., 3:]  # [B,N,3],[B,N,3]
        d_rot = d_rot * gate.squeeze(-1)
        d_trans = d_trans * gate.squeeze(-1)

        return {
            "d_chi": d_chi,
            "d_rigid_rot": d_rot,
            "d_rigid_trans": d_trans,
            "gate": gate,
        }
```

---

### 3. Trainer：采样 t、构造参考桥、计算 loss

Trainer 负责：

* 采样时间 `t` 和噪声 `ξ`；
* 构建参考桥 `x_ref(t)` 和参考速度 `u_ref(t)`；
* 调用 `TorsionFlowNet` 得到预测速度；
* 用 Flow Matching + 几何正则组合总 loss 并反向。

（下面只写一个核心 `training_step` 草图，省略 optimizer / scheduler 等细节）

```python
class Stage2Trainer:
    def __init__(self,
                 model: TorsionFlowNet,
                 fk_module,
                 loss_weights,
                 device: str = "cuda"):
        self.model = model.to(device)
        self.fk = fk_module
        self.w = loss_weights
        self.device = device
        # optimizer, scheduler, etc.

    def sample_reference_bridge(self, batch: Stage2Batch, t: torch.FloatTensor):
        """
        Build reference states x_ref(t) and velocities u_ref(t)
        for both chi and rigids, given apo/holo endpoints.

        Inputs:
            batch: Stage2Batch
            t: [B] sampled in (0,1)

        Returns:
            chi_ref: [B,N,4]
            rigids_ref:  Rigid[B,N]
            d_chi_ref: [B,N,4]
            d_rigid_rot_ref: [B,N,3]
            d_rigid_trans_ref: [B,N,3]
        """
        # 1) unpack endpoints
        # torsion_apo/holo stores full (phi, psi, omega, chi1..4); mainline uses chi slice
        chi0 = batch.torsion_apo[..., 3:7].to(self.device)   # [B,N,4]
        chi1 = batch.torsion_holo[..., 3:7].to(self.device)  # [B,N,4]
        R0 = batch.rigids_apo    # Rigid[B,N] or [B,N,4,4]
        R1 = batch.rigids_holo

        # 2) chi reference
        # wrap angle difference to (-pi,pi]
        delta_chi = wrap_to_pi(chi1 - chi0)

        # gamma(t) schedule (e.g. linear)
        gamma = t.view(-1, 1, 1)  # [B,1,1]
        chi_ref = chi0 + gamma * delta_chi            # [B,N,4]

        # additive noise (optional)
        # sigma(t) ~ lambda * sqrt(t(1-t))
        # ...

        # derivative wrt t
        d_chi_ref = delta_chi       # if gamma(t)=t

        # 3) rigid reference (geodesic on SE(3))
        rigids_ref, d_rot_ref, d_trans_ref = \
            se3_geodesic_bridge(R0, R1, t)  # shapes [B,N] and [B,N,3], [B,N,3]

        return chi_ref, rigids_ref, d_chi_ref, d_rot_ref, d_trans_ref

    def training_step(self, batch: Stage2Batch) -> torch.Tensor:
        batch = move_batch_to_device(batch, self.device)

        B, N, _ = batch.torsion_apo.shape

        # 1) sample random time t in (0,1)
        t = torch.rand(B, device=self.device)

        # 2) reference bridge states & velocities
        (chi_ref, rigids_ref,
         d_chi_ref, d_rot_ref, d_trans_ref) = self.sample_reference_bridge(batch, t)

        # 3) forward pass: predict velocities at x_ref(t)
        out = self.model(
            chi=chi_ref,                # [B,N,4]
            rigids=rigids_ref,
            esm=batch.esm,               # [B,N,d_esm]
            aatype=batch.aatype,         # [B,N]
            lig_points=batch.lig_points, # [B,M,3]
            lig_types=batch.lig_types,   # [B,M,20]
            lig_mask=batch.lig_mask,     # [B,M]
            w_res=batch.w_res,           # [B,N]
            t=t,                         # [B]
        )

        d_chi_pred = out["d_chi"]            # [B,N,4]
        d_rot_pred   = out["d_rigid_rot"]    # [B,N,3]
        d_trans_pred = out["d_rigid_trans"]  # [B,N,3]

        # 4) Flow Matching loss (chi + rigid), pocket-weighted
        w_res = batch.w_res.unsqueeze(-1)    # [B,N,1]
        chi_mask = batch.chi_mask.to(self.device).float()  # [B,N,4]

        fm_chi = ((d_chi_pred - d_chi_ref) ** 2) * w_res * chi_mask
        L_fm_chi = fm_chi.sum() / (chi_mask.sum() + 1e-8)

        fm_rot = ((d_rot_pred - d_rot_ref) ** 2) * w_res
        fm_trans = ((d_trans_pred - d_trans_ref) ** 2) * w_res
        L_fm_rigid = (fm_rot.sum() + fm_trans.sum()) / (w_res.sum() + 1e-8)

        L_FM = self.w["fm_chi"] * L_fm_chi + self.w["fm_rigid"] * L_fm_rigid

        # 5) geometry/path regularization at one or more time points
        # Decode coords from (rigids_ref, chi_ref) via FK
        torsion_full = assemble_torsion_full(rigids=rigids_ref, chi=chi_ref)
        coords_ref = self.fk(rigids_ref, torsion_full, batch.aatype)  # [B,N,14,3]

        # Compute FAPE smoothness, clash, contact monotonicity, Stage-1 prior regularization...
        L_geom = compute_geometry_losses(
            coords_ref=coords_ref,
            chi_ref=chi_ref,
            batch=batch,
            t=t,
        )

        # 6) Optional: endpoint consistency (not shown here to keep it short)

        loss = L_FM + self.w["geom"] * L_geom

        loss.backward()
        # optimizer.step(), zero_grad(), etc.
        return loss
```

`compute_geometry_losses` 里面可以直接调用你现在 Stage‑1 的 FAPE / clash / distance 实现（随机子采样 clash 方案 A），外加一个 pocket‑ligand contact monotonicity term和 Stage‑1 prior term：

* `L_smooth`: 用 FAPE 或 Cα L2 在相邻 t 上做 path smoothness；
* `L_clash`: 调用 Stage‑1 `clash_penalty`；
* `L_contact`: 根据 ligand–pocket soft contact score 的非减约束计算；
* `L_prior`: 对 t>0.5 的帧，让 torsion 靠近 Stage‑1 在同样条件下输出的 holo torsion 分布。

---

如果你愿意，下一步我们可以针对你实际代码结构，把这些伪代码拆成真模块（比如 `src/stage2/models/torsion_flow.py`, `src/stage2/datasets/dataset_bridge.py`, `src/stage2/training/trainer.py`），直接给出更完整的 skeleton。

---

## 附录：顶会/顶刊审稿人 Checklist（用于论文叙事与实验打穿）

> 目的：把“方法闭环”进一步升级成“论文级不可替代贡献 + 可复现实验铁证 + 风险点提前堵死”的执行清单。
> 口径以本文主线 `x(t)=(F(t),χ(t))` 为准（full‑torsion 7D 仅作为 baseline/ablation）。

### 1) 贡献（必须钉成 2–3 条不可替代陈述）

审稿人最常见质疑是“工程拼装”。你需要把贡献写成可验收的 claim（并对应实验表格）：

1. **Path-level biophysical constraints as training-time control**：不是只优化终点，而是把 `clash/contact/smoothness` 提升为路径级可控先验，并量化路径质量指标（而非只报终点误差）。
2. **Pocket-gated vector field + background stability (`L_bg`)**：`gate` 负责“哪里允许动”，`L_bg` 负责“哪里必须别乱动”，并证明两者缺一不可（因果消融）。
3. **Apo+ligand-only inference as a first-class mode**：把真实使用场景写进系统闭环（Stage‑1 生成 pseudo‑holo endpoint，Stage‑2 生成通路），而不是只在“已知 holo”里自洽。

> 写论文时建议用“我们解锁了一个以前做不到的任务/指标”的表述，而不是“更优雅”。

### 2) 数据与划分（决定可接受性的硬门槛）

顶会很常见的拒稿理由：数据构建不透明/不严谨 → 结果不可复现或存在泄漏。建议在方法/附录中强制给出：

- **去冗余 split（至少 protein split）**：按序列同源（如 30%/40% identity）做 train/val/test；数据量允许的话，再加 ligand scaffold split（或至少作为补充分析）。
- **apo/holo pair 质量过滤**：缺失残基、链不一致、对齐失败、配体不一致/过度类似物导致的伪配对要筛掉；否则桥流在学 label noise。
- **分桶评估**：按 apo→holo backbone 变化程度（例如 RMSD / domain motion 指标）分 `small` vs `large` bucket，并分别报告终点+路径指标。

### 3) 路径的含义（提前收敛 claim，避免被问死）

你要主动写清楚：

- **不宣称真实动力学时间**；宣称的是“在几何与接触先验约束下的可解释构象通路（geometrically & sterically plausible transition pathway）”。
- 所有 claim 必须落在可度量指标上：平均 clash%、contact 单调性违例比例、路径平滑度（FAPE/Cα smoothness）、终点误差（pocket iRMSD/FAPE 等）。

### 4) 最容易翻车的点（必须提前堵）

#### A. 口径一致性（主线 vs baseline）

- 主线固定：`(F,χ)`（χ‑only 4D，mask 控制）。
- full‑torsion 7D 只能作为 ablation（appendix），不要在主线实现/接口里混成两套“主线”。

#### B. Gate 的因果性（避免被喷成 heuristic）

必须做最直接的因果消融矩阵（并同时报终点+路径指标）：

1. `gate ≡ 1`（无 gate）
2. `gate = gate(w_res)`（不看 h_i(t)）
3. `gate = gate(h_i(t))`（不看 w_res）
4. `gate + L_bg` vs `gate-only`

并报告：`gate` 是否真的学到“口袋运动学”，而不是把 loss 权重硬编码了一遍。

#### C. 已知 bound pose 假设（把限制变成亮点）

如果任务定位为“已知 pose 的结构解释/路径生成”，请写清楚；同时建议补一个 **pose 噪声敏感性曲线**：

- 对 ligand pose 注入不同 RMSD 的扰动（或用 docked pose），报告终点与路径指标怎么掉；
- 展示 `gate + L_bg + path regs` 对 pose noise 更稳/更可控。

### 5) 顶会级实验包（按优先级）

#### P0（不做基本会被拒）：核心对照 + 消融矩阵

至少四个 baseline（并报终点 + 路径指标）：

1. **Linear interpolation**（rigid geodesic + χ circular interpolation）
2. **Stage‑1 直接预测 holo**（只有终点，无路径）
3. **Stage‑2 (FM only)**（无几何正则/无 prior）
4. **Stage‑2 (FM + path regs + gate + L_bg + late prior)**（全量主方法）

关键消融（建议做成表格矩阵）：

- 去掉 `L_contact`（你把它定义成重要 novel prior，就必须单独打）
- 去掉 `L_prior`（late-time Stage‑1 prior）
- 关掉 rigid head（只动 χ）vs 全开
- χ‑only vs 7D torsion（appendix）

#### P1（增强说服力）：小变构 vs 大变构分桶

- 分桶后分别报告提升/退化；
- NMA‑guided gating 建议只在 `large` bucket 启用，并做闭环（`w_eff`/loss reweighting + time‑decay）验证。

#### P2（Nature 更吃）：2–3 个 case study

- 画 `C(t)`、clash 曲线、关键残基 χ1 翻转时刻；
- 给结构帧序列可视化，讲“机制解释价值”（而不是数值游戏）。

### 6) 复现稳定性（默认配置要写进主方法，不要只放附录）

建议把“稳定性最小闭环配置”写成默认：

- ODE 积分：Heun/RK4（优先于纯 Euler）
- 路径正则时间点数：`K≈3–5`（clash 子采样方案 A），endpoint consistency 稀疏启用
- 旋转 log/exp 数值稳定、mask/padding/atom14 处理一致性（否则路径级 clash 会被放大）

### 7) Rebuttal 预案（审稿人常问三连）

1. **“你是不是工程拼装？”** → 用第 1 节三条 contribution + 对应消融表格回答
2. **“数据是否泄漏/挑样本？”** → 用第 2 节 split/过滤规则与统计表回答
3. **“路径有什么物理意义？”** → 用第 3 节 claim 收敛 + 路径指标曲线回答
