# BINDRAE 技术文档

> 本文档详细介绍 BINDRAE 系统的整体架构、训练流程及常见技术疑问解答，帮助理解系统设计、复杂度分析及实现细节。

---

## 目录

### Part I: 系统概述
1. [BINDRAE 是什么？](#1-bindrae-是什么)
2. [整体系统流程](#2-整体系统流程)
3. [Stage-1：配体条件化的 Holo 解码器](#3-stage-1配体条件化的-holo-解码器)
4. [Stage-2：连续构象路径生成](#4-stage-2连续构象路径生成)

### Part II: 核心技术详解
5. [流形计算：SE(3) 与 S¹ 上的几何操作](#5-流形计算se3-与-s¹-上的几何操作)
6. [训练流程与损失函数](#6-训练流程与损失函数)

### Part III: 常见问题 (FAQ)
7. [计算复杂度分析](#7-计算复杂度分析)
8. [其他常见问题](#8-其他常见问题)

---

# Part I: 系统概述

---

## 1. BINDRAE 是什么？

**BINDRAE** (Bridge-flow INDuced-fit with Rectified Auto-Encoder) 是一个**两阶段蛋白质诱导契合预测系统**，旨在：

- 给定 **apo 蛋白结构**（未结合配体）和 **配体结合位姿**
- 预测 **holo 蛋白结构**（结合配体后的构象）
- 生成从 apo 到 holo 的 **物理合理的连续过渡路径**

### 1.1 核心创新

| 创新点 | 描述 |
| ------ | ---- |
| **两阶段解耦** | Stage-1 学习"holo 应该长什么样"，Stage-2 学习"如何从 apo 走到 holo" |
| **流形上的流匹配** | 在 SE(3)×S¹ 乘积流形上进行几何正确的构象插值与生成 |
| **配体条件化** | 全程以配体位姿作为条件，引导口袋区域的构象变化 |
| **物理约束** | 路径级别的 clash、peptide 几何、接触单调性约束 |

### 1.2 与传统方法的区别

```
传统方法:
  APO + Ligand ─────────────────────────► HOLO
               （一步到位，无中间过程）

BINDRAE:
  APO + Ligand ──► Stage-1 ──► Holo Prior ──► Stage-2 ──► 连续路径 ──► HOLO
                   (学习终态)              (学习动态)    (物理合理)
```

---

## 2. 整体系统流程

### 2.1 完整训练与推理流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BINDRAE 完整系统流程                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         数据准备阶段                                  │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │   APO 结构          HOLO 结构         配体 (Ligand)                  │   │
│  │   ├── N, Cα, C      ├── N, Cα, C      ├── 3D 坐标                   │   │
│  │   ├── χ1-χ4 torsions ├── χ1-χ4 torsions ├── 原子类型                  │   │
│  │   └── ESM-2 embedding└── ESM-2 embedding └── 拓扑特征                  │   │
│  │                                                                     │   │
│  │   ↓                                                                 │   │
│  │   预处理: 计算 w_res (口袋权重), 提取 ligand tokens                    │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Stage-1 训练 (Holo 解码器)                        │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │   输入:                          输出:                               │   │
│  │   ├── APO backbone (N,Cα,C)      ├── 预测的 χ1-χ4 torsions          │   │
│  │   ├── ESM-2 表征 (1280D)         ├── 预测的刚体帧 F (Rigid)           │   │
│  │   ├── Ligand tokens              └── FK 解码的 atom14 坐标           │   │
│  │   └── w_res 口袋权重                                                │   │
│  │                                                                     │   │
│  │   损失函数:                                                          │   │
│  │   L = L_chi (χ角误差) + L_FAPE (帧对齐) + L_clash (空间冲突)          │   │
│  │                                                                     │   │
│  │   关键模块:                                                          │   │
│  │   ESM Adapter → EdgeEmbedder → FlashIPA (×3) → LigandConditioner     │   │
│  │   → TorsionHead → Forward Kinematics (FK)                           │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Stage-2 训练 (连续路径生成)                         │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │   输入:                          输出:                               │   │
│  │   ├── APO 状态 x₀ = (F₀, χ₀)     ├── 速度场 v(x,t) = (ξ, χ̇)         │   │
│  │   ├── HOLO 状态 x₁ = (F₁, χ₁)    ├── 门控 g(t) ∈ (0,1)              │   │
│  │   ├── 时间 t ∈ [0,1]             └── 积分路径 {x(t)}                 │   │
│  │   ├── Stage-1 先验 (可选)                                            │   │
│  │   └── NMA 特征 (可选)                                                │   │
│  │                                                                     │   │
│  │   损失函数:                                                          │   │
│  │   L = L_CFM (流匹配) + L_smooth (平滑) + L_clash (冲突)               │   │
│  │     + L_pep (肽几何) + L_contact (接触) + L_prior (先验)              │   │
│  │     + L_bg (背景稳定) + L_end (端点)                                  │   │
│  │                                                                     │   │
│  │   关键过程:                                                          │   │
│  │   1. 采样参考桥状态 x_ref(t)                                         │   │
│  │   2. 预测速度场 v_θ(x_ref, t)                                        │   │
│  │   3. ODE 积分生成完整路径                                            │   │
│  │   4. 计算路径级几何约束                                              │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                           推理阶段                                    │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │   输入: APO 结构 + Ligand                                            │   │
│  │                                                                     │   │
│  │   Step 1: Stage-1 预测 Holo 先验                                     │   │
│  │           (F̃₁, χ̃₁) = Stage1Model(APO, Ligand)                       │   │
│  │                                                                     │   │
│  │   Step 2: Stage-2 ODE 积分                                           │   │
│  │           for t = 0 → 1:                                            │   │
│  │               v = Stage2Model(x, t, Ligand, Stage1_prior)           │   │
│  │               x ← update(x, v, dt)   # SE(3)×S¹ 流形更新             │   │
│  │                                                                     │   │
│  │   输出: HOLO 结构 + 连续过渡路径 {x(t)}                               │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流总结

| 阶段 | 输入 | 输出 | 模型 |
| ---- | ---- | ---- | ---- |
| **数据准备** | PDB 文件 | (APO, HOLO, Ligand) 三元组 | - |
| **Stage-1 训练** | APO + Ligand | Holo 先验 (F̃, χ̃) | Stage1Model |
| **Stage-2 训练** | APO + HOLO + Ligand + t | 速度场 v(x,t) | TorsionFlowNet |
| **推理** | APO + Ligand | HOLO + 路径 | Stage1 + Stage2 |

---

## 3. Stage-1：配体条件化的 Holo 解码器

### 3.1 任务定义

Stage-1 学习的是：**给定配体位姿，蛋白质的 holo 构象应该是什么样子？**

```
                    ┌─────────────────────────────────────────┐
                    │             Stage-1 Model               │
                    ├─────────────────────────────────────────┤
                    │                                         │
  APO backbone ────►│  ESM Adapter                           │
  (N, Cα, C)        │      ↓                                 │
                    │  EdgeEmbedder (几何编码)                │
  ESM-2 表征 ──────►│      ↓                                 │────► χ1-χ4 torsions
  (1280D)           │  FlashIPA × 3 (几何推理)                │
                    │      ↓                                 │────► Rigid 帧更新
  Ligand tokens ───►│  LigandConditioner (配体条件化)         │
  (coords + types)  │      ↓                                 │────► atom14 坐标
                    │  TorsionHead + FK (解码)                │      (via FK)
                    │                                         │
                    └─────────────────────────────────────────┘
```

### 3.2 关键模块

| 模块 | 功能 | 输入维度 | 输出维度 |
| ---- | ---- | -------- | -------- |
| **ESM Adapter** | 投影 ESM-2 表征 | [B, N, 1280] | [B, N, 384] |
| **EdgeEmbedder** | 构造残基对边特征 | [B, N, 384] + coords | [B, N, N, 128] (低秩) |
| **FlashIPA** | 几何感知自注意力 | s + rigids + edges | s' + rigids' |
| **LigandConditioner** | 配体 cross-attention | s + lig_tokens | s'' |
| **TorsionHead** | 预测扭转角 | [B, N, 384] | [B, N, 7] |
| **FK Module** | 前向运动学解码 | rigids + torsions | atom14 coords |

### 3.3 损失函数

$$
\mathcal{L}_{\text{Stage-1}} = \lambda_\chi \mathcal{L}_\chi + \lambda_{\text{FAPE}} \mathcal{L}_{\text{FAPE}} + \lambda_{\text{clash}} \mathcal{L}_{\text{clash}}
$$

| 损失项 | 公式 | 作用 |
| ------ | ---- | ---- |
| **χ Loss** | $\sum_i w_i \cdot d_{S^1}(\chi_i^{\text{pred}}, \chi_i^{\text{true}})^2$ | 扭转角准确性 |
| **FAPE Loss** | Frame-Aligned Point Error | 局部坐标系下的坐标误差 |
| **Clash Loss** | $\sum_{i<j} \max(0, d_{\text{thresh}} - d_{ij})^2$ | 原子空间冲突惩罚 |

### 3.4 训练配置

```yaml
# Stage-1 典型配置
batch_size: 32
learning_rate: 2e-4
max_epochs: 100
warmup_steps: 2000

# 模型参数
c_s: 384          # 节点隐藏维度
c_p: 128          # 边维度
ipa_depth: 3      # IPA 层数
ipa_heads: 8      # 注意力头数
```

---

## 4. Stage-2：连续构象路径生成

### 4.1 任务定义

Stage-2 学习的是：**如何从 apo 构象连续变化到 holo 构象？**

这不是简单的终点预测，而是学习一个**时间连续的速度场**：

$$
\frac{dx}{dt} = v_\theta(x, t \mid \text{Ligand}, \text{Stage-1 prior})
$$

### 4.2 Conditional Flow Matching (CFM)

我们使用 **Conditional Flow Matching** 框架：

1. **定义参考桥**：给定端点 $x_0$ (apo) 和 $x_1$ (holo)，构造确定性参考路径 $x_t^{\text{ref}}$
2. **学习速度场**：训练网络 $v_\theta$ 匹配参考桥的速度
3. **推理时积分**：从 $x_0$ 出发，沿学到的速度场积分到 $x_1$

```
训练时:
                    ┌─────────────────────────────────────────┐
  x_ref(t) ────────►│                                         │
  (参考桥状态)       │         TorsionFlowNet                  │
                    │                                         │
  t ───────────────►│   预测: v_θ(x_ref, t)                   │────► v_pred
  (时间)            │                                         │
                    │   目标: v_ref = dx_ref/dt               │────► v_ref
  Ligand ──────────►│                                         │
                    │   损失: ||v_pred - v_ref||²             │
  Stage-1 prior ───►│                                         │
                    └─────────────────────────────────────────┘

推理时:
  x₀ (APO) ──► ODE积分 ──► x(0.25) ──► x(0.5) ──► x(0.75) ──► x₁ (HOLO)
               ↑              ↑           ↑           ↑
            v_θ(x,0)      v_θ(x,0.25)  v_θ(x,0.5)  v_θ(x,0.75)
```

### 4.3 参考桥构造

#### χ 角（S¹ 圆周）

```python
delta_chi = wrap_to_pi(chi1 - chi0)  # 最短角距离
gamma = 3*t**2 - 2*t**3              # 平滑插值
chi_ref = wrap_to_pi(chi0 + gamma * delta_chi)
d_chi_ref = gamma' * delta_chi       # 参考速度
```

#### 刚体帧（SE(3)）

```python
# 相对变换
Delta = F0^{-1} @ F1
xi = se3_log(Delta)  # twist 表示

# 测地线插值
F_ref = F0 @ se3_exp(gamma * xi)
d_F_ref = gamma' * xi  # body-frame 速度
```

### 4.4 路径级约束

Stage-2 不仅匹配速度，还要保证**整条路径的物理合理性**：

| 约束 | 公式 | 物理意义 |
| ---- | ---- | -------- |
| **Smoothness** | $\sum_t \|\xi(t) - \xi(t-\Delta t)\|^2$ | 路径平滑，无突变 |
| **Clash** | $\sum_t \mathcal{L}_{\text{clash}}(x(t))$ | 全程无原子冲突 |
| **Peptide Geometry** | 键长/键角约束 | 肽链几何正确 |
| **Contact Monotonicity** | $C(t) \leq C(t')$ for $t < t'$ | 蛋白-配体接触递增 |
| **Stage-1 Prior** | $\|x(t) - \tilde{x}_1\|^2$ for $t > t_{\text{mid}}$ | 后半段靠近 Stage-1 预测 |
| **Background Stability** | $(1-w_i)^\beta \|\xi_i\|^2$ | 非口袋区域保持稳定 |

### 4.5 ODE 积分方法

```python
# Heun 方法（二阶精度）
for k in range(n_steps):
    t = k * dt
    
    # Step 1: 预测器
    v1 = model(x, t)
    x_pred = update(x, v1, dt)
    
    # Step 2: 校正器
    v2 = model(x_pred, t + dt)
    v_avg = 0.5 * (v1 + v2)
    x = update(x, v_avg, dt)
```

---

# Part II: 核心技术详解

---

## 5. 流形计算：SE(3) 与 S¹ 上的几何操作

这是 BINDRAE 的核心理论创新之一。蛋白质构象变化涉及**非欧几里得空间**上的运动，我们的设计严格尊重这一几何结构。

### 5.1 状态空间：乘积流形

蛋白质构象状态 $x = (F, \chi)$ 位于乘积流形上：

$$
\mathcal{M} = \underbrace{\mathrm{SE(3)}^N}_{\text{骨架刚体帧}} \times \underbrace{(S^1)^{4N}}_{\text{侧链扭转角}}
$$

| 组件       | 流形        | 维度 | 物理意义                         |
| ---------- | ----------- | ---- | -------------------------------- |
| $F_i$    | SE(3)       | 6    | 残基 i 的局部坐标系（旋转+平移） |
| $\chi_i$ | $(S^1)^4$ | 4    | 残基 i 的 4 个侧链扭转角         |

**为什么不能用欧氏空间？**

- SE(3) 不是向量空间（旋转矩阵有正交约束）
- $S^1$ 是周期性的（$-\pi$ 和 $\pi$ 是同一点）
- 直接在 $\mathbb{R}^n$ 上插值/优化会破坏几何约束

### 5.2 SE(3) 上的计算：右三化表示

#### 5.2.1 核心思想

在 SE(3) 上，我们使用**右三化（right-trivialized）** 表示：

$$
\dot{F}_i = F_i \cdot \widehat{\xi_i}, \quad \xi_i \in \mathbb{R}^6
$$

其中 $\xi_i = (\omega_i, v_i)$ 是 **body-frame twist**：

- $\omega_i \in \mathbb{R}^3$：局部坐标系下的角速度
- $v_i \in \mathbb{R}^3$：局部坐标系下的线速度

#### 5.2.2 为什么用右三化？

| 表示方式         | 更新公式                            | 等变性 | 问题            |
| ---------------- | ----------------------------------- | ------ | --------------- |
| **左三化** | $\dot{F} = \widehat{\xi} \cdot F$ | ❌     | 全局坐标系依赖  |
| **右三化** | $\dot{F} = F \cdot \widehat{\xi}$ | ✅     | 天然 SE(3) 等变 |

**等变性证明**：若对所有残基施加全局变换 $G \in \mathrm{SE(3)}$，则 $F'_i = G \cdot F_i$，其速度：

$$
\dot{F}'_i = G \cdot \dot{F}_i = G \cdot F_i \cdot \widehat{\xi_i} = F'_i \cdot \widehat{\xi_i}
$$

twist $\xi_i$ **不变**！这意味着网络只需要预测与全局坐标系无关的局部速度。

#### 5.2.3 代码实现

```python
# src/stage2/modules/se3.py

def se3_exp(xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    指数映射：se(3) → SE(3)
    xi: [..., 6] = (omega, v)
    返回: (R, t) 刚体变换
    """
    omega, v = xi[..., :3], xi[..., 3:]
    R = so3_exp(omega)  # Rodrigues 公式
    # ... V 矩阵计算（处理小角度数值稳定性）
    t = V @ v
    return R, t

def se3_log(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    对数映射：SE(3) → se(3)（右三化）
    返回: xi = (omega, v)
    """
    omega = so3_log(R)  # 提取轴角
    # ... V_inv 矩阵计算
    v = V_inv @ t
    return torch.cat([omega, v], dim=-1)
```

### 5.3 S¹ 上的计算：圆周插值

#### 5.3.1 角度差的正确计算

扭转角 $\chi \in (-\pi, \pi]$ 位于圆周上，直接做差会产生错误：

```python
# ❌ 错误：欧氏差
delta = chi1 - chi0  # 若 chi0=-179°, chi1=179°，得到 358° 而非 -2°

# ✅ 正确：wrap 到 [-π, π]
def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

delta = wrap_to_pi(chi1 - chi0)  # 得到最短角距离
```

#### 5.3.2 参考桥插值

```python
# 训练时：sample_reference_bridge()
chi0 = batch.torsion_apo[..., 3:7]   # apo 扭转角
chi1 = batch.torsion_holo[..., 3:7]  # holo 扭转角

delta_chi = wrap_to_pi(chi1 - chi0)  # 最短角差
gamma = 3*t**2 - 2*t**3              # 平滑插值函数
chi_ref = wrap_to_pi(chi0 + gamma * delta_chi)  # 参考桥状态
```

### 5.4 参考桥构造：SE(3) geodesic 插值

#### 5.4.1 数学定义

给定 apo 帧 $F_0$ 和 holo 帧 $F_1$，参考桥定义为 SE(3) 上的测地线：

$$
F_t^{\mathrm{ref}} = F_0 \cdot \exp\bigl(\gamma(t) \cdot \log(F_0^{-1} F_1)\bigr)
$$

其中：

- $\Delta = F_0^{-1} F_1$：从 apo 到 holo 的相对变换
- $\log(\Delta) \in \mathbb{R}^6$：相对变换的 twist 表示
- $\gamma(t) = 3t^2 - 2t^3$：平滑插值（满足 $\gamma'(0) = \gamma'(1) = 0$）

#### 5.4.2 代码实现

```python
# src/stage2/training/trainer.py: sample_reference_bridge()

# 1. 获取端点帧
R0, t0 = self._rigid_to_rt(rigids_apo)   # apo
R1, t1 = self._rigid_to_rt(rigids_holo)  # holo

# 2. 计算相对变换
R0_inv, t0_inv = rigid_inverse(R0, t0)
R_delta, t_delta = rigid_compose(R0_inv, t0_inv, R1, t1)
xi = se3_log(R_delta, t_delta)  # Δ 的 twist 表示 [B, N, 6]

# 3. 测地线插值
gamma = 3*t**2 - 2*t**3
xi_t = xi * gamma  # 缩放 twist
R_inc, t_inc = se3_exp(xi_t)  # 增量变换
R_t, t_t = rigid_compose(R0, t0, R_inc, t_inc)  # F_t = F_0 ∘ exp(γ·ξ)
```

### 5.5 向量场预测与积分

#### 5.5.1 网络输出

TorsionFlowNet 预测的是**切空间速度**：

```python
# 输出
{
    "d_chi": [B, N, 4],       # χ 角速度 (rad/s)
    "d_rigid_rot": [B, N, 3], # body-frame 角速度 ω
    "d_rigid_trans": [B, N, 3], # body-frame 线速度 v
    "gate": [B, N, 1],        # 门控
}
```

#### 5.5.2 ODE 积分更新

```python
# integrate_path() 中的 Heun 积分

# 1. 在当前状态预测速度
out1 = model(chi, rigids, t=t)
d_chi1, d_rot1, d_trans1 = out1['d_chi'], out1['d_rigid_rot'], out1['d_rigid_trans']

# 2. Euler 预测
chi_pred = wrap_to_pi(chi + dt * d_chi1)
xi1 = torch.cat([d_rot1, d_trans1], dim=-1) * dt
R_inc1, t_inc1 = se3_exp(xi1)  # ← 指数映射保证在 SE(3) 上
R_pred, t_pred = rigid_compose(R_curr, t_curr, R_inc1, t_inc1)  # 右乘更新

# 3. 在预测状态再算一次速度
out2 = model(chi_pred, rigids_pred, t=t+dt)

# 4. Heun 修正
d_chi = 0.5 * (d_chi1 + d_chi2)
d_rot = 0.5 * (d_rot1 + d_rot2)
d_trans = 0.5 * (d_trans1 + d_trans2)

# 5. 最终更新
chi = wrap_to_pi(chi + dt * d_chi)
xi = torch.cat([d_rot, d_trans], dim=-1) * dt
R_inc, t_inc = se3_exp(xi)
R_new, t_new = rigid_compose(R_curr, t_curr, R_inc, t_inc)  # 右乘！
```

### 5.6 设计正确性保证

| 设计选择                               | 保证的性质   | 如果不这样做         |
| -------------------------------------- | ------------ | -------------------- |
| 右三化 twist                           | SE(3) 等变性 | 预测依赖全局坐标系   |
| 右乘更新 $F \cdot \exp(\xi \cdot dt)$ | 刚体约束     | 可能产生非正交矩阵   |
| wrap_to_pi                             | 角度连续性   | 跨越 ±π 时出现跳变 |
| 测地线参考桥                           | 最短路径     | 插值可能绕远路       |
| $\gamma(t) = 3t^2 - 2t^3$            | 端点速度为零 | 起止点突变           |

### 5.7 流形计算流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    Stage-2 流形计算流程                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  状态空间: M = SE(3)^N × (S¹)^4N                            │
│                                                             │
│  ┌─────────────┐                                            │
│  │ Apo 构象 x₀ │                                            │
│  └──────┬──────┘                                            │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────────────────────────┐                   │
│  │ for t = 0, dt, 2dt, ..., 1:         │                   │
│  │                                      │                   │
│  │   ξ, χ̇ = Network(x, t)  ← 预测切向量│                   │
│  │                                      │                   │
│  │   F ← F · exp(ξ·dt)     ← SE(3)右乘 │                   │
│  │   χ ← wrap(χ + χ̇·dt)    ← S¹ 周期  │                   │
│  │                                      │                   │
│  └──────────────────────────────────────┘                   │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                           │
│  │ Holo 构象 x₁ │                                           │
│  └──────────────┘                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 训练流程与损失函数

### 6.1 Stage-1 训练流程

```python
def train_step_stage1(batch):
    # 1. 模型前向
    outputs = model(
        esm=batch.esm,
        N=batch.N_apo, Ca=batch.Ca_apo, C=batch.C_apo,
        lig_points=batch.lig_points,
        lig_types=batch.lig_types,
        node_mask=batch.node_mask
    )
    
    # 2. FK 解码
    atom14 = fk_module(outputs['torsions'], outputs['rigids'], batch.aatype)
    
    # 3. 计算损失
    L_chi = chi_loss(outputs['chi'], batch.chi_true, batch.chi_mask)
    L_fape = fape_loss(atom14, batch.atom14_true, outputs['rigids'])
    L_clash = clash_loss(atom14)
    
    loss = L_chi + L_fape + L_clash
    
    # 4. 反向传播
    loss.backward()
    optimizer.step()
```

### 6.2 Stage-2 训练流程

```python
def train_step_stage2(batch):
    # 1. 采样时间
    t = torch.rand(batch_size)
    
    # 2. 构造参考桥状态
    chi_ref, rigids_ref, d_chi_ref, d_rigid_ref = sample_reference_bridge(batch, t)
    
    # 3. (可选) 获取 Stage-1 先验
    stage1_chi, stage1_rigids = stage1_model(batch)
    
    # 4. 预测速度场
    out = model(chi_ref, rigids_ref, t, stage1_prior=(stage1_chi, stage1_rigids))
    
    # 5. CFM 损失
    L_cfm = ((out['d_chi'] - d_chi_ref)**2).mean()
    L_cfm += ((out['d_rigid'] - d_rigid_ref)**2).mean()
    
    # 6. ODE 积分 + 几何约束
    path = integrate_path(batch, rigids_apo, chi_apo)
    L_geom = compute_path_constraints(path, batch)
    
    loss = L_cfm + L_geom
    
    # 7. 反向传播
    loss.backward()
    optimizer.step()
```

### 6.3 损失函数详解

#### Stage-2 总损失

$$
\mathcal{L} = \mathcal{L}_{\text{CFM}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{smooth}} + \lambda_{\text{clash}} \mathcal{L}_{\text{clash}} + \lambda_{\text{pep}} \mathcal{L}_{\text{pep}} + \lambda_{\text{contact}} \mathcal{L}_{\text{contact}} + \lambda_{\text{prior}} \mathcal{L}_{\text{prior}} + \lambda_{\text{bg}} \mathcal{L}_{\text{bg}} + \lambda_{\text{end}} \mathcal{L}_{\text{end}}
$$

| 损失项 | 权重 | 作用 |
| ------ | ---- | ---- |
| $\mathcal{L}_{\text{CFM}}$ | 1.0 | 匹配参考桥速度 |
| $\mathcal{L}_{\text{smooth}}$ | 0.1 | 路径平滑性 |
| $\mathcal{L}_{\text{clash}}$ | 1.0 | 原子冲突惩罚 |
| $\mathcal{L}_{\text{pep}}$ | 0.5 | 肽几何正确性 |
| $\mathcal{L}_{\text{contact}}$ | 0.1 | 接触单调性 |
| $\mathcal{L}_{\text{prior}}$ | 0.5 | Stage-1 先验对齐 |
| $\mathcal{L}_{\text{bg}}$ | 0.1 | 背景稳定性 |
| $\mathcal{L}_{\text{end}}$ | 1.0 | 端点一致性 |

---

# Part III: 常见问题 (FAQ)

---

## 7. 计算复杂度分析

### 7.1 符号定义

| 符号 | 含义          | 典型值               |
| ---- | ------------- | -------------------- |
| B    | Batch size    | 1~4                  |
| N    | 残基数        | 100~500              |
| L    | 配体 token 数 | 30~128               |
| D    | 隐藏维度      | 384                  |
| K    | IPA 层数      | 3                    |
| T    | ODE 积分步数  | 8 (Heun) / 4 (Euler) |
| G    | 几何采样步数  | 4                    |

### 7.2 Stage-2 前向传播次数

| 类型               | 组件               | 次数   | 复杂度       |
| ------------------ | ------------------ | ------ | ------------ |
| **模型前向** | TorsionFlowNet     | 1 + 2T | O(B·K·N²) |
| **FK 解码**  | Forward Kinematics | G + 2  | O(B·N)      |

**详细分解**：

```
训练单 step:
├── CFM 损失计算 ────────────────────── 1 次模型前向
│   └── 在参考桥状态预测速度场
│
├── ODE 积分 (Heun 方法) ────────────── 16 次模型前向
│   └── 每步 2 次：predictor + corrector
│   └── 共 8 步 → 8 × 2 = 16 次
│
└── 几何约束 FK ─────────────────────── 6 次 FK
    ├── Holo 参考: 1 次
    ├── 路径采样: 4 次
    └── 终点 FAPE: 1 次

总计: 17 次模型前向 + 6 次 FK
```

### 7.3 时间复杂度

**单次 TorsionFlowNet 前向**：

$$
T_{\text{forward}} = O(B \cdot K \cdot N^2) + O(B \cdot N \cdot L \cdot D)
$$

主导项：**FlashIPA 的 O(B·K·N²)**

**训练单 step**：

$$
T_{\text{step}} = (1 + 2T) \cdot T_{\text{forward}} + (G + 2) \cdot T_{\text{FK}}
$$

### 7.4 空间复杂度

| 组件          | 空间复杂度       | 典型占用         |
| ------------- | ---------------- | ---------------- |
| 模型参数      | O(D²·K)        | ~50 MB           |
| 梯度          | O(参数量)        | ~50 MB           |
| 优化器 (Adam) | 2×O(参数量)     | ~100 MB          |
| 激活值缓存    | O(B·N·D·T·K) | **1~2 GB** |
| ODE 路径      | O(T·B·N·D)    | ~200 MB          |

**峰值显存**（B=4, N=300, T=8）：**2~4 GB**

### 7.5 Stage-1 vs Stage-2 对比

| 维度               | Stage-1      | Stage-2               |
| ------------------ | ------------ | --------------------- |
| **任务**     | 静态映射     | 学习动态路径          |
| **前向次数** | 1            | 17 (Heun) / 5 (Euler) |
| **约束**     | 终点约束     | 终点 + 路径约束       |
| **典型耗时** | ~0.15 s/step | ~2.5 s/step           |

### 7.6 优化建议

| 策略                   | 效果           | 代价           |
| ---------------------- | -------------- | -------------- |
| Euler 替代 Heun        | 前向次数 17→5 | 精度略降       |
| 减少 T（8→4）         | 前向次数 17→9 | 路径分辨率降低 |
| Gradient Checkpointing | 显存降 50%     | 训练时间增 20% |

---

## 8. 其他常见问题

### 8.1 为什么需要 ODE 积分？推理时也需要吗？

**是的，推理时也需要 ODE 积分**，这是 Flow Matching 的本质特性：

```
推理流程:
输入: apo 构象 x₀
      ↓
for t = 0, dt, 2dt, ..., 1:
    v = model(x, t)      # 查询速度场
    x = x + v * dt       # 积分更新
      ↓
输出: holo 构象 x₁
```

**为什么不直接预测终点？**

| 方案                              | 优点                   | 缺点                 |
| --------------------------------- | ---------------------- | -------------------- |
| **直接预测终点**（Stage-1） | 快，1 次前向           | 无过渡构象，可能穿模 |
| **学习速度场**（Stage-2）   | 生成连续路径，物理合理 | 需要积分，稍慢       |

Stage-2 的价值在于生成**物理可解释的过渡路径**，而非仅预测终点。

### 8.2 FK 是否会成为瓶颈？

**不会**。FK 是 O(N) 的轻量操作，仅占训练时间的 ~1%：

| 指标             | FK      | FlashIPA（单层） |
| ---------------- | ------- | ---------------- |
| 时间复杂度       | O(B·N) | O(B·N²)        |
| 典型耗时         | ~10 ms  | ~50 ms           |
| 调用次数         | 6       | 17 × 3 = 51     |
| **总占比** | ~1%     | **~99%**   |

### 8.3 NMA 特征的作用与计算成本

**NMA (Normal Mode Analysis)** 是可选的物理先验特征：

- **作用**：为大变构（铰链运动）提供"物理上容易动"的信息
- **计算**：离线预计算，O(N³)，~1 秒/样本
- **当前状态**：代码已支持，默认关闭，待消融实验

| 阶段                 | 复杂度             | 成本       |
| -------------------- | ------------------ | ---------- |
| **离线预计算** | O(N³)（特征分解） | ~1 秒/样本 |
| **训练时加载** | O(B·N·K)         | 几乎为零   |
| **模型增量**   | +1% 参数           | 可忽略     |

---

## 附录：关键参数速查

```yaml
# Stage-1 配置 (src/stage1/training/config.py)
batch_size: 32
learning_rate: 2e-4
c_s: 384
c_p: 128
ipa_depth: 3

# Stage-2 配置 (src/stage2/training/config.py)
n_integration_steps: 8    # 减到 4 可加速
n_geom_steps: 4           # 路径约束采样点数
use_nma: false            # NMA 特征开关
nma_dim: 0                # 启用时设为 3~5
use_stage1_prior: true    # Stage-1 先验
```

---

*Last updated: 2026-01-13*
