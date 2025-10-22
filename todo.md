# 配体条件下的蛋白质**动态构象生成**项目手册（v1.0）

> 研究口径（第一性原理）：
> **在连续潜空间 (\mathcal{Z})**（由 ESM‑3 语义 + 几何分支融合得到）**学习从 apo→holo 的时间连续运输（flow/bridge）**；
> **在结构空间 (\mathcal{X})** 用**解码器**将任意时刻的潜变量 (Z_p(t)) 还原为可检验的 3D 构象；
> 通过**几何/化学约束**与**能量或打分蒸馏**，保证路径的物理合理性与可用性。

---

## 0. 产出与边界

* 产出

  1. 一套可复现实验管线（数据→潜空间→连续生成→解码→评测）。
  2. **动态轨迹生成器**（Flow Matching 主干，SB 为可插拔正则）。
  3. 端到端评测：端点（apo/holo）与路径（中间态）的指标。
  4. 论文草稿的**方法与实验章节**“可直接落笔”的素材（公式/伪代码/图示建议）。
* 非目标

  * 本版不解决**长时间尺度热力学**的严格一致性（后续以SB + 能量蒸馏增强）。
  * 暂不做**端到端配体生成**（先做“已知配体条件下”动态构象；协同生成作为扩展）。

---

## 1. 问题定义与符号

**目标**：给定蛋白的 apo 结构与配体，生成一条**从 apo 到 holo 的连续构象路径** ({X_p(t)}_{t\in[0,1]})。

* 编码器

  * 蛋白：(Z_p^{(0)}=E_p(\text{apo}))，(Z_p^{(1)}=E_p(\text{holo}))。
  * 配体：(Z_l=E_l(\text{lig}))。
* 潜空间：(\mathcal{Z}=\mathcal{Z}_p\times \mathcal{Z}_l) 的直积流形，包含

  * 欧氏块 (\mathbb{R}^m)（集体变量/语义嵌入），
  * 角度块 ((S^1)^K)（主链/侧链扭转），
  * 旋转块 (SO(3))（可选的局部刚体自由度）。
* 运输模型

  * **FM**：(\dot Z_p(t)=v_\theta(Z_p(t),t;Z_l))（确定性 ODE）。
  * **SB**：(dZ_p(t)=v_\theta,dt+\sigma(t),dW_t)（随机 SDE）。
* 解码：(X_p(t)=D(Z_p(t)))（内部坐标 + FK 或等变坐标头）。

**不等式目标（直观）**：端点分布匹配 + 路径几何合理
[
Z_p(0)\sim P_0,\quad Z_p(1)\approx P_1(\cdot\mid Z_l),\quad \text{且 }X_p(t)\text{ 满足几何/化学先验。}
]

---

## 2. 系统总览与数据流

1. 数据制备：((\text{apo},\ \text{lig},\ \text{holo})) 三元组 + 质量过滤 + 标注（接触、口袋）。
2. 编码：ESM‑3 残基层语义令牌 + 几何分支（GVP/EGNN）→ 融合 → (Z_p)；配体图/3D → (Z_l)。
3. **RAE 预训练**：训练解码器 (D) 保障 (Z\to X) 的**可逆性/可还原性**。
4. **桥模型训练**：在 (\mathcal{Z}) 上学流/桥（FM 主 + SB 辅），得到 (v_\theta)。
5. 采样与解码：从 (Z_p(0)) 积分至 (Z_p(1))，沿途解码为 (X_p(t))。
6. 评测：端点+路径指标；可选 MD 短程验证。

---

## 3. 数据集与清洗

### 3.1 三元组构建

* 同一蛋白的 **apo 结构** 与 **holo 复合体**（holo 中抽取配体）。
* Ligand 3D 标准化：质子化状态、立体中心、扭转键标记、重复体/重复构象去重。
* 分辨率与质量门槛：X‑ray 分辨率 ≤ 3.0 Å；去除大缺失残基；统一编号与链标识。

### 3.2 同源去泄漏的划分

* **序列相似度聚类**（如 30%/40% 两档）→ 家族分层拆分 train/val/test。
* 按 **配体骨架**（Bemis–Murcko）再做分布平衡，避免配体信息泄漏。

### 3.3 标签与辅助特征

* 口袋掩码、界面残基、残基层接触（距离阈值 8 Å）。
* 可选导出：主链/侧链 torsion 角标签，距离图 (d_{ij})、SASA、二级结构。

---

## 4. 表示学习（Encoder 与融合）

### 4.1 蛋白编码器 (E_p)

* **语义**：ESM‑3 主体冻结；倒数 1–2 层插入 **LoRA/Adapter**（小幅对齐 PPI/口袋域）。
* **几何分支**（推荐 GVP‑GNN 起步）：

  * 图构建：原子级节点 + 共价边 + 半径 (r=8) Å 非键邻域；
  * 节点特征：原子类、SASA、部分电荷、局部参考系坐标；
  * 边特征：距离、单位向量、氢键/盐桥指示；
  * 残基池化：原子 → 残基层局部几何令牌 (\mathbf{g}_i)。
* **融合**（残基层配对）：

  * (\mathbf{z}_i=W[\mathbf{e}_i|\mathbf{g}_i]) 或 Gate/FiLM 或 1–2 层跨注意力；
  * 得到 (\mathbf{Z}*p={\mathbf{z}*i}*{i=1}^{N*{\text{res}}})，维度 (d_z\sim 256\text{–}512)。

### 4.2 配体编码器 (E_l)

* **条件生成模式（A）**：配体图/3D → **欧氏嵌入** (\in\mathbb{R}^{d_l}) 作为**条件**。
* **协同生成模式（B）**：将配体自由度显式参数化为 (\mathbb{R}^3\times SO(3)\times (S^1)^K)，同时作为被生成变量。

---

## 5. 解码器 (D)（结构还原）

### 5.1 内部坐标 + 前向运动学（FK）

* 预测每个残基的 (\hat\phi,\hat\psi,\hat\omega\in S^1) 与 (\hat{\boldsymbol{\chi}}\in (S^1)^{K_i})，**wrap** 差值到 ([-\pi,\pi))。
* 通过 FK 还原原子坐标（用固定键长/角 + 扭转角驱动）。

### 5.2 几何不变损失

* **Torsion**：(\mathcal{L}*\text{torsion}=\sum \mathrm{wrap}*{2\pi}(\hat\theta-\theta)^2)。
* **FAPE**：在每个残基层局部参考系对齐的点误差（平移/旋转不敏感）。
* **距离图**：(\mathcal{L}*\text{dist}=\sum(\hat d*{ij}-d_{ij})^2)。
* **可选**：接触 BCE、密度/表面体素 L2 + TV（后期增强）。

### 5.3 RAE 稳定技巧

* **解码器噪声增强**：训练时对 (Z) 加小噪（欧氏高斯 / von‑Mises / 矩阵Fisher），让 (D) 适应生成时的带噪潜变量。
* **维度依赖时间表偏移**：时间嵌入 (t\mapsto t_m=\frac{\alpha t}{1+(\alpha-1)t},\ \alpha=\sqrt{m/n})（潜令牌总维 (m) 对低维基表 (n) 的缩放）。
* **宽度≥维度**：用于去噪/流场的通道宽 (d\ge \text{latent维})，避免容量瓶颈。

---

## 6. 生成器 (v_\theta)（在潜空间学习“怎么走”）

### 6.1 Flow Matching（主干）

* **训练样本**：((Z_p^{(0)},Z_p^{(1)},Z_l))；采样 (t\sim U(0,1))。
* **geodesic+noise 中间态** (z_t)：

  * 欧氏：(z_t=(1-t)z_0+t z_1+\sigma(t)\varepsilon)；
  * (S^1)：(\theta_t=\mathrm{wrap}(\theta_0+t,\mathrm{wrap}(\theta_1-\theta_0))+\eta_t)；
  * (SO(3))：(R_t=R_0\exp\big(t\log(R_0^\top R_1)\big),\Xi_t)。
* **“真速度”**：欧氏 (z_1-z_0)；(S^1) 常角速度；(SO(3)) (\dot R_t=R_t A, A=\log(R_0^\top R_1))。
* **目标**：(\min_\theta \mathbb{E}|v_\theta(z_t,t;Z_l)-\mathbb{E}[\dot z_t\mid z_t,t,Z_l]|^2)。
* **网络**：Transformer/GNN 混合主干 + 时间/条件注入（FiLM/跨注意力）；必要时加**浅而宽的头**增宽通道。

### 6.2 Schrödinger Bridge（可插拔正则）

* 在参考扩散上最小化路径 KL，学习“最经济”的随机桥；加入**边际匹配/路径 KL/score matching**项。
* 价值：改善多模态路径与热力学一致性（作为 FM 的正则层叠，训练仍稳定）。

### 6.3 几何/化学与能量引导（可选增强）

* **几何正则**：无碰撞（min‑dist penalty）、键长角度/价态先验、H‑bond/疏水互补奖励。
* **能量/力蒸馏**：训练能量头 (E_\phi) 或蒸馏自快速打分器，对速度场加
  (|v_\theta+\lambda,\Pi_T\nabla_z E_\phi|^2)（(\Pi_T) 为流形切空间投影）。

---

## 7. 目标函数与权重（最小可用 + 增强）

**最小可用版本**
[
\mathcal{L}*\text{rec}=\lambda_1\mathcal{L}*\text{torsion}+\lambda_2\mathcal{L}*\text{FAPE}+\lambda_3\mathcal{L}*\text{dist}
]
[
\mathcal{L}*\text{bridge}=\mathcal{L}*\text{FM}(v_\theta;z_t,\dot z_t)
]
[
\mathcal{L}*\text{total}=\mathcal{L}*\text{rec}(Z_p^{(0)})+\mathcal{L}*\text{rec}(Z_p^{(1)})+\alpha,\mathcal{L}*\text{bridge}
]

**增强项（逐步打开）**

* * 接触 (\lambda_4\mathcal{L}_\text{contact})；
* * SB 边际/路径 KL：(\beta,\mathcal{L}_\text{SB})；
* * 几何/能量：(\gamma,\mathcal{L}*\text{geom}+\eta,\mathcal{L}*\text{force})。

**起始权重建议**

* (\lambda_1:\lambda_2:\lambda_3=1:1:0.1)；(\alpha=1)。增强项视稳定性逐步开到 (\beta,\gamma,\eta\in[0.1,1])。

---

## 8. 张量形状与自由度约定

* 批大小：(B)；残基数：(N_r)；配体原子数：(N_l)。
* 蛋白潜令牌：(\mathbf{Z}_p\in\mathbb{R}^{B\times N_r\times d_z})。
* 配体条件嵌入：(\mathbf{Z}_l\in\mathbb{R}^{B\times d_l})（模式 A）。
* 扭转角：(\Theta\in\mathbb{R}^{B\times N_r\times K_{\max}})（按残基掩码）。
* 局部旋转（可选）：(R\in SO(3)^{B\times N_\text{block}})；使用旋转向量或四元数参数化（数值稳定）。
* geodesic 噪声：

  * 欧氏：(\varepsilon\sim\mathcal{N}(0,I))；
  * (S^1)：(\eta\sim\text{von‑Mises}(\kappa))；
  * (SO(3))：(\Xi\sim\text{Matrix‑Fisher}(\mathbf{F}))。

---

## 9. 训练流程（逐步落地）

### 9.1 阶段 1：RAE 预训练（2–3 周）

* 冻结 ESM‑3 主体，仅训 LoRA/Adapter + 解码器。
* 目标：(\mathcal{L}_\text{rec})（扭转 + FAPE + 距离），解码器噪声增强**开启**。
* 早停标准：val FAPE 与 torsion RMSE 稳定下降；碰撞率 < 基线。

### 9.2 阶段 2：桥模型（3–4 周）

* 采用 FM 主干训练 (v_\theta)；中间态用 geodesic+noise 采样；时间表做**维度偏移**。
* 逐步加入几何无碰撞与软价态正则；日志中记录训练时重构出的中间帧。

### 9.3 阶段 3：小幅联合微调（3–4 周）

* 解冻 LoRA/Adapter 与解码器一起**小步**更新；
* 视稳定情况加入 SB 边际 KL 与能量/力蒸馏（小权重起步）。

**优化超参（建议）**

* AdamW，lr 2e‑4（线性 warmup 2k step，余弦退火到 2e‑5），
* β=(0.9,0.95)，weight decay 0.05，梯度裁剪 1.0，EMA 0.9995，
* 混合精度（bf16）+ 梯度累积（按显存调）。

---

## 10. 评测协议

### 10.1 端点质量

* 蛋白：iRMSD（主链/侧链分列）、FNAT（接触恢复）、clash rate。
* 配体（若做姿势）：Pose RMSD、有效率（价态/键长角度/环）。

### 10.2 路径合理性

* **接触形成单调性**：(C(t)) 随 (t) 单调上升（允许小噪动）。
* **短程 MD 验证**：从若干中间帧各跑 200–500 ps，向 holo 收敛比例。
* **能量剖面**（若有能量头）：能垒位置与幅度与经验一致性。

### 10.3 泛化

* 跨家族、跨口袋类型、计算结构（AF/ESMFold）鲁棒性；
* 配体骨架外推的稳定性（仅条件生成模式 A）。

---

## 11. 消融与对照

* 去掉几何分支，仅 ESM‑3 语义。
* 维度时间表偏移 on/off；解码器噪声增强 on/off。
* 宽度 < latent 维 vs. 宽度 ≥ latent 维。
* SB 正则 on/off；能量蒸馏 on/off。
* 模式 A vs. B（协同生成）。

---

## 12. 计算与工程

* 显卡：A100/H100 80GB（或 4×A100 40GB 分布式）。
* 训练时长（单卡等效）：RAE 40–80h；FM 60–120h；联合微调 40–80h（随数据量浮动）。
* 数据管线缓存：预计算 ESM‑3 令牌与几何图，落地为 `.pt`；dataloader 只做轻处理。

---

## 13. 目录结构（建议）

```
project/
  data/
    splits/ (train.json, val.json, test.json)
    processed/ (apo/*.pdb, holo/*.pdb, lig/*.sdf/.mol2, features/*.pt)
  models/
    encoders/ (esm_adapter.py, gvp_gnn.py, ligand_encoder.py)
    decoders/ (torsion_fk.py, fape_loss.py, dist_loss.py)
    bridge/   (flow_field.py, geodesic_ops.py, sb_regularizer.py)
  train/
    train_rae.py
    train_bridge.py
    train_joint.py
  eval/
    metrics_endpoints.py
    metrics_path.py
    md_short_runs.py
  utils/
    geom_so3.py
    geom_s1.py
    noise_samplers.py
    schedule_shift.py
    masking.py
  configs/
    rae.yaml
    bridge.yaml
    joint.yaml
  logs/ ckpts/ scripts/
```

---

## 14. 关键伪代码

### 14.1 geodesic 插值 & 噪声（欧氏 / (S^1) / (SO(3))）

```python
def interp_euclid(z0, z1, t, sigma_t):
    eps = normal_like(z0) * sigma_t
    return (1 - t) * z0 + t * z1 + eps, (z1 - z0)           # state, "true" velocity

def wrap_angle(a):  # map to (-pi, pi]
    return (a + math.pi) % (2 * math.pi) - math.pi

def interp_s1(theta0, theta1, t, kappa):
    d = wrap_angle(theta1 - theta0)
    theta_t = wrap_angle(theta0 + t * d) + von_mises_like(theta0, kappa)
    v_true  = d  # constant angular velocity in wrapped space
    return theta_t, v_true

def so3_log(R):  # rotation matrix -> axis-angle (vec3)
    A = 0.5 * (R - R.T)
    r = vee(A)  # extract vector
    angle = torch.norm(r)  # small-angle handling omitted for brevity
    return r

def so3_exp(r):  # axis-angle -> rotation matrix
    # Rodrigues with safe series for small |r|
    ...

def interp_so3(R0, R1, t, F):
    A = so3_log(R0.T @ R1)          # tangent in Lie algebra
    Rt = R0 @ so3_exp(t * A)        # geodesic
    Xi = matrix_fisher_sample(F)    # noise on SO(3)
    Rt_noisy = Rt @ Xi
    v_true = Rt @ A                 # d/dt Rt = Rt * A
    return Rt_noisy, v_true
```

### 14.2 Flow Matching 训练循环（最小版）

```python
for batch in loader:
    Zp0, Zp1, Zl = batch["Zp_apo"], batch["Zp_holo"], batch["Zl"]

    # 中间态采样（分块：欧氏/S1/SO3）
    t = torch.rand(B, 1, 1)
    zt, v_true = interp_blocks(Zp0, Zp1, t, sigma_t, kappa, F)

    # 维度依赖时间表偏移
    t_shift = schedule_shift(t, m_total, n_base)

    # 预测速度场
    v_pred = flow_field(zt, t_shift, Zl)  # Transformer/GNN + cond

    # FM 损失
    L_fm = mse(v_pred, v_true)

    # 可选 SB 正则、几何/能量约束
    L_sb   = sb_regularizer(...)
    X_hat  = decoder(zt)        # 解码中间态
    L_geom = geometry_penalty(X_hat, ...)

    # 端点重建（解码器噪声增强）
    L_rec = rec_loss(decoder(noise(Zp0)), apo_targets) + \
            rec_loss(decoder(noise(Zp1)), holo_targets)

    loss = L_rec + alpha * L_fm + beta * L_sb + gamma * L_geom
    loss.backward(); clip_grad_norm_(...); opt.step(); ema.update()
```

### 14.3 解码器损失（扭转 + FAPE + 距离）

```python
def torsion_loss(theta_pred, theta_gt, mask):
    d = wrap_angle(theta_pred - theta_gt)
    return (d**2 * mask).sum() / (mask.sum() + 1e-8)

def fape_loss(X_pred, X_gt, frames):
    # transform to local frames, compute invariant distances
    ...

def dist_loss(D_pred, D_gt, mask):
    return ((D_pred - D_gt)**2 * mask).sum() / (mask.sum() + 1e-8)
```

---

## 15. 指标与可视化面板（建议）

* 端点：iRMSD（主/侧链分开）、FNAT、clash%。
* 路径：接触形成度曲线 (C(t))、能量剖面（若有）、MD 短程收敛率。
* 可视化：关键残基 torsion vs. t，口袋开合度（SASA 或距离特征）vs. t。

---

## 16. 风险与兜底

* **数据泄漏**：严格同源聚类拆分 + 配体骨架平衡。
* **训练不稳**：先最小目标（FM + (\mathcal{L}_\text{rec})），SB/能量正则**后开**；学习率与噪声幅度网格搜索。
* **几何崩坏**：强制等变/对齐不变损失；硬阈值碰撞惩罚；扭转角限幅。
* **过拟合**：大量数据增强（随机局部扰动/噪声），权重衰减，EMA。

---

## 17. 时间表（以 12–14 周为例）

1. **W1–2**：数据清洗与三元组构建；ESM‑3 令牌与几何图缓存。
2. **W3–5**：RAE 预训练（扭转+FAPE+距离），达成稳定重建。
3. **W6–9**：FM 训练 + 几何无碰撞；中间态可视化与首轮指标。
4. **W10–12**：联合微调；引入 SB 正则与轻量能量蒸馏；撰写方法/实验。
5. **W13–14**：增补消融、泛化组实验、补图与附录。

---

## 18. 扩展与后续

* **协同生成（模式 B）**：引入配体位姿/扭转的联合流形，做蛋白–配体**协同路径**。
* **能量一致性**：对接打分器/ML 力场蒸馏，或以 SB 为主干建桥。
* **属性引导**：亲和/选择性/溶解度等“属性头”对采样做引导。
* **多体/多配体**：引入变构位点或二聚体耦合的路径学习。

---

## 19. 执行清单（逐条打勾）

* [ ] 三元组（apo/lig/holo）构建与质控
* [ ] 同源聚类拆分与配体骨架平衡
* [ ] ESM‑3 + GVP 编码器缓存
* [ ] 解码器 RAE 预训练（扭转 + FAPE + 距离）
* [ ] FM 训练（geodesic+noise；时间表偏移）
* [ ] 路径几何正则（无碰撞/键长角度）
* [ ] 端点与路径指标面板就绪
* [ ] SB 正则与能量蒸馏（小权重试水）
* [ ] 消融与泛化组
* [ ] 论文草稿方法/实验图表

---

### 备注（实现建议与经验句柄）

* **先小后大**：先在小蛋白与单口袋上打通，再扩到大体系。
* **mask 一切**：扭转角、缺失原子、变长侧链务必用 mask 规避梯度污染。
* **日志可视化**：始终绘制 (C(t))、torsion vs. t、碰撞率 vs. step。
* **数值稳定**：(SO(3)) 小角近似用安全罗德里格斯，S¹ wrap 一定放在 loss 里。

---

以上就是我将要逐步执行的**“从 0 到 1”**项目手册。
如果你同意这份手册作为 v1.0，我们就按第 17 节的时间表，从 **数据三元组与特征缓存** 开始推进。需要我把 **RAE 预训练（阶段 1）** 的 `rae.yaml` 超参与训练脚本骨架（入参/日志字段/保存点命名）也写出来吗？

