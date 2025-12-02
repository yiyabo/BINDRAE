# BINDRAE - Stage-2 实现任务清单

> **项目目标**：实现 apo→holo 连续构象路径的 Conditional Flow Matching / Bridge Flow
>
> **更新时间**：2025-11-27
>
> **前置依赖**：Stage-1 训练完成（χ1 ≈ 75–76%）
>
> **理论文档**：`docs/理论/Stage2.md`（算法蓝图）、`docs/理论/Stage2理论与指导.md`（理论规范）

---

## 📊 项目概览

### 任务定义

给定：
- **apo 结构** `P_apo`（未结合配体的构象）
- **holo 结构** `P_holo`（配体结合后的构象）
- **配体** `L`（含 bound pose）
- **序列** `S`（冻结 ESM-2 表征）

目标：
- 在混合状态空间 `SE(3)^N × (S^1)^K` 上学习一个**时间连续的向量场** `v_Θ(x,t)`
- 使得从 `x(0) = x_apo` 积分到 `x(1)` 能够接近 `x_holo`
- 路径上满足几何/生物物理约束（FAPE 平滑、无严重 clash、pocket contact 单调递增）

### 架构设计

```
                                    ┌─────────────────────────────────────────┐
                                    │           TorsionFlowNet                │
                                    │                                         │
x(t) = (F(t), θ(t)) ──┬──> FK ──> atom14 ──> EdgeEmbedder ──┐               │
                      │                                      │               │
                      │     ESM(冻结) ──> Adapter ───────────┤               │
                      │                                      ↓               │
                      │     L_tok ──> LigandTokenEmbed ──> LigandConditioner │
                      │                                      ↓               │
                      │                              FlashIPA (from Stage-1) │
                      │                                      ↓               │
                      │     t ──> TimeEmbed ──────────> h_i(t) + t_embed     │
                      │                                      ↓               │
                      │     w_res ─────────────────> PocketGateMLP ──> g_i   │
                      │                                      ↓               │
                      │                         ┌────────────┴────────────┐  │
                      │                         ↓                        ↓  │
                      │                   TorsionHead              RigidHead │
                      │                         ↓                        ↓  │
                      │                    dθ/dt * g_i             dF/dt * g_i│
                      │                         └────────────┬────────────┘  │
                      │                                      ↓               │
                      └───────────────────────────────> v_Θ(x,t) ────────────┘
```

### 关键技术决策

| 模块               | 方案                                   | 理由                                       |
| ------------------ | -------------------------------------- | ------------------------------------------ |
| **状态空间**       | SE(3)^N × (S^1)^K（刚体帧 + torsion）  | 物理可解释，与 Stage-1 FK 对接             |
| **参考桥**         | torsion Brownian bridge + SE(3) geodesic | CFM/Bridge Flow 标准做法                 |
| **几何 Encoder**   | 复用 Stage-1 的 Enc_trunk              | 不回写全局 rigids，避免双重更新            |
| **口袋门控**       | PocketGateMLP（w_res + t_embed）       | 聚焦 binding region 的速度场               |
| **路径正则**       | 在 learned path x_Θ(t_k) 上施加        | 比只在参考桥上正则更有物理意义             |
| **Stage-1 先验**   | t > 0.5 时软对齐 θ_stage1              | 后半段落到 Stage-1 holo manifold           |
| **噪声日程**       | 默认 σ(t)=0（deterministic PCFM）      | 先保证训练稳定，再逐步加噪                 |

### 与 Stage-1 的关系

```
Stage-1: 冻结ESM + Adapter + LigandConditioner + FlashIPA + TorsionHead + FK
         → 学习 "给定配体时 holo 应该长什么样"（holo decoder/prior）

Stage-2: 复用 Stage-1 的 Enc_trunk 作为几何 encoder
         + 新增 TimeEmbed + PocketGateMLP + RigidHead
         → 学习 "如何从 apo 连续变化到 holo"（apo→holo bridge flow）
```

---

## ✅ 前置条件（已满足）

### Stage-1 训练完成

- [X] χ1 准确率 ≈ 75–76%（目标 > 70%）
- [X] FAPE ≈ 0.055 Å（优秀）
- [X] Pocket iRMSD ≈ 0.01 Å
- [X] Clash% ≈ 9.4%
- [X] 模型 checkpoint 可用（`checkpoints/stage1_best.pt`）

### Stage-1 可复用模块

- [X] `ESMAdapter`（1280 → 384）
- [X] `EdgeEmbedder`（z_rank=2, num_rbf=16）
- [X] `LigandTokenEmbedding`（20D → 64D）
- [X] `LigandConditioner`（Cross-Attn + FiLM）
- [X] `FlashIPAModule`（3 层 IPA）
- [X] `OpenFoldFK`（torsion → atom14）
- [X] 损失函数：`fape_loss`, `clash_penalty`, `torsion_loss`, `distance_loss`

### 理论文档完成

- [X] `docs/理论/Stage2.md`（算法蓝图 + 伪代码）
- [X] `docs/理论/Stage2理论与指导.md`（理论规范，已定稿）

---

## 🔨 待实现任务

### Phase 0: 数据准备（预计 2-3 天）

#### 0.1 apo-holo 配对数据源评估

**任务描述**：
- [ ] 评估 AHoJ 数据库（apo-holo pairs）
- [ ] 评估 PLINDER 数据库
- [ ] 确定数据来源与下载方式
- [ ] 统计可用 apo-holo-ligand 三元组数量

**输出**：
- 数据源选择报告
- 预估数据量（目标 > 500 对）

#### 0.2 数据预处理管线 (`src/stage2/datasets/preprocess.py`)

**功能需求**：
- [ ] 序列对齐（apo vs holo）
  - 使用 BioPython 或 PyMOL 进行结构对齐
  - 确保残基一一对应
- [ ] torsion 提取
  - 复用 Stage-1 的 torsion 提取代码
  - 同时提取 apo 和 holo 的 (φ, ψ, ω, χ1-4)
- [ ] 刚体帧构建
  - 从 (N, Cα, C) 构建 SE(3) 帧
  - 分别构建 F_apo 和 F_holo
- [ ] 配体特征
  - 复用 Stage-1 的 ligand_utils
  - 提取重原子坐标 + 方向探针 + 20D 类型特征
- [ ] 口袋权重 w_res
  - 基于 holo+ligand 计算（与 Stage-1 一致）
- [ ] ESM 特征
  - 复用 Stage-1 的 ESM 缓存（序列相同）

**接口设计**：

```python
def preprocess_apo_holo_pair(
    apo_pdb: str,
    holo_pdb: str,
    ligand_sdf: str,
    esm_cache_dir: str
) -> Dict:
    """
    预处理单个 apo-holo-ligand 三元组
    
    Returns:
        {
            'pdb_id': str,
            'theta_apo': [N, 7],        # apo torsion
            'theta_holo': [N, 7],       # holo torsion
            'F_apo': (R[N,3,3], t[N,3]),  # apo 刚体帧
            'F_holo': (R[N,3,3], t[N,3]), # holo 刚体帧
            'esm_embed': [N, 1280],
            'lig_points': [M, 3],
            'lig_types': [M, 20],
            'w_res': [N],
            'torsion_mask': [N, 7],
            'bb_mask': [N],
            'sequence': str,
        }
    """
```

#### 0.3 数据集划分与验证

- [ ] 划分 train/val/test（建议 80:10:10）
- [ ] 确保 apo-holo 对不跨 split 泄露
- [ ] 数据质量检查
  - 序列长度分布
  - torsion 差异分布（Δθ_apo_holo）
  - 口袋残基数量分布

**输出文件结构**：

```
data/apo_holo/
├── processed/
│   ├── train/          # 训练集 NPZ 文件
│   ├── val/            # 验证集 NPZ 文件
│   └── test/           # 测试集 NPZ 文件
├── features/
│   └── esm2_cache/     # ESM 特征缓存（可复用 Stage-1）
└── splits/
    └── apo_holo_splits.json
```

---

### Phase 1: 核心模块开发（预计 3-4 天）

#### 1.1 Stage2Batch 数据类 (`src/stage2/datasets/batch.py`)

**功能需求**：
- [ ] 定义 `Stage2Batch` dataclass
- [ ] 支持 apo/holo 端点状态
- [ ] 支持时间采样
- [ ] 支持参考桥构造所需的中间变量

**接口设计**：

```python
@dataclass
class Stage2Batch:
    # 蛋白序列特征
    esm: Tensor              # [B, N, 1280]
    
    # apo 端点
    theta_apo: Tensor        # [B, N, 7]  apo torsion
    R_apo: Tensor            # [B, N, 3, 3]  apo rotation
    t_apo: Tensor            # [B, N, 3]  apo translation
    
    # holo 端点
    theta_holo: Tensor       # [B, N, 7]  holo torsion
    R_holo: Tensor           # [B, N, 3, 3]  holo rotation
    t_holo: Tensor           # [B, N, 3]  holo translation
    
    # 配体
    lig_points: Tensor       # [B, M, 3]
    lig_types: Tensor        # [B, M, 20]
    lig_mask: Tensor         # [B, M]
    
    # 口袋权重
    w_res: Tensor            # [B, N]
    
    # 掩码
    node_mask: Tensor        # [B, N]
    torsion_mask: Tensor     # [B, N, 7]
    bb_mask: Tensor          # [B, N, 3]  backbone (φ,ψ,ω) mask
    chi_mask: Tensor         # [B, N, 4]  sidechain (χ1-4) mask
    
    # 序列类型
    aatype: Tensor           # [B, N]  氨基酸类型（用于 FK）
    
    # Meta
    pdb_ids: List[str]
    n_residues: List[int]
```

#### 1.2 Stage2Dataset (`src/stage2/datasets/dataset_bridge.py`)

**功能需求**：
- [ ] 加载预处理的 apo-holo 数据
- [ ] 构建 Stage2Batch
- [ ] collate 函数支持 padding

**接口设计**：

```python
class Stage2Dataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train'):
        ...
    
    def __getitem__(self, idx: int) -> Dict:
        ...
    
    def __len__(self) -> int:
        ...

def collate_stage2_batch(samples: List[Dict]) -> Stage2Batch:
    ...

def create_stage2_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    ...
```

#### 1.3 参考桥构造 (`src/stage2/models/reference_bridge.py`)

**功能需求**：
- [ ] **torsion 参考桥**
  - wrap_to_pi 计算最短角差 Δθ
  - 线性或 smoothstep 插值函数 γ(t)
  - 可选噪声 σ(t)·ξ（默认 σ=0）
  - 解析速度 u_ref(t)
- [ ] **SE(3) 参考桥**
  - 相对旋转 R_Δ = R_holo @ R_apo^T
  - 对数映射 Ω = log(R_Δ)
  - geodesic 插值 R(t) = exp(γ(t)·Ω) @ R_apo
  - 平移线性插值
  - 可选 Lie algebra 上的噪声
- [ ] **统一接口**

**接口设计**：

```python
class ReferenceBridgeBuilder:
    def __init__(
        self,
        interpolation: str = 'linear',  # 'linear' or 'smoothstep'
        sigma_torsion: float = 0.0,      # torsion 噪声强度
        sigma_rotation: float = 0.0,     # rotation 噪声强度
        sigma_translation: float = 0.0,  # translation 噪声强度
    ):
        ...
    
    def sample_bridge(
        self,
        batch: Stage2Batch,
        t: Tensor,  # [B] or scalar
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        构造时间 t 的参考桥状态和速度
        
        Returns:
            theta_ref: [B, N, 7]   参考 torsion
            R_ref: [B, N, 3, 3]    参考 rotation
            t_ref: [B, N, 3]       参考 translation
            u_theta_ref: [B, N, 7] 参考 torsion 速度
            u_R_ref: [B, N, 3]     参考 rotation 速度（轴角）
            u_t_ref: [B, N, 3]     参考 translation 速度
        """
        ...

def wrap_to_pi(angle: Tensor) -> Tensor:
    """将角度 wrap 到 (-π, π]"""
    ...

def so3_log(R: Tensor) -> Tensor:
    """SO(3) 对数映射：R → 轴角向量"""
    ...

def so3_exp(omega: Tensor) -> Tensor:
    """SO(3) 指数映射：轴角向量 → R"""
    ...
```

#### 1.4 时间嵌入 (`src/stage2/models/time_embed.py`)

**功能需求**：
- [ ] sin/cos 位置编码 或 小 MLP
- [ ] 支持与残基特征拼接

**接口设计**：

```python
class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int = 64, max_period: float = 10000.0):
        ...
    
    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: [B] 时间值，范围 [0, 1]
        Returns:
            t_embed: [B, d_model]
        """
        ...
```

#### 1.5 口袋门控 (`src/stage2/models/pocket_gate.py`)

**功能需求**：
- [ ] 输入：残基特征 h_i(t) + w_res + t_embed
- [ ] 输出：门控值 g_i(t) ∈ (0, 1)
- [ ] 用于缩放向量场，使 pocket 区域有更大的速度

**接口设计**：

```python
class PocketGateMLP(nn.Module):
    def __init__(
        self,
        d_in: int = 384,      # 残基特征维度
        d_time: int = 64,     # 时间嵌入维度
        d_hidden: int = 128,
    ):
        ...
    
    def forward(
        self,
        h: Tensor,         # [B, N, d_in]  残基特征
        w_res: Tensor,     # [B, N]        口袋权重
        t_embed: Tensor,   # [B, d_time]   时间嵌入
    ) -> Tensor:
        """
        Returns:
            gate: [B, N]  门控值 ∈ (0, 1)
        """
        ...
```

#### 1.6 TorsionFlowNet (`src/stage2/models/torsion_flow.py`)

**功能需求**：
- [ ] 复用 Stage-1 的 Enc_trunk（ESMAdapter + EdgeEmbedder + LigandConditioner + FlashIPA）
  - **关键**：Enc_trunk 内部的 rigids 更新不回写到全局状态
- [ ] 新增 TimeEmbedding
- [ ] 新增 PocketGateMLP
- [ ] 新增 RigidHead（预测 dF/dt）
- [ ] 复用 TorsionHead（预测 dθ/dt）
- [ ] 输出向量场 v_Θ(x,t)

**接口设计**：

```python
class TorsionFlowNet(nn.Module):
    def __init__(
        self,
        # Stage-1 复用配置
        c_s: int = 384,
        c_z: int = 128,
        d_lig: int = 64,
        depth: int = 3,
        # Stage-2 新增配置
        d_time: int = 64,
        d_gate_hidden: int = 128,
        # 可选：加载 Stage-1 权重
        stage1_checkpoint: Optional[str] = None,
        freeze_encoder: bool = False,
    ):
        ...
    
    def forward(
        self,
        # 当前状态 x(t)
        theta: Tensor,        # [B, N, 7]
        R: Tensor,            # [B, N, 3, 3]
        trans: Tensor,        # [B, N, 3]
        # 条件
        esm: Tensor,          # [B, N, 1280]
        lig_points: Tensor,   # [B, M, 3]
        lig_types: Tensor,    # [B, M, 20]
        w_res: Tensor,        # [B, N]
        t: Tensor,            # [B] 时间
        # 掩码
        node_mask: Tensor,    # [B, N]
        lig_mask: Tensor,     # [B, M]
        torsion_mask: Tensor, # [B, N, 7]
        aatype: Tensor,       # [B, N]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            v_theta: [B, N, 7]    torsion 速度 (sin/cos 格式需转换)
            v_R: [B, N, 3]        rotation 速度（轴角增量）
            v_trans: [B, N, 3]    translation 速度
        """
        ...
```

---

### Phase 2: 损失函数与训练（预计 3-4 天）

#### 2.1 Flow Matching 损失 (`src/stage2/training/losses.py`)

**功能需求**：
- [ ] L_FM = E[w_res^α · mask · ||v_Θ(X_ref, t) - u_ref||²]
- [ ] 支持 torsion / rotation / translation 分别加权
- [ ] 支持 χ1 置信度权重 r_i（可选）

**接口设计**：

```python
def flow_matching_loss(
    v_theta_pred: Tensor,     # [B, N, 7]
    v_R_pred: Tensor,         # [B, N, 3]
    v_trans_pred: Tensor,     # [B, N, 3]
    u_theta_ref: Tensor,      # [B, N, 7]
    u_R_ref: Tensor,          # [B, N, 3]
    u_trans_ref: Tensor,      # [B, N, 3]
    w_res: Tensor,            # [B, N]
    torsion_mask: Tensor,     # [B, N, 7]
    node_mask: Tensor,        # [B, N]
    alpha: float = 1.0,       # w_res 指数
    r_i: Optional[Tensor] = None,  # [B, N] χ1 置信度
) -> Tensor:
    ...
```

#### 2.2 端点一致性损失

**功能需求**：
- [ ] 从 x_0 积分到 t=1 得到 x_Θ(1)
- [ ] 与真实 x_1（或 Stage-1 pseudo-holo）比较
- [ ] torsion wrap-angle loss + FAPE

**接口设计**：

```python
def endpoint_consistency_loss(
    x_pred_end: Tuple[Tensor, Tensor, Tensor],  # (θ, R, t) at t=1
    x_true_end: Tuple[Tensor, Tensor, Tensor],  # (θ_holo, R_holo, t_holo)
    w_res: Tensor,
    torsion_mask: Tensor,
    node_mask: Tensor,
    aatype: Tensor,
    fk_module,  # OpenFoldFK
) -> Tensor:
    ...
```

#### 2.3 路径几何正则

**功能需求**：
- [ ] **L_smooth**：相邻时间点的 FAPE / Cα RMSD
- [ ] **L_clash**：复用 Stage-1 的随机 chunk clash（方案 A）
- [ ] **L_contact**：pocket-ligand soft contact 单调性
  - C(t) = mean soft_contact(dist(res_i, L))
  - L_contact = Σ max(0, C(t_k) - C(t_{k+1}) - ε)，ε ≥ 0
- [ ] **L_prior**：t > 0.5 时对齐 Stage-1 的 θ_stage1

**接口设计**：

```python
def path_smoothness_loss(
    x_path: List[Tuple[Tensor, Tensor, Tensor]],  # [(θ, R, t)] at t_k
    w_res: Tensor,
    node_mask: Tensor,
    aatype: Tensor,
    fk_module,
) -> Tensor:
    ...

def path_clash_loss(
    x_path: List[Tuple[Tensor, Tensor, Tensor]],
    node_mask: Tensor,
    aatype: Tensor,
    fk_module,
    sample_size: int = 512,  # 每帧采样原子对数
) -> Tensor:
    ...

def contact_monotonicity_loss(
    x_path: List[Tuple[Tensor, Tensor, Tensor]],
    lig_points: Tensor,       # [B, M, 3]
    w_res: Tensor,
    node_mask: Tensor,
    aatype: Tensor,
    fk_module,
    pocket_threshold: float = 0.5,
    epsilon: float = 0.05,    # 允许的最大下降幅度
) -> Tensor:
    ...

def stage1_prior_loss(
    theta_pred: Tensor,       # [B, N, 7]  t > t_mid 时的预测
    theta_stage1: Tensor,     # [B, N, 7]  Stage-1 预测的 holo torsion
    t: Tensor,                # [B] 当前时间
    t_mid: float,             # 阈值（如 0.5）
    w_res: Tensor,
    torsion_mask: Tensor,
) -> Tensor:
    ...
```

#### 2.4 总损失与权重配置

**总损失形式**：

```python
L = L_FM 
  + λ_end * L_endpoint 
  + λ_geom * (L_smooth + L_clash + L_contact + L_prior)
```

**建议初始权重**：

```python
lambda_fm = 1.0
lambda_end = 0.1       # 可隔若干 step 启用
lambda_smooth = 0.1
lambda_clash = 0.1     # 复用 Stage-1 权重
lambda_contact = 0.1
lambda_prior = 0.05    # 小权重，作为平滑先验
alpha = 1.0            # w_res 指数
```

#### 2.5 ODE 积分器 (`src/stage2/training/integrator.py`)

**功能需求**：
- [ ] 支持 Euler / Heun / RK4
- [ ] 支持指定步数 T
- [ ] 支持中间状态输出（用于路径正则）

**接口设计**：

```python
class ODEIntegrator:
    def __init__(
        self,
        method: str = 'heun',  # 'euler', 'heun', 'rk4'
        num_steps: int = 20,
    ):
        ...
    
    def integrate(
        self,
        model: TorsionFlowNet,
        x0: Tuple[Tensor, Tensor, Tensor],  # (θ_0, R_0, t_0)
        conditions: Dict,  # esm, lig_points, lig_types, w_res, masks
        return_path: bool = False,
        path_times: Optional[List[float]] = None,  # 返回指定时间点的状态
    ) -> Union[
        Tuple[Tensor, Tensor, Tensor],  # 仅终点
        Tuple[Tuple, List[Tuple]],      # 终点 + 路径
    ]:
        ...
```

#### 2.6 Stage2Trainer (`src/stage2/training/trainer.py`)

**功能需求**：
- [ ] 训练循环
  - 采样时间 t ~ U(0,1)
  - 构造参考桥 X_ref(t), u_ref(t)
  - 前向 TorsionFlowNet 得到 v_Θ
  - 计算 L_FM
  - （可选）采样 K 个 t_k，积分得到 x_Θ(t_k)，计算几何正则
  - （可选）计算 L_endpoint
- [ ] 验证循环
  - 从 apo 积分到 holo
  - 评估终点误差和路径质量
- [ ] 优化器：AdamW + CosineAnnealingLR
- [ ] 混合精度训练
- [ ] Checkpoint 保存与早停

**接口设计**：

```python
@dataclass
class Stage2TrainingConfig:
    # 数据
    data_dir: str
    batch_size: int = 4
    num_workers: int = 4
    
    # 模型
    stage1_checkpoint: str = 'checkpoints/stage1_best.pt'
    freeze_encoder: bool = False
    
    # 参考桥
    interpolation: str = 'linear'
    sigma_torsion: float = 0.0   # 默认无噪声
    
    # 积分
    integrator_method: str = 'heun'
    integrator_steps: int = 20
    
    # 路径正则
    num_path_points: int = 3     # K=3 个时间点
    compute_endpoint_every: int = 10  # 每 10 步算一次 endpoint
    
    # 损失权重
    lambda_fm: float = 1.0
    lambda_end: float = 0.1
    lambda_smooth: float = 0.1
    lambda_clash: float = 0.1
    lambda_contact: float = 0.1
    lambda_prior: float = 0.05
    alpha: float = 1.0           # w_res 指数
    t_mid: float = 0.5           # L_prior 的时间阈值
    
    # 优化
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    max_epochs: int = 100
    grad_clip: float = 1.0
    mixed_precision: bool = True
    
    # 验证与早停
    val_interval: int = 1
    early_stop_patience: int = 20
    save_top_k: int = 3

class Stage2Trainer:
    def __init__(self, config: Stage2TrainingConfig):
        ...
    
    def train_epoch(self) -> Dict[str, float]:
        ...
    
    def validate(self) -> Dict[str, float]:
        ...
    
    def train(self):
        ...
```

---

### Phase 3: 评估与推理（预计 2 天）

#### 3.1 评估指标

**终点质量**：
- [ ] torsion 角误差（特别是 χ1 命中率）
- [ ] backbone/pocket FAPE
- [ ] pocket Cα iRMSD
- [ ] clash%

**路径质量**：
- [ ] 平均 clash%
- [ ] contact 曲线 C(t) 形状与单调性违例比例
- [ ] 局部 FAPE 平滑度

**接口设计**：

```python
def evaluate_endpoint(
    x_pred_end: Tuple[Tensor, Tensor, Tensor],
    x_true_end: Tuple[Tensor, Tensor, Tensor],
    fk_module,
    aatype: Tensor,
    w_res: Tensor,
    node_mask: Tensor,
    torsion_mask: Tensor,
) -> Dict[str, float]:
    """
    Returns:
        {
            'chi1_acc': float,
            'chi1_rot_acc': float,
            'fape': float,
            'pocket_irmsd': float,
            'clash_pct': float,
        }
    """
    ...

def evaluate_path(
    x_path: List[Tuple[Tensor, Tensor, Tensor]],
    lig_points: Tensor,
    fk_module,
    aatype: Tensor,
    w_res: Tensor,
    node_mask: Tensor,
) -> Dict[str, float]:
    """
    Returns:
        {
            'avg_clash_pct': float,
            'contact_monotonicity_violation': float,
            'path_smoothness': float,
        }
    """
    ...
```

#### 3.2 推理脚本 (`scripts/inference_stage2.py`)

**功能需求**：
- [ ] **模式 1**：已知 apo + holo + ligand（路径重建评估）
- [ ] **模式 2**：仅 apo + ligand（使用 Stage-1 pseudo-holo）
- [ ] 输出路径可视化（多帧 PDB 或轨迹文件）
- [ ] 输出评估指标

---

### Phase 4: 测试与验证（预计 2 天）

#### 4.1 单元测试

- [ ] `test_reference_bridge.py`：参考桥构造正确性
  - wrap_to_pi 边界测试
  - SO(3) log/exp 数值稳定性
  - 插值端点验证（t=0 → apo, t=1 → holo）
- [ ] `test_time_embed.py`：时间嵌入
- [ ] `test_pocket_gate.py`：门控输出范围 (0, 1)
- [ ] `test_torsion_flow_net.py`：端到端前向
- [ ] `test_losses.py`：各损失可微分
- [ ] `test_integrator.py`：ODE 积分精度
- [ ] `test_trainer.py`：完整训练循环

#### 4.2 过拟合测试

- [ ] 单样本过拟合（验证模型可学习性）
- [ ] 小数据集验证（10 个样本）

#### 4.3 全量训练

- [ ] 完整数据集训练
- [ ] 监控指标：L_FM, 终点 χ1, 路径 clash%
- [ ] 调参：权重、噪声日程、积分步数

---

## 📈 验收标准

### 数据指标（终点）

| 指标             | 目标值   | 说明                               |
| ---------------- | -------- | ---------------------------------- |
| **终点 χ1 命中率** | > 70%    | 与 Stage-1 相当（路径不应损害终点） |
| **终点 FAPE**    | < 0.1 Å  | 局部帧对齐误差                     |
| **终点 pocket iRMSD** | < 0.5 Å | 口袋局部对齐                      |
| **终点 Clash%**  | < 10%    | 与 Stage-1 相当                    |

### 数据指标（路径）

| 指标                     | 目标值   | 说明                              |
| ------------------------ | -------- | --------------------------------- |
| **路径平均 Clash%**      | < 15%    | 路径上不应有严重穿插              |
| **Contact 单调性违例**   | < 20%    | C(t) 应大致单调递增               |
| **路径 FAPE 平滑度**     | < 0.2 Å  | 相邻帧之间的变化应平滑            |

### 训练稳定性

- [ ] 损失曲线平滑下降
- [ ] 验证指标稳定收敛
- [ ] 无 NaN/Inf
- [ ] 显存占用可控（< 40GB per sample on A100）

---

## 🚀 执行计划

### Week 1: 数据与基础模块（Day 1-4）

**Day 1-2**:
- [ ] 数据源评估与下载
- [ ] 数据预处理管线

**Day 3**:
- [ ] Stage2Batch 数据类
- [ ] Stage2Dataset 与 DataLoader

**Day 4**:
- [ ] 参考桥构造
- [ ] 时间嵌入
- [ ] 口袋门控

### Week 1-2: 模型与训练（Day 5-8）

**Day 5**:
- [ ] TorsionFlowNet（集成 Stage-1 Enc_trunk）
- [ ] 端到端前向测试

**Day 6**:
- [ ] Flow Matching 损失
- [ ] ODE 积分器

**Day 7**:
- [ ] 路径几何正则损失
- [ ] 端点一致性损失

**Day 8**:
- [ ] Stage2Trainer 完整实现
- [ ] 单样本过拟合测试

### Week 2: 评估与调优（Day 9-10）

**Day 9**:
- [ ] 评估指标实现
- [ ] 推理脚本

**Day 10**:
- [ ] 单元测试完善
- [ ] 小数据集验证
- [ ] 全量训练启动

---

## ⚠️ 注意事项

### 关键约束

1. **Enc_trunk 不回写 rigids**：FlashIPA 内部的 rigids 更新只是 encoder 辅助状态，全局状态 F(t) 只由 v_Θ 控制
2. **默认 σ(t)=0**：先用 deterministic PCFM 保证稳定，再逐步加噪
3. **路径正则在 learned path 上**：不只在参考桥上正则，要对积分得到的 x_Θ(t_k) 施加约束
4. **时间点采样**：每步只采 K=1-2 个 t_k 做几何正则，统计意义上覆盖 [0,1]
5. **Clash 方案 A**：沿用 Stage-1 的随机 chunk 采样（~512 原子对）
6. **L_prior 小权重**：训练时以真 holo 为主监督，L_prior 只是平滑先验

### 常见陷阱

- ❌ Enc_trunk 的 rigids 更新回写到全局状态，导致双重更新
- ❌ wrap_to_pi 实现错误导致 torsion 差异计算错误
- ❌ SO(3) log/exp 在旋转角接近 0 或 π 时数值不稳定
- ❌ 每步对所有时间点算几何正则，导致显存爆炸
- ❌ L_contact 的 ε 符号搞反（应为 ε ≥ 0）
- ❌ 推理时忘记使用 Stage-1 构造 pseudo-holo

### 数值稳定性

- SO(3) log/exp：旋转角接近 0 时用 Taylor 展开或 clamp
- 时间积分：使用 Heun/RK4 而非纯 Euler
- 梯度裁剪：保持 grad_clip=1.0

---

## 📚 参考文献

1. **Flow Matching**: Lipman et al. (ICLR 2023) - CFM 理论基础
2. **Pairwise CFM**: Tong et al. (ICML 2024) - 配对 Flow Matching
3. **Bridge Flow**: Liu et al. (2023) - Bridge Flow 理论
4. **SBALIGN**: Corso et al. (2024) - 蛋白质-配体对接的 Bridge Flow
5. **DiSCO**: Corso et al. (2023) - 扩散模型在结构预测中的应用
6. **ProteinZen**: 潜空间构象流
7. **BINDRAE Stage-1**: `docs/STAGE1_PIPELINE.md`, `docs/STAGE1_SUMMARY.md`
8. **BINDRAE 理论**: `docs/理论/理论与参考.md`, `docs/理论/Stage2.md`, `docs/理论/Stage2理论与指导.md`

---

## 📁 代码结构规划

```
src/stage2/
├── datasets/
│   ├── __init__.py
│   ├── preprocess.py       # 数据预处理
│   ├── batch.py            # Stage2Batch 定义
│   └── dataset_bridge.py   # Stage2Dataset
├── models/
│   ├── __init__.py
│   ├── reference_bridge.py # 参考桥构造
│   ├── time_embed.py       # 时间嵌入
│   ├── pocket_gate.py      # 口袋门控
│   └── torsion_flow.py     # TorsionFlowNet
├── training/
│   ├── __init__.py
│   ├── losses.py           # 所有损失函数
│   ├── integrator.py       # ODE 积分器
│   └── trainer.py          # Stage2Trainer
└── utils/
    ├── __init__.py
    └── so3_utils.py        # SO(3) 工具函数

scripts/
├── preprocess_apo_holo.py  # 数据预处理脚本
├── train_stage2.py         # 训练脚本
└── inference_stage2.py     # 推理脚本

tests/
├── test_reference_bridge.py
├── test_time_embed.py
├── test_pocket_gate.py
├── test_torsion_flow_net.py
├── test_losses_stage2.py
├── test_integrator.py
└── test_trainer_stage2.py
```

---

**最后更新**: 2025-11-27
**负责人**: BINDRAE Team
**状态**: Stage-2 规划完成 → 待实现 🚀
