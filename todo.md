# BINDRAE - Stage-1 IPA 实现任务清单

> **项目目标**：实现基于 FlashIPA 的蛋白质构象生成模型（Stage-1: 自编码器预训练）
>
> **更新时间**：2025-10-25
>
> **训练环境**：Linux + CUDA (2×A100-80GB 首选)

---

## 📊 项目概览

### 架构设计

```
ESM-2(冻结) → Adapter(1280→384) → FlashIPA×3层 → Torsion Head → FK → 全原子坐标
                ↑                                      ↓
            配体条件(Cross-Attn + FiLM)          FAPE + 扭转 + 距离 + clash
```

### 关键技术决策

| 模块                 | 方案                       | 理由                |
| -------------------- | -------------------------- | ------------------- |
| **几何注意力** | FlashIPA (因子化边)        | 线性显存/时间扩展   |
| **配体条件化** | Cross-Attn + 残基级FiLM    | 稳定且易调试        |
| **配体表示**   | 重原子 + 稀疏探针 (M≤128) | 关键方向 + 显存友好 |
| **帧初始化**   | (N,Cα,C) 实时构建         | 轻量且灵活          |
| **边嵌入**     | EdgeEmbedder (1D/2D因子化) | 避免NxN显存爆炸     |
| **帧更新**     | 每层预测增量并裁剪         | 防数值发散          |
| **刚体工具**   | 复用 OpenFold rigid_utils  | 工业级实现          |

---

## ✅ 已完成的工作

### 数据准备（100% 完成）

- [X] CASF-2016 数据集解压与验证
- [X] 蛋白质结构清洗 (283个复合物)
- [X] 配体规范化 (SDF格式)
- [X] 口袋软掩码提取 (5Å + RBF衰减)
- [X] 扭转角GT提取 (φ/ψ/ω/χ)
- [X] 数据集划分 (train:val:test = 226:29:28)
- [X] ESM-2 表征缓存 (283个, 650M模型)
- [X] 数据质量验证报告

**数据文件清单** (2,551个文件):

```
data/casf2016/
├── complexes/        # 283个PDB+SDF
├── processed/
│   ├── pockets/      # 283个口袋PDB
│   └── torsions/     # 283个扭转角NPZ
├── features/
│   └── esm2_cache/   # 283个ESM表征PT
└── splits/           # 数据划分JSON
```

---

# 🔨 待实现任务

### Phase 1: 核心模块开发 (预计2-3天)

#### 1.1 配体Token构建 (`utils/ligand_utils.py`)

**功能需求**:

- 关键原子识别 (RDKit FeatureFactory)
- HBD/HBA (氢键供体/受体)
- Aromatic (芳香中心)
- Charged (带电原子)

- [X] 方向探针生成（这其实是个创新点，反正我是没找到类似的做法。日后需要做消融）
  - 每个关键原子 ≤2 个探针
  - 沿成键方向外扩 1.5-2.0Å
  - 末端原子补反向探针
  - ≥3键按键序+Gasteiger电荷排序
- [X] 重要性采样 (M≤128)
  - 优先级: 带电(100) > HBD(50) > HBA(40) > 芳香(30)
  - 距离权重: 10Å - dist_to_pocket_center
  - 口袋中心: Cα加权质心 (权重=w_res)
- [X] 类型嵌入编码 (12维)
  - 原子类型 (8维): C/N/O/S/P/halogen/metal/probe
  - 药效团 (4维): HBD/HBA/aromatic/charged

**接口设计**:

```python
def detect_key_atoms(mol: Chem.Mol) -> Dict[str, Set[int]]
def build_direction_probes(mol, pos, atom_idx, max_k=2, step=1.5) -> np.ndarray
def build_ligand_tokens(mol, ca_xyz, w_res, max_points=128) -> Tuple[np.ndarray, np.ndarray]
def encode_atom_types(atoms_info) -> np.ndarray  # [M, 12]
```

**依赖**:

- RDKit (ChemicalFeatures, rdMolDescriptors)
- NumPy

---

#### 1.2 刚体帧工具 (`modules/rigid_utils.py`)

**功能需求**:

- [X] 三点构帧 (N, Cα, C → R, t)
  - 使用 OpenFold 的 Rigid 类
  - 支持批量处理 [B, N, 3]
- [X] 帧增量裁剪
  - 旋转: ≤15° (轴角范数裁剪)
  - 平移: ≤1.5Å (逐分量裁剪)
- [X] 刚体噪声注入 (数据增强)
  - 旋转: 均匀 [0, 5°]
  - 平移: 高斯 N(0, 0.5²)
  - Stage-1 前5k step不启用
- [X] Rigid 打包/解包
  - pack_rigids(R, t) → Rigid
  - unpack_rigids(Rigid) → (R, t)

**接口设计**:

```python
def build_frames_from_3_points(N, Ca, C) -> Tuple[torch.Tensor, torch.Tensor]
def clip_update(delta_rot, delta_trans, max_deg=15.0, max_trans=1.5) -> Tuple
def add_rigid_noise(R, t, rot_deg=5.0, trans_std=0.5, enable=True) -> Tuple
def pack_rigids(R, t) -> Rigid
def unpack_rigids(rigids) -> Tuple[torch.Tensor, torch.Tensor]
```

**依赖**:

- OpenFold (openfold.utils.rigid_utils.Rigid)
- PyTorch

---

#### 1.3 边嵌入封装 (`modules/edge_embed.py`) ✅ 已完成

**实现方案**:

- [X] **使用FlashIPA原生EdgeEmbedder**（替代自实现）
  - 模式: flash_1d_bias (线性显存O(N))
  - 因子秩: z_factor_rank=**2**（⚠️ FlashAttention限制headdim≤256）
  - RBF核数: num_rbf=16（项目配置，原生默认32）
- [X] **ProjectEdgeConfig配置适配**
  - 项目配置 → FlashIPA配置转换
  - 参数：c_s=384, c_p=128, **z_rank=2**
- [X] **EdgeEmbedderAdapter简化接口**
  - 原生6参数 → 简化3参数
  - 自动处理侧链坐标（trans_sc用主链代替）
  - 返回dict格式（含自动生成的edge_mask）
- [X] **预留共价边扩展接口**
  - 第一版不实现 (只用几何边)
  - 留待 Phase-2 ablation

**输出格式**:
```python
{
    'z_f1': [B, N, 2, 128],  # 边因子1
    'z_f2': [B, N, 2, 128],  # 边因子2  
    'edge_mask': [B, N, N]    # 边掩码
}
```

**限制说明**: z_rank=2（FlashAttention的headdim≤256限制，详见FlashIPA_USAGE.md）

**测试状态**: ✅ 通过（RTX 4090 D, 50残基显存18.73 MB）

**文档**: `src/stage1/modules/FlashIPA_USAGE.md`

**实际接口**:

```python
from src.stage1.modules.edge_embed import create_edge_embedder

embedder = create_edge_embedder(c_s=384, c_p=128, z_rank=2, num_rbf=16)
outputs = embedder(node_embed, translations, node_mask)
# 返回: {'z_f1', 'z_f2', 'edge_mask', 'raw_output'}
```

**依赖**:

- flash_ipa (EdgeEmbedder, EdgeEmbedderConfig) - 原生库

---

#### 1.4 FlashIPA 几何分支 (`models/ipa.py`) ✅ 已完成

**实现方案**:

- [X] **多层 IPA 堆叠** (depth=3)
  - InvariantPointAttention (FlashIPA原生)
  - 每层: Self-IPA → 帧更新 → 裁剪 → compose → FFN → 残差
- [X] **帧更新预测头** (BackboneUpdateHead)
  - 结构: LayerNorm → Linear(128) → GELU → Linear(6)
  - 输出: [ωx, ωy, ωz, tx, ty, tz] (轴角+平移)
- [X] **逐层裁剪与更新**
  - clip_frame_update (旋转≤15°, 平移≤1.5Å)
  - axis_angle → Rotation → Rigid → compose
- [X] **FFN + LayerNorm + 残差** (IPAFeedForward)

**实际接口**:

```python
from src.stage1.models.ipa import create_flashipa_module

ipa_module = create_flashipa_module(c_s=384, c_z=128, depth=3)
s_geo, rigids_final = ipa_module(s, rigids, z_f1, z_f2, mask)
# 输入: s[B,N,384], rigids(Rigid对象), z_f1/z_f2[B,N,2,128], mask[B,N]
# 返回: s_geo[B,N,384], rigids_final(Rigid对象)
```

**实际超参**:

```yaml
c_s: 384
c_z: 128
c_hidden: 128
no_heads: 8
depth: 3
no_qk_points: 8
no_v_points: 12
z_factor_rank: 2        # ⚠️ 降为2（FlashAttention限制）
dropout: 0.1
attn_dtype: fp16        # headdim_eff=228
```

**测试状态**: ✅ 通过（RTX 4090 D, 20残基显存48.08 MB，参数量9.96M）

**文档**: `src/stage1/modules/FlashIPA_USAGE.md`

**依赖**:

- flash_ipa (InvariantPointAttention, IPAConfig, Rigid, Rotation)
- flash_attn (FlashAttention2核心)
- beartype, jaxtyping (类型检查)

---

#### 1.5 配体条件化模块 (`models/ligand_condition.py`) ✅ 已完成

**实现方案**:

- [X] **配体Token嵌入** (LigandTokenEmbedding)
  - 输入: concat([xyz(3), types(12)]) = 15维
  - 输出: d_lig=64
  - 网络: Linear(15→64) → LayerNorm → GELU → Linear(64→64)
- [X] **Cross-Attention** (ProteinLigandCrossAttention)
  - Q: 蛋白节点 S [B,N,384]
  - K/V: 配体token [B,M,64]
  - 多头: heads=8
  - 完整投影: Q/K/V proj + 输出投影
- [X] **残基级 FiLM 调制** (FiLMModulation)
  - gamma = MLP_gamma(S_cross)
  - beta = MLP_beta(S_cross)
  - S_out = (1 + λ·γ) ⊙ S + λ·β
- [X] **门控 warmup**
  - λ: 0→1 线性 (2000 steps)
  - 支持手动指定或自动计算
- [X] **特殊初始化**
  - gamma最后层: std=0.01, 偏置=1 ✓
  - beta最后层: std=0.01, 偏置=0 ✓

**实际接口**:

```python
from src.stage1.models.ligand_condition import create_ligand_conditioner

conditioner = create_ligand_conditioner(c_s=384, d_lig=64, num_heads=8, warmup_steps=2000)
s_cond = conditioner(protein_features, lig_points, lig_types, 
                     protein_mask, ligand_mask, current_step=1000)
# 输入: protein[B,N,384], lig_points[B,M,3], lig_types[B,M,12], masks
# 返回: s_cond[B,N,384]
```

**测试状态**: ✅ 通过（RTX 4090 D, 30残基+50配体token，显存20.98 MB，参数量548K）

**依赖**:

- PyTorch (nn.Linear, F.softmax等)

---

### Phase 2: 数据流与训练 (预计2天)

#### 2.1 IPA 数据加载器 (`data/dataset_ipa.py`)

**功能需求**:

- [ ] 继承现有 CASF2016Dataset
- [ ] 实时构建局部帧
  - build_frames_from_3_points(N, Ca, C)
- [ ] 实时构建配体tokens
  - build_ligand_tokens(mol, ca_xyz, w_res)
- [ ] 数据增强控制
  - 训练集: step>5000时加噪
  - 验证集: 不加噪
- [ ] 返回 IPABatch (dataclass)

**Batch结构**:

```python
@dataclass
class IPABatch:
    # 蛋白
    esm: Tensor              # [B, N, 1280]
    N: Tensor                # [B, N, 3]
    Ca: Tensor               # [B, N, 3]
    C: Tensor                # [B, N, 3]
    node_mask: Tensor        # [B, N]
  
    # 配体
    lig_points: Tensor       # [B, M, 3]
    lig_types: Tensor        # [B, M, 12]
    lig_mask: Tensor         # [B, M]
  
    # GT
    torsion_gt: Tensor       # [B, N, n_tor, 2]
    xyz_gt: Tensor           # [B, A, 3]
    frames_gt: Tuple         # (R, t)
  
    # 口袋
    w_res: Tensor            # [B, N]
  
    # Meta
    pdb_id: List[str]
```

**依赖**:

- utils.ligand_utils
- modules.rigid_utils
- 现有数据集基类

---

#### 2.2 评估指标 (`utils/metrics.py`)

**功能需求**:

- [ ] 口袋 iRMSD
  - 仅用口袋残基做 Kabsch 对齐
  - 计算口袋重原子 RMSD
- [ ] χ1 命中率
  - 只统计有侧链的残基
  - 阈值: ±20°
  - wrap 角度差
- [ ] Clash 百分比
  - 非键原子对 < 2.0Å
  - 排除1-2, 1-3邻接
- [ ] val-FAPE
  - 基于局部帧对齐
  - 口袋加权

**接口设计**:

```python
def compute_pocket_irmsd(pred_xyz, true_xyz, pocket_mask) -> float
def compute_chi1_accuracy(pred_angles, true_angles, residue_types, threshold=20) -> float
def compute_clash_percentage(xyz, bond_graph) -> float
def compute_fape(pred_xyz, true_xyz, pred_frames, true_frames, w_res) -> float
```

**依赖**:

- scipy.spatial (Kabsch)
- modules.rigid_utils (FAPE)

---

#### 2.3 损失函数 (`modules/losses.py`)

**功能需求**:

- [ ] FAPE 损失
  - 基于局部帧对齐
  - 外层口袋加权
  - 复用 OpenFold 实现
- [ ] 扭转角损失
  - wrap cosine: 1 - cos(Δθ)
  - 残基级加权
- [ ] 距离损失
  - Pair-wise 重原子距离
  - 权重: max(w_i, w_j)
- [ ] 碰撞惩罚
  - 非键原子最小距离
  - Soft penalty

**接口设计**:

```python
def fape_loss(pred_xyz, true_xyz, pred_frames, true_frames, w_res) -> Tensor
def torsion_loss(pred_angles, true_angles, w_res) -> Tensor
def distance_loss(pred_xyz, true_xyz, w_pair) -> Tensor
def clash_penalty(xyz, bond_graph) -> Tensor
```

**损失权重**:

```python
loss = 1.0 * L_fape + 1.0 * L_torsion + 0.1 * L_dist + 0.1 * L_clash
# 口袋权重 warmup: κ(step) = min(1.0, step/2000)
```

---

#### 2.4 训练脚本 (`scripts/train_stage1_ipa.py`)

**功能需求**:

- [ ] 模型初始化
  - ESM-2 冻结 + Adapter
  - FlashIPA 几何分支
  - LigandConditioner
  - Torsion Head + FK 解码器
- [ ] 优化器配置
  - AdamW (lr=1e-4, wd=0.05)
  - Cosine LR (warmup=1000)
  - Grad clip = 1.0
- [ ] 训练循环
  - 前向: ESM → Adapter → IPA → Cond → Decoder → FK
  - 损失: FAPE + 扭转 + 距离 + clash
  - 口袋warmup: 0→1 (2000 steps)
  - 数据增强: step>5000启用
- [ ] 验证与早停
  - 主指标: 口袋 iRMSD
  - 次指标: val-FAPE, χ1命中率, clash%
  - 早停: patience=20 epochs
- [ ] 日志与可视化
  - Tensorboard
  - 定期保存checkpoint
  - 验证集结构可视化

**训练配置** (`configs/stage1_ipa.yaml`):

```yaml
# 模型
model:
  c_s: 384
  c_z: 128
  d_lig: 64
  heads: 8
  depth: 3
  no_qk_points: 8
  no_v_points: 12
  z_factor_rank: 16

# 训练
train:
  lr: 1.0e-4
  weight_decay: 0.05
  warmup_steps: 1000
  max_epochs: 100
  batch_size: 4
  precision: bf16
  grad_clip: 1.0
  dropout: 0.1
  ema: 0.999
  
# 损失
loss:
  w_fape: 1.0
  w_torsion: 1.0
  w_dist: 0.1
  w_clash: 0.1
  pocket_warmup: 2000
  
# 数据增强
augmentation:
  enable_step: 5000
  rot_max_deg: 5.0
  trans_std: 0.5
  
# 限制
limits:
  max_lig_points: 128
  rot_clip_deg: 15.0
  trans_clip: 1.5
  
# 验证
validation:
  interval: 1  # epochs
  early_stop_metric: pocket_irmsd
  patience: 20
  save_top_k: 3
```

---

### Phase 3: 环境配置与依赖 (预计0.5天)

#### 3.1 依赖安装

**新增依赖**:

```bash
# FlashIPA 相关
pip install flash-attn>=2.0.0
pip install git+https://github.com/flagshippioneering/flash_ipa.git

# OpenFold 工具
pip install git+https://github.com/aqlaboratory/openfold.git

# 已有依赖
# - torch>=2.0.0
# - biopython
# - rdkit
# - numpy
# - scipy
```

#### 3.2 环境验证

- [ ] 验证 FlashAttention 安装 (Linux + CUDA)
- [ ] 验证 FlashIPA 可导入
- [ ] 验证 OpenFold rigid_utils 可用
- [ ] 测试 A100 显存占用 (单卡 batch=4)

---

### Phase 4: 测试与验证 (预计1天)

#### 4.1 单元测试

- [ ] 配体token构建测试
  - 测试关键原子识别
  - 测试探针生成
  - 测试重要性采样
- [ ] 帧工具测试
  - 测试三点构帧正确性
  - 测试增量裁剪
  - 测试噪声注入
- [ ] 模型前向测试
  - 测试 FlashIPA 前向
  - 测试配体条件化
  - 测试端到端推理

#### 4.2 过拟合测试

- [ ] 单样本过拟合
  - 用1个PDB训练至loss→0
  - 验证所有模块可学习
- [ ] 小数据集验证
  - 用10个PDB训练
  - 观察指标收敛

#### 4.3 全量训练

- [ ] CASF-2016 完整训练
  - 监控4项指标
  - 验证早停生效
  - 可视化验证集

---

## 📈 验收标准

### 数据指标

| 指标                 | 目标值   | 说明                          |
| -------------------- | -------- | ----------------------------- |
| **val-FAPE**   | < 2.0 Å | 局部帧对齐误差                |
| **口袋 iRMSD** | < 1.5 Å | 口袋局部对齐RMSD (早停主指标) |
| **χ1 命中率** | > 70%    | ±20° 准确率                 |
| **Clash%**     | < 5%     | 碰撞原子对比例                |

### 训练稳定性

- [ ] 损失曲线平滑下降
- [ ] 验证指标稳定收敛
- [ ] 无 NaN/Inf
- [ ] 显存占用 < 70GB (A100-80GB)

### 代码质量

- [ ] 所有模块有docstring
- [ ] 关键函数有类型注解
- [ ] 代码通过 pylint (score>8.0)
- [ ] 单元测试覆盖率 > 80%

---

## 🚀 执行计划

### Week 1: 核心模块 (Day 1-3)

**Day 1**:

- [ ] `utils/ligand_utils.py` (配体token)
- [ ] `modules/rigid_utils.py` (刚体工具)
- [ ] 单元测试

**Day 2**:

- [ ] `modules/edge_embed.py` (边嵌入)
- [ ] `models/ipa.py` (FlashIPA)
- [ ] 模型前向测试

**Day 3**:

- [ ] `models/ligand_condition.py` (配体条件化)
- [ ] 端到端推理测试

### Week 1: 数据与训练 (Day 4-5)

**Day 4**:

- [ ] `data/dataset_ipa.py` (数据加载器)
- [ ] `utils/metrics.py` (评估指标)
- [ ] `modules/losses.py` (损失函数)

**Day 5**:

- [ ] `scripts/train_stage1_ipa.py` (训练脚本)
- [ ] `configs/stage1_ipa.yaml` (配置文件)
- [ ] 单样本过拟合测试

### Week 2: 训练与调优 (Day 6-7)

**Day 6**:

- [ ] 环境配置 (Linux + FlashIPA)
- [ ] 小数据集验证 (10样本)

**Day 7**:

- [ ] 全量训练启动
- [ ] 监控指标与调优

---

## 📝 文档更新计划

### 实现文档

- [ ] `docs/implementation/IPA_ARCHITECTURE.md`

  - FlashIPA 架构详解
  - 配体条件化设计
  - 帧更新机制
- [ ] `docs/implementation/TRAINING_GUIDE.md`

  - 训练流程说明
  - 超参调优建议
  - 常见问题排查

### 进度日志

- [ ] `docs/logs/STAGE1_PROGRESS.md`
  - 每日进度记录
  - 实验结果汇总
  - 问题与解决方案

### 代码注释

- [ ] 所有核心类/函数有详细 docstring
- [ ] 关键算法有行内注释
- [ ] 复杂逻辑有设计说明

---

## 🔧 技术栈总结

### 核心依赖

| 库             | 版本      | 用途           |
| -------------- | --------- | -------------- |
| PyTorch        | ≥2.0.0   | 深度学习框架   |
| FlashAttention | ≥2.0.0   | 高效注意力内核 |
| FlashIPA       | latest    | 线性扩展IPA    |
| OpenFold       | latest    | 刚体工具/FAPE  |
| RDKit          | ≥2023.03 | 配体特征提取   |
| BioPython      | ≥1.80    | 蛋白结构解析   |
| ESM            | ≥2.0.0   | 蛋白语言模型   |

### 计算资源

**开发环境**: Mac (原型验证)
**训练环境**: Linux + 2×A100-80GB
**推荐配置**:

- CUDA ≥11.8
- cuDNN ≥8.0
- 系统内存 ≥128GB
- SSD 存储 ≥500GB

---

## 📚 参考文献

1. **AlphaFold2**: Jumper et al. (Nature 2021) - IPA 原始设计
2. **FlashIPA**: arXiv:2505.11580 - 线性扩展IPA实现
3. **FlashAttention**: Dao et al. (NeurIPS 2022) - 高效注意力机制
4. **OpenFold**: https://github.com/aqlaboratory/openfold - 工业级实现
5. **BINDRAE理论**: `docs/理论/理论与参考.md` - 项目理论纲领

---

## ⚠️ 注意事项

### 关键约束

1. **Mac禁止训练**: 只用于开发调试，正式训练必须在Linux
2. **显存管理**: 使用因子化边嵌入，避免NxN矩阵
3. **数值稳定**: 每层裁剪帧增量，全局梯度裁剪
4. **共价边**: 第一版不做，留待ablation
5. **口袋权重**: 从0开始warmup，避免初期过拟合口袋

### 常见陷阱

- ❌ 忘记冻结ESM-2
- ❌ 验证集也加数据增强
- ❌ gamma/beta初始化错误导致FiLM失效
- ❌ 帧更新不裁剪导致数值爆炸
- ❌ FAPE不用局部帧对齐

---

## 🎯 下一步工作 (Stage-2)

**在 Stage-1 收敛后**:

1. **数据准备**

   - 获取 apo-holo 配对 (AHoJ/PLINDER)
   - 构建三元组: (P_apo, L, P_holo)
2. **模型扩展**

   - Flow Matching / Schrödinger Bridge
   - 潜空间连续路径学习
3. **评估**

   - 构象轨迹质量
   - 中间态合理性
   - 终态收敛性

---

**最后更新**: 2025-10-25
**负责人**: BINDRAE Team
**状态**: 待开工 → 核心模块开发中
