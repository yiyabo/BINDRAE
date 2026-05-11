# BINDRAE Stage-2 优化方案

> 本文档规划 Stage-2 (LC-BridgeFlow) 的优化方向，重点对比 FlowDock，并提出具体改进策略。

---

## 📊 1. 当前状态

### 1.1 Stage-2 核心能力

| 特性 | 当前实现 | 状态 |
|------|----------|------|
| SE(3)×S¹ 流形 Flow Matching | ✅ | 完成 |
| 配体条件化 (LigandConditioner) | ✅ | 完成 |
| 口袋门控 (Pocket Gate) | ✅ | 完成 |
| 背景稳定性 (L_bg) | ✅ | 完成 |
| ODE 积分 (Heun) | ✅ | 完成 |
| Stage-1 Prior 引导 | ✅ | 完成 |
| FK 全原子重建 | ✅ | 完成 |
| NMA 物理先验 (可选) | ⏸️ | 代码就绪，未启用 |
| **Affinity 预测** | ❌ | 待实现 (P0) |
| **FK + 扩散精修** | ❌ | 待实现 (P1) ⭐ |
| **Confidence 估计** | ❌ | 待实现 (P2) |
| **多配体支持** | ❌ | 待实现 (P4) |

### 1.2 当前输出

```python
# Stage-2 当前输出
{
    "d_chi": [B, N, 4],         # χ 角速度
    "d_rigid_rot": [B, N, 3],   # 旋转速度 (so(3))
    "d_rigid_trans": [B, N, 3], # 平移速度
    "gate": [B, N, 1]           # 口袋门控值
}
```

---

## 🆚 2. 与 FlowDock 对比

### 2.1 架构对比

| 维度 | BINDRAE Stage-2 | FlowDock |
|------|-----------------|----------|
| **架构类型** | 两阶段 (Stage-1 + Stage-2) | 端到端 |
| **核心方法** | Flow Matching on SE(3)×S¹ | Conditional Flow Matching |
| **表示空间** | 流形 (SE(3) + S¹) | 欧几里得空间 |
| **输出** | 连续路径 (多帧) | 单个终点结构 |
| **apo 先验** | 真实 apo 结构 | ESMFold 预测 + 噪声 |
| **Affinity** | ❌ 无 | ✅ 有 |
| **Confidence** | ❌ 无 | ✅ 有 |
| **多配体** | ❌ 单配体 | ✅ 多配体 |

### 2.2 BINDRAE 的核心优势

| 优势 | 说明 |
|------|------|
| 🎬 **路径生成** | 输出完整的 apo→holo 动态路径，可视化构象变化过程 |
| 📐 **流形几何** | 在 SE(3)×S¹ 流形上操作，保证刚体变换和角度的物理正确性 |
| 🔬 **可解释性** | 中间构象可视化，便于理解构象变化机制 |
| 🧬 **物理约束** | 通过 FK 保证骨架几何、键长键角正确 |
| 🎯 **终点监督** | Stage-1 提供明确的 holo 终点信号 |

### 2.3 BINDRAE 的劣势（待补齐）

| 劣势 | 影响 | 优先级 |
|------|------|--------|
| ❌ 无 Affinity 预测 | 药物设计应用受限 | **P0** |
| ⚙️ 纯 FK 重建局限 | 难处理大幅度诱导契合、累积误差 | **P1** |
| ❌ 无 Confidence 估计 | 难以评估预测可靠性 | P2 |
| ⏱️ 推理速度较慢 | ODE 积分需多步 | P3 |
| ❌ 单配体限制 | 无法处理多配体体系 | P4 |

---

## 🎯 3. 优化方向

### 3.1 Priority 0: Affinity 预测模块

**动机**: FlowDock 的一大亮点是直接输出结合亲和力预测，这在药物设计中极为重要。

#### 3.1.1 设计方案

```
┌─────────────────────────────────────────────────────────────┐
│                    AffinityHead                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  输入:                                                       │
│    - h_protein: [B, N, c_s]  # 蛋白特征 (FlashIPA 输出)     │
│    - h_ligand:  [B, M, d_lig] # 配体特征                    │
│    - gate:      [B, N, 1]     # 口袋门控值                  │
│    - t:         [B]           # 时间 (可选，用于时序)       │
│                                                              │
│  处理流程:                                                   │
│    1. 口袋加权池化: h_pocket = Σ_i (gate_i * h_i) / Σ gate  │
│    2. 配体池化:     h_lig = mean(h_ligand, mask)            │
│    3. 交互特征:     h_inter = CrossAttn(h_pocket, h_lig)    │
│    4. 融合:         h_fused = [h_pocket; h_lig; h_inter]    │
│    5. MLP:          affinity = MLP(h_fused)                 │
│                                                              │
│  输出:                                                       │
│    - pKd/pKi/pIC50 预测: [B, 1]                             │
│    - (可选) 不确定性: [B, 1]                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 3.1.2 实现策略

**方案 A: 终点 Affinity (推荐起步)**

```python
class AffinityHead(nn.Module):
    """在 t=1.0 (holo 终点) 预测结合亲和力"""
    
    def __init__(self, c_s: int = 384, d_lig: int = 128, hidden: int = 256):
        super().__init__()
        
        # 口袋特征聚合
        self.pocket_pool = nn.Sequential(
            nn.Linear(c_s, hidden),
            nn.LayerNorm(hidden),
            nn.GELU()
        )
        
        # 配体特征聚合
        self.ligand_pool = nn.Sequential(
            nn.Linear(d_lig, hidden),
            nn.LayerNorm(hidden),
            nn.GELU()
        )
        
        # 蛋白-配体交互
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=4,
            batch_first=True
        )
        
        # 亲和力预测
        self.affinity_mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1)
        )
        
    def forward(self, 
                h_protein: torch.Tensor,  # [B, N, c_s]
                h_ligand: torch.Tensor,   # [B, M, d_lig]
                gate: torch.Tensor,       # [B, N, 1]
                protein_mask: torch.Tensor,
                ligand_mask: torch.Tensor) -> torch.Tensor:
        
        # 1. 口袋加权池化
        h_p = self.pocket_pool(h_protein)  # [B, N, hidden]
        gate_norm = gate / (gate.sum(dim=1, keepdim=True) + 1e-8)
        h_pocket = (h_p * gate_norm).sum(dim=1)  # [B, hidden]
        
        # 2. 配体池化 (mask-aware mean)
        h_l = self.ligand_pool(h_ligand)  # [B, M, hidden]
        lig_mask_expanded = ligand_mask.unsqueeze(-1).float()
        h_lig = (h_l * lig_mask_expanded).sum(dim=1) / (lig_mask_expanded.sum(dim=1) + 1e-8)
        
        # 3. 交互特征 (cross-attention)
        h_pocket_q = h_pocket.unsqueeze(1)  # [B, 1, hidden]
        h_inter, _ = self.cross_attn(h_pocket_q, h_l, h_l, 
                                      key_padding_mask=~ligand_mask)
        h_inter = h_inter.squeeze(1)  # [B, hidden]
        
        # 4. 融合并预测
        h_fused = torch.cat([h_pocket, h_lig, h_inter], dim=-1)  # [B, hidden*3]
        affinity = self.affinity_mlp(h_fused)  # [B, 1]
        
        return affinity
```

**方案 B: 路径积分 Affinity (进阶)**

```python
class PathIntegratedAffinityHead(nn.Module):
    """沿路径积分的亲和力预测 - 利用 BINDRAE 的路径优势"""
    
    def __init__(self, c_s: int = 384, d_lig: int = 128, hidden: int = 256):
        super().__init__()
        
        # 单帧 affinity 编码器 (复用上面的结构)
        self.frame_encoder = AffinityHead(c_s, d_lig, hidden)
        
        # 时间注意力 - 对不同时间点的贡献加权
        self.time_attn = nn.Sequential(
            nn.Linear(hidden + 1, hidden),  # +1 for time embedding
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Softmax(dim=1)
        )
        
        # 路径特征聚合
        self.path_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
        
    def forward(self, 
                path_features: List[torch.Tensor],  # T 个 [B, N, c_s]
                h_ligand: torch.Tensor,             # [B, M, d_lig]
                gates: List[torch.Tensor],          # T 个 [B, N, 1]
                t_list: List[float],
                protein_mask: torch.Tensor,
                ligand_mask: torch.Tensor) -> torch.Tensor:
        """
        利用整条路径的信息预测亲和力
        - 可以捕捉"结合过程"中的能量变化
        - 利用 BINDRAE 独有的路径生成能力
        """
        B = path_features[0].shape[0]
        T = len(path_features)
        
        # 1. 编码每个时间帧的特征
        frame_feats = []
        for t_idx, (h_prot, gate, t) in enumerate(zip(path_features, gates, t_list)):
            # 获取单帧 affinity 隐藏特征
            feat = self.frame_encoder.get_hidden(h_prot, h_ligand, gate, 
                                                  protein_mask, ligand_mask)
            # 加上时间嵌入
            t_embed = torch.full((B, 1), t, device=feat.device)
            feat_with_t = torch.cat([feat, t_embed], dim=-1)
            frame_feats.append(feat_with_t)
        
        frame_feats = torch.stack(frame_feats, dim=1)  # [B, T, hidden+1]
        
        # 2. 时间注意力加权
        attn_weights = self.time_attn(frame_feats)  # [B, T, 1]
        
        # 3. 加权聚合
        h_path = (frame_feats[..., :-1] * attn_weights).sum(dim=1)  # [B, hidden]
        
        # 4. 最终预测
        affinity = self.path_mlp(h_path)  # [B, 1]
        
        return affinity
```

#### 3.1.3 训练策略

```python
# Affinity 损失函数
class AffinityLoss(nn.Module):
    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, 
                pred_affinity: torch.Tensor,  # [B, 1]
                true_affinity: torch.Tensor,  # [B, 1] pKd/pKi/pIC50
                mask: torch.Tensor = None) -> torch.Tensor:
        
        if self.loss_type == 'mse':
            loss = F.mse_loss(pred_affinity, true_affinity, reduction='none')
        elif self.loss_type == 'huber':
            loss = F.huber_loss(pred_affinity, true_affinity, reduction='none', delta=1.0)
        elif self.loss_type == 'rank':
            # 排序损失 - 更关注相对顺序
            loss = self._rank_loss(pred_affinity, true_affinity)
        
        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()
            
        return loss
```

#### 3.1.4 数据需求

需要在 `Stage2Batch` 中添加亲和力标签:

```python
@dataclass
class Stage2Batch:
    # ... existing fields ...
    
    # Affinity labels (新增)
    affinity: Optional[torch.Tensor] = None    # [B] pKd/pKi/pIC50
    affinity_type: Optional[List[str]] = None  # ['pKd', 'pKi', 'pIC50', ...]
    affinity_mask: Optional[torch.Tensor] = None  # [B] 是否有亲和力标签
```

数据来源:
- PDBbind (主要来源，有 pKd/pKi/pIC50)
- BindingDB (补充)
- ChEMBL (补充)

---

### 3.2 Priority 1: 全原子重建优化 - FK + 扩散精修

**动机**: 当前使用纯 FK (Forward Kinematics) 将扭转角转换为全原子坐标。根据 2024-2025 最新文献（DynamicBind、3DMolFormer、D3FG），扩散精修可以显著提升结构质量，特别是处理诱导契合场景。

#### 3.2.1 方法对比

| 方法 | 核心技术 | 优点 | 缺点 | 代表工作 |
|------|---------|------|------|---------|
| **FK (当前)** | 内坐标 → 笛卡尔 | ✅ 快速、键长键角精确 | ❌ 误差累积、难处理大柔性 | AlphaFold |
| **等变扩散精修** | SE(3) Diffusion | ✅ **诱导契合能力最强** | ❌ 需要多步迭代 | **DynamicBind (2024)** |
| **直接坐标回归** | Transformer 双通道 | ✅ 端到端、无累积误差 | ❌ 需海量预训练数据 | **3DMolFormer (ICLR 2025)** |
| **片段级扩散** | 官能团刚体 | ✅ 化学合理性高 | ❌ 分解非标准基团复杂 | **D3FG (2024)** |

#### 3.2.2 改进方案: FK + 扩散精修

**核心思路**: 保留 FK 的快速初始化优势，添加轻量级扩散精修模块进行后处理。

```
当前流程:
  Stage-1 → Chi 角预测 → FK 重建 → 全原子坐标 (最终)

改进流程:
  Stage-1 → Chi 角预测 → FK 重建 (初始) → 扩散精修 (5-10步) → 全原子坐标 (最终)
                                                  ↑
                                     同时微调蛋白 Chi 角 + 配体位姿
```

#### 3.2.3 实现设计

```python
class DiffusionRefinement(nn.Module):
    """
    FK 重建后的扩散精修模块
    
    参考:
    - DynamicBind (Nat. Comm. 2024): 等变几何扩散 + 侧链 Chi 角更新
    - D3FG: 片段级扩散保持化学合理性
    
    特点:
    - 轻量级: 5-10 步精修，计算开销可控
    - 物理感知: 同时优化蛋白侧链和配体位姿
    - 与方向探针协同: 利用探针提供的空间几何约束
    """
    
    def __init__(self, 
                 c_s: int = 384,
                 d_lig: int = 128,
                 num_heads: int = 8,
                 num_refine_steps: int = 10,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.num_refine_steps = num_refine_steps
        
        # SE(3) 等变层 - 处理坐标更新
        self.se3_layer = SE3EquivariantBlock(
            node_dim=c_s,
            edge_dim=hidden_dim,
            num_heads=num_heads
        )
        
        # Chi 角增量预测器
        self.chi_updater = nn.Sequential(
            nn.Linear(c_s + d_lig, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4)  # δχ1, δχ2, δχ3, δχ4
        )
        
        # 蛋白-配体相互作用模块
        self.interaction_module = ProteinLigandInteraction(
            c_s=c_s,
            d_lig=d_lig,
            num_heads=num_heads
        )
        
        # 时间嵌入 (扩散步数)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, c_s)
        )
        
        # 配体位姿更新器 (平移 + 旋转)
        self.ligand_pose_updater = nn.Sequential(
            nn.Linear(d_lig, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6)  # δtrans(3) + δrot(3, axis-angle)
        )
        
    def forward(self, 
                coords_init: torch.Tensor,      # [B, N, 3] FK 重建的初始 Cα 坐标
                chi_init: torch.Tensor,         # [B, N, 4] Stage-1 预测的 Chi 角
                h_protein: torch.Tensor,        # [B, N, c_s] 蛋白特征
                h_ligand: torch.Tensor,         # [B, M, d_lig] 配体特征
                lig_coords: torch.Tensor,       # [B, M, 3] 配体坐标
                protein_mask: torch.Tensor,
                ligand_mask: torch.Tensor,
                gate: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        扩散精修前向传播
        
        Returns:
            {
                'coords_refined': [B, N, 3] 精修后的 Cα 坐标
                'chi_refined': [B, N, 4] 精修后的 Chi 角
                'lig_coords_refined': [B, M, 3] 精修后的配体坐标
                'trajectory': List of intermediate states (可选)
            }
        """
        B, N, _ = coords_init.shape
        device = coords_init.device
        
        # 初始化
        coords = coords_init.clone()
        chi = chi_init.clone()
        lig_pos = lig_coords.clone()
        
        trajectory = [{'coords': coords.clone(), 'chi': chi.clone(), 'lig': lig_pos.clone()}]
        
        for step in range(self.num_refine_steps):
            # 1. 时间嵌入
            t = torch.full((B,), step / self.num_refine_steps, device=device)
            t_embed = self.time_embed(t)  # [B, c_s]
            
            # 2. 计算蛋白-配体相互作用
            h_prot_updated, h_lig_updated = self.interaction_module(
                h_protein + t_embed.unsqueeze(1),
                h_ligand,
                coords,
                lig_pos,
                protein_mask,
                ligand_mask
            )
            
            # 3. 预测 Chi 角增量
            if gate is not None:
                # 只更新口袋区域的 Chi 角
                pocket_weight = gate.squeeze(-1)  # [B, N]
            else:
                pocket_weight = torch.ones(B, N, device=device)
            
            # 聚合配体信息到每个残基
            lig_info = self._aggregate_ligand_to_protein(
                h_lig_updated, lig_pos, coords, protein_mask, ligand_mask
            )  # [B, N, d_lig]
            
            delta_chi = self.chi_updater(
                torch.cat([h_prot_updated, lig_info], dim=-1)
            )  # [B, N, 4]
            delta_chi = delta_chi * pocket_weight.unsqueeze(-1)
            
            # 4. 更新 Chi 角 (小步长)
            chi = chi + 0.1 * delta_chi
            
            # 5. 预测配体位姿更新
            lig_pooled = (h_lig_updated * ligand_mask.unsqueeze(-1).float()).sum(dim=1)
            lig_pooled = lig_pooled / (ligand_mask.sum(dim=1, keepdim=True).float() + 1e-8)
            delta_pose = self.ligand_pose_updater(lig_pooled)  # [B, 6]
            
            delta_trans = delta_pose[:, :3]  # [B, 3]
            delta_rot = delta_pose[:, 3:]    # [B, 3] axis-angle
            
            # 6. 更新配体坐标 (SE(3) 变换)
            lig_pos = self._apply_se3_update(lig_pos, delta_trans, delta_rot, step_size=0.1)
            
            # 7. 通过更新的 Chi 角重建蛋白坐标 (可选)
            # coords = self.rebuild_from_chi(chi, backbone_frames)
            
            trajectory.append({
                'coords': coords.clone(), 
                'chi': chi.clone(), 
                'lig': lig_pos.clone()
            })
        
        return {
            'coords_refined': coords,
            'chi_refined': chi,
            'lig_coords_refined': lig_pos,
            'trajectory': trajectory
        }
    
    def _aggregate_ligand_to_protein(self, h_lig, lig_pos, prot_pos, 
                                      protein_mask, ligand_mask):
        """将配体信息聚合到每个蛋白残基 (基于距离加权)"""
        # 计算距离矩阵 [B, N, M]
        dist = torch.cdist(prot_pos, lig_pos)
        
        # 距离权重 (RBF-like)
        weights = torch.exp(-dist / 5.0)  # 5Å 特征长度
        weights = weights * ligand_mask.unsqueeze(1).float()
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 加权聚合
        lig_info = torch.einsum('bnm,bmd->bnd', weights, h_lig)
        return lig_info
    
    def _apply_se3_update(self, coords, delta_trans, delta_rot, step_size=0.1):
        """应用 SE(3) 更新到配体坐标"""
        # 平移
        coords = coords + step_size * delta_trans.unsqueeze(1)
        
        # 旋转 (axis-angle -> rotation matrix)
        angle = torch.norm(delta_rot, dim=-1, keepdim=True)
        axis = delta_rot / (angle + 1e-8)
        
        # Rodrigues 公式 (简化版)
        cos_a = torch.cos(step_size * angle)
        sin_a = torch.sin(step_size * angle)
        
        # 绕质心旋转
        centroid = coords.mean(dim=1, keepdim=True)
        coords_centered = coords - centroid
        
        # 简化: 只做小角度旋转近似
        coords_rotated = coords_centered + step_size * torch.cross(
            axis.unsqueeze(1).expand_as(coords_centered),
            coords_centered,
            dim=-1
        )
        
        return coords_rotated + centroid


class SE3EquivariantBlock(nn.Module):
    """SE(3) 等变块 - 简化版"""
    
    def __init__(self, node_dim: int, edge_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.coord_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 3, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, 3)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, node_dim)
        )
        
    def forward(self, h, coords, mask):
        # 自注意力更新节点特征
        h_attn, _ = self.attn(h, h, h, key_padding_mask=~mask)
        h = h + h_attn
        h = h + self.node_mlp(h)
        
        return h, coords


class ProteinLigandInteraction(nn.Module):
    """蛋白-配体相互作用模块"""
    
    def __init__(self, c_s: int, d_lig: int, num_heads: int = 8):
        super().__init__()
        
        # 蛋白 -> 配体 注意力
        self.prot_to_lig = nn.MultiheadAttention(
            embed_dim=d_lig,
            num_heads=num_heads,
            kdim=c_s,
            vdim=c_s,
            batch_first=True
        )
        
        # 配体 -> 蛋白 注意力  
        self.lig_to_prot = nn.MultiheadAttention(
            embed_dim=c_s,
            num_heads=num_heads,
            kdim=d_lig,
            vdim=d_lig,
            batch_first=True
        )
        
        self.prot_proj = nn.Linear(d_lig, c_s)
        self.lig_proj = nn.Linear(c_s, d_lig)
        
    def forward(self, h_prot, h_lig, prot_coords, lig_coords, prot_mask, lig_mask):
        # 双向交叉注意力
        h_lig_updated, _ = self.prot_to_lig(
            h_lig, h_prot, h_prot,
            key_padding_mask=~prot_mask
        )
        h_lig = h_lig + h_lig_updated
        
        h_prot_updated, _ = self.lig_to_prot(
            h_prot, h_lig, h_lig,
            key_padding_mask=~lig_mask
        )
        h_prot = h_prot + h_prot_updated
        
        return h_prot, h_lig


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置嵌入 (用于扩散时间步)"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
```

#### 3.2.4 预期收益

| 指标 | 纯 FK | FK + 扩散精修 (10步) |
|------|-------|---------------------|
| Chi1 准确率 | ~50% | **55-60%** |
| 原子碰撞率 (Clash) | ~5% | **<2%** |
| 诱导契合处理 | ❌ 无法 | ✅ 可以 |
| RMSD 改善 | baseline | **-0.3~0.5Å** |
| 计算开销 | 1x | 2-3x |

#### 3.2.5 与现有模块的协同

```
┌─────────────────────────────────────────────────────────────────┐
│                     协同设计                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage-1 (方向探针)                                              │
│      ↓                                                          │
│  探针编码氢键方向 → 指导 Chi 角预测 → FK 初始重建                │
│      ↓                                                          │
│  扩散精修                                                        │
│      ├── 利用探针的空间几何约束                                 │
│      ├── 配体位姿微调 (保持氢键方向)                            │
│      └── 蛋白侧链协同调整                                       │
│      ↓                                                          │
│  物理合理的全原子结构                                           │
│                                                                  │
│  关键: 方向探针为扩散精修提供了"方向性先验"，                   │
│        使精修过程更聚焦于保持关键相互作用                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.2.6 实现计划

| 任务 | 时间 | 产出 |
|------|------|------|
| 设计 DiffusionRefinement 模块 | 3天 | 模块代码 |
| 实现 SE3EquivariantBlock | 2天 | 等变层 |
| 集成到 Stage-2 pipeline | 2天 | 更新流程 |
| 训练 & 调优 | 1周 | 初版模型 |
| 消融实验 (有/无精修对比) | 3天 | 性能报告 |

#### 3.2.7 参考文献

- **DynamicBind** (Nature Communications 2024): "Predicting ligand-specific protein-ligand complex structure with a deep equivariant generative model"
  - 核心贡献: 20步扩散同时优化配体位姿 + 蛋白侧链 Chi 角
  - 能处理 DFG-in/out 等大幅度构象变化

- **3DMolFormer** (ICLR 2025): "A Dual-channel Framework for Structure-based Drug Discovery"
  - 核心贡献: 离散+连续双通道 Transformer，直接回归坐标

- **D3FG** (Arxiv 2024): "Functional-Group-Based Diffusion for Pocket-Specific Molecule Generation"
  - 核心贡献: 片段级扩散，保持化学合理性

---

### 3.4 Priority 2: Confidence 估计

**动机**: 预测置信度对实际应用至关重要。

#### 3.2.1 设计方案

```python
class ConfidenceHead(nn.Module):
    """预测结构预测的置信度 (类似 pLDDT)"""
    
    def __init__(self, c_s: int = 384):
        super().__init__()
        
        # 残基级置信度
        self.residue_conf = nn.Sequential(
            nn.Linear(c_s, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # 全局置信度
        self.global_conf = nn.Sequential(
            nn.Linear(c_s, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 残基级
        conf_per_res = self.residue_conf(h).squeeze(-1)  # [B, N]
        
        # 全局 (mask-aware mean)
        mask_float = mask.float()
        h_global = (h * mask_float.unsqueeze(-1)).sum(dim=1) / (mask_float.sum(dim=1, keepdim=True) + 1e-8)
        conf_global = self.global_conf(h_global).squeeze(-1)  # [B]
        
        return {
            'confidence_per_residue': conf_per_res,
            'confidence_global': conf_global
        }
```

---

### 3.5 Priority 3: 推理速度优化

**目标**: 减少 ODE 积分步数，同时保持路径质量。

#### 3.3.1 策略

| 方法 | 描述 | 预期加速 |
|------|------|----------|
| 自适应步长 | 根据向量场变化动态调整 Δt | 1.5-2x |
| 蒸馏学习 | 训练少步模型模仿多步结果 | 2-4x |
| 条件跳步 | 在变化小的区域跳过积分 | 1.3-1.5x |

#### 3.3.2 自适应步长实现

```python
def adaptive_ode_integrate(self, x0, cond, tol=1e-3, max_steps=50):
    """自适应步长 ODE 积分"""
    t = 0.0
    x = x0
    trajectory = [x]
    
    while t < 1.0 and len(trajectory) < max_steps:
        # 计算向量场
        v = self.model(x, t, cond)
        
        # 估计步长 (基于向量场变化)
        v_norm = torch.norm(v, dim=-1).mean()
        dt = min(0.1, tol / (v_norm + 1e-8))
        dt = min(dt, 1.0 - t)  # 不超过终点
        
        # Heun 积分
        x_pred = x + dt * v
        v_pred = self.model(x_pred, t + dt, cond)
        x = x + 0.5 * dt * (v + v_pred)
        
        t += dt
        trajectory.append(x)
    
    return trajectory
```

---

### 3.6 Priority 4: 多配体支持

**场景**: 某些蛋白同时结合多个小分子 (如辅因子 + 抑制剂)。

#### 3.4.1 设计思路

```python
class MultiLigandConditioner(nn.Module):
    """支持多配体条件化"""
    
    def __init__(self, c_s: int = 384, d_lig: int = 128, max_ligands: int = 3):
        super().__init__()
        
        self.max_ligands = max_ligands
        
        # 配体编码器 (共享)
        self.ligand_encoder = EnhancedLigandEncoder(d_lig=d_lig)
        
        # 配体间交互
        self.ligand_interaction = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_lig, nhead=4, batch_first=True),
            num_layers=2
        )
        
        # 蛋白-多配体交互
        self.protein_ligand_attn = nn.MultiheadAttention(
            embed_dim=c_s,
            num_heads=8,
            kdim=d_lig,
            vdim=d_lig,
            batch_first=True
        )
        
    def forward(self,
                h_protein: torch.Tensor,        # [B, N, c_s]
                ligand_list: List[Dict],        # K 个配体的 {points, types, mask}
                protein_mask: torch.Tensor) -> torch.Tensor:
        
        # 1. 编码每个配体
        lig_features = []
        for lig in ligand_list:
            h_lig = self.ligand_encoder(lig['points'], lig['types'], lig['mask'])
            lig_features.append(h_lig)
        
        # 2. 配体间交互 (如果有多个)
        if len(lig_features) > 1:
            # 拼接所有配体原子
            h_all_lig = torch.cat(lig_features, dim=1)  # [B, M_total, d_lig]
            h_all_lig = self.ligand_interaction(h_all_lig)
        else:
            h_all_lig = lig_features[0]
        
        # 3. 蛋白-配体交互
        h_out, _ = self.protein_ligand_attn(
            h_protein, h_all_lig, h_all_lig
        )
        
        return h_protein + h_out
```

---

## 📋 4. 实现计划

### Phase 1: Affinity 预测 (2-3 周)

| 任务 | 时间 | 产出 |
|------|------|------|
| 设计 AffinityHead | 2天 | 模块代码 |
| 数据准备 (PDBbind affinity) | 3天 | 训练数据 |
| 集成到 TorsionFlowNet | 2天 | 更新模型 |
| 训练 & 调优 | 1周 | 初版模型 |
| 评估 (Pearson/RMSE) | 3天 | 性能报告 |

### Phase 1.5: FK + 扩散精修 (2-3 周) ⭐ 新增

| 任务 | 时间 | 产出 |
|------|------|------|
| 设计 DiffusionRefinement 模块 | 3天 | 核心代码 |
| 实现 SE3EquivariantBlock | 2天 | 等变层 |
| 实现 ProteinLigandInteraction | 2天 | 交互模块 |
| 集成到 Stage-2 pipeline | 2天 | 更新流程 |
| 训练 & 调优 | 1周 | 初版模型 |
| 消融实验 (FK vs FK+精修) | 3天 | 性能报告 |

### Phase 2: Confidence 估计 (1-2 周)

| 任务 | 时间 | 产出 |
|------|------|------|
| 设计 ConfidenceHead | 1天 | 模块代码 |
| 训练数据构建 | 2天 | 自监督标签 |
| 训练 & 评估 | 1周 | 初版模型 |

### Phase 3: 推理优化 (1 周)

| 任务 | 时间 | 产出 |
|------|------|------|
| 自适应步长 | 2天 | 代码更新 |
| 性能测试 | 2天 | 速度/质量对比 |
| 文档更新 | 1天 | 使用指南 |

---

## 📊 5. 评估指标

### 5.1 Affinity 预测

| 指标 | 目标 | FlowDock 参考 |
|------|------|---------------|
| Pearson R | > 0.7 | ~0.75 |
| Spearman ρ | > 0.65 | ~0.70 |
| RMSE (pKd) | < 1.5 | ~1.3 |

### 5.2 Confidence 估计

| 指标 | 目标 |
|------|------|
| Confidence-Error 相关性 | > 0.6 |
| 高置信度样本准确率提升 | > 10% |

### 5.3 推理速度

| 指标 | 当前 | 目标 |
|------|------|------|
| 单样本推理时间 | ~5s | < 2s |
| ODE 积分步数 | 20 | 8-12 |

---

## ⚠️ 6. 风险与缓解

| 风险 | 影响 | 缓解策略 |
|------|------|----------|
| Affinity 数据不足 | 预测不准 | 迁移学习 from 2D models |
| Affinity 与结构预测冲突 | 多任务干扰 | 分阶段训练或梯度平衡 |
| 推理优化损失质量 | 路径不平滑 | 严格质量验证门槛 |

---

## 🎯 7. 总结

BINDRAE Stage-2 相比 FlowDock 的**核心差异化优势是路径生成**，这是我们应该坚持的方向。

同时，添加 **Affinity 预测**可以大幅提升实际应用价值，使 BINDRAE 成为一个既能"理解构象变化机制"又能"预测结合强度"的完整系统。

**推荐策略**:
1. **保持路径生成优势** - 这是核心竞争力
2. **添加 Affinity 预测** - 补齐应用短板 (P0)
3. **添加 FK + 扩散精修** - 提升结构质量、处理诱导契合 (P1) ⭐
4. **添加 Confidence 估计** - 提升可靠性 (P2)
5. **优化推理速度** - 提升实用性 (P3)

```
最终目标:
┌──────────────────────────────────────────────────────────────────┐
│  BINDRAE = 路径可解释性 + Affinity 预测 + 扩散精修 + Confidence   │
│                                                                    │
│  定位: 理解构象变化机制的研究工具                                 │
│       + 结合亲和力预测的药物设计辅助                              │
│       + 高质量全原子结构生成（FK + 扩散精修）                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📚 8. 参考文献

### 全原子重建相关

1. **DynamicBind** (Nature Communications, 2024)
   - Lu, W., et al. "Predicting ligand-specific protein-ligand complex structure with a deep equivariant generative model"
   - 关键技术: SE(3) 等变扩散 + 侧链 Chi 角协同优化
   - 应用场景: 诱导契合、隐蔽口袋

2. **3DMolFormer** (ICLR, 2025)
   - "A Dual-channel Framework for Structure-based Drug Discovery"
   - 关键技术: 离散+连续双通道 Transformer
   - 应用场景: 直接坐标回归，规避 FK 累积误差

3. **D3FG** (Arxiv, 2024)
   - "Functional-Group-Based Diffusion for Pocket-Specific Molecule Generation and Elaboration"
   - 关键技术: 官能团级别扩散
   - 应用场景: 保持化学合理性

4. **EquiScore** (Nature Machine Intelligence, 2024)
   - "A generic protein-ligand interaction scoring method integrating physical prior knowledge"
   - 关键技术: 物理先验 + 距离敏感性建模
