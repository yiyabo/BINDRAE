# Stage-1 模型优化方案 (2026.01)

## 📊 一、当前状态

### 1.1 模型架构
- **IPA 深度**: 3 层 (稳定)
- **节点表示**: c_s = 384
- **配体编码**: d_lig = 96 (简单 2 层 MLP)
- **输出头**: torsion_hidden = 256

### 1.2 训练表现
| Epoch | Chi1 | FAPE | Val Loss |
|-------|------|------|----------|
| 1 | 29.0% | 2.617 | 3.53 |
| 4 | 34.2% | 2.572 | 3.45 |
| 7 | 38.0% | 2.560 | 3.40 |
| 预计收敛 | ~50% | ~2.5 | ~3.3 |

### 1.3 已解决的问题
- ✅ `headdim_eff > 256` 维度冲突
- ✅ Shared memory 不足 (num_workers=0)
- ✅ 5层 IPA 不稳定 → 回退到 3 层

---

## 🎯 二、目标

| 阶段 | Chi1 目标 | 难度 |
|------|----------|------|
| 短期 | 50-55% | ⭐ |
| 中期 | 60-70% | ⭐⭐⭐ |
| 长期 | 80%+ | ⭐⭐⭐⭐⭐ |

---

## 🔬 三、最新方法调研 (2024-2026)

### 3.1 几何增强方法

#### CWFBind (2025)
- **核心**: 引入 **局部曲率 (local curvature)** 特征
- **效果**: Pocket 识别 + Pose 准确性提升 14-30%
- **参考**: [arxiv.org/abs/2508.09499](https://arxiv.org/abs/2508.09499)

```python
# 可借鉴：添加几何特征
class GeometryFeatures:
    - local_curvature      # 局部曲率
    - surface_normal       # 表面法线
    - degree_aware_weight  # 度感知权重
    - distance_distribution # 距离分布
```

#### SurfDock (2024)
- **核心**: Surface-informed diffusion 生成模型
- **特点**: 同时考虑表面信息 + 配体柔性
- **参考**: [english.cas.cn](https://english.cas.cn/newsroom/research_news/life/202412/t20241202_893028.shtml)

### 3.2 多任务学习

#### DeepRLI (2025)
- **核心**: 多目标框架 (Docking + Scoring + Screening)
- **架构**: Graph Transformer + 物理约束 + 对比学习
- **效果**: 泛化性显著提升
- **参考**: [pubs.rsc.org](https://pubs.rsc.org/en/content/articlehtml/2025/dd/d4dd00403e)

```python
# 可借鉴：多任务输出头
class MultiTaskHead:
    - chi_head          # 主任务: 侧链角预测
    - contact_head      # 辅助: 接触图预测
    - affinity_head     # 辅助: 亲和力评分
    - clash_head        # 辅助: 碰撞检测
```

### 3.3 交叉注意力增强

#### CAPLA (2023)
- **核心**: Pocket-Ligand 交叉注意力
- **创新**: Dilated convolution 捕获多尺度信息
- **效果**: t-SNE 显示特征区分度大幅提升
- **参考**: [academic.oup.com](https://academic.oup.com/bioinformatics/article/39/2/btad049/6998204)

### 3.4 配体表示增强

#### 预训练配体模型 (2025)
- **方法**: 自监督任务 (扰动重建、距离预测)
- **效果**: Binding affinity + Binding site 识别提升
- **参考**: [bmcbioinformatics.biomedcentral.com](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-025-06064-w)

```python
# 当前配体编码器（过于简单）
class LigandTokenEmbedding:
    # 输入: xyz(3) + atom_types(12) = 15维
    # 处理: 2层 MLP
    # 问题: 没有捕捉配体内部几何关系！

# 建议改进
class EnhancedLigandEncoder:
    # 1. 更丰富的输入特征
    input_features = [
        'xyz',              # 3D 坐标
        'atom_types',       # 原子类型
        'partial_charges',  # 部分电荷
        'hybridization',    # 杂化类型
        'aromaticity',      # 芳香性
        'ring_membership',  # 环成员
        'hydrogen_bonds',   # 氢键供/受体
    ]
    
    # 2. 配体内自注意力
    self_attention = TransformerEncoder(num_layers=2)
    
    # 3. 几何编码
    distance_encoding = RBFDistanceEncoding()
    angle_encoding = AngleEncoding()
```

### 3.5 Co-folding / 联合建模

#### FlowDock (2024)
- **核心**: 从 Apo → Holo 的生成模型
- **特点**: 无需 MSA，支持多靶点配体结合
- **参考**: [arxiv.org/abs/2412.10966](https://arxiv.org/abs/2412.10966)

#### UMOL / Boltz-1/2 / AlphaFold3
- **思路**: 蛋白-配体共同折叠
- **优势**: 捕捉 induced fit 效应
- **警告**: 可能存在训练集记忆问题
- **参考**: [academic.oup.com](https://academic.oup.com/bib/article/doi/10.1093/bib/bbaf454/8246683)

### 3.6 物理约束融合

#### 能量函数 + GNN (2024)
- **方法**: MM-GB/SA, MM-PB/SA 与 GNN 融合
- **效果**: Binding free energy 预测精度提升
- **参考**: [jcheminf.biomedcentral.com](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00912-2)

---

## 🚀 四、优化方案

### 方案 A: 增强配体编码器 ⭐⭐⭐ (推荐优先)

**改动**: 替换简单的 2 层 MLP 为增强版编码器

```python
class EnhancedLigandEncoder(nn.Module):
    """增强版配体编码器"""
    
    def __init__(self, d_lig=128, num_heads=4, num_layers=2):
        super().__init__()
        
        # 1. 扩展原子特征 (15 → 32)
        self.atom_embed = nn.Sequential(
            nn.Linear(32, d_lig),  # partial_charge, hybridization 等
            nn.LayerNorm(d_lig),
            nn.GELU()
        )
        
        # 2. RBF 距离编码
        self.dist_embed = RBFDistanceEncoding(num_rbf=16)
        
        # 3. 配体内自注意力（关键！）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_lig, 
            nhead=num_heads,
            batch_first=True
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, lig_points, lig_types, lig_mask):
        # 原子嵌入
        atom_feat = self.atom_embed(lig_types)  # [B, M, d_lig]
        
        # 距离矩阵编码
        dist_feat = self.dist_embed(lig_points)  # [B, M, M, num_rbf]
        
        # 自注意力（配体原子交互）
        attn_mask = ~lig_mask.unsqueeze(1).expand(-1, lig_mask.size(1), -1)
        out = self.self_attn(atom_feat, src_key_padding_mask=~lig_mask)
        
        return out
```

**预期效果**: Chi1 +5-10%
**风险**: 低
**工作量**: 1-2 天

---

### 方案 B: 添加几何特征 ⭐⭐

**改动**: 在 EdgeEmbedder 中添加几何特征

```python
# 添加到 edge 特征
geometric_features = {
    'distance_rbf': 16,      # RBF 编码的距离
    'angle_encoding': 8,     # 角度编码
    'local_curvature': 4,    # 局部曲率 (CWFBind)
    'surface_distance': 8,   # 到配体表面距离
}
```

**预期效果**: Chi1 +3-5%
**风险**: 中 (需要预计算几何特征)
**工作量**: 2-3 天

---

### 方案 C: 多任务学习 ⭐⭐⭐

**改动**: 添加辅助任务头

```python
class Stage1ModelMultiTask(Stage1Model):
    def __init__(self, config):
        super().__init__(config)
        
        # 主任务: Chi 角预测 (已有)
        self.chi_head = ChiHead(...)
        
        # 辅助任务 1: 接触图预测
        self.contact_head = nn.Linear(config.c_s, 1)
        
        # 辅助任务 2: 距离矩阵预测
        self.dist_head = nn.Linear(config.c_s * 2, 16)  # 距离 bin
        
    def compute_loss(self, pred, target):
        # 主损失
        chi_loss = self.chi_loss(pred['chi'], target['chi'])
        
        # 辅助损失
        contact_loss = self.contact_loss(pred['contact'], target['contact'])
        dist_loss = self.dist_loss(pred['dist'], target['dist'])
        
        # 组合 (可学习权重)
        total = chi_loss + 0.1 * contact_loss + 0.1 * dist_loss
        return total
```

**预期效果**: Chi1 +5-8%, 泛化性提升
**风险**: 中 (需要准备额外标签)
**工作量**: 3-4 天

---

### 方案 D: 4 层 IPA ⭐⭐

**改动**: 小心地增加到 4 层

```python
@classmethod
def stable_deep(cls) -> 'Stage1ModelConfig':
    """4层配置 - 需要更保守的训练"""
    return cls(
        c_s=384,
        c_p=128,
        c_hidden=128,
        no_heads=8,
        depth=4,  # 3 → 4
        # ... 其他保持不变
    )
```

**训练技巧**:
- 学习率: 5e-5 (更低)
- Gradient clip: 0.5
- Warmup: 1000 steps
- LayerScale / DropPath

**预期效果**: Chi1 +5-10%
**风险**: 高 (可能不稳定)
**工作量**: 1 天配置 + 观察训练

---

### 方案 E: 数据增强 ⭐

**改动**: 添加数据增强

```python
class Stage1DataAugmentation:
    def __init__(self):
        self.rotation_aug = True      # 随机旋转
        self.translation_aug = True   # 随机平移
        self.noise_aug = True         # 坐标噪声
        self.conformer_aug = True     # 配体构象采样
        
    def __call__(self, batch):
        if self.rotation_aug:
            batch = self.random_rotation(batch)
        if self.noise_aug:
            batch = self.add_coordinate_noise(batch, std=0.1)
        return batch
```

**预期效果**: 泛化性提升，Chi1 +2-3%
**风险**: 低
**工作量**: 1 天

---

### 方案 F: 使用完整数据集 ⭐

**改动**: 从 10k 扩展到完整数据集

```bash
# 当前: 10k 训练样本
--valid_samples_file train_samples_10k.txt

# 扩展: 完整数据集 (如果有更多)
--valid_samples_file valid_samples.txt
```

**预期效果**: Chi1 +3-5%
**风险**: 低 (只是更多数据)
**工作量**: 数据准备

---

## 📋 五、实验计划

### 5.1 优先级排序

| 优先级 | 方案 | 预期收益 | 风险 | 建议时间 |
|--------|------|---------|------|---------|
| 1 | A: 增强配体编码 | 高 | 低 | 第1周 |
| 2 | E: 数据增强 | 中 | 低 | 第1周 |
| 3 | C: 多任务学习 | 高 | 中 | 第2周 |
| 4 | B: 几何特征 | 中 | 中 | 第2-3周 |
| 5 | D: 4层IPA | 高 | 高 | 第3周 |
| 6 | F: 完整数据 | 中 | 低 | 随时 |

### 5.2 里程碑

| 里程碑 | 目标 | 时间 |
|--------|------|------|
| M0 | 当前 stable_wide 收敛 | 本周 |
| M1 | 方案 A+E 实现，Chi1 55%+ | 1-2 周 |
| M2 | 方案 C 实现，Chi1 60%+ | 3-4 周 |
| M3 | 方案 B+D 尝试，Chi1 65%+ | 5-6 周 |

---

## ⚠️ 六、风险与监控

### 6.1 训练稳定性
- 监控 NaN/Inf 出现频率
- 每 epoch 检查 loss 趋势
- 出现问题立即降低学习率或回退配置

### 6.2 过拟合
- 训练/验证 loss 差距
- 在未见蛋白上测试
- 使用 dropout / weight decay

### 6.3 维度冲突
- FlashIPA 要求 `headdim_eff <= 256`
- 修改 c_hidden, z_factor_rank 前计算
- `headdim_eff = c_hidden + 36 + z_factor_rank * 32`

---

## 📚 七、参考文献

1. **CWFBind** - Local curvature for binding site prediction  
   https://arxiv.org/abs/2508.09499

2. **DeepRLI** - Multi-objective protein-ligand interaction  
   https://pubs.rsc.org/en/content/articlehtml/2025/dd/d4dd00403e

3. **SurfDock** - Surface-informed diffusion docking  
   https://english.cas.cn/newsroom/research_news/life/202412/t20241202_893028.shtml

4. **FlowDock** - Apo to holo generative model  
   https://arxiv.org/abs/2412.10966

5. **CAPLA** - Cross-attention binding affinity  
   https://academic.oup.com/bioinformatics/article/39/2/btad049/6998204

6. **Co-folding methods** - UMOL, Boltz, AlphaFold3  
   https://academic.oup.com/bib/article/doi/10.1093/bib/bbaf454/8246683

7. **Physical energy + GNN** - Binding free energy prediction  
   https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00912-2

---

## 💡 八、讨论要点

1. **配体编码是当前最大瓶颈**
   - 只用 2 层 MLP，没有捕捉配体内部结构
   - 方案 A 应该优先实施

2. **深度 vs 宽度权衡**
   - 5 层不稳定，3 层稳定但容量有限
   - 考虑 4 层 + 更保守训练
   - 或者保持 3 层但加宽配体/输出部分

3. **多任务可能是关键**
   - DeepRLI 证明多任务提升泛化
   - 接触图/距离矩阵可作为辅助监督

4. **数据质量 vs 数量**
   - 清理 CA-only 样本已完成
   - 是否需要更多数据？
   - 数据增强可能比更多数据更有效

5. **80%+ 的路径**
   - 需要更根本的架构改变
   - 参考 AlphaFold3 / FlowDock
   - 可能需要 recycling / diffusion

---

*最后更新: 2026-01-16*
