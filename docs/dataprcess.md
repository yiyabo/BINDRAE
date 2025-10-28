# 蛋白-配体结合亲和力预测：完整数据处理流程

## 完整数据准备流程 (按顺序执行)

```bash
cd /Users/apple/code/BINDRAE

# 1. 数据集整理
python scripts/prepare_casf2016.py

# 2. 配体规范化
python scripts/prepare_ligands.py

# 3. 口袋提取
python scripts/extract_pockets.py

# 4. 扭转角提取
python scripts/extract_torsions.py

# 5. ESM-2 缓存
python scripts/cache_esm2.py

# 6. 数据集划分
python scripts/split_dataset.py

# 7. 数据验证
python scripts/validate_data.py
```

---

## 第1步：数据集整理 (`prepare_casf2016.py`)

### 为什么需要这一步？
原始CASF-2016数据集存在以下问题：
- 数据分散在多个目录，缺乏统一组织
- 包含大量质量问题的样本（低分辨率、结构不完整等）
- 元数据与结构数据分离，难以管理

### 处理目的
1. **数据标准化**：建立统一的目录结构和文件命名规范
2. **质量控制**：过滤低质量样本，确保训练数据可靠性
3. **元数据整合**：将分辨率、结合亲和力等信息与结构数据关联

### 核心处理逻辑
- **分辨率过滤**（≤3.0Å）：确保原子坐标精度
- **蛋白质验证**：检查残基数量、金属原子含量
- **配体验证**：验证原子数、分子量、文件格式
- **文件复制**：将验证通过的样本复制到统一目录

### 最终输出
```
data/casf2016/
├── complexes/           # 清洗后的复合物
│   ├── 1abc/
│   │   ├── protein.pdb
│   │   └── ligand.sdf
│   └── ...
├── meta/              # 元数据
│   ├── INDEX_core.txt  # 有效样本索引
│   └── filtered.csv    # 过滤记录
└── processed/         # 后续处理目录
```

---

## 第2步：配体规范化 (`prepare_ligands.py`)

### 为什么需要这一步？
原始配体文件存在以下问题：
- 包含氢原子，增加计算复杂度
- 价态和立体化学可能不规范
- 缺乏标准化的分子描述符

### 处理目的
1. **分子标准化**：去除氢原子，标准化价态和立体化学
2. **坐标提取**：提取重原子坐标用于后续计算
3. **特征计算**：生成分子描述符用于模型输入

### 核心处理逻辑
- **氢原子去除**：减少计算复杂度，保留重原子信息
- **价态标准化**：确保分子结构的化学合理性
- **立体化学分配**：处理手性中心信息
- **重原子坐标提取**：生成标准化的3D坐标数组
- **分子描述符计算**：分子量、氢键供体/受体、LogP等

### 最终输出
```
features/
├── 1abc_ligand_coords.npy      # 重原子坐标 (N_atoms, 3)
├── 1abc_ligand_props.npy       # 配体属性字典
├── 1abc_ligand_normalized.sdf  # 规范化后的SDF
└── ...
```

---

## 第3步：口袋提取 (`extract_pockets.py`)

### 为什么需要这一步？
蛋白质-配体结合主要发生在局部区域：
- 全蛋白质信息包含大量无关噪声
- 结合口袋是药物设计的关键区域
- 需要量化残基与配体的相互作用强度

### 处理目的
1. **口袋识别**：基于距离确定结合区域
2. **权重计算**：使用RBF核量化残基重要性
3. **区域扩展**：通过图膨胀包含邻近残基

### 核心处理逻辑
- **距离计算**：计算每个残基与配体的最小重原子距离
- **接触识别**：距离≤5Å的残基定义为接触残基
- **图膨胀**：基于序列邻接关系扩展口袋区域（k=1）
- **RBF权重**：使用径向基函数计算软权重（σ=2Å）
- **权重归一化**：将权重标准化到[0,1]范围

### 最终输出
```
pockets/
├── 1abc_w_res.npy          # 残基权重 (N_res,) 值: [0,1]
├── 1abc_pocket_mask.npy    # 口袋掩码 (N_res,) 值: True/False
├── 1abc_residue_info.npy   # 残基信息 (链、名称、编号)
├── 1abc_distances.npy      # 残基-配体距离 (N_res,)
└── ...
```

---

## 第4步：扭转角提取 (`extract_torsions.py`)

### 为什么需要这一步？
蛋白质的3D结构由扭转角决定：
- 主链扭转角（φ, ψ, ω）决定二级结构
- 侧链扭转角（χ1-χ4）决定三级结构细节
- 扭转角是蛋白质构象的基本自由度

### 处理目的
1. **构象描述**：用扭转角描述蛋白质3D结构
2. **特征提取**：为深度学习模型提供结构信息
3. **质量控制**：验证结构完整性和合理性

### 核心处理逻辑
- **主链角度计算**：φ(-C-N-CA-C), ψ(N-CA-C-+N), ω(CA-C-+N-+CA)
- **侧链角度计算**：基于IUPAC标准的χ1-χ4定义
- **序列连续性验证**：确保只计算真正连续残基间的角度
- **坐标有效性检查**：处理缺失原子、altLoc、无效坐标
- **cis/trans识别**：基于ω角判断肽键构型

### 最终输出
```
features/
├── 1abc_torsions.npz
│   ├── phi, psi, omega: (N_res,)           # 主链角度
│   ├── chi: (N_res, 4)                    # 侧链角度
│   ├── bb_mask, chi_mask: boolean arrays    # 有效性掩码
│   └── omega_cis_trans: 0=trans, 1=cis    # 肽键构型
└── ...
```

---

## 第5步：ESM-2 缓存 (`cache_esm2.py`)

### 为什么需要这一步？
原始蛋白质序列无法直接用于深度学习：
- 序列长度可变，需要固定维度表示
- 缺乏预训练的生物学知识
- 重复计算浪费计算资源

### 处理目的
1. **序列编码**：将氨基酸序列转换为高维向量表示
2. **知识迁移**：利用预训练模型的生物学知识
3. **计算优化**：预计算特征避免重复处理

### 核心处理逻辑
- **序列提取**：从PDB文件提取氨基酸序列（三字母→单字母）
- **模型选择**：根据硬件自动选择合适大小的ESM-2模型
- **深度编码**：使用Transformer编码器生成表征
- **特征提取**：生成每残基表征和全序列表征
- **断点续传**：自动跳过已处理样本

### 最终输出
```
features/
├── 1abc_esm.pt
│   ├── per_residue: (N_res, 512/1280)  # 每残基表征
│   ├── sequence: (512/1280,)            # 全序列表征
│   ├── sequence_str: str                 # 原始序列
│   └── n_residues: int                  # 残基数量
└── ...
```

---

## 第6步：数据集划分 (`split_dataset.py`)

### 为什么需要这一步？
机器学习需要合理的数据划分：
- 避免数据泄露，确保模型评估的可靠性
- 需要独立的训练、验证、测试集
- 固定随机种子确保结果可重现

### 处理目的
1. **数据分割**：按8:1:1比例划分训练/验证/测试集
2. **避免泄露**：确保同一PDB ID不跨分割
3. **可重现性**：固定随机种子，支持结果复现

### 核心处理逻辑
- **样本收集**：获取所有有效复合物的PDB ID
- **随机打乱**：使用固定随机种子打乱样本顺序
- **比例划分**：按8:1:1比例分配到train/val/test
- **文件保存**：将划分结果保存为JSON格式

### 最终输出
```
splits/
├── train.json  # 训练集 (80%)
├── val.json    # 验证集 (10%)
└── test.json   # 测试集 (10%)
```

---

## 第7步：数据验证 (`validate_data.py`)

### 为什么需要这一步？
经过多步处理后可能产生新问题：
- 特征提取过程中的数值异常
- 口袋识别算法的边界情况
- 数据转换过程中的信息丢失

### 处理目的
1. **数值稳定性检查**：确保所有特征值有效
2. **生物学合理性验证**：检查配体和口袋的合理性
3. **训练就绪性确认**：验证数据格式符合模型要求

### 核心处理逻辑
- **配体验证**：原子数范围（5-300）、坐标有效性（NaN/Inf检测）
- **口袋验证**：权重分布检查、口袋大小验证（≥5残基）
- **统计报告**：生成详细的质量评估报告
- **问题分类**：按类型统计和报告数据问题

### 最终输出
```
验证报告：
================================================================================
数据一致性验证
================================================================================
总样本数: 270
有效样本: 265 (98.1%)

问题分类:
⚠️  配体原子过少 (<5): 3 个
❌ 无效坐标: 1 个
⚠️  口袋过小 (<5残基): 1 个

✅ 数据质量验证通过！
```

---

## 完整流程的价值

### 数据流转关系
```
原始CASF-2016 → 数据整理 → 配体规范化 → 口袋提取 → 扭转角提取 → ESM-2编码 → 数据划分 → 质量验证
     ↓              ↓           ↓           ↓            ↓          ↓        ↓        ↓
  分散数据      标准化结构    重原子坐标    结合口袋     构象角度   深度特征   训练划分   质量保证
```

### 最终数据集特征
- **样本数量**：250-285个高质量复合物
- **特征维度**：
  - 配体坐标：(N_atoms, 3)
  - 口袋权重：(N_res,)
  - 扭转角度：(N_res, 7)
  - ESM-2特征：(N_res, 512/1280)
- **标签**：logKa结合亲和力值
- **数据划分**：8:1:1的训练/验证/测试集

### 对模型训练的价值
1. **高质量输入**：多层过滤确保数据可靠性
2. **丰富特征**：结合几何、序列、深度学习特征
3. **标准化格式**：支持批量训练和模型开发
4. **可重现性**：固定随机种子和详细日志

通过这个完整的7步数据处理流程，我们将原始的、质量参差不齐的CASF-2016数据集转换为高质量的、标准化的、可直接用于深度学习训练的完整数据集，为蛋白-配体结合亲和力预测模型提供了可靠的数据基础。以后其他的数据库也可以参考此处理流程，我们可以考虑构造一套基于Agent的智能数据处理系统。

---

## 最终数据形式与模型使用

### 完整数据结构

处理完成后，数据集具有以下结构：

```
data/casf2016/
├── complexes/                    # 原始复合物结构
│   ├── 1abc/
│   │   ├── protein.pdb
│   │   └── ligand.sdf
│   └── ...
├── meta/                        # 元数据
│   ├── INDEX_core.txt           # 有效样本索引
│   └── filtered.csv            # 过滤记录
├── processed/
│   ├── features/               # 特征文件
│   │   ├── 1abc_esm.pt        # ESM-2特征
│   │   ├── 1abc_ligand_coords.npy    # 配体坐标
│   │   ├── 1abc_ligand_props.npy     # 配体属性
│   │   └── 1abc_torsions.npz         # 扭转角
│   ├── pockets/                # 口袋数据
│   │   ├── 1abc_w_res.npy            # 残基权重
│   │   ├── 1abc_pocket_mask.npy      # 口袋掩码
│   │   └── 1abc_residue_info.npy     # 残基信息
│   └── splits/                 # 数据划分
│       ├── train.json           # 训练集
│       ├── val.json             # 验证集
│       └── test.json            # 测试集
```

### 单个样本的数据组成

每个PDB ID（如1abc）包含以下数据：

#### 1. ESM-2蛋白质特征 (`1abc_esm.pt`)
```python
{
    'per_residue': torch.Tensor,  # (N_res, 512/1280) - 每残基表征
    'sequence': torch.Tensor,     # (512/1280,) - 全序列表征
    'sequence_str': str,         # 氨基酸序列
    'n_residues': int           # 残基数量
}
```

#### 2. 配体数据
- **坐标** (`1abc_ligand_coords.npy`): (N_atoms, 3) - 重原子3D坐标
- **属性** (`1abc_ligand_props.npy`): 分子量、氢键供体/受体等描述符

#### 3. 口袋数据
- **权重** (`1abc_w_res.npy`): (N_res,) - 残基重要性权重 [0,1]
- **掩码** (`1abc_pocket_mask.npy`): (N_res,) - 口袋区域标识
- **残基信息** (`1abc_residue_info.npy`): 残基详细信息

#### 4. 扭转角 (`1abc_torsions.npz`)
```python
{
    'phi': np.array,           # (N_res,) - 主链φ角
    'psi': np.array,           # (N_res,) - 主链ψ角
    'omega': np.array,         # (N_res,) - 主链ω角
    'chi': np.array,          # (N_res, 4) - 侧链χ角
    'bb_mask': np.array,       # (N_res,) - 主链角度有效性
    'chi_mask': np.array,      # (N_res, 4) - 侧链角度有效性
}
```

#### 5. 标签数据
从`meta/INDEX_core.txt`获取：
- **logKa**: 结合亲和力对数值（预测目标）
- **resolution**: 结构分辨率
- **year**: 发表年份
- **target**: 靶标类型

---

## ESM+几何双分支模型的数据使用

### 模型架构设计

```
输入数据 → ESM分支 → 蛋白质序列特征
        ↓
        几何分支 → 配体+口袋几何特征
        ↓
        特征融合 → 结合亲和力预测
```

### ESM分支数据使用

#### 输入数据
```python
# 加载ESM-2特征
esm_data = torch.load('features/1abc_esm.pt')

# 序列特征 (全局表征)
sequence_features = esm_data['sequence']  # (512,)

# 残基特征 (局部表征)
residue_features = esm_data['per_residue']  # (N_res, 512)

# 口袋权重加权
pocket_weights = np.load('pockets/1abc_w_res.npy')  # (N_res,)
weighted_residue_features = residue_features * pocket_weights.unsqueeze(-1)

# 池化得到固定维度
pooled_features = torch.sum(weighted_residue_features, dim=0)  # (512,)
```

#### 处理逻辑
1. **全局特征**: 直接使用sequence表征
2. **局部特征**: 使用口袋权重加权残基表征
3. **特征融合**: 拼接全局和局部特征
4. **维度统一**: 确保所有样本特征维度一致

### 几何分支数据使用

#### 输入数据
```python
# 配体坐标
ligand_coords = np.load('features/1abc_ligand_coords.npy')  # (N_atoms, 3)

# 扭转角
torsions = np.load('features/1abc_torsions.npz')
phi = torsions['phi']  # (N_res,)
psi = torsions['psi']  # (N_res,)
chi = torsions['chi']  # (N_res, 4)

# 口袋掩码
pocket_mask = np.load('pockets/1abc_pocket_mask.npy')  # (N_res,)

# 配体属性
ligand_props = np.load('features/1abc_ligand_props.npy').item()
```

#### 处理逻辑
1. **配体几何**: 使用3D坐标计算距离、角度等几何特征
2. **蛋白质几何**: 使用扭转角描述局部构象
3. **口袋特征**: 基于掩码提取结合区域几何信息
4. **分子描述符**: 使用配体化学描述符

### 数据加载器实现

```python
class BINDRAEDataset(torch.utils.data.Dataset):
    def __init__(self, pdb_ids, data_dir):
        self.pdb_ids = pdb_ids
        self.data_dir = Path(data_dir)
        
    def __len__(self):
        return len(self.pdb_ids)
    
    def __getitem__(self, idx):
        pdb_id = self.pdb_ids[idx]
        
        # 加载ESM特征
        esm_data = torch.load(self.data_dir / 'features' / f'{pdb_id}_esm.pt')
        
        # 加载几何数据
        ligand_coords = np.load(self.data_dir / 'features' / f'{pdb_id}_ligand_coords.npy')
        torsions = np.load(self.data_dir / 'features' / f'{pdb_id}_torsions.npz')
        pocket_weights = np.load(self.data_dir / 'pockets' / f'{pdb_id}_w_res.npy')
        
        # 加载标签
        metadata = self.load_metadata(pdb_id)
        
        return {
            'pdb_id': pdb_id,
            'esm_sequence': esm_data['sequence'],
            'esm_residues': esm_data['per_residue'],
            'pocket_weights': torch.tensor(pocket_weights),
            'ligand_coords': torch.tensor(ligand_coords),
            'torsions': {k: torch.tensor(v) for k, v in torsions.items()},
            'target': torch.tensor(metadata['logKa'], dtype=torch.float32)
        }
```

### 模型训练流程

```python
# 数据准备
train_ids = json.load(open('processed/splits/train.json'))['pdb_ids']
val_ids = json.load(open('processed/splits/val.json'))['pdb_ids']

train_dataset = BINDRAEDataset(train_ids, 'data/casf2016')
val_dataset = BINDRAEDataset(val_ids, 'data/casf2016')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 模型训练
for batch in train_loader:
    # ESM分支
    esm_seq_features = model.esm_branch(batch['esm_sequence'])
    esm_res_features = model.esm_branch(batch['esm_residues'])
    esm_weighted = esm_res_features * batch['pocket_weights'].unsqueeze(-1)
    esm_pooled = torch.sum(esm_weighted, dim=1)
    
    # 几何分支
    geo_features = model.geometry_branch(
        batch['ligand_coords'], 
        batch['torsions'],
        batch['pocket_weights']
    )
    
    # 特征融合
    combined = torch.cat([esm_seq_features, esm_pooled, geo_features], dim=-1)
    
    # 预测
    prediction = model.predictor(combined)
    
    # 损失计算
    loss = F.mse_loss(prediction, batch['target'])
```

### 数据优势

1. **多模态特征**: 结合序列语义和3D几何信息
2. **高质量输入**: 多层过滤确保数据可靠性
3. **标准化格式**: 支持批量训练和模型开发
4. **可解释性**: 口袋权重提供残基重要性信息
5. **扩展性**: 模块化设计便于添加新特征

通过这种数据组织方式，ESM+几何双分支模型可以充分利用蛋白质的序列语义信息和3D结构信息，实现更准确的结合亲和力预测。