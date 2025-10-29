# FlashIPA 使用文档

## 概述

本项目使用 **FlashIPA 原生实现**（EdgeEmbedder + InvariantPointAttention），通过适配器层提供简化接口。

**核心模块**：
- `EdgeEmbedderAdapter`: 边嵌入适配器
- `FlashIPAModule`: 多层IPA几何分支

---

## ⚠️ 重要限制：z_factor_rank ≤ 2

### FlashAttention2 硬件限制

**问题**：FlashAttention2要求 `headdim_eff ≤ 256`

**计算公式**：
```python
headdim_eff = max(
    c_hidden + 5*no_qk_points + z_factor_rank*no_heads,
    c_hidden + 3*no_v_points + z_factor_rank*(c_z//4)
)

# 我们的配置：
c_hidden=128, no_v_points=12, c_z=128
headdim_eff = 128 + 36 + z_factor_rank*32

# 限制条件：
128 + 36 + z_rank*32 ≤ 256
z_rank ≤ 2.875
→ z_rank ≤ 2
```

### 为什么有这个限制？

**原因**：CUDA Shared Memory 和寄存器限制
- FlashAttention在片上内存（48-64 KB）中计算
- headdim太大会导致shared memory溢出
- **这是所有GPU的通用限制**，不是硬件性能问题

### 性能影响评估

| 配置 | 边表示容量 | 预期性能影响 |
|-----|-----------|------------|
| z_rank=16（原计划） | 2048维 | Baseline |
| **z_rank=2（实际）** | **256维** | **↓ 5-15%** |

**影响**：
- ⚠️ 长程相互作用建模能力下降
- ⚠️ 复杂口袋细节表示减弱
- ✅ 核心几何信息仍在点注意力中（影响有限）

### 缓解策略

如需更强表示能力，可：
1. **增加IPA层数**：depth: 3→4
2. **增加隐藏维度**：c_hidden: 128→192（需重新验证headdim）
3. **使用naive attention**：`use_flash_attn=False`（速度慢20倍）

**项目决策**：先用z_rank=2训练，后续做消融实验对比。

---

## FlashIPA 原生接口

### EdgeEmbedderConfig 参数

```python
from flash_ipa.edge_embedder import EdgeEmbedderConfig

config = EdgeEmbedderConfig(
    c_s=384,                    # 节点单表示维度
    c_p=128,                    # 节点对表示维度（边嵌入维度）
    z_factor_rank=2,            # 因子化秩（⚠️ 必须≤2，见限制说明）
    num_rbf=32,                 # RBF核数量（默认32）
    mode='flash_1d_bias',       # 边嵌入模式
    use_rbf=True,               # 使用RBF距离编码
    feat_dim=64,                # 特征维度
    relpos_k=64,                # 相对位置编码维度
    self_condition=True,        # 自条件化
)
```

### EdgeEmbedder.forward 参数

```python
outputs = edge_embedder(
    node_embed,      # (B, N, c_s) 节点嵌入
    translations,    # (B, N, 3) 主链坐标（Cα）
    trans_sc,        # (B, N, 3) 侧链坐标
    node_mask,       # (B, N) 节点掩码
    edge_embed,      # 预先提供的边嵌入（可选，通常为None）
    edge_mask        # 边掩码（可选，通常为None）
)
```

### FlashIPA 原生输出格式

```python
# 返回 tuple，4个元素
(None, z_f1, z_f2, None)

# z_f1: [B, N, z_factor_rank, c_p]
# z_f2: [B, N, z_factor_rank, c_p]
```

**示例**：
- `z_factor_rank=2`, `c_p=128`
- 输出形状：`[2, 50, 2, 128]`

---

## 项目适配器接口

### 简化的配置

```python
from src.stage1.modules.edge_embed import ProjectEdgeConfig

config = ProjectEdgeConfig(
    c_s=384,             # 节点表示维度
    c_p=128,             # 边表示维度
    z_factor_rank=2,     # 因子秩（⚠️ 必须≤2）
    num_rbf=16,          # RBF核数（项目默认16）
    mode='flash_1d_bias',
    feat_dim=64,
    relpos_k=64,
)
```

### 简化的接口

```python
from src.stage1.modules.edge_embed import create_edge_embedder
import torch

# 创建适配器
embedder = create_edge_embedder(
    c_s=384,
    c_p=128,
    z_rank=2,  # ⚠️ 必须≤2
    num_rbf=16
)

# 前向传播（简化为3个参数）
outputs = embedder(
    node_embed,     # (B, N, 384)
    translations,   # (B, N, 3)
    node_mask       # (B, N)
    # trans_sc 自动用 translations 代替
)

# 获取输出
z_f1 = outputs['z_f1']          # [B, N, 2, 128]
z_f2 = outputs['z_f2']          # [B, N, 2, 128]
edge_mask = outputs['edge_mask'] # [B, N, N]
```

---

## 输出格式说明

### z_f1 和 z_f2 的含义

**形状**：`[B, N, z_factor_rank, c_p]`

- `B`: 批大小
- `N`: 残基数量
- `z_factor_rank`: 因子化秩（**项目使用2**，受FlashAttention限制）
- `c_p`: 节点对表示维度（默认128）

**物理意义**：
- FlashIPA使用**因子化边表示**避免O(N²)显存
- 完整边嵌入：`z_ij = z_f1[i] ⊗ z_f2[j]` (外积)
- 每个残基有2个因子，每个因子128维
- 边表示容量：2 × 128 = **256维**

**显存优势**：
```
传统方法：[B, N, N, 128] = O(N²)
FlashIPA（z_rank=2）：[B, N, 2, 128] = O(N)
```

对于300残基：
- 传统：~46 MB
- FlashIPA：~0.3 MB（节省99%）

### edge_mask

**形状**：`[B, N, N]`

布尔张量，标记有效的边（节点对）。

---

## 与原生FlashIPA的差异

| 项目 | 原生FlashIPA | 项目适配器 |
|-----|-------------|-----------|
| **forward参数** | 6个 | 3个（简化） |
| **trans_sc处理** | 必须提供 | 可选，默认用主链代替 |
| **返回格式** | tuple (4个元素) | dict (含edge_mask) |
| **edge_mask** | 不返回 | 自动生成 |
| **默认num_rbf** | 32 | 16（项目配置） |

### 为什么简化接口？

1. **侧链坐标**：第一阶段只用主链（Cα），不需要侧链
2. **边嵌入**：不预先提供，完全由EdgeEmbedder生成
3. **返回字典**：更清晰，避免记忆tuple顺序

---

## 完整使用示例

```python
import torch
from src.stage1.modules.edge_embed import create_edge_embedder

# 准备数据
B, N = 2, 50  # 2个蛋白，每个50个残基
node_embed = torch.randn(B, N, 384)  # 来自ESM-2 + Adapter
translations = torch.randn(B, N, 3)   # Cα坐标
node_mask = torch.ones(B, N, dtype=torch.bool)
node_mask[:, -5:] = False  # 最后5个残基无效（padding）

# 创建EdgeEmbedder
embedder = create_edge_embedder(
    c_s=384,
    c_p=128,
    z_rank=16,
    num_rbf=16
)

# GPU加速
if torch.cuda.is_available():
    embedder = embedder.cuda()
    node_embed = node_embed.cuda()
    translations = translations.cuda()
    node_mask = node_mask.cuda()

# 前向传播
with torch.no_grad():
    outputs = embedder(node_embed, translations, node_mask)

# 使用输出
z_f1 = outputs['z_f1']  # [2, 50, 16, 128]
z_f2 = outputs['z_f2']  # [2, 50, 16, 128]
edge_mask = outputs['edge_mask']  # [2, 50, 50]

print(f"z_f1形状: {z_f1.shape}")
print(f"z_f2形状: {z_f2.shape}")
print(f"edge_mask形状: {edge_mask.shape}")
```

---

## 测试

运行测试脚本验证功能：

```bash
cd /path/to/BINDRAE
bash scripts/test_flashipa_adapter.sh
```

**预期输出**：
```
✅ 所有测试通过！
  - z_f1: torch.Size([2, 50, 2, 128])
  - z_f2: torch.Size([2, 50, 2, 128])
  - edge_mask: torch.Size([2, 50, 50])
  - GPU显存: ~18 MB
```

---

---

## InvariantPointAttention 模块

### IPAConfig 参数

```python
from flash_ipa.ipa import IPAConfig

config = IPAConfig(
    c_s=384,                # 节点单表示维度
    c_z=128,                # 边表示维度
    c_hidden=128,           # 隐藏层维度
    no_heads=8,             # 注意力头数
    z_factor_rank=2,        # 因子秩（⚠️ 必须≤2）
    no_qk_points=8,         # query/key点数
    no_v_points=12,         # value点数
    use_flash_attn=True,    # 使用FlashAttention加速
    attn_dtype='fp16',      # fp16（headdim>256时必须）
)
```

### IPA.forward 参数

```python
output = ipa(
    s,              # [B, N, c_s] 节点单表示
    z,              # [B, N, N, c_z] 边表示（可选，通常为None）
    z_factor_1,     # [B, N, z_rank, c_z] 边因子1（来自EdgeEmbedder）
    z_factor_2,     # [B, N, z_rank, c_z] 边因子2
    r,              # Rigid对象 [B, N] 刚体帧
    mask,           # [B, N] 节点掩码
)
# 返回: [B, N, c_s] 更新后的节点表示
```

### 项目FlashIPAModule接口

```python
from src.stage1.models.ipa import create_flashipa_module

# 创建模块
ipa_module = create_flashipa_module(
    c_s=384,
    c_z=128,
    depth=3  # IPA层数
)

# 前向传播
s_geo, rigids_final = ipa_module(
    s,          # [B, N, 384] 节点表示
    rigids,     # Rigid对象 初始帧
    z_f1,       # [B, N, 2, 128] 来自EdgeEmbedder
    z_f2,       # [B, N, 2, 128]
    mask        # [B, N]
)
```

**输出**：
- `s_geo`: [B, N, 384] 几何增强的节点表示
- `rigids_final`: Rigid对象（更新后的刚体帧）

---

## 完整使用流程

```python
import torch
from src.stage1.modules.edge_embed import create_edge_embedder
from src.stage1.models.ipa import create_flashipa_module
from flash_ipa.rigid import Rigid, Rotation

# 1. 准备数据
B, N = 2, 50
node_embed = torch.randn(B, N, 384).cuda()  # 来自ESM + Adapter
ca_coords = torch.randn(B, N, 3).cuda()     # Cα坐标
mask = torch.ones(B, N, dtype=torch.bool).cuda()

# 2. 创建EdgeEmbedder
edge_embedder = create_edge_embedder(c_s=384, c_p=128, z_rank=2).cuda()
outputs = edge_embedder(node_embed, ca_coords, mask)
z_f1, z_f2 = outputs['z_f1'], outputs['z_f2']

# 3. 创建初始Rigid帧
rot_identity = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).cuda()
rotation = Rotation(rot_mats=rot_identity)
rigids = Rigid(rots=rotation, trans=ca_coords)

# 4. 创建FlashIPA模块
ipa_module = create_flashipa_module(c_s=384, c_z=128, depth=3).cuda()

# 5. 前向传播
s_geo, rigids_updated = ipa_module(node_embed, rigids, z_f1, z_f2, mask)

print(f"输出: {s_geo.shape}")  # [2, 50, 384]
```

---

## 测试

### EdgeEmbedder测试
```bash
bash scripts/test_flashipa_adapter.sh
```

### FlashIPA模块测试
```bash
bash scripts/test_ipa_module.sh
```

---

## 注意事项

1. **z_factor_rank限制**：必须≤2（FlashAttention硬件限制）
2. **FlashIPA路径**：确保 `/tmp/flash_ipa/src` 在sys.path中
3. **attn_dtype**：headdim>256时必须用fp16
4. **GPU推荐**：FlashIPA优化了GPU性能，CUDA必需
5. **依赖要求**：
   - flash-attn ≥ 2.0（需要与PyTorch版本匹配）
   - beartype, jaxtyping（类型检查）

---

## 已验证的配置

**测试环境**：
- GPU: RTX 4090 D
- PyTorch: 2.6.0+cu124
- CUDA: 12.4
- flash-attn: 2.8.3

**性能数据**：
- EdgeEmbedder (50残基): 18.73 MB
- FlashIPA模块 (20残基, 3层): 48.08 MB
- 参数量: 9,958,146

---

**最后更新**: 2025-10-28  
**状态**: EdgeEmbedder ✅ | FlashIPA ✅ | 配体条件化 ⏳

