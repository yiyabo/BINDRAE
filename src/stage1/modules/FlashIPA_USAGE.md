# FlashIPA EdgeEmbedder 使用文档

## 概述

本项目使用 **FlashIPA 原生 EdgeEmbedder**，通过适配器层提供简化接口。

---

## FlashIPA 原生接口

### EdgeEmbedderConfig 参数

```python
from flash_ipa.edge_embedder import EdgeEmbedderConfig

config = EdgeEmbedderConfig(
    c_s=384,                    # 节点单表示维度
    c_p=128,                    # 节点对表示维度（边嵌入维度）
    z_factor_rank=16,           # 因子化秩
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
- `z_factor_rank=16`, `c_p=128`
- 输出形状：`[2, 50, 16, 128]`

---

## 项目适配器接口

### 简化的配置

```python
from src.stage1.modules.edge_embed import ProjectEdgeConfig

config = ProjectEdgeConfig(
    c_s=384,             # 节点表示维度
    c_p=128,             # 边表示维度
    z_factor_rank=16,    # 因子秩
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
    z_rank=16,
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
z_f1 = outputs['z_f1']          # [B, N, 16, 128]
z_f2 = outputs['z_f2']          # [B, N, 16, 128]
edge_mask = outputs['edge_mask'] # [B, N, N]
```

---

## 输出格式说明

### z_f1 和 z_f2 的含义

**形状**：`[B, N, z_factor_rank, c_p]`

- `B`: 批大小
- `N`: 残基数量
- `z_factor_rank`: 因子化秩（默认16）
- `c_p`: 节点对表示维度（默认128）

**物理意义**：
- FlashIPA使用**因子化边表示**避免O(N²)显存
- 完整边嵌入：`z_ij = z_f1[i] ⊗ z_f2[j]` (外积)
- 每个残基有16个因子，每个因子128维

**显存优势**：
```
传统方法：[B, N, N, 128] = O(N²)
FlashIPA：[B, N, 16, 128] = O(N)
```

对于300残基：
- 传统：~46 MB
- FlashIPA：~2.5 MB（节省95%）

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
  - z_f1: torch.Size([2, 50, 16, 128])
  - z_f2: torch.Size([2, 50, 16, 128])
  - edge_mask: torch.Size([2, 50, 50])
  - GPU显存: ~18 MB
```

---

## 注意事项

1. **FlashIPA路径**：确保 `/tmp/flash_ipa/src` 存在并在sys.path中
2. **GPU推荐**：FlashIPA优化了GPU性能，建议使用CUDA
3. **显存占用**：因子化设计，300残基仅需~50 MB
4. **梯度计算**：完全支持反向传播

---

**最后更新**: 2025-10-28  
**FlashIPA版本**: Latest (from /tmp/flash_ipa)  
**测试环境**: RTX 4090 D, PyTorch 2.6.0+cu124

