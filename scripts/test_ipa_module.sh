#!/bin/bash
# FlashIPA几何分支测试脚本（Linux服务器）

set -e

GREEN='\033[0;32m'
NC='\033[0m'

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

echo "============================================================================"
echo "              FlashIPA 几何分支模块测试"
echo "============================================================================"
echo ""

echo -e "${GREEN}[1/1]${NC} 功能验证..."
python << 'EOF'
import torch
import sys
import os

flash_ipa_path = '/tmp/flash_ipa/src'
if os.path.exists(flash_ipa_path):
    sys.path.insert(0, flash_ipa_path)

from flash_ipa.rigid import Rigid, Rotation
from src.stage1.models.ipa import create_flashipa_module
from src.stage1.modules.edge_embed import create_edge_embedder

print("\n测试配置:")
print(f"  - 批大小: 2")
print(f"  - 残基数: 20")
print(f"  - IPA层数: 3")

# 准备数据
B, N = 2, 20
c_s = 384

node_embed = torch.randn(B, N, c_s)
translations = torch.randn(B, N, 3)
mask = torch.ones(B, N, dtype=torch.bool)

if torch.cuda.is_available():
    node_embed = node_embed.cuda()
    translations = translations.cuda()
    mask = mask.cuda()
    print(f"  - 设备: CUDA")
else:
    print(f"  - 设备: CPU")

# 创建EdgeEmbedder
print("\n创建EdgeEmbedder...")
edge_embedder = create_edge_embedder(c_s=384, c_p=128, z_rank=2)  # 必须与IPA的z_factor_rank一致
if torch.cuda.is_available():
    edge_embedder = edge_embedder.cuda()

# 生成边嵌入
with torch.no_grad():
    edge_outputs = edge_embedder(node_embed, translations, mask)
    z_f1 = edge_outputs['z_f1']
    z_f2 = edge_outputs['z_f2']

print(f"✓ 边嵌入生成成功")
print(f"  - z_f1: {z_f1.shape}")
print(f"  - z_f2: {z_f2.shape}")

# 创建初始Rigid
print("\n创建初始Rigid...")
rot_identity = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
if torch.cuda.is_available():
    rot_identity = rot_identity.cuda()
    translations_device = translations
else:
    translations_device = translations

rotation = Rotation(rot_mats=rot_identity)
rigids = Rigid(rots=rotation, trans=translations_device)
print(f"✓ Rigid对象创建成功")

# 创建IPA模块
print("\n创建FlashIPA模块...")
ipa_module = create_flashipa_module(c_s=384, c_z=128, depth=3)
if torch.cuda.is_available():
    ipa_module = ipa_module.cuda()

print(f"✓ FlashIPA模块创建成功")
print(f"  - 参数量: {sum(p.numel() for p in ipa_module.parameters()):,}")

# 前向传播
print("\n前向传播...")
with torch.no_grad():
    s_geo, rigids_final = ipa_module(node_embed, rigids, z_f1, z_f2, mask)

print(f"✓ 前向传播成功")
print(f"  - s_geo: {s_geo.shape}")
print(f"  - rigids_final: Rigid对象")

# 显存检查
if torch.cuda.is_available():
    mem = torch.cuda.memory_allocated() / 1024**2
    print(f"  - GPU显存: {mem:.2f} MB")

print(f"\n✅ FlashIPA模块工作正常！")
EOF

echo ""
echo "============================================================================"
echo -e "${GREEN}✅ 测试通过！${NC}"
echo "============================================================================"

