#!/bin/bash
# FlashIPA适配器测试脚本（Linux服务器）

set -e

GREEN='\033[0;32m'
NC='\033[0m'

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

echo "============================================================================"
echo "              FlashIPA EdgeEmbedder 适配器测试"
echo "============================================================================"
echo ""

echo -e "${GREEN}[1/2]${NC} 运行单元测试..."
python -m src.stage1.modules.edge_embed

echo ""
echo -e "${GREEN}[2/2]${NC} 功能验证..."
python << 'EOF'
import torch
from src.stage1.modules.edge_embed import create_edge_embedder

B, N = 2, 50
node_embed = torch.randn(B, N, 384)
translations = torch.randn(B, N, 3)
node_mask = torch.ones(B, N, dtype=torch.bool)

if torch.cuda.is_available():
    node_embed = node_embed.cuda()
    translations = translations.cuda()
    node_mask = node_mask.cuda()

embedder = create_edge_embedder(c_s=384, c_p=128, z_rank=16)
if torch.cuda.is_available():
    embedder = embedder.cuda()

with torch.no_grad():
    outputs = embedder(node_embed, translations, node_mask)

print(f"\n✓ 输出:")
print(f"  - z_f1: {outputs['z_f1'].shape}")
print(f"  - z_f2: {outputs['z_f2'].shape}")
print(f"  - edge_mask: {outputs['edge_mask'].shape}")

if torch.cuda.is_available():
    print(f"  - GPU显存: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

print(f"\n✅ 适配器工作正常！")
EOF

echo ""
echo "============================================================================"
echo -e "${GREEN}✅ 所有测试通过！${NC}"
echo "============================================================================"

