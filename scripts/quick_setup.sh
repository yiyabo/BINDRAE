#!/bin/bash
# ============================================================================
# BINDRAE 快速设置脚本（用于已处理数据）
# ============================================================================
# 
# 用途: 当你已经有处理好的数据时，快速验证和设置环境
# 
# 前置条件:
#   1. 已将 data/casf2016/ 目录上传到服务器
#   2. 已激活 conda 环境: conda activate drug
# 
# 使用方法:
#   bash scripts/quick_setup.sh
# 
# ============================================================================

set -e

# 颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

echo "============================================================================"
echo "              BINDRAE 快速环境设置 & 数据验证"
echo "============================================================================"
echo ""

# 1. 检查 conda 环境
echo -e "${GREEN}[1/5]${NC} 检查 conda 环境..."
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${RED}❌ 未激活 conda 环境！${NC}"
    echo "请运行: conda activate drug"
    exit 1
fi
echo -e "  ✓ 当前环境: $CONDA_DEFAULT_ENV"

# 2. 检查 Python 依赖
echo -e "${GREEN}[2/5]${NC} 检查 Python 依赖..."
python -c "import Bio; import rdkit; import numpy; import torch" 2>/dev/null || {
    echo -e "${RED}❌ 缺少依赖！${NC}"
    echo "请运行: pip install biopython rdkit numpy torch"
    exit 1
}
echo -e "  ✓ 核心依赖已安装"

# 3. 检查数据目录
echo -e "${GREEN}[3/5]${NC} 检查数据目录..."
REQUIRED_DIRS=(
    "data/casf2016/complexes"
    "data/casf2016/processed/features"
    "data/casf2016/processed/pockets"
    "data/casf2016/processed/splits"
)

ALL_EXISTS=true
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo -e "  ${RED}❌ 缺少目录: $dir${NC}"
        ALL_EXISTS=false
    else
        echo -e "  ✓ $dir"
    fi
done

if [ "$ALL_EXISTS" = false ]; then
    echo -e "${RED}数据目录不完整！${NC}"
    echo "请确保已上传完整的 data/casf2016/ 目录"
    exit 1
fi

# 4. 统计数据文件
echo -e "${GREEN}[4/5]${NC} 统计数据文件..."
N_COMPLEXES=$(ls data/casf2016/complexes 2>/dev/null | wc -l | tr -d ' ')
N_LIGANDS=$(ls data/casf2016/processed/features/*_ligand_coords.npy 2>/dev/null | wc -l | tr -d ' ')
N_POCKETS=$(ls data/casf2016/processed/pockets/*_w_res.npy 2>/dev/null | wc -l | tr -d ' ')
N_TORSIONS=$(ls data/casf2016/processed/features/*_torsions.npz 2>/dev/null | wc -l | tr -d ' ')
N_ESM=$(ls data/casf2016/processed/features/*_esm.pt 2>/dev/null | wc -l | tr -d ' ')

echo "  - 复合物: $N_COMPLEXES"
echo "  - 配体特征: $N_LIGANDS"
echo "  - 口袋数据: $N_POCKETS"
echo "  - 扭转角: $N_TORSIONS"
echo "  - ESM-2 缓存: $N_ESM"

if [ "$N_COMPLEXES" -lt 280 ] || [ "$N_LIGANDS" -lt 280 ] || [ "$N_POCKETS" -lt 280 ]; then
    echo -e "${YELLOW}⚠️  警告: 数据文件数量少于预期 (应该是 283)${NC}"
fi

# 5. 运行数据验证
echo -e "${GREEN}[5/5]${NC} 运行数据验证..."
if [ -f "scripts/validate_data.py" ]; then
    python scripts/validate_data.py || {
        echo -e "${YELLOW}⚠️  数据验证发现一些问题${NC}"
    }
else
    echo -e "${YELLOW}⚠️  未找到验证脚本，跳过${NC}"
fi

# 完成
echo ""
echo "============================================================================"
echo -e "${GREEN}✅ 环境设置完成！${NC}"
echo "============================================================================"
echo ""
echo "📊 数据统计:"
echo "  - 总复合物: $N_COMPLEXES"
echo "  - 总大小: $(du -sh data/casf2016 2>/dev/null | cut -f1)"
echo ""
echo "🚀 下一步:"
echo "  1. 开始训练: python scripts/train_stage1_ipa.py"
echo "  2. 查看todo: cat todo.md"
echo ""
