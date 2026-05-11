#!/bin/bash
# ============================================================================
# BINDRAE 服务器端数据准备脚本
# ============================================================================
# 
# 用途: 在服务器上从原始 CASF-2016 数据生成所有训练所需文件
# 
# 前置条件:
#   1. 已将 data/CASF-2016.tar.gz 上传到服务器
#   2. 已激活 conda 环境: conda activate drug
#   3. 已安装所有依赖: pip install -r requirements.txt
# 
# 使用方法:
#   bash scripts/setup_server_data.sh
# 
# 处理流程:
#   [1/7] 解压原始数据
#   [2/7] 准备复合物
#   [3/7] 配体规范化
#   [4/7] 口袋软掩码提取
#   [5/7] 扭转角提取
#   [6/7] 数据集划分
#   [7/7] ESM-2 缓存
# 
# ============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

# 日志文件
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/data_setup_$(date +%Y%m%d_%H%M%S).log"

# 日志函数
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# 检查函数
check_file() {
    if [ ! -f "$1" ]; then
        log_error "文件不存在: $1"
        return 1
    fi
    return 0
}

check_conda_env() {
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        log_error "未激活 conda 环境！请运行: conda activate drug"
        exit 1
    fi
    if [ "$CONDA_DEFAULT_ENV" != "drug" ]; then
        log_warning "当前环境是 $CONDA_DEFAULT_ENV，建议使用 drug 环境"
    fi
}

# ============================================================================
# 主流程
# ============================================================================

echo "============================================================================"
echo "                    BINDRAE 服务器端数据准备"
echo "============================================================================"
echo ""
log "开始数据准备流程..."
log "项目路径: $PROJECT_ROOT"
log "日志文件: $LOG_FILE"
echo ""

# 检查环境
log "检查运行环境..."
check_conda_env

# 检查 Python
if ! command -v python &> /dev/null; then
    log_error "未找到 Python！"
    exit 1
fi
PYTHON_VERSION=$(python --version 2>&1)
log_info "Python 版本: $PYTHON_VERSION"

# 检查必要的依赖
log "检查 Python 依赖..."
python -c "import Bio; import rdkit; import numpy; import torch" 2>/dev/null || {
    log_error "缺少必要的 Python 依赖！请运行: pip install -r requirements.txt"
    exit 1
}
log_info "✓ 依赖检查通过"
echo ""

# ============================================================================
# [1/7] 解压原始数据
# ============================================================================
log "====== [1/7] 解压原始 CASF-2016 数据 ======"

if [ ! -f "data/CASF-2016.tar.gz" ]; then
    log_error "未找到 data/CASF-2016.tar.gz"
    log_info "请先将原始数据上传到服务器！"
    exit 1
fi

if [ ! -d "data/CASF-2016" ]; then
    log "正在解压 CASF-2016.tar.gz..."
    cd data
    tar -xzf CASF-2016.tar.gz
    cd "$PROJECT_ROOT"
    log_info "✓ 解压完成"
else
    log_info "✓ CASF-2016 目录已存在，跳过解压"
fi

# 检查解压后的目录
if [ ! -d "data/CASF-2016/coreset" ]; then
    log_error "解压后的目录结构不正确！"
    exit 1
fi
log_info "✓ 原始数据完整性验证通过"
echo ""

# ============================================================================
# [2/7] 准备复合物（清洗、组织目录）
# ============================================================================
log "====== [2/7] 准备复合物结构 ======"

if [ ! -d "data/casf2016/complexes" ]; then
    log "运行 prepare_casf2016.py..."
    python scripts/prepare_casf2016.py 2>&1 | tee -a "$LOG_FILE"
    
    # 检查输出
    if [ ! -d "data/casf2016/complexes" ]; then
        log_error "prepare_casf2016.py 执行失败！"
        exit 1
    fi
    
    # 统计复合物数量
    N_COMPLEXES=$(ls data/casf2016/complexes | wc -l | tr -d ' ')
    log_info "✓ 复合物准备完成: $N_COMPLEXES 个"
else
    N_COMPLEXES=$(ls data/casf2016/complexes | wc -l | tr -d ' ')
    log_info "✓ 复合物目录已存在: $N_COMPLEXES 个"
fi
echo ""

# ============================================================================
# [3/7] 配体规范化
# ============================================================================
log "====== [3/7] 配体规范化处理 ======"

if [ ! -d "data/casf2016/processed/features" ] || [ -z "$(ls -A data/casf2016/processed/features/*_ligand*.npy 2>/dev/null)" ]; then
    log "运行 prepare_ligands.py..."
    python scripts/prepare_ligands.py 2>&1 | tee -a "$LOG_FILE"
    
    # 检查输出
    N_LIGANDS=$(ls data/casf2016/processed/features/*_ligand_coords.npy 2>/dev/null | wc -l | tr -d ' ')
    if [ "$N_LIGANDS" -eq 0 ]; then
        log_error "prepare_ligands.py 执行失败！"
        exit 1
    fi
    
    log_info "✓ 配体规范化完成: $N_LIGANDS 个"
else
    N_LIGANDS=$(ls data/casf2016/processed/features/*_ligand_coords.npy 2>/dev/null | wc -l | tr -d ' ')
    log_info "✓ 配体特征已存在: $N_LIGANDS 个"
fi
echo ""

# ============================================================================
# [4/7] 口袋软掩码提取
# ============================================================================
log "====== [4/7] 提取结合口袋 ======"

if [ ! -d "data/casf2016/processed/pockets" ] || [ -z "$(ls -A data/casf2016/processed/pockets/*_w_res.npy 2>/dev/null)" ]; then
    log "运行 extract_pockets.py..."
    python scripts/extract_pockets.py 2>&1 | tee -a "$LOG_FILE"
    
    # 检查输出
    N_POCKETS=$(ls data/casf2016/processed/pockets/*_w_res.npy 2>/dev/null | wc -l | tr -d ' ')
    if [ "$N_POCKETS" -eq 0 ]; then
        log_error "extract_pockets.py 执行失败！"
        exit 1
    fi
    
    log_info "✓ 口袋提取完成: $N_POCKETS 个"
else
    N_POCKETS=$(ls data/casf2016/processed/pockets/*_w_res.npy 2>/dev/null | wc -l | tr -d ' ')
    log_info "✓ 口袋数据已存在: $N_POCKETS 个"
fi
echo ""

# ============================================================================
# [5/7] 扭转角提取
# ============================================================================
log "====== [5/7] 提取扭转角 (φ/ψ/ω/χ) ======"

if [ ! -d "data/casf2016/processed/features" ] || [ -z "$(ls -A data/casf2016/processed/features/*_torsions.npz 2>/dev/null)" ]; then
    log "运行 extract_torsions.py..."
    python scripts/extract_torsions.py 2>&1 | tee -a "$LOG_FILE"
    
    # 检查输出
    N_TORSIONS=$(ls data/casf2016/processed/features/*_torsions.npz 2>/dev/null | wc -l | tr -d ' ')
    if [ "$N_TORSIONS" -eq 0 ]; then
        log_error "extract_torsions.py 执行失败！"
        exit 1
    fi
    
    log_info "✓ 扭转角提取完成: $N_TORSIONS 个"
else
    N_TORSIONS=$(ls data/casf2016/processed/features/*_torsions.npz 2>/dev/null | wc -l | tr -d ' ')
    log_info "✓ 扭转角数据已存在: $N_TORSIONS 个"
fi
echo ""

# ============================================================================
# [6/7] 数据集划分
# ============================================================================
log "====== [6/7] 数据集划分 (train/val/test) ======"

if [ ! -f "data/casf2016/processed/splits/train.json" ]; then
    log "运行 split_dataset.py..."
    python scripts/split_dataset.py 2>&1 | tee -a "$LOG_FILE"
    
    # 检查输出
    if [ ! -f "data/casf2016/processed/splits/train.json" ]; then
        log_error "split_dataset.py 执行失败！"
        exit 1
    fi
    
    # 统计各集合大小
    N_TRAIN=$(python -c "import json; print(len(json.load(open('data/casf2016/processed/splits/train.json'))))")
    N_VAL=$(python -c "import json; print(len(json.load(open('data/casf2016/processed/splits/val.json'))))")
    N_TEST=$(python -c "import json; print(len(json.load(open('data/casf2016/processed/splits/test.json'))))")
    
    log_info "✓ 数据集划分完成: train=$N_TRAIN, val=$N_VAL, test=$N_TEST"
else
    N_TRAIN=$(python -c "import json; print(len(json.load(open('data/casf2016/processed/splits/train.json'))))")
    N_VAL=$(python -c "import json; print(len(json.load(open('data/casf2016/processed/splits/val.json'))))")
    N_TEST=$(python -c "import json; print(len(json.load(open('data/casf2016/processed/splits/test.json'))))")
    
    log_info "✓ 数据集划分已存在: train=$N_TRAIN, val=$N_VAL, test=$N_TEST"
fi
echo ""

# ============================================================================
# [7/7] ESM-2 编码器缓存
# ============================================================================
log "====== [7/7] 缓存 ESM-2 蛋白质表征 ======"

if [ ! -d "data/casf2016/processed/features/esm2_cache" ] || [ -z "$(ls -A data/casf2016/processed/features/esm2_cache/*.pt 2>/dev/null)" ]; then
    log "运行 cache_esm2.py..."
    log_warning "此步骤可能需要较长时间（约10-30分钟），并需要~3GB显存"
    
    python scripts/cache_esm2.py 2>&1 | tee -a "$LOG_FILE"
    
    # 检查输出
    N_ESM=$(ls data/casf2016/processed/features/esm2_cache/*.pt 2>/dev/null | wc -l | tr -d ' ')
    if [ "$N_ESM" -eq 0 ]; then
        log_error "cache_esm2.py 执行失败！"
        log_warning "ESM-2 缓存失败不会影响其他数据，可以稍后手动运行"
    else
        log_info "✓ ESM-2 缓存完成: $N_ESM 个"
    fi
else
    N_ESM=$(ls data/casf2016/processed/features/esm2_cache/*.pt 2>/dev/null | wc -l | tr -d ' ')
    log_info "✓ ESM-2 缓存已存在: $N_ESM 个"
fi
echo ""

# ============================================================================
# 数据验证
# ============================================================================
log "====== 数据完整性验证 ======"

log "运行 validate_data.py..."
python scripts/validate_data.py 2>&1 | tee -a "$LOG_FILE" || {
    log_warning "数据验证发现一些问题，请查看日志"
}
echo ""

# ============================================================================
# 生成统计报告
# ============================================================================
log "====== 生成数据统计 ======"

# 计算总文件数
TOTAL_FILES=$(find data/casf2016/processed -type f | wc -l | tr -d ' ')

# 计算总大小
TOTAL_SIZE=$(du -sh data/casf2016 | cut -f1)

# 输出统计
echo ""
echo "============================================================================"
echo "                         数据准备完成"
echo "============================================================================"
echo ""
echo "📊 统计信息:"
echo "  - 复合物数量: $N_COMPLEXES"
echo "  - 配体特征: $N_LIGANDS"
echo "  - 口袋数据: $N_POCKETS"
echo "  - 扭转角数据: $N_TORSIONS"
echo "  - ESM-2 缓存: $N_ESM"
echo "  - 训练集: $N_TRAIN"
echo "  - 验证集: $N_VAL"
echo "  - 测试集: $N_TEST"
echo "  - 总文件数: $TOTAL_FILES"
echo "  - 总大小: $TOTAL_SIZE"
echo ""
echo "📁 数据目录: $PROJECT_ROOT/data/casf2016"
echo "📝 日志文件: $LOG_FILE"
echo ""
echo "✅ 所有数据准备完成！可以开始训练了。"
echo ""
echo "下一步:"
echo "  1. 检查数据: python scripts/verify_casf2016.py"
echo "  2. 开始训练: python scripts/train_stage1_ipa.py --config configs/stage1_ipa.yaml"
echo ""
echo "============================================================================"

# 保存统计到文件
STATS_FILE="data/casf2016/SETUP_STATS.txt"
cat > "$STATS_FILE" << EOF
BINDRAE 数据准备统计
生成时间: $(date)
服务器: $(hostname)

复合物数量: $N_COMPLEXES
配体特征: $N_LIGANDS
口袋数据: $N_POCKETS
扭转角数据: $N_TORSIONS
ESM-2 缓存: $N_ESM
训练集: $N_TRAIN
验证集: $N_VAL
测试集: $N_TEST
总文件数: $TOTAL_FILES
总大小: $TOTAL_SIZE

日志文件: $LOG_FILE
EOF

log "统计信息已保存到: $STATS_FILE"
log "数据准备流程全部完成！🎉"
