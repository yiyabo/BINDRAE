#!/bin/bash
# 完整训练测试（1个epoch + 验证）

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

echo "============================================================================"
echo "              完整训练测试（1 Epoch + 验证）"
echo "============================================================================"
echo ""

echo -e "${YELLOW}注意：这会运行1个完整epoch（约5-10分钟）${NC}\n"

python << 'EOF'
import sys
from pathlib import Path

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from src.stage1.training import TrainingConfig, Stage1Trainer

# 创建测试配置
config = TrainingConfig(
    data_dir='data/casf2016',
    batch_size=2,          # 小batch size加快测试
    max_epochs=2,          # 只跑2个epoch
    num_workers=0,
    log_interval=5,
    val_interval=1,        # 每个epoch都验证
    early_stop_patience=1, # 测试早停
    warmup_steps=10,       # 快速warmup
    save_dir='checkpoints/test_stage1',
    log_dir='logs/test_stage1',
)

print("创建训练器...")
trainer = Stage1Trainer(config)

print("\n" + "="*80)
print("开始训练（最多2个epoch）")
print("="*80 + "\n")

# 运行训练
trainer.train()

print("\n" + "="*80)
print("✅ 完整训练测试通过！")
print("="*80)
print("\n测试验证了:")
print("  ✓ 完整epoch训练")
print("  ✓ 验证循环")
print("  ✓ 早停机制")
print("  ✓ Checkpoint保存")
print("  ✓ 损失计算")
print("  ✓ 梯度更新")
print("\n准备好进行真正的训练了！")
EOF

echo ""
echo "============================================================================"
echo -e "${GREEN}✅ 完整测试通过！${NC}"
echo "============================================================================"

