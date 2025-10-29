#!/bin/bash
# 测试训练能否启动（只跑2个step）

set -e

GREEN='\033[0;32m'
NC='\033[0m'

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

echo "============================================================================"
echo "              训练启动测试（快速验证）"
echo "============================================================================"
echo ""

echo -e "${GREEN}测试：能否启动训练并运行2个step${NC}\n"

python << 'EOF'
import sys
from pathlib import Path

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from src.stage1.training import TrainingConfig, Stage1Trainer

# 创建简化配置（只跑2个step）
config = TrainingConfig(
    data_dir='data/casf2016',
    batch_size=2,
    max_epochs=1,  # 只跑1个epoch
    num_workers=0,
    log_interval=1,
    val_interval=1,
)

print("创建训练器...")
trainer = Stage1Trainer(config)

print("\n开始训练（只跑前2个batch）...")

# 手动跑2个step
for i, batch in enumerate(trainer.train_loader):
    if i >= 2:  # 只跑2个step
        break
    
    print(f"\nStep {i+1}/2:")
    losses = trainer.train_step(batch)
    print(f"  - Total loss: {losses['total']:.4f}")
    print(f"  - Torsion loss: {losses['torsion']:.4f}")
    print(f"  - LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")

print(f"\n✅ 训练启动成功！可以开始完整训练了")
EOF

echo ""
echo "============================================================================"
echo -e "${GREEN}✅ 测试通过！训练循环工作正常${NC}"
echo "============================================================================"

