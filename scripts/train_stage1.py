#!/usr/bin/env python3
"""
Stage-1 训练启动脚本

Usage:
    python scripts/train_stage1.py [--options]

Author: BINDRAE Team
Date: 2025-10-28
"""

import sys
from pathlib import Path
import argparse

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.stage1.training.config import TrainingConfig
from src.stage1.training.trainer import Stage1Trainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Stage-1 训练')
    
    # 数据
    parser.add_argument('--data_dir', type=str, default='data/casf2016',
                       help='数据目录')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批大小')
    
    # 训练
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='最大训练轮数')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='梯度裁剪')
    
    # 早停
    parser.add_argument('--patience', type=int, default=20,
                       help='早停patience')

    # χ1 rotamer 辅助损失
    parser.add_argument('--rotamer_loss_weight', type=float, default=0.0,
                       help='χ1 rotamer loss 权重')
    
    # 保存
    parser.add_argument('--save_dir', type=str, default='checkpoints/stage1',
                       help='Checkpoint保存目录')
    parser.add_argument('--log_dir', type=str, default='logs/stage1',
                       help='日志目录')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备')
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='禁用混合精度')
    
    # 恢复训练
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从checkpoint恢复')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建配置
    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        grad_clip=args.grad_clip,
        early_stop_patience=args.patience,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        device=args.device,
        mixed_precision=not args.no_mixed_precision,
        resume_from=args.resume_from,
        w_rotamer=args.rotamer_loss_weight,
    )
    
    print(f"\n{'='*80}")
    print(f"BINDRAE Stage-1 训练")
    print(f"{'='*80}")
    print(f"\n配置:")
    print(f"  - 数据目录: {config.data_dir}")
    print(f"  - 批大小: {config.batch_size}")
    print(f"  - 学习率: {config.lr}")
    print(f"  - 最大轮数: {config.max_epochs}")
    print(f"  - 早停patience: {config.early_stop_patience}")
    print(f"  - 设备: {config.device}")
    print(f"  - 混合精度: {config.mixed_precision}")
    print(f"\n{'='*80}\n")
    
    # 创建训练器
    trainer = Stage1Trainer(config)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()

