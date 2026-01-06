#!/usr/bin/env python3
"""
Stage-2 训练启动脚本

Usage:
    python scripts/train_stage2.py [--options]
"""

import sys
from pathlib import Path
import argparse

# 避免 /dev/shm 限制导致的 DataLoader 崩溃
try:
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.stage2.training.config import TrainingConfig
from src.stage2.training.trainer import Stage2Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Stage-2 训练')

    # 数据
    parser.add_argument('--data_dir', type=str, default='data/apo_holo_triplets',
                        help='数据目录')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批大小')

    # 训练
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='最大训练轮数')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪')

    # Stage-1 prior
    parser.add_argument('--stage1_ckpt', type=str, default='checkpoints/stage1_best.pt',
                        help='Stage-1 checkpoint')
    parser.add_argument('--no_stage1_prior', action='store_true',
                        help='禁用Stage-1 prior')

    # NMA
    parser.add_argument('--use_nma', action='store_true',
                        help='启用NMA特征')
    parser.add_argument('--nma_dim', type=int, default=0,
                        help='NMA特征维度')

    # 保存
    parser.add_argument('--save_dir', type=str, default='checkpoints/stage2',
                        help='Checkpoint保存目录')
    parser.add_argument('--log_dir', type=str, default='logs/stage2',
                        help='日志目录')

    # 设备
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备')
    parser.add_argument('--no_mixed_precision', action='store_true',
                        help='禁用混合精度')

    return parser.parse_args()


def main():
    args = parse_args()

    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        grad_clip=args.grad_clip,
        stage1_ckpt=args.stage1_ckpt,
        use_stage1_prior=not args.no_stage1_prior,
        use_nma=args.use_nma,
        nma_dim=args.nma_dim,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        device=args.device,
        mixed_precision=not args.no_mixed_precision,
    )

    print(f"\n{'='*80}")
    print("BINDRAE Stage-2 训练")
    print(f"{'='*80}")
    print("\n配置:")
    print(f"  - 数据目录: {config.data_dir}")
    print(f"  - 批大小: {config.batch_size}")
    print(f"  - 学习率: {config.lr}")
    print(f"  - 最大轮数: {config.max_epochs}")
    print(f"  - Stage-1 prior: {config.use_stage1_prior}")
    print(f"  - NMA: {config.use_nma}")
    print(f"  - 设备: {config.device}")
    print(f"  - 混合精度: {config.mixed_precision}")
    print(f"\n{'='*80}\n")

    trainer = Stage2Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
