#!/usr/bin/env python3
"""
Stage-1 训练启动脚本

Usage:
    # 单卡训练
    python scripts/train_stage1.py [--options]
    
    # 多卡 DDP 训练
    torchrun --nproc_per_node=2 scripts/train_stage1.py --distributed [--options]

Author: BINDRAE Team
Date: 2025-10-28
"""

import os
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

from src.stage1.training.config import TrainingConfig
from src.stage1.training.trainer import Stage1Trainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Stage-1 训练')
    
    # 数据
    parser.add_argument('--data_dir', type=str, default='data/apo_holo_triplets',
                       help='数据目录')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批大小')
    parser.add_argument('--max_n_res', type=int, default=None,
                       help='每个batch的最大残基数（超过会过滤）')
    parser.add_argument('--valid_samples_file', type=str, default=None,
                       help='训练集有效样本列表（由scripts/validate_triplets_data.py生成）')
    parser.add_argument('--val_samples_file', type=str, default=None,
                       help='验证集样本列表（可选，不指定则使用完整验证集）')
    
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

    # 保存
    parser.add_argument('--save_dir', type=str, default='checkpoints/stage1',
                       help='Checkpoint保存目录')
    parser.add_argument('--log_dir', type=str, default='logs/stage1',
                       help='日志目录')
    
    # 数据加载
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers (如遇shm不足可设为0)')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备（单卡模式）')
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='禁用混合精度')
    
    # 分布式训练
    parser.add_argument('--distributed', action='store_true',
                       help='启用 DDP 分布式训练（配合 torchrun 使用）')
    
    # 恢复训练
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从checkpoint恢复')
    
    args = parser.parse_args()
    
    # 自动检测 torchrun 环境
    if 'LOCAL_RANK' in os.environ:
        args.distributed = True
    
    return args


def main():
    """主函数"""
    args = parse_args()
    
    # 分布式训练时，local_rank 从环境变量获取
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) if args.distributed else 0
    is_main_process = (local_rank == 0)
    
    # 创建配置
    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_n_res=args.max_n_res,
        valid_samples_file=args.valid_samples_file,
        val_samples_file=args.val_samples_file,
        num_workers=args.num_workers,
        lr=args.lr,
        max_epochs=args.max_epochs,
        grad_clip=args.grad_clip,
        early_stop_patience=args.patience,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        device=args.device,
        mixed_precision=not args.no_mixed_precision,
        distributed=args.distributed,
        resume_from=args.resume_from,
    )
    
    # 只在主进程打印配置
    if is_main_process:
    print(f"\n{'='*80}")
    print(f"BINDRAE Stage-1 训练")
    print(f"{'='*80}")
    print(f"\n配置:")
    print(f"  - 数据目录: {config.data_dir}")
    print(f"  - 批大小: {config.batch_size}")
    print(f"  - 最大残基数: {config.max_n_res}")
    print(f"  - 学习率: {config.lr}")
    print(f"  - 最大轮数: {config.max_epochs}")
    print(f"  - 早停patience: {config.early_stop_patience}")
    print(f"  - 设备: {config.device}")
    print(f"  - 混合精度: {config.mixed_precision}")
        print(f"  - 分布式训练: {config.distributed}")
    print(f"\n{'='*80}\n")
    
    # 创建训练器
    trainer = Stage1Trainer(config)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
