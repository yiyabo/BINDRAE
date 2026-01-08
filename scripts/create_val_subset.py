#!/usr/bin/env python3
"""
从验证集中随机采样创建小验证集文件

用于快速调试时减少验证时间

Usage:
    python scripts/create_val_subset.py \
        --data_dir data/apo_holo_triplets \
        --n_samples 1000 \
        --output val_samples_1k.txt
"""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Create validation subset file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录 (如 data/apo_holo_triplets)')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='采样数量 (默认 1000)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件名 (默认: val_samples_{n}.txt)')
    parser.add_argument('--seed', type=int, default=2025,
                       help='随机种子')
    parser.add_argument('--valid_samples_file', type=str, default=None,
                       help='已有的有效样本文件 (用于过滤)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # 读取验证集 split
    split_file = data_dir / 'splits' / 'val.json'
    if not split_file.exists():
        print(f"错误: 找不到验证集 split 文件: {split_file}")
        return 1
    
    with open(split_file) as f:
        val_samples = json.load(f)
    
    print(f"验证集总样本数: {len(val_samples)}")
    
    # 如果指定了 valid_samples_file，则只保留有效的样本
    if args.valid_samples_file:
        valid_file = data_dir / args.valid_samples_file if not Path(args.valid_samples_file).is_absolute() else Path(args.valid_samples_file)
        if valid_file.exists():
            with open(valid_file) as f:
                valid_ids = set(line.strip() for line in f if line.strip())
            original_count = len(val_samples)
            val_samples = [s for s in val_samples if s in valid_ids]
            print(f"过滤后有效样本数: {len(val_samples)} (过滤了 {original_count - len(val_samples)} 个)")
        else:
            print(f"警告: 找不到 valid_samples_file: {valid_file}")
    
    # 随机采样
    random.seed(args.seed)
    n_samples = min(args.n_samples, len(val_samples))
    subset = random.sample(val_samples, n_samples)
    
    # 输出文件
    if args.output:
        output_file = data_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    else:
        output_file = data_dir / f'val_samples_{n_samples}.txt'
    
    with open(output_file, 'w') as f:
        for sample_id in subset:
            f.write(f"{sample_id}\n")
    
    print(f"✓ 已创建验证子集: {output_file}")
    print(f"  - 样本数: {n_samples}")
    print(f"  - 随机种子: {args.seed}")
    
    return 0


if __name__ == '__main__':
    exit(main())
