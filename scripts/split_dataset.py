#!/usr/bin/env python3
"""
数据集划分脚本
- 8:1:1 随机划分（train/val/test）
- 固定随机种子 2025
- 确保同一 PDB ID 不跨分割
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict


class DatasetSplitter:
    """数据集划分器"""
    
    def __init__(self, base_dir: str, train_ratio: float = 0.8, 
                 val_ratio: float = 0.1, test_ratio: float = 0.1, 
                 random_seed: int = 2025):
        """
        Args:
            base_dir: 项目根目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
        """
        self.base_dir = Path(base_dir)
        self.complexes_dir = self.base_dir / "data" / "casf2016" / "complexes"
        self.processed_dir = self.base_dir / "data" / "casf2016" / "processed"
        self.splits_dir = self.processed_dir / "splits"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        
        # 创建输出目录
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def get_all_pdb_ids(self) -> List[str]:
        """获取所有有效的 PDB ID"""
        pdb_ids = []
        
        for complex_dir in self.complexes_dir.iterdir():
            if complex_dir.is_dir():
                pdb_id = complex_dir.name
                
                # 检查必要文件是否存在
                protein_pdb = complex_dir / "protein.pdb"
                ligand_sdf = complex_dir / "ligand.sdf"
                
                if protein_pdb.exists() and ligand_sdf.exists():
                    pdb_ids.append(pdb_id)
        
        return sorted(pdb_ids)
    
    def split_dataset(self, pdb_ids: List[str]) -> Dict[str, List[str]]:
        """
        划分数据集
        
        Args:
            pdb_ids: PDB ID 列表
            
        Returns:
            划分结果字典
        """
        n_total = len(pdb_ids)
        
        # 计算划分数量
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        n_test = n_total - n_train - n_val  # 确保加起来等于总数
        
        # 打乱顺序
        shuffled_ids = pdb_ids.copy()
        random.shuffle(shuffled_ids)
        
        # 划分
        train_ids = shuffled_ids[:n_train]
        val_ids = shuffled_ids[n_train:n_train+n_val]
        test_ids = shuffled_ids[n_train+n_val:]
        
        return {
            'train': sorted(train_ids),
            'val': sorted(val_ids),
            'test': sorted(test_ids)
        }
    
    def save_splits(self, splits: Dict[str, List[str]]):
        """保存划分结果到 JSON 文件"""
        for split_name, pdb_ids in splits.items():
            output_file = self.splits_dir / f"{split_name}.json"
            
            # 保存为带元数据的格式
            data = {
                'split': split_name,
                'n_samples': len(pdb_ids),
                'random_seed': self.random_seed,
                'pdb_ids': pdb_ids
            }
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"  ✓ {split_name:5s}: {len(pdb_ids):3d} 个样本 → {output_file.name}")
    
    def run(self):
        """运行划分流程"""
        print("="*80)
        print("数据集划分")
        print("="*80)
        print(f"输入目录: {self.complexes_dir}")
        print(f"输出目录: {self.splits_dir}")
        print(f"随机种子: {self.random_seed}")
        print(f"划分比例: {self.train_ratio:.1f} / {self.val_ratio:.1f} / {self.test_ratio:.1f}")
        print()
        
        # 获取所有 PDB ID
        print("收集有效复合物...")
        pdb_ids = self.get_all_pdb_ids()
        print(f"✓ 发现 {len(pdb_ids)} 个有效复合物")
        print()
        
        # 划分数据集
        print("划分数据集...")
        splits = self.split_dataset(pdb_ids)
        print()
        
        # 保存结果
        print("保存划分结果...")
        self.save_splits(splits)
        print()
        
        # 打印统计
        print("="*80)
        print("划分完成统计")
        print("="*80)
        print(f"总样本数:   {len(pdb_ids)}")
        print(f"训练集:     {len(splits['train']):3d} ({100*len(splits['train'])/len(pdb_ids):.1f}%)")
        print(f"验证集:     {len(splits['val']):3d} ({100*len(splits['val'])/len(pdb_ids):.1f}%)")
        print(f"测试集:     {len(splits['test']):3d} ({100*len(splits['test'])/len(pdb_ids):.1f}%)")
        
        # 验收检查
        expected_train = int(len(pdb_ids) * 0.8)
        expected_val = int(len(pdb_ids) * 0.1)
        
        print(f"\n验收检查:")
        if abs(len(splits['train']) - expected_train) <= 1:
            print(f"  ✅ 训练集数量正确 (~{expected_train})")
        else:
            print(f"  ⚠️  训练集数量异常 (预期 ~{expected_train}, 实际 {len(splits['train'])})")
        
        if abs(len(splits['val']) - expected_val) <= 1:
            print(f"  ✅ 验证集数量正确 (~{expected_val})")
        else:
            print(f"  ⚠️  验证集数量异常 (预期 ~{expected_val}, 实际 {len(splits['val'])})")
        
        # 检查是否有重复
        all_ids = set(splits['train']) | set(splits['val']) | set(splits['test'])
        if len(all_ids) == len(pdb_ids):
            print(f"  ✅ 无重复样本")
        else:
            print(f"  ❌ 存在重复样本！")
        
        print("\n✓ 数据集划分完成！")
        print(f"\n输出文件:")
        print(f"  - train.json  (训练集)")
        print(f"  - val.json    (验证集)")
        print(f"  - test.json   (测试集)")
        print(f"\n提示: 后续可通过序列聚类（MMseqs2/CD-HIT）替换 val/test 列表")


def main():
    """主函数"""
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/apple/code/BINDRAE"
    
    # 可以通过命令行参数调整比例和种子
    train_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    val_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    test_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
    random_seed = int(sys.argv[5]) if len(sys.argv) > 5 else 2025
    
    splitter = DatasetSplitter(base_dir, train_ratio, val_ratio, test_ratio, random_seed)
    splitter.run()


if __name__ == "__main__":
    main()
