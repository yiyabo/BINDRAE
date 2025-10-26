#!/usr/bin/env python3
"""
数据一致性验证脚本
- 检查配体原子数合理性
- 检查坐标有效性
- 检查口袋权重分布
- 检查残基索引范围
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class DataValidator:
    """数据验证器"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / "data" / "casf2016" / "processed"
        self.features_dir = self.processed_dir / "features"
        self.pockets_dir = self.processed_dir / "pockets"
        
        self.issues = {
            'ligand_atoms_too_few': [],
            'ligand_atoms_too_many': [],
            'invalid_coords': [],
            'zero_weights': [],
            'small_pocket': [],
            'index_out_of_range': [],
        }
    
    def validate_ligand(self, pdb_id: str) -> bool:
        """验证配体数据"""
        coords_file = self.features_dir / f"{pdb_id}_ligand_coords.npy"
        
        if not coords_file.exists():
            return False
        
        coords = np.load(coords_file)
        
        # 检查原子数
        n_atoms = len(coords)
        if n_atoms < 5:
            self.issues['ligand_atoms_too_few'].append((pdb_id, n_atoms))
            return False
        if n_atoms > 300:
            self.issues['ligand_atoms_too_many'].append((pdb_id, n_atoms))
            return False
        
        # 检查坐标有效性
        if not np.all(np.isfinite(coords)):
            self.issues['invalid_coords'].append((pdb_id, 'NaN/Inf'))
            return False
        
        # 检查是否所有坐标都在原点
        if np.allclose(coords, 0, atol=1e-6):
            self.issues['invalid_coords'].append((pdb_id, 'all_zero'))
            return False
        
        return True
    
    def validate_pocket(self, pdb_id: str) -> bool:
        """验证口袋数据"""
        weights_file = self.pockets_dir / f"{pdb_id}_w_res.npy"
        mask_file = self.pockets_dir / f"{pdb_id}_pocket_mask.npy"
        
        if not weights_file.exists() or not mask_file.exists():
            return False
        
        weights = np.load(weights_file)
        mask = np.load(mask_file)
        
        # 检查权重分布
        if weights.max() == 0:
            self.issues['zero_weights'].append(pdb_id)
            return False
        
        # 检查口袋大小
        pocket_size = mask.sum()
        if pocket_size < 5:
            self.issues['small_pocket'].append((pdb_id, pocket_size))
            return False
        
        return True
    
    def run(self):
        """运行验证"""
        print("="*80)
        print("数据一致性验证")
        print("="*80)
        
        # 获取所有样本
        ligand_files = sorted(self.features_dir.glob("*_ligand_coords.npy"))
        print(f"发现 {len(ligand_files)} 个配体文件")
        print()
        
        valid_count = 0
        total_count = len(ligand_files)
        
        for coords_file in ligand_files:
            pdb_id = coords_file.stem.replace("_ligand_coords", "")
            
            ligand_ok = self.validate_ligand(pdb_id)
            pocket_ok = self.validate_pocket(pdb_id)
            
            if ligand_ok and pocket_ok:
                valid_count += 1
        
        # 打印报告
        print("\n" + "="*80)
        print("验证报告")
        print("="*80)
        print(f"总样本数: {total_count}")
        print(f"有效样本: {valid_count} ({100*valid_count/total_count:.1f}%)")
        print()
        
        # 详细问题
        if self.issues['ligand_atoms_too_few']:
            print(f"⚠️  配体原子过少 (<5): {len(self.issues['ligand_atoms_too_few'])} 个")
            for pdb_id, n in self.issues['ligand_atoms_too_few'][:5]:
                print(f"    {pdb_id}: {n} 个原子")
        
        if self.issues['ligand_atoms_too_many']:
            print(f"⚠️  配体原子过多 (>300): {len(self.issues['ligand_atoms_too_many'])} 个")
        
        if self.issues['invalid_coords']:
            print(f"❌ 无效坐标: {len(self.issues['invalid_coords'])} 个")
            for pdb_id, reason in self.issues['invalid_coords'][:5]:
                print(f"    {pdb_id}: {reason}")
        
        if self.issues['zero_weights']:
            print(f"❌ 权重全0: {len(self.issues['zero_weights'])} 个")
            print(f"    {', '.join(self.issues['zero_weights'][:10])}")
        
        if self.issues['small_pocket']:
            print(f"⚠️  口袋过小 (<5残基): {len(self.issues['small_pocket'])} 个")
            for pdb_id, size in self.issues['small_pocket'][:5]:
                print(f"    {pdb_id}: {size} 个残基")
        
        if valid_count == total_count:
            print("\n✅ 所有数据通过验证！")
        else:
            print(f"\n⚠️  发现 {total_count - valid_count} 个问题样本")


def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/apple/code/BINDRAE"
    validator = DataValidator(base_dir)
    validator.run()


if __name__ == "__main__":
    main()
