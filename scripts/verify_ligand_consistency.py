#!/usr/bin/env python3
"""
验证配体数据一致性

检查 *_ligand_coords.npy 和 *_ligand_normalized.sdf 是否一致

Usage:
    python scripts/verify_ligand_consistency.py
"""

import sys
import numpy as np
from pathlib import Path
from rdkit import Chem

def verify_consistency(base_dir: Path):
    """验证所有配体数据一致性"""
    
    features_dir = base_dir / "data" / "casf2016" / "processed" / "features"
    
    if not features_dir.exists():
        print(f"❌ 特征目录不存在: {features_dir}")
        return
    
    # 查找所有坐标文件
    coord_files = sorted(features_dir.glob("*_ligand_coords.npy"))
    
    print("="*80)
    print(f"配体数据一致性验证")
    print("="*80)
    print(f"特征目录: {features_dir}")
    print(f"发现 {len(coord_files)} 个坐标文件\n")
    
    stats = {
        'total': 0,
        'consistent': 0,
        'inconsistent': 0,
        'no_sdf': 0,
        'sdf_load_failed': 0,
        'inconsistent_ids': []
    }
    
    for coord_file in coord_files:
        stats['total'] += 1
        pdb_id = coord_file.stem.replace('_ligand_coords', '')
        
        # 加载坐标
        coords = np.load(coord_file)
        n_coords = len(coords)
        
        # 查找对应的SDF
        sdf_file = features_dir / f"{pdb_id}_ligand_normalized.sdf"
        
        if not sdf_file.exists():
            stats['no_sdf'] += 1
            print(f"  ⚠️  {pdb_id}: 无SDF文件 (coords={n_coords})")
            continue
        
        # 加载SDF
        try:
            supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=False)
            mol = supplier[0]
            
            if mol is None:
                stats['sdf_load_failed'] += 1
                print(f"  ❌ {pdb_id}: SDF加载失败")
                continue
            
            n_mol = mol.GetNumAtoms()
            
            # 检查一致性
            if n_mol == n_coords:
                stats['consistent'] += 1
                if stats['total'] % 50 == 0:
                    print(f"  ✓ {pdb_id}: 一致 ({n_coords}原子)")
            else:
                stats['inconsistent'] += 1
                stats['inconsistent_ids'].append(pdb_id)
                print(f"  ❌ {pdb_id}: 不一致 (mol={n_mol}, coords={n_coords}, diff={abs(n_mol-n_coords)})")
        
        except Exception as e:
            stats['sdf_load_failed'] += 1
            print(f"  ❌ {pdb_id}: SDF处理异常 - {e}")
    
    # 统计报告
    print("\n" + "="*80)
    print("验证报告")
    print("="*80)
    print(f"总计:           {stats['total']}")
    print(f"一致:           {stats['consistent']} ({100*stats['consistent']/max(1,stats['total']):.1f}%)")
    print(f"不一致:         {stats['inconsistent']}")
    print(f"无SDF:          {stats['no_sdf']}")
    print(f"SDF加载失败:    {stats['sdf_load_failed']}")
    
    if stats['inconsistent_ids']:
        print(f"\n不一致的条目:")
        for pdb_id in stats['inconsistent_ids'][:20]:
            print(f"  - {pdb_id}")
        if len(stats['inconsistent_ids']) > 20:
            print(f"  ... 还有 {len(stats['inconsistent_ids'])-20} 个")
    
    # 建议
    print("\n" + "="*80)
    if stats['inconsistent'] > 0 or stats['sdf_load_failed'] > 0:
        print("⚠️  发现数据不一致！")
        print("\n建议操作:")
        print("1. 重新运行数据预处理:")
        print("   python scripts/prepare_ligands.py")
        print("\n2. 新的预处理会:")
        print("   - 严格验证原子数一致性")
        print("   - 自动删除不一致的SDF")
        print("   - 保存元数据标记")
        print("\n3. 不一致的样本会降级为'纯坐标模式'")
        print("   （无HBD/HBA信息，但仍可训练）")
    else:
        print("✅ 所有数据一致！无需重新处理。")
    
    print("="*80)


def main():
    """主函数"""
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    
    verify_consistency(base_dir)


if __name__ == "__main__":
    main()
