#!/usr/bin/env python3
"""
快速验证 CASF-2016 数据集的完整性和可读性
"""

import sys
from pathlib import Path
from collections import defaultdict

def verify_dataset(base_dir: str):
    """验证数据集"""
    base_path = Path(base_dir)
    complexes_dir = base_path / "data" / "casf2016" / "complexes"
    meta_dir = base_path / "data" / "casf2016" / "meta"
    
    print("="*80)
    print("CASF-2016 数据集完整性验证")
    print("="*80)
    
    # 检查目录存在
    if not complexes_dir.exists():
        print(f"❌ 错误: 复合物目录不存在: {complexes_dir}")
        return False
    
    # 读取索引文件
    index_file = meta_dir / "INDEX_core.txt"
    if not index_file.exists():
        print(f"❌ 错误: 索引文件不存在: {index_file}")
        return False
    
    indexed_ids = set()
    with open(index_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                pdb_id = line.split()[0]
                indexed_ids.add(pdb_id)
    
    print(f"✓ 索引文件包含 {len(indexed_ids)} 个条目")
    
    # 检查复合物目录
    complex_dirs = [d for d in complexes_dir.iterdir() if d.is_dir()]
    print(f"✓ 发现 {len(complex_dirs)} 个复合物目录")
    
    # 验证每个复合物
    stats = {
        'total': 0,
        'has_protein': 0,
        'has_ligand': 0,
        'has_pocket': 0,
        'complete': 0,
        'missing_protein': [],
        'missing_ligand': []
    }
    
    for complex_dir in sorted(complex_dirs):
        pdb_id = complex_dir.name
        stats['total'] += 1
        
        protein_file = complex_dir / "protein.pdb"
        ligand_file = complex_dir / "ligand.sdf"
        pocket_file = complex_dir / "pocket.pdb"
        
        has_protein = protein_file.exists() and protein_file.stat().st_size > 0
        has_ligand = ligand_file.exists() and ligand_file.stat().st_size > 0
        has_pocket = pocket_file.exists() and pocket_file.stat().st_size > 0
        
        if has_protein:
            stats['has_protein'] += 1
        else:
            stats['missing_protein'].append(pdb_id)
        
        if has_ligand:
            stats['has_ligand'] += 1
        else:
            stats['missing_ligand'].append(pdb_id)
        
        if has_pocket:
            stats['has_pocket'] += 1
        
        if has_protein and has_ligand:
            stats['complete'] += 1
    
    # 打印统计
    print("\n" + "-"*80)
    print("数据完整性统计:")
    print("-"*80)
    print(f"  总复合物数:       {stats['total']}")
    print(f"  包含蛋白质:       {stats['has_protein']} ({100*stats['has_protein']/stats['total']:.1f}%)")
    print(f"  包含配体:         {stats['has_ligand']} ({100*stats['has_ligand']/stats['total']:.1f}%)")
    print(f"  包含口袋:         {stats['has_pocket']} ({100*stats['has_pocket']/stats['total']:.1f}%)")
    print(f"  完整条目:         {stats['complete']} ({100*stats['complete']/stats['total']:.1f}%)")
    
    # 检查缺失
    if stats['missing_protein']:
        print(f"\n⚠️  缺失蛋白质的条目: {', '.join(stats['missing_protein'])}")
    
    if stats['missing_ligand']:
        print(f"\n⚠️  缺失配体的条目: {', '.join(stats['missing_ligand'])}")
    
    # 验证索引一致性
    dir_ids = {d.name for d in complex_dirs}
    only_in_index = indexed_ids - dir_ids
    only_in_dirs = dir_ids - indexed_ids
    
    print("\n" + "-"*80)
    print("索引一致性检查:")
    print("-"*80)
    
    if only_in_index:
        print(f"⚠️  索引中存在但目录缺失: {', '.join(sorted(only_in_index))}")
    
    if only_in_dirs:
        print(f"⚠️  目录存在但索引缺失: {', '.join(sorted(only_in_dirs))}")
    
    if not only_in_index and not only_in_dirs:
        print("✓ 索引与目录完全一致")
    
    # 读取过滤记录
    filtered_file = meta_dir / "filtered.csv"
    if filtered_file.exists():
        with open(filtered_file, 'r') as f:
            lines = f.readlines()
            filtered_count = len(lines) - 1  # 减去标题行
        print(f"\n✓ 过滤记录: {filtered_count} 个条目")
    
    # 最终结论
    print("\n" + "="*80)
    if stats['complete'] == stats['total'] and not only_in_index and not only_in_dirs:
        print("✅ 数据集验证通过！所有复合物均完整且一致。")
        return True
    else:
        print("⚠️  数据集存在部分问题，请检查上述警告。")
        return False


if __name__ == "__main__":
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/apple/code/BINDRAE"
    verify_dataset(base_dir)
