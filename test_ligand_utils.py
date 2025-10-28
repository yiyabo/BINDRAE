"""
测试 ligand_utils 模块
"""

import numpy as np
from pathlib import Path
from utils.ligand_utils import build_ligand_tokens_from_file

def test_ligand_tokens():
    """测试配体 token 构建"""
    
    # 测试数据路径
    base_dir = Path(__file__).parent
    features_dir = base_dir / "data" / "casf2016" / "processed" / "features"
    complexes_dir = base_dir / "data" / "casf2016" / "complexes"
    
    # 选择一个样本测试
    pdb_id = "1a30"
    ligand_coords_file = features_dir / f"{pdb_id}_ligand_coords.npy"
    ligand_sdf_file = complexes_dir / pdb_id / "ligand.sdf"
    
    if not ligand_coords_file.exists():
        print(f"⚠️  测试数据不存在: {ligand_coords_file}")
        print("请先运行数据准备脚本")
        return
    
    print("="*80)
    print("测试配体 Token 构建")
    print("="*80)
    print(f"样本: {pdb_id}")
    print()
    
    # 构建 tokens
    tokens = build_ligand_tokens_from_file(
        ligand_coords_file,
        ligand_sdf_file if ligand_sdf_file.exists() else None
    )
    
    # 打印结果
    print(f"✓ Token 数量: {len(tokens['coords'])}")
    print(f"  - 重原子: {(~tokens['is_probe']).sum()}")
    print(f"  - 探针: {tokens['is_probe'].sum()}")
    print()
    
    print(f"✓ 坐标形状: {tokens['coords'].shape}")
    print(f"✓ 类型编码形状: {tokens['types'].shape}")
    print(f"✓ 重要性权重: min={tokens['importance'].min():.3f}, max={tokens['importance'].max():.3f}")
    print()
    
    # 统计原子类型
    type_counts = tokens['types'].sum(axis=0)
    type_names = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', '芳香', '正电', '负电']
    print("原子类型分布:")
    for name, count in zip(type_names, type_counts):
        if count > 0:
            print(f"  {name:4s}: {int(count):3d}")
    
    print()
    print("✅ 测试通过！")


if __name__ == "__main__":
    test_ligand_tokens()
