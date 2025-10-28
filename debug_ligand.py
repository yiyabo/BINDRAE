"""
调试配体加载和特征检测
"""

import numpy as np
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, ChemicalFeatures
    RDKIT_AVAILABLE = True
except ImportError as e:
    print(f"❌ RDKit 导入失败: {e}")
    exit(1)

def debug_ligand():
    """调试配体加载"""
    
    base_dir = Path(__file__).parent
    pdb_id = "1a30"
    
    # 文件路径
    ligand_sdf = base_dir / "data" / "casf2016" / "complexes" / pdb_id / "ligand.sdf"
    ligand_coords = base_dir / "data" / "casf2016" / "processed" / "features" / f"{pdb_id}_ligand_coords.npy"
    
    print("="*80)
    print("配体加载调试")
    print("="*80)
    print(f"SDF 文件: {ligand_sdf}")
    print(f"存在: {ligand_sdf.exists()}")
    print()
    
    # 1. 加载 SDF
    print("[1] 加载 SDF 文件...")
    supplier = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False, sanitize=False)
    mol = supplier[0]
    
    if mol is None:
        print("❌ 分子加载失败！")
        return
    
    print(f"✓ 分子加载成功")
    print(f"  - 原子数（含H）: {mol.GetNumAtoms()}")
    print()
    
    # 2. 去除氢原子
    print("[2] 去除氢原子...")
    mol = Chem.RemoveHs(mol, sanitize=False)
    print(f"✓ 重原子数: {mol.GetNumAtoms()}")
    print()
    
    # 3. 尝试标准化
    print("[3] 标准化分子...")
    try:
        Chem.SanitizeMol(mol)
        print("✓ 标准化成功")
    except Exception as e:
        print(f"⚠️  标准化失败: {e}")
        print("  继续使用未标准化的分子")
    print()
    
    # 4. 检查元素类型
    print("[4] 元素类型:")
    elements = {}
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        elements[elem] = elements.get(elem, 0) + 1
        if elements[elem] <= 3:  # 只打印前3个
            print(f"  原子 {atom.GetIdx()}: {elem}, 芳香={atom.GetIsAromatic()}, 电荷={atom.GetFormalCharge()}")
    print(f"\n  元素统计: {elements}")
    print()
    
    # 5. 检测 HBD/HBA
    print("[5] 检测 HBD/HBA...")
    try:
        from rdkit import RDConfig
        import os
        fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        features = factory.GetFeaturesForMol(mol)
        
        hbd = []
        hba = []
        for feat in features:
            if feat.GetFamily() == 'Donor':
                hbd.extend(feat.GetAtomIds())
            elif feat.GetFamily() == 'Acceptor':
                hba.extend(feat.GetAtomIds())
        
        print(f"  HBD 原子: {hbd}")
        print(f"  HBA 原子: {hba}")
        
        if len(hbd) == 0 and len(hba) == 0:
            print("  ⚠️  未检测到任何 HBD/HBA！")
    except Exception as e:
        print(f"  ❌ Feature 检测失败: {e}")
    print()
    
    # 6. 检查邻近关系
    print("[6] 检查邻近关系（前3个原子）...")
    coords_array = np.load(ligand_coords)
    for i in range(min(3, mol.GetNumAtoms())):
        atom = mol.GetAtomWithIdx(i)
        neighbors = list(atom.GetNeighbors())
        print(f"  原子 {i} ({atom.GetSymbol()}): {len(neighbors)} 个邻居 → {[n.GetIdx() for n in neighbors]}")
    print()
    
    # 7. 测试探针生成
    print("[7] 测试探针生成...")
    if len(hbd) > 0 or len(hba) > 0:
        test_atom_idx = (hbd + hba)[0]
        atom = mol.GetAtomWithIdx(test_atom_idx)
        neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors()]
        neighbor_coords = [coords_array[n.GetIdx()] for n in neighbors if n.GetIdx() < len(coords_array)]
        
        print(f"  测试原子 {test_atom_idx} ({atom.GetSymbol()})")
        print(f"  邻居数: {len(neighbor_coords)}")
        
        if len(neighbor_coords) > 0:
            print(f"  ✓ 应该能生成探针")
        else:
            print(f"  ⚠️  无邻居坐标，无法生成探针")
    else:
        print("  ⚠️  无 HBD/HBA，不生成探针")
    
    print()
    print("="*80)
    print("调试完成")
    print("="*80)


if __name__ == "__main__":
    debug_ligand()
