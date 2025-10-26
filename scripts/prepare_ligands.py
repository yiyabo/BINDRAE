#!/usr/bin/env python3
"""
配体规范化脚本
- 去除氢原子
- 标准化价态和立体化学
- 提取重原子坐标
- 生成配体特征
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("❌ 错误: RDKit 未安装，请先安装: conda install -c conda-forge rdkit")
    sys.exit(1)


class LigandProcessor:
    """配体处理器"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.complexes_dir = self.base_dir / "data" / "casf2016" / "complexes"
        self.processed_dir = self.base_dir / "data" / "casf2016" / "processed"
        self.features_dir = self.processed_dir / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'failed_ids': []
        }
    
    def normalize_ligand(self, sdf_path: Path) -> Optional[Chem.Mol]:
        """
        规范化配体分子
        
        Args:
            sdf_path: SDF 文件路径
            
        Returns:
            规范化后的分子对象，失败返回 None
        """
        try:
            # 读取 SDF - 使用宽松模式
            supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
            mol = supplier[0]
            
            if mol is None:
                return None
            
            # 去除氢原子
            mol = Chem.RemoveHs(mol, sanitize=False)
            
            # 尝试标准化价态（使用部分标准化，容忍错误）
            try:
                # 使用部分标准化，只做必要的检查
                Chem.SanitizeMol(mol, 
                                sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
            except:
                # 如果完全标准化失败，尝试最小标准化
                try:
                    Chem.SanitizeMol(mol, 
                                    sanitizeOps=Chem.SANITIZE_FINDRADICALS |
                                                Chem.SANITIZE_SETAROMATICITY |
                                                Chem.SANITIZE_SETCONJUGATION,
                                    catchErrors=True)
                except:
                    # 即使标准化失败，仍然保留坐标
                    pass
            
            # 尝试分配立体化学（可选，失败不影响使用）
            try:
                Chem.AssignStereochemistry(mol, 
                                          cleanIt=True, 
                                          force=False, 
                                          flagPossibleStereoCenters=True)
            except:
                pass
            
            # 确保分子至少有坐标
            if mol.GetNumConformers() == 0:
                return None
            
            return mol
            
        except Exception as e:
            # 静默失败，只记录真正的错误
            return None
    
    def extract_heavy_atoms_coords(self, mol: Chem.Mol) -> np.ndarray:
        """
        提取重原子坐标
        
        Args:
            mol: RDKit 分子对象
            
        Returns:
            重原子坐标数组，形状 (N_atoms, 3)
        """
        conformer = mol.GetConformer()
        coords = []
        
        for atom in mol.GetAtoms():
            # 只保留重原子（非氢）
            if atom.GetAtomicNum() > 1:  # 氢的原子序数是1
                pos = conformer.GetAtomPosition(atom.GetIdx())
                coords.append([pos.x, pos.y, pos.z])
        
        return np.array(coords, dtype=np.float32)
    
    def get_ligand_properties(self, mol: Chem.Mol) -> dict:
        """
        计算配体属性
        
        Args:
            mol: RDKit 分子对象
            
        Returns:
            属性字典
        """
        props = {
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            'num_atoms': mol.GetNumAtoms(),
            'mol_weight': Descriptors.MolWt(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_hbd': Descriptors.NumHDonors(mol),  # 氢键供体
            'num_hba': Descriptors.NumHAcceptors(mol),  # 氢键受体
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),  # 拓扑极性表面积
        }
        
        return props
    
    def process_complex(self, pdb_id: str) -> bool:
        """
        处理单个复合物的配体
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            是否成功
        """
        self.stats['total'] += 1
        
        # 输入路径
        ligand_sdf = self.complexes_dir / pdb_id / "ligand.sdf"
        
        if not ligand_sdf.exists():
            print(f"  ⚠️  {pdb_id}: 配体文件不存在")
            self.stats['failed'] += 1
            self.stats['failed_ids'].append(pdb_id)
            return False
        
        # 规范化配体
        mol = self.normalize_ligand(ligand_sdf)
        
        if mol is None:
            print(f"  ❌ {pdb_id}: 配体规范化失败")
            self.stats['failed'] += 1
            self.stats['failed_ids'].append(pdb_id)
            return False
        
        # 提取重原子坐标
        coords = self.extract_heavy_atoms_coords(mol)
        
        if len(coords) == 0:
            print(f"  ❌ {pdb_id}: 无重原子坐标")
            self.stats['failed'] += 1
            self.stats['failed_ids'].append(pdb_id)
            return False
        
        # 获取属性
        props = self.get_ligand_properties(mol)
        
        # 保存结果
        output_prefix = self.features_dir / pdb_id
        
        # 保存重原子坐标
        np.save(f"{output_prefix}_ligand_coords.npy", coords)
        
        # 保存属性
        np.save(f"{output_prefix}_ligand_props.npy", props)
        
        # 可选: 保存规范化后的 SDF
        writer = Chem.SDWriter(f"{output_prefix}_ligand_normalized.sdf")
        writer.write(mol)
        writer.close()
        
        self.stats['success'] += 1
        
        if self.stats['total'] % 50 == 0:
            print(f"  进度: {self.stats['total']} 处理完成 "
                  f"(成功: {self.stats['success']}, 失败: {self.stats['failed']})")
        
        return True
    
    def run(self):
        """运行批量处理"""
        print("="*80)
        print("配体规范化与特征提取")
        print("="*80)
        print(f"输入目录: {self.complexes_dir}")
        print(f"输出目录: {self.features_dir}")
        print()
        
        # 获取所有复合物
        complex_dirs = sorted([d for d in self.complexes_dir.iterdir() if d.is_dir()])
        print(f"发现 {len(complex_dirs)} 个复合物")
        print()
        
        # 处理每个复合物
        print("开始处理配体...")
        for complex_dir in complex_dirs:
            pdb_id = complex_dir.name
            self.process_complex(pdb_id)
        
        # 打印统计
        print("\n" + "="*80)
        print("处理完成统计")
        print("="*80)
        print(f"总数:   {self.stats['total']}")
        print(f"成功:   {self.stats['success']} "
              f"({100*self.stats['success']/max(1,self.stats['total']):.1f}%)")
        print(f"失败:   {self.stats['failed']}")
        
        if self.stats['failed_ids']:
            print(f"\n失败的条目: {', '.join(self.stats['failed_ids'][:10])}")
            if len(self.stats['failed_ids']) > 10:
                print(f"  ... 还有 {len(self.stats['failed_ids'])-10} 个")
        
        print("\n✓ 配体规范化完成！")
        print(f"\n输出文件:")
        print(f"  - <PDBID>_ligand_coords.npy      # 重原子坐标 (N_atoms, 3)")
        print(f"  - <PDBID>_ligand_props.npy       # 配体属性字典")
        print(f"  - <PDBID>_ligand_normalized.sdf  # 规范化后的SDF")


def main():
    """主函数"""
    # 使用项目根目录（脚本所在目录的上一级）
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    
    # 允许命令行指定
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    
    processor = LigandProcessor(str(base_dir))
    processor.run()


if __name__ == "__main__":
    main()
