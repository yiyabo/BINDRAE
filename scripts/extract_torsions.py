#!/usr/bin/env python3
"""
提取蛋白质主链和侧链扭转角
- 直接从原子坐标计算四点二面角（IUPAC定义）
- 输出 sin/cos 或 wrap 到 [-π,π]
- 处理缺失原子、Pro/Gly、altLoc
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
    from Bio.PDB.vectors import calc_dihedral, Vector
    from Bio.PDB.Residue import Residue
except ImportError as e:
    print(f"❌ 错误: BioPython 导入失败 - {e}")
    print("请安装: pip install biopython")
    sys.exit(1)


# 侧链扭转角定义（IUPAC标准）
CHI_ANGLES_ATOMS = {
    'ALA': [],
    'CYS': [['N', 'CA', 'CB', 'SG']],
    'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']],
    'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'GLY': [],
    'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
    'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], 
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'], ['CB', 'CG', 'SD', 'CE']],
    'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']],
    'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], 
            ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    'SER': [['N', 'CA', 'CB', 'OG']],
    'THR': [['N', 'CA', 'CB', 'OG1']],
    'VAL': [['N', 'CA', 'CB', 'CG1']],
    'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
}


class TorsionExtractor:
    """扭转角提取器"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.complexes_dir = self.base_dir / "data" / "casf2016" / "complexes"
        self.processed_dir = self.base_dir / "data" / "casf2016" / "processed"
        self.features_dir = self.processed_dir / "features"

        self.parser = PDBParser(QUIET=True)
        
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'bb_coverage': [],
            'chi_coverage': [],
            'failed_ids': []
        }
    
    def get_atom_coord(self, residue, atom_name: str) -> Optional[np.ndarray]:
        """
        获取原子坐标，处理 altLoc 并验证坐标有效性
        
        Args:
            residue: Bio.PDB Residue 对象
            atom_name: 原子名称
            
        Returns:
            坐标或 None（如果原子缺失或坐标无效）
        """
        if atom_name not in residue:
            return None
        
        atom = residue[atom_name]
        
        # 处理 altLoc：选择最高占据率
        if atom.is_disordered():
            selected_atom = atom.selected_child
            coord = selected_atom.coord
        else:
            coord = atom.coord
        
        # ✅ 验证坐标有效性
        if coord is None:
            return None
        
        # 转换为 numpy 数组以便检查
        coord_array = np.array(coord, dtype=np.float32)
        
        # 检查是否是占位符坐标 (0, 0, 0)
        # 注意：真实蛋白质不太可能所有原子都在原点
        if np.allclose(coord_array, [0, 0, 0], atol=1e-6):
            return None
        
        # 检查是否包含 NaN 或 inf
        if not np.all(np.isfinite(coord_array)):
            return None
        
        return coord_array
    
    def calc_dihedral_angle(self, residue: Residue, atom_names: List[str], 
                           prev_residue: Optional[Residue] = None, 
                           next_residue: Optional[Residue] = None) -> Optional[float]:
        """
        计算四点二面角
        
        Args:
            residue: 当前残基
            atom_names: 4个原子名称
            prev_residue: 前一个残基（用于 phi）
            next_residue: 后一个残基（用于 psi）
            
        Returns:
            角度（弧度）或 None
        """
        coords = []
        
        for atom_name in atom_names:
            # 处理跨残基的原子
            if atom_name == '-C' and prev_residue is not None:
                coord = self.get_atom_coord(prev_residue, 'C')
            elif atom_name == '+N' and next_residue is not None:
                coord = self.get_atom_coord(next_residue, 'N')
            else:
                coord = self.get_atom_coord(residue, atom_name)
            
            if coord is None:
                return None
            coords.append(coord)
        
        # 使用 BioPython 计算二面角
        try:
            v1, v2, v3, v4 = [Vector(c) for c in coords]
            angle = calc_dihedral(v1, v2, v3, v4)
            return float(angle)
        except (KeyError, ValueError, TypeError):
            # KeyError: 原子不存在
            # ValueError: 向量计算错误
            # TypeError: 坐标类型错误
            return None
    
    @staticmethod
    def is_sequential_neighbor(res1: Optional[Residue], res2: Optional[Residue]) -> bool:
        """
        检查两个残基是否是序列邻居（同链且resid相差1）
        
        Args:
            res1: 第一个残基
            res2: 第二个残基
            
        Returns:
            是否为序列邻居
        """
        if res1 is None or res2 is None:
            return False
        
        # 检查是否在同一条链
        if res1.get_parent().id != res2.get_parent().id:
            return False
        
        # 检查resid是否相差1（考虑插入码）
        resid1 = res1.id[1]
        resid2 = res2.id[1]
        
        return abs(resid1 - resid2) == 1
    
    def extract_backbone_angles(self, residues: List) -> Dict:
        """
        提取主链扭转角 φ, ψ, ω
        
        注意：只计算真正序列连续的残基间的扭转角
        
        Args:
            residues: 残基列表
            
        Returns:
            角度字典
        """
        n_res = len(residues)
        
        phi = np.zeros(n_res, dtype=np.float32)
        psi = np.zeros(n_res, dtype=np.float32)
        omega = np.zeros(n_res, dtype=np.float32)
        
        bb_mask = np.zeros(n_res, dtype=bool)
        omega_cis_trans = np.zeros(n_res, dtype=np.int8)  # 0: trans, 1: cis
        
        # 添加cis/trans阈值常量
        CIS_TRANS_THRESHOLD = np.pi / 6  # 30度
        
        for i, residue in enumerate(residues):
            # 获取前后残基，并验证是否序列连续
            prev_res = residues[i-1] if i > 0 else None
            next_res = residues[i+1] if i < n_res - 1 else None
            
            # ✅ 验证序列连续性
            if prev_res and not self.is_sequential_neighbor(prev_res, residue):
                prev_res = None  # 不连续，不计算φ
            
            if next_res and not self.is_sequential_neighbor(residue, next_res):
                next_res = None  # 不连续，不计算ψ/ω
            
            # φ (phi): -C -- N - CA - C
            if prev_res is not None:
                phi_angle = self.calc_dihedral_angle(residue, ['-C', 'N', 'CA', 'C'], 
                                                     prev_residue=prev_res)
                if phi_angle is not None:
                    phi[i] = phi_angle
                    bb_mask[i] = True
            
            # ψ (psi): N - CA - C -- +N
            if next_res is not None:
                psi_angle = self.calc_dihedral_angle(residue, ['N', 'CA', 'C', '+N'],
                                                     next_residue=next_res)
                if psi_angle is not None:
                    psi[i] = psi_angle
                    if not bb_mask[i]:  # 如果 phi 失败但 psi 成功，也标记为有效
                        bb_mask[i] = True
            
            # ω (omega): CA - C -- +N - +CA
            if next_res is not None:
                omega_angle = self.calc_dihedral_angle(residue, ['CA', 'C', '+N', '+CA'],
                                                       next_residue=next_res)
                if omega_angle is not None:
                    omega[i] = omega_angle
                    # 判断 cis/trans
                    if abs(omega_angle) < CIS_TRANS_THRESHOLD:
                        omega_cis_trans[i] = 1  # cis
                    else:
                        omega_cis_trans[i] = 0  # trans
        
        return {
            'phi': phi,
            'psi': psi,
            'omega': omega,
            'bb_mask': bb_mask,
            'omega_cis_trans': omega_cis_trans
        }
    
    def extract_sidechain_angles(self, residues: List) -> Dict:
        """
        提取侧链扭转角 χ1, χ2, χ3, χ4
        
        Args:
            residues: 残基列表
            
        Returns:
            角度字典
        """
        n_res = len(residues)
        max_chi = 4
        
        chi = np.zeros((n_res, max_chi), dtype=np.float32)
        chi_mask = np.zeros((n_res, max_chi), dtype=bool)
        
        for i, residue in enumerate(residues):
            resname = residue.get_resname().strip()
            
            # 获取该残基的 χ 定义
            if resname not in CHI_ANGLES_ATOMS:
                # 非标准氨基酸，跳过
                continue
            
            chi_defs = CHI_ANGLES_ATOMS[resname]
            
            for j, atom_names in enumerate(chi_defs):
                if j >= max_chi:
                    break
                
                chi_angle = self.calc_dihedral_angle(residue, atom_names)
                
                if chi_angle is not None:
                    chi[i, j] = chi_angle
                    chi_mask[i, j] = True
        
        return {
            'chi': chi,
            'chi_mask': chi_mask
        }
    
    def extract_torsions(self, pdb_id: str) -> bool:
        """
        提取单个复合物的扭转角
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            是否成功
        """
        self.stats['total'] += 1
        
        protein_pdb = self.complexes_dir / pdb_id / "protein.pdb"
        
        if not protein_pdb.exists():
            self.stats['failed'] += 1
            self.stats['failed_ids'].append(pdb_id)
            return False
        
        try:
            # 读取结构
            structure = self.parser.get_structure(pdb_id, str(protein_pdb))
            
            # 收集所有标准氨基酸残基
            residues = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if is_aa(residue, standard=True):
                            residues.append(residue)
            
            if len(residues) == 0:
                self.stats['failed'] += 1
                self.stats['failed_ids'].append(pdb_id)
                return False
            
            # 提取主链角
            bb_angles = self.extract_backbone_angles(residues)
            
            # 提取侧链角
            sc_angles = self.extract_sidechain_angles(residues)
            
            # 计算覆盖率
            bb_coverage = bb_angles['bb_mask'].mean()
            chi_coverage = sc_angles['chi_mask'].sum() / (len(residues) * 4)
            
            self.stats['bb_coverage'].append(bb_coverage)
            self.stats['chi_coverage'].append(chi_coverage)
            
            # 保存结果
            output_file = self.features_dir / f"{pdb_id}_torsions.npz"
            
            np.savez_compressed(
                output_file,
                phi=bb_angles['phi'],
                psi=bb_angles['psi'],
                omega=bb_angles['omega'],
                chi=sc_angles['chi'],
                bb_mask=bb_angles['bb_mask'],
                chi_mask=sc_angles['chi_mask'],
                omega_cis_trans=bb_angles['omega_cis_trans'],
                n_residues=len(residues)
            )
            
            self.stats['success'] += 1
            
            if self.stats['total'] % 50 == 0:
                avg_bb = np.mean(self.stats['bb_coverage']) if self.stats['bb_coverage'] else 0
                avg_chi = np.mean(self.stats['chi_coverage']) if self.stats['chi_coverage'] else 0
                print(f"  进度: {self.stats['total']} | "
                      f"主链覆盖: {100*avg_bb:.1f}% | "
                      f"侧链覆盖: {100*avg_chi:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"  ❌ {pdb_id}: {str(e)}")
            self.stats['failed'] += 1
            self.stats['failed_ids'].append(pdb_id)
            return False
    
    def run(self):
        """运行批量提取"""
        print("="*80)
        print("蛋白质扭转角提取")
        print("="*80)
        print(f"输入目录: {self.complexes_dir}")
        print(f"输出目录: {self.features_dir}")
        print()
        
        # 获取所有复合物
        complex_dirs = sorted([d for d in self.complexes_dir.iterdir() if d.is_dir()])
        print(f"发现 {len(complex_dirs)} 个复合物")
        print()
        
        # 处理每个复合物
        print("开始提取扭转角...")
        for complex_dir in complex_dirs:
            pdb_id = complex_dir.name
            self.extract_torsions(pdb_id)
        
        # 打印统计
        print("\n" + "="*80)
        print("提取完成统计")
        print("="*80)
        print(f"总数:     {self.stats['total']}")
        print(f"成功:     {self.stats['success']} "
              f"({100*self.stats['success']/max(1,self.stats['total']):.1f}%)")
        print(f"失败:     {self.stats['failed']}")
        
        if self.stats['bb_coverage']:
            avg_bb = np.mean(self.stats['bb_coverage'])
            avg_chi = np.mean(self.stats['chi_coverage'])
            print(f"\n覆盖率统计:")
            print(f"  - 主链角度 (φ/ψ/ω): {100*avg_bb:.2f}%")
            print(f"  - 侧链角度 (χ):    {100*avg_chi:.2f}%")
            
            # 验收检查
            if avg_bb >= 0.98:
                print(f"\n✅ 主链覆盖率达标 (≥98%)")
            else:
                print(f"\n⚠️  主链覆盖率未达标 ({100*avg_bb:.2f}% < 98%)")
        
        if self.stats['failed_ids']:
            print(f"\n失败的条目: {', '.join(self.stats['failed_ids'][:10])}")
            if len(self.stats['failed_ids']) > 10:
                print(f"  ... 还有 {len(self.stats['failed_ids'])-10} 个")
        
        print("\n✓ 扭转角提取完成！")
        print(f"\n输出文件: <PDBID>_torsions.npz")
        print(f"  - phi, psi, omega: (N_res,)")
        print(f"  - chi: (N_res, 4)")
        print(f"  - bb_mask, chi_mask: boolean arrays")
        print(f"  - omega_cis_trans: 0=trans, 1=cis")


def main():
    """主函数"""
    # 使用项目根目录
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    
    # 允许命令行指定
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    
    extractor = TorsionExtractor(str(base_dir))
    extractor.run()


if __name__ == "__main__":
    main()
