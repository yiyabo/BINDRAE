#!/usr/bin/env python3
"""
提取蛋白质结合口袋 soft mask
- 基于配体重原子坐标计算残基-配体距离
- 5Å 接触半径 + 图膨胀 k=1
- RBF 核衰减 (σ=2Å)
- 归一化到 [0,1]
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, List, Set, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    from Bio.PDB import PDBParser, Selection
    from Bio.PDB.Polypeptide import is_aa
    BIOPYTHON_AVAILABLE = True
except ImportError:
    print("❌ 错误: BioPython 未安装，请先安装: conda install -c conda-forge biopython")
    sys.exit(1)


class PocketExtractor:
    """结合口袋提取器"""
    
    # 金属离子列表（需要保留）
    METAL_IONS = {'ZN', 'MG', 'CA', 'FE', 'MN', 'CU', 'NI', 'CO', 'CD', 'NA', 'K'}
    
    # 水分子（忽略）
    WATER_RESIDUES = {'HOH', 'WAT', 'H2O', 'DOD', 'D2O'}
    
    def __init__(self, base_dir: str, contact_radius: float = 5.0,
                 k_hops: int = 1, rbf_sigma: float = 2.0):
        """
        Args:
            base_dir: 项目根目录
            contact_radius: 接触半径 (Å)
            k_hops: 图膨胀层数
            rbf_sigma: RBF 核标准差 (Å)
        """
        self.base_dir = Path(base_dir)
        self.complexes_dir = self.base_dir / "data" / "casf2016" / "complexes"
        self.processed_dir = self.base_dir / "data" / "casf2016" / "processed"
        self.features_dir = self.processed_dir / "features"
        self.pockets_dir = self.processed_dir / "pockets"

        self.pockets_dir.mkdir(parents=True, exist_ok=True)
        
        # 参数
        self.contact_radius = contact_radius
        self.k_hops = k_hops
        self.rbf_sigma = rbf_sigma
        
        # 统计
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'failed_ids': [],
            'avg_pocket_size': [],
            'avg_weight': []
        }
        
        # PDB 解析器
        self.parser = PDBParser(QUIET=True)
    
    def get_residue_heavy_atoms(self, residue) -> np.ndarray:
        """
        获取残基的所有重原子坐标
        
        Args:
            residue: Bio.PDB Residue 对象
            
        Returns:
            重原子坐标数组 (N_atoms, 3)
        """
        coords = []
        for atom in residue:
            if atom.element != 'H':  # 排除氢原子
                coords.append(atom.coord)
        
        return np.array(coords) if coords else np.array([]).reshape(0, 3)
    
    def compute_min_distance(self, res_coords: np.ndarray, 
                            lig_coords: np.ndarray) -> float:
        """
        计算残基与配体的最小重原子距离
        
        Args:
            res_coords: 残基重原子坐标 (M, 3)
            lig_coords: 配体重原子坐标 (N, 3)
            
        Returns:
            最小距离 (Å)
        """
        if len(res_coords) == 0 or len(lig_coords) == 0:
            return float('inf')
        
        # 计算所有原子对之间的距离
        # 使用广播: (M, 1, 3) - (1, N, 3) = (M, N, 3)
        diff = res_coords[:, None, :] - lig_coords[None, :, :]
        distances = np.linalg.norm(diff, axis=2)  # (M, N)
        
        # 返回最小距离
        return np.min(distances)
    
    def get_residue_neighbors(self, residues: List, residue_info: List[Dict], 
                              contact_set: Set[int]) -> Set[int]:
        """
        获取接触残基的序列邻居（图膨胀 k=1）
        
        注意：基于实际的 chain + resid 判断邻接关系，而非简单的索引
        
        Args:
            residues: 残基列表
            residue_info: 残基信息列表
            contact_set: 接触残基索引集合
            
        Returns:
            扩展后的残基索引集合
        """
        expanded_set = contact_set.copy()
        
        # 构建 (chain, resid) -> idx 映射
        resid_to_idx = {}
        for idx, info in enumerate(residue_info):
            chain = info['chain']
            resid = info['resid']
            resid_to_idx[(chain, resid)] = idx
        
        # 对每个接触残基，查找实际的序列邻居
        for idx in contact_set:
            info = residue_info[idx]
            chain = info['chain']
            resid = info['resid']
            
            # 查找 resid-1 和 resid+1 在同一条链上
            prev_key = (chain, resid - 1)
            next_key = (chain, resid + 1)
            
            if prev_key in resid_to_idx:
                expanded_set.add(resid_to_idx[prev_key])
            if next_key in resid_to_idx:
                expanded_set.add(resid_to_idx[next_key])
        
        return expanded_set
    
    def rbf_kernel(self, distance: float) -> float:
        """
        RBF 核函数
        
        Args:
            distance: 距离 (Å)
            
        Returns:
            权重值 [0, 1]
        """
        return np.exp(-0.5 * (distance / self.rbf_sigma) ** 2)
    
    def extract_pocket(self, pdb_id: str) -> bool:
        """
        提取单个复合物的口袋 soft mask
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            是否成功
        """
        self.stats['total'] += 1
        
        # 输入路径
        protein_pdb = self.complexes_dir / pdb_id / "protein.pdb"
        ligand_coords_file = self.features_dir / f"{pdb_id}_ligand_coords.npy"
        
        # 检查文件存在
        if not protein_pdb.exists():
            print(f"  ⚠️  {pdb_id}: 蛋白质文件不存在")
            self.stats['failed'] += 1
            self.stats['failed_ids'].append(pdb_id)
            return False
        
        if not ligand_coords_file.exists():
            print(f"  ⚠️  {pdb_id}: 配体坐标文件不存在 (请先运行 prepare_ligands.py)")
            self.stats['failed'] += 1
            self.stats['failed_ids'].append(pdb_id)
            return False
        
        try:
            # 读取蛋白质结构
            structure = self.parser.get_structure(pdb_id, str(protein_pdb))
            
            # 读取配体重原子坐标
            ligand_coords = np.load(ligand_coords_file)
            
            # 收集所有残基（标准氨基酸 + 金属离子，排除水）
            residues = []
            residue_info = []  # 用于记录残基信息
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        resname = residue.get_resname().strip()
                        
                        # 忽略水分子
                        if resname in self.WATER_RESIDUES:
                            continue
                        
                        # 保留标准氨基酸和金属离子
                        if is_aa(residue, standard=True) or resname in self.METAL_IONS:
                            residues.append(residue)
                            residue_info.append({
                                'chain': chain.id,
                                'resname': resname,
                                'resid': residue.id[1],
                                'type': 'metal' if resname in self.METAL_IONS else 'amino_acid'
                            })
            
            if len(residues) == 0:
                print(f"  ❌ {pdb_id}: 无有效残基")
                self.stats['failed'] += 1
                self.stats['failed_ids'].append(pdb_id)
                return False
            
            # 计算每个残基与配体的最小距离
            distances = np.zeros(len(residues), dtype=np.float32)
            
            for i, residue in enumerate(residues):
                res_coords = self.get_residue_heavy_atoms(residue)
                distances[i] = self.compute_min_distance(res_coords, ligand_coords)
            
            # Step 1: 识别接触残基 (距离 <= contact_radius)
            contact_mask = distances <= self.contact_radius
            contact_indices = set(np.where(contact_mask)[0].tolist())
            
            if len(contact_indices) == 0:
                print(f"  ⚠️  {pdb_id}: 无接触残基 (可能配体坐标有问题)")
                # 不算失败，继续处理
            
            # Step 2: 图膨胀 k 跳（基于实际 chain+resid）
            expanded_indices = contact_indices.copy()
            for _ in range(self.k_hops):
                expanded_indices = self.get_residue_neighbors(residues, residue_info, expanded_indices)
            
            # Step 3: 计算 soft weights (RBF 衰减)
            weights = np.zeros(len(residues), dtype=np.float32)
            
            for i in expanded_indices:
                # 对膨胀后的残基应用 RBF 核
                weights[i] = self.rbf_kernel(distances[i])
            
            # Step 4: 归一化到 [0, 1]
            if weights.max() > 0:
                weights = weights / weights.max()
            
            # 创建硬掩码 (用于快速筛选)
            pocket_mask = np.zeros(len(residues), dtype=bool)
            pocket_mask[list(expanded_indices)] = True
            
            # 保存结果
            output_prefix = self.pockets_dir / pdb_id
            
            # 保存权重
            np.save(f"{output_prefix}_w_res.npy", weights)
            
            # 保存硬掩码
            np.save(f"{output_prefix}_pocket_mask.npy", pocket_mask)
            
            # 保存残基信息 (用于后续分析)
            np.save(f"{output_prefix}_residue_info.npy", residue_info)
            
            # 保存距离 (用于调试)
            np.save(f"{output_prefix}_distances.npy", distances)
            
            # 统计
            self.stats['success'] += 1
            self.stats['avg_pocket_size'].append(pocket_mask.sum())
            self.stats['avg_weight'].append(weights[weights > 0].mean() if weights.sum() > 0 else 0)
            
            if self.stats['total'] % 50 == 0:
                print(f"  进度: {self.stats['total']} 处理完成 "
                      f"(成功: {self.stats['success']}, 失败: {self.stats['failed']})")
            
            return True
            
        except Exception as e:
            print(f"  ❌ {pdb_id}: 处理失败 - {str(e)}")
            self.stats['failed'] += 1
            self.stats['failed_ids'].append(pdb_id)
            return False
    
    def run(self):
        """运行批量提取"""
        print("="*80)
        print("结合口袋 Soft Mask 提取")
        print("="*80)
        print(f"参数配置:")
        print(f"  - 接触半径 (r_c):    {self.contact_radius} Å")
        print(f"  - 图膨胀层数 (k):    {self.k_hops}")
        print(f"  - RBF 标准差 (σ):    {self.rbf_sigma} Å")
        print(f"  - 距离计算方式:      任意重原子最近距离")
        print(f"  - 保留金属离子:      是")
        print(f"  - 忽略水分子:        是")
        print()
        print(f"输入目录: {self.complexes_dir}")
        print(f"配体特征: {self.features_dir}")
        print(f"输出目录: {self.pockets_dir}")
        print()
        
        # 获取所有复合物
        complex_dirs = sorted([d for d in self.complexes_dir.iterdir() if d.is_dir()])
        print(f"发现 {len(complex_dirs)} 个复合物")
        print()
        
        # 处理每个复合物
        print("开始提取口袋...")
        for complex_dir in complex_dirs:
            pdb_id = complex_dir.name
            self.extract_pocket(pdb_id)
        
        # 打印统计
        print("\n" + "="*80)
        print("提取完成统计")
        print("="*80)
        print(f"总数:     {self.stats['total']}")
        print(f"成功:     {self.stats['success']} "
              f"({100*self.stats['success']/max(1,self.stats['total']):.1f}%)")
        print(f"失败:     {self.stats['failed']}")
        
        if self.stats['avg_pocket_size']:
            print(f"\n口袋统计:")
            print(f"  - 平均口袋大小: {np.mean(self.stats['avg_pocket_size']):.1f} 个残基")
            print(f"  - 口袋大小范围: [{np.min(self.stats['avg_pocket_size'])}, "
                  f"{np.max(self.stats['avg_pocket_size'])}]")
            print(f"  - 平均权重值:   {np.mean(self.stats['avg_weight']):.3f}")
        
        if self.stats['failed_ids']:
            print(f"\n失败的条目: {', '.join(self.stats['failed_ids'][:10])}")
            if len(self.stats['failed_ids']) > 10:
                print(f"  ... 还有 {len(self.stats['failed_ids'])-10} 个")
        
        print("\n✓ 口袋提取完成！")
        print(f"\n输出文件:")
        print(f"  - <PDBID>_w_res.npy          # 残基权重 (N_res,) 值: [0,1]")
        print(f"  - <PDBID>_pocket_mask.npy    # 口袋掩码 (N_res,) 值: True/False")
        print(f"  - <PDBID>_residue_info.npy   # 残基信息 (链、名称、编号)")
        print(f"  - <PDBID>_distances.npy      # 残基-配体距离 (N_res,)")


def main():
    """主函数"""
    # 使用项目根目录
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent
    
    # 允许命令行指定
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    
    # 可以通过命令行参数调整超参数
    contact_radius = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    k_hops = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    rbf_sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    
    extractor = PocketExtractor(str(base_dir), contact_radius, k_hops, rbf_sigma)
    extractor.run()


if __name__ == "__main__":
    main()
