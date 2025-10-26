#!/usr/bin/env python3
"""
CASF-2016 数据集整理与完整性校验脚本
组织目录结构并过滤异常样本
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
from collections import defaultdict
import warnings

# 尝试导入必要的库
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Will skip detailed ligand validation.")
    RDKIT_AVAILABLE = False

try:
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Polypeptide import is_aa
    BIOPYTHON_AVAILABLE = True
except ImportError:
    print("Warning: BioPython not available. Will skip detailed protein validation.")
    BIOPYTHON_AVAILABLE = False


class CASF2016Organizer:
    """CASF-2016 数据集整理器"""
    
    def __init__(self, raw_data_dir: str, output_dir: str):
        """
        Args:
            raw_data_dir: 原始 CASF-2016 解压目录
            output_dir: 输出目录根路径
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        
        # 定义目录结构
        self.complexes_dir = self.output_dir / "data" / "casf2016" / "complexes"
        self.meta_dir = self.output_dir / "data" / "casf2016" / "meta"
        self.processed_dir = self.output_dir / "data" / "casf2016" / "processed"
        self.pockets_dir = self.processed_dir / "pockets"
        self.features_dir = self.processed_dir / "features"
        self.splits_dir = self.processed_dir / "splits"
        
        # CoreSet 数据文件
        self.coreset_file = self.raw_data_dir / "power_scoring" / "CoreSet.dat"
        self.coreset_complexes = self.raw_data_dir / "coreset"
        
        # 统计信息
        self.stats = {
            'total': 0,
            'success': 0,
            'filtered': 0,
            'missing_protein': 0,
            'missing_ligand': 0,
            'high_resolution': 0,
            'metal_rich': 0,
            'parse_error': 0,
            'other_error': 0
        }
        
        # 过滤记录
        self.filtered_records = []
        
    def create_directories(self):
        """创建所有必要的目录"""
        dirs = [
            self.complexes_dir,
            self.meta_dir,
            self.pockets_dir,
            self.features_dir,
            self.splits_dir
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ 创建目录结构完成")
        print(f"  - 复合物目录: {self.complexes_dir}")
        print(f"  - 元数据目录: {self.meta_dir}")
        print(f"  - 处理数据目录: {self.processed_dir}")
    
    def parse_coreset_metadata(self) -> Dict[str, Dict]:
        """解析 CoreSet.dat 获取元数据"""
        metadata = {}
        
        if not self.coreset_file.exists():
            print(f"Warning: CoreSet.dat not found at {self.coreset_file}")
            return metadata
        
        with open(self.coreset_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 6:
                    pdb_id = parts[0]
                    metadata[pdb_id] = {
                        'resolution': float(parts[1]),
                        'year': int(parts[2]),
                        'logKa': float(parts[3]),
                        'affinity': parts[4],
                        'target': int(parts[5])
                    }
        
        print(f"✓ 解析 CoreSet 元数据: {len(metadata)} 个条目")
        return metadata
    
    def check_protein_validity(self, protein_path: Path) -> Tuple[bool, Optional[str], Dict]:
        """检查蛋白质文件有效性"""
        info = {
            'has_metal': False,
            'metal_count': 0,
            'residue_count': 0,
            'missing_residues': 0
        }
        
        if not protein_path.exists():
            return False, "protein file missing", info
        
        if not BIOPYTHON_AVAILABLE:
            # 简单检查文件是否为空
            if protein_path.stat().st_size < 100:
                return False, "protein file too small", info
            return True, None, info
        
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('protein', str(protein_path))
            
            metal_atoms = ['ZN', 'MG', 'CA', 'FE', 'MN', 'CU', 'NI', 'CO', 'CD']
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        # 统计残基
                        if is_aa(residue, standard=True):
                            info['residue_count'] += 1
                        
                        # 检测金属
                        resname = residue.get_resname().strip()
                        if resname in metal_atoms:
                            info['has_metal'] = True
                            info['metal_count'] += 1
            
            # 金属过多的过滤阈值
            if info['metal_count'] > 5:
                return False, f"too many metals ({info['metal_count']})", info
            
            # 残基数太少可能有问题
            if info['residue_count'] < 10:
                return False, f"too few residues ({info['residue_count']})", info
            
            return True, None, info
            
        except Exception as e:
            return False, f"parse error: {str(e)}", info
    
    def check_ligand_validity(self, ligand_path: Path) -> Tuple[bool, Optional[str], Dict]:
        """检查配体文件有效性"""
        info = {
            'atom_count': 0,
            'mol_weight': 0.0,
            'format': ligand_path.suffix
        }
        
        if not ligand_path.exists():
            return False, "ligand file missing", info
        
        if not RDKIT_AVAILABLE:
            # 简单检查文件是否为空
            if ligand_path.stat().st_size < 50:
                return False, "ligand file too small", info
            return True, None, info
        
        try:
            # 尝试读取配体
            if ligand_path.suffix == '.sdf':
                mol = Chem.SDMolSupplier(str(ligand_path), removeHs=False)[0]
            elif ligand_path.suffix == '.mol2':
                mol = Chem.MolFromMol2File(str(ligand_path), removeHs=False)
            else:
                return False, f"unsupported format: {ligand_path.suffix}", info
            
            if mol is None:
                return False, "failed to parse ligand", info
            
            info['atom_count'] = mol.GetNumAtoms()
            info['mol_weight'] = Descriptors.MolWt(mol)
            
            # 基本有效性检查
            if info['atom_count'] < 3:
                return False, "too few atoms", info
            
            if info['mol_weight'] < 50:
                return False, "molecular weight too low", info
            
            return True, None, info
            
        except Exception as e:
            return False, f"parse error: {str(e)}", info
    
    def process_complex(self, pdb_id: str, metadata: Dict) -> bool:
        """处理单个复合物"""
        self.stats['total'] += 1
        
        # 源文件路径
        complex_dir = self.coreset_complexes / pdb_id
        protein_pdb = complex_dir / f"{pdb_id}_protein.pdb"
        ligand_sdf = complex_dir / f"{pdb_id}_ligand.sdf"
        pocket_pdb = complex_dir / f"{pdb_id}_pocket.pdb"
        
        # 目标目录
        target_dir = self.complexes_dir / pdb_id
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取分辨率
        resolution = metadata.get(pdb_id, {}).get('resolution', 999.0)
        
        # 检查分辨率
        if resolution > 3.0:
            self.stats['high_resolution'] += 1
            self.stats['filtered'] += 1
            self.filtered_records.append({
                'pdb_id': pdb_id,
                'reason': 'high_resolution',
                'resolution': resolution,
                'details': f'resolution {resolution} > 3.0 Å'
            })
            return False
        
        # 检查蛋白质
        protein_valid, protein_error, protein_info = self.check_protein_validity(protein_pdb)
        if not protein_valid:
            reason_key = 'missing_protein' if 'missing' in protein_error else 'parse_error'
            self.stats[reason_key] += 1
            self.stats['filtered'] += 1
            self.filtered_records.append({
                'pdb_id': pdb_id,
                'reason': reason_key,
                'resolution': resolution,
                'details': protein_error,
                **protein_info
            })
            return False
        
        # 检查配体
        ligand_valid, ligand_error, ligand_info = self.check_ligand_validity(ligand_sdf)
        if not ligand_valid:
            reason_key = 'missing_ligand' if 'missing' in ligand_error else 'parse_error'
            self.stats[reason_key] += 1
            self.stats['filtered'] += 1
            self.filtered_records.append({
                'pdb_id': pdb_id,
                'reason': reason_key,
                'resolution': resolution,
                'details': ligand_error,
                **ligand_info
            })
            return False
        
        # 金属过多检查
        if protein_info.get('metal_count', 0) > 5:
            self.stats['metal_rich'] += 1
            self.stats['filtered'] += 1
            self.filtered_records.append({
                'pdb_id': pdb_id,
                'reason': 'metal_rich',
                'resolution': resolution,
                'details': f"metal atoms: {protein_info['metal_count']}",
                **protein_info
            })
            return False
        
        # 复制文件到目标目录
        try:
            shutil.copy2(protein_pdb, target_dir / "protein.pdb")
            shutil.copy2(ligand_sdf, target_dir / "ligand.sdf")
            
            # 如果存在 pocket，也复制
            if pocket_pdb.exists():
                shutil.copy2(pocket_pdb, target_dir / "pocket.pdb")
            
            self.stats['success'] += 1
            return True
            
        except Exception as e:
            self.stats['other_error'] += 1
            self.stats['filtered'] += 1
            self.filtered_records.append({
                'pdb_id': pdb_id,
                'reason': 'copy_error',
                'resolution': resolution,
                'details': str(e)
            })
            return False
    
    def save_index_file(self, metadata: Dict, successful_ids: List[str]):
        """保存索引文件"""
        index_file = self.meta_dir / "INDEX_core.txt"
        
        with open(index_file, 'w') as f:
            f.write("# CASF-2016 Core Set Index\n")
            f.write("# Format: PDB_ID  Resolution  Year  logKa  Affinity  Target\n")
            f.write("#" + "="*80 + "\n")
            
            for pdb_id in sorted(successful_ids):
                meta = metadata.get(pdb_id, {})
                f.write(f"{pdb_id}\t"
                       f"{meta.get('resolution', 'N/A')}\t"
                       f"{meta.get('year', 'N/A')}\t"
                       f"{meta.get('logKa', 'N/A')}\t"
                       f"{meta.get('affinity', 'N/A')}\t"
                       f"{meta.get('target', 'N/A')}\n")
        
        print(f"✓ 保存索引文件: {index_file}")
        print(f"  包含 {len(successful_ids)} 个有效复合物")
    
    def save_filtered_records(self):
        """保存过滤记录到 CSV"""
        filtered_file = self.meta_dir / "filtered.csv"
        
        if not self.filtered_records:
            print("✓ 没有过滤的条目")
            return
        
        fieldnames = ['pdb_id', 'reason', 'resolution', 'details']
        # 添加额外的字段
        extra_fields = set()
        for record in self.filtered_records:
            extra_fields.update(record.keys())
        extra_fields -= set(fieldnames)
        fieldnames.extend(sorted(extra_fields))
        
        with open(filtered_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.filtered_records)
        
        print(f"✓ 保存过滤记录: {filtered_file}")
        print(f"  过滤了 {len(self.filtered_records)} 个条目")
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*80)
        print("数据整理统计")
        print("="*80)
        print(f"总条目数:          {self.stats['total']}")
        print(f"成功处理:          {self.stats['success']} "
              f"({100*self.stats['success']/max(1,self.stats['total']):.1f}%)")
        print(f"过滤条目:          {self.stats['filtered']} "
              f"({100*self.stats['filtered']/max(1,self.stats['total']):.1f}%)")
        print("\n过滤原因分布:")
        print(f"  - 缺失蛋白:      {self.stats['missing_protein']}")
        print(f"  - 缺失配体:      {self.stats['missing_ligand']}")
        print(f"  - 分辨率过高:    {self.stats['high_resolution']}")
        print(f"  - 金属过多:      {self.stats['metal_rich']}")
        print(f"  - 解析错误:      {self.stats['parse_error']}")
        print(f"  - 其他错误:      {self.stats['other_error']}")
        print("="*80)
    
    def run(self):
        """运行完整流程"""
        print("="*80)
        print("CASF-2016 数据集整理与完整性校验")
        print("="*80)
        print(f"原始数据目录: {self.raw_data_dir}")
        print(f"输出目录: {self.output_dir}")
        print()
        
        # 1. 创建目录
        self.create_directories()
        print()
        
        # 2. 解析元数据
        metadata = self.parse_coreset_metadata()
        print()
        
        # 3. 获取所有 PDB ID
        if self.coreset_complexes.exists():
            pdb_ids = [d.name for d in self.coreset_complexes.iterdir() if d.is_dir()]
        else:
            print(f"Error: CoreSet complexes directory not found: {self.coreset_complexes}")
            return
        
        print(f"发现 {len(pdb_ids)} 个复合物目录")
        print()
        
        # 4. 处理每个复合物
        print("开始处理复合物...")
        successful_ids = []
        
        for i, pdb_id in enumerate(sorted(pdb_ids), 1):
            if i % 50 == 0 or i == len(pdb_ids):
                print(f"  进度: {i}/{len(pdb_ids)} ({100*i/len(pdb_ids):.1f}%)")
            
            if self.process_complex(pdb_id, metadata):
                successful_ids.append(pdb_id)
        
        print()
        
        # 5. 保存索引和过滤记录
        self.save_index_file(metadata, successful_ids)
        self.save_filtered_records()
        print()
        
        # 6. 打印统计
        self.print_statistics()
        
        print("\n✓ 数据整理完成！")
        print(f"\n后续可以在以下目录继续处理:")
        print(f"  - 复合物: {self.complexes_dir}")
        print(f"  - 口袋提取: {self.pockets_dir}")
        print(f"  - 特征计算: {self.features_dir}")
        print(f"  - 数据划分: {self.splits_dir}")


def main():
    """主函数"""
    # 默认路径
    raw_data_dir = "/Users/apple/code/BINDRAE/data/CASF-2016"
    output_dir = "/Users/apple/code/BINDRAE"
    
    # 可以从命令行参数获取
    if len(sys.argv) > 1:
        raw_data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # 运行整理器
    organizer = CASF2016Organizer(raw_data_dir, output_dir)
    organizer.run()


if __name__ == "__main__":
    main()
