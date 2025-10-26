#!/usr/bin/env python3
"""
ESM-2 编码器缓存脚本

功能：
- 智能选择模型大小（Mac: 8M/35M, Linux: 650M）
- 缓存每个蛋白的 ESM-2 表征
- 支持断点续传
- 显存优化（梯度检查点、混合精度）

输出：
- features/<PDBID>_esm.pt
  - 'per_residue': (N_res, d_model) - 每残基表征
  - 'sequence': (d_model,) - 全序列表征
  - 'sequence_str': str - 氨基酸序列
"""

import os
import sys
import platform
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
except ImportError as e:
    print(f"❌ 错误: BioPython 导入失败 - {e}")
    print("请安装: pip install biopython")
    sys.exit(1)

# 三字母到单字母的映射
AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

try:
    import esm
except ImportError as e:
    print(f"❌ 错误: ESM 导入失败 - {e}")
    print("请安装: pip install fair-esm")
    sys.exit(1)


class ESM2Cache:
    """ESM-2 编码器缓存器（智能模型选择）"""
    
    # 模型配置
    MODEL_CONFIG = {
        'Darwin': {  # macOS
            'name': 'esm2_t6_8M_UR50D',  # 最小模型（Mac 友好）
            'fallback': 'esm2_t12_35M_UR50D',  # 备选（如果内存够）
            'batch_size': 1,
            'description': 'Mac 系统 - 使用轻量模型'
        },
        'Linux': {  # 服务器
            'name': 'esm2_t33_650M_UR50D',  # 标准模型
            'fallback': 'esm2_t30_150M_UR50D',  # 备选（显存不足时）
            'batch_size': 4,
            'description': 'Linux 系统 - 使用标准模型'
        }
    }
    
    def __init__(self, base_dir: str, use_fallback: bool = False):
        self.base_dir = Path(base_dir)
        self.complexes_dir = self.base_dir / "data" / "casf2016" / "complexes"
        self.features_dir = self.base_dir / "data" / "casf2016" / "processed" / "features"
        
        # 检测系统并选择模型
        self.system = platform.system()
        self.device = self._get_device()
        self.model_name, self.batch_size = self._select_model(use_fallback)
        
        print(f"\n{'='*80}")
        print(f"ESM-2 编码器缓存")
        print(f"{'='*80}")
        print(f"系统: {self.system}")
        print(f"设备: {self.device}")
        print(f"模型: {self.model_name}")
        print(f"批大小: {self.batch_size}")
        print(f"\n输入目录: {self.complexes_dir}")
        print(f"输出目录: {self.features_dir}")
        
        # 加载模型
        self.model, self.alphabet = self._load_model()
        self.batch_converter = self.alphabet.get_batch_converter()
        
        self.parser = PDBParser(QUIET=True)
        
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'cached': 0,
            'failed_ids': []
        }
    
    def _get_device(self) -> torch.device:
        """获取计算设备"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():  # Apple Silicon
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def _select_model(self, use_fallback: bool) -> Tuple[str, int]:
        """
        智能选择模型
        
        Returns:
            (model_name, batch_size)
        """
        if self.system not in self.MODEL_CONFIG:
            # 未知系统，默认使用 Linux 配置
            print(f"⚠️  未知系统 {self.system}，使用 Linux 配置")
            config = self.MODEL_CONFIG['Linux']
        else:
            config = self.MODEL_CONFIG[self.system]
        
        print(f"\n📌 {config['description']}")
        
        model_name = config['fallback'] if use_fallback else config['name']
        batch_size = config['batch_size']
        
        return model_name, batch_size
    
    def _load_model(self):
        """加载 ESM-2 模型"""
        print(f"\n正在加载模型: {self.model_name}...")
        
        try:
            model, alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
            model = model.to(self.device)
            model.eval()
            
            # 显存优化
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            print(f"✓ 模型加载成功")
            print(f"  - 参数量: {sum(p.numel() for p in model.parameters()):,}")
            
            # 获取嵌入维度（兼容不同ESM版本）
            if hasattr(model, 'args'):
                embed_dim = model.args.embed_dim
            elif hasattr(model, 'embed_dim'):
                embed_dim = model.embed_dim
            else:
                # 从第一层获取
                embed_dim = model.embed_tokens.embedding_dim
            
            print(f"  - 表征维度: {embed_dim}")
            
            # 存储为实例变量
            self.embed_dim = embed_dim
            
            return model, alphabet
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            
            # 尝试备选模型
            if not hasattr(self, '_tried_fallback'):
                self._tried_fallback = True
                print(f"\n尝试备选模型...")
                config = self.MODEL_CONFIG.get(self.system, self.MODEL_CONFIG['Linux'])
                self.model_name = config['fallback']
                return self._load_model()
            else:
                sys.exit(1)
    
    def extract_sequence(self, pdb_id: str) -> Optional[str]:
        """
        从 PDB 文件提取氨基酸序列
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            氨基酸序列字符串（单字母代码）
        """
        protein_pdb = self.complexes_dir / pdb_id / "protein.pdb"
        
        if not protein_pdb.exists():
            return None
        
        try:
            structure = self.parser.get_structure(pdb_id, str(protein_pdb))
            
            # 收集所有标准氨基酸残基
            residues = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if is_aa(residue, standard=True):
                            residues.append(residue)
            
            if len(residues) == 0:
                return None
            
            # 转换为单字母序列
            sequence = ""
            for res in residues:
                try:
                    resname = res.get_resname().strip()
                    aa = AA_MAP.get(resname, 'X')  # 非标准残基用 X
                    sequence += aa
                except Exception:
                    sequence += "X"
            
            return sequence
            
        except Exception as e:
            print(f"  ⚠️  {pdb_id}: 序列提取失败 - {e}")
            return None
    
    def encode_sequence(self, pdb_id: str, sequence: str) -> Optional[Dict]:
        """
        用 ESM-2 编码序列
        
        Args:
            pdb_id: PDB ID
            sequence: 氨基酸序列
            
        Returns:
            编码结果字典
        """
        try:
            # 准备输入
            data = [(pdb_id, sequence)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            
            # 前向传播（无梯度）
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[self.model.num_layers])
            
            # 提取表征
            # representations: (batch, seq_len, d_model)
            per_residue = results['representations'][self.model.num_layers][0, 1:-1]  # 去掉 <cls> 和 <eos>
            
            # 全序列表征（使用 <cls> token）
            sequence_repr = results['representations'][self.model.num_layers][0, 0]
            
            return {
                'per_residue': per_residue.cpu(),  # (N_res, d_model)
                'sequence': sequence_repr.cpu(),    # (d_model,)
                'sequence_str': sequence,           # str
                'n_residues': len(sequence)
            }
            
        except Exception as e:
            print(f"  ⚠️  {pdb_id}: 编码失败 - {e}")
            return None
    
    def cache_protein(self, pdb_id: str) -> bool:
        """
        缓存单个蛋白的 ESM-2 表征
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            是否成功
        """
        self.stats['total'] += 1
        
        output_file = self.features_dir / f"{pdb_id}_esm.pt"
        
        # 跳过已缓存
        if output_file.exists():
            self.stats['cached'] += 1
            return True
        
        # 提取序列
        sequence = self.extract_sequence(pdb_id)
        if sequence is None:
            self.stats['failed'] += 1
            self.stats['failed_ids'].append(pdb_id)
            return False
        
        # 编码
        encoding = self.encode_sequence(pdb_id, sequence)
        if encoding is None:
            self.stats['failed'] += 1
            self.stats['failed_ids'].append(pdb_id)
            return False
        
        # 保存
        torch.save(encoding, output_file)
        
        self.stats['success'] += 1
        
        # 进度报告
        if self.stats['total'] % 10 == 0:
            print(f"  进度: {self.stats['total']} | "
                  f"成功: {self.stats['success']} | "
                  f"缓存: {self.stats['cached']} | "
                  f"失败: {self.stats['failed']}")
        
        return True
    
    def run(self):
        """运行缓存"""
        # 获取所有蛋白
        pdb_ids = sorted([d.name for d in self.complexes_dir.iterdir() if d.is_dir()])
        
        print(f"\n发现 {len(pdb_ids)} 个蛋白\n")
        print("开始缓存...\n")
        
        for pdb_id in pdb_ids:
            self.cache_protein(pdb_id)
        
        # 最终统计
        print(f"\n{'='*80}")
        print(f"缓存完成统计")
        print(f"{'='*80}")
        print(f"总数:     {self.stats['total']}")
        print(f"成功:     {self.stats['success']} ({100*self.stats['success']/max(1,self.stats['total']):.1f}%)")
        print(f"已缓存:   {self.stats['cached']} (跳过)")
        print(f"失败:     {self.stats['failed']}")
        
        if self.stats['failed_ids']:
            print(f"\n失败的条目: {', '.join(self.stats['failed_ids'])}")
        
        print(f"\n✓ ESM-2 缓存完成！")
        print(f"\n输出文件:")
        print(f"  - <PDBID>_esm.pt")
        print(f"    - 'per_residue': (N_res, {self.embed_dim}) - 每残基表征")
        print(f"    - 'sequence': ({self.embed_dim},) - 全序列表征")
        print(f"    - 'sequence_str': str - 氨基酸序列")
        print(f"    - 'n_residues': int - 残基数量")


def main():
    import argparse
    
    # 默认使用项目根目录
    script_dir = Path(__file__).resolve().parent
    default_base_dir = str(script_dir.parent)
    
    parser = argparse.ArgumentParser(description='ESM-2 编码器缓存（智能模型选择）')
    parser.add_argument('--base_dir', type=str, default=default_base_dir,
                       help=f'项目根目录（默认: 脚本所在项目根目录）')
    parser.add_argument('--fallback', action='store_true',
                       help='使用备选模型（显存不足时）')
    
    args = parser.parse_args()
    
    # 运行缓存
    cache = ESM2Cache(args.base_dir, use_fallback=args.fallback)
    cache.run()


if __name__ == "__main__":
    main()
