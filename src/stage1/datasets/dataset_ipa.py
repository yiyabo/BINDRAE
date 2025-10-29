"""
IPA 数据加载器

功能：
1. 加载CASF-2016数据集
2. 实时构建局部帧（从N, Cα, C）
3. 实时构建配体tokens
4. 返回IPABatch格式

Author: BINDRAE Team
Date: 2025-10-28
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# FlashIPA路径
flash_ipa_path = '/tmp/flash_ipa/src'
if os.path.exists(flash_ipa_path) and flash_ipa_path not in sys.path:
    sys.path.insert(0, flash_ipa_path)

from flash_ipa.rigid import Rigid, Rotation
from utils.ligand_utils import build_ligand_tokens_from_file


# ============================================================================
# IPABatch 数据结构
# ============================================================================

@dataclass
class IPABatch:
    """
    IPA训练的批数据
    
    所有Tensor都已经是PyTorch格式，可直接用于训练
    """
    # 蛋白
    esm: torch.Tensor           # [B, N, 1280] ESM-2表征
    N: torch.Tensor             # [B, N, 3] 主链N坐标
    Ca: torch.Tensor            # [B, N, 3] 主链Cα坐标
    C: torch.Tensor             # [B, N, 3] 主链C坐标
    node_mask: torch.Tensor     # [B, N] 节点掩码
    
    # 配体
    lig_points: torch.Tensor    # [B, M, 3] 配体坐标（重原子+探针）
    lig_types: torch.Tensor     # [B, M, 12] 配体类型编码
    lig_mask: torch.Tensor      # [B, M] 配体掩码
    
    # Ground Truth
    torsion_angles: torch.Tensor  # [B, N, 7] phi/psi/omega/chi1-4
    torsion_mask: torch.Tensor    # [B, N, 7] 扭转角掩码
    
    # 口袋
    w_res: torch.Tensor         # [B, N] 口袋权重
    
    # Meta
    pdb_ids: List[str]          # 样本ID列表
    n_residues: List[int]       # 每个样本的残基数


# ============================================================================
# PDB坐标提取
# ============================================================================

def extract_backbone_coords(pdb_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从PDB文件提取主链N, Cα, C坐标
    
    Args:
        pdb_file: PDB文件路径
        
    Returns:
        N_coords: [N_res, 3]
        Ca_coords: [N_res, 3]
        C_coords: [N_res, 3]
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', str(pdb_file))
    
    N_coords = []
    Ca_coords = []
    C_coords = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # 只处理标准氨基酸
                if residue.get_id()[0] == ' ':  # 标准残基
                    try:
                        N = residue['N'].get_coord()
                        Ca = residue['CA'].get_coord()
                        C = residue['C'].get_coord()
                        
                        N_coords.append(N)
                        Ca_coords.append(Ca)
                        C_coords.append(C)
                    except KeyError:
                        # 缺少主链原子，跳过
                        continue
    
    N_coords = np.array(N_coords, dtype=np.float32)
    Ca_coords = np.array(Ca_coords, dtype=np.float32)
    C_coords = np.array(C_coords, dtype=np.float32)
    
    return N_coords, Ca_coords, C_coords


# ============================================================================
# Dataset 类
# ============================================================================

class CASF2016IPADataset(Dataset):
    """
    CASF-2016 IPA数据集
    
    加载：
    - ESM-2表征
    - 主链坐标（N, Cα, C）
    - 配体tokens
    - 扭转角GT
    - 口袋权重
    """
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 max_lig_tokens: int = 128):
        """
        Args:
            data_dir: 数据根目录（casf2016/）
            split: 'train', 'val', 或 'test'
            max_lig_tokens: 配体token最大数量
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_lig_tokens = max_lig_tokens
        
        # 目录
        self.complexes_dir = self.data_dir / "complexes"
        self.features_dir = self.data_dir / "processed" / "features"
        self.pockets_dir = self.data_dir / "processed" / "pockets"
        
        # 加载split
        split_file = self.data_dir / "processed" / "splits" / f"{split}.json"
        import json
        with open(split_file, 'r') as f:
            split_data = json.load(f)
            # split文件是字典格式: {"pdb_ids": [...]}
            if isinstance(split_data, dict):
                self.pdb_ids = split_data.get('pdb_ids', split_data)
            else:
                self.pdb_ids = split_data
        
        print(f"✓ 加载 {split} 集: {len(self.pdb_ids)} 个样本")
    
    def __len__(self) -> int:
        return len(self.pdb_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        加载单个样本
        
        Returns:
            样本字典（numpy格式，collate时转为tensor）
        """
        pdb_id = self.pdb_ids[idx]
        
        # 1. 加载ESM特征
        esm_file = self.features_dir / f"{pdb_id}_esm.pt"
        esm_data = torch.load(esm_file, weights_only=False)
        esm_features = esm_data['per_residue'].numpy()  # [N, 1280]
        
        # 2. 提取主链坐标
        protein_pdb = self.complexes_dir / pdb_id / "protein.pdb"
        N_coords, Ca_coords, C_coords = extract_backbone_coords(protein_pdb)
        
        # 3. 加载配体tokens
        ligand_coords_file = self.features_dir / f"{pdb_id}_ligand_coords.npy"
        ligand_sdf_file = self.features_dir / f"{pdb_id}_ligand_normalized.sdf"
        
        lig_tokens = build_ligand_tokens_from_file(
            ligand_coords_file,
            ligand_sdf_file,
            max_tokens=self.max_lig_tokens
        )
        
        # 4. 加载扭转角
        torsions_file = self.features_dir / f"{pdb_id}_torsions.npz"
        torsions_data = np.load(torsions_file)
        
        # 组装扭转角 [N, 7]: phi, psi, omega, chi1-4
        n_res = len(esm_features)
        torsion_angles = np.zeros((n_res, 7), dtype=np.float32)
        torsion_mask = np.zeros((n_res, 7), dtype=bool)
        
        torsion_angles[:, 0] = torsions_data['phi']
        torsion_angles[:, 1] = torsions_data['psi']
        torsion_angles[:, 2] = torsions_data['omega']
        torsion_angles[:, 3:] = torsions_data['chi'][:, :4]
        
        torsion_mask[:, 0] = torsions_data['bb_mask']
        torsion_mask[:, 1] = torsions_data['bb_mask']
        torsion_mask[:, 2] = torsions_data['bb_mask']
        torsion_mask[:, 3:] = torsions_data['chi_mask'][:, :4]
        
        # 5. 加载口袋权重
        w_res_file = self.pockets_dir / f"{pdb_id}_w_res.npy"
        w_res = np.load(w_res_file)
        
        # 确保所有数据长度一致（以ESM特征为准）
        # 有时w_res可能包含非标准残基，需要截断或padding
        if len(w_res) != n_res:
            if len(w_res) > n_res:
                # 截断
                w_res = w_res[:n_res]
            else:
                # Padding
                w_res_padded = np.zeros(n_res, dtype=np.float32)
                w_res_padded[:len(w_res)] = w_res
                w_res = w_res_padded
        
        # 验证所有坐标长度
        assert len(N_coords) == n_res, f"N coords mismatch: {len(N_coords)} vs {n_res}"
        assert len(Ca_coords) == n_res, f"Ca coords mismatch: {len(Ca_coords)} vs {n_res}"
        assert len(C_coords) == n_res, f"C coords mismatch: {len(C_coords)} vs {n_res}"
        assert len(w_res) == n_res, f"w_res mismatch: {len(w_res)} vs {n_res}"
        
        return {
            'pdb_id': pdb_id,
            'esm': esm_features,           # [N, 1280]
            'N': N_coords,                 # [N, 3]
            'Ca': Ca_coords,               # [N, 3]
            'C': C_coords,                 # [N, 3]
            'lig_points': lig_tokens['coords'],   # [M, 3]
            'lig_types': lig_tokens['types'],     # [M, 12]
            'torsion_angles': torsion_angles,     # [N, 7]
            'torsion_mask': torsion_mask,         # [N, 7]
            'w_res': w_res,                # [N]
            'n_residues': n_res,
        }


# ============================================================================
# 批处理函数
# ============================================================================

def collate_ipa_batch(samples: List[Dict]) -> IPABatch:
    """
    将多个样本组合成批
    
    处理：
    - Padding到最大长度
    - 创建掩码
    - 转换为Tensor
    
    Args:
        samples: Dataset返回的样本列表
        
    Returns:
        IPABatch对象
    """
    batch_size = len(samples)
    
    # 找到最大长度
    max_n_res = max(s['n_residues'] for s in samples)
    max_lig_tokens = max(len(s['lig_points']) for s in samples)
    
    # 初始化batch arrays
    esm_batch = np.zeros((batch_size, max_n_res, 1280), dtype=np.float32)
    N_batch = np.zeros((batch_size, max_n_res, 3), dtype=np.float32)
    Ca_batch = np.zeros((batch_size, max_n_res, 3), dtype=np.float32)
    C_batch = np.zeros((batch_size, max_n_res, 3), dtype=np.float32)
    node_mask_batch = np.zeros((batch_size, max_n_res), dtype=bool)
    
    lig_points_batch = np.zeros((batch_size, max_lig_tokens, 3), dtype=np.float32)
    lig_types_batch = np.zeros((batch_size, max_lig_tokens, 12), dtype=np.float32)
    lig_mask_batch = np.zeros((batch_size, max_lig_tokens), dtype=bool)
    
    torsion_angles_batch = np.zeros((batch_size, max_n_res, 7), dtype=np.float32)
    torsion_mask_batch = np.zeros((batch_size, max_n_res, 7), dtype=bool)
    
    w_res_batch = np.zeros((batch_size, max_n_res), dtype=np.float32)
    
    pdb_ids = []
    n_residues_list = []
    
    # 填充数据
    for i, sample in enumerate(samples):
        n_res = sample['n_residues']
        n_lig = len(sample['lig_points'])
        
        esm_batch[i, :n_res] = sample['esm']
        N_batch[i, :n_res] = sample['N']
        Ca_batch[i, :n_res] = sample['Ca']
        C_batch[i, :n_res] = sample['C']
        node_mask_batch[i, :n_res] = True
        
        lig_points_batch[i, :n_lig] = sample['lig_points']
        lig_types_batch[i, :n_lig] = sample['lig_types']
        lig_mask_batch[i, :n_lig] = True
        
        torsion_angles_batch[i, :n_res] = sample['torsion_angles']
        torsion_mask_batch[i, :n_res] = sample['torsion_mask']
        
        w_res_batch[i, :n_res] = sample['w_res']
        
        pdb_ids.append(sample['pdb_id'])
        n_residues_list.append(n_res)
    
    # 转换为Tensor
    return IPABatch(
        esm=torch.from_numpy(esm_batch),
        N=torch.from_numpy(N_batch),
        Ca=torch.from_numpy(Ca_batch),
        C=torch.from_numpy(C_batch),
        node_mask=torch.from_numpy(node_mask_batch),
        lig_points=torch.from_numpy(lig_points_batch),
        lig_types=torch.from_numpy(lig_types_batch),
        lig_mask=torch.from_numpy(lig_mask_batch),
        torsion_angles=torch.from_numpy(torsion_angles_batch),
        torsion_mask=torch.from_numpy(torsion_mask_batch),
        w_res=torch.from_numpy(w_res_batch),
        pdb_ids=pdb_ids,
        n_residues=n_residues_list,
    )


# ============================================================================
# 工厂函数
# ============================================================================

def create_ipa_dataloader(data_dir: str,
                         split: str = 'train',
                         batch_size: int = 4,
                         shuffle: bool = True,
                         num_workers: int = 0,
                         **kwargs):
    """
    创建IPA DataLoader
    
    Args:
        data_dir: 数据目录
        split: 'train', 'val', 或 'test'
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 数据加载线程数
        **kwargs: 其他Dataset参数
        
    Returns:
        DataLoader
        
    Example:
        >>> train_loader = create_ipa_dataloader(
        ...     'data/casf2016', split='train', batch_size=4
        ... )
        >>> for batch in train_loader:
        ...     print(batch.esm.shape)  # [4, max_N, 1280]
    """
    from torch.utils.data import DataLoader
    
    dataset = CASF2016IPADataset(data_dir, split=split, **kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_ipa_batch,
        pin_memory=True  # 加速GPU传输
    )
    
    return dataloader


