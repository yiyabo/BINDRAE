"""
配体 Token 构建工具

功能：
1. 从 ligand_coords.npy 提取重原子坐标
2. RDKit 检测关键原子类型 (HBD/HBA/芳香/带电)
3. 生成方向探针 (≤2个/原子)
4. 重要性采样 (M≤128)
5. 12维原子类型嵌入

设计决策：
- 探针：基于邻近原子的法向量
- 采样：优先保留关键原子(HBD/HBA/带电)
- 类型编码：12维 one-hot (C/N/O/S/P/F/Cl/Br/I/芳香/带电+/-/other)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem, RDConfig
    from rdkit.Chem import AllChem, Descriptors, ChemicalFeatures
    import os
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Ligand processing will be limited.")


# ============================================================================
# 常量定义
# ============================================================================

# 原子类型编码 (12维)
ATOM_TYPE_MAPPING = {
    'C': 0,
    'N': 1,
    'O': 2,
    'S': 3,
    'P': 4,
    'F': 5,
    'Cl': 6,
    'Br': 7,
    'I': 8,
    'aromatic': 9,      # 芳香原子标记
    'positive': 10,      # 带正电
    'negative': 11,      # 带负电
}

# 方向探针配置
MAX_PROBES_PER_ATOM = 2  # 每个原子最多2个探针
PROBE_DISTANCE = 1.5     # 探针距离原子的距离 (Å)

# 重要性采样配置
MAX_LIGAND_TOKENS = 128  # 配体token上限


# ============================================================================
# 核心类
# ============================================================================

class LigandTokenBuilder:
    """配体 Token 构建器"""
    
    def __init__(self, max_tokens: int = MAX_LIGAND_TOKENS):
        """
        Args:
            max_tokens: 最大token数 (重原子+探针)
        """
        self.max_tokens = max_tokens
        
        # RDKit Feature Factory (用于检测HBD/HBA)
        if RDKIT_AVAILABLE:
            fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            self.feature_factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        else:
            self.feature_factory = None
    
    def build_tokens(self, ligand_coords: np.ndarray, 
                    ligand_mol: Optional[Chem.Mol] = None) -> Dict[str, np.ndarray]:
        """
        构建配体 tokens
        
        Args:
            ligand_coords: 重原子坐标 (N_atoms, 3)
            ligand_mol: RDKit 分子对象 (可选，用于类型检测)
            
        Returns:
            {
                'coords': (M, 3) - token坐标 (重原子+探针)
                'types': (M, 12) - 原子类型 one-hot
                'is_probe': (M,) - 是否为探针
                'atom_indices': (M,) - 对应的原子索引 (-1表示探针)
                'importance': (M,) - 重要性权重
            }
        """
        n_atoms = len(ligand_coords)
        
        # 1. 检测原子类型和重要性
        atom_info = self._analyze_atoms(ligand_mol, n_atoms)
        
        # 2. 生成方向探针
        probe_coords, probe_atom_indices = self._generate_probes(
            ligand_coords, ligand_mol, atom_info
        )
        
        # 3. 合并重原子和探针
        all_coords = np.vstack([ligand_coords, probe_coords]) if len(probe_coords) > 0 else ligand_coords
        all_atom_indices = np.concatenate([
            np.arange(n_atoms),
            probe_atom_indices
        ]) if len(probe_coords) > 0 else np.arange(n_atoms)
        
        is_probe = np.concatenate([
            np.zeros(n_atoms, dtype=bool),
            np.ones(len(probe_coords), dtype=bool)
        ]) if len(probe_coords) > 0 else np.zeros(n_atoms, dtype=bool)
        
        # 4. 重要性采样 (如果超过上限)
        if len(all_coords) > self.max_tokens:
            keep_indices = self._importance_sampling(
                all_coords, all_atom_indices, is_probe, atom_info
            )
            all_coords = all_coords[keep_indices]
            all_atom_indices = all_atom_indices[keep_indices]
            is_probe = is_probe[keep_indices]
        
        # 5. 编码原子类型
        types = self._encode_atom_types(all_atom_indices, is_probe, atom_info)
        
        # 6. 计算重要性权重
        importance = self._compute_importance(all_atom_indices, is_probe, atom_info)
        
        return {
            'coords': all_coords.astype(np.float32),
            'types': types.astype(np.float32),
            'is_probe': is_probe,
            'atom_indices': all_atom_indices.astype(np.int32),
            'importance': importance.astype(np.float32)
        }
    
    def _analyze_atoms(self, mol: Optional[Chem.Mol], n_atoms: int) -> Dict:
        """
        分析原子类型和特性
        
        Returns:
            {
                'elements': List[str] - 元素符号
                'aromatic': np.ndarray - 是否芳香
                'charge': np.ndarray - 电荷
                'hbd': List[int] - 氢键供体原子索引
                'hba': List[int] - 氢键受体原子索引
            }
        """
        info = {
            'elements': ['C'] * n_atoms,
            'aromatic': np.zeros(n_atoms, dtype=bool),
            'charge': np.zeros(n_atoms, dtype=np.float32),
            'hbd': [],
            'hba': [],
        }
        
        if mol is None or not RDKIT_AVAILABLE:
            return info
        
        # 提取原子信息（只处理前 n_atoms 个）
        mol_n_atoms = mol.GetNumAtoms()
        if mol_n_atoms != n_atoms:
            # 分子原子数与坐标不匹配，只取最小值
            import warnings
            warnings.warn(f"Molecule has {mol_n_atoms} atoms but coords have {n_atoms}. Using min={min(mol_n_atoms, n_atoms)}")
            process_n = min(mol_n_atoms, n_atoms)
        else:
            process_n = n_atoms
        
        for i in range(process_n):
            atom = mol.GetAtomWithIdx(i)
            info['elements'][i] = atom.GetSymbol()
            info['aromatic'][i] = atom.GetIsAromatic()
            info['charge'][i] = atom.GetFormalCharge()
        
        # 检测 HBD/HBA (使用 RDKit Feature Factory)
        if self.feature_factory is not None:
            try:
                features = self.feature_factory.GetFeaturesForMol(mol)
                for feat in features:
                    if feat.GetFamily() == 'Donor':
                        info['hbd'].extend(feat.GetAtomIds())
                    elif feat.GetFamily() == 'Acceptor':
                        info['hba'].extend(feat.GetAtomIds())
            except Exception:
                pass  # 忽略特征检测失败
        
        return info
    
    def _generate_probes(self, coords: np.ndarray, 
                        mol: Optional[Chem.Mol],
                        atom_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        为关键原子生成方向探针
        
        策略：
        - HBD/HBA: 生成1-2个探针，方向指向孤对电子/氢
        - 芳香环: 生成垂直于环平面的探针
        - 其他: 不生成探针
        
        Returns:
            probe_coords: (N_probes, 3)
            probe_atom_indices: (N_probes,) - 对应的原子索引
        """
        probe_coords = []
        probe_atom_indices = []
        
        if mol is None or not RDKIT_AVAILABLE:
            return np.array(probe_coords), np.array(probe_atom_indices, dtype=np.int32)
        
        # 为 HBD/HBA 生成探针
        key_atoms = set(atom_info['hbd'] + atom_info['hba'])
        
        for atom_idx in key_atoms:
            if atom_idx >= len(coords):
                continue
            
            atom_coord = coords[atom_idx]
            atom = mol.GetAtomWithIdx(atom_idx)
            
            # 获取邻近原子
            neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors()]
            neighbor_coords = [coords[n.GetIdx()] for n in neighbors if n.GetIdx() < len(coords)]
            
            if len(neighbor_coords) == 0:
                continue
            
            # 计算探针方向
            probe_directions = self._compute_probe_directions(
                atom_coord, neighbor_coords, max_probes=MAX_PROBES_PER_ATOM
            )
            
            for direction in probe_directions:
                probe_coord = atom_coord + PROBE_DISTANCE * direction
                probe_coords.append(probe_coord)
                probe_atom_indices.append(atom_idx)
        
        if len(probe_coords) == 0:
            return np.array([]), np.array([], dtype=np.int32)
        
        return np.array(probe_coords), np.array(probe_atom_indices, dtype=np.int32)
    
    def _compute_probe_directions(self, atom_coord: np.ndarray,
                                neighbor_coords: List[np.ndarray],
                                max_probes: int = 2) -> List[np.ndarray]:
        """
        计算探针方向 (基于邻近原子的法向量)
        
        Args:
            atom_coord: 中心原子坐标 (3,)
            neighbor_coords: 邻近原子坐标列表
            max_probes: 最多生成几个探针
            
        Returns:
            方向向量列表 (单位向量)
        """
        directions = []
        
        if len(neighbor_coords) == 1:
            # 只有1个邻居：反向
            vec = atom_coord - neighbor_coords[0]
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                directions.append(vec / norm)
        
        elif len(neighbor_coords) == 2:
            # 2个邻居：角平分线的反向
            vec1 = neighbor_coords[0] - atom_coord
            vec2 = neighbor_coords[1] - atom_coord
            vec1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
            vec2 = vec2 / (np.linalg.norm(vec2) + 1e-8)
            bisector = -(vec1 + vec2)
            norm = np.linalg.norm(bisector)
            if norm > 1e-6:
                directions.append(bisector / norm)
        
        elif len(neighbor_coords) >= 3:
            # 3+个邻居：使用法向量
            # 取前3个计算平面法向量
            vecs = [neighbor_coords[i] - atom_coord for i in range(min(3, len(neighbor_coords)))]
            v1, v2 = vecs[0], vecs[1]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normal = normal / norm
                directions.append(normal)
                if max_probes >= 2:
                    directions.append(-normal)  # 两侧都加
        
        return directions[:max_probes]
    
    def _importance_sampling(self, coords: np.ndarray,
                            atom_indices: np.ndarray,
                            is_probe: np.ndarray,
                            atom_info: Dict) -> np.ndarray:
        """
        重要性采样，保留最重要的 max_tokens 个
        
        优先级：
        1. HBD/HBA 原子及其探针
        2. 带电原子
        3. 芳香原子
        4. 其他重原子
        5. 其他探针
        
        Returns:
            keep_indices: 保留的索引
        """
        n_tokens = len(coords)
        importance_scores = np.zeros(n_tokens)
        
        for i in range(n_tokens):
            if is_probe[i]:
                # 探针：继承对应原子的重要性
                orig_atom_idx = atom_indices[i]
                score = 0.0
                if orig_atom_idx in atom_info['hbd'] or orig_atom_idx in atom_info['hba']:
                    score = 3.0  # HBD/HBA探针
                elif abs(atom_info['charge'][orig_atom_idx]) > 0.1:
                    score = 2.0  # 带电原子探针
                elif atom_info['aromatic'][orig_atom_idx]:
                    score = 1.0  # 芳香原子探针
                else:
                    score = 0.5  # 普通探针
            else:
                # 重原子
                atom_idx = atom_indices[i]
                score = 1.0  # 基础分
                
                if atom_idx in atom_info['hbd'] or atom_idx in atom_info['hba']:
                    score += 5.0  # HBD/HBA
                if abs(atom_info['charge'][atom_idx]) > 0.1:
                    score += 3.0  # 带电
                if atom_info['aromatic'][atom_idx]:
                    score += 2.0  # 芳香
                
                # 元素类型加分
                element = atom_info['elements'][atom_idx]
                if element in ['N', 'O', 'S', 'P']:
                    score += 1.0  # 杂原子
            
            importance_scores[i] = score
        
        # 按重要性排序，保留前 max_tokens 个
        keep_indices = np.argsort(-importance_scores)[:self.max_tokens]
        # 恢复原始顺序（重原子在前，探针在后），方便后续处理
        # 注意：重要性权重已保存在 importance 数组中，这里排序不影响
        keep_indices = np.sort(keep_indices)
        
        return keep_indices
    
    def _encode_atom_types(self, atom_indices: np.ndarray,
                        is_probe: np.ndarray,
                        atom_info: Dict) -> np.ndarray:
        """
        编码原子类型为 12 维 one-hot
        
        维度：[C, N, O, S, P, F, Cl, Br, I, 芳香, 正电, 负电]
        
        注意：芳香性是叠加属性，可以与元素类型同时为1
            例如芳香碳：types[i, 0]=1 且 types[i, 9]=1
        
        Returns:
            types: (M, 12)
        """
        n_tokens = len(atom_indices)
        types = np.zeros((n_tokens, 12), dtype=np.float32)
        
        for i in range(n_tokens):
            atom_idx = atom_indices[i]
            
            if atom_idx < 0 or atom_idx >= len(atom_info['elements']):
                continue  # 无效索引
            
            # 元素类型
            element = atom_info['elements'][atom_idx]
            if element in ATOM_TYPE_MAPPING:
                types[i, ATOM_TYPE_MAPPING[element]] = 1.0
            
            # 芳香性
            if atom_info['aromatic'][atom_idx]:
                types[i, ATOM_TYPE_MAPPING['aromatic']] = 1.0
            
            # 电荷
            charge = atom_info['charge'][atom_idx]
            if charge > 0.1:
                types[i, ATOM_TYPE_MAPPING['positive']] = 1.0
            elif charge < -0.1:
                types[i, ATOM_TYPE_MAPPING['negative']] = 1.0
        
        return types
    
    def _compute_importance(self, atom_indices: np.ndarray,
                        is_probe: np.ndarray,
                        atom_info: Dict) -> np.ndarray:
        """
        计算重要性权重 (用于注意力加权)
        
        Returns:
            importance: (M,) 值域 [0, 1]
        """
        n_tokens = len(atom_indices)
        importance = np.ones(n_tokens, dtype=np.float32)
        
        for i in range(n_tokens):
            atom_idx = atom_indices[i]
            
            if atom_idx < 0 or atom_idx >= len(atom_info['elements']):
                importance[i] = 0.5
                continue
            
            score = 0.5  # 基础值
            
            if atom_idx in atom_info['hbd'] or atom_idx in atom_info['hba']:
                score += 0.3
            if abs(atom_info['charge'][atom_idx]) > 0.1:
                score += 0.2
            
            importance[i] = min(score, 1.0)
        
        return importance


# ============================================================================
# 便捷函数
# ============================================================================

def build_ligand_tokens_from_file(ligand_coords_file: Path,
                                ligand_sdf_file: Optional[Path] = None,
                                max_tokens: int = MAX_LIGAND_TOKENS) -> Dict[str, np.ndarray]:
    """
    从文件构建配体 tokens
    
    Args:
        ligand_coords_file: *_ligand_coords.npy 文件路径
        ligand_sdf_file: ligand.sdf 文件路径 (可选)
        max_tokens: 最大 token 数
        
    Returns:
        配体 tokens 字典
    """
    # 加载坐标
    coords = np.load(ligand_coords_file)
    
    # 加载分子 (如果提供了 SDF)
    mol = None
    if ligand_sdf_file is not None and ligand_sdf_file.exists() and RDKIT_AVAILABLE:
        try:
            # 先尝试保留氢原子，如果失败再去除
            supplier = Chem.SDMolSupplier(str(ligand_sdf_file), removeHs=False, sanitize=False)
            mol = supplier[0]
            if mol is not None:
                # 去除氢原子
                mol = Chem.RemoveHs(mol, sanitize=False)
                # 尝试标准化（忽略失败）
                try:
                    Chem.SanitizeMol(mol)
                except:
                    # 标准化失败，手动初始化必要信息
                    try:
                        mol.UpdatePropertyCache(strict=False)
                        Chem.GetSymmSSSR(mol)  # 初始化环信息
                    except:
                        pass  # 如果还失败就放弃
        except Exception as e:
            # 记录错误但不中断
            import warnings
            warnings.warn(f"Failed to load molecule from {ligand_sdf_file}: {e}")
            mol = None
    
    # 构建 tokens
    builder = LigandTokenBuilder(max_tokens=max_tokens)
    return builder.build_tokens(coords, mol)


def encode_ligand_batch(ligand_tokens_list: List[Dict[str, np.ndarray]],
                    max_seq_len: int = MAX_LIGAND_TOKENS) -> Dict[str, np.ndarray]:
    """
    批量编码配体 tokens (用于 DataLoader)
    
    Args:
        ligand_tokens_list: 配体 tokens 列表
        max_seq_len: 最大序列长度 (padding)
        
    Returns:
        {
            'coords': (B, M, 3)
            'types': (B, M, 12)
            'mask': (B, M) - padding mask
            'importance': (B, M)
        }
    """
    batch_size = len(ligand_tokens_list)
    
    coords = np.zeros((batch_size, max_seq_len, 3), dtype=np.float32)
    types = np.zeros((batch_size, max_seq_len, 12), dtype=np.float32)
    mask = np.zeros((batch_size, max_seq_len), dtype=bool)
    importance = np.zeros((batch_size, max_seq_len), dtype=np.float32)
    
    for i, tokens in enumerate(ligand_tokens_list):
        n_tokens = len(tokens['coords'])
        coords[i, :n_tokens] = tokens['coords']
        types[i, :n_tokens] = tokens['types']
        mask[i, :n_tokens] = True
        importance[i, :n_tokens] = tokens['importance']
    
    return {
        'coords': coords,
        'types': types,
        'mask': mask,
        'importance': importance
    }
