"""
配体 Token 构建工具

功能：
1. 从 ligand_coords.npy 提取坐标（重原子+极性氢）
2. RDKit 检测关键原子类型 (HBD/HBA/芳香/带电)
3. 生成方向探针（仅为HBA）
4. 重要性采样 (M≤128)
5. 13维原子类型嵌入

设计决策（方案A：保留极性氢）：
- ✅ 保留极性氢：N-H, O-H, S-H（功能性关键，参与氢键形成）
- ✅ 移除非极性氢：C-H（方向性弱，信息冗余）
- ✅ HBD策略：直接使用极性氢原子位置（完全准确）
- ✅ HBA策略：生成孤对电子方向探针（计算近似）
- ✅ 采样：优先保留关键原子(HBD/HBA/带电)
- ✅ 类型编码：13维 one-hot (C/N/O/S/P/F/Cl/Br/I/H/芳香/带电+/-)

科研理由：
1. 氢键是最重要的相互作用（占60-70%），高度方向性
2. HBD的方向性 = 氢原子的真实位置，保留极性氢提供完全准确的方向信息
3. HBA的方向性 = 孤对电子方向，需要计算探针近似
4. 与蛋白质处理一致（蛋白质也保留极性氢）
5. 符合项目核心创新：显式编码相互作用方向性
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from rdkit import Chem, RDConfig, RDLogger
    from rdkit.Chem import AllChem, Descriptors, ChemicalFeatures
    import os
    
    # 关闭RDKit所有警告（避免训练时刷屏）
    RDLogger.DisableLog('rdApp.*')
    
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Ligand processing will be limited.")


# ============================================================================
# 常量定义
# ============================================================================

# 原子类型编码 (13维) - 添加了氢原子类型
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
    'H': 9,          # 极性氢 (N-H, O-H, S-H)
    'aromatic': 10,   # 芳香原子标记
    'positive': 11,   # 带正电
    'negative': 12,   # 带负电
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
            ligand_coords: 坐标 (N_atoms, 3) - 包含重原子+极性氢
            ligand_mol: RDKit 分子对象 (可选，用于类型检测)
            
        Returns:
            {
                'coords': (M, 3) - token坐标 (原子+探针)
                'types': (M, 13) - 原子类型 one-hot
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
        
        # 严格的数据一致性校验
        mol_n_atoms = mol.GetNumAtoms()
        if mol_n_atoms != n_atoms:
            # 严重错误：分子和坐标数量不匹配
            raise ValueError(
                f"🚨 严重数据不一致！\n"
                f"RDKit解析分子: {mol_n_atoms}个原子\n"
                f"坐标文件: {n_atoms}个原子\n"
                f"差值: {abs(mol_n_atoms - n_atoms)}个原子\n"
                f"这表明数据预处理存在问题，请检查:\n"
                f"1. SDF文件是否包含氢原子\n"
                f"2. 坐标文件是否正确生成\n"
                f"3. 分子标准化是否正确执行\n"
                f"📍 数据来源: 请检查数据预处理脚本"
            )

        # 数据一致性验证通过，安全提取原子信息
        for i in range(n_atoms):
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
        
        策略（方案A：保留极性氢）：
        - HBD（氢键供体）: **不生成探针**，直接使用极性氢原子位置
        - HBA（氢键受体）: 生成1-2个探针，方向指向孤对电子
        - 芳香环: 可选，生成垂直于环平面的探针
        
        理由：
        - 极性氢（N-H, O-H, S-H）已经保留在坐标中
        - 氢原子本身就是HBD的"探针"，位置完全准确
        - HBA的孤对电子不可见，需要计算探针方向
        
        Returns:
            probe_coords: (N_probes, 3)
            probe_atom_indices: (N_probes,) - 对应的原子索引
        """
        probe_coords = []
        probe_atom_indices = []
        
        if mol is None or not RDKIT_AVAILABLE:
            return np.array(probe_coords), np.array(probe_atom_indices, dtype=np.int32)
        
        # ✅ 只为 HBA（氢键受体）生成探针
        # HBD（氢键供体）的氢原子已经在坐标中，不需要额外探针
        key_atoms = set(atom_info['hba'])  # 只取 HBA，不包含 HBD
        
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
        
        优先级（方案A：保留极性氢）：
        1. 极性氢（H）- 真正的 HBD，最高优先级
        2. HBD/HBA 重原子（N, O, S）及其探针
        3. 带电原子
        4. 芳香原子
        5. 其他重原子
        6. 其他探针
        
        注意：
        - info['hbd'] 存的是重原子索引（N, O, S），不是氢原子
        - 极性氢才是真正的 HBD，必须给予最高优先级，防止被采样丢弃
        - HBA 探针（孤对电子方向）也很重要
        
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
                if orig_atom_idx in atom_info['hba']:  # 只有 HBA 才有探针
                    score = 5.0  # HBA 探针（孤对电子方向）- 高优先级
                elif abs(atom_info['charge'][orig_atom_idx]) > 0.1:
                    score = 3.0  # 带电原子探针
                elif atom_info['aromatic'][orig_atom_idx]:
                    score = 2.0  # 芳香原子探针
                else:
                    score = 1.0  # 普通探针
            else:
                # 原子（包括重原子和极性氢）
                atom_idx = atom_indices[i]
                element = atom_info['elements'][atom_idx]
                score = 1.0  # 基础分
                
                # ✅ 极性氢：与 HBD/HBA 重原子同等重要
                if element == 'H':
                    score += 3.0  # 极性氢（降低权重，避免过度主导）
                
                # HBD/HBA 重原子（N, O, S）
                elif atom_idx in atom_info['hbd'] or atom_idx in atom_info['hba']:
                    score += 5.0  # HBD/HBA 重原子（保持较高权重）
                
                # 带电原子
                if abs(atom_info['charge'][atom_idx]) > 0.1:
                    score += 3.0  # 带电
                
                # 芳香原子
                if atom_info['aromatic'][atom_idx]:
                    score += 2.0  # 芳香
                
                # 杂原子加分（除了氢）
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
        编码原子类型为 13 维 one-hot
        
        维度：[C, N, O, S, P, F, Cl, Br, I, H, 芳香, 正电, 负电]
        
        注意：
        - 芳香性/电荷是叠加属性，可以与元素类型同时为1
        - 例如芳香碳：types[i, 0]=1 且 types[i, 10]=1
        - 极性氢：types[i, 9]=1（保留真实位置）
        
        Returns:
            types: (M, 13)
        """
        n_tokens = len(atom_indices)
        types = np.zeros((n_tokens, 13), dtype=np.float32)
        
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
        
        方案A：极性氢和HBA探针都应该有高权重
        
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
            element = atom_info['elements'][atom_idx]
            
            # ✅ 极性氢：中等权重（避免过度主导）
            if element == 'H':
                score += 0.2  # 极性氢权重 = 0.7（降低权重）
            
            # HBD/HBA 重原子
            elif atom_idx in atom_info['hbd'] or atom_idx in atom_info['hba']:
                score += 0.3  # HBD/HBA 重原子权重 = 0.8（保持较高）
            
            # 带电原子
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
        # 直接加载（预处理保留了极性氢，移除了非极性氢）
        supplier = Chem.SDMolSupplier(str(ligand_sdf_file), removeHs=False, sanitize=False)
        mol = supplier[0]
        
        if mol is None:
            raise ValueError(
                f"❌ 无法加载配体分子: {ligand_sdf_file}\n"
                f"这表明SDF文件损坏或格式错误。\n"
                f"请重新运行数据预处理: python scripts/prepare_ligands.py"
            )
        
        # ✅ 严格验证原子数一致性（科研代码不允许不一致）
        if mol.GetNumAtoms() != len(coords):
            raise ValueError(
                f"🚨 数据不一致错误！\n"
                f"配体: {ligand_sdf_file.stem}\n"
                f"SDF分子: {mol.GetNumAtoms()} 个原子\n"
                f"坐标文件: {len(coords)} 个原子\n"
                f"差异: {abs(mol.GetNumAtoms() - len(coords))} 个原子\n\n"
                f"这是严重的数据预处理问题，不能继续训练！\n"
                f"解决方案:\n"
                f"1. 验证数据: python scripts/verify_ligand_consistency.py\n"
                f"2. 重新预处理: python scripts/prepare_ligands.py\n"
                f"3. 确保预处理时验证通过"
            )
        
        # 初始化分子信息（必需，失败则报错）
        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)  # 初始化环信息
        except Exception as e:
            raise ValueError(
                f"❌ 配体分子初始化失败: {ligand_sdf_file.stem}\n"
                f"错误: {e}\n"
                f"这表明分子结构有问题，请检查SDF文件。"
            )
    
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
            'types': (B, M, 13) - 包含氢原子类型
            'mask': (B, M) - padding mask
            'importance': (B, M)
        }
    """
    batch_size = len(ligand_tokens_list)
    
    coords = np.zeros((batch_size, max_seq_len, 3), dtype=np.float32)
    types = np.zeros((batch_size, max_seq_len, 13), dtype=np.float32)
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
