"""
残基常量数据

从OpenFold提取的标准原子坐标和刚体帧数据

Reference: AlphaFold2 / OpenFold
"""

import numpy as np

# 20种标准氨基酸（单字母代码顺序）
restypes = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
]

# 单字母 ↔ 三字母转换
restype_1to3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}

restype_3to1 = {v: k for k, v in restype_1to3.items()}

# Chi角掩码（每种残基有哪些chi角）
# [chi1, chi2, chi3, chi4]
chi_angles_mask = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 1.0, 0.0, 0.0],  # PRO
    [1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 0.0, 0.0, 0.0],  # VAL
]

# Atom14编码：每个残基最多14个重原子
# 格式: [原子名, rigid_group_idx, (x, y, z)]
# rigid_group: 0=backbone, 3=psi, 4=chi1, 5=chi2, 6=chi3, 7=chi4

rigid_group_atom_positions = {
    "ALA": [
        ["N", 0, (-0.525, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.529, -0.774, -1.205)],
        ["O", 3, (0.627, 1.062, 0.000)],
    ],
    "ARG": [
        ["N", 0, (-0.524, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.524, -0.778, -1.209)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.616, 1.390, -0.000)],
        ["CD", 5, (0.564, 1.414, 0.000)],
        ["NE", 6, (0.539, 1.357, -0.000)],
        ["NH1", 7, (0.206, 2.301, 0.000)],
        ["NH2", 7, (2.078, 0.978, -0.000)],
        ["CZ", 7, (0.758, 1.093, -0.000)],
    ],
    "ASN": [
        ["N", 0, (-0.536, 1.357, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.531, -0.787, -1.200)],
        ["O", 3, (0.625, 1.062, 0.000)],
        ["CG", 4, (0.584, 1.399, 0.000)],
        ["ND2", 5, (0.593, -1.188, 0.001)],
        ["OD1", 5, (0.633, 1.059, 0.000)],
    ],
    "ASP": [
        ["N", 0, (-0.525, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, 0.000, -0.000)],
        ["CB", 0, (-0.526, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.593, 1.398, -0.000)],
        ["OD1", 5, (0.610, 1.091, 0.000)],
        ["OD2", 5, (0.592, -1.101, -0.003)],
    ],
    "CYS": [
        ["N", 0, (-0.522, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, 0.000)],
        ["CB", 0, (-0.519, -0.773, -1.212)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["SG", 4, (0.728, 1.653, 0.000)],
    ],
    "GLN": [
        ["N", 0, (-0.526, 1.361, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.779, -1.207)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.615, 1.393, 0.000)],
        ["CD", 5, (0.587, 1.399, -0.000)],
        ["NE2", 6, (0.593, -1.189, -0.001)],
        ["OE1", 6, (0.634, 1.060, 0.000)],
    ],
    "GLU": [
        ["N", 0, (-0.528, 1.361, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.526, -0.781, -1.207)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.615, 1.392, 0.000)],
        ["CD", 5, (0.600, 1.397, 0.000)],
        ["OE1", 6, (0.607, 1.095, -0.000)],
        ["OE2", 6, (0.589, -1.104, -0.001)],
    ],
    "GLY": [
        ["N", 0, (-0.572, 1.337, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.517, -0.000, -0.000)],
        ["O", 3, (0.626, 1.062, -0.000)],
    ],
    "HIS": [
        ["N", 0, (-0.527, 1.360, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.778, -1.208)],
        ["O", 3, (0.625, 1.063, 0.000)],
        ["CG", 4, (0.600, 1.370, -0.000)],
        ["CD2", 5, (0.889, -1.021, 0.003)],
        ["ND1", 5, (0.744, 1.160, -0.000)],
        ["CE1", 5, (2.030, 0.851, 0.002)],
        ["NE2", 5, (2.145, -0.466, 0.004)],
    ],
    "ILE": [
        ["N", 0, (-0.493, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.536, -0.793, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.534, 1.437, -0.000)],
        ["CG2", 4, (0.540, -0.785, -1.199)],
        ["CD1", 5, (0.619, 1.391, 0.000)],
    ],
    "LEU": [
        ["N", 0, (-0.520, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.773, -1.214)],
        ["O", 3, (0.625, 1.063, -0.000)],
        ["CG", 4, (0.678, 1.371, 0.000)],
        ["CD1", 5, (0.530, 1.430, -0.000)],
        ["CD2", 5, (0.535, -0.774, 1.200)],
    ],
    "LYS": [
        ["N", 0, (-0.526, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.524, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.619, 1.390, 0.000)],
        ["CD", 5, (0.559, 1.417, 0.000)],
        ["CE", 6, (0.560, 1.416, 0.000)],
        ["NZ", 7, (0.554, 1.387, 0.000)],
    ],
    "MET": [
        ["N", 0, (-0.521, 1.364, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.210)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["CG", 4, (0.613, 1.391, -0.000)],
        ["SD", 5, (0.703, 1.695, 0.000)],
        ["CE", 6, (0.320, 1.786, -0.000)],
    ],
    "PHE": [
        ["N", 0, (-0.518, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, -0.000)],
        ["CB", 0, (-0.525, -0.776, -1.212)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.377, 0.000)],
        ["CD1", 5, (0.709, 1.195, -0.000)],
        ["CD2", 5, (0.706, -1.196, 0.000)],
        ["CE1", 5, (2.102, 1.198, -0.000)],
        ["CE2", 5, (2.098, -1.201, -0.000)],
        ["CZ", 5, (2.794, -0.003, -0.001)],
    ],
    "PRO": [
        ["N", 0, (-0.566, 1.351, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, 0.000)],
        ["CB", 0, (-0.546, -0.611, -1.293)],
        ["O", 3, (0.621, 1.066, 0.000)],
        ["CG", 4, (0.382, 1.445, 0.0)],
        ["CD", 5, (0.477, 1.424, 0.0)],
    ],
    "SER": [
        ["N", 0, (-0.529, 1.360, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.518, -0.777, -1.211)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["OG", 4, (0.503, 1.325, 0.000)],
    ],
    "THR": [
        ["N", 0, (-0.517, 1.364, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, -0.000)],
        ["CB", 0, (-0.516, -0.793, -1.215)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG2", 4, (0.550, -0.718, -1.228)],
        ["OG1", 4, (0.472, 1.353, 0.000)],
    ],
    "TRP": [
        ["N", 0, (-0.521, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.212)],
        ["O", 3, (0.627, 1.062, 0.000)],
        ["CG", 4, (0.609, 1.370, -0.000)],
        ["CD1", 5, (0.824, 1.091, 0.000)],
        ["CD2", 5, (0.854, -1.148, -0.005)],
        ["CE2", 5, (2.186, -0.678, -0.007)],
        ["CE3", 5, (0.622, -2.530, -0.007)],
        ["NE1", 5, (2.140, 0.690, -0.004)],
        ["CH2", 5, (3.028, -2.890, -0.013)],
        ["CZ2", 5, (3.283, -1.543, -0.011)],
        ["CZ3", 5, (1.715, -3.389, -0.011)],
    ],
    "TYR": [
        ["N", 0, (-0.522, 1.362, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.776, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.382, -0.000)],
        ["CD1", 5, (0.716, 1.195, -0.000)],
        ["CD2", 5, (0.713, -1.194, -0.001)],
        ["CE1", 5, (2.107, 1.200, -0.002)],
        ["CE2", 5, (2.104, -1.201, -0.003)],
        ["OH", 5, (4.168, -0.002, -0.005)],
        ["CZ", 5, (2.791, -0.001, -0.003)],
    ],
    "VAL": [
        ["N", 0, (-0.494, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.533, -0.795, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.540, 1.429, -0.000)],
        ["CG2", 4, (0.533, -0.776, 1.203)],
    ],
}

# Atom14名称映射（每个残基最多14个重原子）
restype_name_to_atom14_names = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
}

# 残基类型顺序（对应索引0-19）
restype_order = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
}

# 每个原子属于哪个rigid group
# 0=backbone, 3=psi, 4=chi1, 5=chi2, 6=chi3, 7=chi4
restype_atom14_to_rigid_group = {
    "ALA": [0, 0, 0, 3, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    "ARG": [0, 0, 0, 3, 0, 4, 5, 6, 7, 7, 7, -1, -1, -1],
    "ASN": [0, 0, 0, 3, 0, 4, 5, 5, -1, -1, -1, -1, -1, -1],
    "ASP": [0, 0, 0, 3, 0, 4, 5, 5, -1, -1, -1, -1, -1, -1],
    "CYS": [0, 0, 0, 3, 0, 4, -1, -1, -1, -1, -1, -1, -1, -1],
    "GLN": [0, 0, 0, 3, 0, 4, 5, 6, 6, -1, -1, -1, -1, -1],
    "GLU": [0, 0, 0, 3, 0, 4, 5, 6, 6, -1, -1, -1, -1, -1],
    "GLY": [0, 0, 0, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    "HIS": [0, 0, 0, 3, 0, 4, 5, 5, 5, 5, -1, -1, -1, -1],
    "ILE": [0, 0, 0, 3, 0, 4, 4, 5, -1, -1, -1, -1, -1, -1],
    "LEU": [0, 0, 0, 3, 0, 4, 5, 5, -1, -1, -1, -1, -1, -1],
    "LYS": [0, 0, 0, 3, 0, 4, 5, 6, 7, -1, -1, -1, -1, -1],
    "MET": [0, 0, 0, 3, 0, 4, 5, 6, -1, -1, -1, -1, -1, -1],
    "PHE": [0, 0, 0, 3, 0, 4, 5, 5, 5, 5, 5, -1, -1, -1],
    "PRO": [0, 0, 0, 3, 0, 4, 5, -1, -1, -1, -1, -1, -1, -1],
    "SER": [0, 0, 0, 3, 0, 4, -1, -1, -1, -1, -1, -1, -1, -1],
    "THR": [0, 0, 0, 3, 0, 4, 4, -1, -1, -1, -1, -1, -1, -1],
    "TRP": [0, 0, 0, 3, 0, 4, 5, 5, 5, 5, 5, 5, 5, 5],
    "TYR": [0, 0, 0, 3, 0, 4, 5, 5, 5, 5, 5, 5, -1, -1],
    "VAL": [0, 0, 0, 3, 0, 4, 4, -1, -1, -1, -1, -1, -1, -1],
}

def build_atom14_constants():
    """
    构建atom14常量张量
    
    Returns:
        {
            'restype_atom14_positions': [21, 14, 3] 每种残基的atom14局部坐标
            'restype_atom14_to_group': [21, 14] 每个atom属于哪个rigid group
            'restype_atom14_mask': [21, 14] 每个atom是否存在
        }
    """
    import numpy as np
    
    restype_atom14_positions = np.zeros([21, 14, 3], dtype=np.float32)
    restype_atom14_to_group = np.zeros([21, 14], dtype=np.int64)
    restype_atom14_mask = np.zeros([21, 14], dtype=np.float32)
    
    # 填充20种氨基酸的数据
    for restype_idx, (restype_letter, resname) in enumerate(restype_1to3.items()):
        atom_list = restype_name_to_atom14_names[resname]
        group_list = restype_atom14_to_rigid_group[resname]
        
        # 从rigid_group_atom_positions查找坐标
        atom_pos_dict = {atom_name: pos for atom_name, _, pos in rigid_group_atom_positions[resname]}
        
        for atom14_idx, (atom_name, group_idx) in enumerate(zip(atom_list, group_list)):
            if atom_name and group_idx >= 0:
                restype_atom14_mask[restype_idx, atom14_idx] = 1.0
                restype_atom14_to_group[restype_idx, atom14_idx] = group_idx
                if atom_name in atom_pos_dict:
                    restype_atom14_positions[restype_idx, atom14_idx] = atom_pos_dict[atom_name]
    
    # UNK残基（索引20）- 全0
    
    return {
        'restype_atom14_positions': restype_atom14_positions,
        'restype_atom14_to_group': restype_atom14_to_group,
        'restype_atom14_mask': restype_atom14_mask,
    }


# ========================================================================
# Chi angle atom definitions (standard AlphaFold2/OpenFold convention)
# Each chi angle is defined by 4 atoms; the torsion rotates around
# the bond between atoms[1] and atoms[2].
# ========================================================================

chi_angles_atoms = {
    "ALA": [],
    "ARG": [["N","CA","CB","CG"], ["CA","CB","CG","CD"],
            ["CB","CG","CD","NE"], ["CG","CD","NE","CZ"]],
    "ASN": [["N","CA","CB","CG"], ["CA","CB","CG","OD1"]],
    "ASP": [["N","CA","CB","CG"], ["CA","CB","CG","OD1"]],
    "CYS": [["N","CA","CB","SG"]],
    "GLN": [["N","CA","CB","CG"], ["CA","CB","CG","CD"],
            ["CB","CG","CD","OE1"]],
    "GLU": [["N","CA","CB","CG"], ["CA","CB","CG","CD"],
            ["CB","CG","CD","OE1"]],
    "GLY": [],
    "HIS": [["N","CA","CB","CG"], ["CA","CB","CG","ND1"]],
    "ILE": [["N","CA","CB","CG1"], ["CA","CB","CG1","CD1"]],
    "LEU": [["N","CA","CB","CG"], ["CA","CB","CG","CD1"]],
    "LYS": [["N","CA","CB","CG"], ["CA","CB","CG","CD"],
            ["CB","CG","CD","CE"], ["CG","CD","CE","NZ"]],
    "MET": [["N","CA","CB","CG"], ["CA","CB","CG","SD"],
            ["CB","CG","SD","CE"]],
    "PHE": [["N","CA","CB","CG"], ["CA","CB","CG","CD1"]],
    "PRO": [["N","CA","CB","CG"], ["CA","CB","CG","CD"]],
    "SER": [["N","CA","CB","OG"]],
    "THR": [["N","CA","CB","OG1"]],
    "TRP": [["N","CA","CB","CG"], ["CA","CB","CG","CD1"]],
    "TYR": [["N","CA","CB","CG"], ["CA","CB","CG","CD1"]],
    "VAL": [["N","CA","CB","CG1"]],
}


def _make_rigid_transformation_4x4(ex, ey, translation):
    """
    Build a [4,4] homogeneous transformation matrix from:
      ex:          unit x-axis direction  (shape [3])
      ey:          vector in the xy-plane (shape [3], will be orthogonalized)
      translation: origin position        (shape [3])

    Returns:
      np.ndarray of shape [4,4], dtype float64 (caller casts to float32).
    """
    # Orthogonalize ey against ex and normalize
    ex = ex / (np.linalg.norm(ex) + 1e-20)
    ey = ey - np.dot(ey, ex) * ex
    ey = ey / (np.linalg.norm(ey) + 1e-20)
    ez = np.cross(ex, ey)

    result = np.eye(4)
    # Rotation columns = basis vectors (each column is a basis vector in
    # the parent frame).  Row-major: result[row, col].
    # Column 0 = ex, Column 1 = ey, Column 2 = ez  =>  R @ [1,0,0] = ex
    result[0, 0] = ex[0];  result[0, 1] = ey[0];  result[0, 2] = ez[0]
    result[1, 0] = ex[1];  result[1, 1] = ey[1];  result[1, 2] = ez[1]
    result[2, 0] = ex[2];  result[2, 1] = ey[2];  result[2, 2] = ez[2]
    result[0, 3] = translation[0]
    result[1, 3] = translation[1]
    result[2, 3] = translation[2]
    return result


def _invert_4x4(T):
    """Invert a [4,4] homogeneous rigid transformation (R|t ; 0 1)."""
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def make_default_frames():
    """
    Build ``restype_rigid_group_default_frame`` -- a [21, 8, 4, 4] float32
    array of homogeneous transformation matrices.

    For each of the 20 standard amino acids and 8 rigid groups (backbone,
    pre-omega, phi, psi, chi1 .. chi4) this gives the default (zero-torsion)
    frame of each group expressed *relative to its parent group*.

    Parent chain:  0->backbone  1->0  2->0  3->0  4->0  5->4  6->5  7->6

    The algorithm builds backbone-frame atom positions iteratively:
      1. Group-0 atoms are already in backbone frame.
      2. For each chi group g (4 .. 7), the default frame is computed from
         the three defining atoms (atoms[0], atoms[1], atoms[2] of the
         corresponding chi angle).  The frame has:
           - origin at atoms[2]
           - x-axis along atoms[1] -> atoms[2]  (the torsion bond axis)
           - y-axis in the atoms[0]-atoms[1]-atoms[2] plane
         After building group g's frame, the positions of atoms belonging
         to group g are transformed from their local coords into the
         backbone frame, making them available for group g+1.
      3. Frames for groups 5-7 are re-expressed relative to the parent
         group via: frame_rel = inv(parent_frame) @ child_frame.
    """
    FRAME_PARENT = [0, 0, 0, 0, 0, 4, 5, 6]

    result = np.zeros([21, 8, 4, 4], dtype=np.float64)
    # Initialize all to identity
    for i in range(21):
        for g in range(8):
            result[i, g] = np.eye(4)

    for restype_letter in restypes:
        resname = restype_1to3[restype_letter]
        aa_idx = restype_order[restype_letter]

        # -- Collect group-0 (backbone) atom positions in backbone frame --
        # These are the "seed" positions we know directly.
        atom_pos_local = {}   # atom_name -> (group_idx, np.array position in that group's frame)
        atom_pos_bb = {}      # atom_name -> np.array position in backbone frame

        for atom_name, grp, pos in rigid_group_atom_positions[resname]:
            p = np.array(pos, dtype=np.float64)
            atom_pos_local[atom_name] = (grp, p)
            if grp == 0:
                atom_pos_bb[atom_name] = p

        # Convenience: get backbone atom positions (with safe defaults)
        def _get_bb(name, default=None):
            if name in atom_pos_bb:
                return atom_pos_bb[name]
            if default is not None:
                return default
            return np.zeros(3, dtype=np.float64)

        pos_N  = _get_bb("N",  np.array([-0.525, 1.363, 0.0]))
        pos_CA = _get_bb("CA", np.array([ 0.000, 0.000, 0.0]))
        pos_C  = _get_bb("C",  np.array([ 1.526, 0.000, 0.0]))

        # ---- Group 0 (backbone): identity ----
        # Already set to identity above.

        # ---- Group 1 (pre-omega): identity ----
        # Already identity.

        # ---- Group 2 (phi): rotation around N-CA bond, origin at N ----
        ex = pos_CA - pos_N
        ey = pos_C - pos_N
        result[aa_idx, 2] = _make_rigid_transformation_4x4(ex, ey, pos_N)

        # ---- Group 3 (psi): rotation around CA-C bond, origin at C ----
        ex = pos_C - pos_CA
        ey = pos_N - pos_CA
        result[aa_idx, 3] = _make_rigid_transformation_4x4(ex, ey, pos_C)

        # ---- Groups 4-7 (chi1 .. chi4) ----
        chi_atoms = chi_angles_atoms.get(resname, [])
        chi_mask_this = chi_angles_mask[aa_idx]

        # We will accumulate absolute (backbone-frame) default frames for
        # each chi group, so we can compute relative frames for child groups.
        abs_frames = {}   # group_idx -> 4x4 matrix in backbone frame

        for chi_idx in range(4):
            group_idx = chi_idx + 4  # groups 4,5,6,7

            if chi_idx >= len(chi_atoms) or chi_mask_this[chi_idx] < 0.5:
                # No chi angle for this group -- leave as identity
                abs_frames[group_idx] = np.eye(4)
                continue

            a0_name, a1_name, a2_name, _a3_name = chi_atoms[chi_idx]

            # We need a0, a1, a2 positions in backbone frame.
            # For chi1: all three (N, CA, CB) are group-0 atoms => already known.
            # For chi2+: the first atom of the chi definition that's in a higher
            #   group will have been transformed to backbone frame in a
            #   previous iteration.
            #
            # Strategy: for each atom, check if we already have it in atom_pos_bb.
            # If not, it's in a group we already processed -- we must have
            # added it via the frame composition below.

            # Resolve positions -- atoms not yet in atom_pos_bb are a problem
            # we handle by transforming them from their local frame using
            # the already-computed absolute default frame.
            for aname in [a0_name, a1_name, a2_name]:
                if aname not in atom_pos_bb:
                    grp_of_atom, local_p = atom_pos_local[aname]
                    if grp_of_atom in abs_frames:
                        T = abs_frames[grp_of_atom]
                        p_homo = np.array([local_p[0], local_p[1], local_p[2], 1.0])
                        p_bb = (T @ p_homo)[:3]
                        atom_pos_bb[aname] = p_bb

            p0 = atom_pos_bb.get(a0_name, np.zeros(3))
            p1 = atom_pos_bb.get(a1_name, np.zeros(3))
            p2 = atom_pos_bb.get(a2_name, np.zeros(3))

            # Build the absolute (backbone-frame) default frame for this group:
            #   origin at p2 (the "child" pivot atom)
            #   x-axis along p1 -> p2  (the torsion rotation axis)
            #   y  in the p0-p1-p2 plane
            ex = p2 - p1
            ey = p0 - p1
            T_abs = _make_rigid_transformation_4x4(ex, ey, p2)
            abs_frames[group_idx] = T_abs

            # Now transform all atoms in this group to backbone frame
            # so they're available for subsequent chi definitions.
            for atom_name, grp, pos in rigid_group_atom_positions[resname]:
                if grp == group_idx and atom_name not in atom_pos_bb:
                    p_local = np.array(pos, dtype=np.float64)
                    p_homo = np.array([p_local[0], p_local[1], p_local[2], 1.0])
                    p_bb = (T_abs @ p_homo)[:3]
                    atom_pos_bb[atom_name] = p_bb

            # Store the frame: for groups 4 this is relative to backbone
            # (which is identity, so abs == rel).  For groups 5-7 we need
            # to re-express relative to the parent group's default frame.
            parent_grp = FRAME_PARENT[group_idx]
            if parent_grp == 0:
                result[aa_idx, group_idx] = T_abs.astype(np.float64)
            else:
                T_parent = abs_frames.get(parent_grp, np.eye(4))
                T_rel = _invert_4x4(T_parent) @ T_abs
                result[aa_idx, group_idx] = T_rel.astype(np.float64)

    # Index 20 = UNK => all identity (already set).
    return result.astype(np.float32)


# Module-level constant: computed once at import time.
restype_rigid_group_default_frame = make_default_frames()

