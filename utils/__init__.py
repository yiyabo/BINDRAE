"""
BINDRAE 工具模块
"""

# 配体工具
from .ligand_utils import (
    LigandTokenBuilder,
    build_ligand_tokens_from_file,
    encode_ligand_batch,
    ATOM_TYPE_MAPPING,
    MAX_LIGAND_TOKENS,
)

# 刚体工具
from .rigid_utils import (
    Rigid,
    RigidBatch,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    random_rotation_matrix,
    add_rigid_noise,
    build_backbone_frames,
    clip_rotation_matrix,
)

# 评估指标
from .metrics import (
    compute_pocket_irmsd,
    compute_chi1_accuracy,
    compute_clash_percentage,
    compute_fape,
    kabsch_align,
    compute_rmsd,
)

__all__ = [
    # 配体工具
    'LigandTokenBuilder',
    'build_ligand_tokens_from_file',
    'encode_ligand_batch',
    'ATOM_TYPE_MAPPING',
    'MAX_LIGAND_TOKENS',
    # 刚体工具
    'Rigid',
    'RigidBatch',
    'rotation_matrix_to_quaternion',
    'quaternion_to_rotation_matrix',
    'random_rotation_matrix',
    'add_rigid_noise',
    'build_backbone_frames',
    'clip_rotation_matrix',
    # 评估指标
    'compute_pocket_irmsd',
    'compute_chi1_accuracy',
    'compute_clash_percentage',
    'compute_fape',
    'kabsch_align',
    'compute_rmsd',
]
