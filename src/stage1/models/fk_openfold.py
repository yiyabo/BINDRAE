"""
OpenFold式FK实现（使用FlashIPA的Rigid）

完整复刻OpenFold的torsion→frames→atom14方法
使用预计算的 default frames 确保扭转旋转轴正确对齐

Author: BINDRAE Team
Date: 2025-10-28
Updated: 2026-04-06  -- 修复chi旋转轴对齐bug，引入default_frames
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

# 项目路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# FlashIPA路径 (项目内 vendor 目录)
flash_ipa_path = str(project_root / 'vendor' / 'flash_ipa' / 'src')
if os.path.exists(flash_ipa_path) and flash_ipa_path not in sys.path:
    sys.path.insert(0, flash_ipa_path)

from flash_ipa.rigid import Rigid, Rotation

from src.stage1.data.residue_constants import (
    rigid_group_atom_positions,
    restype_1to3,
    restype_3to1,
    restype_order,
    restype_atom14_to_rigid_group,
    chi_angles_mask,
    restype_rigid_group_default_frame,
)


def reorder_torsions_to_openfold(torsions_sincos: torch.Tensor) -> torch.Tensor:
    """
    Reorder stored torsions from (phi, psi, omega, chi1..4)
    to the OpenFold FK convention (omega, phi, psi, chi1..4).
    """
    if torsions_sincos.shape[-2:] != (7, 2):
        raise ValueError(
            f"Expected torsions_sincos shape [..., 7, 2], got {tuple(torsions_sincos.shape)}"
        )

    return torch.cat(
        [
            torsions_sincos[..., 2:3, :],
            torsions_sincos[..., 0:2, :],
            torsions_sincos[..., 3:, :],
        ],
        dim=-2,
    )


class OpenFoldFK(nn.Module):
    """
    OpenFold式FK（完整实现，含default frame旋转对齐）

    流程：
    1. 扭转角(sin,cos) → 8个刚体帧旋转（绕x轴）
    2. default_frame.compose(torsion_rotation) → 局部帧
    3. parent_frame.compose(local_frame) → 全局帧
    4. frame.apply(文献坐标) → 全局原子坐标

    Frame 层次链（OpenFold 标准）：
    - Frame 0 (backbone): 单位帧，相对于 backbone_rigids
    - Frame 1 (pre-omega): 单位帧
    - Frame 2 (phi): 旋转轴=N→CA，原点=N，相对于 Frame 0
    - Frame 3 (psi): 旋转轴=CA→C，原点=C，相对于 Frame 0
    - Frame 4 (chi1): 旋转轴=CA→CB，原点=CB，相对于 Frame 0
    - Frame 5 (chi2): 相对于 Frame 4
    - Frame 6 (chi3): 相对于 Frame 5
    - Frame 7 (chi4): 相对于 Frame 6

    default_frames [21, 8, 4, 4] 编码了每个group在其parent坐标系下的
    默认（零扭转角）变换，包含正确的旋转对齐（x轴=扭转键方向）。
    扭转角旋转在default frame之后compose，确保绕正确的键轴旋转。
    """

    # Frame 的 parent chain：每个 frame 相对于哪个 frame 定义
    FRAME_PARENT = [0, 0, 0, 0, 0, 4, 5, 6]

    def __init__(self):
        super().__init__()

        # 构建残基常量（20种氨基酸的atom14数据）
        self._build_residue_constants()

        # 注册 default frames buffer [21, 8, 4, 4]
        self.register_buffer(
            'default_frames',
            torch.from_numpy(restype_rigid_group_default_frame)
        )

        print("✓ OpenFoldFK初始化:")
        print(f"  - 残基类型: 20")
        print(f"  - Atom14编码")
        print(f"  - Frame 层次链: 0→backbone, 5→Frame4, 6→Frame5, 7→Frame6")
        print(f"  - 使用FlashIPA Rigid + default_frames旋转对齐")

    def _build_residue_constants(self):
        """构建残基常量张量"""
        from src.stage1.data.residue_constants import build_atom14_constants

        # 构建atom14常量
        constants = build_atom14_constants()

        # 注册为buffer（可在GPU上使用）
        self.register_buffer('restype_atom14_positions',
                           torch.from_numpy(constants['restype_atom14_positions']))
        self.register_buffer('restype_atom14_to_group',
                           torch.from_numpy(constants['restype_atom14_to_group']))
        self.register_buffer('restype_atom14_mask',
                           torch.from_numpy(constants['restype_atom14_mask']))

    @staticmethod
    def _frame_from_4x4(T44):
        """
        Extract a Rigid from a [..., 4, 4] homogeneous matrix tensor.

        Args:
            T44: [..., 4, 4] float tensor
        Returns:
            Rigid with rotation [..., 3, 3] and translation [..., 3]
        """
        R = T44[..., :3, :3]   # [..., 3, 3]
        t = T44[..., :3, 3]    # [..., 3]
        return Rigid(rots=Rotation(rot_mats=R), trans=t)

    def torsion_angles_to_frames(self,
                                torsions_sincos: torch.Tensor,
                                backbone_rigids: Rigid,
                                aatype: torch.Tensor) -> List[Rigid]:
        """
        From torsion angles generate 8 rigid frames following OpenFold hierarchy.

        Uses precomputed default_frames to correctly orient the torsion
        rotation axis for each rigid group before applying the torsion angle.

        Args:
            torsions_sincos: [B, N, 7, 2]  (omega, phi, psi, chi1..chi4)
            backbone_rigids: Rigid [B, N]
            aatype: [B, N] residue type indices (0-19, or 20 for UNK)

        Returns:
            all_frames: List[Rigid] length 8, each [B, N]
        """
        B, N = torsions_sincos.shape[:2]
        device = torsions_sincos.device

        # --- Build torsion rotation matrices (rotation around x-axis) ---
        # torsions_sincos: [B, N, 7, 2]  indices 0..6 = omega, phi, psi, chi1-4
        sin_angles = torsions_sincos[..., 0]  # [B, N, 7]
        cos_angles = torsions_sincos[..., 1]  # [B, N, 7]

        # Prepend group 0 (identity: sin=0, cos=1)
        sin_all = torch.cat([
            torch.zeros(B, N, 1, device=device),
            sin_angles
        ], dim=-1)  # [B, N, 8]

        cos_all = torch.cat([
            torch.ones(B, N, 1, device=device),
            cos_angles
        ], dim=-1)  # [B, N, 8]

        zeros = torch.zeros_like(sin_all)
        ones = torch.ones_like(sin_all)

        # rot_mats[..., g, :, :] = Rx(angle_g)
        # Rotation around x-axis:
        # [[1,   0,      0    ],
        #  [0,  cos,   -sin   ],
        #  [0,  sin,    cos   ]]
        rot_mats = torch.stack([
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cos_all, -sin_all], dim=-1),
            torch.stack([zeros, sin_all, cos_all], dim=-1),
        ], dim=-2)  # [B, N, 8, 3, 3]

        # --- Gather per-residue default frames ---
        # aatype: [B, N] -> clamp to valid range [0, 20]
        aa_clamped = aatype.clamp(0, 20)
        # default_frames: [21, 8, 4, 4]
        # per_res_defaults: [B, N, 8, 4, 4]
        per_res_defaults = self.default_frames[aa_clamped]

        # --- Compose frames ---
        all_frames = []

        for group_idx in range(8):
            # Torsion rotation for this group: [B, N, 3, 3]
            torsion_rot = rot_mats[:, :, group_idx, :, :]
            torsion_rigid = Rigid(
                rots=Rotation(rot_mats=torsion_rot),
                trans=torch.zeros(B, N, 3, device=device)
            )

            # Default frame for this group (relative to parent): [B, N, 4, 4]
            default_4x4 = per_res_defaults[:, :, group_idx]
            default_rigid = self._frame_from_4x4(default_4x4)

            # local_frame = default_frame.compose(torsion_rotation)
            # This first orients the frame so x-axis aligns with the bond,
            # then applies the torsion rotation around that axis.
            local_frame = default_rigid.compose(torsion_rigid)

            # Apply parent transform to get global frame
            parent_idx = self.FRAME_PARENT[group_idx]
            if parent_idx == 0 or group_idx <= 4:
                # Groups 0-4: parent is backbone
                frame_to_global = backbone_rigids.compose(local_frame)
            else:
                # Groups 5-7: parent is the preceding chi frame
                parent_frame = all_frames[parent_idx]
                frame_to_global = parent_frame.compose(local_frame)

            all_frames.append(frame_to_global)

        return all_frames

    def frames_to_atom14_pos(self,
                            all_frames: List[Rigid],
                            aatype: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        从8个刚体帧生成atom14坐标（完整实现，含侧链）

        OpenFold方法：
        1. 根据残基类型查询atom14文献坐标
        2. 根据atom→group映射，应用对应的刚体帧
        3. 得到全局坐标
        """
        B, N = aatype.shape
        device = aatype.device

        # 获取每个残基的atom14数据 [B, N, 14, 3/1]
        # 使用aatype索引
        lit_positions = self.restype_atom14_positions[aatype]  # [B, N, 14, 3]
        atom_to_group = self.restype_atom14_to_group[aatype]  # [B, N, 14]
        atom_mask = self.restype_atom14_mask[aatype]  # [B, N, 14]

        # 初始化全局坐标
        atom14_pos = torch.zeros(B, N, 14, 3, device=device)

        # 对每个atom，应用其对应的刚体帧
        for atom_idx in range(14):
            # 该atom在不同残基中属于哪个group
            groups = atom_to_group[:, :, atom_idx]  # [B, N]

            # 文献坐标
            lit_pos = lit_positions[:, :, atom_idx, :]  # [B, N, 3]

            # 对每个group应用对应的帧
            # 简化：逐group处理
            for group_idx in range(8):
                # 找到属于该group的atom
                mask = (groups == group_idx)  # [B, N]

                if mask.any():
                    # 应用该group的帧
                    frame = all_frames[group_idx]

                    # 变换坐标
                    global_pos = frame.apply(lit_pos)  # [B, N, 3]

                    # 只更新属于该group的atom
                    atom14_pos[:, :, atom_idx, :] = torch.where(
                        mask.unsqueeze(-1),
                        global_pos,
                        atom14_pos[:, :, atom_idx, :]
                    )

        return {
            'atom14_pos': atom14_pos,
            'atom14_mask': atom_mask,
        }

    def forward(self,
                torsions_sincos: torch.Tensor,
                backbone_rigids: Rigid,
                aatype: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        OpenFold式FK前向传播

        Args:
            torsions_sincos: [B, N, 7, 2] 扭转角(sin,cos)
            backbone_rigids: Rigid对象 [B, N] 主链帧
            aatype: [B, N] 残基类型索引(0-19)

        Returns:
            {
                'atom14_pos': [B, N, 14, 3] Atom14坐标
                'atom14_mask': [B, N, 14] Atom14掩码
            }
        """
        # 步骤1: torsion → 8个frames
        all_frames = self.torsion_angles_to_frames(torsions_sincos, backbone_rigids, aatype)

        # 步骤2: frames → atom14坐标
        result = self.frames_to_atom14_pos(all_frames, aatype)

        return result


def create_openfold_fk() -> OpenFoldFK:
    """创建OpenFoldFK模块"""
    return OpenFoldFK()
