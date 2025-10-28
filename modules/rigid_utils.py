"""
刚体帧操作工具

功能：
1. Rigid 类：表示 3D 刚体变换 (旋转 + 平移)
2. 从 3 点构建局部帧 (N-Cα-C)
3. 刚体组合、逆变换
4. 坐标变换
5. 刚体噪声生成

设计参考：
- AlphaFold2 的 Rigid 类
- SE(3) 群操作
- 用于 IPA 的局部帧构建
"""

import numpy as np
from typing import Tuple, Optional, Union
import warnings

# ============================================================================
# 常量定义
# ============================================================================

# 数值稳定性
EPS = 1e-8

# 噪声参数（用于数据增强）
DEFAULT_ROTATION_NOISE_STD = 0.1  # 旋转噪声标准差 (弧度)
DEFAULT_TRANSLATION_NOISE_STD = 0.5  # 平移噪声标准差 (Å)


# ============================================================================
# Rigid 类
# ============================================================================

class Rigid:
    """
    3D 刚体变换类
    
    表示 SE(3) 群中的元素，由旋转矩阵 R (3x3) 和平移向量 t (3,) 组成
    变换作用：x' = R @ x + t
    
    Attributes:
        rotation: (3, 3) 旋转矩阵
        translation: (3,) 平移向量
    """
    
    def __init__(self, rotation: np.ndarray, translation: np.ndarray):
        """
        Args:
            rotation: (3, 3) 旋转矩阵
            translation: (3,) 平移向量
        """
        assert rotation.shape == (3, 3), f"Rotation must be (3,3), got {rotation.shape}"
        assert translation.shape == (3,), f"Translation must be (3,), got {translation.shape}"
        
        self.rotation = rotation.astype(np.float32)
        self.translation = translation.astype(np.float32)
    
    @classmethod
    def identity(cls) -> 'Rigid':
        """创建单位刚体变换"""
        return cls(
            rotation=np.eye(3, dtype=np.float32),
            translation=np.zeros(3, dtype=np.float32)
        )
    
    @classmethod
    def from_3_points(cls, 
                     p1: np.ndarray, 
                     p2: np.ndarray, 
                     p3: np.ndarray) -> 'Rigid':
        """
        从 3 个点构建局部坐标系
        
        用途：蛋白质主链局部帧 (N, Cα, C)
        - 原点：p2 (Cα)
        - X 轴：p2 → p3 (Cα → C)
        - Y 轴：垂直于 (p1-p2) 和 (p3-p2) 平面
        - Z 轴：X × Y
        
        Args:
            p1: (3,) 第一个点 (N)
            p2: (3,) 第二个点 (Cα, 原点)
            p3: (3,) 第三个点 (C)
            
        Returns:
            Rigid 对象
        """
        # 原点是 p2
        origin = p2.astype(np.float32)
        
        # 构建局部坐标系
        # X 轴：p2 → p3
        v1 = p3 - p2
        e1 = v1 / (np.linalg.norm(v1) + EPS)
        
        # 临时向量：p2 → p1
        v2 = p1 - p2
        
        # Y 轴：垂直于 v1 和 v2
        e2 = v2 - np.dot(v2, e1) * e1  # Gram-Schmidt 正交化
        e2_norm = np.linalg.norm(e2)
        
        # 处理三点共线的边界情况
        if e2_norm < 1e-6:
            # 三点接近共线，使用备用方案：找一个与e1垂直的向量
            if abs(e1[2]) < 0.9:
                e2 = np.cross(e1, np.array([0, 0, 1]))
            else:
                e2 = np.cross(e1, np.array([1, 0, 0]))
            e2 = e2 / (np.linalg.norm(e2) + EPS)
        else:
            e2 = e2 / e2_norm
        
        # Z 轴：X × Y
        e3 = np.cross(e1, e2)
        
        # 旋转矩阵：列向量是局部坐标系的基
        rotation = np.stack([e1, e2, e3], axis=1).astype(np.float32)
        
        return cls(rotation=rotation, translation=origin)
    
    def apply(self, coords: np.ndarray) -> np.ndarray:
        """
        应用刚体变换到坐标
        
        Args:
            coords: (..., 3) 坐标
            
        Returns:
            transformed: (..., 3) 变换后的坐标
        """
        # x' = R @ x + t
        return coords @ self.rotation.T + self.translation
    
    def invert_apply(self, coords: np.ndarray) -> np.ndarray:
        """
        应用逆刚体变换到坐标（世界坐标 → 局部坐标）
        
        Args:
            coords: (..., 3) 世界坐标
            
        Returns:
            local_coords: (..., 3) 局部坐标
        """
        # x_local = R^T @ (x - t)
        return (coords - self.translation) @ self.rotation
    
    def compose(self, other: 'Rigid') -> 'Rigid':
        """
        组合两个刚体变换：self ∘ other
        
        结果变换先应用 other，再应用 self
        
        Args:
            other: 另一个刚体变换
            
        Returns:
            组合后的刚体变换
        """
        # R_new = R_self @ R_other
        # t_new = R_self @ t_other + t_self
        new_rotation = self.rotation @ other.rotation
        new_translation = self.rotation @ other.translation + self.translation
        return Rigid(new_rotation, new_translation)
    
    def inverse(self) -> 'Rigid':
        """
        计算逆刚体变换
        
        Returns:
            逆变换
        """
        # R_inv = R^T
        # t_inv = -R^T @ t
        inv_rotation = self.rotation.T
        inv_translation = -inv_rotation @ self.translation
        return Rigid(inv_rotation, inv_translation)
    
    def to_tensor_7(self) -> np.ndarray:
        """
        转换为 7 维表示 [qw, qx, qy, qz, tx, ty, tz]
        
        用于神经网络输入/输出
        
        Returns:
            (7,) 四元数 + 平移
        """
        quat = rotation_matrix_to_quaternion(self.rotation)
        return np.concatenate([quat, self.translation])
    
    @classmethod
    def from_tensor_7(cls, tensor: np.ndarray) -> 'Rigid':
        """
        从 7 维表示创建 Rigid
        
        Args:
            tensor: (7,) [qw, qx, qy, qz, tx, ty, tz]
            
        Returns:
            Rigid 对象
        """
        quat = tensor[:4]
        translation = tensor[4:]
        rotation = quaternion_to_rotation_matrix(quat)
        return cls(rotation, translation)
    
    def __repr__(self) -> str:
        return f"Rigid(rotation=\n{self.rotation},\ntranslation={self.translation})"


# ============================================================================
# 旋转矩阵 ↔ 四元数转换
# ============================================================================

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    旋转矩阵 → 四元数 [w, x, y, z]
    
    Args:
        R: (3, 3) 旋转矩阵
        
    Returns:
        quat: (4,) 四元数
    """
    # Shepperd's method (数值稳定)
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    quat = np.array([w, x, y, z], dtype=np.float32)
    return quat / (np.linalg.norm(quat) + EPS)


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    四元数 [w, x, y, z] → 旋转矩阵
    
    Args:
        quat: (4,) 四元数
        
    Returns:
        R: (3, 3) 旋转矩阵
    """
    # 归一化
    quat = quat / (np.linalg.norm(quat) + EPS)
    w, x, y, z = quat
    
    # 转换公式
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float32)
    
    return R


# ============================================================================
# 刚体噪声生成
# ============================================================================

def random_rotation_matrix(std: float = DEFAULT_ROTATION_NOISE_STD) -> np.ndarray:
    """
    生成随机旋转矩阵（小角度噪声）
    
    策略：从轴角表示采样
    - 随机旋转轴（单位球面均匀采样）
    - 旋转角度 ~ N(0, std^2)
    
    Args:
        std: 旋转角度标准差 (弧度)
        
    Returns:
        R: (3, 3) 旋转矩阵
    """
    # 随机旋转轴
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + EPS)
    
    # 旋转角度
    angle = np.random.randn() * std
    
    # 轴角 → 旋转矩阵 (Rodrigues' formula)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
    return R.astype(np.float32)


def add_rigid_noise(rigid: Rigid,
                   rotation_std: float = DEFAULT_ROTATION_NOISE_STD,
                   translation_std: float = DEFAULT_TRANSLATION_NOISE_STD) -> Rigid:
    """
    为刚体变换添加噪声（数据增强）
    
    Args:
        rigid: 原始刚体变换
        rotation_std: 旋转噪声标准差 (弧度)
        translation_std: 平移噪声标准差 (Å)
        
    Returns:
        带噪声的刚体变换
    """
    # 旋转噪声
    noise_R = random_rotation_matrix(rotation_std)
    new_rotation = noise_R @ rigid.rotation
    
    # 平移噪声
    noise_t = np.random.randn(3) * translation_std
    new_translation = rigid.translation + noise_t
    
    return Rigid(new_rotation, new_translation)


# ============================================================================
# 批量操作（用于 DataLoader）
# ============================================================================

class RigidBatch:
    """
    批量刚体变换
    
    用于高效处理一批刚体变换
    
    Attributes:
        rotation: (B, 3, 3) 旋转矩阵
        translation: (B, 3) 平移向量
    """
    
    def __init__(self, rotation: np.ndarray, translation: np.ndarray):
        """
        Args:
            rotation: (B, 3, 3) 旋转矩阵
            translation: (B, 3) 平移向量
        """
        assert rotation.ndim == 3 and rotation.shape[-2:] == (3, 3)
        assert translation.ndim == 2 and translation.shape[-1] == 3
        assert rotation.shape[0] == translation.shape[0]
        
        self.rotation = rotation.astype(np.float32)
        self.translation = translation.astype(np.float32)
    
    @classmethod
    def from_list(cls, rigids: list) -> 'RigidBatch':
        """从 Rigid 列表创建批量"""
        rotations = np.stack([r.rotation for r in rigids])
        translations = np.stack([r.translation for r in rigids])
        return cls(rotations, translations)
    
    def apply(self, coords: np.ndarray) -> np.ndarray:
        """
        批量应用刚体变换
        
        Args:
            coords: (B, N, 3) 坐标
            
        Returns:
            transformed: (B, N, 3) 变换后的坐标
        """
        # x' = x @ R^T + t
        return np.einsum('bni,bji->bnj', coords, self.rotation) + self.translation[:, None, :]
    
    def invert_apply(self, coords: np.ndarray) -> np.ndarray:
        """
        批量应用逆变换
        
        Args:
            coords: (B, N, 3) 世界坐标
            
        Returns:
            local_coords: (B, N, 3) 局部坐标
        """
        # x_local = (x - t) @ R
        return np.einsum('bni,bji->bnj', coords - self.translation[:, None, :], 
                        self.rotation.transpose(0, 2, 1))
    
    def __len__(self) -> int:
        return len(self.rotation)
    
    def __getitem__(self, idx: int) -> Rigid:
        """获取单个刚体变换"""
        return Rigid(self.rotation[idx], self.translation[idx])


# ============================================================================
# 辅助函数
# ============================================================================

def build_backbone_frames(ca_coords: np.ndarray,
                         n_coords: np.ndarray,
                         c_coords: np.ndarray) -> RigidBatch:
    """
    构建蛋白质主链局部帧
    
    Args:
        ca_coords: (N, 3) Cα 坐标
        n_coords: (N, 3) N 坐标
        c_coords: (N, 3) C 坐标
        
    Returns:
        RigidBatch: (N,) 局部帧
    """
    n_residues = len(ca_coords)
    rigids = []
    
    for i in range(n_residues):
        try:
            rigid = Rigid.from_3_points(n_coords[i], ca_coords[i], c_coords[i])
            rigids.append(rigid)
        except Exception as e:
            warnings.warn(f"Failed to build frame for residue {i}: {e}")
            # 使用单位帧
            rigids.append(Rigid.identity())
    
    return RigidBatch.from_list(rigids)


def clip_rotation_matrix(R: np.ndarray, max_angle: float = np.pi / 6) -> np.ndarray:
    """
    裁剪旋转矩阵的角度（防止过大的旋转）
    
    Args:
        R: (3, 3) 旋转矩阵
        max_angle: 最大允许角度 (弧度)
        
    Returns:
        裁剪后的旋转矩阵
    """
    # 转换为四元数
    quat = rotation_matrix_to_quaternion(R)
    w = quat[0]
    
    # 计算旋转角度：θ = 2 * arccos(w)
    angle = 2 * np.arccos(np.clip(w, -1, 1))
    
    # 如果超过最大角度，缩放
    if angle > max_angle:
        scale = max_angle / (angle + EPS)
        # 调整四元数
        xyz = quat[1:]
        new_w = np.cos(max_angle / 2)
        new_xyz = xyz / (np.linalg.norm(xyz) + EPS) * np.sin(max_angle / 2)
        quat = np.concatenate([[new_w], new_xyz])
    
    return quaternion_to_rotation_matrix(quat)
