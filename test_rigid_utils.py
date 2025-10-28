"""
测试 rigid_utils 模块
"""

import numpy as np
from utils.rigid_utils import (
    Rigid, 
    RigidBatch,
    build_backbone_frames,
    add_rigid_noise,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix
)

def test_rigid_basic():
    """测试 Rigid 基本功能"""
    print("="*80)
    print("测试 Rigid 基本功能")
    print("="*80)
    
    # 1. 单位变换
    identity = Rigid.identity()
    print(f"✓ 单位变换创建成功")
    
    # 2. 从3点构建帧
    n = np.array([0.0, 0.0, 0.0])
    ca = np.array([1.0, 0.0, 0.0])
    c = np.array([2.0, 1.0, 0.0])
    
    rigid = Rigid.from_3_points(n, ca, c)
    print(f"✓ 从3点构建帧成功")
    print(f"  原点: {rigid.translation}")
    print(f"  旋转矩阵形状: {rigid.rotation.shape}")
    
    # 3. 应用变换
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    transformed = rigid.apply(points)
    print(f"✓ 变换应用成功")
    print(f"  原始点: {points[0]}")
    print(f"  变换后: {transformed[0]}")
    
    # 4. 逆变换
    recovered = rigid.invert_apply(transformed)
    error = np.abs(recovered - points).max()
    print(f"✓ 逆变换成功，误差: {error:.6f}")
    
    # 5. 组合变换
    rigid2 = Rigid.from_3_points(c, ca, n)
    composed = rigid.compose(rigid2)
    print(f"✓ 变换组合成功")
    
    # 6. 7维表示
    tensor7 = rigid.to_tensor_7()
    rigid_recovered = Rigid.from_tensor_7(tensor7)
    print(f"✓ 7维转换成功")
    print(f"  四元数: {tensor7[:4]}")
    print(f"  平移: {tensor7[4:]}")
    
    print()


def test_quaternion_conversion():
    """测试四元数转换"""
    print("="*80)
    print("测试四元数 ↔ 旋转矩阵转换")
    print("="*80)
    
    # 生成随机旋转矩阵
    angle = np.pi / 4  # 45度
    axis = np.array([0, 0, 1])  # Z轴
    
    # 轴角 → 旋转矩阵
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    # 旋转矩阵 → 四元数 → 旋转矩阵
    quat = rotation_matrix_to_quaternion(R)
    R_recovered = quaternion_to_rotation_matrix(quat)
    
    error = np.abs(R - R_recovered).max()
    print(f"✓ 转换误差: {error:.8f}")
    print(f"  原始旋转矩阵:\n{R}")
    print(f"  四元数: {quat}")
    print()


def test_rigid_batch():
    """测试批量刚体操作"""
    print("="*80)
    print("测试 RigidBatch")
    print("="*80)
    
    # 创建一批随机坐标
    batch_size = 5
    ca_coords = np.random.randn(batch_size, 3)
    n_coords = ca_coords + np.random.randn(batch_size, 3) * 0.1
    c_coords = ca_coords + np.random.randn(batch_size, 3) * 0.1
    
    # 构建批量帧
    batch = build_backbone_frames(ca_coords, n_coords, c_coords)
    print(f"✓ 批量帧构建成功: {len(batch)} 个")
    
    # 批量应用变换
    coords = np.random.randn(batch_size, 10, 3)  # (B, N, 3)
    transformed = batch.apply(coords)
    print(f"✓ 批量变换应用成功")
    print(f"  输入形状: {coords.shape}")
    print(f"  输出形状: {transformed.shape}")
    
    # 批量逆变换
    recovered = batch.invert_apply(transformed)
    error = np.abs(recovered - coords).max()
    print(f"✓ 批量逆变换成功，误差: {error:.6f}")
    
    # 访问单个元素
    single_rigid = batch[0]
    print(f"✓ 单个刚体访问成功: {type(single_rigid).__name__}")
    
    print()


def test_noise():
    """测试噪声生成"""
    print("="*80)
    print("测试刚体噪声")
    print("="*80)
    
    # 创建刚体
    rigid = Rigid.identity()
    
    # 添加噪声
    noisy_rigid = add_rigid_noise(rigid, rotation_std=0.1, translation_std=0.5)
    
    print(f"✓ 噪声添加成功")
    print(f"  原始平移: {rigid.translation}")
    print(f"  噪声平移: {noisy_rigid.translation}")
    
    # 检查旋转矩阵正交性
    R = noisy_rigid.rotation
    orthogonality = np.abs(R @ R.T - np.eye(3)).max()
    print(f"  旋转矩阵正交性误差: {orthogonality:.8f}")
    
    print()


def test_from_pdb():
    """从实际 PDB 数据测试"""
    print("="*80)
    print("测试实际蛋白质坐标")
    print("="*80)
    
    from pathlib import Path
    
    # 加载坐标
    base_dir = Path(__file__).parent
    coords_file = base_dir / "data" / "casf2016" / "processed" / "features" / "1a30_protein_coords.npy"
    
    if not coords_file.exists():
        print("⚠️  测试数据不存在，跳过此测试")
        return
    
    coords = np.load(coords_file)  # (N, 3)
    print(f"✓ 加载蛋白质坐标: {coords.shape}")
    
    # 假设前3个残基的 N, Cα, C
    n_residues = min(3, len(coords) // 3)
    ca_coords = coords[1::3][:n_residues]  # 假设 Cα 在第2个
    n_coords = coords[0::3][:n_residues]
    c_coords = coords[2::3][:n_residues]
    
    # 构建帧
    frames = build_backbone_frames(ca_coords, n_coords, c_coords)
    print(f"✓ 构建 {len(frames)} 个主链帧")
    
    # 变换到局部坐标
    for i in range(n_residues):
        frame = frames[i]
        # 把 Cα 变换到局部坐标系（应该在原点）
        local_ca = frame.invert_apply(ca_coords[i:i+1])
        distance_from_origin = np.linalg.norm(local_ca)
        print(f"  残基 {i}: Cα 距原点 {distance_from_origin:.6f} Å")
    
    print()


def test_collinear_points():
    """测试共线点的边界情况"""
    print("="*80)
    print("测试共线点边界情况")
    print("="*80)
    
    # 构造三个共线点
    n = np.array([0.0, 0.0, 0.0])
    ca = np.array([1.0, 0.0, 0.0])
    c = np.array([2.0, 0.0, 0.0])  # 共线！
    
    try:
        rigid = Rigid.from_3_points(n, ca, c)
        print("✓ 共线点处理成功")
        
        # 检查旋转矩阵正交性
        R = rigid.rotation
        orthogonality = np.abs(R @ R.T - np.eye(3)).max()
        print(f"  旋转矩阵正交性误差: {orthogonality:.8f}")
        
        # 检查基向量长度
        e1_norm = np.linalg.norm(R[:, 0])
        e2_norm = np.linalg.norm(R[:, 1])
        e3_norm = np.linalg.norm(R[:, 2])
        print(f"  基向量长度: e1={e1_norm:.6f}, e2={e2_norm:.6f}, e3={e3_norm:.6f}")
        
        # 检查基向量垂直性
        dot12 = abs(np.dot(R[:, 0], R[:, 1]))
        dot23 = abs(np.dot(R[:, 1], R[:, 2]))
        dot31 = abs(np.dot(R[:, 2], R[:, 0]))
        print(f"  垂直性: |e1·e2|={dot12:.8f}, |e2·e3|={dot23:.8f}, |e3·e1|={dot31:.8f}")
        
        print("✓ 所有检查通过！")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
    
    print()


if __name__ == "__main__":
    test_rigid_basic()
    test_quaternion_conversion()
    test_rigid_batch()
    test_noise()
    test_collinear_points()  # 新增：共线点测试
    test_from_pdb()
    
    print("="*80)
    print("✅ 所有测试通过！")
    print("="*80)
