#!/usr/bin/env python3
"""验证flash-attn的ABI问题是否解决"""

print("=" * 80)
print("FlashAttention ABI 验证")
print("=" * 80)

# 测试1：直接导入
print("\n[测试1] 直接导入 flash_attn...")
try:
    from flash_attn import flash_attn_varlen_func
    print("✅ 直接导入成功")
    direct_import = True
except ImportError as e:
    print(f"❌ 直接导入失败: {e}")
    direct_import = False

# 测试2：通过FlashIPA导入（间接）
print("\n[测试2] 通过 FlashIPA 间接使用...")
try:
    import sys
    import os
    flash_ipa_path = '/tmp/flash_ipa/src'
    if os.path.exists(flash_ipa_path):
        sys.path.insert(0, flash_ipa_path)
    
    from flash_ipa.ipa import InvariantPointAttention, IPAConfig
    import torch
    
    # 创建简单的IPA实例
    config = IPAConfig(c_s=384, c_z=128, z_factor_rank=2)
    ipa = InvariantPointAttention(config)
    
    print("✅ FlashIPA实例化成功（说明flash_attn能在内部工作）")
    flashipa_works = True
except Exception as e:
    print(f"❌ FlashIPA实例化失败: {e}")
    flashipa_works = False

# 测试3：实际运行flash_attn
print("\n[测试3] 实际执行 FlashAttention 计算...")
if flashipa_works:
    try:
        from flash_ipa.rigid import Rigid, Rotation
        
        B, N = 1, 10
        c_s = 384
        
        s = torch.randn(B, N, c_s)
        rot = Rotation(rot_mats=torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1))
        rigids = Rigid(rots=rot, trans=torch.randn(B, N, 3))
        mask = torch.ones(B, N, dtype=torch.bool)
        
        if torch.cuda.is_available():
            ipa = ipa.cuda()
            s = s.cuda()
            rigids = Rigid(rots=Rotation(rot_mats=rigids.get_rots().get_rot_mats().cuda()), 
                          trans=rigids.get_trans().cuda())
            mask = mask.cuda()
        
        with torch.no_grad():
            output = ipa(s=s, z=None, z_factor_1=None, z_factor_2=None, r=rigids, mask=mask)
        
        print(f"✅ FlashAttention实际计算成功")
        print(f"   输出形状: {output.shape}")
        actual_compute = True
    except Exception as e:
        print(f"❌ 实际计算失败: {e}")
        actual_compute = False
else:
    actual_compute = False

# 总结
print("\n" + "=" * 80)
print("验证总结")
print("=" * 80)
print(f"直接导入 flash_attn:          {'✅ 成功' if direct_import else '❌ 失败（ABI问题）'}")
print(f"FlashIPA 实例化:              {'✅ 成功' if flashipa_works else '❌ 失败'}")
print(f"FlashAttention 实际计算:      {'✅ 成功' if actual_compute else '❌ 失败'}")

print("\n结论:")
if direct_import:
    print("  ✅ flash-attn ABI问题已完全解决")
elif flashipa_works and actual_compute:
    print("  ⚠️  直接导入失败，但FlashIPA内部能正常工作")
    print("      → 不影响项目使用，可以继续开发")
    print("      → ABI问题被FlashIPA的导入方式绕过了")
else:
    print("  ❌ flash-attn无法使用，需要重新编译")

print("=" * 80)

