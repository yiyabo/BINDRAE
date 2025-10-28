#!/usr/bin/env python3
"""探索FlashIPA的InvariantPointAttention接口"""

import sys
import os

flash_ipa_path = '/tmp/flash_ipa/src'
if os.path.exists(flash_ipa_path):
    sys.path.insert(0, flash_ipa_path)

try:
    from flash_ipa import InvariantPointAttention, IPAConfig
    print("✅ InvariantPointAttention 导入成功\n")
    
    import inspect
    
    # 检查IPAConfig参数
    print("=" * 80)
    print("IPAConfig 参数:")
    print("=" * 80)
    config_sig = inspect.signature(IPAConfig.__init__)
    for param_name, param in config_sig.parameters.items():
        if param_name != 'self':
            print(f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
            if param.default != inspect.Parameter.empty:
                print(f"    默认值: {param.default}")
    
    # 检查IPA的forward方法
    print("\n" + "=" * 80)
    print("InvariantPointAttention.forward 参数:")
    print("=" * 80)
    forward_sig = inspect.signature(InvariantPointAttention.forward)
    for param_name, param in forward_sig.parameters.items():
        if param_name != 'self':
            print(f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
    
    # 打印文档
    print("\n" + "=" * 80)
    print("IPAConfig 文档:")
    print("=" * 80)
    print(IPAConfig.__doc__)
    
    print("\n" + "=" * 80)
    print("InvariantPointAttention 文档:")
    print("=" * 80)
    print(InvariantPointAttention.__doc__)
    
    # 尝试创建实例
    print("\n" + "=" * 80)
    print("测试实例化:")
    print("=" * 80)
    
    config = IPAConfig(
        c_s=384,
        c_z=128,
    )
    print(f"✓ IPAConfig 创建成功")
    print(f"  配置: {config}")
    
    ipa = InvariantPointAttention(config)
    print(f"\n✓ InvariantPointAttention 创建成功")
    print(f"  参数量: {sum(p.numel() for p in ipa.parameters()):,}")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("\n可能的原因:")
    print("  1. FlashIPA路径不对")
    print("  2. 模块名称不对")
    print("\n请尝试:")
    print("  - 检查 /tmp/flash_ipa/src 是否存在")
    print("  - 查看 flash_ipa 目录下有哪些文件")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

