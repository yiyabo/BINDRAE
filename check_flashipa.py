#!/usr/bin/env python3
"""检查FlashIPA的实际接口"""

try:
    from flash_ipa import EdgeEmbedder, EdgeEmbedderConfig
    print("✅ FlashIPA导入成功\n")
    
    # 检查EdgeEmbedderConfig的参数
    import inspect
    config_sig = inspect.signature(EdgeEmbedderConfig.__init__)
    print("EdgeEmbedderConfig参数:")
    for param_name, param in config_sig.parameters.items():
        if param_name != 'self':
            print(f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
            if param.default != inspect.Parameter.empty:
                print(f"    默认值: {param.default}")
    
    # 检查EdgeEmbedder的forward方法
    forward_sig = inspect.signature(EdgeEmbedder.forward)
    print("\nEdgeEmbedder.forward参数:")
    for param_name, param in forward_sig.parameters.items():
        if param_name != 'self':
            print(f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")
    
    # 打印文档字符串
    print("\nEdgeEmbedder文档:")
    print(EdgeEmbedder.__doc__)
    
    print("\nEdgeEmbedderConfig文档:")
    print(EdgeEmbedderConfig.__doc__)
    
except ImportError as e:
    print(f"❌ FlashIPA未安装: {e}")
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

