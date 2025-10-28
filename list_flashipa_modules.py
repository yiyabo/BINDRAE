#!/usr/bin/env python3
"""列出FlashIPA的所有可用模块"""

import sys
import os

flash_ipa_path = '/tmp/flash_ipa/src'
if os.path.exists(flash_ipa_path):
    sys.path.insert(0, flash_ipa_path)
    print(f"✓ FlashIPA路径已添加: {flash_ipa_path}\n")
else:
    print(f"❌ FlashIPA路径不存在: {flash_ipa_path}\n")

# 列出目录结构
print("=" * 80)
print("FlashIPA 目录结构:")
print("=" * 80)

if os.path.exists('/tmp/flash_ipa/src/flash_ipa'):
    for root, dirs, files in os.walk('/tmp/flash_ipa/src/flash_ipa'):
        level = root.replace('/tmp/flash_ipa/src/flash_ipa', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.py'):
                print(f'{subindent}{file}')

# 尝试导入flash_ipa并查看其属性
print("\n" + "=" * 80)
print("flash_ipa 模块内容:")
print("=" * 80)

try:
    import flash_ipa
    print(f"flash_ipa 路径: {flash_ipa.__file__ if hasattr(flash_ipa, '__file__') else 'unknown'}")
    print(f"\n可用属性/模块:")
    for attr in dir(flash_ipa):
        if not attr.startswith('_'):
            obj = getattr(flash_ipa, attr)
            print(f"  - {attr}: {type(obj).__name__}")
except Exception as e:
    print(f"❌ 导入 flash_ipa 失败: {e}")

# 尝试查找IPA相关模块
print("\n" + "=" * 80)
print("搜索IPA相关模块:")
print("=" * 80)

ipa_modules = []
if os.path.exists('/tmp/flash_ipa/src/flash_ipa'):
    for root, dirs, files in os.walk('/tmp/flash_ipa/src/flash_ipa'):
        for file in files:
            if 'ipa' in file.lower() and file.endswith('.py'):
                full_path = os.path.join(root, file)
                rel_path = full_path.replace('/tmp/flash_ipa/src/', '')
                module_path = rel_path.replace('/', '.').replace('.py', '')
                ipa_modules.append((file, module_path))
                print(f"  {file} → {module_path}")

# 尝试导入找到的IPA模块
if ipa_modules:
    print("\n" + "=" * 80)
    print("尝试导入IPA模块:")
    print("=" * 80)
    
    for filename, module_path in ipa_modules:
        try:
            module = __import__(module_path, fromlist=[''])
            print(f"\n✓ {module_path} 导入成功")
            print(f"  内容: {[x for x in dir(module) if not x.startswith('_')]}")
        except Exception as e:
            print(f"\n✗ {module_path} 导入失败: {e}")

