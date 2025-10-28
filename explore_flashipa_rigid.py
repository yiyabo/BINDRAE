#!/usr/bin/env python3
"""探索FlashIPA的Rigid类"""

import sys
import os

flash_ipa_path = '/tmp/flash_ipa/src'
if os.path.exists(flash_ipa_path):
    sys.path.insert(0, flash_ipa_path)

print("=" * 80)
print("读取 flash_ipa/rigid.py 源代码")
print("=" * 80)

rigid_file = '/tmp/flash_ipa/src/flash_ipa/rigid.py'

if os.path.exists(rigid_file):
    with open(rigid_file, 'r') as f:
        lines = f.readlines()
    
    # 只打印前100行（包含Rigid类定义）
    print("\n前100行:")
    print("-" * 80)
    for i, line in enumerate(lines[:100], 1):
        print(f"{i:3d}| {line}", end='')
    
    print("\n" + "-" * 80)
    print(f"总行数: {len(lines)}")
else:
    print(f"❌ 文件不存在: {rigid_file}")

# 尝试查找Rigid类的关键方法
print("\n" + "=" * 80)
print("搜索关键方法:")
print("=" * 80)

content = ''.join(lines) if os.path.exists(rigid_file) else ""

import re

methods = ['__init__', 'from_tensor_7', 'from_3_points', 'compose', 'apply', 'invert_apply']
for method in methods:
    pattern = rf'def\s+{method}\(self.*?\):'
    match = re.search(pattern, content)
    if match:
        print(f"\n✓ 找到 {method}:")
        print(f"  {match.group(0)}")

