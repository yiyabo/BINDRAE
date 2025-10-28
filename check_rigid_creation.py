#!/usr/bin/env python3
"""检查如何创建和compose Rigid对象"""

import sys
import os
import re

flash_ipa_path = '/tmp/flash_ipa/src'
if os.path.exists(flash_ipa_path):
    sys.path.insert(0, flash_ipa_path)

# 读取rigid.py找Rotation类
rigid_file = '/tmp/flash_ipa/src/flash_ipa/rigid.py'

with open(rigid_file, 'r') as f:
    content = f.read()

print("=" * 80)
print("查找 Rotation 类")
print("=" * 80)

# 查找Rotation类定义
rotation_match = re.search(r'class\s+Rotation.*?(?=\nclass\s+[A-Z])', content, re.DOTALL)
if rotation_match:
    rotation_class = rotation_match.group(0)
    lines = rotation_class.split('\n')[:100]
    
    print("\nRotation 类（前100行）:")
    print("-" * 80)
    for i, line in enumerate(lines, 1):
        print(f"{i:3d}| {line}")
    
    # 查找from_matrix方法
    from_matrix_match = re.search(r'def\s+from_matrix\(.*?\):', rotation_class, re.DOTALL)
    if from_matrix_match:
        print("\n" + "=" * 80)
        print("✓ 找到 Rotation.from_matrix:")
        print("=" * 80)
        print(from_matrix_match.group(0))

