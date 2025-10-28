#!/usr/bin/env python3
"""精确定位Rigid类定义"""

import re

rigid_file = '/tmp/flash_ipa/src/flash_ipa/rigid.py'

print("=" * 80)
print("搜索 Rigid 类")
print("=" * 80)

with open(rigid_file, 'r') as f:
    lines = f.readlines()

# 查找包含"class Rigid"的行
for i, line in enumerate(lines):
    if re.match(r'^class\s+Rigid', line):
        print(f"\n找到 Rigid 类定义在第 {i+1} 行")
        print(f"{'-'*80}")
        
        # 打印类定义及其后150行
        start = i
        end = min(i + 150, len(lines))
        
        for j in range(start, end):
            print(f"{j+1:4d}| {lines[j]}", end='')
        
        print(f"\n{'-'*80}")
        break

print("\n" + "=" * 80)
print("完成")
print("=" * 80)

