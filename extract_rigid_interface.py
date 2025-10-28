#!/usr/bin/env python3
"""智能提取Rigid类的接口（签名+文档）"""

import re

rigid_file = '/tmp/flash_ipa/src/flash_ipa/rigid.py'

print("=" * 80)
print("FlashIPA Rigid 类接口提取")
print("=" * 80)

with open(rigid_file, 'r') as f:
    content = f.read()

# 查找Rigid类
rigid_match = re.search(r'class\s+Rigid.*?(?=\nclass\s+[A-Z]|\Z)', content, re.DOTALL)

if not rigid_match:
    print("❌ 未找到Rigid类")
    exit(1)

rigid_class = rigid_match.group(0)

# 提取类定义头部（到第一个def）
header_match = re.search(r'class\s+Rigid.*?(?=\n    def)', rigid_class, re.DOTALL)
if header_match:
    print("\n" + "=" * 80)
    print("类定义头部:")
    print("=" * 80)
    print(header_match.group(0))

# 提取关键方法
methods_to_extract = [
    '__init__',
    'from_3_points',
    'from_tensor_4x4',
    'from_tensor_7',
    'compose',
    'apply',
    'invert_apply',
    'invert',
    'stop_rot_gradient',
]

print("\n" + "=" * 80)
print("关键方法签名和文档:")
print("=" * 80)

for method_name in methods_to_extract:
    # 匹配方法定义到下一个方法或类结束
    pattern = rf'(    def\s+{method_name}\(.*?\n.*?)(?=\n    def|\n\nclass|\Z)'
    match = re.search(pattern, rigid_class, re.DOTALL)
    
    if match:
        method_text = match.group(1)
        
        # 只保留签名和文档字符串（去掉实现代码）
        # 匹配：def ... : 和紧接着的"""..."""
        sig_doc_pattern = r'(    def\s+.*?:\s*(?:\n\s*""".*?""")?)(?=\n\s{8}[^"\s]|\n\s{4}def|\Z)'
        sig_doc_match = re.search(sig_doc_pattern, method_text, re.DOTALL)
        
        if sig_doc_match:
            print(f"\n{'-'*80}")
            print(f"方法: {method_name}")
            print(f"{'-'*80}")
            print(sig_doc_match.group(1))
        else:
            # 备选：至少显示签名
            sig_match = re.search(r'    def\s+.*?:', method_text)
            if sig_match:
                print(f"\n{'-'*80}")
                print(f"方法: {method_name}")
                print(f"{'-'*80}")
                print(sig_match.group(0))
    else:
        print(f"\n✗ 未找到方法: {method_name}")

print("\n" + "=" * 80)
print("完成")
print("=" * 80)

