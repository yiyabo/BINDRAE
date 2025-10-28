#!/usr/bin/env python3
"""直接读取FlashIPA的ipa.py源代码"""

import os
import re

ipa_file = '/tmp/flash_ipa/src/flash_ipa/ipa.py'

if not os.path.exists(ipa_file):
    print(f"❌ 文件不存在: {ipa_file}")
    exit(1)

print("=" * 80)
print("读取 flash_ipa/ipa.py")
print("=" * 80)

with open(ipa_file, 'r') as f:
    content = f.read()

# 查找类定义
print("\n" + "=" * 80)
print("类定义:")
print("=" * 80)

class_pattern = r'class\s+(\w+).*?:'
classes = re.findall(class_pattern, content)
for cls in classes:
    print(f"  - {cls}")

# 查找IPAConfig或类似的配置类
print("\n" + "=" * 80)
print("查找配置类 (Config):")
print("=" * 80)

config_match = re.search(r'class\s+\w*Config.*?:\s*\n(.*?)(?=\nclass|\Z)', content, re.DOTALL)
if config_match:
    config_section = config_match.group(0)
    # 提取前50行
    lines = config_section.split('\n')[:50]
    for line in lines:
        print(line)
else:
    print("未找到Config类")

# 查找InvariantPointAttention的__init__
print("\n" + "=" * 80)
print("查找 InvariantPointAttention.__init__:")
print("=" * 80)

init_pattern = r'class\s+InvariantPointAttention.*?def\s+__init__\(self.*?\):'
init_match = re.search(init_pattern, content, re.DOTALL)
if init_match:
    init_text = init_match.group(0)
    print(init_text)
else:
    print("未找到 InvariantPointAttention.__init__")

# 查找forward方法签名
print("\n" + "=" * 80)
print("查找 InvariantPointAttention.forward 签名:")
print("=" * 80)

# 找到InvariantPointAttention类后查找forward
ipa_class_match = re.search(r'class\s+InvariantPointAttention.*?(?=\nclass|\Z)', content, re.DOTALL)
if ipa_class_match:
    ipa_class_content = ipa_class_match.group(0)
    forward_match = re.search(r'def\s+forward\(self.*?\):', ipa_class_content, re.DOTALL)
    if forward_match:
        forward_sig = forward_match.group(0)
        print(forward_sig)
        
        # 提取参数
        params_match = re.search(r'def\s+forward\(self,\s*(.*?)\):', forward_sig, re.DOTALL)
        if params_match:
            params_str = params_match.group(1)
            params = [p.strip() for p in params_str.split(',')]
            print("\n参数列表:")
            for param in params:
                print(f"  - {param}")

print("\n" + "=" * 80)
print("完成")
print("=" * 80)

