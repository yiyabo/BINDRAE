#!/usr/bin/env python3
"""完整解析FK模板内容"""

import pickle
from pathlib import Path
import json

fk_file = Path('data/casf2016/processed/fk_template.pkl')

print("="*80)
print("FK模板完整内容")
print("="*80)

with open(fk_file, 'rb') as f:
    fk_template = pickle.load(f)

print(f"\n模板类型: {type(fk_template)}")

if isinstance(fk_template, dict):
    print(f"\n顶层键值: {list(fk_template.keys())}\n")
    
    for key, value in fk_template.items():
        print(f"\n{'='*60}")
        print(f"键: {key}")
        print(f"{'='*60}")
        print(f"类型: {type(value)}")
        
        if isinstance(value, dict):
            print(f"子键数量: {len(value)}")
            if len(value) <= 25:  # 如果不多，全部打印
                for k, v in value.items():
                    print(f"  {k}: {v if not isinstance(v, (list, dict)) else f'{type(v).__name__}(len={len(v)})'}")
            else:
                # 只打印前10个
                for i, (k, v) in enumerate(list(value.items())[:10]):
                    print(f"  {k}: {v if not isinstance(v, (list, dict)) else f'{type(v).__name__}(len={len(v)})'}")
                print(f"  ... 还有{len(value)-10}个")
        elif isinstance(value, (list, tuple)):
            print(f"长度: {len(value)}")
            if len(value) <= 10:
                for i, item in enumerate(value):
                    print(f"  [{i}]: {item}")
        else:
            print(f"值: {value}")

print("\n" + "="*80)

