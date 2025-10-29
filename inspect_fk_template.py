#!/usr/bin/env python3
"""检查FK模板内容"""

import pickle
from pathlib import Path

fk_file = Path('data/casf2016/processed/fk_template.pkl')

with open(fk_file, 'rb') as f:
    fk_template = pickle.load(f)

print("FK模板内容:")
print(f"类型: {type(fk_template)}")

if isinstance(fk_template, dict):
    print(f"\n键值:")
    for k, v in fk_template.items():
        print(f"  {k}: {type(v)}")
        if isinstance(v, dict) and len(v) < 30:
            for kk, vv in v.items():
                print(f"    {kk}: {vv if not isinstance(vv, (list, dict)) else type(vv)}")

