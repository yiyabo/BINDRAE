#!/usr/bin/env python3
"""查看氨基酸侧链拓扑的详细结构"""

import pickle
from pathlib import Path

fk_file = Path('data/casf2016/processed/fk_template.pkl')

with open(fk_file, 'rb') as f:
    fk_template = pickle.load(f)

residue_topology = fk_template['residue_topology']

print("="*80)
print("氨基酸侧链拓扑详细结构")
print("="*80)

# 查看几个代表性残基
test_residues = ['GLY', 'ALA', 'SER', 'LYS', 'PHE', 'ARG']

for res_name in test_residues:
    print(f"\n{'-'*60}")
    print(f"{res_name}（{['无侧链', '小侧链', '羟基', '长链', '芳香环', '复杂'][test_residues.index(res_name)]}）")
    print(f"{'-'*60}")
    
    topo = residue_topology[res_name]
    print(f"类型: {type(topo)}")
    print(f"长度: {len(topo)}")
    
    if len(topo) > 0:
        print(f"\n内容:")
        for i, item in enumerate(topo):
            print(f"  [{i}]: {item}")
            if isinstance(item, dict):
                for k, v in item.items():
                    print(f"      {k}: {v}")
            elif isinstance(item, (list, tuple)):
                print(f"      元素: {item}")

print("\n" + "="*80)
print("总结:")
print("="*80)
print("\n各残基侧链原子数:")
for res_name, topo in residue_topology.items():
    n_atoms = len(topo) + 1 if len(topo) > 0 else 0  # +1是CB
    print(f"  {res_name}: {n_atoms}个侧链原子")

