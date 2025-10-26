#!/usr/bin/env python3
"""
准备 Forward Kinematics (FK) 模板
- 标准键长/键角（Engh & Huber 1991）
- 残基侧链拓扑树
- 支持从扭转角重建原子坐标
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


# 标准键长（Å）- Engh & Huber 1991
STANDARD_BOND_LENGTHS = {
    # 主链键
    'N-CA': 1.458,
    'CA-C': 1.525,
    'C-O': 1.231,
    'C-N': 1.329,  # 肽键
    
    # 常见侧链键
    'CA-CB': 1.530,
    'CB-CG': 1.520,
    'CG-CD': 1.520,
    'CD-CE': 1.520,
    'CE-NZ': 1.489,  # Lys
    
    'CB-SG': 1.808,  # Cys
    'CB-OG': 1.417,  # Ser
    'CB-OG1': 1.433, # Thr
    
    'CG-SD': 1.803,  # Met
    'SD-CE': 1.791,  # Met
    
    'CG-OD1': 1.249, # Asp/Asn
    'CG-ND2': 1.328, # Asn
    
    'CD-OE1': 1.249, # Glu/Gln
    'CD-NE2': 1.328, # Gln
    
    'CG-CD1': 1.387, # Phe/Tyr/His
    'CG-CD2': 1.387,
    'CD-NE': 1.456,  # Arg
    'NE-CZ': 1.329,  # Arg
    'CZ-NH1': 1.326, # Arg
    
    'CD-NE1': 1.374, # Trp
    'CG-ND1': 1.381, # His
}

# 标准键角（度）
STANDARD_BOND_ANGLES = {
    # 主链键角
    'N-CA-C': 111.2,
    'CA-C-N': 116.2,
    'C-N-CA': 121.7,
    'CA-C-O': 120.8,
    
    # 侧链键角
    'N-CA-CB': 110.5,
    'C-CA-CB': 110.1,
    'CA-CB-CG': 113.4,
    'CB-CG-CD': 113.4,
    'CG-CD-CE': 112.4,
    
    # Pro 特殊
    'N-CA-CB-PRO': 103.0,
    'CA-CB-CG-PRO': 104.0,
}

# 残基侧链拓扑树（从 CA 开始）
RESIDUE_TOPOLOGY = {
    'ALA': [
        ('CA', 'CB', []),
    ],
    'CYS': [
        ('CA', 'CB', [
            ('CB', 'SG', []),
        ]),
    ],
    'ASP': [
        ('CA', 'CB', [
            ('CB', 'CG', [
                ('CG', 'OD1', []),
                ('CG', 'OD2', []),
            ]),
        ]),
    ],
    'GLU': [
        ('CA', 'CB', [
            ('CB', 'CG', [
                ('CG', 'CD', [
                    ('CD', 'OE1', []),
                    ('CD', 'OE2', []),
                ]),
            ]),
        ]),
    ],
    'PHE': [
        ('CA', 'CB', [
            ('CB', 'CG', [
                ('CG', 'CD1', [
                    ('CD1', 'CE1', [
                        ('CE1', 'CZ', []),
                    ]),
                ]),
                ('CG', 'CD2', [
                    ('CD2', 'CE2', [
                        ('CE2', 'CZ', []),
                    ]),
                ]),
            ]),
        ]),
    ],
    'GLY': [],  # 无侧链
    'HIS': [
        ('CA', 'CB', [
            ('CB', 'CG', [
                ('CG', 'ND1', [
                    ('ND1', 'CE1', [
                        ('CE1', 'NE2', [
                            ('NE2', 'CD2', []),
                        ]),
                    ]),
                ]),
                ('CG', 'CD2', []),
            ]),
        ]),
    ],
    'ILE': [
        ('CA', 'CB', [
            ('CB', 'CG1', [
                ('CG1', 'CD1', []),
            ]),
            ('CB', 'CG2', []),
        ]),
    ],
    'LYS': [
        ('CA', 'CB', [
            ('CB', 'CG', [
                ('CG', 'CD', [
                    ('CD', 'CE', [
                        ('CE', 'NZ', []),
                    ]),
                ]),
            ]),
        ]),
    ],
    'LEU': [
        ('CA', 'CB', [
            ('CB', 'CG', [
                ('CG', 'CD1', []),
                ('CG', 'CD2', []),
            ]),
        ]),
    ],
    'MET': [
        ('CA', 'CB', [
            ('CB', 'CG', [
                ('CG', 'SD', [
                    ('SD', 'CE', []),
                ]),
            ]),
        ]),
    ],
    'ASN': [
        ('CA', 'CB', [
            ('CB', 'CG', [
                ('CG', 'OD1', []),
                ('CG', 'ND2', []),
            ]),
        ]),
    ],
    'PRO': [
        ('CA', 'CB', [
            ('CB', 'CG', [
                ('CG', 'CD', [
                    ('CD', 'N', []),  # 环状连接
                ]),
            ]),
        ]),
    ],
    'GLN': [
        ('CA', 'CB', [
            ('CB', 'CG', [
                ('CG', 'CD', [
                    ('CD', 'OE1', []),
                    ('CD', 'NE2', []),
                ]),
            ]),
        ]),
    ],
    'ARG': [
        ('CA', 'CB', [
            ('CB', 'CG', [
                ('CG', 'CD', [
                    ('CD', 'NE', [
                        ('NE', 'CZ', [
                            ('CZ', 'NH1', []),
                            ('CZ', 'NH2', []),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ],
    'SER': [
        ('CA', 'CB', [
            ('CB', 'OG', []),
        ]),
    ],
    'THR': [
        ('CA', 'CB', [
            ('CB', 'OG1', []),
            ('CB', 'CG2', []),
        ]),
    ],
    'VAL': [
        ('CA', 'CB', [
            ('CB', 'CG1', []),
            ('CB', 'CG2', []),
        ]),
    ],
    'TRP': [
        ('CA', 'CB', [
            ('CB', 'CG', [
                ('CG', 'CD1', [
                    ('CD1', 'NE1', [
                        ('NE1', 'CE2', [
                            ('CE2', 'CZ2', [
                                ('CZ2', 'CH2', [
                                    ('CH2', 'CZ3', [
                                        ('CZ3', 'CE3', [
                                            ('CE3', 'CD2', []),
                                        ]),
                                    ]),
                                ]),
                            ]),
                            ('CE2', 'CD2', []),
                        ]),
                    ]),
                ]),
                ('CG', 'CD2', []),
            ]),
        ]),
    ],
    'TYR': [
        ('CA', 'CB', [
            ('CB', 'CG', [
                ('CG', 'CD1', [
                    ('CD1', 'CE1', [
                        ('CE1', 'CZ', [
                            ('CZ', 'OH', []),
                        ]),
                    ]),
                ]),
                ('CG', 'CD2', [
                    ('CD2', 'CE2', [
                        ('CE2', 'CZ', []),
                    ]),
                ]),
            ]),
        ]),
    ],
}


class FKTemplateBuilder:
    """FK 模板构建器"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / "data" / "casf2016" / "processed"
    
    def build_template(self) -> Dict:
        """构建FK模板"""
        template = {
            'bond_lengths': STANDARD_BOND_LENGTHS,
            'bond_angles': STANDARD_BOND_ANGLES,
            'residue_topology': RESIDUE_TOPOLOGY,
            'metadata': {
                'reference': 'Engh & Huber 1991',
                'units': {
                    'bond_lengths': 'Angstrom',
                    'bond_angles': 'degrees'
                }
            }
        }
        
        return template
    
    def save_template(self, template: Dict, output_path: Path):
        """保存模板到 pickle 文件"""
        with open(output_path, 'wb') as f:
            pickle.dump(template, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def verify_template(self, template_path: Path) -> bool:
        """验证模板文件"""
        try:
            with open(template_path, 'rb') as f:
                template = pickle.load(f)
            
            # 检查必需字段
            required_keys = ['bond_lengths', 'bond_angles', 'residue_topology']
            for key in required_keys:
                if key not in template:
                    print(f"  ❌ 缺少字段: {key}")
                    return False
            
            # 检查残基拓扑完整性
            aa_list = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 
                      'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 
                      'THR', 'VAL', 'TRP', 'TYR']
            
            missing_aa = []
            for aa in aa_list:
                if aa not in template['residue_topology']:
                    missing_aa.append(aa)
            
            if missing_aa:
                print(f"  ⚠️  缺少残基拓扑: {', '.join(missing_aa)}")
            
            print(f"  ✓ 模板包含 {len(template['bond_lengths'])} 个键长")
            print(f"  ✓ 模板包含 {len(template['bond_angles'])} 个键角")
            print(f"  ✓ 模板包含 {len(template['residue_topology'])} 个残基拓扑")
            
            return True
            
        except Exception as e:
            print(f"  ❌ 验证失败: {e}")
            return False
    
    def run(self):
        """运行构建流程"""
        print("="*80)
        print("FK 模板准备")
        print("="*80)
        print(f"输出目录: {self.processed_dir}")
        print()
        
        # 构建模板
        print("构建 FK 模板...")
        template = self.build_template()
        print(f"  ✓ 键长: {len(template['bond_lengths'])} 个")
        print(f"  ✓ 键角: {len(template['bond_angles'])} 个")
        print(f"  ✓ 残基拓扑: {len(template['residue_topology'])} 种氨基酸")
        print()
        
        # 保存模板
        output_path = self.processed_dir / "fk_template.pkl"
        print(f"保存模板到: {output_path}")
        self.save_template(template, output_path)
        print(f"  ✓ 文件大小: {output_path.stat().st_size / 1024:.1f} KB")
        print()
        
        # 验证模板
        print("验证模板...")
        if self.verify_template(output_path):
            print("  ✅ 模板验证通过")
        else:
            print("  ❌ 模板验证失败")
            return
        
        print("\n" + "="*80)
        print("✓ FK 模板准备完成！")
        print("="*80)
        print(f"\n输出文件: {output_path.name}")
        print(f"\n使用方法:")
        print(f"```python")
        print(f"import pickle")
        print(f"with open('{output_path.name}', 'rb') as f:")
        print(f"    template = pickle.load(f)")
        print(f"bond_length = template['bond_lengths']['N-CA']")
        print(f"```")


def main():
    """主函数"""
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/apple/code/BINDRAE"
    
    builder = FKTemplateBuilder(base_dir)
    builder.run()


if __name__ == "__main__":
    main()
