"""
训练配置

Author: BINDRAE Team
Date: 2025-10-28
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Stage-1训练配置"""
    
    # 数据
    data_dir: str = 'data/casf2016'
    batch_size: int = 4
    num_workers: int = 4
    
    # 优化器
    lr: float = 5e-4  # 折中值（1e-4太慢，1e-3可能不稳定）
    weight_decay: float = 0.05
    grad_clip: float = 1.0  # 梯度裁剪保护NaN
    
    # 学习率调度
    warmup_steps: int = 1000
    max_epochs: int = 100
    
    # 损失权重
    w_fape: float = 1.0
    w_torsion: float = 1.0
    w_dist: float = 0.1
    w_clash: float = 0.1
    
    # Warmup
    pocket_warmup_steps: int = 2000  # 口袋权重warmup
    ligand_gate_warmup_steps: int = 2000  # 配体门控warmup
    
    # 验证与早停
    val_interval: int = 1  # 每几个epoch验证一次
    early_stop_patience: int = 20
    save_top_k: int = 3
    
    # 日志
    log_dir: str = 'logs/stage1'
    save_dir: str = 'checkpoints/stage1'
    log_interval: int = 10  # 每几个step记录一次
    
    # 设备
    device: str = 'cuda:1'
    mixed_precision: bool = True  # fp16混合精度
    
    # 其他
    seed: int = 2025
    resume_from: Optional[str] = None  # checkpoint路径

