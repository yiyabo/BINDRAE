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
    
    # 模型规模
    model_size: str = 'small'  # 'small', 'stable_wide', 'enhanced_ligand', 'medium', 'large', 'wide_shallow'
    
    # 数据
    data_dir: str = 'data/apo_holo_triplets'
    batch_size: int = 4
    num_workers: int = 4
    max_n_res: Optional[int] = None  # 限制每个batch的最大残基数（None=不限制）
    valid_samples_file: Optional[str] = None  # 训练集有效样本列表（建议使用 train_valid.txt）
    val_samples_file: Optional[str] = None  # 验证集有效样本列表（None=使用原始 val split）
    sample_metadata_file: Optional[str] = None  # 样本长度/复杂度元数据（由validate_triplets_data.py生成）
    length_bucketed_sampling: bool = False  # 是否启用长度感知的 DDP batch 组织
    bucket_size_multiplier: int = 8  # bucket 大小 = global_batch_size * multiplier
    residue_budget: Optional[int] = None  # 每个 rank 每步的总残基预算（可变 local batch）
    
    # 优化器
    lr: float = 5e-4  # 折中值（1e-4太慢，1e-3可能不稳定）
    weight_decay: float = 0.05
    grad_clip: float = 1.0  # 梯度裁剪保护NaN
    
    # 学习率调度
    warmup_steps: int = 1000
    lr_scheduler: str = 'plateau'  # 'plateau', 'cosine'
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    min_lr_scale: float = 0.01
    max_epochs: int = 100
    
    # 损失权重
    w_fape: float = 1.0
    w_chi: float = 1.0
    w_clash: float = 0.1
    lambda_pchi1: float = 0.0
    pchi1_mask_mode: str = 'soft'  # 'soft', 'ligand_facing', 'holo_pocket'
    pchi1_start_step: int = 0
    pchi1_ramp_steps: int = 0
    use_pocket_chi1_expert: bool = False
    pocket_chi1_expert_hidden: int = 128
    pocket_chi1_expert_layers: int = 2
    pocket_chi1_gate_threshold: float = 0.5
    pocket_chi1_residual_scale: float = 0.25
    lambda_chi1_rotamer: float = 0.0
    lambda_contact: float = 0.0
    lambda_candidate_chi1: float = 0.0
    lambda_ligand_contrastive: float = 0.0
    ligand_contrastive_margin: float = 0.2
    lambda_candidate_rerank: float = 0.0
    candidate_rerank_margin: float = 0.1
    candidate_rerank_rank_margin: float = 0.0
    candidate_rerank_contact_only: bool = True
    candidate_rerank_decoy_kind: str = 'translated'  # 'translated' | 'nolig' | 'shuffled'
    candidate_scorer_lr_scale: float = 1.0  # LR multiplier for randomly-initialized candidate scorer
    lambda_geometry_chi1: float = 0.0
    lambda_base_prior: float = 0.0  # weight for base-prior-only CE (trains Dunbrack-like branch independently)
    geometry_scorer_lr_scale: float = 1.0
    geometry_scorer_use_sgeo: bool = False
    geometry_scorer_sgeo_dim: int = 32
    geometry_scorer_use_typed_energy: bool = False
    geometry_scorer_typed_pair_dim: int = 64
    geometry_scorer_typed_cutoff: float = 6.0
    geometry_scorer_typed_init_scale: float = 0.1
    lambda_typed_candidate_energy: float = 0.0
    typed_candidate_decoy_kind: str = 'scrambled'  # 'scrambled' | 'shuffled' | 'nolig' | 'translated'
    typed_candidate_contact_only: bool = True
    typed_candidate_margin: float = 0.05
    typed_candidate_noharm_weight: float = 0.1
    typed_candidate_noncontact_zero_weight: float = 0.05

    # Phase-1 residual retraining (GPT-5.5 Pro plan)
    geometry_scorer_bounded_residual: bool = False
    geometry_scorer_residual_max: float = 5.0
    geometry_scorer_residual_tau: float = 2.0
    geometry_scorer_gate_norm: bool = False
    geometry_scorer_gate_clamp: float = 6.0
    geometry_scorer_gate_init_bias: float = 0.0
    reset_residual_and_gate_on_resume: bool = False
    freeze_base_mlp: bool = False
    freeze_gate_mlp: bool = False  # Phase-1 v2: gate fixed at 1.0, no learning
    detach_base_for_residual: bool = False
    gate_warmup_open_steps: int = 0
    residual_beta_warmup_steps: int = 0
    residual_beta_min: float = 0.1
    # Likelihood-ratio losses on G_i = log_softmax(full) - log_softmax(base)
    lambda_g_lift_switch: float = 0.0
    lambda_g_noharm: float = 0.0
    lambda_g_zero_noncontact: float = 0.0
    g_lift_margin: float = 0.5
    g_noharm_margin: float = 0.5
    g_noncontact_threshold: float = 8.0  # CA-ligand distance threshold (Å)
    # Phase-1 v2 G-vector posterior losses
    lambda_g_switch_dir: float = 0.0       # CE on G/T toward holo bin
    lambda_g_switch_amp: float = 0.0       # relu(m_amp - G_holo)
    lambda_g_switch_rank: float = 0.0      # relu(m_rank - (G_holo - max_other))
    lambda_g_antiharm: float = 0.0         # relu(max_nonapo_G - tau) on apo-correct
    lambda_g_decoy: float = 0.0            # G_correct[holo] > G_decoy[holo] + m
    g_switch_temperature: float = 1.0      # T in CE(G/T, holo)
    g_switch_amp_margin: float = 0.05
    g_switch_rank_margin: float = 0.05
    g_antiharm_tau: float = 0.05
    g_decoy_margin: float = 0.05
    g_switch_contact_only: bool = True     # restrict switch losses to contact residues
    g_decoy_kind: str = 'translated'       # 'translated' | 'nolig' | 'shuffled'
    g_decoy_translation_offset: float = 100.0  # Å offset for translated decoy
    lambda_switch_bce: float = 0.0
    lambda_rescue_noharm: float = 0.0
    lambda_ligand_residual: float = 0.0
    rescue_noharm_margin: float = 0.5
    ligand_residual_margin: float = 0.3
    use_nolig_contrastive: bool = False
    chi1_rotamer_hidden: int = 128
    candidate_chi1_hidden: int = 128
    contact_hidden: int = 128
    contact_loss_pos_weight: float = 0.0  # <=0 means batch-balanced automatic weight
    freeze_stage1_backbone_for_posteriors: bool = False
    unfreeze_ligand_conditioner_for_posteriors: bool = False
    unfreeze_last_ipa_blocks_for_posteriors: int = 0
    
    # Pocket Routing Adapter
    use_pocket_routing_adapter: bool = False
    pocket_routing_hidden: int = 128
    pocket_routing_layers: int = 2
    pocket_routing_gate_threshold: float = 0.3
    pocket_routing_residual_scale: float = 0.5
    
    # Warmup
    pocket_warmup_steps: int = 2000  # 口袋权重warmup
    ligand_gate_warmup_steps: int = 2000  # 配体门控warmup
    
    # 验证与早停
    val_interval: int = 1  # 每几个epoch验证一次
    early_stop_patience: int = 20
    save_top_k: int = 3
    selection_metric: str = 'total'  # 选模/早停指标，默认与历史行为兼容
    save_latest_checkpoint: bool = True  # 每次验证后保存 latest_model.pt
    save_epoch_checkpoints: bool = False  # 可选：保存 epoch_{k}.pt
    metrics_filename: str = 'metrics.jsonl'  # 结构化 epoch 指标日志
    compute_slow_metrics: bool = False  # 是否计算较慢的验证指标（iRMSD/clash/contact）
    enable_dual_mask_audit: bool = False  # 是否启用 dual-mask 审计指标
    audit_filename: str = 'audit_metrics_latest.json'  # 审计指标快照文件（写入 log_dir）
    
    # 日志
    log_dir: str = 'logs/stage1'
    save_dir: str = 'checkpoints/stage1'
    log_interval: int = 10  # 每几个step记录一次
    
    # 设备
    device: str = 'cuda'
    mixed_precision: bool = True  # AMP 混合精度
    amp_dtype: str = 'auto'  # auto=优先bf16，其次fp16
    
    # 分布式训练 (DDP)
    distributed: bool = False  # 是否启用分布式训练
    local_rank: int = -1  # 本地 GPU rank（由 torchrun 自动设置）
    world_size: int = 1  # 总 GPU 数量
    ddp_find_unused_parameters: bool = False  # fullscratch multi-head runs may leave auxiliary heads unused
    
    # 其他
    seed: int = 2025
    resume_from: Optional[str] = None  # checkpoint路径
