#!/usr/bin/env python3
"""
Stage-1 训练启动脚本

Usage:
    # 单卡训练
    python scripts/train_stage1.py [--options]
    
    # 多卡 DDP 训练
    torchrun --nproc_per_node=2 scripts/train_stage1.py --distributed [--options]

Author: BINDRAE Team
Date: 2025-10-28
"""

import os
import sys
from pathlib import Path
import argparse

# 避免 /dev/shm 限制导致的 DataLoader 崩溃
try:
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.stage1.training.config import TrainingConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Stage-1 训练')
    
    # 模型规模
    parser.add_argument('--model_size', type=str, default='small',
                       choices=['small', 'stable_wide', 'medium', 'large', 'wide_shallow', 'enhanced_ligand'],
                       help='模型规模: small(5M), stable_wide(7M), enhanced_ligand(8M,推荐), medium(10M), large(40M), wide_shallow(25M)')
    
    # 数据
    parser.add_argument('--data_dir', type=str, default='data/apo_holo_triplets',
                       help='数据目录')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批大小')
    parser.add_argument('--max_n_res', type=int, default=None,
                       help='每个batch的最大残基数（超过会过滤）')
    parser.add_argument('--valid_samples_file', type=str, default=None,
                       help='训练集有效样本列表（建议使用 train_valid.txt，而不是全局 valid_samples.txt）')
    parser.add_argument('--val_samples_file', type=str, default=None,
                       help='验证集有效样本列表（可选；不指定则使用原始 val split，不再复用训练清单）')
    parser.add_argument('--sample_metadata_file', type=str, default=None,
                       help='样本元数据 JSON（由scripts/validate_triplets_data.py生成）')
    parser.add_argument('--length_bucketed_sampling', action='store_true',
                       help='启用长度感知的 DDP batch 组织')
    parser.add_argument('--bucket_size_multiplier', type=int, default=8,
                       help='bucket 大小倍率：global_batch_size * multiplier')
    parser.add_argument('--residue_budget', type=int, default=None,
                       help='每个 rank 每步的总残基预算（启用后 local batch 可变）')
    
    # 训练
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='学习率调度 warmup/cosine 切换步数')
    parser.add_argument('--lr_scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine'],
                       help='学习率调度：plateau=验证退化时自动降LR，cosine=余弦衰减')
    parser.add_argument('--plateau_factor', type=float, default=0.5,
                       help='plateau 调度降LR倍率')
    parser.add_argument('--plateau_patience', type=int, default=2,
                       help='plateau 调度容忍的连续坏验证轮数')
    parser.add_argument('--min_lr_scale', type=float, default=0.01,
                       help='最小学习率相对初始LR的比例')
    parser.add_argument('--pocket_warmup_steps', type=int, default=2000,
                       help='口袋残基权重 warmup 步数')
    parser.add_argument('--ligand_gate_warmup_steps', type=int, default=2000,
                       help='配体门控 warmup 步数')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='最大训练轮数')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='梯度裁剪')
    
    # 损失权重
    parser.add_argument('--w_fape', type=float, default=1.0,
                       help='FAPE 损失权重（默认 1.0）')
    parser.add_argument('--w_chi', type=float, default=1.0,
                       help='Chi 损失权重（默认 1.0）')
    parser.add_argument('--w_clash', type=float, default=0.1,
                       help='Clash 损失权重（默认 0.1）')
    parser.add_argument('--lambda_pchi1', type=float, default=0.0,
                       help='E5 pocket χ1 auxiliary loss 权重（默认 0.0=关闭）')
    parser.add_argument('--pchi1_mask_mode', type=str, default='soft',
                       choices=['soft', 'ligand_facing', 'holo_pocket'],
                       help='pchi1 辅助项支持集：soft=沿用 w_res_warmed，ligand_facing=硬配体接触口袋，holo_pocket=硬holo距离口袋')
    parser.add_argument('--pchi1_start_step', type=int, default=0,
                       help='pchi1 辅助项开始生效的 step（默认 0）')
    parser.add_argument('--pchi1_ramp_steps', type=int, default=0,
                       help='pchi1 从 0 ramp 到目标 lambda 的步数（默认 0=无 ramp）')
    parser.add_argument('--use_pocket_chi1_expert', action='store_true',
                       help='启用 late pocket-only chi1 residual expert')
    parser.add_argument('--pocket_chi1_expert_hidden', type=int, default=128,
                       help='pocket chi1 expert 隐藏维度')
    parser.add_argument('--pocket_chi1_expert_layers', type=int, default=2,
                       help='pocket chi1 expert 层数')
    parser.add_argument('--pocket_chi1_gate_threshold', type=float, default=0.5,
                       help='pocket chi1 expert 的 w_res gate 阈值')
    parser.add_argument('--pocket_chi1_residual_scale', type=float, default=0.25,
                        help='pocket chi1 residual 的缩放系数')
    parser.add_argument('--lambda_chi1_rotamer', type=float, default=0.0,
                       help='χ1 rotamer posterior auxiliary loss 权重（默认 0.0=关闭）')
    parser.add_argument('--lambda_contact', type=float, default=0.0,
                       help='residue-ligand contact posterior auxiliary loss 权重（默认 0.0=关闭）')
    parser.add_argument('--lambda_candidate_chi1', type=float, default=0.0,
                       help='candidate-aware χ1 scorer CE loss 权重（默认 0.0=关闭）')
    parser.add_argument('--lambda_ligand_contrastive', type=float, default=0.0,
                       help='correct-vs-shuffled ligand χ1 contrastive loss 权重（默认 0.0=关闭）')
    parser.add_argument('--ligand_contrastive_margin', type=float, default=0.2,
                       help='ligand contrastive margin for gold χ1 candidate logit')
    parser.add_argument('--lambda_candidate_rerank', type=float, default=0.0,
                       help='explicit correct-vs-decoy ligand-causal candidate reranking loss weight')
    parser.add_argument('--candidate_rerank_margin', type=float, default=0.1,
                       help='margin for correct ligand holo-bin log-prob over decoy')
    parser.add_argument('--candidate_rerank_rank_margin', type=float, default=0.0,
                       help='optional margin for holo bin over other bins under correct ligand')
    parser.add_argument('--candidate_rerank_contact_only', action='store_true', default=True,
                       help='restrict candidate reranking to CA-ligand contact switch residues')
    parser.add_argument('--candidate_rerank_no_contact_only', dest='candidate_rerank_contact_only', action='store_false')
    parser.add_argument('--candidate_rerank_decoy_kind', type=str, default='translated',
                       choices=['translated', 'nolig', 'shuffled'],
                       help='decoy type for explicit candidate reranking')
    parser.add_argument('--candidate_scorer_lr_scale', type=float, default=1.0,
                       help='LR multiplier for candidate scorer (randomly initialized)')
    parser.add_argument('--lambda_geometry_chi1', type=float, default=0.0,
                       help='geometry-bypass χ1 scorer CE loss weight')
    parser.add_argument('--lambda_base_prior', type=float, default=0.0,
                       help='base-prior-only CE loss weight (trains Dunbrack-like branch independently)')
    parser.add_argument('--geometry_scorer_lr_scale', type=float, default=1.0,
                       help='LR multiplier for geometry scorer')
    parser.add_argument('--geometry_scorer_use_sgeo', action='store_true',
                       help='M3 mode: add s_geo projection to geometry scorer input')
    parser.add_argument('--geometry_scorer_sgeo_dim', type=int, default=32,
                       help='s_geo projection dim for M3 mode')
    parser.add_argument('--geometry_scorer_use_typed_energy', action='store_true',
                       help='enable typed residue-atom/ligand-token interaction energy in geometry scorer')
    parser.add_argument('--geometry_scorer_typed_pair_dim', type=int, default=64,
                       help='typed interaction embedding dimension')
    parser.add_argument('--geometry_scorer_typed_cutoff', type=float, default=6.0,
                       help='typed interaction atom-ligand cutoff in Angstrom')
    parser.add_argument('--geometry_scorer_typed_init_scale', type=float, default=0.1,
                       help='initial scale for typed interaction energy')
    parser.add_argument('--lambda_typed_candidate_energy', type=float, default=0.0,
                       help='typed candidate correct-vs-decoy energy loss weight')
    parser.add_argument('--typed_candidate_decoy_kind', type=str, default='scrambled',
                       choices=['scrambled', 'shuffled', 'nolig', 'translated'],
                       help='decoy type for typed candidate energy loss')
    parser.add_argument('--typed_candidate_contact_only', action='store_true', default=True,
                       help='restrict typed candidate energy loss to CA-ligand contact switch residues')
    parser.add_argument('--typed_candidate_no_contact_only', dest='typed_candidate_contact_only', action='store_false')
    parser.add_argument('--typed_candidate_margin', type=float, default=0.05,
                       help='margin for correct typed energy over decoy typed energy')
    parser.add_argument('--typed_candidate_noharm_weight', type=float, default=0.1,
                       help='auxiliary no-harm weight for typed candidate energy')
    parser.add_argument('--typed_candidate_noncontact_zero_weight', type=float, default=0.05,
                       help='auxiliary non-contact zero-energy weight for typed candidate energy')
    parser.add_argument('--lambda_switch_bce', type=float, default=0.0,
                       help='switch prediction BCE loss weight')
    parser.add_argument('--lambda_rescue_noharm', type=float, default=0.0,
                       help='rescue/no-harm margin loss weight')
    parser.add_argument('--lambda_ligand_residual', type=float, default=0.0,
                       help='ligand residual contrastive loss weight')
    parser.add_argument('--rescue_noharm_margin', type=float, default=0.5,
                       help='margin for rescue/no-harm loss')
    parser.add_argument('--ligand_residual_margin', type=float, default=0.3,
                       help='margin for ligand residual loss')
    parser.add_argument('--use_nolig_contrastive', action='store_true',
                       help='use no-ligand forward for contrastive')
    # === Phase-1 residual retraining (GPT-5.5 Pro plan) ===
    parser.add_argument('--reset_residual_and_gate_on_resume', action='store_true',
                       help='Reinitialize residual_mlp + gate_mlp after loading checkpoint (Phase-1)')
    parser.add_argument('--freeze_base_mlp', action='store_true',
                       help='Freeze geometry_candidate_scorer.base_mlp during training (Phase-1)')
    parser.add_argument('--detach_base_for_residual', action='store_true',
                       help='Use base.detach() in scorer forward so residual loss does not update base_mlp')
    parser.add_argument('--gate_warmup_open_steps', type=int, default=0,
                       help='Force gate=1.0 for first N steps (residual warmup, Phase-1)')
    parser.add_argument('--residual_beta_warmup_steps', type=int, default=0,
                       help='Linear ramp residual scale beta over N steps (Phase-1)')
    parser.add_argument('--residual_beta_min', type=float, default=0.1,
                       help='Initial beta during residual warmup ramp')
    parser.add_argument('--geometry_scorer_bounded_residual', action='store_true',
                       help='Apply tanh bound on residual logits')
    parser.add_argument('--geometry_scorer_residual_max', type=float, default=5.0)
    parser.add_argument('--geometry_scorer_residual_tau', type=float, default=2.0)
    parser.add_argument('--geometry_scorer_gate_norm', action='store_true',
                       help='Add LayerNorm on gate input features')
    parser.add_argument('--geometry_scorer_gate_clamp', type=float, default=6.0,
                       help='Clamp raw gate logit to +/- this value before sigmoid (0=disable)')
    parser.add_argument('--geometry_scorer_gate_init_bias', type=float, default=0.0,
                       help='Initial bias on gate final layer (sigmoid open factor)')
    parser.add_argument('--lambda_g_lift_switch', type=float, default=0.0,
                       help='Likelihood-ratio lift loss weight on switch residues (G_holo > G_apo + m)')
    parser.add_argument('--lambda_g_noharm', type=float, default=0.0,
                       help='Likelihood-ratio no-harm loss weight on apo-correct residues')
    parser.add_argument('--lambda_g_zero_noncontact', type=float, default=0.0,
                       help='||G_i||^2 regularizer on non-contact residues')
    parser.add_argument('--g_lift_margin', type=float, default=0.5,
                       help='Margin m for g_lift_switch_loss')
    parser.add_argument('--g_noharm_margin', type=float, default=0.5,
                       help='Margin m for g_noharm_loss')
    parser.add_argument('--g_noncontact_threshold', type=float, default=8.0,
                       help='CA-ligand distance (Å) above which residue is considered non-contact')
    # Phase-1 v2 G-vector posterior losses
    parser.add_argument('--lambda_g_switch_dir', type=float, default=0.0,
                       help='v2: CE on G/T toward holo bin (switch direction)')
    parser.add_argument('--lambda_g_switch_amp', type=float, default=0.0,
                       help='v2: hinge ensuring G_holo >= margin on switch residues')
    parser.add_argument('--lambda_g_switch_rank', type=float, default=0.0,
                       help='v2: hinge ensuring G_holo - max_other_G >= margin')
    parser.add_argument('--lambda_g_antiharm', type=float, default=0.0,
                       help='v2: relu(max_nonapo_G - tau) on apo-correct (replaces g_noharm)')
    parser.add_argument('--lambda_g_decoy', type=float, default=0.0,
                       help='v2: G_correct[holo] - G_decoy[holo] >= margin (ligand causality)')
    parser.add_argument('--g_switch_temperature', type=float, default=1.0,
                       help='v2: T in CE(G/T, holo)')
    parser.add_argument('--g_switch_amp_margin', type=float, default=0.05)
    parser.add_argument('--g_switch_rank_margin', type=float, default=0.05)
    parser.add_argument('--g_antiharm_tau', type=float, default=0.05)
    parser.add_argument('--g_decoy_margin', type=float, default=0.05)
    parser.add_argument('--g_switch_contact_only', action='store_true', default=True,
                       help='v2: restrict switch losses to contact residues (CA-ligand <= threshold)')
    parser.add_argument('--g_switch_no_contact_only', dest='g_switch_contact_only', action='store_false')
    parser.add_argument('--g_decoy_kind', type=str, default='translated',
                       choices=['translated', 'nolig', 'shuffled'])
    parser.add_argument('--g_decoy_translation_offset', type=float, default=100.0,
                       help='Å offset for translated-ligand decoy')
    parser.add_argument('--freeze_gate_mlp', action='store_true',
                       help='v2: freeze gate_mlp + gate_norm parameters (gate fixed at 1.0)')
    parser.add_argument('--chi1_rotamer_hidden', type=int, default=128,
                       help='χ1 rotamer posterior head hidden dim')
    parser.add_argument('--candidate_chi1_hidden', type=int, default=128,
                       help='candidate-aware χ1 scorer hidden dim')
    parser.add_argument('--contact_hidden', type=int, default=128,
                       help='contact posterior head hidden dim')
    parser.add_argument('--contact_loss_pos_weight', type=float, default=0.0,
                       help='contact BCE positive class weight；<=0 使用 batch-balanced 自动权重')
    parser.add_argument('--freeze_stage1_backbone_for_posteriors', action='store_true',
                       help='冻结 Stage-1 trunk/几何头，只训练 chi1 rotamer/contact posterior heads')
    parser.add_argument('--unfreeze_ligand_conditioner_for_posteriors', action='store_true',
                       help='posterior 训练时额外解冻 ligand conditioner，用于测试 ligand-conditioning bottleneck')
    parser.add_argument('--unfreeze_last_ipa_blocks_for_posteriors', type=int, default=0,
                       help='posterior 训练时额外解冻最后 N 个 IPA blocks（默认 0）')
    
    # Pocket Routing Adapter
    parser.add_argument('--use_pocket_routing_adapter', action='store_true',
                       help='Enable pocket-gated routing adapter between LigandConditioner and IPA')
    parser.add_argument('--pocket_routing_hidden', type=int, default=128,
                       help='Pocket routing adapter hidden dim')
    parser.add_argument('--pocket_routing_layers', type=int, default=2,
                       help='Pocket routing adapter MLP layers')
    parser.add_argument('--pocket_routing_gate_threshold', type=float, default=0.3,
                       help='w_res threshold for pocket routing gate')
    parser.add_argument('--pocket_routing_residual_scale', type=float, default=0.5,
                       help='Residual scale for pocket routing adapter')
    
    # 早停
    parser.add_argument('--patience', type=int, default=20,
                       help='早停patience')
    parser.add_argument('--selection_metric', type=str, default='total',
                       choices=[
                           'total', 'chi', 'fape', 'clash', 'candidate_rerank', 'typed_candidate_energy', 'chi1_acc', 'pocket_chi1_acc',
                           'chi1_rotamer_acc', 'contact_posterior_f1', 'pocket_irmsd', 'clash_pct',
                           'ligand_lift_switch_rotamer_acc',
                           'ligand_lift_apo_wrong_rotamer_acc',
                           'ligand_lift_contact_rotamer_acc',
                            'ligand_lift_contact_switch_rotamer_acc',
                            'ligand_lift_pocket_switch_rotamer_acc',
                            'ligand_decoy_lift_contact_switch_rotamer_acc',
                            'typed_energy_gap_switch',
                            'typed_energy_gap_contact',
                            'typed_energy_gap_contact_switch',
                            'typed_energy_gap_pocket_switch',
                            'candidate_decoy_lift_switch_rotamer_acc',
                            'candidate_decoy_lift_apo_wrong_rotamer_acc',
                            'candidate_decoy_lift_contact_rotamer_acc',
                            'candidate_decoy_lift_contact_switch_rotamer_acc',
                            'candidate_decoy_lift_pocket_switch_rotamer_acc',
                        ],
                       help='选模/早停指标：默认 total 保持历史行为兼容')

    # 保存
    parser.add_argument('--save_dir', type=str, default='checkpoints/stage1',
                       help='Checkpoint保存目录')
    parser.add_argument('--log_dir', type=str, default='logs/stage1',
                       help='日志目录')
    parser.add_argument('--metrics_filename', type=str, default='metrics.jsonl',
                       help='结构化 epoch 指标日志文件名（写入 log_dir）')
    parser.add_argument('--audit_filename', type=str, default='audit_metrics_latest.json',
                       help='审计指标快照文件名（写入 log_dir）')
    parser.add_argument('--no_save_latest_checkpoint', action='store_true',
                       help='禁用 latest_model.pt 保存')
    parser.add_argument('--save_epoch_checkpoints', action='store_true',
                       help='额外保存每个验证 epoch 的 epoch_{k}.pt')
    parser.add_argument('--compute_slow_metrics', action='store_true',
                       help='启用较慢的验证指标（如 pocket iRMSD / clash / contact）')
    parser.add_argument('--enable_dual_mask_audit', action='store_true',
                       help='启用 dual-mask 审计指标（apo/holo/ligand-facing 子集）')
    
    # 数据加载
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers (如遇shm不足可设为0)')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda',
                       help='训练设备（单卡模式）')
    parser.add_argument('--no_mixed_precision', action='store_true',
                       help='禁用混合精度')
    parser.add_argument('--amp_dtype', type=str, default='auto',
                       choices=['auto', 'fp16', 'bf16'],
                       help='AMP 精度类型：auto=优先 bf16（支持时），否则 fp16')
    
    # 分布式训练
    parser.add_argument('--distributed', action='store_true',
                       help='启用 DDP 分布式训练（配合 torchrun 使用）')
    parser.add_argument('--ddp_find_unused_parameters', action='store_true',
                       help='DDP unused-parameter detection for multi-head objectives where some heads are inactive')
    
    # 恢复训练
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从checkpoint恢复')
    
    args = parser.parse_args()
    
    # 自动检测 torchrun 环境
    if 'LOCAL_RANK' in os.environ:
        args.distributed = True
    
    return args


def main():
    """主函数"""
    args = parse_args()
    from src.stage1.training.trainer import Stage1Trainer
    
    # 分布式训练时，local_rank 从环境变量获取
    local_rank = int(os.environ.get('LOCAL_RANK', 0)) if args.distributed else 0
    is_main_process = (local_rank == 0)
    
    # 创建配置
    config = TrainingConfig(
        model_size=args.model_size,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_n_res=args.max_n_res,
        valid_samples_file=args.valid_samples_file,
        val_samples_file=args.val_samples_file,
        sample_metadata_file=args.sample_metadata_file,
        length_bucketed_sampling=args.length_bucketed_sampling,
        bucket_size_multiplier=args.bucket_size_multiplier,
        residue_budget=args.residue_budget,
        num_workers=args.num_workers,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler=args.lr_scheduler,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        min_lr_scale=args.min_lr_scale,
        pocket_warmup_steps=args.pocket_warmup_steps,
        ligand_gate_warmup_steps=args.ligand_gate_warmup_steps,
        max_epochs=args.max_epochs,
        grad_clip=args.grad_clip,
        w_fape=args.w_fape,
        w_chi=args.w_chi,
        w_clash=args.w_clash,
        lambda_pchi1=args.lambda_pchi1,
        pchi1_mask_mode=args.pchi1_mask_mode,
        pchi1_start_step=args.pchi1_start_step,
        pchi1_ramp_steps=args.pchi1_ramp_steps,
        use_pocket_chi1_expert=args.use_pocket_chi1_expert,
        pocket_chi1_expert_hidden=args.pocket_chi1_expert_hidden,
        pocket_chi1_expert_layers=args.pocket_chi1_expert_layers,
        pocket_chi1_gate_threshold=args.pocket_chi1_gate_threshold,
        pocket_chi1_residual_scale=args.pocket_chi1_residual_scale,
        lambda_chi1_rotamer=args.lambda_chi1_rotamer,
        lambda_contact=args.lambda_contact,
        lambda_candidate_chi1=args.lambda_candidate_chi1,
        lambda_ligand_contrastive=args.lambda_ligand_contrastive,
        ligand_contrastive_margin=args.ligand_contrastive_margin,
        lambda_candidate_rerank=args.lambda_candidate_rerank,
        candidate_rerank_margin=args.candidate_rerank_margin,
        candidate_rerank_rank_margin=args.candidate_rerank_rank_margin,
        candidate_rerank_contact_only=args.candidate_rerank_contact_only,
        candidate_rerank_decoy_kind=args.candidate_rerank_decoy_kind,
        candidate_scorer_lr_scale=args.candidate_scorer_lr_scale,
        lambda_geometry_chi1=args.lambda_geometry_chi1,
        lambda_base_prior=args.lambda_base_prior,
        geometry_scorer_lr_scale=args.geometry_scorer_lr_scale,
        geometry_scorer_use_sgeo=args.geometry_scorer_use_sgeo,
        geometry_scorer_sgeo_dim=args.geometry_scorer_sgeo_dim,
        geometry_scorer_use_typed_energy=args.geometry_scorer_use_typed_energy,
        geometry_scorer_typed_pair_dim=args.geometry_scorer_typed_pair_dim,
        geometry_scorer_typed_cutoff=args.geometry_scorer_typed_cutoff,
        geometry_scorer_typed_init_scale=args.geometry_scorer_typed_init_scale,
        lambda_typed_candidate_energy=args.lambda_typed_candidate_energy,
        typed_candidate_decoy_kind=args.typed_candidate_decoy_kind,
        typed_candidate_contact_only=args.typed_candidate_contact_only,
        typed_candidate_margin=args.typed_candidate_margin,
        typed_candidate_noharm_weight=args.typed_candidate_noharm_weight,
        typed_candidate_noncontact_zero_weight=args.typed_candidate_noncontact_zero_weight,
        lambda_switch_bce=args.lambda_switch_bce,
        lambda_rescue_noharm=args.lambda_rescue_noharm,
        lambda_ligand_residual=args.lambda_ligand_residual,
        rescue_noharm_margin=args.rescue_noharm_margin,
        ligand_residual_margin=args.ligand_residual_margin,
        use_nolig_contrastive=args.use_nolig_contrastive,
        # Phase-1 controls
        reset_residual_and_gate_on_resume=args.reset_residual_and_gate_on_resume,
        freeze_base_mlp=args.freeze_base_mlp,
        detach_base_for_residual=args.detach_base_for_residual,
        gate_warmup_open_steps=args.gate_warmup_open_steps,
        residual_beta_warmup_steps=args.residual_beta_warmup_steps,
        residual_beta_min=args.residual_beta_min,
        geometry_scorer_bounded_residual=args.geometry_scorer_bounded_residual,
        geometry_scorer_residual_max=args.geometry_scorer_residual_max,
        geometry_scorer_residual_tau=args.geometry_scorer_residual_tau,
        geometry_scorer_gate_norm=args.geometry_scorer_gate_norm,
        geometry_scorer_gate_clamp=args.geometry_scorer_gate_clamp,
        geometry_scorer_gate_init_bias=args.geometry_scorer_gate_init_bias,
        lambda_g_lift_switch=args.lambda_g_lift_switch,
        lambda_g_noharm=args.lambda_g_noharm,
        lambda_g_zero_noncontact=args.lambda_g_zero_noncontact,
        g_lift_margin=args.g_lift_margin,
        g_noharm_margin=args.g_noharm_margin,
        g_noncontact_threshold=args.g_noncontact_threshold,
        # Phase-1 v2
        lambda_g_switch_dir=args.lambda_g_switch_dir,
        lambda_g_switch_amp=args.lambda_g_switch_amp,
        lambda_g_switch_rank=args.lambda_g_switch_rank,
        lambda_g_antiharm=args.lambda_g_antiharm,
        lambda_g_decoy=args.lambda_g_decoy,
        g_switch_temperature=args.g_switch_temperature,
        g_switch_amp_margin=args.g_switch_amp_margin,
        g_switch_rank_margin=args.g_switch_rank_margin,
        g_antiharm_tau=args.g_antiharm_tau,
        g_decoy_margin=args.g_decoy_margin,
        g_switch_contact_only=args.g_switch_contact_only,
        g_decoy_kind=args.g_decoy_kind,
        g_decoy_translation_offset=args.g_decoy_translation_offset,
        freeze_gate_mlp=args.freeze_gate_mlp,
        chi1_rotamer_hidden=args.chi1_rotamer_hidden,
        candidate_chi1_hidden=args.candidate_chi1_hidden,
        contact_hidden=args.contact_hidden,
        contact_loss_pos_weight=args.contact_loss_pos_weight,
        freeze_stage1_backbone_for_posteriors=args.freeze_stage1_backbone_for_posteriors,
        unfreeze_ligand_conditioner_for_posteriors=args.unfreeze_ligand_conditioner_for_posteriors,
        unfreeze_last_ipa_blocks_for_posteriors=args.unfreeze_last_ipa_blocks_for_posteriors,
        use_pocket_routing_adapter=args.use_pocket_routing_adapter,
        pocket_routing_hidden=args.pocket_routing_hidden,
        pocket_routing_layers=args.pocket_routing_layers,
        pocket_routing_gate_threshold=args.pocket_routing_gate_threshold,
        pocket_routing_residual_scale=args.pocket_routing_residual_scale,
        early_stop_patience=args.patience,
        selection_metric=args.selection_metric,
        save_latest_checkpoint=not args.no_save_latest_checkpoint,
        save_epoch_checkpoints=args.save_epoch_checkpoints,
        metrics_filename=args.metrics_filename,
        audit_filename=args.audit_filename,
        compute_slow_metrics=args.compute_slow_metrics,
        enable_dual_mask_audit=args.enable_dual_mask_audit,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        device=args.device,
        mixed_precision=not args.no_mixed_precision,
        amp_dtype=args.amp_dtype,
        distributed=args.distributed,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        resume_from=args.resume_from,
    )

    if config.selection_metric in {'pocket_irmsd', 'clash_pct'} and not config.compute_slow_metrics:
        config.compute_slow_metrics = True

    # 只在主进程打印配置
    if is_main_process:
        print(f"\n{'='*80}")
        print(f"BINDRAE Stage-1 训练")
        print(f"{'='*80}")
        print(f"\n配置:")
        print(f"  - 模型规模: {config.model_size}")
        print(f"  - 数据目录: {config.data_dir}")
        print(f"  - 批大小: {config.batch_size}")
        print(f"  - 最大残基数: {config.max_n_res}")
        print(f"  - 长度 bucket: {config.length_bucketed_sampling}")
        print(f"  - 残基预算: {config.residue_budget}")
        print(f"  - 学习率: {config.lr}")
        print(f"  - 调度 warmup: {config.warmup_steps}")
        print(f"  - LR scheduler: {config.lr_scheduler}")
        if config.lr_scheduler == 'plateau':
            print(f"  - plateau factor: {config.plateau_factor}")
            print(f"  - plateau patience: {config.plateau_patience}")
            print(f"  - min lr scale: {config.min_lr_scale}")
        print(f"  - pocket warmup: {config.pocket_warmup_steps}")
        print(f"  - ligand gate warmup: {config.ligand_gate_warmup_steps}")
        print(f"  - 最大轮数: {config.max_epochs}")
        print(f"  - 早停patience: {config.early_stop_patience}")
        print(f"  - 选模指标: {config.selection_metric}")
        print(f"  - 损失权重: w_fape={config.w_fape}, w_chi={config.w_chi}, w_clash={config.w_clash}, lambda_pchi1={config.lambda_pchi1}, lambda_chi1_rotamer={config.lambda_chi1_rotamer}, lambda_contact={config.lambda_contact}, lambda_candidate_chi1={config.lambda_candidate_chi1}, lambda_ligand_contrastive={config.lambda_ligand_contrastive}, lambda_candidate_rerank={config.lambda_candidate_rerank}, lambda_typed_candidate_energy={config.lambda_typed_candidate_energy}")
        print(f"  - ligand contrastive margin: {config.ligand_contrastive_margin}")
        print(f"  - candidate rerank: margin={config.candidate_rerank_margin}, rank_margin={config.candidate_rerank_rank_margin}, contact_only={config.candidate_rerank_contact_only}, decoy={config.candidate_rerank_decoy_kind}")
        print(f"  - typed candidate energy: enabled={config.geometry_scorer_use_typed_energy}, decoy={config.typed_candidate_decoy_kind}, margin={config.typed_candidate_margin}, contact_only={config.typed_candidate_contact_only}")
        print(f"  - pchi1 mask: {config.pchi1_mask_mode}")
        print(f"  - pchi1 start step: {config.pchi1_start_step}")
        print(f"  - pchi1 ramp steps: {config.pchi1_ramp_steps}")
        print(f"  - posterior heads: chi1_rotamer={config.lambda_chi1_rotamer > 0.0}, contact={config.lambda_contact > 0.0}, candidate_chi1={(config.lambda_candidate_chi1 > 0.0 or config.lambda_ligand_contrastive > 0.0 or config.lambda_candidate_rerank > 0.0)}")
        print(f"  - freeze Stage-1 backbone for posteriors: {config.freeze_stage1_backbone_for_posteriors}")
        print(f"  - pocket chi1 expert: {config.use_pocket_chi1_expert}")
        if config.use_pocket_chi1_expert:
            print(f"  - pocket chi1 expert hidden: {config.pocket_chi1_expert_hidden}")
            print(f"  - pocket chi1 expert layers: {config.pocket_chi1_expert_layers}")
            print(f"  - pocket chi1 gate threshold: {config.pocket_chi1_gate_threshold}")
            print(f"  - pocket chi1 residual scale: {config.pocket_chi1_residual_scale}")
        print(f"  - latest checkpoint: {config.save_latest_checkpoint}")
        print(f"  - epoch checkpoints: {config.save_epoch_checkpoints}")
        print(f"  - slow metrics: {config.compute_slow_metrics}")
        print(f"  - dual-mask audit: {config.enable_dual_mask_audit}")
        print(f"  - metrics 文件: {Path(config.log_dir) / config.metrics_filename}")
        print(f"  - audit 文件: {Path(config.log_dir) / config.audit_filename}")
        print(f"  - 设备: {config.device}")
        print(f"  - 混合精度: {config.mixed_precision}")
        print(f"  - AMP dtype: {config.amp_dtype}")
        print(f"  - 分布式训练: {config.distributed}")
        print(f"\n{'='*80}\n")
    
    # 创建训练器
    trainer = Stage1Trainer(config)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
