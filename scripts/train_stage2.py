#!/usr/bin/env python3
"""
Stage-2 训练启动脚本

Usage:
    python scripts/train_stage2.py [--options]
"""

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

from src.stage2.training.config import TrainingConfig
from src.stage2.training.trainer import Stage2Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Stage-2 训练')

    # 数据
    parser.add_argument('--data_dir', type=str, default='data/apo_holo_triplets',
                        help='数据目录')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批大小')
    parser.add_argument('--valid_samples_file', type=str, default=None,
                        help='训练样本筛选文件')
    parser.add_argument('--val_samples_file', type=str, default=None,
                        help='验证样本筛选文件')

    # 训练
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='最大训练轮数')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='梯度累积步数')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker 数')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='学习率 warmup 步数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--val_t', type=float, default=0.5,
                        help='Validation t value; set negative for random validation t')

    # ESM representation adapter
    parser.add_argument('--esm_fusion_enabled', action='store_true',
                        help='Enable last-K ESM layer fusion before the Stage-2 geometry trunk')
    parser.add_argument('--esm_num_layers', type=int, default=1,
                        help='Number of ESM layers expected when --esm_fusion_enabled is active')
    parser.add_argument('--esm_fusion_mode', type=str, default='sum',
                        choices=['sum', 'mean', 'softmax_weighted'],
                        help='How to fuse [B,N,K,D] ESM features')
    parser.add_argument('--esm_layer_dropout', type=float, default=0.0,
                        help='Dropout applied to learned ESM layer weights during training')

    # Stage-1 prior
    parser.add_argument('--stage1_ckpt', type=str, default='checkpoints/stage1_best.pt',
                        help='Stage-1 checkpoint')
    parser.add_argument('--no_stage1_prior', action='store_true',
                        help='禁用Stage-1 prior')
    parser.add_argument('--stage1_prior_mode', type=str, default='stage1',
                        choices=[
                            'stage1', 'apo_chi', 'holo_chi', 'holo_chi_apo_rigid',
                            'apo_chi_holo_rigid', 'holo_ligand_facing_chi', 'noisy_stage1_chi'
                        ],
                        help='Stage-1 prior diagnostic mode')
    parser.add_argument('--stage1_prior_noise_scale', type=float, default=0.0,
                        help='noisy_stage1_chi 模式下添加到 chi prior 的噪声尺度（弧度）')
    parser.add_argument('--no_stage1_rigid_prior', action='store_true',
                        help='仅使用 Stage-1 χ prior，不注入/惩罚 Stage-1 rigid prior')
    parser.add_argument('--stage1_chi_feature_scale', type=float, default=1.0,
                        help='Stage-1 delta-χ input feature scaling factor')
    parser.add_argument('--w_prior', type=float, default=0.1,
                        help='Stage-1 prior loss 权重')
    parser.add_argument('--t_mid', type=float, default=0.5,
                        help='Prior loss 生效的起始时间')
    parser.add_argument('--use_pocket_local_prior', action='store_true',
                        help='仅在 pocket 残基上施加 Stage-1 prior loss')
    parser.add_argument('--prior_pocket_threshold', type=float, default=0.3,
                        help='Pocket-local prior 的 w_res 阈值')
    parser.add_argument('--interaction_prior_ckpt', type=str, default=None,
                        help='显式 residue-ligand interaction prior checkpoint')
    parser.add_argument('--w_interaction_prior', type=float, default=0.0,
                        help='Interaction prior soft contact loss 权重')
    parser.add_argument('--interaction_prior_min_score', type=float, default=0.0,
                        help='只在 prior probability 超过该阈值的残基上施加 interaction prior')
    parser.add_argument('--interaction_prior_temperature', type=float, default=1.0,
                        help='Interaction prior logit temperature')
    parser.add_argument('--interaction_prior_contact_dist', type=float, default=4.5,
                        help='Interaction prior soft contact 距离阈值')
    parser.add_argument('--interaction_prior_contact_tau', type=float, default=0.75,
                        help='Interaction prior soft contact sigmoid 温度')
    parser.add_argument('--interaction_prior_t_mid', type=float, default=0.3,
                        help='Interaction prior loss 生效的起始路径时间')
    parser.add_argument('--interaction_prior_feature_mode', type=str, default='none',
                        choices=['none', 'prior', 'zero', 'oracle_contact'],
                        help='将 interaction prior 作为 Stage-2 per-residue 输入特征: none/prior/zero/oracle_contact')
    parser.add_argument('--interaction_prior_feature_scale', type=float, default=1.0,
                        help='Interaction prior 输入特征缩放')
    parser.add_argument('--stage1v2_posterior_feature_mode', type=str, default='none',
                        choices=[
                            'none', 'zero', 'student', 'student_shuffled',
                            'oracle_holo_truth', 'external_teacher_cached',
                            'oracle_motion', 'oracle_motion_residue_shuffled',
                            'oracle_motion_sample_shuffled',
                        ],
                        help='Stage-1-v2 posterior cache feature mode')
    parser.add_argument('--stage1v2_train_cache_dir', type=str, default=None,
                        help='Stage-1-v2 student/external posterior cache dir for train split')
    parser.add_argument('--stage1v2_val_cache_dir', type=str, default=None,
                        help='Stage-1-v2 student/external posterior cache dir for val split')
    parser.add_argument('--stage1v2_train_label_dir', type=str, default=None,
                        help='Stage-1-v2 holo_truth/oracle label dir for train split')
    parser.add_argument('--stage1v2_val_label_dir', type=str, default=None,
                        help='Stage-1-v2 holo_truth/oracle label dir for val split')
    parser.add_argument('--stage1v2_posterior_feature_names', type=str,
                        default='contact_prob,active_prob,approach_prob,release_prob,confidence,teacher_min_dist_pred_norm,signed_delta_dist_pred_norm',
                        help='Comma-separated Stage-1-v2 per-residue feature names')
    parser.add_argument('--stage1v2_posterior_feature_scale', type=float, default=1.0,
                        help='Stage-1-v2 posterior feature scale before model input')
    parser.add_argument('--stage1v2_loss_weight_mode', type=str, default='none',
                        choices=['none', 'contact', 'active', 'contact_active'],
                        help='Use Stage-1-v2 posterior to reweight Stage-2 FM/path losses')
    parser.add_argument('--stage1v2_loss_weight_alpha', type=float, default=0.0,
                        help='Strength for Stage-1-v2 posterior loss reweighting')
    parser.add_argument('--w_stage1v2_guidance', type=float, default=0.0,
                        help='Weight for Stage-1-v2 posterior contact guidance loss')
    parser.add_argument('--stage1v2_guidance_feature', type=str, default='contact_prob',
                        help='Stage-1-v2 posterior feature used as contact-guidance target')
    parser.add_argument('--stage1v2_guidance_min_prob', type=float, default=0.0,
                        help='Only guide residues with posterior probability above this value')
    parser.add_argument('--stage1v2_guidance_t_mid', type=float, default=0.3,
                        help='Path time from which Stage-1-v2 contact guidance is active')
    parser.add_argument('--repa_enabled', action='store_true',
                        help='Enable REPA-style hidden-state alignment to Stage-1-v2/oracle-motion features')
    parser.add_argument('--repa_weight', type=float, default=0.0,
                        help='Loss weight for REPA-style hidden-state alignment')
    parser.add_argument('--repa_dim', type=int, default=128,
                        help='Hidden width of the REPA projection head')
    parser.add_argument('--repa_loss_type', type=str, default='cosine',
                        choices=['cosine', 'mse'],
                        help='REPA alignment loss type')
    parser.add_argument('--repa_mask_mode', type=str, default='motion_active_or_pocket',
                        choices=['node', 'pocket', 'motion_active', 'motion_active_or_pocket'],
                        help='Residue mask used by REPA alignment')
    parser.add_argument('--repa_target_shuffle_mode', type=str, default='none',
                        choices=['none', 'residue'],
                        help='Shuffle only the REPA target while leaving Stage-2 conditioning features unchanged')
    parser.add_argument('--w_contact', type=float, default=0.1,
                        help='Stage-2 path contact loss 权重')
    parser.add_argument('--contact_loss_mode', type=str, default='holo_target',
                        choices=['holo_target', 'monotonic_increase'],
                        help='Contact path loss: holo_target fits apo->holo contact trajectory; monotonic_increase is legacy')
    parser.add_argument('--contact_eps', type=float, default=0.0,
                        help='Contact monotonicity deadband for legacy monotonic_increase mode')
    parser.add_argument('--contact_direction_eps', type=float, default=1e-3,
                        help='Minimum apo-holo contact-score delta used for direction accuracy')

    # NMA
    parser.add_argument('--use_nma', action='store_true',
                        help='启用NMA特征')
    parser.add_argument('--nma_dim', type=int, default=0,
                        help='NMA特征维度')

    # 保存
    parser.add_argument('--save_dir', type=str, default='checkpoints/stage2',
                        help='Checkpoint保存目录')
    parser.add_argument('--log_dir', type=str, default='logs/stage2',
                        help='日志目录')

    # 设备
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备')
    parser.add_argument('--no_mixed_precision', action='store_true',
                        help='禁用混合精度')
    parser.add_argument('--amp_dtype', type=str, default='bf16',
                        choices=['auto', 'bf16', 'fp16'],
                        help='混合精度 dtype')

    # 分布式
    parser.add_argument('--distributed', action='store_true',
                        help='启用 DDP 分布式训练')

    # Smoke-test / geometry control
    parser.add_argument('--n_integration_steps', type=int, default=5,
                        help='路径积分步数')
    parser.add_argument('--integration_chi_clip', type=float, default=1.0,
                        help='路径积分时 chi velocity 裁剪；<=0 表示不裁剪')
    parser.add_argument('--integration_rot_clip', type=float, default=0.1,
                        help='路径积分时 SO(3) log velocity 裁剪；<=0 表示不裁剪')
    parser.add_argument('--integration_trans_clip', type=float, default=0.2,
                        help='路径积分时 translation velocity 裁剪；<=0 表示不裁剪')
    parser.add_argument('--n_geom_steps', type=int, default=4,
                        help='几何损失采样步数')
    parser.add_argument('--geom_loss_every_n_steps', type=int, default=5,
                        help='每 N 个优化步计算一次几何损失')

    return parser.parse_args()


def main():
    args = parse_args()

    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        valid_samples_file=args.valid_samples_file,
        val_samples_file=args.val_samples_file,
        lr=args.lr,
        max_epochs=args.max_epochs,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.accum_steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        val_t=args.val_t if args.val_t >= 0.0 else None,
        esm_fusion_enabled=args.esm_fusion_enabled,
        esm_num_layers=args.esm_num_layers,
        esm_fusion_mode=args.esm_fusion_mode,
        esm_layer_dropout=args.esm_layer_dropout,
        stage1_ckpt=args.stage1_ckpt,
        use_stage1_prior=not args.no_stage1_prior,
        stage1_prior_mode=args.stage1_prior_mode,
        stage1_prior_noise_scale=args.stage1_prior_noise_scale,
        use_stage1_rigid_prior=not args.no_stage1_rigid_prior,
        stage1_chi_feature_scale=args.stage1_chi_feature_scale,
        use_pocket_local_prior=args.use_pocket_local_prior,
        prior_pocket_threshold=args.prior_pocket_threshold,
        interaction_prior_ckpt=args.interaction_prior_ckpt,
        w_interaction_prior=args.w_interaction_prior,
        interaction_prior_min_score=args.interaction_prior_min_score,
        interaction_prior_temperature=args.interaction_prior_temperature,
        interaction_prior_contact_dist=args.interaction_prior_contact_dist,
        interaction_prior_contact_tau=args.interaction_prior_contact_tau,
        interaction_prior_t_mid=args.interaction_prior_t_mid,
        interaction_prior_feature_mode=args.interaction_prior_feature_mode,
        interaction_prior_feature_scale=args.interaction_prior_feature_scale,
        stage1v2_posterior_feature_mode=args.stage1v2_posterior_feature_mode,
        stage1v2_train_cache_dir=args.stage1v2_train_cache_dir,
        stage1v2_val_cache_dir=args.stage1v2_val_cache_dir,
        stage1v2_train_label_dir=args.stage1v2_train_label_dir,
        stage1v2_val_label_dir=args.stage1v2_val_label_dir,
        stage1v2_posterior_feature_names=args.stage1v2_posterior_feature_names,
        stage1v2_posterior_feature_scale=args.stage1v2_posterior_feature_scale,
        stage1v2_loss_weight_mode=args.stage1v2_loss_weight_mode,
        stage1v2_loss_weight_alpha=args.stage1v2_loss_weight_alpha,
        w_stage1v2_guidance=args.w_stage1v2_guidance,
        stage1v2_guidance_feature=args.stage1v2_guidance_feature,
        stage1v2_guidance_min_prob=args.stage1v2_guidance_min_prob,
        stage1v2_guidance_t_mid=args.stage1v2_guidance_t_mid,
        repa_enabled=args.repa_enabled,
        repa_weight=args.repa_weight,
        repa_dim=args.repa_dim,
        repa_loss_type=args.repa_loss_type,
        repa_mask_mode=args.repa_mask_mode,
        repa_target_shuffle_mode=args.repa_target_shuffle_mode,
        w_contact=args.w_contact,
        contact_loss_mode=args.contact_loss_mode,
        contact_eps=args.contact_eps,
        contact_direction_eps=args.contact_direction_eps,
        w_prior=args.w_prior,
        t_mid=args.t_mid,
        use_nma=args.use_nma,
        nma_dim=args.nma_dim,
        n_integration_steps=args.n_integration_steps,
        integration_chi_clip=args.integration_chi_clip,
        integration_rot_clip=args.integration_rot_clip,
        integration_trans_clip=args.integration_trans_clip,
        n_geom_steps=args.n_geom_steps,
        geom_loss_every_n_steps=args.geom_loss_every_n_steps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        device=args.device,
        mixed_precision=not args.no_mixed_precision,
        amp_dtype=args.amp_dtype,
        distributed=args.distributed,
    )

    print(f"\n{'='*80}")
    print("BINDRAE Stage-2 训练")
    print(f"{'='*80}")
    print("\n配置:")
    print(f"  - 数据目录: {config.data_dir}")
    print(f"  - 批大小: {config.batch_size}")
    print(f"  - 学习率: {config.lr}")
    print(f"  - 最大轮数: {config.max_epochs}")
    print(f"  - ESM fusion enabled: {config.esm_fusion_enabled}")
    print(f"  - ESM num layers: {config.esm_num_layers}")
    print(f"  - ESM fusion mode: {config.esm_fusion_mode}")
    print(f"  - ESM layer dropout: {config.esm_layer_dropout}")
    print(f"  - Stage-1 prior: {config.use_stage1_prior}")
    print(f"  - Stage-1 prior mode: {config.stage1_prior_mode}")
    print(f"  - Stage-1 prior noise scale: {config.stage1_prior_noise_scale}")
    print(f"  - Stage-1 rigid prior: {config.use_stage1_rigid_prior}")
    print(f"  - Stage-1 chi feature scale: {config.stage1_chi_feature_scale}")
    print(f"  - Stage-1 ckpt: {config.stage1_ckpt if config.use_stage1_prior else 'OFF'}")
    print(f"  - pocket-local prior: {config.use_pocket_local_prior}")
    print(f"  - prior_pocket_threshold: {config.prior_pocket_threshold}")
    print(f"  - interaction prior ckpt: {config.interaction_prior_ckpt or 'OFF'}")
    print(f"  - w_interaction_prior: {config.w_interaction_prior}")
    print(f"  - interaction_prior_min_score: {config.interaction_prior_min_score}")
    print(f"  - interaction_prior_temperature: {config.interaction_prior_temperature}")
    print(f"  - interaction_prior_contact_dist: {config.interaction_prior_contact_dist}")
    print(f"  - interaction_prior_contact_tau: {config.interaction_prior_contact_tau}")
    print(f"  - interaction_prior_t_mid: {config.interaction_prior_t_mid}")
    print(f"  - interaction_prior_feature_mode: {config.interaction_prior_feature_mode}")
    print(f"  - interaction_prior_feature_scale: {config.interaction_prior_feature_scale}")
    print(f"  - stage1v2_posterior_feature_mode: {config.stage1v2_posterior_feature_mode}")
    print(f"  - stage1v2_train_cache_dir: {config.stage1v2_train_cache_dir or 'OFF'}")
    print(f"  - stage1v2_val_cache_dir: {config.stage1v2_val_cache_dir or 'OFF'}")
    print(f"  - stage1v2_train_label_dir: {config.stage1v2_train_label_dir or 'OFF'}")
    print(f"  - stage1v2_val_label_dir: {config.stage1v2_val_label_dir or 'OFF'}")
    print(f"  - stage1v2_features: {config.stage1v2_posterior_feature_names}")
    print(f"  - stage1v2_feature_scale: {config.stage1v2_posterior_feature_scale}")
    print(f"  - stage1v2_loss_weight_mode: {config.stage1v2_loss_weight_mode}")
    print(f"  - stage1v2_loss_weight_alpha: {config.stage1v2_loss_weight_alpha}")
    print(f"  - w_stage1v2_guidance: {config.w_stage1v2_guidance}")
    print(f"  - stage1v2_guidance_feature: {config.stage1v2_guidance_feature}")
    print(f"  - stage1v2_guidance_min_prob: {config.stage1v2_guidance_min_prob}")
    print(f"  - stage1v2_guidance_t_mid: {config.stage1v2_guidance_t_mid}")
    print(f"  - REPA enabled: {config.repa_enabled}")
    print(f"  - REPA weight: {config.repa_weight}")
    print(f"  - REPA dim: {config.repa_dim}")
    print(f"  - REPA loss type: {config.repa_loss_type}")
    print(f"  - REPA mask mode: {config.repa_mask_mode}")
    print(f"  - REPA target shuffle mode: {config.repa_target_shuffle_mode}")
    print(f"  - w_contact: {config.w_contact}")
    print(f"  - contact_loss_mode: {config.contact_loss_mode}")
    print(f"  - contact_eps: {config.contact_eps}")
    print(f"  - contact_direction_eps: {config.contact_direction_eps}")
    print(f"  - w_prior: {config.w_prior}")
    print(f"  - t_mid: {config.t_mid}")
    print(f"  - warmup_steps: {config.warmup_steps}")
    print(f"  - num_workers: {config.num_workers}")
    print(f"  - n_integration_steps: {config.n_integration_steps}")
    print(f"  - integration_chi_clip: {config.integration_chi_clip}")
    print(f"  - integration_rot_clip: {config.integration_rot_clip}")
    print(f"  - integration_trans_clip: {config.integration_trans_clip}")
    print(f"  - n_geom_steps: {config.n_geom_steps}")
    print(f"  - geom_loss_every_n_steps: {config.geom_loss_every_n_steps}")
    print(f"  - seed: {config.seed}")
    print(f"  - val_t: {config.val_t}")
    print(f"  - NMA: {config.use_nma}")
    print(f"  - 设备: {config.device}")
    print(f"  - 混合精度: {config.mixed_precision}")
    print(f"  - AMP dtype: {config.amp_dtype}")
    print(f"\n{'='*80}\n")

    trainer = Stage2Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
