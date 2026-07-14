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
    parser.add_argument('--val_batch_size', type=int, default=None,
                        help='Validation batch size; defaults to training batch size')
    parser.add_argument('--valid_samples_file', type=str, default=None,
                        help='训练样本筛选文件')
    parser.add_argument('--val_samples_file', type=str, default=None,
                        help='验证样本筛选文件')
    parser.add_argument('--val_split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='Dataset split used for validation loader')
    parser.add_argument('--trust_prechecked_samples', action='store_true',
                        help='Skip startup file/cache existence scans when valid sample files were prechecked')

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
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='DataLoader prefetch batches per worker when num_workers > 0')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='学习率 warmup 步数')
    parser.add_argument('--early_stop_patience', type=int, default=20,
                        help='Validation epochs without improvement before early stopping')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--val_t', type=float, default=0.5,
                        help='Validation t value; set negative for random validation t')
    parser.add_argument('--path_parameterization', type=str, default='flow',
                        choices=[
                            'flow',
                            'boundary_residual_v1',
                            'boundary_residual',
                            'projected_flow',
                            'bridge_timewarp_v1',
                            'phase_orthogonal_residual_v1',
                        ],
                        help='Path construction used by geometry losses/evaluation')
    parser.add_argument('--boundary_residual_envelope', type=str, default='sin2',
                        choices=['sin2', 'poly'],
                        help='Endpoint-zero envelope for boundary_residual paths')
    parser.add_argument('--boundary_residual_scale', type=float, default=1.0,
                        help='Scale applied to boundary_residual model outputs')
    parser.add_argument('--terminal_projection_schedule', type=str, default='smootherstep',
                        choices=['smoothstep', 'smootherstep', 'late_smoother', 'quadratic'],
                        help='Correction schedule for projected_flow terminal projection')
    parser.add_argument('--time_warp_logit_scale', type=float, default=1.0,
                        help='Scale applied to bridge_timewarp_v1 logits before softplus rates')
    parser.add_argument('--time_warp_rate_eps', type=float, default=1e-3,
                        help='Minimum positive interval rate for bridge_timewarp_v1')
    parser.add_argument('--time_warp_rate_clip', type=float, default=10.0,
                        help='Maximum interval rate for bridge_timewarp_v1; <=0 disables clipping')
    parser.add_argument('--phase_residual_tau_mode', type=str, default='learned',
                        choices=['learned', 'identity'],
                        help='Use learned residue phase or synchronous identity phase')
    parser.add_argument('--phase_residual_bridge_mode', type=str,
                        default='se3_geodesic',
                        choices=['se3_geodesic', 'cartesian_backbone'],
                        help='Reference backbone bridge used by the phase model')
    parser.add_argument('--phase_residual_envelope', type=str, default='poly',
                        choices=['poly', 'sin2'],
                        help='Endpoint-zero envelope for phase residuals')
    parser.add_argument('--phase_residual_scale', type=float, default=1.0,
                        help='Global scale applied after tangent-normal projection')
    parser.add_argument('--phase_residual_rotation_metric_scale', type=float, default=1.0,
                        help='Characteristic rotation scale in radians for the product metric')
    parser.add_argument('--phase_residual_translation_metric_scale', type=float, default=1.0,
                        help='Characteristic translation scale in Angstrom for the product metric')
    parser.add_argument('--phase_residual_chi_metric_scale', type=float, default=1.0,
                        help='Characteristic chi scale in radians for the product metric')
    parser.add_argument('--phase_residual_min_tangent_norm', type=float, default=1e-3,
                        help='Disable spatial residuals below this endpoint-motion norm')
    parser.add_argument('--phase_residual_max_metric_norm', type=float, default=0.0,
                        help='Per-residue product-metric residual cap; <=0 disables the cap')
    parser.add_argument('--phase_residual_peptide_retraction', action='store_true',
                        help='Apply differentiable bounded peptide retraction to interior phase-normal frames')
    parser.add_argument('--phase_residual_peptide_retraction_iterations', type=int, default=8)
    parser.add_argument('--phase_residual_peptide_retraction_relaxation', type=float, default=0.75)
    parser.add_argument('--phase_residual_peptide_retraction_anchor_strength', type=float, default=0.02)
    parser.add_argument('--phase_residual_peptide_retraction_max_translation', type=float, default=1.0)
    parser.add_argument('--phase_residual_peptide_retraction_activation_loss_threshold', type=float, default=0.0)
    parser.add_argument('--init_from_checkpoint', type=str, default=None,
                        help='Warm-start model weights only; does not restore optimizer, scheduler, or epoch')
    parser.add_argument('--teacher_residual_cache_dir', type=str, default=None,
                        help='Optional cache of free-flow teacher residuals relative to the apo-holo bridge')
    parser.add_argument('--w_teacher_residual', type=float, default=0.0,
                        help='Loss weight for boundary-residual teacher-shape distillation')
    parser.add_argument('--teacher_residual_loss_type', type=str, default='mse',
                        choices=['mse', 'huber'],
                        help='Loss type for teacher residual distillation')
    parser.add_argument('--teacher_residual_huber_delta', type=float, default=1.0,
                        help='Smooth-L1 beta for teacher residual Huber loss')
    parser.add_argument('--teacher_residual_t_min', type=float, default=0.08,
                        help='Earliest interior t supervised by teacher residual cache')
    parser.add_argument('--teacher_residual_t_max', type=float, default=0.92,
                        help='Latest interior t supervised by teacher residual cache')
    parser.add_argument('--teacher_residual_mask_mode', type=str, default='motion_active_or_pocket',
                        choices=[
                            'node',
                            'pocket',
                            'motion_active',
                            'motion_active_or_pocket',
                            'clash_relief',
                            'clash_relief_or_motion_active',
                            'clash_relief_or_pocket',
                        ],
                        help='Residue mask used for teacher residual distillation')
    parser.add_argument('--teacher_residual_clash_weight_threshold', type=float, default=1e-4,
                        help='Minimum cached clash-relief weight for clash-relief teacher masks')
    parser.add_argument('--teacher_residual_missing_policy', type=str, default='error',
                        choices=['error', 'skip'],
                        help='How to handle samples missing teacher residual cache files')
    parser.add_argument('--phase_teacher_cache_dir', type=str, default=None,
                        help='Optional phase_teacher_v1 pseudo-label cache directory')
    parser.add_argument('--w_phase_teacher', type=float, default=0.0,
                        help='Loss weight for confidence-weighted phase pseudo-teacher distillation')
    parser.add_argument('--phase_teacher_loss_type', type=str, default='huber',
                        choices=['mse', 'huber'])
    parser.add_argument('--phase_teacher_huber_delta', type=float, default=0.1)
    parser.add_argument('--phase_teacher_mask_mode', type=str, default='contact_event',
                        choices=['contact_event', 'formed_contact', 'approach', 'active', 'pocket', 'node'])
    parser.add_argument('--phase_teacher_min_confidence', type=float, default=0.05)
    parser.add_argument('--phase_teacher_missing_policy', type=str, default='error',
                        choices=['error', 'skip'])
    parser.add_argument('--phase_teacher_head_only', action='store_true',
                        help='Freeze all parameters except time_warp_head for a phase learnability diagnostic')
    parser.add_argument('--phase_teacher_residual_heads_only', action='store_true',
                        help='Freeze the shared trunk and train only phase/residual heads')
    parser.add_argument('--phase_normal_cache_dir', type=str, default=None,
                        help='Audited md_phase_normal_v1 cache for phase-normal residual heads')
    parser.add_argument('--w_phase_normal_residual', type=float, default=0.0,
                        help='Loss weight for audited MD normal-residual supervision')
    parser.add_argument('--phase_normal_residual_loss_type', type=str, default='huber',
                        choices=['mse', 'huber'])
    parser.add_argument('--phase_normal_residual_huber_delta', type=float, default=0.25)
    parser.add_argument('--phase_normal_missing_policy', type=str, default='error',
                        choices=['error', 'skip'])

    # ESM representation adapter
    parser.add_argument('--esm_fusion_enabled', action='store_true',
                        help='Enable last-K ESM layer fusion before the Stage-2 geometry trunk')
    parser.add_argument('--esm_num_layers', type=int, default=1,
                        help='Number of ESM layers expected when --esm_fusion_enabled is active')
    parser.add_argument('--esm_fusion_mode', type=str, default='sum',
                        choices=['sum', 'mean', 'softmax_weighted', 'gated_residual'],
                        help='How to fuse [B,N,K,D] ESM features')
    parser.add_argument('--esm_layer_dropout', type=float, default=0.0,
                        help='Dropout applied to learned ESM layer weights during training')
    parser.add_argument('--esm_layer_entropy_weight', type=float, default=0.0,
                        help='Entropy regularization weight for gated_residual ESM layer weights (penalizes uniform weights)')
    parser.add_argument('--esm_gate_bias', type=float, default=-3.0,
                        help='Initial bias for gated_residual earlier-layer gates')
    parser.add_argument('--esm_gate_context_mode', type=str, default='none',
                        choices=['none', 'pocket_motion'],
                        help='Optional context appended to gated_residual gate input')

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
    parser.add_argument('--repa_target_mode', type=str, default='full',
                        choices=['full', 'motion_continuous'],
                        help='Feature subset used as the REPA alignment target')
    parser.add_argument('--repa_target_shuffle_mode', type=str, default='none',
                        choices=['none', 'residue'],
                        help='Shuffle only the REPA target while leaving Stage-2 conditioning features unchanged')
    parser.add_argument('--w_contact', type=float, default=0.1,
                        help='Stage-2 path contact loss 权重')
    parser.add_argument('--w_fm_chi', type=float, default=1.0,
                        help='CFM chi velocity loss weight')
    parser.add_argument('--w_fm_rigid', type=float, default=1.0,
                        help='CFM rigid velocity loss weight')
    parser.add_argument('--w_bg', type=float, default=0.1,
                        help='Background stability loss weight')
    parser.add_argument('--w_phase_residual_magnitude', type=float, default=0.01,
                        help='Capacity penalty on projected spatial residual magnitude')
    parser.add_argument('--w_phase_residual_temporal_smooth', type=float, default=0.01,
                        help='Temporal smoothness penalty on projected spatial residuals')
    parser.add_argument('--w_phase_residual_neighbor_smooth', type=float, default=0.01,
                        help='Adjacent-residue smoothness penalty on projected spatial residuals')
    parser.add_argument('--w_smooth', type=float, default=0.05,
                        help='Path smoothness loss weight')
    parser.add_argument('--w_clash', type=float, default=0.1,
                        help='Path clash loss weight')
    parser.add_argument('--w_ligand_clearance', type=float, default=0.0,
                        help='Path sidechain-ligand clearance loss weight')
    parser.add_argument('--ligand_clearance_dist', type=float, default=2.2,
                        help='Minimum sidechain-ligand distance encouraged along interior path')
    parser.add_argument('--ligand_clearance_mask_mode', type=str, default='pocket',
                        choices=['pocket', 'node', 'motion_active', 'pocket_or_motion_active'],
                        help='Residues supervised by ligand clearance loss')
    parser.add_argument('--ligand_clearance_loss_mode', type=str, default='all',
                        choices=['all', 'hard_negative'],
                        help='all averages clearance over the mask; hard_negative only supervises current clash residues')
    parser.add_argument('--ligand_clearance_hard_negative_dist', type=float, default=2.2,
                        help='Current path distance cutoff used to select hard-negative ligand clashes')
    parser.add_argument('--ligand_clearance_t_min', type=float, default=0.05,
                        help='Earliest path time supervised by ligand clearance loss')
    parser.add_argument('--ligand_clearance_t_max', type=float, default=0.95,
                        help='Latest path time supervised by ligand clearance loss')
    parser.add_argument('--w_bridge_anchor', type=float, default=0.0,
                        help='Weight for bridge-anchor regularization on non-clash path states')
    parser.add_argument('--bridge_anchor_mask_mode', type=str, default='non_clash_node',
                        choices=['non_clash_node', 'non_clash_pocket', 'node', 'pocket'],
                        help='Residues regularized toward the analytic bridge')
    parser.add_argument('--bridge_anchor_t_min', type=float, default=0.05,
                        help='Earliest path time supervised by bridge-anchor regularization')
    parser.add_argument('--bridge_anchor_t_max', type=float, default=0.95,
                        help='Latest path time supervised by bridge-anchor regularization')
    parser.add_argument('--w_pep', type=float, default=0.1,
                        help='Peptide geometry loss weight')
    parser.add_argument('--w_end', type=float, default=0.1,
                        help='Endpoint loss weight')
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
    parser.add_argument('--checkpoint_every_n_epochs', type=int, default=0,
                        help='Save epoch_XXXX.pt every N epochs after validation; <=0 disables')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Explicit Stage-2 checkpoint path to resume from')
    parser.add_argument('--no_auto_resume', action='store_true',
                        help='Disable automatic resume from save_dir/last_checkpoint.pt')

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
    parser.add_argument('--length_bucketed_train', action='store_true',
                        help='Use length-bucketed DDP train batches for variable-length proteins')
    parser.add_argument('--length_bucket_multiplier', type=int, default=8,
                        help='Bucket size multiplier for length-bucketed training batches')
    parser.add_argument('--length_bucket_drop_last', dest='length_bucket_drop_last',
                        action='store_true', default=True,
                        help='Drop the last incomplete length-bucketed global batch')
    parser.add_argument('--no_length_bucket_drop_last', dest='length_bucket_drop_last',
                        action='store_false',
                        help='Keep the last incomplete length-bucketed global batch')
    parser.add_argument('--length_bucket_lengths_file', type=str, default=None,
                        help='Optional TSV/CSV/whitespace file with sample_id and n_residues for length bucketing')
    parser.add_argument('--length_bucket_residue_budget', type=int, default=None,
                        help='Optional per-rank residue budget for length-bucketed variable batch packing')
    parser.add_argument('--progress_log_every', type=int, default=100,
                        help='Training progress log interval in batches; <=0 disables sparse batch logging')

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
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        valid_samples_file=args.valid_samples_file,
        val_samples_file=args.val_samples_file,
        val_split=args.val_split,
        trust_prechecked_samples=args.trust_prechecked_samples,
        lr=args.lr,
        max_epochs=args.max_epochs,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.accum_steps,
        warmup_steps=args.warmup_steps,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        val_t=args.val_t if args.val_t >= 0.0 else None,
        path_parameterization=args.path_parameterization,
        boundary_residual_envelope=args.boundary_residual_envelope,
        boundary_residual_scale=args.boundary_residual_scale,
        terminal_projection_schedule=args.terminal_projection_schedule,
        time_warp_logit_scale=args.time_warp_logit_scale,
        time_warp_rate_eps=args.time_warp_rate_eps,
        time_warp_rate_clip=args.time_warp_rate_clip,
        phase_residual_tau_mode=args.phase_residual_tau_mode,
        phase_residual_bridge_mode=args.phase_residual_bridge_mode,
        phase_residual_envelope=args.phase_residual_envelope,
        phase_residual_scale=args.phase_residual_scale,
        phase_residual_rotation_metric_scale=args.phase_residual_rotation_metric_scale,
        phase_residual_translation_metric_scale=args.phase_residual_translation_metric_scale,
        phase_residual_chi_metric_scale=args.phase_residual_chi_metric_scale,
        phase_residual_min_tangent_norm=args.phase_residual_min_tangent_norm,
        phase_residual_max_metric_norm=args.phase_residual_max_metric_norm,
        phase_residual_peptide_retraction=args.phase_residual_peptide_retraction,
        phase_residual_peptide_retraction_iterations=args.phase_residual_peptide_retraction_iterations,
        phase_residual_peptide_retraction_relaxation=args.phase_residual_peptide_retraction_relaxation,
        phase_residual_peptide_retraction_anchor_strength=args.phase_residual_peptide_retraction_anchor_strength,
        phase_residual_peptide_retraction_max_translation=args.phase_residual_peptide_retraction_max_translation,
        phase_residual_peptide_retraction_activation_loss_threshold=args.phase_residual_peptide_retraction_activation_loss_threshold,
        init_from_checkpoint=args.init_from_checkpoint,
        teacher_residual_cache_dir=args.teacher_residual_cache_dir,
        w_teacher_residual=args.w_teacher_residual,
        teacher_residual_loss_type=args.teacher_residual_loss_type,
        teacher_residual_huber_delta=args.teacher_residual_huber_delta,
        teacher_residual_t_min=args.teacher_residual_t_min,
        teacher_residual_t_max=args.teacher_residual_t_max,
        teacher_residual_mask_mode=args.teacher_residual_mask_mode,
        teacher_residual_clash_weight_threshold=args.teacher_residual_clash_weight_threshold,
        teacher_residual_missing_policy=args.teacher_residual_missing_policy,
        phase_teacher_cache_dir=args.phase_teacher_cache_dir,
        w_phase_teacher=args.w_phase_teacher,
        phase_teacher_loss_type=args.phase_teacher_loss_type,
        phase_teacher_huber_delta=args.phase_teacher_huber_delta,
        phase_teacher_mask_mode=args.phase_teacher_mask_mode,
        phase_teacher_min_confidence=args.phase_teacher_min_confidence,
        phase_teacher_missing_policy=args.phase_teacher_missing_policy,
        phase_teacher_head_only=args.phase_teacher_head_only,
        phase_teacher_residual_heads_only=args.phase_teacher_residual_heads_only,
        phase_normal_cache_dir=args.phase_normal_cache_dir,
        w_phase_normal_residual=args.w_phase_normal_residual,
        phase_normal_residual_loss_type=args.phase_normal_residual_loss_type,
        phase_normal_residual_huber_delta=args.phase_normal_residual_huber_delta,
        phase_normal_missing_policy=args.phase_normal_missing_policy,
        esm_fusion_enabled=args.esm_fusion_enabled,
        esm_num_layers=args.esm_num_layers,
        esm_fusion_mode=args.esm_fusion_mode,
        esm_layer_dropout=args.esm_layer_dropout,
        esm_layer_entropy_weight=args.esm_layer_entropy_weight,
        esm_gate_bias=args.esm_gate_bias,
        esm_gate_context_mode=args.esm_gate_context_mode,
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
        repa_target_mode=args.repa_target_mode,
        repa_target_shuffle_mode=args.repa_target_shuffle_mode,
        w_fm_chi=args.w_fm_chi,
        w_fm_rigid=args.w_fm_rigid,
        w_bg=args.w_bg,
        w_phase_residual_magnitude=args.w_phase_residual_magnitude,
        w_phase_residual_temporal_smooth=args.w_phase_residual_temporal_smooth,
        w_phase_residual_neighbor_smooth=args.w_phase_residual_neighbor_smooth,
        w_smooth=args.w_smooth,
        w_clash=args.w_clash,
        w_ligand_clearance=args.w_ligand_clearance,
        ligand_clearance_dist=args.ligand_clearance_dist,
        ligand_clearance_mask_mode=args.ligand_clearance_mask_mode,
        ligand_clearance_loss_mode=args.ligand_clearance_loss_mode,
        ligand_clearance_hard_negative_dist=args.ligand_clearance_hard_negative_dist,
        ligand_clearance_t_min=args.ligand_clearance_t_min,
        ligand_clearance_t_max=args.ligand_clearance_t_max,
        w_bridge_anchor=args.w_bridge_anchor,
        bridge_anchor_mask_mode=args.bridge_anchor_mask_mode,
        bridge_anchor_t_min=args.bridge_anchor_t_min,
        bridge_anchor_t_max=args.bridge_anchor_t_max,
        w_pep=args.w_pep,
        w_contact=args.w_contact,
        w_end=args.w_end,
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
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        resume_from=args.resume_from,
        auto_resume=not args.no_auto_resume,
        device=args.device,
        mixed_precision=not args.no_mixed_precision,
        amp_dtype=args.amp_dtype,
        distributed=args.distributed,
        length_bucketed_train=args.length_bucketed_train,
        length_bucket_multiplier=args.length_bucket_multiplier,
        length_bucket_drop_last=args.length_bucket_drop_last,
        length_bucket_lengths_file=args.length_bucket_lengths_file,
        length_bucket_residue_budget=args.length_bucket_residue_budget,
        progress_log_every=args.progress_log_every,
    )

    print(f"\n{'='*80}")
    print("BINDRAE Stage-2 训练")
    print(f"{'='*80}")
    print("\n配置:")
    print(f"  - 数据目录: {config.data_dir}")
    print(f"  - validation split: {config.val_split}")
    print(f"  - trust prechecked samples: {config.trust_prechecked_samples}")
    print(f"  - 批大小: {config.batch_size}")
    print(f"  - val 批大小: {config.val_batch_size or config.batch_size}")
    print(f"  - 学习率: {config.lr}")
    print(f"  - 最大轮数: {config.max_epochs}")
    print(f"  - checkpoint every N epochs: {config.checkpoint_every_n_epochs}")
    print(f"  - ESM fusion enabled: {config.esm_fusion_enabled}")
    print(f"  - ESM num layers: {config.esm_num_layers}")
    print(f"  - ESM fusion mode: {config.esm_fusion_mode}")
    print(f"  - ESM layer dropout: {config.esm_layer_dropout}")
    print(f"  - ESM gate bias: {config.esm_gate_bias}")
    print(f"  - ESM gate context: {config.esm_gate_context_mode}")
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
    print(f"  - REPA target mode: {config.repa_target_mode}")
    print(f"  - REPA target shuffle mode: {config.repa_target_shuffle_mode}")
    print(f"  - w_contact: {config.w_contact}")
    print(f"  - contact_loss_mode: {config.contact_loss_mode}")
    print(f"  - contact_eps: {config.contact_eps}")
    print(f"  - contact_direction_eps: {config.contact_direction_eps}")
    print(f"  - w_prior: {config.w_prior}")
    print(f"  - t_mid: {config.t_mid}")
    print(f"  - warmup_steps: {config.warmup_steps}")
    print(f"  - early_stop_patience: {config.early_stop_patience}")
    print(f"  - num_workers: {config.num_workers}")
    print(f"  - prefetch_factor: {config.prefetch_factor}")
    print(f"  - n_integration_steps: {config.n_integration_steps}")
    print(f"  - integration_chi_clip: {config.integration_chi_clip}")
    print(f"  - integration_rot_clip: {config.integration_rot_clip}")
    print(f"  - integration_trans_clip: {config.integration_trans_clip}")
    print(f"  - n_geom_steps: {config.n_geom_steps}")
    print(f"  - geom_loss_every_n_steps: {config.geom_loss_every_n_steps}")
    print(f"  - seed: {config.seed}")
    print(f"  - val_t: {config.val_t}")
    print(f"  - path_parameterization: {config.path_parameterization}")
    print(f"  - boundary_residual_envelope: {config.boundary_residual_envelope}")
    print(f"  - boundary_residual_scale: {config.boundary_residual_scale}")
    print(f"  - terminal_projection_schedule: {config.terminal_projection_schedule}")
    print(f"  - time_warp_logit_scale: {config.time_warp_logit_scale}")
    print(f"  - time_warp_rate_eps: {config.time_warp_rate_eps}")
    print(f"  - time_warp_rate_clip: {config.time_warp_rate_clip}")
    print(f"  - phase_residual_tau_mode: {config.phase_residual_tau_mode}")
    print(f"  - phase_residual_bridge_mode: {config.phase_residual_bridge_mode}")
    print(f"  - phase_residual_envelope: {config.phase_residual_envelope}")
    print(f"  - phase_residual_scale: {config.phase_residual_scale}")
    print(f"  - phase_residual_metric_scales: rot={config.phase_residual_rotation_metric_scale} trans={config.phase_residual_translation_metric_scale} chi={config.phase_residual_chi_metric_scale}")
    print(f"  - phase_residual_min_tangent_norm: {config.phase_residual_min_tangent_norm}")
    print(f"  - phase_residual_max_metric_norm: {config.phase_residual_max_metric_norm}")
    print(f"  - phase_residual_peptide_retraction: {config.phase_residual_peptide_retraction}")
    print(f"  - phase_residual_peptide_retraction_iterations: {config.phase_residual_peptide_retraction_iterations}")
    print(f"  - phase_residual_peptide_retraction_max_translation: {config.phase_residual_peptide_retraction_max_translation}")
    print(f"  - init_from_checkpoint: {config.init_from_checkpoint or 'OFF'}")
    print(f"  - teacher_residual_cache_dir: {config.teacher_residual_cache_dir or 'OFF'}")
    print(f"  - w_teacher_residual: {config.w_teacher_residual}")
    print(f"  - phase_teacher_cache_dir: {config.phase_teacher_cache_dir or 'OFF'}")
    print(f"  - w_phase_teacher: {config.w_phase_teacher}")
    print(f"  - phase_teacher_head_only: {config.phase_teacher_head_only}")
    print(f"  - phase_teacher_residual_heads_only: {config.phase_teacher_residual_heads_only}")
    print(f"  - phase_normal_cache_dir: {config.phase_normal_cache_dir or 'OFF'}")
    print(f"  - w_phase_normal_residual: {config.w_phase_normal_residual}")
    print(f"  - phase_normal_residual_loss_type: {config.phase_normal_residual_loss_type}")
    print(f"  - phase_normal_missing_policy: {config.phase_normal_missing_policy}")
    print(f"  - teacher_residual_loss_type: {config.teacher_residual_loss_type}")
    print(f"  - teacher_residual_huber_delta: {config.teacher_residual_huber_delta}")
    print(f"  - teacher_residual_t_range: {config.teacher_residual_t_min}-{config.teacher_residual_t_max}")
    print(f"  - teacher_residual_mask_mode: {config.teacher_residual_mask_mode}")
    print(f"  - teacher_residual_clash_weight_threshold: {config.teacher_residual_clash_weight_threshold}")
    print(f"  - teacher_residual_missing_policy: {config.teacher_residual_missing_policy}")
    print(f"  - ligand_clearance: w={config.w_ligand_clearance} dist={config.ligand_clearance_dist} hard_dist={config.ligand_clearance_hard_negative_dist} mode={config.ligand_clearance_loss_mode} mask={config.ligand_clearance_mask_mode} t={config.ligand_clearance_t_min}-{config.ligand_clearance_t_max}")
    print(f"  - bridge_anchor: w={config.w_bridge_anchor} mask={config.bridge_anchor_mask_mode} t={config.bridge_anchor_t_min}-{config.bridge_anchor_t_max}")
    print(f"  - loss weights: fm_chi={config.w_fm_chi} fm_rigid={config.w_fm_rigid} bg={config.w_bg} smooth={config.w_smooth} clash={config.w_clash} ligand_clearance={config.w_ligand_clearance} bridge_anchor={config.w_bridge_anchor} pep={config.w_pep} end={config.w_end}")
    print(f"  - phase residual loss weights: magnitude={config.w_phase_residual_magnitude} temporal={config.w_phase_residual_temporal_smooth} neighbor={config.w_phase_residual_neighbor_smooth}")
    print(f"  - resume_from: {config.resume_from or 'OFF'}")
    print(f"  - auto_resume: {config.auto_resume}")
    print(f"  - NMA: {config.use_nma}")
    print(f"  - 设备: {config.device}")
    print(f"  - 混合精度: {config.mixed_precision}")
    print(f"  - AMP dtype: {config.amp_dtype}")
    print(f"  - length_bucketed_train: {config.length_bucketed_train}")
    print(f"  - length_bucket_multiplier: {config.length_bucket_multiplier}")
    print(f"  - length_bucket_drop_last: {config.length_bucket_drop_last}")
    print(f"  - length_bucket_lengths_file: {config.length_bucket_lengths_file or 'OFF'}")
    print(f"  - length_bucket_residue_budget: {config.length_bucket_residue_budget or 'OFF'}")
    print(f"  - progress_log_every: {config.progress_log_every}")
    print(f"\n{'='*80}\n")

    trainer = Stage2Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
