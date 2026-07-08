"""Stage-2 trainer (bridge flow on apo->holo paths)."""

import datetime
import hashlib
import json
import math
import os
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# FlashIPA path (项目内 vendor 目录)
_project_root = Path(__file__).resolve().parent.parent.parent.parent
flash_ipa_path = str(_project_root / 'vendor' / 'flash_ipa' / 'src')
if os.path.exists(flash_ipa_path) and flash_ipa_path not in sys.path:
    sys.path.insert(0, flash_ipa_path)

from flash_ipa.rigid import Rigid, Rotation

from .config import TrainingConfig
from ..datasets import create_stage2_dataloader
from ..models import TorsionFlowNet, TorsionFlowNetConfig
from ..modules import (
    se3_log,
    se3_exp,
    rigid_inverse,
    rigid_compose,
    wrap_to_pi,
    compute_contact_score,
    compute_peptide_loss,
    compute_w_eff,
)
from src.stage1.models import Stage1Model, Stage1ModelConfig
from src.stage1.models.fk_openfold import create_openfold_fk, reorder_torsions_to_openfold
from src.stage1.models.interaction_prior import (
    load_interaction_prior,
    min_sidechain_ligand_dist,
    sidechain_atom_mask,
)
from src.stage1.datasets.samplers import DistributedLengthBatchSampler
from src.stage1.modules.losses import clash_penalty, fape_loss


class Stage2Trainer:
    """Stage-2 trainer."""

    _LOSS_KEYS = (
        'total',
        'total_no_repa',
        'fm_chi',
        'fm_rigid',
        'teacher_residual',
        'teacher_residual_rigid',
        'teacher_residual_chi',
        'teacher_residual_target_norm',
        'teacher_residual_pred_norm',
        'teacher_residual_t_error',
        'teacher_residual_weight_mean',
        'teacher_residual_mask_frac',
        'bg',
        'smooth',
        'clash',
        'ligand_clearance',
        'ligand_clearance_active_frac',
        'ligand_clearance_min_dist',
        'bridge_anchor',
        'bridge_anchor_mask_frac',
        'bridge_anchor_residual_norm',
        'pep',
        'contact',
        'stage1v2_guidance',
        'repa',
        'prior',
        'interaction_prior',
        'interaction_prior_final',
        'contact_score_apo',
        'contact_score_final',
        'contact_score_holo',
        'contact_score_gain',
        'contact_score_holo_delta',
        'contact_score_direction_acc',
        'contact_score_holo_gap_abs',
        'contact_score_sidechain_apo',
        'contact_score_sidechain_final',
        'contact_score_sidechain_holo',
        'contact_score_sidechain_gain',
        'contact_score_sidechain_holo_gap_abs',
        'contact_score_sidechain_formed_recall',
        'end_rigid',
        'end_chi',
        'end_fape',
        'end_rigid_uw',
        'end_chi_uw',
        'end',
        'esm_entropy',
        'esm_layer_weight_entropy_raw',
        'esm_layer_weight_max',
        'esm_gate_mean',
        'esm_gate_pocket_mean',
        'esm_gate_nonpocket_mean',
        'esm_layer_weight_0',
        'esm_layer_weight_1',
        'esm_layer_weight_2',
        'esm_layer_weight_3',
        'esm_layer_weight_4',
        'esm_layer_weight_5',
        'esm_layer_weight_6',
        'esm_layer_weight_7',
        'esm_layer_weight_8',
        'esm_layer_weight_9',
    )

    _REPA_MOTION_CONTINUOUS_FEATURES = (
        'delta_trans_local_x_norm',
        'delta_trans_local_y_norm',
        'delta_trans_local_z_norm',
        'delta_rot_log_x_norm',
        'delta_rot_log_y_norm',
        'delta_rot_log_z_norm',
        'delta_chi1_sin',
        'delta_chi2_sin',
        'delta_chi3_sin',
        'delta_chi4_sin',
        'delta_chi1_cos',
        'delta_chi2_cos',
        'delta_chi3_cos',
        'delta_chi4_cos',
        'trans_mag_norm',
        'rot_angle_norm',
        'max_abs_delta_chi_norm',
    )

    @staticmethod
    def _resolve_stage1_model_config(ckpt: Dict) -> Stage1ModelConfig:
        saved_config = ckpt.get('config') if isinstance(ckpt, dict) else None
        if isinstance(saved_config, Stage1ModelConfig):
            return saved_config

        model_size = getattr(saved_config, 'model_size', None)
        if isinstance(saved_config, dict):
            model_size = saved_config.get('model_size', model_size)

        if model_size is None:
            return Stage1ModelConfig()

        builders = {
            'small': Stage1ModelConfig.small,
            'stable_wide': Stage1ModelConfig.stable_wide,
            'medium': Stage1ModelConfig.medium,
            'large': Stage1ModelConfig.large,
            'wide_shallow': Stage1ModelConfig.wide_shallow,
            'enhanced_ligand': Stage1ModelConfig.enhanced_ligand,
        }
        try:
            return builders[str(model_size).lower()]()
        except KeyError as exc:
            raise ValueError(f"Unsupported Stage-1 model_size in checkpoint: {model_size}") from exc

    @staticmethod
    def _parse_stage1v2_feature_names(raw: str) -> Tuple[str, ...]:
        names = tuple(item.strip() for item in str(raw or "").split(",") if item.strip())
        if not names:
            raise ValueError("stage1v2_posterior_feature_names cannot be empty")
        return names

    def _resolve_repa_target_indices(self) -> Optional[Tuple[int, ...]]:
        if self.config.repa_target_mode == 'full':
            return None
        if self.config.repa_target_mode != 'motion_continuous':
            raise ValueError(f"Unsupported repa_target_mode={self.config.repa_target_mode}")
        name_to_idx = {name: idx for idx, name in enumerate(self.stage1v2_feature_names)}
        missing = [
            name for name in self._REPA_MOTION_CONTINUOUS_FEATURES
            if name not in name_to_idx
        ]
        if missing:
            raise ValueError(
                f"repa_target_mode=motion_continuous requires missing features: {missing}"
            )
        return tuple(name_to_idx[name] for name in self._REPA_MOTION_CONTINUOUS_FEATURES)

    def _stage1v2_feature_dir_for_split(self, split: str) -> Optional[str]:
        mode = self.config.stage1v2_posterior_feature_mode
        if mode in {'none', 'zero'}:
            return None
        if mode == 'oracle_holo_truth':
            return (
                self.config.stage1v2_train_label_dir
                if split == 'train'
                else self.config.stage1v2_val_label_dir
            )
        return (
            self.config.stage1v2_train_cache_dir
            if split == 'train'
            else self.config.stage1v2_val_cache_dir
        )

    def _load_length_bucket_costs(self) -> List[int]:
        lengths_file = self.config.length_bucket_lengths_file
        if not lengths_file:
            return self.train_loader.dataset.get_sample_lengths()

        path = Path(lengths_file)
        if not path.is_absolute() and not path.exists():
            path = Path(self.config.data_dir) / path
        if not path.exists():
            raise FileNotFoundError(f"length_bucket_lengths_file not found: {path}")

        length_by_id: Dict[str, int] = {}
        with open(path, 'r') as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.replace(',', '\t').split()
                if len(parts) < 2:
                    continue
                sample_id, value = parts[0], parts[1]
                if line_no == 1 and sample_id.lower() in {'sample_id', 'id'}:
                    continue
                try:
                    length_by_id[sample_id] = int(value)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no} invalid n_residues={value!r}") from exc

        lengths: List[int] = []
        missing: List[str] = []
        for idx, sample in enumerate(self.train_loader.dataset.samples):
            sample_id = sample.get('id', f'sample_{idx}')
            n_res = length_by_id.get(sample_id)
            if n_res is None:
                missing.append(sample_id)
                if len(missing) >= 8:
                    break
            else:
                lengths.append(n_res)
        if missing:
            raise ValueError(
                f"{path} is missing training sample lengths; examples={missing}"
            )
        return lengths

    @staticmethod
    def _format_length_stats(lengths: List[int]) -> str:
        if not lengths:
            return "empty"
        ordered = sorted(lengths)

        def quantile(q: float) -> int:
            pos = min(int(round(q * (len(ordered) - 1))), len(ordered) - 1)
            return ordered[pos]

        return (
            f"min={ordered[0]} median={quantile(0.5)} "
            f"p95={quantile(0.95)} p99={quantile(0.99)} max={ordered[-1]}"
        )

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.distributed = config.distributed
        self.local_rank = 0
        self.world_size = 1
        self.is_main_process = True

        if self.distributed:
            os.environ.setdefault('NCCL_TIMEOUT', '1800000')
            os.environ.setdefault('NCCL_DEBUG', 'WARN')
            os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '1')
            os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
            dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=1800))
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = dist.get_world_size()
            self.is_main_process = (self.local_rank == 0)
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            if self.is_main_process:
                print(f"[DDP] Initialized: world_size={self.world_size}, local_rank={self.local_rank}")
        else:
            self.device = torch.device(config.device)

        torch.manual_seed(config.seed + self.local_rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed + self.local_rank)

        if config.use_nma and config.nma_dim <= 0:
            raise ValueError("use_nma=True but nma_dim <= 0")
        if not config.use_nma and config.nma_dim > 0:
            raise ValueError("nma_dim > 0 but use_nma is False")
        allowed_prior_modes = {
            'stage1', 'apo_chi', 'holo_chi', 'holo_chi_apo_rigid',
            'apo_chi_holo_rigid', 'holo_ligand_facing_chi', 'noisy_stage1_chi'
        }
        if config.stage1_prior_mode not in allowed_prior_modes:
            raise ValueError(f"Unsupported stage1_prior_mode={config.stage1_prior_mode}")
        if config.esm_num_layers < 1:
            raise ValueError(f"esm_num_layers must be >= 1, got {config.esm_num_layers}")
        allowed_esm_fusion_modes = {'sum', 'mean', 'softmax_weighted', 'gated_residual'}
        if config.esm_fusion_mode not in allowed_esm_fusion_modes:
            raise ValueError(f"Unsupported esm_fusion_mode={config.esm_fusion_mode}")
        if not 0.0 <= config.esm_layer_dropout < 1.0:
            raise ValueError(
                f"esm_layer_dropout must be in [0, 1), got {config.esm_layer_dropout}"
            )
        allowed_esm_gate_context_modes = {'none', 'pocket_motion'}
        if config.esm_gate_context_mode not in allowed_esm_gate_context_modes:
            raise ValueError(f"Unsupported esm_gate_context_mode={config.esm_gate_context_mode}")
        if config.esm_gate_context_mode != 'none' and config.esm_fusion_mode != 'gated_residual':
            raise ValueError("esm_gate_context_mode requires esm_fusion_mode=gated_residual")
        allowed_interaction_feature_modes = {'none', 'prior', 'zero', 'oracle_contact'}
        if config.interaction_prior_feature_mode not in allowed_interaction_feature_modes:
            raise ValueError(
                f"Unsupported interaction_prior_feature_mode={config.interaction_prior_feature_mode}"
            )
        allowed_stage1v2_feature_modes = {
            'none', 'zero', 'student', 'student_shuffled',
            'oracle_holo_truth', 'external_teacher_cached',
            'oracle_motion', 'oracle_motion_residue_shuffled',
            'oracle_motion_sample_shuffled',
        }
        if config.stage1v2_posterior_feature_mode not in allowed_stage1v2_feature_modes:
            raise ValueError(
                f"Unsupported stage1v2_posterior_feature_mode={config.stage1v2_posterior_feature_mode}"
            )
        allowed_repa_loss_types = {'cosine', 'mse'}
        if config.repa_loss_type not in allowed_repa_loss_types:
            raise ValueError(f"Unsupported repa_loss_type={config.repa_loss_type}")
        allowed_repa_mask_modes = {'node', 'pocket', 'motion_active', 'motion_active_or_pocket'}
        if config.repa_mask_mode not in allowed_repa_mask_modes:
            raise ValueError(f"Unsupported repa_mask_mode={config.repa_mask_mode}")
        allowed_repa_target_modes = {'full', 'motion_continuous'}
        if config.repa_target_mode not in allowed_repa_target_modes:
            raise ValueError(f"Unsupported repa_target_mode={config.repa_target_mode}")
        allowed_repa_target_shuffle_modes = {'none', 'residue'}
        if config.repa_target_shuffle_mode not in allowed_repa_target_shuffle_modes:
            raise ValueError(
                f"Unsupported repa_target_shuffle_mode={config.repa_target_shuffle_mode}"
            )
        if config.repa_weight < 0.0:
            raise ValueError(f"repa_weight must be >= 0, got {config.repa_weight}")
        if config.repa_dim <= 0:
            raise ValueError(f"repa_dim must be > 0, got {config.repa_dim}")
        allowed_stage1v2_loss_weight_modes = {'none', 'contact', 'active', 'contact_active'}
        if config.stage1v2_loss_weight_mode not in allowed_stage1v2_loss_weight_modes:
            raise ValueError(
                f"Unsupported stage1v2_loss_weight_mode={config.stage1v2_loss_weight_mode}"
            )
        allowed_contact_loss_modes = {'holo_target', 'monotonic_increase'}
        if config.contact_loss_mode not in allowed_contact_loss_modes:
            raise ValueError(f"Unsupported contact_loss_mode={config.contact_loss_mode}")
        allowed_path_parameterizations = {
            'flow', 'boundary_residual_v1', 'boundary_residual', 'projected_flow'
        }
        if config.path_parameterization not in allowed_path_parameterizations:
            raise ValueError(f"Unsupported path_parameterization={config.path_parameterization}")
        allowed_boundary_envelopes = {'sin2', 'poly'}
        if config.boundary_residual_envelope not in allowed_boundary_envelopes:
            raise ValueError(f"Unsupported boundary_residual_envelope={config.boundary_residual_envelope}")
        allowed_teacher_residual_mask_modes = {
            'node',
            'pocket',
            'motion_active',
            'motion_active_or_pocket',
            'clash_relief',
            'clash_relief_or_motion_active',
            'clash_relief_or_pocket',
        }
        if config.teacher_residual_mask_mode not in allowed_teacher_residual_mask_modes:
            raise ValueError(
                f"Unsupported teacher_residual_mask_mode={config.teacher_residual_mask_mode}"
            )
        if config.teacher_residual_clash_weight_threshold < 0.0:
            raise ValueError("teacher_residual_clash_weight_threshold must be >= 0")
        allowed_teacher_residual_missing = {'error', 'skip'}
        if config.teacher_residual_missing_policy not in allowed_teacher_residual_missing:
            raise ValueError(
                "Unsupported teacher_residual_missing_policy="
                f"{config.teacher_residual_missing_policy}"
            )
        if config.w_teacher_residual > 0.0 and not config.teacher_residual_cache_dir:
            raise ValueError("w_teacher_residual > 0 requires teacher_residual_cache_dir")
        if config.w_teacher_residual > 0.0 and config.path_parameterization not in {
            'boundary_residual_v1', 'boundary_residual'
        }:
            raise ValueError("teacher residual distillation requires boundary_residual path mode")
        allowed_teacher_residual_losses = {'mse', 'huber'}
        if config.teacher_residual_loss_type not in allowed_teacher_residual_losses:
            raise ValueError(
                f"Unsupported teacher_residual_loss_type={config.teacher_residual_loss_type}"
            )
        if config.teacher_residual_huber_delta <= 0.0:
            raise ValueError("teacher_residual_huber_delta must be > 0")
        if not (0.0 <= config.teacher_residual_t_min < config.teacher_residual_t_max <= 1.0):
            raise ValueError(
                "teacher_residual_t_min/max must satisfy 0 <= min < max <= 1"
            )
        allowed_projection_schedules = {'smoothstep', 'smootherstep', 'late_smoother', 'quadratic'}
        if config.terminal_projection_schedule not in allowed_projection_schedules:
            raise ValueError(
                f"Unsupported terminal_projection_schedule={config.terminal_projection_schedule}"
            )
        if config.n_integration_steps <= 0:
            raise ValueError(f"n_integration_steps must be > 0, got {config.n_integration_steps}")
        if config.n_geom_steps <= 0:
            raise ValueError(f"n_geom_steps must be > 0, got {config.n_geom_steps}")
        if config.geom_loss_every_n_steps <= 0:
            raise ValueError(
                f"geom_loss_every_n_steps must be > 0, got {config.geom_loss_every_n_steps}"
            )
        if config.ligand_clearance_dist <= 0.0:
            raise ValueError("ligand_clearance_dist must be > 0")
        if config.ligand_clearance_hard_negative_dist <= 0.0:
            raise ValueError("ligand_clearance_hard_negative_dist must be > 0")
        if not (0.0 <= config.ligand_clearance_t_min < config.ligand_clearance_t_max <= 1.0):
            raise ValueError(
                "ligand_clearance_t_min/max must satisfy 0 <= min < max <= 1"
            )
        allowed_clearance_loss_modes = {'all', 'hard_negative'}
        if config.ligand_clearance_loss_mode not in allowed_clearance_loss_modes:
            raise ValueError(
                "Unsupported ligand_clearance_loss_mode="
                f"{config.ligand_clearance_loss_mode}"
            )
        allowed_clearance_masks = {'pocket', 'node', 'motion_active', 'pocket_or_motion_active'}
        if config.ligand_clearance_mask_mode not in allowed_clearance_masks:
            raise ValueError(
                "Unsupported ligand_clearance_mask_mode="
                f"{config.ligand_clearance_mask_mode}"
            )
        if not (0.0 <= config.bridge_anchor_t_min < config.bridge_anchor_t_max <= 1.0):
            raise ValueError(
                "bridge_anchor_t_min/max must satisfy 0 <= min < max <= 1"
            )
        allowed_anchor_masks = {'non_clash_node', 'non_clash_pocket', 'node', 'pocket'}
        if config.bridge_anchor_mask_mode not in allowed_anchor_masks:
            raise ValueError(
                "Unsupported bridge_anchor_mask_mode="
                f"{config.bridge_anchor_mask_mode}"
            )
        if (
            config.path_parameterization in {'boundary_residual_v1', 'boundary_residual'}
            and config.geom_loss_every_n_steps > 1
            and self.is_main_process
        ):
            print(
                "WARNING: boundary_residual_v1 learns from path geometry/contact losses; "
                "geom_loss_every_n_steps > 1 makes residual supervision sparse."
            )
        if config.interaction_prior_feature_mode == 'prior' and not config.interaction_prior_ckpt:
            raise ValueError("interaction_prior_ckpt must be set when interaction_prior_feature_mode=prior")
        if config.w_interaction_prior > 0.0 and not config.interaction_prior_ckpt:
            raise ValueError("interaction_prior_ckpt must be set when w_interaction_prior > 0")
        train_stage1v2_dir = self._stage1v2_feature_dir_for_split('train')
        val_stage1v2_dir = self._stage1v2_feature_dir_for_split('val')
        if config.stage1v2_posterior_feature_mode not in {'none', 'zero'}:
            if not train_stage1v2_dir or not val_stage1v2_dir:
                raise ValueError(
                    f"stage1v2_posterior_feature_mode={config.stage1v2_posterior_feature_mode} "
                    "requires train/val cache or label directories"
                )
        if (
            config.stage1v2_posterior_feature_mode == 'none'
            and (
                config.stage1v2_loss_weight_mode != 'none'
                or config.w_stage1v2_guidance > 0.0
            )
        ):
            raise ValueError(
                "Stage-1-v2 posterior loss weighting/guidance requires "
                "stage1v2_posterior_feature_mode != none"
            )
        self.stage1v2_feature_names = self._parse_stage1v2_feature_names(
            config.stage1v2_posterior_feature_names
        )
        self.repa_target_indices = self._resolve_repa_target_indices()
        self.repa_target_dim = (
            len(self.stage1v2_feature_names)
            if self.repa_target_indices is None
            else len(self.repa_target_indices)
        )
        if (
            config.repa_enabled or config.repa_weight > 0.0
        ) and config.stage1v2_posterior_feature_mode in {'none', 'zero'}:
            raise ValueError("REPA alignment requires non-zero Stage-1-v2/oracle feature mode")
        if config.repa_enabled and config.repa_weight <= 0.0 and self.is_main_process:
            print("WARNING: repa_enabled=True but repa_weight <= 0; REPA head will train with zero weight")
        if config.repa_weight > 0.0 and not config.repa_enabled:
            raise ValueError("repa_weight > 0 requires repa_enabled=True")
        if config.length_bucketed_train and not config.distributed:
            raise ValueError("length_bucketed_train currently requires distributed=True")
        if config.length_bucket_multiplier <= 0:
            raise ValueError(
                f"length_bucket_multiplier must be positive, got {config.length_bucket_multiplier}"
            )
        if config.prefetch_factor <= 0:
            raise ValueError(
                f"prefetch_factor must be positive, got {config.prefetch_factor}"
            )

        # Model
        print("Creating Stage-2 model...")
        self.interaction_prior_feature_dim = 1 if config.interaction_prior_feature_mode != 'none' else 0
        self.stage1v2_feature_dim = (
            len(self.stage1v2_feature_names)
            if config.stage1v2_posterior_feature_mode != 'none'
            else 0
        )
        interaction_prior_feature_dim = self.interaction_prior_feature_dim + self.stage1v2_feature_dim
        esm_gate_context_dim = 2 if config.esm_gate_context_mode == 'pocket_motion' else 0
        model_config = TorsionFlowNetConfig(
            esm_fusion_enabled=config.esm_fusion_enabled,
            esm_num_layers=config.esm_num_layers,
            esm_fusion_mode=config.esm_fusion_mode,
            esm_layer_dropout=config.esm_layer_dropout,
            esm_gate_bias=config.esm_gate_bias,
            esm_gate_context_dim=esm_gate_context_dim,
            nma_dim=config.nma_dim,
            stage1_chi_feature_scale=config.stage1_chi_feature_scale,
            interaction_prior_feature_dim=interaction_prior_feature_dim,
            interaction_prior_feature_scale=1.0,
            repa_enabled=config.repa_enabled,
            repa_dim=config.repa_dim,
            repa_target_dim=self.repa_target_dim,
        )
        self.model = TorsionFlowNet(model_config).to(self.device)
        resume_target_exists = bool(config.resume_from) or (
            bool(config.auto_resume) and (Path(config.save_dir) / 'last_checkpoint.pt').is_file()
        )
        if config.init_from_checkpoint and not resume_target_exists:
            self._init_model_from_checkpoint(config.init_from_checkpoint)
        elif config.init_from_checkpoint and self.is_main_process:
            print("[Init] Skipping init_from_checkpoint because this run will resume from its own checkpoint")

        # Stage-1 prior model
        self.stage1_model = None
        if config.use_stage1_prior:
            mode_requires_model = config.stage1_prior_mode in {'stage1', 'noisy_stage1_chi'}
            if mode_requires_model and not config.stage1_ckpt:
                raise ValueError("stage1_ckpt must be provided when use_stage1_prior=True")
            if mode_requires_model:
                self.stage1_model = self._load_stage1_model(config.stage1_ckpt)

        # FK module
        self.fk_module = create_openfold_fk().to(self.device)

        # Explicit local interaction prior.  This is a soft Stage-2 geometry
        # regularizer, not a deterministic holo endpoint.
        self.interaction_prior_model = None
        needs_interaction_prior_model = bool(config.interaction_prior_ckpt)
        if needs_interaction_prior_model:
            self.interaction_prior_model = load_interaction_prior(
                config.interaction_prior_ckpt,
                self.device,
            )
            if self.is_main_process:
                print(f"✓ Loaded interaction prior: {config.interaction_prior_ckpt}")

        # Data
        if self.is_main_process:
            print("Creating dataloaders...")
        self.train_loader = create_stage2_dataloader(
            config.data_dir,
            split='train',
            batch_size=config.batch_size,
            shuffle=not self.distributed,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            require_nma=config.use_nma,
            valid_samples_file=config.valid_samples_file,
            esm_num_layers=(config.esm_num_layers if config.esm_fusion_enabled else 1),
            stage1v2_posterior_cache_dir=train_stage1v2_dir,
            stage1v2_posterior_feature_mode=config.stage1v2_posterior_feature_mode,
            stage1v2_posterior_feature_names=config.stage1v2_posterior_feature_names,
            trust_prechecked_samples=config.trust_prechecked_samples,
        )
        self.val_loader = create_stage2_dataloader(
            config.data_dir,
            split=config.val_split,
            batch_size=config.val_batch_size or config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            require_nma=config.use_nma,
            valid_samples_file=config.val_samples_file,
            esm_num_layers=(config.esm_num_layers if config.esm_fusion_enabled else 1),
            stage1v2_posterior_cache_dir=val_stage1v2_dir,
            stage1v2_posterior_feature_mode=config.stage1v2_posterior_feature_mode,
            stage1v2_posterior_feature_names=config.stage1v2_posterior_feature_names,
            trust_prechecked_samples=config.trust_prechecked_samples,
        )
        train_size = len(self.train_loader.dataset)
        val_size = len(self.val_loader.dataset)
        if train_size <= 0:
            raise ValueError(
                "Stage-2 train dataset is empty after filtering. "
                f"valid_samples_file={config.valid_samples_file!r}"
            )
        if val_size <= 0:
            raise ValueError(
                "Stage-2 validation dataset is empty after filtering. "
                f"val_split={config.val_split!r} val_samples_file={config.val_samples_file!r}"
            )

        # Wrap model with DDP
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                # Stage-2 computes one loss from several model forwards
                # (reference CFM plus path integration). Keep optional heads as
                # zero-valued graph dependencies inside the model forward, then
                # use the faster non-unused DDP path here.
                find_unused_parameters=False,
            )
            if config.length_bucketed_train:
                lengths_obj = [None]
                if self.local_rank == 0:
                    lengths_obj[0] = self._load_length_bucket_costs()
                dist.broadcast_object_list(lengths_obj, src=0)
                sample_lengths = lengths_obj[0]
                if sample_lengths is None:
                    raise RuntimeError("Failed to broadcast Stage-2 sample lengths")
                train_batch_sampler = DistributedLengthBatchSampler(
                    sample_lengths,
                    batch_size=config.batch_size,
                    num_replicas=self.world_size,
                    rank=self.local_rank,
                    shuffle=True,
                    drop_last=config.length_bucket_drop_last,
                    seed=config.seed,
                    bucket_size_multiplier=config.length_bucket_multiplier,
                    residue_budget=config.length_bucket_residue_budget,
                )
                loader_kwargs = {
                    'num_workers': config.num_workers,
                    'collate_fn': self.train_loader.collate_fn,
                    'pin_memory': True,
                    'batch_sampler': train_batch_sampler,
                }
                if config.num_workers > 0:
                    loader_kwargs.update(
                        persistent_workers=True,
                        prefetch_factor=config.prefetch_factor,
                    )
                self.train_loader = torch.utils.data.DataLoader(
                    self.train_loader.dataset,
                    **loader_kwargs,
                )
                if self.is_main_process:
                    print(
                        "[DDP] Length-bucketed train batches enabled: "
                        f"bucket_multiplier={config.length_bucket_multiplier} "
                        f"drop_last={config.length_bucket_drop_last} "
                        f"residue_budget={config.length_bucket_residue_budget or 'OFF'} "
                        f"lengths={self._format_length_stats(sample_lengths)}"
                    )
            else:
                train_sampler = DistributedSampler(
                    self.train_loader.dataset,
                    num_replicas=self.world_size,
                    rank=self.local_rank,
                    shuffle=True,
                )
                loader_kwargs = {
                    'batch_size': config.batch_size,
                    'shuffle': False,
                    'num_workers': config.num_workers,
                    'collate_fn': self.train_loader.collate_fn,
                    'pin_memory': True,
                    'sampler': train_sampler,
                }
                if config.num_workers > 0:
                    loader_kwargs.update(
                        persistent_workers=True,
                        prefetch_factor=config.prefetch_factor,
                    )
                self.train_loader = torch.utils.data.DataLoader(
                    self.train_loader.dataset,
                    **loader_kwargs,
                )
            val_sampler = DistributedSampler(
                self.val_loader.dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=False,
            )
            loader_kwargs = {
                'batch_size': config.val_batch_size or config.batch_size,
                'shuffle': False,
                'num_workers': config.num_workers,
                'collate_fn': self.val_loader.collate_fn,
                'pin_memory': True,
                'sampler': val_sampler,
            }
            if config.num_workers > 0:
                loader_kwargs.update(
                    persistent_workers=True,
                    prefetch_factor=config.prefetch_factor,
                )
            self.val_loader = torch.utils.data.DataLoader(
                self.val_loader.dataset,
                **loader_kwargs,
            )
            if self.is_main_process:
                print(f"[DDP] Model wrapped with DistributedDataParallel")

        # Optimizer
        if self.is_main_process:
            print("Creating optimizer...")
        model_params = self.model.module.parameters() if self.distributed else self.model.parameters()
        self.optimizer = torch.optim.AdamW(
            model_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        self.grad_accum_steps = max(getattr(config, 'grad_accum_steps', 1), 1)
        updates_per_epoch = math.ceil(len(self.train_loader) / self.grad_accum_steps)
        total_steps = updates_per_epoch * config.max_epochs
        self.scheduler_t_max = max(total_steps - config.warmup_steps, 1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.scheduler_t_max,
            eta_min=config.lr * 0.01,
        )

        self.autocast_dtype = self._resolve_amp_dtype()
        self.scaler = GradScaler() if self.autocast_dtype == torch.float16 else None

        self.current_epoch = 0
        self.global_step = 0
        self.optimizer_step_count = 0
        self.best_val_metric = float('inf')
        self.patience_counter = 0

        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        self._maybe_resume()

        if self.is_main_process:
            print("✓ Stage-2 Trainer initialized")
            print(f"  - params: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"  - train samples: {len(self.train_loader.dataset)}")
            print(f"  - val samples: {len(self.val_loader.dataset)}")
            print(f"  - total steps: {total_steps:,}")
            print(f"  - world_size: {self.world_size}")
            print(f"  - effective batch_size: {config.batch_size * self.world_size}")
            print(f"  - resume epoch: {self.current_epoch}  global_step: {self.global_step}")

    def _resolve_amp_dtype(self) -> Optional[torch.dtype]:
        if not self.config.mixed_precision or self.device.type != 'cuda':
            return None
        amp_dtype = str(getattr(self.config, 'amp_dtype', 'bf16')).lower()
        if amp_dtype == 'auto':
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if amp_dtype == 'bf16':
            if not torch.cuda.is_bf16_supported():
                raise ValueError("amp_dtype=bf16 requested but CUDA device does not support bf16")
            return torch.bfloat16
        if amp_dtype == 'fp16':
            return torch.float16
        raise ValueError(f"Unsupported amp_dtype={self.config.amp_dtype}")

    def _check_finite_losses(self, losses: Dict[str, torch.Tensor], context: str) -> None:
        bad = []
        for name, value in losses.items():
            if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
                bad.append(name)
        if bad:
            details = {
                name: float(losses[name].detach().float().nan_to_num(posinf=1e30, neginf=-1e30).mean().item())
                for name in bad
            }
            raise FloatingPointError(f"Non-finite Stage-2 loss in {context}: {details}")

    def _check_finite_gradients(self) -> None:
        bad = []
        for name, param in self.model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                bad.append(name)
                if len(bad) >= 8:
                    break
        if bad:
            raise FloatingPointError(f"Non-finite Stage-2 gradients: {bad}")

    def _init_model_from_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"init_from_checkpoint='{checkpoint_path}' not found")
        if self.is_main_process:
            print(f"[Init] Loading model weights only from: {path}")
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        state = ckpt.get('model_state_dict') if isinstance(ckpt, dict) else None
        if state is None:
            raise ValueError(f"Checkpoint lacks model_state_dict: {path}")
        result = self.model.load_state_dict(state, strict=False)
        unexpected = list(result.unexpected_keys)
        missing = list(result.missing_keys)
        if unexpected or missing:
            raise RuntimeError(
                f"Warm-start state_dict mismatch unexpected={unexpected}, missing={missing}"
            )
        if self.is_main_process:
            print("[Init] Model warm-start OK; optimizer/scheduler/epoch remain fresh")

    def _all_reduce_loss_sums(self, sums: Dict[str, float], count: int) -> Tuple[Dict[str, float], int]:
        if not self.distributed:
            return sums, count
        values = [sums[k] for k in self._LOSS_KEYS] + [float(count)]
        tensor = torch.tensor(values, device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced = {k: float(tensor[i].item()) for i, k in enumerate(self._LOSS_KEYS)}
        return reduced, int(tensor[-1].item())

    @property
    def _raw_model(self):
        return self.model.module if self.distributed else self.model

    def _compute_interaction_prior_scores(self, batch, atom14_pos, atom14_mask) -> torch.Tensor:
        if self.interaction_prior_model is None:
            return batch.w_res.new_zeros(batch.w_res.shape)
        sc_mask = sidechain_atom_mask(atom14_mask.bool(), batch.node_mask.bool())
        with torch.no_grad():
            logits = self.interaction_prior_model(
                atom14_pos.float(),
                sc_mask,
                atom14_pos[:, :, 1].float(),
                batch.aatype,
                batch.lig_points.float(),
                batch.lig_types.float(),
                batch.lig_mask.bool(),
                batch.node_mask.bool(),
            )
        temp = max(float(self.config.interaction_prior_temperature), 1e-6)
        return torch.sigmoid(logits / temp) * batch.node_mask.float()

    def _differentiable_min_sidechain_ligand_dist(
        self,
        atom14_pos: torch.Tensor,
        atom_mask: torch.Tensor,
        lig_points: torch.Tensor,
        lig_mask: torch.Tensor,
        node_mask: torch.Tensor,
        residue_chunk: int = 64,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        bsz, n_res, n_atom, _ = atom14_pos.shape
        n_lig = lig_points.shape[1]
        out = atom14_pos.new_full((bsz, n_res), 50.0)
        if n_lig == 0:
            return out

        chunk = max(int(residue_chunk), 1)
        for start in range(0, n_res, chunk):
            end = min(start + chunk, n_res)
            c = end - start
            atom_chunk = atom14_pos[:, start:end].reshape(bsz, c, n_atom, 1, 3)
            lig_chunk = lig_points[:, None, None, :, :]
            dist = (atom_chunk - lig_chunk).square().sum(dim=-1).clamp_min(eps).sqrt()
            valid = (
                atom_mask[:, start:end, :, None].bool()
                & lig_mask[:, None, None, :].bool()
                & node_mask[:, start:end, None, None].bool()
            )
            dist = dist.masked_fill(~valid, 50.0)
            out[:, start:end] = dist.amin(dim=(-1, -2))
        return out

    def _interaction_prior_loss(self, prior_prob, atom14_pos, atom14_mask, batch) -> torch.Tensor:
        sc_mask = sidechain_atom_mask(atom14_mask.bool(), batch.node_mask.bool())
        min_dist = self._differentiable_min_sidechain_ligand_dist(
            atom14_pos.float(),
            sc_mask,
            batch.lig_points.float(),
            batch.lig_mask.bool(),
            batch.node_mask.bool(),
            residue_chunk=64,
        )
        contact_prob = torch.sigmoid(
            (float(self.config.interaction_prior_contact_dist) - min_dist.clamp(max=50.0))
            / max(float(self.config.interaction_prior_contact_tau), 1e-6)
        )
        weights = prior_prob.detach() * batch.node_mask.float()
        if self.config.interaction_prior_min_score > 0.0:
            weights = weights * (prior_prob.detach() >= float(self.config.interaction_prior_min_score)).float()
        denom = weights.sum().clamp(min=1e-8)
        loss = -torch.log(contact_prob.clamp(min=1e-6, max=1.0))
        return (loss * weights).sum() / denom

    def _sidechain_contact_score_and_dist(
        self,
        atom14_pos: torch.Tensor,
        atom14_mask: torch.Tensor,
        batch,
        w_res: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sc_mask = sidechain_atom_mask(atom14_mask.bool(), batch.node_mask.bool())
        min_dist = self._differentiable_min_sidechain_ligand_dist(
            atom14_pos.float(),
            sc_mask,
            batch.lig_points.float(),
            batch.lig_mask.bool(),
            batch.node_mask.bool(),
            residue_chunk=64,
        )
        contact_prob = torch.sigmoid(
            (float(self.config.interaction_prior_contact_dist) - min_dist.clamp(max=50.0))
            / max(float(self.config.interaction_prior_contact_tau), 1e-6)
        )
        pocket_mask = ((w_res > float(self.config.pocket_threshold)) & batch.node_mask.bool()).float()
        score = (contact_prob * pocket_mask).sum(dim=-1) / pocket_mask.sum(dim=-1).clamp(min=1e-8)
        return score, min_dist

    def _formed_contact_recall(
        self,
        d_apo: torch.Tensor,
        d_final: torch.Tensor,
        d_holo: torch.Tensor,
        batch,
    ) -> torch.Tensor:
        cutoff = float(self.config.interaction_prior_contact_dist)
        formed = (d_apo > cutoff) & (d_holo <= cutoff) & batch.node_mask.bool()
        recovered = (d_final <= cutoff) & formed
        denom = formed.float().sum()
        if float(denom.detach().item()) <= 0.0:
            return d_apo.new_tensor(0.0)
        return recovered.float().sum() / denom.clamp(min=1.0)

    def _compute_apo_interaction_prior_prob(self, batch, rigids_apo: Rigid) -> Optional[torch.Tensor]:
        if self.interaction_prior_model is None:
            if self.config.interaction_prior_feature_mode == 'zero':
                return batch.w_res.new_zeros(batch.w_res.shape)
            return None

        phi_psi_omega = batch.torsion_apo[..., :3]
        phi_psi_omega_sincos = torch.stack(
            [torch.sin(phi_psi_omega), torch.cos(phi_psi_omega)], dim=-1
        )
        apo_chi_sincos = torch.stack(
            [torch.sin(batch.torsion_apo[..., 3:7]), torch.cos(batch.torsion_apo[..., 3:7])],
            dim=-1,
        )
        apo_torsions_sincos = reorder_torsions_to_openfold(
            torch.cat([phi_psi_omega_sincos, apo_chi_sincos], dim=2)
        )
        atom14_apo = self.fk_module(apo_torsions_sincos, rigids_apo, batch.aatype)
        return self._compute_interaction_prior_scores(
            batch,
            atom14_apo['atom14_pos'],
            atom14_apo['atom14_mask'].bool(),
        )

    def _compute_holo_oracle_contact_prob(self, batch, rigids_holo: Rigid) -> torch.Tensor:
        phi_psi_omega = batch.torsion_holo[..., :3]
        phi_psi_omega_sincos = torch.stack(
            [torch.sin(phi_psi_omega), torch.cos(phi_psi_omega)], dim=-1
        )
        holo_chi_sincos = torch.stack(
            [torch.sin(batch.torsion_holo[..., 3:7]), torch.cos(batch.torsion_holo[..., 3:7])],
            dim=-1,
        )
        holo_torsions_sincos = reorder_torsions_to_openfold(
            torch.cat([phi_psi_omega_sincos, holo_chi_sincos], dim=2)
        )
        with torch.no_grad():
            atom14_holo = self.fk_module(holo_torsions_sincos, rigids_holo, batch.aatype)
            sc_mask = sidechain_atom_mask(atom14_holo['atom14_mask'].bool(), batch.node_mask.bool())
            min_dist = self._differentiable_min_sidechain_ligand_dist(
                atom14_holo['atom14_pos'].float(),
                sc_mask,
                batch.lig_points.float(),
                batch.lig_mask.bool(),
                batch.node_mask.bool(),
                residue_chunk=64,
            )
            prob = torch.sigmoid(
                (float(self.config.interaction_prior_contact_dist) - min_dist.clamp(max=50.0))
                / max(float(self.config.interaction_prior_contact_tau), 1e-6)
            )
            return prob * batch.node_mask.float()

    def _interaction_prior_feature(self, prior_prob: Optional[torch.Tensor], batch) -> Optional[torch.Tensor]:
        mode = self.config.interaction_prior_feature_mode
        if mode == 'none':
            return None
        if mode == 'zero':
            return batch.w_res.new_zeros(batch.w_res.shape)
        if prior_prob is None:
            raise RuntimeError(
                f"interaction_prior_feature_mode={mode} but prior_prob is unavailable"
            )
        return prior_prob.detach()

    def _combined_prior_features(
        self,
        interaction_prior_feature: Optional[torch.Tensor],
        batch,
    ) -> Optional[torch.Tensor]:
        features = []
        if self.interaction_prior_feature_dim > 0:
            if interaction_prior_feature is None:
                interaction_prior_feature = batch.w_res.new_zeros(batch.w_res.shape)
            if interaction_prior_feature.ndim == 2:
                interaction_prior_feature = interaction_prior_feature.unsqueeze(-1)
            if interaction_prior_feature.shape[-1] != self.interaction_prior_feature_dim:
                raise ValueError(
                    "interaction prior feature dim mismatch: "
                    f"{interaction_prior_feature.shape[-1]} != {self.interaction_prior_feature_dim}"
                )
            features.append(
                interaction_prior_feature.detach().float()
                * float(self.config.interaction_prior_feature_scale)
            )

        if self.stage1v2_feature_dim > 0:
            stage1v2_features = batch.stage1v2_posterior_features
            if stage1v2_features is None:
                stage1v2_features = batch.w_res.new_zeros(
                    (*batch.w_res.shape, self.stage1v2_feature_dim)
                )
            if stage1v2_features.shape[-1] != self.stage1v2_feature_dim:
                raise ValueError(
                    "Stage-1-v2 posterior feature dim mismatch: "
                    f"{stage1v2_features.shape[-1]} != {self.stage1v2_feature_dim}"
                )
            features.append(
                stage1v2_features.detach().float()
                * float(self.config.stage1v2_posterior_feature_scale)
            )

        if not features:
            return None
        out = torch.cat(features, dim=-1)
        return out * batch.node_mask.unsqueeze(-1).float()

    def _esm_gate_context(self, batch) -> Optional[torch.Tensor]:
        match self.config.esm_gate_context_mode:
            case 'none':
                return None
            case 'pocket_motion':
                motion = batch.w_res.new_zeros(batch.w_res.shape)
                if batch.stage1v2_posterior_features is not None:
                    try:
                        motion_idx = self.stage1v2_feature_names.index('motion_active')
                    except ValueError:
                        motion_idx = -1
                    if motion_idx >= 0:
                        motion = batch.stage1v2_posterior_features[..., motion_idx].detach().float()
                context = torch.stack([batch.w_res.detach().float(), motion], dim=-1)
                return context * batch.node_mask.unsqueeze(-1).float()
            case unreachable:
                raise ValueError(f"Unsupported esm_gate_context_mode={unreachable}")

    @staticmethod
    def _clip_velocity(x: torch.Tensor, limit: float) -> torch.Tensor:
        limit = float(limit)
        if limit <= 0.0:
            return x
        return x.clamp(min=-limit, max=limit)

    _STAGE1V2_FEATURE_ALIASES = {
        'contact_prob': ('contact_prob', 'contact_holo', 'pocket_mask'),
        'active_prob': ('active_prob', 'switch_prob'),
        'switch_prob': ('switch_prob', 'active_prob'),
        'motion_active': ('motion_active', 'active_prob', 'switch_prob'),
        'pocket_mask': ('pocket_mask', 'contact_prob', 'contact_holo'),
        'teacher_min_dist_pred': ('teacher_min_dist_pred', 'teacher_min_dist'),
        'teacher_min_dist': ('teacher_min_dist', 'teacher_min_dist_pred'),
        'signed_delta_dist_pred': ('signed_delta_dist_pred', 'signed_delta_dist'),
        'signed_delta_dist': ('signed_delta_dist', 'signed_delta_dist_pred'),
        'teacher_min_dist_pred_norm': ('teacher_min_dist_pred_norm', 'teacher_min_dist_norm'),
        'teacher_min_dist_norm': ('teacher_min_dist_norm', 'teacher_min_dist_pred_norm'),
        'signed_delta_dist_pred_norm': ('signed_delta_dist_pred_norm', 'signed_delta_dist_norm'),
        'signed_delta_dist_norm': ('signed_delta_dist_norm', 'signed_delta_dist_pred_norm'),
    }

    def _stage1v2_feature_tensor(
        self,
        batch,
        feature_name: str,
        *,
        strict: bool = True,
    ) -> Optional[torch.Tensor]:
        if self.stage1v2_feature_dim <= 0 or batch.stage1v2_posterior_features is None:
            if strict:
                raise RuntimeError(
                    f"Stage-1-v2 feature {feature_name!r} requested but posterior features are unavailable"
                )
            return None
        candidates = self._STAGE1V2_FEATURE_ALIASES.get(feature_name, (feature_name,))
        idx = next(
            (i for i, name in enumerate(self.stage1v2_feature_names) if name in candidates),
            None,
        )
        if idx is None:
            if strict:
                raise KeyError(
                    f"Stage-1-v2 feature {feature_name!r} not found in "
                    f"{self.stage1v2_feature_names}"
                )
            return None
        return batch.stage1v2_posterior_features[..., idx].detach().float()

    def _stage1v2_loss_signal(self, batch) -> Optional[torch.Tensor]:
        mode = self.config.stage1v2_loss_weight_mode
        alpha = float(self.config.stage1v2_loss_weight_alpha)
        if mode == 'none' or alpha <= 0.0:
            return None
        if mode == 'contact':
            signal = self._stage1v2_feature_tensor(batch, 'contact_prob')
        elif mode == 'active':
            signal = self._stage1v2_feature_tensor(batch, 'active_prob')
        elif mode == 'contact_active':
            contact = self._stage1v2_feature_tensor(batch, 'contact_prob')
            active = self._stage1v2_feature_tensor(batch, 'active_prob')
            signal = torch.maximum(contact, active)
        else:
            raise ValueError(f"Unsupported stage1v2_loss_weight_mode={mode}")
        return signal.clamp(min=0.0, max=1.0) * batch.node_mask.float()

    def _stage1v2_loss_weights(self, base_weights: torch.Tensor, batch) -> torch.Tensor:
        signal = self._stage1v2_loss_signal(batch)
        if signal is None:
            return base_weights
        return base_weights * (1.0 + float(self.config.stage1v2_loss_weight_alpha) * signal)

    def _stage1v2_guidance_prob(self, batch) -> Optional[torch.Tensor]:
        if float(self.config.w_stage1v2_guidance) <= 0.0:
            return None
        prob = self._stage1v2_feature_tensor(batch, self.config.stage1v2_guidance_feature)
        return prob.clamp(min=0.0, max=1.0) * batch.node_mask.float()

    def _stage1v2_guidance_loss(
        self,
        guidance_prob: torch.Tensor,
        atom14_pos: torch.Tensor,
        atom14_mask: torch.Tensor,
        batch,
    ) -> torch.Tensor:
        sc_mask = sidechain_atom_mask(atom14_mask.bool(), batch.node_mask.bool())
        min_dist = self._differentiable_min_sidechain_ligand_dist(
            atom14_pos.float(),
            sc_mask,
            batch.lig_points.float(),
            batch.lig_mask.bool(),
            batch.node_mask.bool(),
            residue_chunk=64,
        )
        contact_prob = torch.sigmoid(
            (float(self.config.interaction_prior_contact_dist) - min_dist.clamp(max=50.0))
            / max(float(self.config.interaction_prior_contact_tau), 1e-6)
        )
        weights = guidance_prob.detach() * batch.node_mask.float()
        min_prob = float(self.config.stage1v2_guidance_min_prob)
        if min_prob > 0.0:
            weights = weights * (guidance_prob.detach() >= min_prob).float()
        denom = weights.sum().clamp(min=1e-8)
        loss = -torch.log(contact_prob.clamp(min=1e-6, max=1.0))
        return (loss * weights).sum() / denom

    def _ligand_clearance_residue_mask(self, batch) -> torch.Tensor:
        node = batch.node_mask.bool()
        pocket = (batch.w_res > float(self.config.pocket_threshold)) & node
        pocket_feature = self._stage1v2_feature_tensor(batch, 'pocket_mask', strict=False)
        if pocket_feature is not None:
            pocket = pocket | ((pocket_feature > 0.5) & node)

        motion = torch.zeros_like(batch.w_res, dtype=torch.bool)
        motion_feature = self._stage1v2_feature_tensor(batch, 'motion_active', strict=False)
        if motion_feature is not None:
            motion = (motion_feature > 0.5) & node

        mode = self.config.ligand_clearance_mask_mode
        if mode == 'node':
            return node
        if mode == 'pocket':
            return pocket
        if mode == 'motion_active':
            return motion
        if mode == 'pocket_or_motion_active':
            return pocket | motion
        raise ValueError(f"Unsupported ligand_clearance_mask_mode={mode}")

    def _bridge_anchor_residue_mask(
        self,
        batch,
        clearance_mask: torch.Tensor,
        hard_negative_mask: torch.Tensor,
    ) -> torch.Tensor:
        node = batch.node_mask.bool()
        mode = self.config.bridge_anchor_mask_mode
        if mode == 'node':
            return node
        if mode == 'pocket':
            return clearance_mask & node
        if mode == 'non_clash_pocket':
            return clearance_mask & node & (~hard_negative_mask)
        if mode == 'non_clash_node':
            return node & (~hard_negative_mask)
        raise ValueError(f"Unsupported bridge_anchor_mask_mode={mode}")

    def _repa_alignment_mask(self, batch) -> torch.Tensor:
        mode = self.config.repa_mask_mode
        node = batch.node_mask.bool()
        if mode == 'node':
            return node.float()

        pocket = (batch.w_res > float(self.config.pocket_threshold)) & node
        pocket_feature = self._stage1v2_feature_tensor(batch, 'pocket_mask', strict=False)
        if pocket_feature is not None:
            pocket = pocket | ((pocket_feature > 0.5) & node)

        if mode == 'pocket':
            return pocket.float()

        motion = self._stage1v2_feature_tensor(batch, 'motion_active', strict=False)
        if motion is None:
            motion_mask = pocket
        else:
            motion_mask = (motion > 0.5) & node

        if mode == 'motion_active':
            return motion_mask.float()
        if mode == 'motion_active_or_pocket':
            return (motion_mask | pocket).float()
        raise ValueError(f"Unsupported repa_mask_mode={mode}")

    def _repa_alignment_loss(self, out: Dict[str, torch.Tensor], batch) -> torch.Tensor:
        if not self.config.repa_enabled or float(self.config.repa_weight) <= 0.0:
            return batch.w_res.new_tensor(0.0)
        student = out.get('repa_student')
        if student is None:
            raise RuntimeError("REPA is enabled but model output does not contain repa_student")
        target = self._repa_target_features(batch)
        if student.shape != target.shape:
            raise ValueError(
                f"REPA student/target shape mismatch: {tuple(student.shape)} != {tuple(target.shape)}"
            )
        mask = self._repa_alignment_mask(batch) * batch.node_mask.float()
        denom = mask.sum().clamp(min=1e-8)
        if self.config.repa_loss_type == 'cosine':
            per_res = 1.0 - (
                F.normalize(student.float(), dim=-1, eps=1e-6)
                * F.normalize(target, dim=-1, eps=1e-6)
            ).sum(dim=-1)
        elif self.config.repa_loss_type == 'mse':
            per_res = (student.float() - target).square().mean(dim=-1)
        else:
            raise ValueError(f"Unsupported repa_loss_type={self.config.repa_loss_type}")
        return (per_res * mask).sum() / denom

    @staticmethod
    def _residue_shuffle_tensor(
        target: torch.Tensor,
        node_mask: torch.Tensor,
        sample_ids: Optional[List[str]] = None,
        extra_seed: str = "repa_residue_shuffle",
    ) -> torch.Tensor:
        """Shuffle valid residue rows with a stable per-sample RNG seed."""
        shuffled = target.clone()
        valid_mask = node_mask.bool()
        if sample_ids is not None and len(sample_ids) < target.shape[0]:
            raise ValueError(
                f"sample_ids length {len(sample_ids)} is smaller than batch size {target.shape[0]}"
            )
        for batch_idx in range(target.shape[0]):
            valid_idx = valid_mask[batch_idx].nonzero(as_tuple=False).flatten()
            if valid_idx.numel() <= 1:
                continue
            seed_key = (
                sample_ids[batch_idx] if sample_ids is not None else f"_idx_{batch_idx}"
            )
            seed_str = f"{seed_key}|{extra_seed}"
            seed = int.from_bytes(
                hashlib.sha256(seed_str.encode("utf-8")).digest()[:8],
                byteorder="little",
                signed=False,
            ) % (2**31)
            cpu_gen = torch.Generator(device="cpu").manual_seed(seed)
            perm_local = torch.randperm(valid_idx.numel(), generator=cpu_gen).to(
                valid_idx.device
            )
            if torch.equal(perm_local, torch.arange(valid_idx.numel(), device=valid_idx.device)):
                perm_local = torch.roll(perm_local, shifts=1)
            permuted_idx = valid_idx[perm_local]
            shuffled[batch_idx, valid_idx] = target[batch_idx, permuted_idx]
        return shuffled

    def _repa_target_features(self, batch) -> torch.Tensor:
        if batch.stage1v2_posterior_features is None:
            raise RuntimeError("REPA alignment requires batch.stage1v2_posterior_features")
        target = (
            batch.stage1v2_posterior_features.detach().float()
            * float(self.config.stage1v2_posterior_feature_scale)
        )
        if self.repa_target_indices is not None:
            target = target[..., list(self.repa_target_indices)]
        if self.config.repa_target_shuffle_mode == 'none':
            return target
        if self.config.repa_target_shuffle_mode == 'residue':
            sample_ids = getattr(batch, "pdb_ids", None)
            return self._residue_shuffle_tensor(
                target,
                batch.node_mask,
                sample_ids=sample_ids,
            )
        raise ValueError(f"Unsupported repa_target_shuffle_mode={self.config.repa_target_shuffle_mode}")

    # FK module buffers are deterministic constants recomputed from
    # residue_constants.  They may be absent in older checkpoints that
    # pre-date the FK refactor, or present with a stale shape.  Either way
    # the freshly-constructed module already holds the correct values,
    # so we tolerate mismatches *only* for these keys.
    _FK_BUFFER_KEYS = frozenset({
        'fk_module.default_frames',
        'fk_module.restype_atom14_positions',
        'fk_module.restype_atom14_to_group',
        'fk_module.restype_atom14_mask',
    })

    def _load_stage1_model(self, ckpt_path: str) -> Stage1Model:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        model_config = self._resolve_stage1_model_config(ckpt)
        model = Stage1Model(model_config).to(self.device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        result = model.load_state_dict(state_dict, strict=False)
        unexpected = [k for k in result.unexpected_keys if k not in self._FK_BUFFER_KEYS]
        missing = [k for k in result.missing_keys if k not in self._FK_BUFFER_KEYS]
        if unexpected or missing:
            raise RuntimeError(
                f"Stage-1 checkpoint mismatch (ignoring FK buffers): "
                f"unexpected={unexpected}, missing={missing}"
            )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print(f"✓ Loaded Stage-1 prior: {ckpt_path}")
        return model

    def _build_rigids_from_backbone(self, N, Ca, C, mask, eps: float = 1e-8) -> Rigid:
        # e1: CA -> C
        e1 = C - Ca
        e1 = e1 / (torch.norm(e1, dim=-1, keepdim=True) + eps)

        # u: CA -> N
        u = N - Ca
        proj = (u * e1).sum(dim=-1, keepdim=True) * e1
        e2 = u - proj
        e2 = e2 / (torch.norm(e2, dim=-1, keepdim=True) + eps)

        # e3: right-handed
        e3 = torch.cross(e1, e2, dim=-1)

        R = torch.stack([e1, e2, e3], dim=-1)
        t = Ca

        if mask is not None:
            mask_exp = mask.unsqueeze(-1).unsqueeze(-1)
            eye = torch.eye(3, device=R.device).view(1, 1, 3, 3)
            R = torch.where(mask_exp, R, eye)
            t = torch.where(mask.unsqueeze(-1), t, torch.zeros_like(t))

        rotation = Rotation(rot_mats=R)
        return Rigid(rots=rotation, trans=t)

    def _rigid_to_rt(self, rigids: Rigid) -> Tuple[torch.Tensor, torch.Tensor]:
        R = rigids.get_rots().get_rot_mats()
        t = rigids.get_trans()
        return R, t

    def _rt_to_rigid(self, R: torch.Tensor, t: torch.Tensor) -> Rigid:
        return Rigid(rots=Rotation(rot_mats=R), trans=t)

    def _model_autocast(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return autocast(dtype=self.autocast_dtype)

    def _model_forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Run the vector-field network under AMP, but return fp32 velocities.

        Stage-2 geometry losses backpropagate through SE(3) log/exp, FK, and
        FAPE. Those operations are numerically fragile in bf16, so only the
        network matmul-heavy forward is autocast; downstream geometry remains
        fp32.
        """
        with self._model_autocast():
            out = self.model(**kwargs)
        return {
            key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
            for key, value in out.items()
        }

    def _compute_stage1_outputs(self, batch) -> Tuple[torch.Tensor, Rigid]:
        mode = getattr(self.config, 'stage1_prior_mode', 'stage1')
        if mode == 'apo_chi':
            return batch.torsion_apo[..., 3:7], self._build_rigids_from_backbone(
                batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
            )
        if mode == 'holo_chi':
            return batch.torsion_holo[..., 3:7], self._build_rigids_from_backbone(
                batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
            )
        if mode == 'holo_chi_apo_rigid':
            return batch.torsion_holo[..., 3:7], self._build_rigids_from_backbone(
                batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
            )
        if mode == 'apo_chi_holo_rigid':
            return batch.torsion_apo[..., 3:7], self._build_rigids_from_backbone(
                batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
            )
        if mode == 'holo_ligand_facing_chi':
            pocket_mask = (batch.w_res > self.config.prior_pocket_threshold).unsqueeze(-1)
            pred_chi = torch.where(pocket_mask, batch.torsion_holo[..., 3:7], batch.torsion_apo[..., 3:7])
            return pred_chi, self._build_rigids_from_backbone(
                batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
            )
        if self.stage1_model is None:
            raise RuntimeError("Stage-1 prior requested but model is not loaded")

        stage1_batch = SimpleNamespace(
            esm=batch.esm,
            N_apo=batch.N_apo,
            Ca_apo=batch.Ca_apo,
            C_apo=batch.C_apo,
            node_mask=batch.node_mask,
            lig_points=batch.lig_points,
            lig_types=batch.lig_types,
            lig_mask=batch.lig_mask,
            torsion_apo=batch.torsion_apo,
            sequences=batch.sequences,
        )

        with torch.no_grad():
            out = self.stage1_model(stage1_batch, current_step=0)
            pred_chi = torch.atan2(out['pred_chi'][..., 0], out['pred_chi'][..., 1])
            rigids_stage1 = out['rigids_final']
            if mode == 'noisy_stage1_chi' and self.config.stage1_prior_noise_scale > 0.0:
                noise = self.config.stage1_prior_noise_scale * torch.randn_like(pred_chi)
                pred_chi = wrap_to_pi(pred_chi + noise)
        return pred_chi, rigids_stage1

    def sample_reference_bridge(self, batch, t: torch.Tensor):
        # Chi endpoints
        chi0 = batch.torsion_apo[..., 3:7]
        chi1 = batch.torsion_holo[..., 3:7]
        delta_chi = wrap_to_pi(chi1 - chi0)
        gamma = (3 * t**2 - 2 * t**3).view(-1, 1, 1)
        dgamma = (6 * t - 6 * t**2).view(-1, 1, 1)
        chi_ref = chi0 + gamma * delta_chi
        d_chi_ref = dgamma * delta_chi

        # Rigids endpoints
        rigids_apo = self._build_rigids_from_backbone(
            batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
        )
        rigids_holo = self._build_rigids_from_backbone(
            batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
        )
        R0, t0 = self._rigid_to_rt(rigids_apo)
        R1, t1 = self._rigid_to_rt(rigids_holo)

        # Relative transform
        R0_inv, t0_inv = rigid_inverse(R0, t0)
        R_delta, t_delta = rigid_compose(R0_inv, t0_inv, R1, t1)
        xi = se3_log(R_delta, t_delta)

        # Interpolate
        xi_t = xi * gamma
        R_inc, t_inc = se3_exp(xi_t)
        R_t, t_t = rigid_compose(R0, t0, R_inc, t_inc)
        rigids_ref = self._rt_to_rigid(R_t, t_t)

        d_rot_ref = xi[..., :3] * dgamma
        d_trans_ref = xi[..., 3:] * dgamma

        # Clip reference velocities to prevent explosion
        d_chi_ref = d_chi_ref.clamp(min=-5.0, max=5.0)
        d_rot_ref = d_rot_ref.clamp(min=-5.0, max=5.0)
        d_trans_ref = d_trans_ref.clamp(min=-5.0, max=5.0)

        return chi_ref, rigids_ref, d_chi_ref, d_rot_ref, d_trans_ref, rigids_apo, rigids_holo

    def _interpolate_endpoints(
        self,
        batch,
        rigids_apo: Rigid,
        rigids_holo: Rigid,
        t_value: float,
    ) -> Tuple[Rigid, torch.Tensor]:
        t_scalar = float(t_value)
        gamma = 3.0 * t_scalar * t_scalar - 2.0 * t_scalar * t_scalar * t_scalar
        chi0 = batch.torsion_apo[..., 3:7]
        chi1 = batch.torsion_holo[..., 3:7]
        chi_t = wrap_to_pi(chi0 + gamma * wrap_to_pi(chi1 - chi0))

        R0, t0 = self._rigid_to_rt(rigids_apo)
        R1, t1 = self._rigid_to_rt(rigids_holo)
        R0_inv, t0_inv = rigid_inverse(R0, t0)
        R_delta, t_delta = rigid_compose(R0_inv, t0_inv, R1, t1)
        xi = se3_log(R_delta, t_delta)
        R_inc, t_inc = se3_exp(xi * gamma)
        R_t, trans_t = rigid_compose(R0, t0, R_inc, t_inc)
        return self._rt_to_rigid(R_t, trans_t), chi_t

    def _boundary_residual_envelope(self, t_value: float) -> float:
        t_scalar = min(max(float(t_value), 0.0), 1.0)
        if t_scalar <= 0.0 or t_scalar >= 1.0:
            return 0.0
        if self.config.boundary_residual_envelope == 'poly':
            return 4.0 * t_scalar * (1.0 - t_scalar)
        return math.sin(math.pi * t_scalar) ** 2

    def _boundary_residual_envelope_tensor(self, t: torch.Tensor) -> torch.Tensor:
        t_clamped = t.float().clamp(0.0, 1.0)
        if self.config.boundary_residual_envelope == 'poly':
            beta = 4.0 * t_clamped * (1.0 - t_clamped)
        else:
            beta = torch.sin(math.pi * t_clamped) ** 2
        interior = (t_clamped > 0.0) & (t_clamped < 1.0)
        return torch.where(interior, beta, torch.zeros_like(beta))

    @staticmethod
    def _safe_sample_id(sample_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id))

    def _teacher_residual_cache_path(self, sample_id: str) -> Path:
        if not self.config.teacher_residual_cache_dir:
            raise RuntimeError("teacher_residual_cache_dir is not set")
        return Path(self.config.teacher_residual_cache_dir) / f"{self._safe_sample_id(sample_id)}.npz"

    def _load_teacher_residual_targets(
        self,
        batch,
        t: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self.config.teacher_residual_cache_dir or self.config.w_teacher_residual <= 0.0:
            return None

        bsz, max_n = batch.node_mask.shape
        target_rot = torch.zeros((bsz, max_n, 3), dtype=torch.float32)
        target_trans = torch.zeros((bsz, max_n, 3), dtype=torch.float32)
        target_chi = torch.zeros((bsz, max_n, 4), dtype=torch.float32)
        target_mask = torch.zeros((bsz, max_n), dtype=torch.bool)
        target_chi_mask = torch.zeros((bsz, max_n, 4), dtype=torch.bool)
        target_motion_active = torch.zeros((bsz, max_n), dtype=torch.bool)
        target_pocket = torch.zeros((bsz, max_n), dtype=torch.bool)
        target_clash_relief_weight = torch.zeros((bsz, max_n), dtype=torch.float32)
        t_error = torch.zeros((bsz,), dtype=torch.float32)
        loaded = 0
        loaded_clash_relief = 0
        missing = []

        t_cpu = t.detach().float().cpu().tolist()
        for b, sample_id in enumerate(getattr(batch, "pdb_ids", [])):
            n_res = int(batch.n_residues[b])
            if not (
                float(self.config.teacher_residual_t_min)
                <= float(t_cpu[b])
                <= float(self.config.teacher_residual_t_max)
            ):
                continue
            path = self._teacher_residual_cache_path(str(sample_id))
            if not path.is_file():
                missing.append(str(path))
                continue

            with np.load(path, allow_pickle=False) as data:
                cached_id = str(data["sample_id"].item()) if "sample_id" in data else str(sample_id)
                if cached_id != str(sample_id):
                    raise ValueError(f"{path} sample_id={cached_id!r}, expected {sample_id!r}")
                cached_n = int(data["n_residues"].item()) if "n_residues" in data else n_res
                if cached_n != n_res:
                    raise ValueError(f"{path} n_residues={cached_n}, expected {n_res}")
                t_values = np.asarray(data["t_values"], dtype=np.float32)
                if t_values.ndim != 1 or t_values.size == 0:
                    raise ValueError(f"{path} has invalid t_values")
                idx = int(np.abs(t_values - float(t_cpu[b])).argmin())
                target_rot[b, :n_res] = torch.from_numpy(np.asarray(data["residual_rot"][idx, :n_res]))
                target_trans[b, :n_res] = torch.from_numpy(np.asarray(data["residual_trans"][idx, :n_res]))
                target_chi[b, :n_res] = torch.from_numpy(np.asarray(data["residual_chi"][idx, :n_res]))
                if "node_mask" in data:
                    target_mask[b, :n_res] = torch.from_numpy(np.asarray(data["node_mask"][:n_res]).astype(bool))
                else:
                    target_mask[b, :n_res] = batch.node_mask[b, :n_res].detach().cpu()
                if "chi_mask" in data:
                    target_chi_mask[b, :n_res] = torch.from_numpy(
                        np.asarray(data["chi_mask"][:n_res]).astype(bool)
                    )
                else:
                    target_chi_mask[b, :n_res] = batch.chi_mask[b, :n_res].detach().cpu()
                if "motion_active" in data:
                    target_motion_active[b, :n_res] = torch.from_numpy(
                        np.asarray(data["motion_active"][:n_res]).astype(bool)
                    )
                if "pocket_mask" in data:
                    target_pocket[b, :n_res] = torch.from_numpy(
                        np.asarray(data["pocket_mask"][:n_res]).astype(bool)
                    )
                else:
                    target_pocket[b, :n_res] = (
                        batch.w_res[b, :n_res].detach().cpu()
                        > float(self.config.pocket_threshold)
                    )
                if "clash_relief_weight" in data:
                    weight = np.asarray(data["clash_relief_weight"], dtype=np.float32)
                    if weight.ndim != 2:
                        raise ValueError(f"{path} clash_relief_weight must have shape [T, N]")
                    target_clash_relief_weight[b, :n_res] = torch.from_numpy(
                        np.nan_to_num(weight[idx, :n_res], nan=0.0, posinf=0.0, neginf=0.0)
                    ).clamp(min=0.0)
                    loaded_clash_relief += 1
                t_error[b] = abs(float(t_values[idx]) - float(t_cpu[b]))
                loaded += 1

        if missing and self.config.teacher_residual_missing_policy == 'error':
            shown = ", ".join(missing[:3])
            extra = "" if len(missing) <= 3 else f" ... (+{len(missing) - 3})"
            raise FileNotFoundError(f"Missing teacher residual cache: {shown}{extra}")
        if loaded == 0:
            return None

        target_rot = target_rot.to(self.device)
        target_trans = target_trans.to(self.device)
        target_chi = target_chi.to(self.device)
        target_mask = target_mask.to(self.device) & batch.node_mask.bool()
        target_chi_mask = target_chi_mask.to(self.device) & batch.chi_mask.bool()
        target_motion_active = target_motion_active.to(self.device)
        target_pocket = target_pocket.to(self.device)
        target_clash_relief_weight = target_clash_relief_weight.to(self.device).clamp(min=0.0)
        if not target_motion_active.any():
            residual_mag = (
                torch.linalg.norm(target_rot, dim=-1)
                + torch.linalg.norm(target_trans, dim=-1)
                + (target_chi.abs() * target_chi_mask.float()).sum(dim=-1)
            )
            target_motion_active = residual_mag > 1e-4

        mask_mode = self.config.teacher_residual_mask_mode
        if mask_mode.startswith('clash_relief') and loaded_clash_relief != loaded:
            raise ValueError(
                "teacher_residual_mask_mode="
                f"{mask_mode} requires v2 teacher residual cache with clash_relief_weight "
                f"for every loaded sample, got {loaded_clash_relief}/{loaded}"
            )
        clash_relief = (
            target_clash_relief_weight
            > float(self.config.teacher_residual_clash_weight_threshold)
        )
        if mask_mode == 'node':
            loss_mask = target_mask
        elif mask_mode == 'pocket':
            loss_mask = target_mask & target_pocket
        elif mask_mode == 'motion_active':
            loss_mask = target_mask & target_motion_active
        elif mask_mode == 'motion_active_or_pocket':
            loss_mask = target_mask & (target_motion_active | target_pocket)
        elif mask_mode == 'clash_relief':
            loss_mask = target_mask & clash_relief
        elif mask_mode == 'clash_relief_or_motion_active':
            loss_mask = target_mask & (clash_relief | target_motion_active)
        else:
            loss_mask = target_mask & (clash_relief | target_pocket)

        base_weight = torch.ones_like(target_clash_relief_weight)
        relief_weight = target_clash_relief_weight.clamp(min=0.0)
        if mask_mode.startswith('clash_relief'):
            teacher_weight = torch.where(
                relief_weight > 0.0,
                relief_weight,
                base_weight,
            )
        else:
            teacher_weight = base_weight
        teacher_weight = teacher_weight * loss_mask.float()

        return {
            'rot': target_rot,
            'trans': target_trans,
            'chi': target_chi,
            'mask': loss_mask,
            'weight': teacher_weight,
            'chi_mask': target_chi_mask & loss_mask.unsqueeze(-1),
            't_error': t_error.to(self.device),
        }

    def boundary_residual_path(
        self,
        batch,
        rigids_apo: Rigid,
        rigids_holo: Rigid,
        stage1_chi=None,
        stage1_rigids=None,
        stage1_chi_mask=None,
        interaction_prior=None,
        esm_gate_context=None,
    ) -> Tuple[List[Rigid], List[torch.Tensor], List[float]]:
        n_steps = self.config.n_integration_steps
        rigids_list: List[Rigid] = []
        chi_list: List[torch.Tensor] = []
        t_list: List[float] = []
        bsz = batch.esm.shape[0]
        residual_scale = float(self.config.boundary_residual_scale)

        for k in range(int(n_steps) + 1):
            t_val = k / max(int(n_steps), 1)
            interp_rigids, interp_chi = self._interpolate_endpoints(batch, rigids_apo, rigids_holo, t_val)
            beta = self._boundary_residual_envelope(t_val)

            if beta == 0.0:
                rigids_list.append(interp_rigids)
                chi_list.append(interp_chi)
                t_list.append(t_val)
                continue

            t_tensor = torch.full((bsz,), t_val, device=self.device)
            out = self._model_forward(
                chi=interp_chi,
                rigids=interp_rigids,
                esm=batch.esm,
                lig_points=batch.lig_points,
                lig_types=batch.lig_types,
                lig_mask=batch.lig_mask,
                w_res=batch.w_res,
                t=t_tensor,
                node_mask=batch.node_mask,
                nma_features=batch.nma_features,
                stage1_chi=stage1_chi,
                stage1_rigids=stage1_rigids,
                stage1_chi_mask=stage1_chi_mask,
                interaction_prior=interaction_prior,
                esm_gate_context=esm_gate_context,
                current_step=self.global_step,
            )
            d_chi = self._clip_velocity(out['d_chi'], self.config.integration_chi_clip)
            d_rot = self._clip_velocity(out['d_rigid_rot'], self.config.integration_rot_clip)
            d_trans = self._clip_velocity(out['d_rigid_trans'], self.config.integration_trans_clip)
            residual_weight = beta * residual_scale

            chi_t = wrap_to_pi(interp_chi + residual_weight * d_chi)
            R_ref, t_ref = self._rigid_to_rt(interp_rigids)
            R_res, t_res = se3_exp(torch.cat([d_rot, d_trans], dim=-1) * residual_weight)
            R_t, trans_t = rigid_compose(R_ref, t_ref, R_res, t_res)
            rigids_list.append(self._rt_to_rigid(R_t, trans_t))
            chi_list.append(chi_t)
            t_list.append(t_val)

        return rigids_list, chi_list, t_list

    def _terminal_projection_weight(self, t_value: float) -> float:
        t_scalar = min(max(float(t_value), 0.0), 1.0)
        if self.config.terminal_projection_schedule == 'quadratic':
            return t_scalar * t_scalar
        if self.config.terminal_projection_schedule == 'smoothstep':
            return 3.0 * t_scalar * t_scalar - 2.0 * t_scalar * t_scalar * t_scalar
        if self.config.terminal_projection_schedule == 'late_smoother':
            late_t = t_scalar**3
            return (
                10.0 * late_t**3
                - 15.0 * late_t**4
                + 6.0 * late_t**5
            )
        return (
            10.0 * t_scalar**3
            - 15.0 * t_scalar**4
            + 6.0 * t_scalar**5
        )

    def project_terminal_path(
        self,
        rigids_list: List[Rigid],
        chi_list: List[torch.Tensor],
        t_list: List[float],
        rigids_holo: Rigid,
        chi_holo: torch.Tensor,
    ) -> Tuple[List[Rigid], List[torch.Tensor], List[float]]:
        """Project a free path onto the holo endpoint with a smooth SE(3)/chi correction."""
        if not rigids_list or not chi_list or len(rigids_list) != len(chi_list):
            raise ValueError("terminal projection requires matching non-empty rigid/chi paths")

        final_R, final_t = self._rigid_to_rt(rigids_list[-1])
        holo_R, holo_t = self._rigid_to_rt(rigids_holo)
        final_R_inv, final_t_inv = rigid_inverse(final_R, final_t)
        delta_R, delta_t = rigid_compose(final_R_inv, final_t_inv, holo_R, holo_t)
        delta_xi = se3_log(delta_R, delta_t)
        delta_chi = wrap_to_pi(chi_holo - chi_list[-1])

        projected_rigids: List[Rigid] = []
        projected_chi: List[torch.Tensor] = []
        for rigids_t, chi_t, t_value in zip(rigids_list, chi_list, t_list):
            weight = self._terminal_projection_weight(float(t_value))
            if weight <= 0.0:
                projected_rigids.append(rigids_t)
                projected_chi.append(chi_t)
                continue
            if weight >= 1.0:
                projected_rigids.append(rigids_holo)
                projected_chi.append(chi_holo)
                continue

            R_t, trans_t = self._rigid_to_rt(rigids_t)
            R_corr, t_corr = se3_exp(delta_xi * weight)
            R_proj, trans_proj = rigid_compose(R_t, trans_t, R_corr, t_corr)
            projected_rigids.append(self._rt_to_rigid(R_proj, trans_proj))
            projected_chi.append(wrap_to_pi(chi_t + weight * delta_chi))

        return projected_rigids, projected_chi, list(t_list)

    def integrate_path(self,
                       batch,
                       rigids0: Rigid,
                       chi0: torch.Tensor,
                       stage1_chi=None,
                       stage1_rigids=None,
                       stage1_chi_mask=None,
                       interaction_prior=None,
                       esm_gate_context=None) -> Tuple[List[Rigid], List[torch.Tensor], List[float]]:
        """Integrate vector field from t=0 to t=1, return states."""
        n_steps = self.config.n_integration_steps
        dt = 1.0 / n_steps

        rigids = rigids0
        chi = chi0

        rigids_list = [rigids]
        chi_list = [chi]
        t_list = [0.0]

        for k in range(n_steps):
            t = torch.full((chi.shape[0],), k * dt, device=self.device)
            t_next = torch.full((chi.shape[0],), (k + 1) * dt, device=self.device)

            # Step 1: velocity at current state
            out1 = self._model_forward(
                chi=chi,
                rigids=rigids,
                esm=batch.esm,
                lig_points=batch.lig_points,
                lig_types=batch.lig_types,
                lig_mask=batch.lig_mask,
                w_res=batch.w_res,
                t=t,
                node_mask=batch.node_mask,
                nma_features=batch.nma_features,
                stage1_chi=stage1_chi,
                stage1_rigids=stage1_rigids,
                stage1_chi_mask=stage1_chi_mask,
                interaction_prior=interaction_prior,
                esm_gate_context=esm_gate_context,
                current_step=self.global_step,
            )

            d_chi1 = out1['d_chi']
            d_rot1 = out1['d_rigid_rot']
            d_trans1 = out1['d_rigid_trans']

            # Keep integration stable, but make the cutoffs explicit because
            # induced-fit translations can be several Angstrom.
            d_chi1 = self._clip_velocity(d_chi1, self.config.integration_chi_clip)
            d_rot1 = self._clip_velocity(d_rot1, self.config.integration_rot_clip)
            d_trans1 = self._clip_velocity(d_trans1, self.config.integration_trans_clip)

            # Predictor (Euler)
            chi_pred = wrap_to_pi(chi + dt * d_chi1)
            xi1 = torch.cat([d_rot1, d_trans1], dim=-1) * dt
            R_inc1, t_inc1 = se3_exp(xi1)
            R_curr, t_curr = self._rigid_to_rt(rigids)
            R_pred, t_pred = rigid_compose(R_curr, t_curr, R_inc1, t_inc1)
            rigids_pred = self._rt_to_rigid(R_pred, t_pred)

            # Step 2: velocity at predicted state
            out2 = self._model_forward(
                chi=chi_pred,
                rigids=rigids_pred,
                esm=batch.esm,
                lig_points=batch.lig_points,
                lig_types=batch.lig_types,
                lig_mask=batch.lig_mask,
                w_res=batch.w_res,
                t=t_next,
                node_mask=batch.node_mask,
                nma_features=batch.nma_features,
                stage1_chi=stage1_chi,
                stage1_rigids=stage1_rigids,
                stage1_chi_mask=stage1_chi_mask,
                interaction_prior=interaction_prior,
                esm_gate_context=esm_gate_context,
                current_step=self.global_step,
            )

            d_chi2 = out2['d_chi']
            d_rot2 = out2['d_rigid_rot']
            d_trans2 = out2['d_rigid_trans']

            d_chi2 = self._clip_velocity(d_chi2, self.config.integration_chi_clip)
            d_rot2 = self._clip_velocity(d_rot2, self.config.integration_rot_clip)
            d_trans2 = self._clip_velocity(d_trans2, self.config.integration_trans_clip)

            # Corrector (Heun)
            d_chi = 0.5 * (d_chi1 + d_chi2)
            d_rot = 0.5 * (d_rot1 + d_rot2)
            d_trans = 0.5 * (d_trans1 + d_trans2)

            chi = wrap_to_pi(chi + dt * d_chi)

            xi = torch.cat([d_rot, d_trans], dim=-1) * dt
            R_inc, t_inc = se3_exp(xi)
            R_new, t_new = rigid_compose(R_curr, t_curr, R_inc, t_inc)
            rigids = self._rt_to_rigid(R_new, t_new)

            rigids_list.append(rigids)
            chi_list.append(chi)
            t_list.append((k + 1) * dt)

        return rigids_list, chi_list, t_list

    def compute_losses(
        self,
        batch,
        t: torch.Tensor,
        *,
        force_geom: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Reference bridge
        (chi_ref, rigids_ref, d_chi_ref,
         d_rot_ref, d_trans_ref, rigids_apo, rigids_holo) = self.sample_reference_bridge(batch, t)

        # Effective weights
        w_eff = compute_w_eff(
            batch.w_res,
            batch.nma_features if self.config.use_nma else None,
            nma_lambda=self.config.nma_lambda,
            nma_time_decay=self.config.nma_time_decay,
            t=t,
        )
        w_pow = w_eff ** self.config.alpha
        loss_w = self._stage1v2_loss_weights(w_pow, batch)
        if self.config.use_pocket_local_prior:
            prior_mask = (batch.w_res > self.config.prior_pocket_threshold).float()
            prior_w = loss_w * prior_mask
        else:
            prior_w = loss_w

        if self.config.interaction_prior_feature_mode == 'oracle_contact':
            interaction_prior_prob = self._compute_holo_oracle_contact_prob(batch, rigids_holo)
        else:
            interaction_prior_prob = self._compute_apo_interaction_prior_prob(batch, rigids_apo)
        interaction_prior_feature = self._interaction_prior_feature(interaction_prior_prob, batch)
        combined_prior_features = self._combined_prior_features(interaction_prior_feature, batch)
        esm_gate_context = self._esm_gate_context(batch)
        stage1v2_guidance_prob = self._stage1v2_guidance_prob(batch)

        stage1_chi = None
        stage1_rigids = None
        stage1_chi_mask = None
        if self.config.use_stage1_prior:
            stage1_chi, stage1_rigids = self._compute_stage1_outputs(batch)
            if not self.config.use_stage1_rigid_prior:
                stage1_rigids = None
            if self.config.use_pocket_local_prior:
                stage1_chi_mask = (batch.w_res > self.config.prior_pocket_threshold).float()

        # Predict velocities at reference state
        out = self._model_forward(
            chi=chi_ref,
            rigids=rigids_ref,
            esm=batch.esm,
            lig_points=batch.lig_points,
            lig_types=batch.lig_types,
            lig_mask=batch.lig_mask,
            w_res=batch.w_res,
            t=t,
            node_mask=batch.node_mask,
            nma_features=batch.nma_features,
            stage1_chi=stage1_chi,
            stage1_rigids=stage1_rigids,
            stage1_chi_mask=stage1_chi_mask,
            interaction_prior=combined_prior_features,
            esm_gate_context=esm_gate_context,
            current_step=self.global_step,
            return_repa=self.config.repa_enabled and float(self.config.repa_weight) > 0.0,
        )

        d_chi_pred = out['d_chi']
        d_rot_pred = out['d_rigid_rot']
        d_trans_pred = out['d_rigid_trans']

        boundary_residual_mode = self.config.path_parameterization in {
            'boundary_residual_v1', 'boundary_residual'
        }

        # FM loss for free-flow modes. In boundary_residual_v1 the same logged
        # keys are residual regularizers: the model output is an endpoint-zero
        # displacement around the analytic apo-holo bridge, not a bridge
        # velocity target.
        chi_mask = batch.chi_mask.float()
        if boundary_residual_mode:
            chi_target = torch.zeros_like(d_chi_pred)
            rot_target = torch.zeros_like(d_rot_pred)
            trans_target = torch.zeros_like(d_trans_pred)
        else:
            chi_target = d_chi_ref
            rot_target = d_rot_ref
            trans_target = d_trans_ref

        fm_chi = ((d_chi_pred - chi_target) ** 2) * loss_w.unsqueeze(-1) * chi_mask
        fm_chi_denom = (chi_mask * loss_w.unsqueeze(-1)).sum().clamp(min=1e-8)
        L_fm_chi = fm_chi.sum() / fm_chi_denom

        fm_rot = ((d_rot_pred - rot_target) ** 2) * loss_w.unsqueeze(-1)
        fm_trans = ((d_trans_pred - trans_target) ** 2) * loss_w.unsqueeze(-1)
        L_fm_rigid = (fm_rot.sum() + fm_trans.sum()) / (loss_w.sum() + 1e-8)

        L_teacher_residual = chi_ref.new_tensor(0.0)
        L_teacher_residual_rigid = chi_ref.new_tensor(0.0)
        L_teacher_residual_chi = chi_ref.new_tensor(0.0)
        teacher_residual_target_norm = chi_ref.new_tensor(0.0)
        teacher_residual_pred_norm = chi_ref.new_tensor(0.0)
        teacher_residual_t_error = chi_ref.new_tensor(0.0)
        teacher_residual_weight_mean = chi_ref.new_tensor(0.0)
        teacher_residual_mask_frac = chi_ref.new_tensor(0.0)
        if boundary_residual_mode and self.config.w_teacher_residual > 0.0:
            teacher_targets = self._load_teacher_residual_targets(batch, t)
            if teacher_targets is not None:
                beta = self._boundary_residual_envelope_tensor(t).view(-1, 1, 1)
                residual_scale = beta * float(self.config.boundary_residual_scale)
                pred_rot_residual = d_rot_pred.float() * residual_scale
                pred_trans_residual = d_trans_pred.float() * residual_scale
                pred_chi_residual = d_chi_pred.float() * residual_scale
                target_rot = teacher_targets['rot'].float()
                target_trans = teacher_targets['trans'].float()
                target_chi = teacher_targets['chi'].float()
                teacher_mask = teacher_targets['mask'].float()
                teacher_weight = teacher_targets['weight'].float()
                teacher_chi_mask = teacher_targets['chi_mask'].float()
                rigid_denom = teacher_weight.sum().clamp(min=1.0)
                if self.config.teacher_residual_loss_type == 'huber':
                    beta = float(self.config.teacher_residual_huber_delta)
                    rot_loss = F.smooth_l1_loss(
                        pred_rot_residual,
                        target_rot,
                        reduction='none',
                        beta=beta,
                    ).sum(dim=-1)
                    trans_loss = F.smooth_l1_loss(
                        pred_trans_residual,
                        target_trans,
                        reduction='none',
                        beta=beta,
                    ).sum(dim=-1)
                    chi_delta = wrap_to_pi(pred_chi_residual - target_chi)
                    chi_loss = F.smooth_l1_loss(
                        chi_delta,
                        torch.zeros_like(chi_delta),
                        reduction='none',
                        beta=beta,
                    )
                else:
                    rot_loss = ((pred_rot_residual - target_rot) ** 2).sum(dim=-1)
                    trans_loss = ((pred_trans_residual - target_trans) ** 2).sum(dim=-1)
                    chi_loss = wrap_to_pi(pred_chi_residual - target_chi) ** 2

                L_teacher_residual_rigid = (
                    (rot_loss + trans_loss) * teacher_weight
                ).sum() / rigid_denom
                teacher_chi_weight = teacher_weight.unsqueeze(-1) * teacher_chi_mask
                chi_denom = teacher_chi_weight.sum().clamp(min=1.0)
                L_teacher_residual_chi = (
                    chi_loss * teacher_chi_weight
                ).sum() / chi_denom
                L_teacher_residual = L_teacher_residual_rigid + L_teacher_residual_chi

                target_residual_norm = (
                    torch.linalg.norm(target_rot, dim=-1)
                    + torch.linalg.norm(target_trans, dim=-1)
                    + (target_chi.abs() * teacher_chi_mask).sum(dim=-1)
                )
                pred_residual_norm = (
                    torch.linalg.norm(pred_rot_residual, dim=-1)
                    + torch.linalg.norm(pred_trans_residual, dim=-1)
                    + (pred_chi_residual.abs() * teacher_chi_mask).sum(dim=-1)
                )
                teacher_residual_target_norm = (
                    (target_residual_norm * teacher_weight).sum() / rigid_denom
                ).detach()
                teacher_residual_pred_norm = (
                    (pred_residual_norm * teacher_weight).sum() / rigid_denom
                ).detach()
                teacher_residual_t_error = teacher_targets['t_error'].mean().detach()
                active_teacher = teacher_mask > 0.0
                if active_teacher.any():
                    teacher_residual_weight_mean = teacher_weight[active_teacher].mean().detach()
                teacher_residual_mask_frac = (
                    teacher_mask.sum() / batch.node_mask.float().sum().clamp(min=1.0)
                ).detach()

        # Background stability
        bg_w = (1.0 - w_eff).clamp(min=0.0) ** self.config.bg_beta
        L_bg = (
            (bg_w * (d_rot_pred ** 2).sum(dim=-1)).sum() +
            (bg_w * (d_trans_pred ** 2).sum(dim=-1)).sum() +
            (bg_w.unsqueeze(-1) * (d_chi_pred ** 2) * chi_mask).sum()
        ) / (bg_w.sum() + 1e-8)

        # Integrate path for geometry (only every N steps for performance)
        compute_geom = force_geom or (
            self.global_step % self.config.geom_loss_every_n_steps == 0
        )

        L_smooth = chi_ref.new_tensor(0.0)
        L_clash = chi_ref.new_tensor(0.0)
        L_ligand_clearance = chi_ref.new_tensor(0.0)
        ligand_clearance_active_frac = chi_ref.new_tensor(0.0)
        ligand_clearance_min_dist = chi_ref.new_tensor(0.0)
        L_bridge_anchor = chi_ref.new_tensor(0.0)
        bridge_anchor_mask_frac = chi_ref.new_tensor(0.0)
        bridge_anchor_residual_norm = chi_ref.new_tensor(0.0)
        L_pep = chi_ref.new_tensor(0.0)
        L_contact = chi_ref.new_tensor(0.0)
        L_stage1v2_guidance = chi_ref.new_tensor(0.0)
        L_repa = self._repa_alignment_loss(out, batch)
        L_prior = chi_ref.new_tensor(0.0)
        L_interaction_prior = chi_ref.new_tensor(0.0)
        L_interaction_prior_final = chi_ref.new_tensor(0.0)
        contact_score_apo = chi_ref.new_tensor(0.0)
        contact_score_final = chi_ref.new_tensor(0.0)
        contact_score_holo = chi_ref.new_tensor(0.0)
        contact_score_gain = chi_ref.new_tensor(0.0)
        contact_score_holo_delta = chi_ref.new_tensor(0.0)
        contact_score_direction_acc = chi_ref.new_tensor(0.0)
        contact_score_holo_gap_abs = chi_ref.new_tensor(0.0)
        contact_score_sidechain_apo = chi_ref.new_tensor(0.0)
        contact_score_sidechain_final = chi_ref.new_tensor(0.0)
        contact_score_sidechain_holo = chi_ref.new_tensor(0.0)
        contact_score_sidechain_gain = chi_ref.new_tensor(0.0)
        contact_score_sidechain_holo_gap_abs = chi_ref.new_tensor(0.0)
        contact_score_sidechain_formed_recall = chi_ref.new_tensor(0.0)

        contact_scores = []
        contact_score_times = []
        stage1v2_guidance_terms = 0
        ligand_clearance_terms = 0
        ligand_clearance_active_terms = []
        ligand_clearance_min_terms = []
        bridge_anchor_terms = 0
        bridge_anchor_mask_terms = []
        bridge_anchor_residual_terms = []

        # Initialize endpoint losses to zero
        L_end = chi_ref.new_tensor(0.0)
        L_end_chi = chi_ref.new_tensor(0.0)
        L_end_fape = chi_ref.new_tensor(0.0)
        L_end_rigid = chi_ref.new_tensor(0.0)
        L_end_rigid_uw = chi_ref.new_tensor(0.0)
        L_end_chi_uw = chi_ref.new_tensor(0.0)

        if compute_geom:
            if boundary_residual_mode:
                rigids_list, chi_list, t_list = self.boundary_residual_path(
                    batch,
                    rigids_apo,
                    rigids_holo,
                    stage1_chi=stage1_chi,
                    stage1_rigids=stage1_rigids,
                    stage1_chi_mask=stage1_chi_mask,
                    interaction_prior=combined_prior_features,
                    esm_gate_context=esm_gate_context,
                )
            else:
                rigids_list, chi_list, t_list = self.integrate_path(
                    batch,
                    rigids_apo,
                    batch.torsion_apo[..., 3:7],
                    stage1_chi=stage1_chi,
                    stage1_rigids=stage1_rigids,
                    stage1_chi_mask=stage1_chi_mask,
                    interaction_prior=combined_prior_features,
                    esm_gate_context=esm_gate_context,
                )
                if self.config.path_parameterization == 'projected_flow':
                    rigids_list, chi_list, t_list = self.project_terminal_path(
                        rigids_list,
                        chi_list,
                        t_list,
                        rigids_holo,
                        batch.torsion_holo[..., 3:7],
                    )

            # Integration is part of the scientific contract. Do not silently
            # downgrade to FM-only if the path state is numerically invalid.
            has_nonfinite = any(
                (not torch.isfinite(r.get_trans()).all())
                or (not torch.isfinite(r.get_rots().get_rot_mats()).all())
                for r in rigids_list
            ) or any(not torch.isfinite(c).all() for c in chi_list)

            if has_nonfinite:
                raise FloatingPointError(f"Non-finite Stage-2 integrated path at step {self.global_step}")

            # Select geometry steps
            n_geom = min(self.config.n_geom_steps, len(t_list))
            geom_indices = torch.linspace(0, len(t_list) - 1, steps=n_geom).long().tolist()

            # Geometry losses along path
            prev_R, prev_t = None, None
            prev_chi = None

            phi_psi_omega = batch.torsion_apo[..., :3]
            phi_psi_omega_sincos = torch.stack(
                [torch.sin(phi_psi_omega), torch.cos(phi_psi_omega)], dim=-1
            )
            for idx in geom_indices:
                rigids_t = rigids_list[idx]
                chi_t = chi_list[idx]
                t_val = t_list[idx]

                # FK decode
                chi_sincos = torch.stack([torch.sin(chi_t), torch.cos(chi_t)], dim=-1)
                torsions_sincos = reorder_torsions_to_openfold(
                    torch.cat([phi_psi_omega_sincos, chi_sincos], dim=2)
                )
                atom14 = self.fk_module(torsions_sincos, rigids_t, batch.aatype)

                atom14_pos = atom14['atom14_pos'].clamp(min=-1000.0, max=1000.0)
                atom14_mask = atom14['atom14_mask'].bool()

                # Smoothness (between consecutive geometry steps)
                R_t, t_t = self._rigid_to_rt(rigids_t)
                if prev_R is not None:
                    R_inv, t_inv = rigid_inverse(prev_R, prev_t)
                    R_delta, t_delta = rigid_compose(R_inv, t_inv, R_t, t_t)
                    xi_delta = se3_log(R_delta, t_delta)
                    L_smooth = L_smooth + ((xi_delta ** 2).sum(dim=-1) * loss_w).sum() / (loss_w.sum() + 1e-8)

                    d_chi = wrap_to_pi(chi_t - prev_chi)
                    chi_smooth = ((d_chi ** 2) * chi_mask) * loss_w.unsqueeze(-1)
                    chi_smooth_denom = (chi_mask * loss_w.unsqueeze(-1)).sum().clamp(min=1e-8)
                    L_smooth = L_smooth + chi_smooth.sum() / chi_smooth_denom

                prev_R, prev_t = R_t, t_t
                prev_chi = chi_t

                # Clash (mask invalid atoms to avoid padded clashes)
                valid_atom = atom14_mask & batch.node_mask.unsqueeze(-1)
                flat_atoms = atom14_pos.reshape(atom14_pos.shape[0], -1, 3)
                flat_atom_mask = valid_atom.reshape(valid_atom.shape[0], -1)
                L_clash = L_clash + clash_penalty(
                    flat_atoms,
                    clash_threshold=2.2,
                    aatype=batch.aatype,
                    atom_mask=flat_atom_mask,
                )

                clearance_time_active = (
                    float(self.config.ligand_clearance_t_min)
                    <= float(t_val)
                    <= float(self.config.ligand_clearance_t_max)
                )
                anchor_time_active = (
                    float(self.config.bridge_anchor_t_min)
                    <= float(t_val)
                    <= float(self.config.bridge_anchor_t_max)
                )
                needs_ligand_distance = (
                    float(self.config.w_ligand_clearance) > 0.0
                    and clearance_time_active
                ) or (
                    float(self.config.w_bridge_anchor) > 0.0
                    and anchor_time_active
                )
                if needs_ligand_distance:
                    clearance_mask = self._ligand_clearance_residue_mask(batch)
                    min_dist = self._differentiable_min_sidechain_ligand_dist(
                        atom14_pos.float(),
                        valid_atom,
                        batch.lig_points.float(),
                        batch.lig_mask.bool(),
                        batch.node_mask.bool(),
                        residue_chunk=64,
                    )
                    hard_negative_all = (
                        min_dist.detach()
                        < float(self.config.ligand_clearance_hard_negative_dist)
                    ) & batch.node_mask.bool()
                    hard_negative = hard_negative_all & clearance_mask
                    clearance_violation = torch.relu(
                        min_dist.new_tensor(float(self.config.ligand_clearance_dist))
                        - min_dist.clamp(max=50.0)
                    )
                    clearance_w_all = clearance_mask.float()
                    clearance_denom_all = clearance_w_all.sum().clamp(min=1.0)
                    if (
                        float(self.config.w_ligand_clearance) > 0.0
                        and clearance_time_active
                    ):
                        if self.config.ligand_clearance_loss_mode == 'hard_negative':
                            clearance_w = hard_negative.float()
                        else:
                            clearance_w = clearance_w_all
                        denom = clearance_w.sum().clamp(min=1.0)
                        L_ligand_clearance = L_ligand_clearance + (
                            clearance_violation.pow(2) * clearance_w
                        ).sum() / denom
                        ligand_clearance_active_terms.append(
                            (hard_negative.float() * clearance_w_all).sum()
                            / clearance_denom_all
                        )
                        ligand_clearance_min_terms.append(
                            (min_dist.clamp(max=50.0) * clearance_w_all).sum()
                            / clearance_denom_all
                        )
                        ligand_clearance_terms += 1

                    if (
                        float(self.config.w_bridge_anchor) > 0.0
                        and anchor_time_active
                    ):
                        anchor_mask = self._bridge_anchor_residue_mask(
                            batch,
                            clearance_mask,
                            hard_negative_all,
                        )
                        anchor_w = anchor_mask.float()
                        anchor_denom = anchor_w.sum().clamp(min=1.0)
                        bridge_rigids, bridge_chi = self._interpolate_endpoints(
                            batch,
                            rigids_apo,
                            rigids_holo,
                            float(t_val),
                        )
                        R_bridge, t_bridge = self._rigid_to_rt(bridge_rigids)
                        R_bridge_inv, t_bridge_inv = rigid_inverse(R_bridge, t_bridge)
                        R_t_cur, t_t_cur = self._rigid_to_rt(rigids_t)
                        R_delta_bridge, t_delta_bridge = rigid_compose(
                            R_bridge_inv,
                            t_bridge_inv,
                            R_t_cur,
                            t_t_cur,
                        )
                        xi_bridge = se3_log(R_delta_bridge, t_delta_bridge)
                        rigid_anchor = (xi_bridge ** 2).sum(dim=-1)
                        chi_delta_bridge = wrap_to_pi(chi_t - bridge_chi)
                        chi_anchor = (
                            (chi_delta_bridge ** 2) * chi_mask
                        ).sum(dim=-1) / chi_mask.sum(dim=-1).clamp(min=1.0)
                        anchor_residual = rigid_anchor + chi_anchor
                        L_bridge_anchor = L_bridge_anchor + (
                            anchor_residual * anchor_w
                        ).sum() / anchor_denom
                        bridge_anchor_mask_terms.append(
                            anchor_w.sum()
                            / batch.node_mask.float().sum().clamp(min=1.0)
                        )
                        bridge_anchor_residual_terms.append(
                            (anchor_residual.sqrt() * anchor_w).sum()
                            / anchor_denom
                        )
                        bridge_anchor_terms += 1

                # Peptide geometry
                L_pep = L_pep + compute_peptide_loss(
                    atom14_pos,
                    atom14_mask,
                    batch.node_mask,
                    bond_len=self.config.pep_bond_len,
                    angle_cacn=self.config.pep_angle_cacn,
                    angle_cnca=self.config.pep_angle_cnca,
                    angle_weight=self.config.pep_angle_weight,
                )

                # Contact score
                C_t = compute_contact_score(
                    atom14_pos,
                    valid_atom,
                    batch.lig_points,
                    batch.lig_mask,
                    w_eff,
                    pocket_threshold=self.config.pocket_threshold,
                    d_c=self.config.contact_d0,
                    tau=self.config.contact_tau,
                )
                contact_scores.append(C_t)
                contact_score_times.append(float(t_val))

                # Prior (late time)
                if stage1_chi is not None and t_val >= self.config.t_mid:
                    if prior_w.sum() > 0:
                        d_chi_prior = wrap_to_pi(chi_t - stage1_chi)
                        prior_term = ((d_chi_prior ** 2) * chi_mask) * prior_w.unsqueeze(-1)
                        prior_denom = (chi_mask * prior_w.unsqueeze(-1)).sum().clamp(min=1e-8)
                        L_prior = L_prior + prior_term.sum() / prior_denom

                        if stage1_rigids is not None:
                            R_t, t_t = self._rigid_to_rt(rigids_t)
                            R1, t1 = self._rigid_to_rt(stage1_rigids)
                            R_inv, t_inv = rigid_inverse(R_t, t_t)
                            R_delta, t_delta = rigid_compose(R_inv, t_inv, R1, t1)
                            xi_prior = se3_log(R_delta, t_delta)
                            prior_rigid = ((xi_prior ** 2).sum(dim=-1) * prior_w).sum() / (prior_w.sum() + 1e-8)
                            L_prior = L_prior + prior_rigid

                if (
                    interaction_prior_prob is not None
                    and t_val >= self.config.interaction_prior_t_mid
                ):
                    L_interaction_prior = L_interaction_prior + self._interaction_prior_loss(
                        interaction_prior_prob,
                        atom14_pos,
                        valid_atom,
                        batch,
                    )
                if (
                    stage1v2_guidance_prob is not None
                    and t_val >= float(self.config.stage1v2_guidance_t_mid)
                ):
                    L_stage1v2_guidance = L_stage1v2_guidance + self._stage1v2_guidance_loss(
                        stage1v2_guidance_prob,
                        atom14_pos,
                        valid_atom,
                        batch,
                    )
                    stage1v2_guidance_terms += 1

            # Endpoint loss
            # Precompute holo atom14 for FAPE
            torsion_holo = batch.torsion_holo
            phi_psi_omega_holo = torsion_holo[..., :3]
            phi_psi_omega_holo_sincos = torch.stack(
                [torch.sin(phi_psi_omega_holo), torch.cos(phi_psi_omega_holo)], dim=-1
            )
            chi_holo = torsion_holo[..., 3:7]
            chi_holo_sincos = torch.stack([torch.sin(chi_holo), torch.cos(chi_holo)], dim=-1)
            torsions_holo_sincos = reorder_torsions_to_openfold(
                torch.cat([phi_psi_omega_holo_sincos, chi_holo_sincos], dim=2)
            )
            atom14_holo = self.fk_module(torsions_holo_sincos, rigids_holo, batch.aatype)
            holo_valid_atom = (
                atom14_holo['atom14_mask'].bool()
                & batch.node_mask.unsqueeze(-1)
            )
            contact_score_holo_batch = compute_contact_score(
                atom14_holo['atom14_pos'].clamp(min=-1000.0, max=1000.0),
                holo_valid_atom,
                batch.lig_points,
                batch.lig_mask,
                w_eff,
                pocket_threshold=self.config.pocket_threshold,
                d_c=self.config.contact_d0,
                tau=self.config.contact_tau,
            )

            if self.config.contact_loss_mode == 'monotonic_increase':
                for k in range(len(contact_scores) - 1):
                    delta = contact_scores[k] - contact_scores[k + 1] - self.config.contact_eps
                    L_contact = L_contact + torch.relu(delta).pow(2).mean()
            elif contact_scores:
                contact_start = contact_scores[0].detach()
                contact_target_delta = (contact_score_holo_batch - contact_start).detach()
                for C_t, t_val in zip(contact_scores, contact_score_times):
                    target_t = contact_start + float(t_val) * contact_target_delta
                    L_contact = L_contact + (C_t - target_t).abs().mean()
                L_contact = L_contact / max(len(contact_scores), 1)

            if stage1v2_guidance_terms > 0:
                L_stage1v2_guidance = L_stage1v2_guidance / stage1v2_guidance_terms
            if ligand_clearance_terms > 0:
                L_ligand_clearance = L_ligand_clearance / ligand_clearance_terms
                ligand_clearance_active_frac = torch.stack(
                    ligand_clearance_active_terms
                ).mean()
                ligand_clearance_min_dist = torch.stack(
                    ligand_clearance_min_terms
                ).mean()
            if bridge_anchor_terms > 0:
                L_bridge_anchor = L_bridge_anchor / bridge_anchor_terms
                bridge_anchor_mask_frac = torch.stack(
                    bridge_anchor_mask_terms
                ).mean()
                bridge_anchor_residual_norm = torch.stack(
                    bridge_anchor_residual_terms
                ).mean()

            rigids_final = rigids_list[-1]
            chi_final = chi_list[-1]
            R_final, t_final = self._rigid_to_rt(rigids_final)
            R_holo, t_holo = self._rigid_to_rt(rigids_holo)
            R_inv, t_inv = rigid_inverse(R_final, t_final)
            R_delta, t_delta = rigid_compose(R_inv, t_inv, R_holo, t_holo)
            xi_end = se3_log(R_delta, t_delta)
            L_end_rigid = ((xi_end ** 2).sum(dim=-1) * loss_w).sum() / (loss_w.sum() + 1e-8)
            node_w = batch.node_mask.float()
            L_end_rigid_uw = ((xi_end ** 2).sum(dim=-1) * node_w).sum() / node_w.sum().clamp(min=1e-8)

            d_chi_end = wrap_to_pi(chi_final - batch.torsion_holo[..., 3:7])
            end_chi_term = ((d_chi_end ** 2) * chi_mask) * loss_w.unsqueeze(-1)
            end_chi_denom = (chi_mask * loss_w.unsqueeze(-1)).sum().clamp(min=1e-8)
            L_end_chi = end_chi_term.sum() / end_chi_denom
            L_end_chi_uw = ((d_chi_end ** 2) * chi_mask).sum() / chi_mask.sum().clamp(min=1e-8)

            # FAPE endpoint
            chi_sincos = torch.stack([torch.sin(chi_final), torch.cos(chi_final)], dim=-1)
            torsions_final_sincos = reorder_torsions_to_openfold(
                torch.cat([phi_psi_omega_sincos, chi_sincos], dim=2)
            )
            atom14_final = self.fk_module(torsions_final_sincos, rigids_final, batch.aatype)
            final_valid_atom = (
                atom14_final['atom14_mask'].bool()
                & batch.node_mask.unsqueeze(-1)
            )

            if contact_scores:
                contact_score_apo = contact_scores[0].mean()
                contact_score_final = contact_scores[-1].mean()
                contact_score_gain = contact_score_final - contact_score_apo
                contact_score_holo = contact_score_holo_batch.mean()
                contact_score_holo_delta_batch = contact_score_holo_batch - contact_scores[0]
                contact_score_holo_delta = contact_score_holo_delta_batch.mean()
                contact_score_holo_gap_abs = (contact_scores[-1] - contact_score_holo_batch).abs().mean()
                contact_score_actual_delta = contact_scores[-1] - contact_scores[0]
                direction_mask = (
                    contact_score_holo_delta_batch.abs()
                    > float(self.config.contact_direction_eps)
                )
                if direction_mask.any():
                    direction_ok = (
                        contact_score_actual_delta[direction_mask]
                        * contact_score_holo_delta_batch[direction_mask]
                    ) > 0
                    contact_score_direction_acc = direction_ok.float().mean()

                atom14_apo = self.fk_module(
                    reorder_torsions_to_openfold(
                        torch.cat([
                            phi_psi_omega_sincos,
                            torch.stack(
                                [
                                    torch.sin(batch.torsion_apo[..., 3:7]),
                                    torch.cos(batch.torsion_apo[..., 3:7]),
                                ],
                                dim=-1,
                            ),
                        ], dim=2)
                    ),
                    rigids_apo,
                    batch.aatype,
                )
                sidechain_apo_batch, sidechain_d_apo = self._sidechain_contact_score_and_dist(
                    atom14_apo['atom14_pos'].clamp(min=-1000.0, max=1000.0),
                    atom14_apo['atom14_mask'].bool() & batch.node_mask.unsqueeze(-1),
                    batch,
                    w_eff,
                )
                sidechain_final_batch, sidechain_d_final = self._sidechain_contact_score_and_dist(
                    atom14_final['atom14_pos'].clamp(min=-1000.0, max=1000.0),
                    final_valid_atom,
                    batch,
                    w_eff,
                )
                sidechain_holo_batch, sidechain_d_holo = self._sidechain_contact_score_and_dist(
                    atom14_holo['atom14_pos'].clamp(min=-1000.0, max=1000.0),
                    holo_valid_atom,
                    batch,
                    w_eff,
                )
                contact_score_sidechain_apo = sidechain_apo_batch.mean()
                contact_score_sidechain_final = sidechain_final_batch.mean()
                contact_score_sidechain_holo = sidechain_holo_batch.mean()
                contact_score_sidechain_gain = contact_score_sidechain_final - contact_score_sidechain_apo
                contact_score_sidechain_holo_gap_abs = (
                    sidechain_final_batch - sidechain_holo_batch
                ).abs().mean()
                contact_score_sidechain_formed_recall = self._formed_contact_recall(
                    sidechain_d_apo,
                    sidechain_d_final,
                    sidechain_d_holo,
                    batch,
                )

            if interaction_prior_prob is not None:
                L_interaction_prior_final = self._interaction_prior_loss(
                    interaction_prior_prob,
                    atom14_final['atom14_pos'].clamp(min=-1000.0, max=1000.0),
                    final_valid_atom,
                    batch,
                )

            pred_R = rigids_final.get_rots().get_rot_mats()
            pred_t = rigids_final.get_trans()
            true_R = rigids_holo.get_rots().get_rot_mats()
            true_t = rigids_holo.get_trans()
            atom14_overlap_mask = (
                atom14_final['atom14_mask'].bool()
                & atom14_holo['atom14_mask'].bool()
                & batch.node_mask.unsqueeze(-1)
            )

            L_end_fape = fape_loss(
                atom14_final['atom14_pos'],
                atom14_holo['atom14_pos'],
                (pred_R, pred_t),
                (true_R, true_t),
                w_res=None,
                atom_mask=atom14_overlap_mask,
            )

            L_end = (self.config.w_end_chi * L_end_chi +
                     self.config.w_end_fape * L_end_fape +
                     L_end_rigid)

        total_no_repa = (
            self.config.w_fm_chi * L_fm_chi.clamp(max=100.0) +
            self.config.w_fm_rigid * L_fm_rigid.clamp(max=100.0) +
            self.config.w_teacher_residual * L_teacher_residual.clamp(max=100.0) +
            self.config.w_bg * L_bg.clamp(max=100.0) +
            self.config.w_smooth * L_smooth.clamp(max=100.0) +
            self.config.w_clash * L_clash.clamp(max=100.0) +
            self.config.w_ligand_clearance * L_ligand_clearance.clamp(max=100.0) +
            self.config.w_bridge_anchor * L_bridge_anchor.clamp(max=100.0) +
            self.config.w_pep * L_pep.clamp(max=100.0) +
            self.config.w_contact * L_contact.clamp(max=100.0) +
            self.config.w_stage1v2_guidance * L_stage1v2_guidance.clamp(max=100.0) +
            self.config.w_prior * L_prior.clamp(max=100.0) +
            self.config.w_interaction_prior * L_interaction_prior.clamp(max=100.0) +
            self.config.w_end * L_end.clamp(max=100.0)
        )
        total = total_no_repa + self.config.repa_weight * L_repa.clamp(max=100.0)

        zero = chi_ref.new_tensor(0.0)
        L_esm_entropy = zero
        esm_layer_weight_entropy_raw = zero
        esm_layer_weight_max = zero
        esm_gate_mean = zero
        esm_gate_pocket_mean = zero
        esm_gate_nonpocket_mean = zero
        esm_layer_weight_logs = {
            f'esm_layer_weight_{idx}': zero
            for idx in range(10)
        }

        esm_lw = out.get("esm_layer_weights")
        if esm_lw is not None:
            esm_lw_flat = esm_lw.float().reshape(-1)
            entropy = -(esm_lw_flat * torch.log(esm_lw_flat + 1e-8)).sum()
            esm_layer_weight_entropy_raw = entropy.detach()
            esm_layer_weight_max = esm_lw_flat.max().detach()
            for idx in range(min(10, esm_lw_flat.numel())):
                esm_layer_weight_logs[f'esm_layer_weight_{idx}'] = esm_lw_flat[idx].detach()
            if self.config.esm_layer_entropy_weight > 0.0:
                L_esm_entropy = self.config.esm_layer_entropy_weight * entropy
                total = total + L_esm_entropy

        esm_gates = out.get("esm_layer_gates")
        if esm_gates is not None:
            gates = esm_gates.float()
            node_mask = batch.node_mask.bool()
            gate_mask = node_mask.unsqueeze(-1).expand_as(gates)
            gate_mask_f = gate_mask.float()
            esm_gate_mean = (
                (gates * gate_mask_f).sum()
                / gate_mask_f.sum().clamp(min=1.0)
            ).detach()

            pocket_mask = (
                (batch.w_res > float(self.config.pocket_threshold))
                & node_mask
            ).unsqueeze(-1).expand_as(gates)
            pocket_mask_f = pocket_mask.float()
            esm_gate_pocket_mean = (
                (gates * pocket_mask_f).sum()
                / pocket_mask_f.sum().clamp(min=1.0)
            ).detach()

            nonpocket_mask = (
                (batch.w_res <= float(self.config.pocket_threshold))
                & node_mask
            ).unsqueeze(-1).expand_as(gates)
            nonpocket_mask_f = nonpocket_mask.float()
            esm_gate_nonpocket_mean = (
                (gates * nonpocket_mask_f).sum()
                / nonpocket_mask_f.sum().clamp(min=1.0)
            ).detach()

        return {
            'total': total,
            'total_no_repa': total_no_repa,
            'fm_chi': L_fm_chi,
            'fm_rigid': L_fm_rigid,
            'teacher_residual': L_teacher_residual,
            'teacher_residual_rigid': L_teacher_residual_rigid,
            'teacher_residual_chi': L_teacher_residual_chi,
            'teacher_residual_target_norm': teacher_residual_target_norm,
            'teacher_residual_pred_norm': teacher_residual_pred_norm,
            'teacher_residual_t_error': teacher_residual_t_error,
            'teacher_residual_weight_mean': teacher_residual_weight_mean,
            'teacher_residual_mask_frac': teacher_residual_mask_frac,
            'bg': L_bg,
            'smooth': L_smooth,
            'clash': L_clash,
            'ligand_clearance': L_ligand_clearance,
            'ligand_clearance_active_frac': ligand_clearance_active_frac,
            'ligand_clearance_min_dist': ligand_clearance_min_dist,
            'bridge_anchor': L_bridge_anchor,
            'bridge_anchor_mask_frac': bridge_anchor_mask_frac,
            'bridge_anchor_residual_norm': bridge_anchor_residual_norm,
            'pep': L_pep,
            'contact': L_contact,
            'stage1v2_guidance': L_stage1v2_guidance,
            'repa': L_repa,
            'prior': L_prior,
            'interaction_prior': L_interaction_prior,
            'interaction_prior_final': L_interaction_prior_final,
            'contact_score_apo': contact_score_apo,
            'contact_score_final': contact_score_final,
            'contact_score_holo': contact_score_holo,
            'contact_score_gain': contact_score_gain,
            'contact_score_holo_delta': contact_score_holo_delta,
            'contact_score_direction_acc': contact_score_direction_acc,
            'contact_score_holo_gap_abs': contact_score_holo_gap_abs,
            'contact_score_sidechain_apo': contact_score_sidechain_apo,
            'contact_score_sidechain_final': contact_score_sidechain_final,
            'contact_score_sidechain_holo': contact_score_sidechain_holo,
            'contact_score_sidechain_gain': contact_score_sidechain_gain,
            'contact_score_sidechain_holo_gap_abs': contact_score_sidechain_holo_gap_abs,
            'contact_score_sidechain_formed_recall': contact_score_sidechain_formed_recall,
            'end_rigid': L_end_rigid,
            'end_chi': L_end_chi,
            'end_fape': L_end_fape,
            'end_rigid_uw': L_end_rigid_uw,
            'end_chi_uw': L_end_chi_uw,
            'end': L_end,
            'esm_entropy': L_esm_entropy,
            'esm_layer_weight_entropy_raw': esm_layer_weight_entropy_raw,
            'esm_layer_weight_max': esm_layer_weight_max,
            'esm_gate_mean': esm_gate_mean,
            'esm_gate_pocket_mean': esm_gate_pocket_mean,
            'esm_gate_nonpocket_mean': esm_gate_nonpocket_mean,
            **esm_layer_weight_logs,
        }

    def train_step(self, batch, *, accum_steps: int, should_step: bool) -> Dict[str, float]:
        self.model.train()

        batch = self._batch_to_device(batch)
        t = torch.rand(batch.esm.shape[0], device=self.device)

        if self.autocast_dtype is not None:
            losses = self.compute_losses(batch, t)
            loss = losses['total']
            self._check_finite_losses(losses, f"train step {self.global_step}")

            loss = loss / accum_steps
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if should_step:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                self._check_finite_gradients()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if self.optimizer_step_count >= self.config.warmup_steps:
                    self.scheduler.step()
                self.optimizer_step_count += 1
        else:
            losses = self.compute_losses(batch, t)
            loss = losses['total']
            self._check_finite_losses(losses, f"train step {self.global_step}")
            loss = loss / accum_steps
            loss.backward()
            if should_step:
                self._check_finite_gradients()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.optimizer_step_count >= self.config.warmup_steps:
                    self.scheduler.step()
                self.optimizer_step_count += 1

        self.global_step += 1

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    def _batch_to_device(self, batch):
        non_blocking = self.device.type == 'cuda'
        batch.esm = batch.esm.to(self.device, non_blocking=non_blocking)
        batch.torsion_apo = batch.torsion_apo.to(self.device, non_blocking=non_blocking)
        batch.torsion_holo = batch.torsion_holo.to(self.device, non_blocking=non_blocking)
        batch.bb_mask = batch.bb_mask.to(self.device, non_blocking=non_blocking)
        batch.chi_mask = batch.chi_mask.to(self.device, non_blocking=non_blocking)
        batch.node_mask = batch.node_mask.to(self.device, non_blocking=non_blocking)
        batch.N_apo = batch.N_apo.to(self.device, non_blocking=non_blocking)
        batch.Ca_apo = batch.Ca_apo.to(self.device, non_blocking=non_blocking)
        batch.C_apo = batch.C_apo.to(self.device, non_blocking=non_blocking)
        batch.N_holo = batch.N_holo.to(self.device, non_blocking=non_blocking)
        batch.Ca_holo = batch.Ca_holo.to(self.device, non_blocking=non_blocking)
        batch.C_holo = batch.C_holo.to(self.device, non_blocking=non_blocking)
        batch.lig_points = batch.lig_points.to(self.device, non_blocking=non_blocking)
        batch.lig_types = batch.lig_types.to(self.device, non_blocking=non_blocking)
        batch.lig_mask = batch.lig_mask.to(self.device, non_blocking=non_blocking)
        batch.w_res = batch.w_res.to(self.device, non_blocking=non_blocking)
        if batch.stage1v2_posterior_features is not None:
            batch.stage1v2_posterior_features = batch.stage1v2_posterior_features.to(
                self.device,
                non_blocking=non_blocking,
            )
        if self.config.use_nma and batch.nma_features is None:
            raise ValueError("use_nma=True but batch.nma_features is None")
        if batch.nma_features is not None:
            batch.nma_features = batch.nma_features.to(self.device, non_blocking=non_blocking)
        batch.aatype = batch.aatype.to(self.device, non_blocking=non_blocking)
        return batch

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        epoch_losses = {key: 0.0 for key in self._LOSS_KEYS}

        # Set epoch for regular DistributedSampler or length-aware batch sampler.
        if self.distributed:
            for sampler in (
                getattr(self.train_loader, 'sampler', None),
                getattr(self.train_loader, 'batch_sampler', None),
            ):
                if hasattr(sampler, 'set_epoch'):
                    sampler.set_epoch(self.current_epoch)

        use_tqdm = self.is_main_process and sys.stderr.isatty()
        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch:3d}',
            ncols=120,
            leave=True,
            disable=not use_tqdm,
        )
        n_batches = len(self.train_loader)
        prev_step_end = time.perf_counter()
        for batch_idx, batch in enumerate(pbar):
            batch_ready_time = time.perf_counter()
            window_start = (batch_idx // self.grad_accum_steps) * self.grad_accum_steps
            window_end = min(window_start + self.grad_accum_steps, n_batches)
            accum_steps = window_end - window_start
            should_step = (batch_idx + 1 == window_end)

            step_losses = self.train_step(batch, accum_steps=accum_steps, should_step=should_step)
            step_end_time = time.perf_counter()
            for k in epoch_losses:
                epoch_losses[k] += step_losses[k]

            if use_tqdm:
                pbar.set_postfix({
                    'loss': f"{step_losses['total']:.3f}",
                    'fm': f"{step_losses['fm_chi'] + step_losses['fm_rigid']:.3f}",
                })
            elif (
                self.is_main_process
                and self.config.progress_log_every > 0
                and (
                    batch_idx == 0
                    or batch_idx + 1 == n_batches
                    or (batch_idx + 1) % self.config.progress_log_every == 0
                )
            ):
                fm = step_losses['fm_chi'] + step_losses['fm_rigid']
                data_wait = batch_ready_time - prev_step_end
                step_time = step_end_time - batch_ready_time
                print(
                    f"[Train] epoch={self.current_epoch} "
                    f"batch={batch_idx + 1}/{n_batches} "
                    f"loss={step_losses['total']:.4f} fm={fm:.4f} "
                    f"data_wait={data_wait:.3f}s step={step_time:.3f}s",
                    flush=True,
                )
            prev_step_end = step_end_time

        epoch_losses, n_batches = self._all_reduce_loss_sums(epoch_losses, n_batches)
        denom = max(n_batches, 1)
        return {k: v / denom for k, v in epoch_losses.items()}

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        val_losses = {key: 0.0 for key in self._LOSS_KEYS}
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation', ncols=120, leave=False):
                batch = self._batch_to_device(batch)
                if self.config.val_t is None:
                    t = torch.rand(batch.esm.shape[0], device=self.device)
                else:
                    t = torch.full((batch.esm.shape[0],), float(self.config.val_t), device=self.device)
                losses = self.compute_losses(batch, t, force_geom=True)
                self._check_finite_losses(losses, "validation")
                for k in val_losses:
                    val_losses[k] += losses[k].item()
                n_batches += 1

        val_losses, n_batches = self._all_reduce_loss_sums(val_losses, n_batches)
        denom = max(n_batches, 1)
        return {k: v / denom for k, v in val_losses.items()}

    def save_checkpoint(self, filepath: str, verbose: bool = True):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'optimizer_step_count': self.optimizer_step_count,
            'patience_counter': self.patience_counter,
            'model_state_dict': self._raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config,
        }
        if self.scaler is not None:
            payload['scaler_state_dict'] = self.scaler.state_dict()
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        if verbose:
            print(f"  ✓ Saved: {path.name}")

    @staticmethod
    def _config_value(config_obj, name: str):
        if config_obj is None:
            return None
        if isinstance(config_obj, dict):
            return config_obj.get(name)
        return getattr(config_obj, name, None)

    def _validate_resume_config(self, ckpt_config) -> None:
        if ckpt_config is None:
            if self.is_main_process:
                print("[Resume] WARNING: checkpoint has no config; skipping config compatibility checks")
            return

        strict_fields = (
            # Dataset / feature-cache contract.
            'data_dir',
            'valid_samples_file',
            'val_samples_file',
            'trust_prechecked_samples',
            'batch_size',
            'val_batch_size',
            'seed',
            'stage1v2_train_cache_dir',
            'stage1v2_val_cache_dir',
            'stage1v2_train_label_dir',
            'stage1v2_val_label_dir',
            'stage1v2_posterior_feature_scale',
            # Model architecture / input feature dimensions.
            'esm_fusion_enabled',
            'esm_num_layers',
            'esm_fusion_mode',
            'esm_layer_dropout',
            'esm_layer_entropy_weight',
            'use_nma',
            'nma_dim',
            'stage1_prior_mode',
            'use_stage1_prior',
            'use_stage1_rigid_prior',
            'stage1_chi_feature_scale',
            'stage1v2_posterior_feature_mode',
            'stage1v2_posterior_feature_names',
            'interaction_prior_feature_mode',
            'interaction_prior_feature_scale',
            'repa_enabled',
            'repa_dim',
            'repa_loss_type',
            'repa_mask_mode',
            'repa_target_mode',
            'repa_target_shuffle_mode',
            # Optimizer state is restored, so these CLI values should not drift silently.
            'lr',
            'weight_decay',
            'warmup_steps',
            'grad_accum_steps',
            'path_parameterization',
            'boundary_residual_envelope',
            'boundary_residual_scale',
            'terminal_projection_schedule',
            'length_bucket_residue_budget',
            # Loss contract.
            'contact_loss_mode',
            'w_fm_chi',
            'w_fm_rigid',
            'w_bg',
            'w_smooth',
            'w_clash',
            'w_pep',
            'w_contact',
            'w_prior',
            'w_interaction_prior',
            'w_stage1v2_guidance',
            'w_end',
        )
        mismatches = []
        backward_defaults = {
            'path_parameterization': 'flow',
            'boundary_residual_envelope': 'sin2',
            'boundary_residual_scale': 1.0,
            'terminal_projection_schedule': 'smootherstep',
        }
        for field in strict_fields:
            old = self._config_value(ckpt_config, field)
            if old is None and field in backward_defaults:
                old = backward_defaults[field]
            new = getattr(self.config, field, None)
            if old != new:
                mismatches.append((field, old, new))

        if mismatches:
            detail = ", ".join(
                f"{field}: checkpoint={old!r} current={new!r}"
                for field, old, new in mismatches
            )
            raise ValueError(
                "Refusing to resume Stage-2 checkpoint with incompatible config. "
                f"{detail}. Use a fresh save_dir or disable auto-resume for a new run."
            )

    def _maybe_resume(self) -> None:
        explicit = self.config.resume_from
        auto_path = Path(self.config.save_dir) / 'last_checkpoint.pt'

        if explicit:
            ckpt_path = Path(explicit)
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"resume_from='{explicit}' not found"
                )
        elif self.config.auto_resume and auto_path.is_file():
            ckpt_path = auto_path
        else:
            return

        if self.is_main_process:
            print(f"[Resume] Loading checkpoint: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        self._validate_resume_config(ckpt.get('config') if isinstance(ckpt, dict) else None)

        load_result = self._raw_model.load_state_dict(
            ckpt['model_state_dict'], strict=False
        )
        unexpected = list(load_result.unexpected_keys)
        missing = list(load_result.missing_keys)
        if unexpected or missing:
            raise RuntimeError(
                f"Stage-2 checkpoint state_dict mismatch "
                f"unexpected={unexpected}, missing={missing}"
            )

        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if getattr(self.scheduler, 'T_max', None) != self.scheduler_t_max:
            old_t_max = getattr(self.scheduler, 'T_max', None)
            self.scheduler.T_max = self.scheduler_t_max
            if self.is_main_process:
                print(
                    f"[Resume] Adjusted scheduler T_max from {old_t_max} to "
                    f"{self.scheduler_t_max} for current max_epochs={self.config.max_epochs}"
                )

        if self.scaler is not None and 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])

        self.current_epoch = int(ckpt.get('epoch', 0)) + 1
        self.global_step = int(ckpt.get('global_step', 0))
        self.optimizer_step_count = int(ckpt.get('optimizer_step_count', self.global_step))
        self.patience_counter = int(ckpt.get('patience_counter', 0))
        self.best_val_metric = float(ckpt.get('best_val_metric', float('inf')))

        if self.is_main_process:
            print(
                f"[Resume] OK  next_epoch={self.current_epoch}  "
                f"global_step={self.global_step}  best_val={self.best_val_metric:.4f}"
            )

        if self.distributed:
            dist.barrier()

    def train(self):
        if self.is_main_process:
            print(f"\n{'='*80}")
            print("Start training - Stage-2")
            print(f"{'='*80}\n")

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch

            train_losses = self.train_epoch()
            if self.is_main_process:
                train_info = (
                    f"Epoch {epoch:3d} | Loss: {train_losses['total']:.4f} "
                    f"FM:{train_losses['fm_chi'] + train_losses['fm_rigid']:.2f} "
                    f"Smooth:{train_losses['smooth']:.2f} "
                    f"IPrior:{train_losses['interaction_prior']:.2f} "
                    f"REPA:{train_losses['repa']:.3f}"
                )
                print(train_info)

            if epoch % 1 == 0:
                val_results = self.validate()
                should_stop = False
                if self.is_main_process:
                    val_info = (
                        f" | Val Loss: {val_results['total']:.4f} "
                        f"FM:{val_results['fm_chi'] + val_results['fm_rigid']:.2f} "
                        f"Prior:{val_results['prior']:.2f} "
                        f"IPrior:{val_results['interaction_prior']:.2f} "
                        f"REPA:{val_results['repa']:.3f} "
                        f"End:{val_results['end']:.2f} "
                        f"Contact:{val_results['contact']:.3f}"
                    )
                    print(val_info)
                    metrics_path = Path(self.config.log_dir) / 'metrics.jsonl'
                    record = {
                        'epoch': epoch,
                        **{f'train_{k}': v for k, v in train_losses.items()},
                        **{f'val_{k}': v for k, v in val_results.items()},
                    }
                    with metrics_path.open('a', encoding='utf-8') as f:
                        f.write(json.dumps(record) + '\n')

                    current_metric = val_results.get('total_no_repa', val_results['total'])
                    if current_metric < self.best_val_metric:
                        self.best_val_metric = current_metric
                        self.patience_counter = 0
                        save_path = Path(self.config.save_dir) / 'best_model.pt'
                        self.save_checkpoint(str(save_path), verbose=False)
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.config.early_stop_patience:
                            print("Early stopping triggered")
                            should_stop = True

                    last_path = Path(self.config.save_dir) / 'last_checkpoint.pt'
                    self.save_checkpoint(str(last_path), verbose=False)

                if self.distributed:
                    stop_tensor = torch.tensor(
                        [1 if should_stop else 0],
                        device=self.device,
                        dtype=torch.int32,
                    )
                    dist.broadcast(stop_tensor, src=0)
                    should_stop = bool(stop_tensor.item())

                if should_stop:
                    break

        if self.is_main_process:
            final_path = Path(self.config.save_dir) / 'final_model.pt'
            self.save_checkpoint(str(final_path))

        if self.distributed:
            dist.destroy_process_group()
