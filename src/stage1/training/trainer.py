"""
Stage-1 trainer (apo + ligand -> holo-like prior).

支持单卡和多卡 DDP 训练。
"""

import math
import time
import os
import sys
import json
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# FlashIPA Rigid (与 FK 模块一致).  This must be on sys.path before importing
# Stage1Model through src.stage1.models; the package initializer intentionally
# skips Stage1Model exports if flash_ipa cannot be imported.
project_root = Path(__file__).resolve().parent.parent.parent.parent
flash_ipa_path = str(project_root / 'vendor' / 'flash_ipa' / 'src')
if os.path.exists(flash_ipa_path) and flash_ipa_path not in sys.path:
    sys.path.insert(0, flash_ipa_path)

from .config import TrainingConfig
from ..models.stage1_model import Stage1Model, Stage1ModelConfig
from ..models.fk_openfold import reorder_torsions_to_openfold
from ..datasets import (
    create_stage1_dataloader,
    ApoHoloTripletDataset,
    collate_stage1_batch,
    DistributedLengthBatchSampler,
)
from ..datasets.dataset_stage1 import compute_pocket_weights
from ..modules.losses import (
    binary_contact_loss,
    candidate_decoy_rerank_loss,
    chi1_rotamer_loss,
    clash_penalty,
    fape_loss,
    g_antiharm_loss,
    g_decoy_contrastive_loss,
    g_lift_switch_loss,
    g_noharm_loss,
    g_switch_amp_loss,
    g_switch_dir_loss,
    g_switch_rank_loss,
    g_zero_noncontact_loss,
    ligand_contrastive_chi1_loss,
    ligand_residual_chi1_loss,
    rescue_noharm_loss,
    switch_bce_loss,
    typed_candidate_energy_loss,
    torsion_sincos_loss,
)
from utils.metrics import (
    compute_angle_errors_deg,
    compute_chi12_accuracy,
    compute_clash_percentage,
    compute_pocket_clash_percentage,
    compute_pocket_irmsd,
    compute_residue_contact_mask,
)

from flash_ipa.rigid import Rigid, Rotation
from functools import partial
from dataclasses import replace


# ---------------------------------------------------------------------
# Sequence -> aatype mapping
# ---------------------------------------------------------------------
AA_RESTYPE_MAP = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
}


def sequences_to_aatype(sequences, max_len: int, device: torch.device) -> torch.Tensor:
    """Convert list of sequences to [B, N] aatype tensor."""
    B = len(sequences)
    aatype = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            if j >= max_len:
                break
            aatype[i, j] = AA_RESTYPE_MAP.get(aa, 0)  # default to ALA for unknown
    return aatype


class Stage1Trainer:
    """Stage-1 trainer with DDP support."""

    _METRIC_DIRECTIONS = {
        'total': 'min',
        'chi': 'min',
        'fape': 'min',
        'clash': 'min',
        'candidate_rerank': 'min',
        'typed_candidate_energy': 'min',
        'chi1_acc': 'max',
        'pocket_chi1_acc': 'max',
        'chi1_rotamer_acc': 'max',
        'ligand_lift_switch_rotamer_acc': 'max',
        'ligand_lift_apo_wrong_rotamer_acc': 'max',
        'ligand_lift_contact_rotamer_acc': 'max',
        'ligand_lift_contact_switch_rotamer_acc': 'max',
        'ligand_lift_pocket_switch_rotamer_acc': 'max',
        'ligand_decoy_lift_contact_switch_rotamer_acc': 'max',
        'typed_energy_gap_switch': 'max',
        'typed_energy_gap_contact': 'max',
        'typed_energy_gap_contact_switch': 'max',
        'typed_energy_gap_pocket_switch': 'max',
        'candidate_decoy_lift_switch_rotamer_acc': 'max',
        'candidate_decoy_lift_apo_wrong_rotamer_acc': 'max',
        'candidate_decoy_lift_contact_rotamer_acc': 'max',
        'candidate_decoy_lift_contact_switch_rotamer_acc': 'max',
        'candidate_decoy_lift_pocket_switch_rotamer_acc': 'max',
        'contact_posterior_f1': 'max',
        'pocket_irmsd': 'min',
        'clash_pct': 'min',
    }

    def _resolve_amp_dtype(self) -> Optional[torch.dtype]:
        if not self.config.mixed_precision or self.device.type != 'cuda':
            return None

        amp_dtype = getattr(self.config, 'amp_dtype', 'auto').lower()
        bf16_supported = bool(getattr(torch.cuda, 'is_bf16_supported', lambda: False)())

        if amp_dtype == 'auto':
            return torch.bfloat16 if bf16_supported else torch.float16
        if amp_dtype == 'bf16':
            if not bf16_supported:
                raise ValueError("amp_dtype=bf16 requested but this CUDA device does not support bf16")
            return torch.bfloat16
        if amp_dtype == 'fp16':
            return torch.float16

        raise ValueError(f"Unsupported amp_dtype: {self.config.amp_dtype}")

    def _autocast_context(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return autocast(dtype=self.autocast_dtype)

    def _warmup_lr_scale(self, step: int) -> float:
        if self.config.warmup_steps <= 0:
            return 1.0
        if step >= self.config.warmup_steps:
            return 1.0
        return max(step + 1, 1) / max(self.config.warmup_steps, 1)

    def _apply_warmup_lr(self):
        if self.config.warmup_steps <= 0:
            return
        if self.global_step < self.config.warmup_steps:
            scale = self._warmup_lr_scale(self.global_step)
            for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
                param_group['lr'] = base_lr * scale
        elif self.global_step == self.config.warmup_steps:
            for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
                param_group['lr'] = base_lr

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]['lr'])

    def _metric_direction(self) -> str:
        metric = getattr(self.config, 'selection_metric', 'total')
        if metric not in self._METRIC_DIRECTIONS:
            allowed = ', '.join(sorted(self._METRIC_DIRECTIONS))
            raise ValueError(f"Unsupported selection_metric: {metric}. Allowed: {allowed}")
        return self._METRIC_DIRECTIONS[metric]

    def _initial_best_metric(self) -> float:
        return float('inf') if self._metric_direction() == 'min' else float('-inf')

    def _extract_selection_metric(self, val_results: Dict[str, float]) -> float:
        metric_name = getattr(self.config, 'selection_metric', 'total')
        if metric_name not in val_results:
            available = ', '.join(sorted(val_results.keys()))
            raise KeyError(f"selection_metric '{metric_name}' missing from val results. Available: {available}")
        value = float(val_results[metric_name])
        if not math.isfinite(value):
            return float('inf') if self._metric_direction() == 'min' else float('-inf')
        return value

    def _is_better_metric(self, candidate: float, incumbent: float) -> bool:
        direction = self._metric_direction()
        if direction == 'min':
            return candidate < incumbent
        return candidate > incumbent

    def _metrics_log_path(self) -> Path:
        return Path(self.config.log_dir) / self.config.metrics_filename

    def _audit_log_path(self) -> Path:
        return Path(self.config.log_dir) / self.config.audit_filename

    def _to_serializable(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.item())
            return value.detach().cpu().tolist()
        if isinstance(value, Path):
            return str(value)
        return value

    def _append_metrics_record(self, record: Dict):
        if not self.is_main_process:
            return
        path = self._metrics_log_path()
        serializable = {k: self._to_serializable(v) for k, v in record.items()}
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(serializable, ensure_ascii=False) + '\n')

    def _write_audit_snapshot(self, payload: Dict):
        if not self.is_main_process:
            return
        path = self._audit_log_path()
        serializable = {k: self._to_serializable(v) for k, v in payload.items()}
        with path.open('w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _nanmean_from_sum_count(total: float, count: int) -> float:
        return float(total / count) if count > 0 else float('nan')

    @staticmethod
    def _chi1_rotamer_labels(chi1_angles: torch.Tensor) -> torch.Tensor:
        angle = ((chi1_angles + math.pi) % (2 * math.pi)) - math.pi
        centers = torch.tensor([-math.pi / 3, math.pi / 3, math.pi], device=angle.device, dtype=angle.dtype)
        diff = angle.unsqueeze(-1) - centers
        circ_dist = ((diff + math.pi) % (2 * math.pi) - math.pi).abs()
        return circ_dist.argmin(dim=-1)  # 0=g-, 1=g+, 2=t

    @staticmethod
    def _valid_ligand_coords(lig_points: torch.Tensor, lig_mask: torch.Tensor) -> np.ndarray:
        valid_mask = lig_mask.bool()
        if not valid_mask.any():
            return np.zeros((0, 3), dtype=np.float32)
        return lig_points[valid_mask].detach().cpu().numpy().astype(np.float32)

    def _compute_holo_distance_mask(self,
                                    ca_holo: torch.Tensor,
                                    lig_points: torch.Tensor,
                                    lig_mask: torch.Tensor,
                                    node_mask: torch.Tensor) -> torch.Tensor:
        lig_coords = self._valid_ligand_coords(lig_points, lig_mask)
        if lig_coords.shape[0] == 0:
            return torch.zeros_like(node_mask, dtype=torch.bool)
        w_res_holo = compute_pocket_weights(
            ca_holo.detach().cpu().numpy().astype(np.float32),
            lig_coords,
        )
        w_res_holo = torch.from_numpy(w_res_holo).to(device=node_mask.device)
        return (w_res_holo > 0.5) & node_mask.bool()

    def _compute_ligand_contact_mask(self,
                                     atom14_holo: torch.Tensor,
                                     atom14_holo_mask: torch.Tensor,
                                     lig_points: torch.Tensor,
                                     lig_mask: torch.Tensor,
                                     node_mask: torch.Tensor,
                                     contact_threshold: float = 4.5) -> torch.Tensor:
        lig_coords = self._valid_ligand_coords(lig_points, lig_mask)
        if lig_coords.shape[0] == 0:
            return torch.zeros_like(node_mask, dtype=torch.bool)

        contact_mask_np = compute_residue_contact_mask(
            atom14_holo,
            atom14_holo_mask,
            lig_coords,
            contact_threshold=contact_threshold,
        )
        contact_mask = torch.from_numpy(contact_mask_np).to(device=node_mask.device)
        return contact_mask.bool() & node_mask.bool()

    @staticmethod
    def _make_nolig_batch(batch):
        """Zero out ligand for no-ligand contrastive. DDP-safe."""
        return replace(
            batch,
            lig_points=torch.zeros_like(batch.lig_points),
            lig_types=torch.zeros_like(batch.lig_types),
            lig_mask=torch.zeros_like(batch.lig_mask),
        )

    @staticmethod
    def _make_batch_shuffled_ligand(batch):
        B = batch.lig_points.shape[0]
        if B < 2:
            # B=1: use same ligand as decoy (contrastive loss will be ~0 but keeps DDP in sync)
            return replace(batch, lig_points=batch.lig_points, lig_types=batch.lig_types, lig_mask=batch.lig_mask)
        shift = 1 + (B // 2)
        perm = torch.arange(B, device=batch.lig_points.device).roll(shifts=shift)
        return replace(
            batch,
            lig_points=batch.lig_points.index_select(0, perm),
            lig_types=batch.lig_types.index_select(0, perm),
            lig_mask=batch.lig_mask.index_select(0, perm),
        )

    @staticmethod
    def _make_scrambled_ligand_types(batch):
        """Keep ligand coordinates fixed but rotate valid token type rows."""
        lig_types = batch.lig_types.clone()
        for i in range(batch.lig_types.shape[0]):
            valid = torch.nonzero(batch.lig_mask[i].bool(), as_tuple=False).squeeze(-1)
            if valid.numel() > 1:
                lig_types[i, valid] = batch.lig_types[i, valid.roll(shifts=1)]
        return replace(batch, lig_types=lig_types)

    @staticmethod
    def _make_translated_batch(batch, offset: float = 100.0):
        """Translate ligand far away from the protein.  Preserves ligand identity
        (atom types, intra-ligand geometry) but removes spatial contact, so the
        residual scorer sees the same chemistry but no protein-ligand interface.

        This is the preferred decoy for ligand-causal contrastive losses
        (Phase-1 v2): a ligand-conditioned model that still produces the same
        posterior for translated ligand is *not* using ligand identity-in-pocket
        as evidence.
        """
        offset_vec = torch.tensor(
            [offset, offset, offset],
            device=batch.lig_points.device,
            dtype=batch.lig_points.dtype,
        ).view(1, 1, 3)
        return replace(
            batch,
            lig_points=batch.lig_points + offset_vec,
        )

    def _make_typed_candidate_decoy_batch(self, batch):
        kind = getattr(self.config, 'typed_candidate_decoy_kind', 'scrambled')
        if kind == 'nolig':
            return self._make_nolig_batch(batch)
        if kind == 'shuffled':
            return self._make_batch_shuffled_ligand(batch)
        if kind == 'translated':
            return self._make_translated_batch(
                batch, offset=float(getattr(self.config, 'g_decoy_translation_offset', 100.0)),
            )
        return self._make_scrambled_ligand_types(batch)


    @staticmethod
    def _audit_mask_items(enable_dual_mask_audit: bool) -> List[Tuple[str, str]]:
        items: List[Tuple[str, str]] = [('pocket', 'apo_distance_mask')]
        if enable_dual_mask_audit:
            items.extend([
                ('holo_pocket', 'holo_distance_mask'),
                ('ligand_facing', 'ligand_contact_mask'),
            ])
        return items

    def __init__(self, config: TrainingConfig):
        self.config = config
        if self.config.selection_metric in {'pocket_irmsd', 'clash_pct'} and not self.config.compute_slow_metrics:
            self.config.compute_slow_metrics = True
            print(f"[INFO] Auto-enabled compute_slow_metrics because selection_metric={self.config.selection_metric}")
        
        # ========== 分布式初始化 ==========
        self.distributed = config.distributed
        self.local_rank = 0
        self.world_size = 1
        self.is_main_process = True
        
        if self.distributed:
            # 初始化分布式进程组
            if not dist.is_initialized():
                timeout_sec = int(os.environ.get("DDP_TIMEOUT", "7200"))
                dist.init_process_group(
                    backend='nccl',
                    timeout=timedelta(seconds=timeout_sec),
                )
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.world_size = dist.get_world_size()
            self.is_main_process = (self.local_rank == 0)
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            if self.is_main_process:
                print(f"[DDP] Initialized: world_size={self.world_size}, local_rank={self.local_rank}")
        else:
            self.device = torch.device(config.device)

        # 设置随机种子（每个 rank 不同以获得不同的数据增强）
        seed = config.seed + self.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ========== 创建模型 ==========
        if self.is_main_process:
            print(f"Creating model (size={config.model_size})...")
        
        # 根据配置选择模型规模
        model_size = config.model_size.lower()
        if model_size == 'small':
            model_config = Stage1ModelConfig.small()
        elif model_size == 'stable_wide':
            model_config = Stage1ModelConfig.stable_wide()
        elif model_size == 'medium':
            model_config = Stage1ModelConfig.medium()
        elif model_size == 'large':
            model_config = Stage1ModelConfig.large()
        elif model_size == 'wide_shallow':
            model_config = Stage1ModelConfig.wide_shallow()
        elif model_size == 'enhanced_ligand':
            model_config = Stage1ModelConfig.enhanced_ligand()
        else:
            raise ValueError(f"Unknown model_size: {model_size}. Use 'small', 'stable_wide', 'enhanced_ligand', 'medium', 'large', or 'wide_shallow'")

        model_config.warmup_steps = config.ligand_gate_warmup_steps
        model_config.use_pocket_chi1_expert = config.use_pocket_chi1_expert
        model_config.pocket_chi1_expert_hidden = config.pocket_chi1_expert_hidden
        model_config.pocket_chi1_expert_layers = config.pocket_chi1_expert_layers
        model_config.pocket_chi1_gate_threshold = config.pocket_chi1_gate_threshold
        model_config.pocket_chi1_residual_scale = config.pocket_chi1_residual_scale
        model_config.use_chi1_rotamer_posterior = config.lambda_chi1_rotamer > 0.0
        model_config.chi1_rotamer_hidden = config.chi1_rotamer_hidden
        model_config.use_candidate_chi1_scorer = (
            config.lambda_candidate_chi1 > 0.0
            or config.lambda_ligand_contrastive > 0.0
            or getattr(config, 'lambda_candidate_rerank', 0.0) > 0.0
        )
        model_config.candidate_chi1_hidden = config.candidate_chi1_hidden
        # Create geometry scorer whenever any of its consuming losses is active
        # (covers Phase-1 v2: full CE may be 0 but G-vector losses use base/residual).
        model_config.use_geometry_candidate_scorer = (
            config.lambda_geometry_chi1 > 0.0
            or config.lambda_base_prior > 0.0
            or config.lambda_g_lift_switch > 0.0
            or config.lambda_g_noharm > 0.0
            or config.lambda_g_zero_noncontact > 0.0
            or getattr(config, 'lambda_g_switch_dir', 0.0) > 0.0
            or getattr(config, 'lambda_g_switch_amp', 0.0) > 0.0
            or getattr(config, 'lambda_g_switch_rank', 0.0) > 0.0
            or getattr(config, 'lambda_g_antiharm', 0.0) > 0.0
            or getattr(config, 'lambda_g_decoy', 0.0) > 0.0
            or getattr(config, 'lambda_typed_candidate_energy', 0.0) > 0.0
        )
        model_config.geometry_scorer_hidden = getattr(config, 'geometry_scorer_hidden', 64)
        model_config.geometry_scorer_num_rbf = getattr(config, 'geometry_scorer_num_rbf', 16)
        model_config.geometry_scorer_use_sgeo = getattr(config, 'geometry_scorer_use_sgeo', False)
        model_config.geometry_scorer_sgeo_dim = getattr(config, 'geometry_scorer_sgeo_dim', 32)
        model_config.geometry_scorer_bounded_residual = getattr(config, 'geometry_scorer_bounded_residual', False)
        model_config.geometry_scorer_residual_max = getattr(config, 'geometry_scorer_residual_max', 5.0)
        model_config.geometry_scorer_residual_tau = getattr(config, 'geometry_scorer_residual_tau', 2.0)
        model_config.geometry_scorer_gate_norm = getattr(config, 'geometry_scorer_gate_norm', False)
        model_config.geometry_scorer_gate_clamp = getattr(config, 'geometry_scorer_gate_clamp', 6.0)
        model_config.geometry_scorer_gate_init_bias = getattr(config, 'geometry_scorer_gate_init_bias', 0.0)
        model_config.geometry_scorer_use_typed_energy = getattr(config, 'geometry_scorer_use_typed_energy', False)
        model_config.geometry_scorer_typed_pair_dim = getattr(config, 'geometry_scorer_typed_pair_dim', 64)
        model_config.geometry_scorer_typed_cutoff = getattr(config, 'geometry_scorer_typed_cutoff', 6.0)
        model_config.geometry_scorer_typed_init_scale = getattr(config, 'geometry_scorer_typed_init_scale', 0.1)
        model_config.use_contact_posterior = config.lambda_contact > 0.0
        model_config.contact_hidden = config.contact_hidden
        model_config.use_pocket_routing_adapter = config.use_pocket_routing_adapter
        model_config.pocket_routing_hidden = config.pocket_routing_hidden
        model_config.pocket_routing_layers = config.pocket_routing_layers
        model_config.pocket_routing_gate_threshold = config.pocket_routing_gate_threshold
        model_config.pocket_routing_residual_scale = config.pocket_routing_residual_scale
        
        self.model = Stage1Model(model_config).to(self.device)

        if config.freeze_stage1_backbone_for_posteriors:
            self._freeze_stage1_backbone_for_posteriors()

        # DDP 包装模型
        if self.distributed:
            ddp_find_unused = bool(getattr(config, 'ddp_find_unused_parameters', False))
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=ddp_find_unused,
            )
            if self.is_main_process:
                print(f"[DDP] Model wrapped with DistributedDataParallel (find_unused_parameters={ddp_find_unused})")

        # ========== 创建数据加载器 ==========
        if self.is_main_process:
            print("Creating dataloaders...")
        
        self.train_sampler: Optional[object] = None
        self.val_sampler: Optional[DistributedSampler] = None
        
        if self.distributed:
            # 分布式模式：手动创建 dataset 和 sampler
            sample_metadata_file = config.sample_metadata_file or 'sample_metadata.json'
            train_dataset = ApoHoloTripletDataset(
                config.data_dir,
                split='train',
                valid_samples_file=config.valid_samples_file,
                sample_metadata_file=sample_metadata_file,
                require_atom14=False,
            )
            collate_fn = partial(collate_stage1_batch, max_n_res=config.max_n_res)
            persistent_workers = config.num_workers > 0

            if config.length_bucketed_sampling or config.residue_budget is not None:
                sample_costs = train_dataset.get_sample_residue_counts(sample_metadata_file)
                if config.max_n_res is not None:
                    keep_indices = [
                        idx for idx, cost in enumerate(sample_costs)
                        if cost <= config.max_n_res
                    ]
                    filtered_count = len(sample_costs) - len(keep_indices)
                    if filtered_count > 0 and self.is_main_process:
                        print(
                            f"  Filtered {filtered_count} training samples "
                            f"above max_n_res={config.max_n_res} before batching"
                        )
                    train_dataset.samples = [train_dataset.samples[idx] for idx in keep_indices]
                    sample_costs = [sample_costs[idx] for idx in keep_indices]

                self.train_sampler = DistributedLengthBatchSampler(
                    sample_costs,
                    batch_size=config.batch_size,
                    num_replicas=self.world_size,
                    rank=self.local_rank,
                    shuffle=True,
                    drop_last=True,
                    seed=config.seed,
                    bucket_size_multiplier=config.bucket_size_multiplier,
                    residue_budget=config.residue_budget,
                )
                self.train_loader = DataLoader(
                    train_dataset,
                    batch_sampler=self.train_sampler,
                    num_workers=config.num_workers,
                    collate_fn=collate_fn,
                    pin_memory=True,
                    persistent_workers=persistent_workers,
                )
            else:
                self.train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=self.world_size,
                    rank=self.local_rank,
                    shuffle=True,
                )
                self.train_loader = DataLoader(
                    train_dataset,
                    batch_size=config.batch_size,
                    sampler=self.train_sampler,
                    num_workers=config.num_workers,
                    collate_fn=collate_fn,
                    pin_memory=True,
                    persistent_workers=persistent_workers,
                    drop_last=True,  # DDP 需要保证每个 rank batch 数相同
                )
            
            # 验证集默认使用原始 val split；不要隐式复用训练集清单，避免数据泄漏/错分。
            val_samples = config.val_samples_file
            if (
                self.is_main_process
                and config.valid_samples_file is not None
                and config.val_samples_file is None
            ):
                print(
                    "[WARN] val_samples_file not provided; validation will use the raw "
                    "val split instead of reusing valid_samples_file."
                )
            if self.is_main_process:
                val_dataset = ApoHoloTripletDataset(
                    config.data_dir,
                    split='val',
                    valid_samples_file=val_samples,
                    sample_metadata_file=sample_metadata_file,
                    require_atom14=False,
                )
                val_num_workers = min(config.num_workers, 2)
                self.val_loader = DataLoader(
                    val_dataset,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=val_num_workers,
                    collate_fn=collate_fn,
                    pin_memory=True,
                    persistent_workers=val_num_workers > 0,
                    drop_last=False,
                )
            else:
                self.val_loader = None
        else:
            # 单卡模式：使用原有的工厂函数
            self.train_loader = create_stage1_dataloader(
                config.data_dir,
                split='train',
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                max_n_res=config.max_n_res,
                valid_samples_file=config.valid_samples_file,
                sample_metadata_file=config.sample_metadata_file,
                require_atom14=False,
            )
            val_num_workers = min(config.num_workers, 2)
            # 验证集默认使用原始 val split；不要隐式复用训练集清单，避免数据泄漏/错分。
            val_samples = config.val_samples_file
            if (
                self.is_main_process
                and config.valid_samples_file is not None
                and config.val_samples_file is None
            ):
                print(
                    "[WARN] val_samples_file not provided; validation will use the raw "
                    "val split instead of reusing valid_samples_file."
                )
            self.val_loader = create_stage1_dataloader(
                config.data_dir,
                split='val',
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=val_num_workers,
                max_n_res=config.max_n_res,
                valid_samples_file=val_samples,
                sample_metadata_file=config.sample_metadata_file,
                require_atom14=False,
            )

        # ========== 优化器 ==========
        if self.is_main_process:
            print("Creating optimizer...")

        param_groups = self._build_param_groups(config)
        if not param_groups:
            raise ValueError("No trainable model parameters. Check posterior-head and freeze settings.")
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        scheduler_name = getattr(config, 'lr_scheduler', 'plateau').lower()
        if scheduler_name not in {'plateau', 'cosine'}:
            raise ValueError(f"Unsupported lr_scheduler: {config.lr_scheduler}")
        self.lr_scheduler_name = scheduler_name

        # LR scheduler
        total_steps = len(self.train_loader) * config.max_epochs
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        if self.lr_scheduler_name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(total_steps - config.warmup_steps, 1),
                eta_min=config.lr * config.min_lr_scale,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.plateau_factor,
                patience=config.plateau_patience,
                min_lr=[lr * config.min_lr_scale for lr in self.base_lrs],
            )

        self.autocast_dtype = self._resolve_amp_dtype()
        self.scaler = GradScaler() if self.autocast_dtype == torch.float16 else None

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = self._initial_best_metric()
        self.patience_counter = 0
        self.resumed_from: Optional[Path] = None

        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)

        # Phase-1: optionally reset residual + gate (after checkpoint load) and freeze base.
        # This may rebind nn.Module submodules, so optimizer/scheduler must be rebuilt
        # to reference the fresh parameters.
        phase1_changed = self._apply_phase1_module_controls()
        if phase1_changed:
            self._rebuild_optimizer_and_scheduler()

        # 只在主进程创建目录和打印信息
        if self.is_main_process:
            Path(config.save_dir).mkdir(parents=True, exist_ok=True)
            Path(config.log_dir).mkdir(parents=True, exist_ok=True)
            metrics_path = self._metrics_log_path()
            if metrics_path.exists() and self.resumed_from is None:
                metrics_path.unlink()
            audit_path = self._audit_log_path()
            if audit_path.exists() and self.resumed_from is None:
                audit_path.unlink()

            print("✓ Trainer initialized")
            n_trainable_params = sum(p.numel() for p in self._trainable_model_parameters())
            n_total_params = sum(p.numel() for p in self._model.parameters())
            print(f"  - params: {n_trainable_params:,} trainable / {n_total_params:,} total")
            print(f"  - train samples: {len(self.train_loader.dataset)}")
            print(f"  - val samples: {len(self.val_loader.dataset)}")
            print(f"  - total steps: {total_steps:,}")
            print(f"  - lr scheduler: {self.lr_scheduler_name}")
            print(f"  - selection metric: {self.config.selection_metric} ({self._metric_direction()})")
            print(f"  - amp dtype: {str(self.autocast_dtype).replace('torch.', '') if self.autocast_dtype is not None else 'disabled'}")
            if self.distributed:
                print(f"  - world_size: {self.world_size}")
                print(f"  - effective batch: {config.batch_size * self.world_size}")
                print(f"  - length bucketed sampling: {config.length_bucketed_sampling}")
                print(f"  - residue budget: {config.residue_budget}")
                print(f"  - pocket warmup: {config.pocket_warmup_steps}")
                print(f"  - ligand gate warmup: {config.ligand_gate_warmup_steps}")
            if self.resumed_from is not None:
                print(f"  - resumed from: {self.resumed_from}")

        self._apply_warmup_lr()

    def _load_checkpoint(self, filepath: str):
        ckpt_path = Path(filepath)
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        branch_resume = ckpt_path.resolve().parent != Path(self.config.save_dir).resolve()

        load_result = self._model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=not branch_resume,
        )
        if branch_resume and self.is_main_process:
            missing = list(load_result.missing_keys)
            unexpected = list(load_result.unexpected_keys)
            if missing:
                print(f"[INFO] Branch resume missing model keys initialized from scratch: {missing}")
            if unexpected:
                print(f"[INFO] Branch resume ignored unexpected model keys: {unexpected}")

        if not branch_resume and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif branch_resume and self.is_main_process:
            print("[INFO] Branch resume uses a fresh optimizer/scheduler for architecture changes")
        if not branch_resume and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if not branch_resume:
            self.best_val_metric = float(checkpoint.get('best_val_metric', self._initial_best_metric()))
        else:
            self.best_val_metric = self._initial_best_metric()

        source_epoch = int(checkpoint.get('epoch', -1)) + 1
        source_global_step = int(checkpoint.get('global_step', 0))
        if branch_resume:
            self.current_epoch = 0
            self.global_step = 0
        else:
            self.current_epoch = source_epoch
            self.global_step = source_global_step
        self.patience_counter = 0
        self.resumed_from = ckpt_path

        if self.is_main_process:
            resume_mode = 'branch-resume' if branch_resume else 'full-resume'
            print(f"[INFO] Loaded checkpoint ({resume_mode}): {ckpt_path}")
            if branch_resume:
                print(
                    f"[INFO] Branch source epoch={source_epoch} global_step={source_global_step}; "
                    f"starting new run at epoch=0 global_step=0"
                )
            else:
                print(f"[INFO] Resume epoch={self.current_epoch} global_step={self.global_step}")

    def compute_pocket_warmup(self, step: int) -> float:
        if step >= self.config.pocket_warmup_steps:
            return 1.0
        return step / max(self.config.pocket_warmup_steps, 1)

    def compute_pchi1_lambda(self, step: int) -> float:
        target = float(getattr(self.config, 'lambda_pchi1', 0.0))
        if target <= 0.0:
            return 0.0

        start_step = max(int(getattr(self.config, 'pchi1_start_step', 0)), 0)
        ramp_steps = max(int(getattr(self.config, 'pchi1_ramp_steps', 0)), 0)

        if step < start_step:
            return 0.0
        if ramp_steps == 0:
            return target

        progress = min(max(step - start_step, 0) / ramp_steps, 1.0)
        return target * progress

    def _compute_pchi1_residue_mask(self,
                                    batch,
                                    target_atom14: torch.Tensor,
                                    target_atom14_mask: torch.Tensor) -> torch.Tensor:
        mask_mode = getattr(self.config, 'pchi1_mask_mode', 'soft')
        node_mask = batch.node_mask.bool()

        if mask_mode == 'soft':
            return node_mask

        masks = []
        for i in range(batch.Ca_holo.shape[0]):
            node_mask_i = node_mask[i]
            if mask_mode == 'ligand_facing':
                residue_mask_i = self._compute_ligand_contact_mask(
                    target_atom14[i],
                    target_atom14_mask[i],
                    batch.lig_points[i],
                    batch.lig_mask[i],
                    node_mask_i,
                )
            elif mask_mode == 'holo_pocket':
                residue_mask_i = self._compute_holo_distance_mask(
                    batch.Ca_holo[i],
                    batch.lig_points[i],
                    batch.lig_mask[i],
                    node_mask_i,
                )
            else:
                raise ValueError(f"Unsupported pchi1_mask_mode: {mask_mode}")

            masks.append(residue_mask_i)

        return torch.stack(masks, dim=0)

    def _compute_batch_ligand_contact_mask(self,
                                           batch,
                                           target_atom14: torch.Tensor,
                                           target_atom14_mask: torch.Tensor) -> torch.Tensor:
        masks = []
        node_mask = batch.node_mask.bool()
        for i in range(target_atom14.shape[0]):
            masks.append(self._compute_ligand_contact_mask(
                target_atom14[i],
                target_atom14_mask[i],
                batch.lig_points[i],
                batch.lig_mask[i],
                node_mask[i],
            ))
        return torch.stack(masks, dim=0)

    def _compute_ca_ligand_non_contact_mask(self, batch, threshold: float = 8.0) -> torch.Tensor:
        """[B, N] bool, True where CA-ligand min-distance > threshold (Å).

        Cheap proxy for "the ligand is far enough away that it should not influence
        this residue's rotamer posterior" (used by g_zero_noncontact_loss).
        """
        ca = batch.Ca_apo  # [B, N, 3]
        lig = batch.lig_points  # [B, L, 3]
        lig_mask = batch.lig_mask.bool()
        diff = ca.unsqueeze(2) - lig.unsqueeze(1)  # [B, N, L, 3]
        dist = torch.linalg.norm(diff, dim=-1)
        dist = dist.masked_fill(~lig_mask.unsqueeze(1), float('inf'))
        min_dist = dist.min(dim=-1).values  # [B, N]
        return (min_dist > float(threshold)) & batch.node_mask.bool()

    def _compute_ca_ligand_contact_mask(self, batch, threshold: float = 8.0) -> torch.Tensor:
        """[B, N] bool, True where CA-ligand min-distance <= threshold (Å).

        Inverse of the non-contact mask, used to restrict v2 switch losses
        (direction / amplitude / rank / decoy) to ligand-facing residues.
        """
        ca = batch.Ca_apo
        lig = batch.lig_points
        lig_mask = batch.lig_mask.bool()
        diff = ca.unsqueeze(2) - lig.unsqueeze(1)
        dist = torch.linalg.norm(diff, dim=-1)
        dist = dist.masked_fill(~lig_mask.unsqueeze(1), float('inf'))
        min_dist = dist.min(dim=-1).values
        return (min_dist <= float(threshold)) & batch.node_mask.bool()

    @staticmethod
    def _binary_contact_pos_weight(contact_mask: torch.Tensor,
                                   residue_mask: torch.Tensor,
                                   configured_weight: float,
                                   eps: float = 1e-8) -> torch.Tensor:
        if configured_weight > 0.0:
            return contact_mask.float().new_tensor(configured_weight)

        valid = residue_mask.bool()
        positives = (contact_mask.bool() & valid).float().sum()
        negatives = ((~contact_mask.bool()) & valid).float().sum()
        if positives <= 0:
            return contact_mask.float().new_tensor(1.0)
        return negatives / (positives + eps)

    def _build_frames_from_backbone(self, N, Ca, C, mask, eps: float = 1e-6):
        """Build frames from backbone with NaN protection."""
        device = Ca.device
        default_e1 = torch.tensor([1.0, 0.0, 0.0], device=device)
        default_e2 = torch.tensor([0.0, 1.0, 0.0], device=device)
        default_e3 = torch.tensor([0.0, 0.0, 1.0], device=device)
        
        # e1: CA -> C
        e1 = C - Ca
        e1_norm = torch.norm(e1, dim=-1, keepdim=True)
        e1_valid = e1_norm > eps
        # 只对有效向量归一化，无效的用默认单位向量
        e1_normalized = e1 / torch.clamp(e1_norm, min=eps)
        e1 = torch.where(e1_valid, e1_normalized, default_e1.expand_as(e1))

        # u: CA -> N
        u = N - Ca
        proj = (u * e1).sum(dim=-1, keepdim=True) * e1
        e2 = u - proj
        e2_norm = torch.norm(e2, dim=-1, keepdim=True)
        e2_valid = e2_norm > eps
        e2_normalized = e2 / torch.clamp(e2_norm, min=eps)
        e2 = torch.where(e2_valid, e2_normalized, default_e2.expand_as(e2))

        # e3: cross product
        e3 = torch.cross(e1, e2, dim=-1)
        e3_norm = torch.norm(e3, dim=-1, keepdim=True)
        e3_valid = e3_norm > eps
        e3_normalized = e3 / torch.clamp(e3_norm, min=eps)
        e3 = torch.where(e3_valid, e3_normalized, default_e3.expand_as(e3))

        R = torch.stack([e1, e2, e3], dim=-1)
        t = Ca

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
            eye = torch.eye(3, device=device).view(1, 1, 3, 3)
            R = torch.where(mask_expanded, R, eye)
            t = torch.where(mask.unsqueeze(-1), t, torch.zeros_like(t))

        # Replace any remaining NaN
        R = torch.where(torch.isnan(R), torch.eye(3, device=device).view(1, 1, 3, 3).expand_as(R), R)
        t = torch.where(torch.isnan(t), torch.zeros_like(t), t)

        return R, t

    def _build_rigids_from_backbone(self, N, Ca, C, mask) -> Rigid:
        """Build Rigid object from backbone coordinates for FK."""
        R, t = self._build_frames_from_backbone(N, Ca, C, mask)
        rotation = Rotation(rot_mats=R)
        return Rigid(rots=rotation, trans=t)

    @property
    def _model(self):
        """获取实际模型（DDP 模式下返回 .module）"""
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _freeze_stage1_backbone_for_posteriors(self):
        """Freeze the deterministic Stage-1 trunk and train selected posterior modules."""
        trainable_prefixes = ['chi1_rotamer_head.', 'contact_head.', 'candidate_chi1_scorer.', 'geometry_candidate_scorer.']
        if getattr(self.config, 'unfreeze_ligand_conditioner_for_posteriors', False):
            trainable_prefixes.append('ligand_conditioner.')

        n_last_ipa = int(getattr(self.config, 'unfreeze_last_ipa_blocks_for_posteriors', 0) or 0)
        if n_last_ipa < 0:
            raise ValueError("unfreeze_last_ipa_blocks_for_posteriors must be >= 0")
        if n_last_ipa > 0:
            ipa_blocks = getattr(getattr(self._model, 'ipa_module', None), 'ipa_blocks', None)
            if ipa_blocks is None:
                raise ValueError("Cannot unfreeze final IPA blocks: model has no ipa_module.ipa_blocks")
            n_blocks = len(ipa_blocks)
            start = max(0, n_blocks - n_last_ipa)
            trainable_prefixes.extend(f'ipa_module.ipa_blocks.{idx}.' for idx in range(start, n_blocks))
            trainable_prefixes.append('ipa_module.final_norm.')

        trainable_prefixes_tuple = tuple(trainable_prefixes)
        for name, param in self._model.named_parameters():
            param.requires_grad = name.startswith(trainable_prefixes_tuple)

        trainable = [name for name, param in self._model.named_parameters() if param.requires_grad]
        if self.is_main_process:
            print("[INFO] Frozen Stage-1 trunk; trainable posterior parameter prefixes:")
            for prefix in trainable_prefixes_tuple:
                print(f"  * {prefix}")
            print("[INFO] Trainable posterior parameters:")
            for name in trainable:
                print(f"  - {name}")

    def _trainable_model_parameters(self):
        return [param for param in self._model.parameters() if param.requires_grad]

    def _apply_phase1_module_controls(self) -> bool:
        """Apply Phase-1 module-level controls after model creation / checkpoint load.

        Operates on the underlying nn.Module (not the DDP wrapper) so the DDP
        wrapper must be rebuilt afterwards if module submodules were rebound.
        Returns True if any change was applied (caller must rebuild
        DDP+optimizer+scheduler).
        """
        scorer = getattr(self._model, 'geometry_candidate_scorer', None)
        if scorer is None:
            return False
        changed = False
        if getattr(self.config, 'reset_residual_and_gate_on_resume', False):
            scorer.reset_residual_and_gate()
            changed = True
            if self.is_main_process:
                print("[INFO] Phase-1: residual_mlp and gate_mlp re-initialized from scratch")
        if getattr(self.config, 'freeze_base_mlp', False):
            scorer.freeze_base(True)
            changed = True
            if self.is_main_process:
                n_frozen = sum(1 for p in scorer.base_mlp.parameters() if not p.requires_grad)
                print(f"[INFO] Phase-1: geometry_candidate_scorer.base_mlp frozen ({n_frozen} param tensors)")
        if getattr(self.config, 'freeze_gate_mlp', False):
            scorer.freeze_gate(True)
            changed = True
            if self.is_main_process:
                n_frozen = sum(1 for p in scorer.gate_mlp.parameters() if not p.requires_grad)
                print(f"[INFO] Phase-1 v2: geometry_candidate_scorer.gate_mlp frozen ({n_frozen} param tensors); gate fixed=1.0")
        return changed

    def _rebuild_optimizer_and_scheduler(self) -> None:
        """Recreate optimizer + scheduler after Phase-1 module changes (reset/freeze).

        Phase-1 retraining starts a fresh optimizer state since the residual/gate
        parameters are new objects.  When DDP-wrapped, the DDP wrapper must also
        be rebuilt so it picks up the new parameter tensors for synchronization.
        """
        # Re-wrap DDP if needed: unwrap to the bare module, then re-wrap.
        if self.distributed and isinstance(self.model, DDP):
            bare_model = self.model.module
            ddp_find_unused = bool(getattr(self.config, 'ddp_find_unused_parameters', False))
            self.model = DDP(
                bare_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=ddp_find_unused,
            )
            if self.is_main_process:
                print(f"[INFO] Phase-1: DDP wrapper rebuilt after submodule reset (find_unused_parameters={ddp_find_unused})")

        param_groups = self._build_param_groups(self.config)
        if not param_groups:
            raise ValueError("Phase-1: no trainable parameters after reset/freeze")
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]

        # Recreate scheduler with the same policy
        if self.lr_scheduler_name == 'cosine':
            total_steps = len(self.train_loader) * self.config.max_epochs
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, total_steps - self.config.warmup_steps),
                eta_min=min(self.base_lrs) * self.config.min_lr_scale,
            )
        else:
            direction = self._metric_direction()
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min' if direction == 'min' else 'max',
                factor=self.config.plateau_factor,
                patience=self.config.plateau_patience,
                min_lr=[lr * self.config.min_lr_scale for lr in self.base_lrs],
            )
        if self.is_main_process:
            print(f"[INFO] Phase-1: optimizer + scheduler rebuilt with fresh state")
            for i, group in enumerate(self.optimizer.param_groups):
                n_p = sum(p.numel() for p in group['params'] if p.requires_grad)
                print(f"  group {i}: lr={group['lr']:.2e}, n_trainable_params={n_p:,}")

    def _geometry_scorer_kwargs(self, step: int, training: bool) -> Dict[str, Any]:
        """Compute Phase-1 runtime kwargs forwarded to GeometryCandidateScorer.

        - base_detach: only when training (validation must reflect real gradients-free output)
        - gate_override: 1.0 during the first gate_warmup_open_steps steps (training only)
        - beta: linear ramp from residual_beta_min -> 1.0 over residual_beta_warmup_steps
        """
        if not training:
            return {'base_detach': False, 'gate_override': None, 'beta': 1.0}
        kwargs: Dict[str, Any] = {}
        kwargs['base_detach'] = bool(getattr(self.config, 'detach_base_for_residual', False))
        gate_warmup = int(getattr(self.config, 'gate_warmup_open_steps', 0) or 0)
        kwargs['gate_override'] = 1.0 if (gate_warmup > 0 and step < gate_warmup) else None
        beta_warmup = int(getattr(self.config, 'residual_beta_warmup_steps', 0) or 0)
        if beta_warmup <= 0:
            kwargs['beta'] = 1.0
        else:
            beta_min = float(getattr(self.config, 'residual_beta_min', 0.1))
            frac = min(max(step / float(beta_warmup), 0.0), 1.0)
            kwargs['beta'] = beta_min + (1.0 - beta_min) * frac
        return kwargs

    def _build_param_groups(self, config):
        scorer_lr_scale = getattr(config, 'candidate_scorer_lr_scale', 1.0)
        geo_lr_scale = getattr(config, 'geometry_scorer_lr_scale', 1.0)
        has_separate_lr = (scorer_lr_scale != 1.0) or (geo_lr_scale != 1.0)
        if not has_separate_lr:
            params = self._trainable_model_parameters()
            return [{'params': params}] if params else []

        scorer_params = []
        geo_params = []
        other_params = []
        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue
            if 'candidate_chi1_scorer' in name:
                scorer_params.append(param)
            elif 'geometry_candidate_scorer' in name:
                geo_params.append(param)
            else:
                other_params.append(param)

        groups = []
        if other_params:
            groups.append({'params': other_params})
        if scorer_params and scorer_lr_scale != 1.0:
            groups.append({'params': scorer_params, 'lr': config.lr * scorer_lr_scale})
            if self.is_main_process:
                print(f"[INFO] Candidate scorer LR = {config.lr * scorer_lr_scale:.1e} ({scorer_lr_scale}x base)")
        elif scorer_params:
            groups[0]['params'].extend(scorer_params) if groups else groups.append({'params': scorer_params})
        if geo_params and geo_lr_scale != 1.0:
            groups.append({'params': geo_params, 'lr': config.lr * geo_lr_scale})
            if self.is_main_process:
                print(f"[INFO] Geometry scorer LR = {config.lr * geo_lr_scale:.1e} ({geo_lr_scale}x base)")
        elif geo_params:
            if groups:
                groups[0]['params'].extend(geo_params)
            else:
                groups.append({'params': geo_params})
        return groups

    def _generate_atom14_holo_with_fk(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        用 FK 从 holo backbone + torsion_holo 生成 atom14_holo
        
        这是 SQ1 理论方案：用 FK 动态生成真值以保持一致性
        """
        # 1. 从 holo backbone 构建 Rigid frames
        holo_rigids = self._build_rigids_from_backbone(
            batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
        )
        
        # 2. torsion_holo [B, N, 7] -> sin/cos [B, N, 7, 2]
        torsion_sincos = torch.stack([
            torch.sin(batch.torsion_holo),
            torch.cos(batch.torsion_holo)
        ], dim=-1)
        torsion_sincos = reorder_torsions_to_openfold(torsion_sincos)
        
        # 3. 从 sequences 生成 aatype
        B, N = batch.torsion_holo.shape[:2]
        aatype = sequences_to_aatype(batch.sequences, N, self.device)
        
        # 4. 调用 FK 生成 atom14（使用实际模型）
        fk_result = self._model.fk_module(torsion_sincos, holo_rigids, aatype)
        
        return fk_result['atom14_pos'], fk_result['atom14_mask']

    def _build_atom14_targets(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_atom14 = batch.atom14_holo
        raw_mask = batch.atom14_holo_mask

        if raw_atom14 is not None and raw_mask is not None:
            raw_mask = raw_mask.bool() & (raw_atom14.abs().sum(dim=-1) > 0)
            use_raw = raw_mask.view(raw_mask.shape[0], -1).any(dim=1)
            if use_raw.all():
                return raw_atom14, raw_mask
        else:
            raw_mask = None
            use_raw = None

        fk_atom14, fk_mask = self._generate_atom14_holo_with_fk(batch)
        fk_mask = fk_mask.bool()

        if raw_atom14 is None or raw_mask is None or use_raw is None:
            return fk_atom14, fk_mask

        target_atom14 = torch.where(use_raw[:, None, None, None], raw_atom14, fk_atom14)
        target_mask = torch.where(use_raw[:, None, None], raw_mask, fk_mask)
        return target_atom14, target_mask

    def compute_loss(self, outputs: Dict, batch, step: int) -> Dict[str, torch.Tensor]:
        pred_chi_sincos = outputs['pred_chi']  # [B,N,4,2]
        pocket_chi1_delta = outputs.get('pocket_chi1_delta', None)

        # Pocket warmup for chi supervision
        kappa = self.compute_pocket_warmup(step)
        w_res_warmed = batch.w_res * kappa + (1 - kappa) * 0.1

        loss_chi = torsion_sincos_loss(
            pred_chi_sincos,
            batch.chi_holo,
            batch.chi_mask,
            w_res_warmed,
        )

        current_lambda_pchi1 = self.compute_pchi1_lambda(step)

        target_atom14, target_atom14_mask = self._build_atom14_targets(batch)
        if batch.node_mask is not None:
            target_atom14_mask = target_atom14_mask & batch.node_mask.unsqueeze(-1)

        if current_lambda_pchi1 > 0.0:
            pchi1_residue_mask = self._compute_pchi1_residue_mask(batch, target_atom14, target_atom14_mask)
            pchi1_chi_mask = batch.chi_mask[:, :, :1] & pchi1_residue_mask.unsqueeze(-1)

            if getattr(self.config, 'pchi1_mask_mode', 'soft') == 'soft':
                pchi1_w_res = w_res_warmed
            else:
                pchi1_w_res = pchi1_residue_mask.float()

            loss_pchi1 = torsion_sincos_loss(
                pred_chi_sincos[:, :, :1, :],
                batch.chi_holo[:, :, :1],
                pchi1_chi_mask,
                pchi1_w_res,
            )
        else:
            loss_pchi1 = loss_chi.new_zeros(())

        chi1_rotamer_logits = outputs.get('chi1_rotamer_logits', None)
        if self.config.lambda_chi1_rotamer > 0.0 and chi1_rotamer_logits is not None:
            loss_chi1_rotamer = chi1_rotamer_loss(
                chi1_rotamer_logits,
                batch.chi_holo[:, :, 0],
                batch.chi_mask[:, :, 0] & batch.node_mask.bool(),
                w_res=w_res_warmed,
            )
        else:
            loss_chi1_rotamer = loss_chi.new_zeros(())

        contact_logits = outputs.get('contact_logits', None)
        if self.config.lambda_contact > 0.0 and contact_logits is not None:
            contact_mask = self._compute_batch_ligand_contact_mask(batch, target_atom14, target_atom14_mask)
            contact_residue_mask = batch.node_mask.bool()
            pos_weight = self._binary_contact_pos_weight(
                contact_mask,
                contact_residue_mask,
                float(getattr(self.config, 'contact_loss_pos_weight', 0.0)),
            )
            loss_contact = binary_contact_loss(
                contact_logits,
                contact_mask,
                residue_mask=contact_residue_mask,
                pos_weight=pos_weight,
            )
        else:
            loss_contact = loss_chi.new_zeros(())

        candidate_chi1_logits = outputs.get('candidate_chi1_logits', None)
        if self.config.lambda_candidate_chi1 > 0.0 and candidate_chi1_logits is not None:
            loss_candidate_chi1 = chi1_rotamer_loss(
                candidate_chi1_logits,
                batch.chi_holo[:, :, 0],
                batch.chi_mask[:, :, 0] & batch.node_mask.bool(),
                w_res=w_res_warmed,
            )
        else:
            loss_candidate_chi1 = loss_chi.new_zeros(())

        contrastive_decoy_logits = outputs.get('candidate_chi1_decoy_logits', None)
        loss_candidate_rerank = loss_chi.new_zeros(())
        if (
            self.config.lambda_ligand_contrastive > 0.0
            and candidate_chi1_logits is not None
            and contrastive_decoy_logits is not None
        ):
            loss_ligand_contrastive = ligand_contrastive_chi1_loss(
                candidate_chi1_logits,
                contrastive_decoy_logits,
                batch.chi_holo[:, :, 0],
                batch.chi_mask[:, :, 0] & batch.node_mask.bool(),
                w_res=w_res_warmed,
                margin=float(getattr(self.config, 'ligand_contrastive_margin', 0.2)),
            )
        else:
            loss_ligand_contrastive = loss_chi.new_zeros(())

        if (
            getattr(self.config, 'lambda_candidate_rerank', 0.0) > 0.0
            and candidate_chi1_logits is not None
            and contrastive_decoy_logits is not None
        ):
            rerank_contact_mask = None
            if bool(getattr(self.config, 'candidate_rerank_contact_only', True)):
                rerank_contact_mask = self._compute_ca_ligand_contact_mask(
                    batch,
                    threshold=float(getattr(self.config, 'g_noncontact_threshold', 8.0)),
                )
            loss_candidate_rerank = candidate_decoy_rerank_loss(
                candidate_chi1_logits,
                contrastive_decoy_logits,
                batch.torsion_apo[:, :, 3],
                batch.chi_holo[:, :, 0],
                batch.chi_mask[:, :, 0] & batch.node_mask.bool(),
                contact_mask=rerank_contact_mask,
                w_res=w_res_warmed,
                decoy_margin=float(getattr(self.config, 'candidate_rerank_margin', 0.1)),
                rank_margin=float(getattr(self.config, 'candidate_rerank_rank_margin', 0.0)),
            )

        geometry_chi1_logits = outputs.get('geometry_chi1_logits', None)
        if self.config.lambda_geometry_chi1 > 0.0 and geometry_chi1_logits is not None:
            loss_geometry_chi1 = chi1_rotamer_loss(
                geometry_chi1_logits,
                batch.chi_holo[:, :, 0],
                batch.chi_mask[:, :, 0] & batch.node_mask.bool(),
                w_res=w_res_warmed,
            )
        else:
            loss_geometry_chi1 = loss_chi.new_zeros(())

        # Base prior loss: train the Dunbrack-like branch independently
        geometry_chi1_base_logits = outputs.get('geometry_chi1_base_logits', None)
        if self.config.lambda_base_prior > 0.0 and geometry_chi1_base_logits is not None:
            loss_base_prior = chi1_rotamer_loss(
                geometry_chi1_base_logits,
                batch.chi_holo[:, :, 0],
                batch.chi_mask[:, :, 0] & batch.node_mask.bool(),
                w_res=w_res_warmed,
            )
        else:
            loss_base_prior = loss_chi.new_zeros(())

        # Phase 2: switch-aware and ligand-causal losses
        geom_logits_for_phase2 = geometry_chi1_logits if geometry_chi1_logits is not None else candidate_chi1_logits
        if self.config.lambda_switch_bce > 0.0 and geom_logits_for_phase2 is not None:
            loss_switch_bce = switch_bce_loss(
                geom_logits_for_phase2,
                batch.torsion_apo[:, :, 3],
                batch.chi_holo[:, :, 0],
                batch.chi_mask[:, :, 0] & batch.node_mask.bool(),
                w_res=w_res_warmed,
            )
        else:
            loss_switch_bce = loss_chi.new_zeros(())

        if self.config.lambda_rescue_noharm > 0.0 and geom_logits_for_phase2 is not None:
            loss_rescue_noharm = rescue_noharm_loss(
                geom_logits_for_phase2,
                batch.torsion_apo[:, :, 3],
                batch.chi_holo[:, :, 0],
                batch.chi_mask[:, :, 0] & batch.node_mask.bool(),
                w_res=w_res_warmed,
                margin=float(getattr(self.config, 'rescue_noharm_margin', 0.5)),
            )
        else:
            loss_rescue_noharm = loss_chi.new_zeros(())

        contrastive_nolig_logits = outputs.get('geometry_chi1_nolig_logits', None)
        if self.config.lambda_ligand_residual > 0.0 and geom_logits_for_phase2 is not None and contrastive_nolig_logits is not None:
            loss_ligand_residual = ligand_residual_chi1_loss(
                geom_logits_for_phase2,
                contrastive_nolig_logits,
                batch.chi_holo[:, :, 0],
                batch.chi_mask[:, :, 0] & batch.node_mask.bool(),
                w_res=w_res_warmed,
                margin=float(getattr(self.config, 'ligand_residual_margin', 0.3)),
            )
        else:
            loss_ligand_residual = loss_chi.new_zeros(())

        # === Phase-1: likelihood-ratio losses on G_i = log_softmax(full) - log_softmax(base) ===
        loss_g_lift = loss_chi.new_zeros(())
        loss_g_noharm = loss_chi.new_zeros(())
        loss_g_zero = loss_chi.new_zeros(())
        # === Phase-1 v2 G-vector losses ===
        loss_g_switch_dir = loss_chi.new_zeros(())
        loss_g_switch_amp = loss_chi.new_zeros(())
        loss_g_switch_rank = loss_chi.new_zeros(())
        loss_g_antiharm = loss_chi.new_zeros(())
        loss_g_decoy = loss_chi.new_zeros(())
        loss_typed_candidate_energy = loss_chi.new_zeros(())
        if (geometry_chi1_logits is not None and geometry_chi1_base_logits is not None):
            chi_holo_t = batch.chi_holo[:, :, 0]
            chi_mask_t = batch.chi_mask[:, :, 0] & batch.node_mask.bool()
            apo_chi1_t = batch.torsion_apo[:, :, 3]
            if self.config.lambda_g_lift_switch > 0.0:
                loss_g_lift = g_lift_switch_loss(
                    geometry_chi1_logits, geometry_chi1_base_logits,
                    apo_chi1_t, chi_holo_t, chi_mask_t,
                    margin=float(getattr(self.config, 'g_lift_margin', 0.5)),
                    w_res=w_res_warmed,
                )
            if self.config.lambda_g_noharm > 0.0:
                loss_g_noharm = g_noharm_loss(
                    geometry_chi1_logits, geometry_chi1_base_logits,
                    apo_chi1_t, chi_holo_t, chi_mask_t,
                    margin=float(getattr(self.config, 'g_noharm_margin', 0.5)),
                    w_res=w_res_warmed,
                )
            if self.config.lambda_g_zero_noncontact > 0.0:
                non_contact = self._compute_ca_ligand_non_contact_mask(
                    batch, threshold=float(getattr(self.config, 'g_noncontact_threshold', 8.0)),
                )
                loss_g_zero = g_zero_noncontact_loss(
                    geometry_chi1_logits, geometry_chi1_base_logits, chi_mask_t, non_contact,
                )
            # ----- v2 losses -----
            v2_active = (
                self.config.lambda_g_switch_dir > 0.0
                or self.config.lambda_g_switch_amp > 0.0
                or self.config.lambda_g_switch_rank > 0.0
                or self.config.lambda_g_antiharm > 0.0
                or self.config.lambda_g_decoy > 0.0
            )
            if v2_active:
                contact_only = bool(getattr(self.config, 'g_switch_contact_only', True))
                contact_mask_v2 = None
                if contact_only:
                    contact_mask_v2 = self._compute_ca_ligand_contact_mask(
                        batch, threshold=float(getattr(self.config, 'g_noncontact_threshold', 8.0)),
                    )
                if self.config.lambda_g_switch_dir > 0.0:
                    loss_g_switch_dir = g_switch_dir_loss(
                        geometry_chi1_logits, geometry_chi1_base_logits,
                        apo_chi1_t, chi_holo_t, chi_mask_t,
                        contact_mask=contact_mask_v2,
                        temperature=float(getattr(self.config, 'g_switch_temperature', 1.0)),
                    )
                if self.config.lambda_g_switch_amp > 0.0:
                    loss_g_switch_amp = g_switch_amp_loss(
                        geometry_chi1_logits, geometry_chi1_base_logits,
                        apo_chi1_t, chi_holo_t, chi_mask_t,
                        contact_mask=contact_mask_v2,
                        margin=float(getattr(self.config, 'g_switch_amp_margin', 0.05)),
                    )
                if self.config.lambda_g_switch_rank > 0.0:
                    loss_g_switch_rank = g_switch_rank_loss(
                        geometry_chi1_logits, geometry_chi1_base_logits,
                        apo_chi1_t, chi_holo_t, chi_mask_t,
                        contact_mask=contact_mask_v2,
                        margin=float(getattr(self.config, 'g_switch_rank_margin', 0.05)),
                    )
                if self.config.lambda_g_antiharm > 0.0:
                    loss_g_antiharm = g_antiharm_loss(
                        geometry_chi1_logits, geometry_chi1_base_logits,
                        apo_chi1_t, chi_holo_t, chi_mask_t,
                        contact_mask=contact_mask_v2,
                        tau=float(getattr(self.config, 'g_antiharm_tau', 0.05)),
                    )
                if self.config.lambda_g_decoy > 0.0:
                    decoy_logits = outputs.get('geometry_chi1_decoy_logits', None)
                    if decoy_logits is not None:
                        loss_g_decoy = g_decoy_contrastive_loss(
                            geometry_chi1_logits, decoy_logits, geometry_chi1_base_logits,
                            apo_chi1_t, chi_holo_t, chi_mask_t,
                            contact_mask=contact_mask_v2,
                            margin=float(getattr(self.config, 'g_decoy_margin', 0.05)),
                        )
            if self.config.lambda_typed_candidate_energy > 0.0:
                typed_energy = outputs.get('geometry_chi1_typed_energy', None)
                typed_decoy_energy = outputs.get('geometry_chi1_typed_decoy_energy', None)
                if typed_energy is not None and typed_decoy_energy is not None:
                    typed_contact_mask = None
                    if bool(getattr(self.config, 'typed_candidate_contact_only', True)):
                        typed_contact_mask = self._compute_ca_ligand_contact_mask(
                            batch, threshold=float(getattr(self.config, 'g_noncontact_threshold', 8.0)),
                        )
                    typed_noncontact = self._compute_ca_ligand_non_contact_mask(
                        batch, threshold=float(getattr(self.config, 'g_noncontact_threshold', 8.0)),
                    )
                    loss_typed_candidate_energy = typed_candidate_energy_loss(
                        typed_energy,
                        typed_decoy_energy,
                        apo_chi1_t,
                        chi_holo_t,
                        chi_mask_t,
                        contact_mask=typed_contact_mask,
                        non_contact_mask=typed_noncontact,
                        margin=float(getattr(self.config, 'typed_candidate_margin', 0.05)),
                        noharm_weight=float(getattr(self.config, 'typed_candidate_noharm_weight', 0.1)),
                        noncontact_zero_weight=float(getattr(self.config, 'typed_candidate_noncontact_zero_weight', 0.05)),
                    )

        # FAPE on atom14 (if available)
        pred_atom14 = outputs['atom14_pos']
        pred_atom14_mask = outputs['atom14_mask'].bool()
        if batch.node_mask is not None:
            pred_atom14_mask = pred_atom14_mask & batch.node_mask.unsqueeze(-1)
        pred_R = outputs['rigids_final'].get_rots().get_rot_mats()
        pred_t = outputs['rigids_final'].get_trans()

        true_R, true_t = self._build_frames_from_backbone(
            batch.N_holo,
            batch.Ca_holo,
            batch.C_holo,
            batch.node_mask,
        )

        fape_mask = batch.node_mask.float() if batch.node_mask is not None else None
        loss_fape = fape_loss(
            pred_atom14,
            target_atom14,
            (pred_R, pred_t),
            (true_R, true_t),
            w_res=fape_mask,
            atom_mask=pred_atom14_mask & target_atom14_mask,
        )

        # Clash on predicted atoms
        B, N, A, _ = pred_atom14.shape
        pred_all_atoms = pred_atom14.reshape(B, -1, 3)
        pred_all_atom_mask = pred_atom14_mask.reshape(B, -1)
        atom_aatype = sequences_to_aatype(batch.sequences, N, pred_atom14.device)
        loss_clash = clash_penalty(
            pred_all_atoms,
            clash_threshold=2.2,
            aatype=atom_aatype,
            atom_mask=pred_all_atom_mask,
        )

        total_loss = (
            self.config.w_fape * loss_fape +
            self.config.w_chi * loss_chi +
            self.config.w_clash * loss_clash +
            current_lambda_pchi1 * loss_pchi1 +
            self.config.lambda_chi1_rotamer * loss_chi1_rotamer +
            self.config.lambda_contact * loss_contact +
            self.config.lambda_candidate_chi1 * loss_candidate_chi1 +
            self.config.lambda_ligand_contrastive * loss_ligand_contrastive +
            self.config.lambda_candidate_rerank * loss_candidate_rerank +
            self.config.lambda_geometry_chi1 * loss_geometry_chi1 +
            self.config.lambda_base_prior * loss_base_prior +
            self.config.lambda_switch_bce * loss_switch_bce +
            self.config.lambda_rescue_noharm * loss_rescue_noharm +
            self.config.lambda_ligand_residual * loss_ligand_residual +
            self.config.lambda_g_lift_switch * loss_g_lift +
            self.config.lambda_g_noharm * loss_g_noharm +
            self.config.lambda_g_zero_noncontact * loss_g_zero +
            self.config.lambda_g_switch_dir * loss_g_switch_dir +
            self.config.lambda_g_switch_amp * loss_g_switch_amp +
            self.config.lambda_g_switch_rank * loss_g_switch_rank +
            self.config.lambda_g_antiharm * loss_g_antiharm +
            self.config.lambda_g_decoy * loss_g_decoy +
            self.config.lambda_typed_candidate_energy * loss_typed_candidate_energy
        )

        if pocket_chi1_delta is not None:
            loss_expert_delta = pocket_chi1_delta.norm(dim=-1).mean()
        else:
            loss_expert_delta = loss_chi.new_zeros(())

        return {
            'total': total_loss,
            'chi': loss_chi,
            'pchi1': loss_pchi1,
            'chi1_rotamer': loss_chi1_rotamer,
            'contact': loss_contact,
            'candidate_chi1': loss_candidate_chi1,
            'ligand_contrastive': loss_ligand_contrastive,
            'candidate_rerank': loss_candidate_rerank,
            'geometry_chi1': loss_geometry_chi1,
            'base_prior': loss_base_prior,
            'switch_bce': loss_switch_bce,
            'rescue_noharm': loss_rescue_noharm,
            'ligand_residual': loss_ligand_residual,
            'g_lift_switch': loss_g_lift,
            'g_noharm': loss_g_noharm,
            'g_zero_noncontact': loss_g_zero,
            'g_switch_dir': loss_g_switch_dir,
            'g_switch_amp': loss_g_switch_amp,
            'g_switch_rank': loss_g_switch_rank,
            'g_antiharm': loss_g_antiharm,
            'g_decoy': loss_g_decoy,
            'typed_candidate_energy': loss_typed_candidate_energy,
            'expert_delta': loss_expert_delta,
            'clash': loss_clash,
            'fape': loss_fape,
        }

    def _check_nan(self, tensor: torch.Tensor, name: str) -> bool:
        """检查 tensor 是否包含 NaN/Inf"""
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan or has_inf:
            print(f"[NaN DEBUG] {name}: nan={has_nan}, inf={has_inf}, shape={tensor.shape}")
            return True
        return False

    def _debug_batch(self, batch):
        """调试输入数据"""
        issues = []
        if self._check_nan(batch.esm, "esm"): issues.append("esm")
        if self._check_nan(batch.N_apo, "N_apo"): issues.append("N_apo")
        if self._check_nan(batch.Ca_apo, "Ca_apo"): issues.append("Ca_apo")
        if self._check_nan(batch.N_holo, "N_holo"): issues.append("N_holo")
        if self._check_nan(batch.Ca_holo, "Ca_holo"): issues.append("Ca_holo")
        if self._check_nan(batch.chi_holo, "chi_holo"): issues.append("chi_holo")
        if self._check_nan(batch.torsion_holo, "torsion_holo"): issues.append("torsion_holo")
        if self._check_nan(batch.lig_points, "lig_points"): issues.append("lig_points")
        
        # 检查骨架坐标是否有异常值（太大或全零）
        ca_max = batch.Ca_apo.abs().max().item()
        ca_min = batch.Ca_apo[batch.node_mask.bool()].abs().min().item() if batch.node_mask.any() else 0
        if ca_max > 1000:
            print(f"[NaN DEBUG] Ca_apo 坐标过大: max={ca_max}")
            issues.append("Ca_apo_large")
        
        # 检查是否有全零的残基（有效位置）
        valid_mask = batch.node_mask.bool()
        ca_norms = torch.norm(batch.Ca_apo, dim=-1)  # [B, N]
        zero_ca = (ca_norms[valid_mask] < 1e-6).sum().item()
        if zero_ca > 0:
            print(f"[NaN DEBUG] 有 {zero_ca} 个有效残基的 Ca 坐标接近零")
            issues.append("Ca_zero")
        
        if issues:
            print(f"[NaN DEBUG] 输入数据有问题: {issues}")
            print(f"[NaN DEBUG] sequences[0]: {batch.sequences[0][:50]}...")
            return False
        return True

    def _debug_outputs(self, outputs):
        """调试模型输出"""
        issues = []
        for key, val in outputs.items():
            if isinstance(val, torch.Tensor):
                if self._check_nan(val, f"output.{key}"):
                    issues.append(key)
        if issues:
            print(f"[NaN DEBUG] 模型输出有问题: {issues}")
            return False
        return True

    def train_step(self, batch) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        batch = self._batch_to_device(batch)

        # === 输入检查：提前检测异常数据 ===
        def has_bad_values(t, name=None):
            if torch.isnan(t).any() or torch.isinf(t).any():
                return True
            # 检查坐标是否有极端值（超过 1000 Å）
            if t.abs().max() > 10000:
                return True
            return False
        
        input_bad_local = (
            has_bad_values(batch.Ca_apo) or 
            has_bad_values(batch.Ca_holo) or
            has_bad_values(batch.esm) or
            has_bad_values(batch.lig_points)
        )
        
        # DDP 模式：同步输入检查状态，任一 rank 有问题则所有 rank 都跳过
        if self.distributed:
            bad_flag = torch.tensor([1.0 if input_bad_local else 0.0], device=self.device)
            dist.all_reduce(bad_flag, op=dist.ReduceOp.MAX)
            input_bad = bad_flag.item() > 0
        else:
            input_bad = input_bad_local
        
        if input_bad:
            if self.is_main_process and input_bad_local:
                print(f"[WARN] Bad input data at step {self.global_step}, skipping batch (pdb_ids: {batch.pdb_ids[:3]}...)")
            self.global_step += 1
            return {
                'total': float('nan'), 'chi': float('nan'), 'pchi1': float('nan'),
                'chi1_rotamer': float('nan'), 'contact': float('nan'),
                'candidate_chi1': float('nan'), 'ligand_contrastive': float('nan'),
                'candidate_rerank': float('nan'),
                'geometry_chi1': float('nan'),
                'base_prior': float('nan'),
                'switch_bce': float('nan'), 'rescue_noharm': float('nan'), 'ligand_residual': float('nan'),
                'g_lift_switch': float('nan'), 'g_noharm': float('nan'), 'g_zero_noncontact': float('nan'),
                'g_switch_dir': float('nan'), 'g_switch_amp': float('nan'), 'g_switch_rank': float('nan'),
                'g_antiharm': float('nan'), 'g_decoy': float('nan'),
                'typed_candidate_energy': float('nan'),
                'expert_delta': float('nan'), 'fape': float('nan'), 'clash': float('nan')
            }

        def tensor_has_bad_values(t: torch.Tensor) -> bool:
            return torch.isnan(t).any().item() or torch.isinf(t).any().item()

        # Forward pass
        forward_error = False
        outputs = {}
        scorer_kwargs_train = self._geometry_scorer_kwargs(self.global_step, training=True)
        # The "no-ligand" pass for contrastive losses must always include the residual
        # path with full beta and no detach: it is a pure inference of "what would
        # happen without ligand?" used as a contrastive denominator.
        scorer_kwargs_nolig = self._geometry_scorer_kwargs(self.global_step, training=False)

        def _v2_decoy_batch(b):
            """Build the v2 G-decoy batch according to config.g_decoy_kind."""
            kind = getattr(self.config, 'g_decoy_kind', 'translated')
            if kind == 'nolig':
                return self._make_nolig_batch(b)
            if kind == 'shuffled':
                return self._make_batch_shuffled_ligand(b)
            return self._make_translated_batch(
                b, offset=float(getattr(self.config, 'g_decoy_translation_offset', 100.0)),
            )

        def _candidate_rerank_decoy_batch(b):
            """Build the candidate-rerank decoy batch without affecting legacy shuffled contrastive."""
            kind = getattr(self.config, 'candidate_rerank_decoy_kind', 'translated')
            if kind == 'nolig':
                return self._make_nolig_batch(b)
            if kind == 'shuffled':
                return self._make_batch_shuffled_ligand(b)
            return self._make_translated_batch(
                b, offset=float(getattr(self.config, 'g_decoy_translation_offset', 100.0)),
            )

        def _do_forward_with_aux():
            outs = self.model(batch, self.global_step, geometry_scorer_kwargs=scorer_kwargs_train)
            if (
                getattr(self.config, 'lambda_ligand_contrastive', 0.0) > 0.0
                or getattr(self.config, 'lambda_candidate_rerank', 0.0) > 0.0
            ):
                if getattr(self.config, 'lambda_candidate_rerank', 0.0) > 0.0:
                    decoy_batch = _candidate_rerank_decoy_batch(batch)
                else:
                    decoy_batch = self._make_batch_shuffled_ligand(batch)
                if decoy_batch is not None:
                    decoy_outputs = self.model(decoy_batch, self.global_step, geometry_scorer_kwargs=scorer_kwargs_train)
                    outs['candidate_chi1_decoy_logits'] = decoy_outputs.get('candidate_chi1_logits', None)
            if getattr(self.config, 'lambda_ligand_residual', 0.0) > 0.0 or getattr(self.config, 'use_nolig_contrastive', False):
                nolig_batch = self._make_nolig_batch(batch)
                nolig_outputs = self.model(nolig_batch, self.global_step, geometry_scorer_kwargs=scorer_kwargs_nolig)
                outs['geometry_chi1_nolig_logits'] = nolig_outputs.get('geometry_chi1_logits', None)
            # v2 G-decoy contrastive forward
            if getattr(self.config, 'lambda_g_decoy', 0.0) > 0.0:
                v2_decoy = _v2_decoy_batch(batch)
                v2_decoy_outputs = self.model(v2_decoy, self.global_step, geometry_scorer_kwargs=scorer_kwargs_train)
                outs['geometry_chi1_decoy_logits'] = v2_decoy_outputs.get('geometry_chi1_logits', None)
            if getattr(self.config, 'lambda_typed_candidate_energy', 0.0) > 0.0:
                typed_decoy = self._make_typed_candidate_decoy_batch(batch)
                typed_decoy_outputs = self.model(typed_decoy, self.global_step, geometry_scorer_kwargs=scorer_kwargs_train)
                outs['geometry_chi1_typed_decoy_energy'] = typed_decoy_outputs.get('geometry_chi1_typed_energy', None)
                outs['geometry_chi1_decoy_logits'] = typed_decoy_outputs.get('geometry_chi1_logits', None)
            return outs

        try:
            if self.autocast_dtype is not None:
                with self._autocast_context():
                    outputs = _do_forward_with_aux()
                    losses = self.compute_loss(outputs, batch, self.global_step)
                    loss = losses['total']
            else:
                outputs = _do_forward_with_aux()
                losses = self.compute_loss(outputs, batch, self.global_step)
                loss = losses['total']
        except RuntimeError as e:
            # 捕获 CUDA 错误等
            if self.is_main_process:
                print(f"[WARN] Runtime error at step {self.global_step}: {str(e)[:80]}, skipping batch")
            forward_error = True
            loss = torch.tensor(float('nan'), device=self.device)
            losses = {
                'total': loss, 'chi': loss, 'pchi1': loss,
                'chi1_rotamer': loss, 'contact': loss,
                'candidate_chi1': loss, 'ligand_contrastive': loss, 'candidate_rerank': loss,
                'geometry_chi1': loss,
                'base_prior': loss,
                'switch_bce': loss, 'rescue_noharm': loss, 'ligand_residual': loss,
                'g_lift_switch': loss, 'g_noharm': loss, 'g_zero_noncontact': loss,
                'g_switch_dir': loss, 'g_switch_amp': loss, 'g_switch_rank': loss,
                'g_antiharm': loss, 'g_decoy': loss,
                'typed_candidate_energy': loss,
                'expert_delta': loss, 'fape': loss, 'clash': loss
            }

        # === NaN/Inf 检查：DDP 模式下同步跳过状态 ===
        # 同时检查输出张量与刚体状态，防止 loss 仍有限但中间量已坏掉。
        outputs_bad_local = False
        if not forward_error:
            for _val in outputs.values():
                if isinstance(_val, torch.Tensor) and _val.is_floating_point():
                    if tensor_has_bad_values(_val):
                        outputs_bad_local = True
                        break
            if not outputs_bad_local and 'rigids_final' in outputs:
                rigid = outputs['rigids_final']
                rigid_trans = rigid.get_trans()
                rigid_rots = rigid.get_rots().get_rot_mats()
                outputs_bad_local = (
                    tensor_has_bad_values(rigid_trans) or
                    tensor_has_bad_values(rigid_rots)
                )

        losses_bad_local = any(
            isinstance(v, torch.Tensor) and v.is_floating_point() and tensor_has_bad_values(v)
            for v in losses.values()
        )
        loss_requires_grad_local = isinstance(loss, torch.Tensor) and loss.requires_grad
        loss_is_bad_local = (
            forward_error
            or outputs_bad_local
            or losses_bad_local
            or not loss_requires_grad_local
        )

        # DDP 模式：同步坏 batch 状态，任一 rank 有问题则所有 rank 都跳过 backward。
        if self.distributed:
            nan_flag = torch.tensor([1.0 if loss_is_bad_local else 0.0], device=self.device)
            dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
            skip_backward = nan_flag.item() > 0
        else:
            skip_backward = loss_is_bad_local

        if skip_backward:
            if self.is_main_process and loss_is_bad_local:
                if not loss_requires_grad_local and not (forward_error or outputs_bad_local or losses_bad_local):
                    print(
                        f"[WARN] Detached loss at step {self.global_step}, "
                        f"skipping batch (pdb_ids: {batch.pdb_ids[:3]}...)"
                    )
                else:
                    print(
                        f"[WARN] NaN/Inf loss detected at step {self.global_step}, "
                        f"skipping batch (pdb_ids: {batch.pdb_ids[:3]}...)"
                    )
            self.optimizer.zero_grad()  # 清除任何残留梯度
            self.global_step += 1
            return {k: float('nan') for k in losses.keys()}

        # Backward pass (所有 rank 同时执行)
        model_params = self._model.parameters()
        loss_for_backward = loss
        if self.distributed:
            local_batch_size = torch.tensor(
                [float(batch.esm.shape[0])],
                device=self.device,
            )
            dist.all_reduce(local_batch_size, op=dist.ReduceOp.SUM)
            global_batch_size = max(local_batch_size.item(), 1.0)
            loss_for_backward = loss * (
                batch.esm.shape[0] * self.world_size / global_batch_size
            )

        if self.distributed:
            if self.scaler is not None:
                self.scaler.scale(loss_for_backward).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss_for_backward.backward()
        else:
            try:
                if self.scaler is not None:
                    self.scaler.scale(loss_for_backward).backward()
                    self.scaler.unscale_(self.optimizer)
                else:
                    loss_for_backward.backward()
            except RuntimeError as e:
                if self.is_main_process:
                    print(f"[WARN] Backward error at step {self.global_step}: {str(e)[:120]}, skipping batch")
                self.optimizer.zero_grad()
                self.global_step += 1
                return {k: float('nan') for k in losses.keys()}

        grad_norm = torch.nn.utils.clip_grad_norm_(model_params, self.config.grad_clip)
        grad_norm_value = float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        grad_bad_local = not math.isfinite(grad_norm_value)

        if self.distributed:
            grad_flag = torch.tensor([1.0 if grad_bad_local else 0.0], device=self.device)
            dist.all_reduce(grad_flag, op=dist.ReduceOp.MAX)
            skip_step = grad_flag.item() > 0
        else:
            skip_step = grad_bad_local

        if skip_step:
            sample_preview = ', '.join(batch.pdb_ids[:3])
            if grad_bad_local:
                print(
                    f"[WARN][rank {self.local_rank}] Non-finite grad norm at step "
                    f"{self.global_step}, skipping optimizer step (pdb_ids: {sample_preview})"
                )
            elif self.is_main_process:
                print(
                    f"[WARN] Non-finite grad norm on a remote rank at step "
                    f"{self.global_step}, skipping optimizer step"
                )
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.update()
            self.global_step += 1
            self._apply_warmup_lr()
            return {k: float('nan') for k in losses.keys()}

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if (
            self.lr_scheduler_name == 'cosine'
            and self.global_step >= self.config.warmup_steps
        ):
            self.scheduler.step()

        self.global_step += 1
        self._apply_warmup_lr()

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    def _batch_to_device(self, batch):
        batch.esm = batch.esm.to(self.device)
        batch.N_apo = batch.N_apo.to(self.device)
        batch.Ca_apo = batch.Ca_apo.to(self.device)
        batch.C_apo = batch.C_apo.to(self.device)
        batch.N_holo = batch.N_holo.to(self.device)
        batch.Ca_holo = batch.Ca_holo.to(self.device)
        batch.C_holo = batch.C_holo.to(self.device)
        batch.node_mask = batch.node_mask.to(self.device)
        batch.lig_points = batch.lig_points.to(self.device)
        batch.lig_types = batch.lig_types.to(self.device)
        batch.lig_mask = batch.lig_mask.to(self.device)
        batch.chi_holo = batch.chi_holo.to(self.device)
        batch.chi_mask = batch.chi_mask.to(self.device)
        batch.torsion_apo = batch.torsion_apo.to(self.device)
        batch.torsion_holo = batch.torsion_holo.to(self.device)
        batch.w_res = batch.w_res.to(self.device)
        if batch.atom14_holo is not None:
            batch.atom14_holo = batch.atom14_holo.to(self.device)
        if batch.atom14_holo_mask is not None:
            batch.atom14_holo_mask = batch.atom14_holo_mask.to(self.device)
        return batch

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        
        # DDP: 每个 epoch 更新 sampler 以获得不同的 shuffle
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.current_epoch)

        epoch_losses = {
            'total': 0.0,
            'chi': 0.0,
            'pchi1': 0.0,
            'chi1_rotamer': 0.0,
            'contact': 0.0,
            'candidate_chi1': 0.0,
            'ligand_contrastive': 0.0,
            'candidate_rerank': 0.0,
            'geometry_chi1': 0.0,
            'base_prior': 0.0,
            'switch_bce': 0.0,
            'rescue_noharm': 0.0,
            'ligand_residual': 0.0,
            'g_lift_switch': 0.0,
            'g_noharm': 0.0,
            'g_zero_noncontact': 0.0,
            'g_switch_dir': 0.0,
            'g_switch_amp': 0.0,
            'g_switch_rank': 0.0,
            'g_antiharm': 0.0,
            'g_decoy': 0.0,
            'typed_candidate_energy': 0.0,
            'expert_delta': 0.0,
            'clash': 0.0,
            'fape': 0.0,
        }

        # 只在主进程显示进度条
        if self.is_main_process:
            pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch:3d}', ncols=120, leave=True)
        else:
            pbar = self.train_loader
            
        n_batches = 0
        n_skipped = 0
        for batch in pbar:
            if batch is None:
                continue
            step_losses = self.train_step(batch)
            
            # 跳过 NaN batch（不计入平均）
            if math.isnan(step_losses['total']):
                n_skipped += 1
                if self.is_main_process and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix({
                        'loss': 'nan',
                        'skipped': n_skipped,
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.1e}",
                    }, refresh=True)
                continue
                
            for k in epoch_losses:
                epoch_losses[k] += step_losses[k]
            n_batches += 1

            if self.is_main_process and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'loss': f"{step_losses['total']:.3f}",
                    'chi': f"{step_losses['chi']:.3f}",
                    'pchi1': f"{step_losses['pchi1']:.3f}",
                    'rot': f"{step_losses['chi1_rotamer']:.3f}",
                    'ct': f"{step_losses['contact']:.3f}",
                    'cand': f"{step_losses['candidate_chi1']:.3f}",
                    'lcon': f"{step_losses['ligand_contrastive']:.3f}",
                    'crnk': f"{step_losses['candidate_rerank']:.3f}",
                    'typed': f"{step_losses['typed_candidate_energy']:.3f}",
                    'exp': f"{step_losses['expert_delta']:.3f}",
                    'fape': f"{step_losses['fape']:.3f}",
                    'clash': f"{step_losses['clash']:.3f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.1e}",
                }, refresh=True)

        if n_skipped > 0 and self.is_main_process:
            print(f"  [INFO] Epoch {self.current_epoch}: skipped {n_skipped} NaN batches")

        n_batches = max(n_batches, 1)
        return {k: v / n_batches for k, v in epoch_losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        验证函数。
        
        DDP 模式下只在主进程执行验证，其他进程等待 barrier。
        这样可以完全避免验证阶段的 collective 同步问题。
        """
        self.model.eval()

        # DDP 模式：非主进程只等待 barrier，不执行验证
        if self.distributed and not self.is_main_process:
            dist.barrier()  # 等待主进程完成验证
            # 返回空结果，train 函数中只有主进程会使用验证结果
            return {
                'total': 0.0, 'chi': 0.0, 'pchi1': 0.0, 'clash': 0.0, 'fape': 0.0,
                'chi1_rotamer': 0.0,
                'contact': 0.0,
                'candidate_chi1': 0.0,
                'ligand_contrastive': 0.0,
                'candidate_rerank': 0.0,
                'typed_candidate_energy': 0.0,
                'chi1_acc': 0.0,
                'pocket_chi1_acc': 0.0,
                'pocket_chi1_mae_deg': 0.0,
                'pocket_chi1_median_deg': 0.0,
                'pocket_chi12_acc': 0.0,
                'pocket_contact_precision': 0.0,
                'pocket_contact_recall': 0.0,
                'pocket_contact_f1': 0.0,
                'pocket_contact_iou': 0.0,
                'pocket_irmsd': 0.0,
                'clash_pct': 0.0,
                'pocket_clash_pct': 0.0,
                'chi1_rotamer_acc': 0.0,
                'contact_posterior_precision': 0.0,
                'contact_posterior_recall': 0.0,
                'contact_posterior_f1': 0.0,
            }

        # === 以下只在主进程执行 ===
        val_losses = {
            'total': 0.0,
            'chi': 0.0,
            'pchi1': 0.0,
            'chi1_rotamer': 0.0,
            'contact': 0.0,
            'candidate_chi1': 0.0,
            'ligand_contrastive': 0.0,
            'candidate_rerank': 0.0,
            'geometry_chi1': 0.0,
            'base_prior': 0.0,
            'switch_bce': 0.0,
            'rescue_noharm': 0.0,
            'ligand_residual': 0.0,
            'g_lift_switch': 0.0,
            'g_noharm': 0.0,
            'g_zero_noncontact': 0.0,
            'g_switch_dir': 0.0,
            'g_switch_amp': 0.0,
            'g_switch_rank': 0.0,
            'g_antiharm': 0.0,
            'g_decoy': 0.0,
            'typed_candidate_energy': 0.0,
            'expert_delta': 0.0,
            'clash': 0.0,
            'fape': 0.0,
        }

        compute_slow_metrics = getattr(self.config, 'compute_slow_metrics', False)
        enable_dual_mask_audit = getattr(self.config, 'enable_dual_mask_audit', False)

        mask_items = self._audit_mask_items(enable_dual_mask_audit)

        metric_stats = {
            'chi1_hits': 0.0,
            'chi1_total': 0,
            'chi1_rotamer_hits': 0.0,
            'chi1_rotamer_total': 0,
            'contact_post_tp': 0,
            'contact_post_fp': 0,
            'contact_post_fn': 0,
            'clash_pct_sum': 0.0,
            'clash_pct_count': 0,
        }
        ligand_lift_stats = {
            key: {'full_hits': 0.0, 'base_hits': 0.0, 'decoy_hits': 0.0, 'total': 0}
            for key in ('switch', 'apo_wrong', 'contact', 'contact_switch', 'pocket_switch')
        }
        candidate_lift_stats = {
            key: {'full_hits': 0.0, 'decoy_hits': 0.0, 'total': 0}
            for key in ('switch', 'apo_wrong', 'contact', 'contact_switch', 'pocket_switch')
        }
        typed_energy_stats = {
            key: {'gap_sum': 0.0, 'correct_sum': 0.0, 'decoy_sum': 0.0, 'abs_sum': 0.0, 'total': 0}
            for key in ('switch', 'contact', 'contact_switch', 'pocket_switch')
        }
        angle_buffers: Dict[str, List[np.ndarray]] = {}
        chi_hits: Dict[str, float] = {}
        chi_totals: Dict[str, int] = {}
        chi12_hits: Dict[str, float] = {}
        chi12_totals: Dict[str, int] = {}
        contact_confusion: Dict[str, Dict[str, int]] = {}
        irmsd_sums: Dict[str, float] = {}
        irmsd_counts: Dict[str, int] = {}
        pocket_clash_sums: Dict[str, float] = {}
        pocket_clash_counts: Dict[str, int] = {}

        for metric_key, _ in mask_items:
            angle_buffers[metric_key] = []
            chi_hits[metric_key] = 0.0
            chi_totals[metric_key] = 0
            chi12_hits[metric_key] = 0.0
            chi12_totals[metric_key] = 0
            contact_confusion[metric_key] = {'tp': 0, 'fp': 0, 'fn': 0, 'union': 0}
            irmsd_sums[metric_key] = 0.0
            irmsd_counts[metric_key] = 0
            pocket_clash_sums[metric_key] = 0.0
            pocket_clash_counts[metric_key] = 0

        # 验证时使用原始模型（不带 DDP 包装）
        eval_model = self._model
        scorer_kwargs_eval = self._geometry_scorer_kwargs(self.global_step, training=False)

        pbar = tqdm(self.val_loader, desc='  Validating', ncols=120, leave=False)

        n_batches = 0
        for batch in pbar:
            if batch is None:
                continue
            batch = self._batch_to_device(batch)
            outputs = eval_model(batch, self.global_step, geometry_scorer_kwargs=scorer_kwargs_eval)
            if (
                getattr(self.config, 'lambda_ligand_contrastive', 0.0) > 0.0
                or getattr(self.config, 'lambda_candidate_rerank', 0.0) > 0.0
            ):
                if getattr(self.config, 'lambda_candidate_rerank', 0.0) > 0.0:
                    decoy_kind = getattr(self.config, 'candidate_rerank_decoy_kind', 'translated')
                    if decoy_kind == 'nolig':
                        decoy_batch = self._make_nolig_batch(batch)
                    elif decoy_kind == 'shuffled':
                        decoy_batch = self._make_batch_shuffled_ligand(batch)
                    else:
                        decoy_batch = self._make_translated_batch(
                            batch,
                            offset=float(getattr(self.config, 'g_decoy_translation_offset', 100.0)),
                        )
                else:
                    decoy_batch = self._make_batch_shuffled_ligand(batch)
                if decoy_batch is not None:
                    decoy_outputs = eval_model(decoy_batch, self.global_step, geometry_scorer_kwargs=scorer_kwargs_eval)
                    outputs['candidate_chi1_decoy_logits'] = decoy_outputs.get('candidate_chi1_logits', None)
            if getattr(self.config, 'lambda_g_decoy', 0.0) > 0.0 and outputs.get('geometry_chi1_logits') is not None:
                decoy_kind = getattr(self.config, 'g_decoy_kind', 'translated')
                if decoy_kind == 'nolig':
                    geom_decoy_batch = self._make_nolig_batch(batch)
                elif decoy_kind == 'shuffled':
                    geom_decoy_batch = self._make_batch_shuffled_ligand(batch)
                else:
                    geom_decoy_batch = self._make_translated_batch(
                        batch,
                        offset=float(getattr(self.config, 'g_decoy_translation_offset', 100.0)),
                    )
                geom_decoy_outputs = eval_model(geom_decoy_batch, self.global_step, geometry_scorer_kwargs=scorer_kwargs_eval)
                outputs['geometry_chi1_decoy_logits'] = geom_decoy_outputs.get('geometry_chi1_logits', None)
            if getattr(self.config, 'lambda_typed_candidate_energy', 0.0) > 0.0 and outputs.get('geometry_chi1_typed_energy') is not None:
                typed_decoy_batch = self._make_typed_candidate_decoy_batch(batch)
                typed_decoy_outputs = eval_model(typed_decoy_batch, self.global_step, geometry_scorer_kwargs=scorer_kwargs_eval)
                outputs['geometry_chi1_typed_decoy_energy'] = typed_decoy_outputs.get('geometry_chi1_typed_energy', None)
                outputs['geometry_chi1_decoy_logits'] = typed_decoy_outputs.get('geometry_chi1_logits', None)
            losses = self.compute_loss(outputs, batch, self.global_step)
            n_batches += 1

            for k in val_losses:
                val_losses[k] += losses[k].item()

            pred_chi = torch.atan2(outputs['pred_chi'][..., 0], outputs['pred_chi'][..., 1])
            target_atom14, target_atom14_mask = self._build_atom14_targets(batch)
            pred_atom14 = outputs['atom14_pos']
            pred_atom14_mask = outputs['atom14_mask'].bool() & batch.node_mask.unsqueeze(-1)
            target_atom14_mask = target_atom14_mask.bool() & batch.node_mask.unsqueeze(-1)
            ca_contact_mask_batch = self._compute_ca_ligand_contact_mask(
                batch,
                threshold=float(getattr(self.config, 'g_noncontact_threshold', 8.0)),
            )

            B = batch.Ca_holo.shape[0]
            for i in range(B):
                node_mask_i = batch.node_mask[i].bool()
                chi1_valid_mask = batch.chi_mask[i, :, 0].bool() & node_mask_i
                chi1_errors = compute_angle_errors_deg(
                    pred_chi[i, :, 0],
                    batch.chi_holo[i, :, 0],
                    chi1_valid_mask,
                )
                if chi1_errors.size > 0:
                    metric_stats['chi1_hits'] += float((chi1_errors < 20.0).sum())
                    metric_stats['chi1_total'] += int(chi1_errors.size)

                rotamer_probs = outputs.get('chi1_rotamer_probs', None)
                candidate_probs = outputs.get('candidate_chi1_probs', None)
                geometry_probs = outputs.get('geometry_chi1_probs', None)
                if rotamer_probs is None:
                    rotamer_probs = candidate_probs
                if rotamer_probs is None:
                    rotamer_probs = geometry_probs
                if rotamer_probs is not None:
                    rotamer_pred = rotamer_probs[i].argmax(dim=-1)
                    rotamer_true = self._chi1_rotamer_labels(batch.chi_holo[i, :, 0])
                    rotamer_valid = chi1_valid_mask
                    if rotamer_valid.any():
                        metric_stats['chi1_rotamer_hits'] += float((rotamer_pred[rotamer_valid] == rotamer_true[rotamer_valid]).sum().item())
                        metric_stats['chi1_rotamer_total'] += int(rotamer_valid.sum().item())

                candidate_logits = outputs.get('candidate_chi1_logits', None)
                candidate_decoy_logits = outputs.get('candidate_chi1_decoy_logits', None)
                if candidate_logits is not None and candidate_decoy_logits is not None:
                    candidate_pred = candidate_logits[i].argmax(dim=-1)
                    candidate_decoy_pred = candidate_decoy_logits[i].argmax(dim=-1)
                    holo_bins = self._chi1_rotamer_labels(batch.chi_holo[i, :, 0])
                    apo_bins = self._chi1_rotamer_labels(batch.torsion_apo[i, :, 3])
                    switch_mask = chi1_valid_mask & (apo_bins != holo_bins)
                    contact_mask = chi1_valid_mask & ca_contact_mask_batch[i]
                    pocket_mask = chi1_valid_mask & (batch.w_res[i] > 0.5)
                    candidate_masks = {
                        'switch': switch_mask,
                        'apo_wrong': switch_mask,
                        'contact': contact_mask,
                        'contact_switch': contact_mask & switch_mask,
                        'pocket_switch': pocket_mask & switch_mask,
                    }
                    for lift_key, lift_mask in candidate_masks.items():
                        if not lift_mask.any():
                            continue
                        stats = candidate_lift_stats[lift_key]
                        true_subset = holo_bins[lift_mask]
                        stats['full_hits'] += float((candidate_pred[lift_mask] == true_subset).sum().item())
                        stats['decoy_hits'] += float((candidate_decoy_pred[lift_mask] == true_subset).sum().item())
                        stats['total'] += int(lift_mask.sum().item())

                geometry_logits = outputs.get('geometry_chi1_logits', None)
                geometry_base_logits = outputs.get('geometry_chi1_base_logits', None)
                if geometry_logits is not None and geometry_base_logits is not None:
                    geom_pred = geometry_logits[i].argmax(dim=-1)
                    base_pred = geometry_base_logits[i].argmax(dim=-1)
                    decoy_logits = outputs.get('geometry_chi1_decoy_logits', None)
                    decoy_pred = decoy_logits[i].argmax(dim=-1) if decoy_logits is not None else None
                    holo_bins = self._chi1_rotamer_labels(batch.chi_holo[i, :, 0])
                    apo_bins = self._chi1_rotamer_labels(batch.torsion_apo[i, :, 3])
                    switch_mask = chi1_valid_mask & (apo_bins != holo_bins)
                    contact_mask = chi1_valid_mask & ca_contact_mask_batch[i]
                    pocket_mask = chi1_valid_mask & (batch.w_res[i] > 0.5)
                    lift_masks = {
                        'switch': switch_mask,
                        'apo_wrong': switch_mask,
                        'contact': contact_mask,
                        'contact_switch': contact_mask & switch_mask,
                        'pocket_switch': pocket_mask & switch_mask,
                    }
                    for lift_key, lift_mask in lift_masks.items():
                        if not lift_mask.any():
                            continue
                        stats = ligand_lift_stats[lift_key]
                        true_subset = holo_bins[lift_mask]
                        stats['full_hits'] += float((geom_pred[lift_mask] == true_subset).sum().item())
                        stats['base_hits'] += float((base_pred[lift_mask] == true_subset).sum().item())
                        if decoy_pred is not None:
                            stats['decoy_hits'] += float((decoy_pred[lift_mask] == true_subset).sum().item())
                        stats['total'] += int(lift_mask.sum().item())

                typed_energy = outputs.get('geometry_chi1_typed_energy', None)
                typed_decoy_energy = outputs.get('geometry_chi1_typed_decoy_energy', None)
                if typed_energy is not None and typed_decoy_energy is not None:
                    holo_bins = self._chi1_rotamer_labels(batch.chi_holo[i, :, 0])
                    apo_bins = self._chi1_rotamer_labels(batch.torsion_apo[i, :, 3])
                    switch_mask = chi1_valid_mask & (apo_bins != holo_bins)
                    contact_mask = chi1_valid_mask & ca_contact_mask_batch[i]
                    pocket_mask = chi1_valid_mask & (batch.w_res[i] > 0.5)
                    gather_idx = holo_bins.clamp(min=0).unsqueeze(-1)
                    correct_holo_energy = typed_energy[i].gather(-1, gather_idx).squeeze(-1)
                    decoy_holo_energy = typed_decoy_energy[i].gather(-1, gather_idx).squeeze(-1)
                    typed_masks = {
                        'switch': switch_mask,
                        'contact': contact_mask,
                        'contact_switch': contact_mask & switch_mask,
                        'pocket_switch': pocket_mask & switch_mask,
                    }
                    for typed_key, typed_mask in typed_masks.items():
                        if not typed_mask.any():
                            continue
                        gap = correct_holo_energy[typed_mask] - decoy_holo_energy[typed_mask]
                        stats = typed_energy_stats[typed_key]
                        stats['gap_sum'] += float(gap.sum().item())
                        stats['correct_sum'] += float(correct_holo_energy[typed_mask].sum().item())
                        stats['decoy_sum'] += float(decoy_holo_energy[typed_mask].sum().item())
                        stats['abs_sum'] += float(gap.abs().sum().item())
                        stats['total'] += int(typed_mask.sum().item())

                mask_dict = {
                    'apo_distance_mask': (batch.w_res[i] > 0.5) & node_mask_i,
                }
                if enable_dual_mask_audit:
                    mask_dict['holo_distance_mask'] = self._compute_holo_distance_mask(
                        batch.Ca_holo[i],
                        batch.lig_points[i],
                        batch.lig_mask[i],
                        node_mask_i,
                    )
                    mask_dict['ligand_contact_mask'] = self._compute_ligand_contact_mask(
                        target_atom14[i],
                        target_atom14_mask[i],
                        batch.lig_points[i],
                        batch.lig_mask[i],
                        node_mask_i,
                    )

                contact_probs = outputs.get('contact_probs', None)
                if contact_probs is not None:
                    if 'ligand_contact_mask' in mask_dict:
                        true_contact_t = mask_dict['ligand_contact_mask']
                    else:
                        true_contact_t = self._compute_ligand_contact_mask(
                            target_atom14[i],
                            target_atom14_mask[i],
                            batch.lig_points[i],
                            batch.lig_mask[i],
                            node_mask_i,
                        )
                    pred_contact_t = (contact_probs[i] >= 0.5) & node_mask_i
                    metric_stats['contact_post_tp'] += int((pred_contact_t & true_contact_t).sum().item())
                    metric_stats['contact_post_fp'] += int((pred_contact_t & (~true_contact_t) & node_mask_i).sum().item())
                    metric_stats['contact_post_fn'] += int(((~pred_contact_t) & true_contact_t).sum().item())

                for metric_key, mask_name in mask_items:
                    residue_mask = mask_dict[mask_name]
                    chi1_mask = chi1_valid_mask & residue_mask
                    chi1_subset_errors = compute_angle_errors_deg(
                        pred_chi[i, :, 0],
                        batch.chi_holo[i, :, 0],
                        chi1_mask,
                    )
                    if chi1_subset_errors.size > 0:
                        chi_hits[metric_key] += float((chi1_subset_errors < 20.0).sum())
                        chi_totals[metric_key] += int(chi1_subset_errors.size)
                        angle_buffers[metric_key].append(chi1_subset_errors)

                    chi12_mask = batch.chi_mask[i, :, :2].bool() & residue_mask.unsqueeze(-1)
                    valid_chi12_count = int((chi12_mask[:, 0] & chi12_mask[:, 1]).sum().item())
                    if valid_chi12_count > 0:
                        chi12_acc = compute_chi12_accuracy(
                            pred_chi[i, :, :2],
                            batch.chi_holo[i, :, :2],
                            chi12_mask,
                        )
                        if not math.isnan(chi12_acc):
                            chi12_hits[metric_key] += float(chi12_acc * valid_chi12_count)
                            chi12_totals[metric_key] += valid_chi12_count

                    if compute_slow_metrics:
                        lig_coords = self._valid_ligand_coords(batch.lig_points[i], batch.lig_mask[i])
                        if lig_coords.shape[0] > 0:
                            pred_contact_mask = compute_residue_contact_mask(
                                pred_atom14[i],
                                pred_atom14_mask[i],
                                lig_coords,
                            )
                            true_contact_mask = compute_residue_contact_mask(
                                target_atom14[i],
                                target_atom14_mask[i],
                                lig_coords,
                            )
                            subset_np = residue_mask.detach().cpu().numpy().astype(bool)
                            pred_subset = pred_contact_mask & subset_np
                            true_subset = true_contact_mask & subset_np
                            tp = int(np.logical_and(pred_subset, true_subset).sum())
                            fp = int(np.logical_and(pred_subset, ~true_subset).sum())
                            fn = int(np.logical_and(~pred_subset, true_subset).sum())
                            union = int(np.logical_or(pred_subset, true_subset).sum())
                            contact_confusion[metric_key]['tp'] += tp
                            contact_confusion[metric_key]['fp'] += fp
                            contact_confusion[metric_key]['fn'] += fn
                            contact_confusion[metric_key]['union'] += union

                            irmsd = compute_pocket_irmsd(
                                pred_atom14[i, :, 1],
                                batch.Ca_holo[i],
                                subset_np,
                            )
                            if not math.isnan(irmsd):
                                irmsd_sums[metric_key] += irmsd
                                irmsd_counts[metric_key] += 1

                            pocket_clash_pct = compute_pocket_clash_percentage(
                                pred_atom14[i],
                                pred_atom14_mask[i],
                                residue_mask,
                            )
                            if not math.isnan(pocket_clash_pct):
                                pocket_clash_sums[metric_key] += pocket_clash_pct
                                pocket_clash_counts[metric_key] += 1

                if compute_slow_metrics:
                    valid_atom_mask = pred_atom14_mask[i].reshape(-1)
                    coords_i = pred_atom14[i].reshape(-1, 3)[valid_atom_mask]
                    if coords_i.shape[0] > 1:
                        clash_pct = compute_clash_percentage(coords_i)
                        metric_stats['clash_pct_sum'] += clash_pct
                        metric_stats['clash_pct_count'] += 1

            chi1_acc = self._nanmean_from_sum_count(metric_stats['chi1_hits'], metric_stats['chi1_total'])

            pbar.set_postfix({
                'v_loss': f"{losses['total'].item():.3f}",
                'chi1': f"{chi1_acc:5.1%}",
            }, refresh=False)

        # DDP: 通知其他进程验证完成
        if self.distributed:
            dist.barrier()

        if n_batches == 0:
            return {
                'total': float('nan'),
                'chi': float('nan'),
                'pchi1': float('nan'),
                'chi1_rotamer': float('nan'),
                'contact': float('nan'),
                'typed_candidate_energy': float('nan'),
                'clash': float('nan'),
                'fape': float('nan'),
                'chi1_acc': float('nan'),
                'pocket_chi1_acc': float('nan'),
                'pocket_chi1_mae_deg': float('nan'),
                'pocket_chi1_median_deg': float('nan'),
                'pocket_chi12_acc': float('nan'),
                'pocket_contact_precision': float('nan'),
                'pocket_contact_recall': float('nan'),
                'pocket_contact_f1': float('nan'),
                'pocket_contact_iou': float('nan'),
                'pocket_irmsd': float('nan'),
                'clash_pct': float('nan'),
                'pocket_clash_pct': float('nan'),
                'chi1_rotamer_acc': float('nan'),
                'contact_posterior_precision': float('nan'),
                'contact_posterior_recall': float('nan'),
                'contact_posterior_f1': float('nan'),
            }

        val_losses = {k: v / n_batches for k, v in val_losses.items()}
        val_metrics = {
            'chi1_acc': self._nanmean_from_sum_count(metric_stats['chi1_hits'], metric_stats['chi1_total']),
            'chi1_rotamer_acc': self._nanmean_from_sum_count(metric_stats['chi1_rotamer_hits'], metric_stats['chi1_rotamer_total']),
            'clash_pct': self._nanmean_from_sum_count(metric_stats['clash_pct_sum'], metric_stats['clash_pct_count']) if compute_slow_metrics else float('nan'),
        }
        for lift_key, stats in ligand_lift_stats.items():
            full_acc = self._nanmean_from_sum_count(stats['full_hits'], stats['total'])
            base_acc = self._nanmean_from_sum_count(stats['base_hits'], stats['total'])
            decoy_acc = self._nanmean_from_sum_count(stats['decoy_hits'], stats['total'])
            prefix = f'ligand_lift_{lift_key}_rotamer'
            val_metrics[f'{prefix}_n'] = float(stats['total'])
            val_metrics[f'{prefix}_full_acc'] = full_acc
            val_metrics[f'{prefix}_base_acc'] = base_acc
            val_metrics[f'{prefix}_acc'] = full_acc - base_acc if math.isfinite(full_acc) and math.isfinite(base_acc) else float('nan')
            val_metrics[f'{prefix}_decoy_acc'] = decoy_acc
            val_metrics[f'ligand_decoy_lift_{lift_key}_rotamer_acc'] = (
                full_acc - decoy_acc if math.isfinite(full_acc) and math.isfinite(decoy_acc) else float('nan')
            )
        for lift_key, stats in candidate_lift_stats.items():
            full_acc = self._nanmean_from_sum_count(stats['full_hits'], stats['total'])
            decoy_acc = self._nanmean_from_sum_count(stats['decoy_hits'], stats['total'])
            prefix = f'candidate_decoy_lift_{lift_key}_rotamer'
            val_metrics[f'{prefix}_n'] = float(stats['total'])
            val_metrics[f'{prefix}_full_acc'] = full_acc
            val_metrics[f'{prefix}_decoy_acc'] = decoy_acc
            val_metrics[f'{prefix}_acc'] = (
                full_acc - decoy_acc if math.isfinite(full_acc) and math.isfinite(decoy_acc) else float('nan')
            )
        for typed_key, stats in typed_energy_stats.items():
            n_typed = stats['total']
            val_metrics[f'typed_energy_gap_{typed_key}_n'] = float(n_typed)
            val_metrics[f'typed_energy_gap_{typed_key}'] = self._nanmean_from_sum_count(stats['gap_sum'], n_typed)
            val_metrics[f'typed_energy_correct_{typed_key}'] = self._nanmean_from_sum_count(stats['correct_sum'], n_typed)
            val_metrics[f'typed_energy_decoy_{typed_key}'] = self._nanmean_from_sum_count(stats['decoy_sum'], n_typed)
            val_metrics[f'typed_energy_abs_gap_{typed_key}'] = self._nanmean_from_sum_count(stats['abs_sum'], n_typed)
        post_precision = float(metric_stats['contact_post_tp'] / max(metric_stats['contact_post_tp'] + metric_stats['contact_post_fp'], 1))
        post_recall = float(metric_stats['contact_post_tp'] / max(metric_stats['contact_post_tp'] + metric_stats['contact_post_fn'], 1))
        val_metrics['contact_posterior_precision'] = post_precision
        val_metrics['contact_posterior_recall'] = post_recall
        val_metrics['contact_posterior_f1'] = float(2 * post_precision * post_recall / max(post_precision + post_recall, 1e-8))

        for metric_key, _ in mask_items:
            prefix = metric_key
            val_metrics[f'{prefix}_chi1_acc'] = self._nanmean_from_sum_count(chi_hits[metric_key], chi_totals[metric_key])
            if angle_buffers[metric_key]:
                errors = np.concatenate(angle_buffers[metric_key])
                val_metrics[f'{prefix}_chi1_mae_deg'] = float(errors.mean())
                val_metrics[f'{prefix}_chi1_median_deg'] = float(np.median(errors))
            else:
                val_metrics[f'{prefix}_chi1_mae_deg'] = float('nan')
                val_metrics[f'{prefix}_chi1_median_deg'] = float('nan')

            val_metrics[f'{prefix}_chi12_acc'] = self._nanmean_from_sum_count(chi12_hits[metric_key], chi12_totals[metric_key])

            if compute_slow_metrics:
                confusion = contact_confusion[metric_key]
                precision = float(confusion['tp'] / max(confusion['tp'] + confusion['fp'], 1))
                recall = float(confusion['tp'] / max(confusion['tp'] + confusion['fn'], 1))
                f1 = float(2 * precision * recall / max(precision + recall, 1e-8))
                iou = float(confusion['tp'] / max(confusion['union'], 1))
                val_metrics[f'{prefix}_contact_precision'] = precision
                val_metrics[f'{prefix}_contact_recall'] = recall
                val_metrics[f'{prefix}_contact_f1'] = f1
                val_metrics[f'{prefix}_contact_iou'] = iou
                val_metrics[f'{prefix}_irmsd'] = self._nanmean_from_sum_count(irmsd_sums[metric_key], irmsd_counts[metric_key])
                val_metrics[f'{prefix}_clash_pct'] = self._nanmean_from_sum_count(pocket_clash_sums[metric_key], pocket_clash_counts[metric_key])
            else:
                val_metrics[f'{prefix}_contact_precision'] = float('nan')
                val_metrics[f'{prefix}_contact_recall'] = float('nan')
                val_metrics[f'{prefix}_contact_f1'] = float('nan')
                val_metrics[f'{prefix}_contact_iou'] = float('nan')
                val_metrics[f'{prefix}_irmsd'] = float('nan')
                val_metrics[f'{prefix}_clash_pct'] = float('nan')

        return {**val_losses, **val_metrics}

    def save_checkpoint(self, filepath: str, verbose: bool = True):
        """保存 checkpoint（只在主进程执行）"""
        if not self.is_main_process:
            return
            
        # DDP 模式下保存 .module 的状态
        model_state = self._model.state_dict()
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'selection_metric': self.config.selection_metric,
            'config': self.config,
        }, filepath)
        if verbose:
            print(f"  ✓ Saved: {Path(filepath).name}")

    def train(self):
        if self.is_main_process:
            print(f"\n{'='*80}")
            print("Start training - Stage-1")
            if self.distributed:
                print(f"  [DDP] {self.world_size} GPUs, effective batch size: {self.config.batch_size * self.world_size}")
            print(f"{'='*80}\n")

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch

            train_losses = self.train_epoch()
            
            # DDP: 同步后再验证
            if self.distributed:
                dist.barrier()
            
            train_info = (
                f"Epoch {epoch:3d} | "
                f"Loss: {train_losses['total']:.4f} "
                f"(Chi:{train_losses['chi']:.2f} "
                f"PChi1:{train_losses['pchi1']:.2f} "
                f"Rot:{train_losses['chi1_rotamer']:.2f} "
                f"Ct:{train_losses['contact']:.2f} "
                f"Cand:{train_losses['candidate_chi1']:.2f} "
                f"LCon:{train_losses['ligand_contrastive']:.2f} "
                f"CRnk:{train_losses['candidate_rerank']:.2f} "
                f"Typ:{train_losses['typed_candidate_energy']:.2f} "
                f"Exp:{train_losses['expert_delta']:.2f} "
                f"F:{train_losses['fape']:.2f} "
                f"C:{train_losses['clash']:.2f} "
                f"BP:{train_losses['base_prior']:.2f} "
                f"GL:{train_losses['g_lift_switch']:.2f} "
                f"GN:{train_losses['g_noharm']:.2f} "
                f"GZ:{train_losses['g_zero_noncontact']:.3f} "
                f"GD:{train_losses['g_switch_dir']:.3f} "
                f"GA:{train_losses['g_switch_amp']:.3f} "
                f"GR:{train_losses['g_switch_rank']:.3f} "
                f"GH:{train_losses['g_antiharm']:.3f} "
                f"GX:{train_losses['g_decoy']:.3f})"
            )

            if epoch % self.config.val_interval == 0:
                val_results = self.validate()

                # DDP: 验证后同步
                if self.distributed:
                    dist.barrier()
                should_stop = False
                if self.is_main_process:
                    pocket_irmsd_str = f"{val_results['pocket_irmsd']:.2f}" if math.isfinite(val_results['pocket_irmsd']) else 'NA'
                    clash_pct_str = f"{val_results['clash_pct']*100:.1f}%" if math.isfinite(val_results['clash_pct']) else 'NA'
                    val_info = (
                        f" | Val Loss: {val_results['total']:.4f} "
                        f"chi1:{val_results['chi1_acc']:5.1%} "
                        f"FAPE:{val_results['fape']:.3f} "
                        f"iRMSD:{pocket_irmsd_str} "
                        f"Clash:{clash_pct_str}"
                    )

                    current_metric = self._extract_selection_metric(val_results)
                    if self.config.selection_metric != 'total':
                        val_info += f" Sel[{self.config.selection_metric}]={current_metric:.4f}"
                    scheduler_metric = val_results['total'] if math.isfinite(val_results['total']) else float('inf')
                    if self.lr_scheduler_name == 'plateau':
                        metric_tensor = torch.tensor([scheduler_metric], device=self.device)
                        if self.distributed:
                            dist.broadcast(metric_tensor, src=0)
                        self.scheduler.step(float(metric_tensor.item()))
                    if self.config.save_latest_checkpoint:
                        latest_path = Path(self.config.save_dir) / 'latest_model.pt'
                        self.save_checkpoint(str(latest_path), verbose=False)
                    if self.config.save_epoch_checkpoints:
                        epoch_path = Path(self.config.save_dir) / f'epoch_{epoch:03d}.pt'
                        self.save_checkpoint(str(epoch_path), verbose=False)

                    record = {
                        'epoch': epoch,
                        'global_step': self.global_step,
                        'selection_metric_name': self.config.selection_metric,
                        'selection_metric_value': current_metric,
                        'best_val_metric': self.best_val_metric,
                        'lr': self._current_lr(),
                    }
                    record.update({f'train_{k}': v for k, v in train_losses.items()})
                    record.update({f'val_{k}': v for k, v in val_results.items()})

                    is_best = self._is_better_metric(current_metric, self.best_val_metric)
                    if is_best:
                        self.best_val_metric = current_metric
                        self.patience_counter = 0
                        save_path = Path(self.config.save_dir) / 'best_model.pt'
                        self.save_checkpoint(str(save_path), verbose=False)
                        val_info += " | ⭐ Best"
                    else:
                        self.patience_counter += 1
                        val_info += f" | Patience {self.patience_counter}/{self.config.early_stop_patience}"

                    record['best_val_metric'] = self.best_val_metric
                    record['patience_counter'] = self.patience_counter
                    record['is_best'] = is_best
                    self._append_metrics_record(record)
                    if self.config.compute_slow_metrics or self.config.enable_dual_mask_audit:
                        audit_payload = {
                            'epoch': epoch,
                            'global_step': self.global_step,
                            'selection_metric_name': self.config.selection_metric,
                            'selection_metric_value': current_metric,
                            'best_val_metric': self.best_val_metric,
                            'is_best': is_best,
                            'lr': self._current_lr(),
                            'val_metrics': val_results,
                        }
                        self._write_audit_snapshot(audit_payload)

                    val_info += f" | LR {self._current_lr():.2e}"
                    print(train_info + val_info)

                    if self.patience_counter >= self.config.early_stop_patience:
                        print("Early stopping triggered")
                        should_stop = True

                elif self.lr_scheduler_name == 'plateau':
                    metric_tensor = torch.tensor([val_results['total']], device=self.device)
                    dist.broadcast(metric_tensor, src=0)
                    scheduler_metric = float(metric_tensor.item())
                    if not math.isfinite(scheduler_metric):
                        scheduler_metric = float('inf')
                    self.scheduler.step(scheduler_metric)

                if self.distributed:
                    stop_flag = torch.tensor([1 if should_stop else 0], device=self.device)
                    dist.broadcast(stop_flag, src=0)
                    should_stop = stop_flag.item() > 0

                if should_stop:
                    break
            else:
                if self.is_main_process:
                    print(train_info)
        
        # DDP: 训练结束时清理
        if self.distributed:
            dist.destroy_process_group()
