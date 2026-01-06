"""
Stage-1 trainer (apo + ligand -> holo-like prior).
"""

import math
import time
from pathlib import Path
from typing import Dict

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainingConfig
from ..models import Stage1Model, Stage1ModelConfig
from ..datasets import create_stage1_dataloader
from ..modules.losses import fape_loss, torsion_loss, clash_penalty
from utils.metrics import compute_pocket_irmsd, compute_chi1_accuracy, compute_clash_percentage

# FlashIPA Rigid (与 FK 模块一致)
import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
flash_ipa_path = str(project_root / 'vendor' / 'flash_ipa' / 'src')
if os.path.exists(flash_ipa_path) and flash_ipa_path not in sys.path:
    sys.path.insert(0, flash_ipa_path)
from flash_ipa.rigid import Rigid, Rotation


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
    """Stage-1 trainer."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # Model
        print("Creating model...")
        model_config = Stage1ModelConfig()
        self.model = Stage1Model(model_config).to(self.device)

        # Data
        print("Creating dataloaders...")
        self.train_loader = create_stage1_dataloader(
            config.data_dir,
            split='train',
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            max_n_res=config.max_n_res,
            require_atom14=False,  # atom14_holo 可选，缺失时跳过 FAPE
        )
        self.val_loader = create_stage1_dataloader(
            config.data_dir,
            split='val',
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            max_n_res=config.max_n_res,
            require_atom14=False,
        )

        # Optimizer
        print("Creating optimizer...")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # LR scheduler
        total_steps = len(self.train_loader) * config.max_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - config.warmup_steps, 1),
            eta_min=config.lr * 0.01,
        )

        self.scaler = GradScaler() if config.mixed_precision else None

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf')
        self.patience_counter = 0

        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        print("✓ Trainer initialized")
        print(f"  - params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  - train samples: {len(self.train_loader.dataset)}")
        print(f"  - val samples: {len(self.val_loader.dataset)}")
        print(f"  - total steps: {total_steps:,}")

    def compute_pocket_warmup(self, step: int) -> float:
        if step >= self.config.pocket_warmup_steps:
            return 1.0
        return step / max(self.config.pocket_warmup_steps, 1)

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

    def _generate_atom14_holo_with_fk(self, batch) -> torch.Tensor:
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
        
        # 3. 从 sequences 生成 aatype
        B, N = batch.torsion_holo.shape[:2]
        aatype = sequences_to_aatype(batch.sequences, N, self.device)
        
        # 4. 调用 FK 生成 atom14
        fk_result = self.model.fk_module(torsion_sincos, holo_rigids, aatype)
        
        return fk_result['atom14_pos'], fk_result['atom14_mask']

    def compute_loss(self, outputs: Dict, batch, step: int) -> Dict[str, torch.Tensor]:
        pred_chi_sincos = outputs['pred_chi']  # [B,N,4,2]
        pred_chi = torch.atan2(pred_chi_sincos[..., 0], pred_chi_sincos[..., 1])

        # Pocket warmup for chi supervision
        kappa = self.compute_pocket_warmup(step)
        w_res_warmed = batch.w_res * kappa + (1 - kappa) * 0.1

        loss_chi = torsion_loss(
            pred_chi,
            batch.chi_holo,
            batch.chi_mask,
            w_res_warmed,
        )

        # FAPE on atom14 (if available)
        pred_atom14 = outputs['atom14_pos']
        pred_R = outputs['rigids_final'].get_rots().get_rot_mats()
        pred_t = pred_atom14[:, :, 1]

        true_R, true_t = self._build_frames_from_backbone(
            batch.N_holo,
            batch.Ca_holo,
            batch.C_holo,
            batch.node_mask,
        )

        # FAPE loss: 用 FK 动态生成 atom14_holo 真值（保证一致性）
        # 如果原始 atom14_holo 为空或全零，就用 FK 生成
        atom14_holo_valid = (
            batch.atom14_holo is not None and 
            batch.atom14_holo.abs().sum() > 0
        )
        
        if atom14_holo_valid:
            # 使用原始 atom14_holo 数据
            target_atom14 = batch.atom14_holo
        else:
            # 用 FK 从 holo backbone + torsion_holo 动态生成
            target_atom14, _ = self._generate_atom14_holo_with_fk(batch)
        
        fape_mask = batch.node_mask.float() if batch.node_mask is not None else None
        loss_fape = fape_loss(
            pred_atom14,
            target_atom14,
            (pred_R, pred_t),
            (true_R, true_t),
            w_res=fape_mask,
        )

        # Clash on predicted atoms
        B, N, A, _ = pred_atom14.shape
        pred_all_atoms = pred_atom14.reshape(B, -1, 3)
        if batch.node_mask is not None:
            valid_atom_mask = batch.node_mask.unsqueeze(-1).expand(-1, -1, A).reshape(B, -1)
            pred_all_atoms = pred_all_atoms.masked_fill(~valid_atom_mask.unsqueeze(-1), 1e6)
        loss_clash = clash_penalty(pred_all_atoms, clash_threshold=2.2)

        total_loss = (
            self.config.w_fape * loss_fape +
            self.config.w_chi * loss_chi +
            self.config.w_clash * loss_clash
        )

        return {
            'total': total_loss,
            'chi': loss_chi,
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

        if self.scaler is not None:
            with autocast():
                outputs = self.model(batch, self.global_step)
                losses = self.compute_loss(outputs, batch, self.global_step)
                loss = losses['total']

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(batch, self.global_step)
            losses = self.compute_loss(outputs, batch, self.global_step)
            loss = losses['total']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

        if self.global_step >= self.config.warmup_steps:
            self.scheduler.step()

        self.global_step += 1

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

        epoch_losses = {
            'total': 0.0,
            'chi': 0.0,
            'clash': 0.0,
            'fape': 0.0,
        }

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch:3d}', ncols=120, leave=True)
        n_batches = 0
        for batch in pbar:
            if batch is None:
                continue
            step_losses = self.train_step(batch)
            for k in epoch_losses:
                epoch_losses[k] += step_losses[k]
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{step_losses['total']:.3f}",
                'chi': f"{step_losses['chi']:.3f}",
                'fape': f"{step_losses['fape']:.3f}",
                'clash': f"{step_losses['clash']:.3f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.1e}",
            }, refresh=True)

        n_batches = max(n_batches, 1)
        return {k: v / n_batches for k, v in epoch_losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()

        val_losses = {
            'total': 0.0,
            'chi': 0.0,
            'clash': 0.0,
            'fape': 0.0,
        }

        val_metrics = {
            'chi1_acc': 0.0,
        }
        pocket_irmsd_sum = 0.0
        clash_pct_sum = 0.0
        n_structures = 0

        pbar = tqdm(self.val_loader, desc='  Validating', ncols=120, leave=False)
        n_batches = 0
        for batch in pbar:
            if batch is None:
                continue
            batch = self._batch_to_device(batch)
            outputs = self.model(batch, self.global_step)
            losses = self.compute_loss(outputs, batch, self.global_step)
            n_batches += 1

            for k in val_losses:
                val_losses[k] += losses[k].item()

            pred_chi = torch.atan2(outputs['pred_chi'][..., 0], outputs['pred_chi'][..., 1])
            chi1_acc = compute_chi1_accuracy(
                pred_chi[:, :, 0].flatten(),
                batch.chi_holo[:, :, 0].flatten(),
                batch.chi_mask[:, :, 0].flatten(),
            )
            val_metrics['chi1_acc'] += chi1_acc

            B = batch.Ca_holo.shape[0]
            for i in range(B):
                pocket_mask = (batch.w_res[i] > 0.5)
                if pocket_mask.any():
                    irmsd = compute_pocket_irmsd(
                        outputs['atom14_pos'][i, :, 1],
                        batch.Ca_holo[i],
                        pocket_mask,
                    )
                    if not math.isnan(irmsd):
                        pocket_irmsd_sum += irmsd

                valid_atom_mask = outputs['atom14_mask'][i].bool().view(-1)
                coords_i = outputs['atom14_pos'][i].view(-1, 3)[valid_atom_mask]
                if coords_i.shape[0] > 1:
                    clash_pct = compute_clash_percentage(coords_i)
                    clash_pct_sum += clash_pct
                n_structures += 1

            pbar.set_postfix({
                'v_loss': f"{losses['total'].item():.3f}",
                'chi1': f"{chi1_acc:5.1%}",
            }, refresh=False)

        if n_batches == 0:
            return {
                'total': float('nan'),
                'chi': float('nan'),
                'clash': float('nan'),
                'fape': float('nan'),
                'chi1_acc': float('nan'),
                'pocket_irmsd': float('nan'),
                'clash_pct': float('nan'),
            }

        val_losses = {k: v / n_batches for k, v in val_losses.items()}
        val_metrics['chi1_acc'] = val_metrics['chi1_acc'] / n_batches
        if n_structures > 0:
            val_metrics['pocket_irmsd'] = pocket_irmsd_sum / n_structures
            val_metrics['clash_pct'] = clash_pct_sum / n_structures
        else:
            val_metrics['pocket_irmsd'] = float('nan')
            val_metrics['clash_pct'] = float('nan')

        return {**val_losses, **val_metrics}

    def save_checkpoint(self, filepath: str, verbose: bool = True):
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config,
        }, filepath)
        if verbose:
            print(f"  ✓ Saved: {Path(filepath).name}")

    def train(self):
        print(f"\n{'='*80}")
        print("Start training - Stage-1")
        print(f"{'='*80}\n")

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch

            train_losses = self.train_epoch()
            train_info = (
                f"Epoch {epoch:3d} | "
                f"Loss: {train_losses['total']:.4f} "
                f"(Chi:{train_losses['chi']:.2f} "
                f"F:{train_losses['fape']:.2f} "
                f"C:{train_losses['clash']:.2f})"
            )

            if epoch % self.config.val_interval == 0:
                val_results = self.validate()

                val_info = (
                    f" | Val Loss: {val_results['total']:.4f} "
                    f"chi1:{val_results['chi1_acc']:5.1%} "
                    f"FAPE:{val_results['fape']:.3f} "
                    f"iRMSD:{val_results['pocket_irmsd']:.2f} "
                    f"Clash:{val_results['clash_pct']*100:.1f}%"
                )

                current_metric = val_results['total']
                if current_metric < self.best_val_metric:
                    self.best_val_metric = current_metric
                    self.patience_counter = 0
                    save_path = Path(self.config.save_dir) / 'best_model.pt'
                    self.save_checkpoint(str(save_path), verbose=False)
                    val_info += " | ⭐ Best"
                else:
                    self.patience_counter += 1
                    val_info += f" | Patience {self.patience_counter}/{self.config.early_stop_patience}"

                print(train_info + val_info)

                if self.patience_counter >= self.config.early_stop_patience:
                    print("Early stopping triggered")
                    break
            else:
                print(train_info)
