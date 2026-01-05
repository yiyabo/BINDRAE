"""Stage-2 trainer (bridge flow on apo->holo paths)."""

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import sys
import os

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
from src.stage1.models.fk_openfold import create_openfold_fk
from src.stage1.modules.losses import clash_penalty, fape_loss


class Stage2Trainer:
    """Stage-2 trainer."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        if config.use_nma and config.nma_dim <= 0:
            raise ValueError("use_nma=True but nma_dim <= 0")
        if not config.use_nma and config.nma_dim > 0:
            raise ValueError("nma_dim > 0 but use_nma is False")

        # Model
        print("Creating Stage-2 model...")
        model_config = TorsionFlowNetConfig(nma_dim=config.nma_dim)
        self.model = TorsionFlowNet(model_config).to(self.device)

        # Stage-1 prior model
        self.stage1_model = None
        if config.use_stage1_prior:
            if not config.stage1_ckpt:
                raise ValueError("stage1_ckpt must be provided when use_stage1_prior=True")
            self.stage1_model = self._load_stage1_model(config.stage1_ckpt)

        # FK module
        self.fk_module = create_openfold_fk().to(self.device)

        # Data
        print("Creating dataloaders...")
        self.train_loader = create_stage2_dataloader(
            config.data_dir,
            split='train',
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            require_nma=config.use_nma,
        )
        self.val_loader = create_stage2_dataloader(
            config.data_dir,
            split='val',
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            require_nma=config.use_nma,
        )

        # Optimizer
        print("Creating optimizer...")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

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

        print("✓ Stage-2 Trainer initialized")
        print(f"  - params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  - train samples: {len(self.train_loader.dataset)}")
        print(f"  - val samples: {len(self.val_loader.dataset)}")
        print(f"  - total steps: {total_steps:,}")

    def _load_stage1_model(self, ckpt_path: str) -> Stage1Model:
        model_config = Stage1ModelConfig()
        model = Stage1Model(model_config).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=False)
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

    def _compute_stage1_outputs(self, batch) -> Tuple[torch.Tensor, Rigid]:
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

        return chi_ref, rigids_ref, d_chi_ref, d_rot_ref, d_trans_ref, rigids_apo, rigids_holo

    def integrate_path(self,
                       batch,
                       rigids0: Rigid,
                       chi0: torch.Tensor,
                       stage1_chi=None,
                       stage1_rigids=None) -> Tuple[List[Rigid], List[torch.Tensor], List[float]]:
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
            out1 = self.model(
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
                current_step=self.global_step,
            )

            d_chi1 = out1['d_chi']
            d_rot1 = out1['d_rigid_rot']
            d_trans1 = out1['d_rigid_trans']

            # Predictor (Euler)
            chi_pred = wrap_to_pi(chi + dt * d_chi1)
            xi1 = torch.cat([d_rot1, d_trans1], dim=-1) * dt
            R_inc1, t_inc1 = se3_exp(xi1)
            R_curr, t_curr = self._rigid_to_rt(rigids)
            R_pred, t_pred = rigid_compose(R_curr, t_curr, R_inc1, t_inc1)
            rigids_pred = self._rt_to_rigid(R_pred, t_pred)

            # Step 2: velocity at predicted state
            out2 = self.model(
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
                current_step=self.global_step,
            )

            d_chi2 = out2['d_chi']
            d_rot2 = out2['d_rigid_rot']
            d_trans2 = out2['d_rigid_trans']

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

    def compute_losses(self, batch, t: torch.Tensor) -> Dict[str, torch.Tensor]:
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

        stage1_chi = None
        stage1_rigids = None
        if self.config.use_stage1_prior:
            stage1_chi, stage1_rigids = self._compute_stage1_outputs(batch)

        # Predict velocities at reference state
        out = self.model(
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
            current_step=self.global_step,
        )

        d_chi_pred = out['d_chi']
        d_rot_pred = out['d_rigid_rot']
        d_trans_pred = out['d_rigid_trans']

        # FM loss
        chi_mask = batch.chi_mask.float()
        fm_chi = ((d_chi_pred - d_chi_ref) ** 2) * w_pow.unsqueeze(-1) * chi_mask
        fm_chi_denom = (chi_mask * w_pow.unsqueeze(-1)).sum().clamp(min=1e-8)
        L_fm_chi = fm_chi.sum() / fm_chi_denom

        fm_rot = ((d_rot_pred - d_rot_ref) ** 2) * w_pow
        fm_trans = ((d_trans_pred - d_trans_ref) ** 2) * w_pow
        L_fm_rigid = (fm_rot.sum() + fm_trans.sum()) / (w_pow.sum() + 1e-8)

        # Background stability
        bg_w = (1.0 - w_eff).clamp(min=0.0) ** self.config.bg_beta
        L_bg = (
            (bg_w * (d_rot_pred ** 2).sum(dim=-1)).sum() +
            (bg_w * (d_trans_pred ** 2).sum(dim=-1)).sum() +
            (bg_w.unsqueeze(-1) * (d_chi_pred ** 2) * chi_mask).sum()
        ) / (bg_w.sum() + 1e-8)

        # Integrate path for geometry
        rigids_list, chi_list, t_list = self.integrate_path(
            batch,
            rigids_apo,
            batch.torsion_apo[..., 3:7],
            stage1_chi=stage1_chi,
            stage1_rigids=stage1_rigids,
        )

        # Select geometry steps
        n_geom = min(self.config.n_geom_steps, len(t_list))
        geom_indices = torch.linspace(0, len(t_list) - 1, steps=n_geom).long().tolist()

        L_smooth = chi_ref.new_tensor(0.0)
        L_clash = chi_ref.new_tensor(0.0)
        L_pep = chi_ref.new_tensor(0.0)
        L_contact = chi_ref.new_tensor(0.0)
        L_prior = chi_ref.new_tensor(0.0)

        contact_scores = []

        # Precompute holo atom14 for endpoint FAPE
        torsion_holo = batch.torsion_holo
        phi_psi_omega_holo = torsion_holo[..., :3]
        phi_psi_omega_sincos = torch.stack(
            [torch.sin(phi_psi_omega_holo), torch.cos(phi_psi_omega_holo)], dim=-1
        )
        chi_holo = torsion_holo[..., 3:7]
        chi_holo_sincos = torch.stack([torch.sin(chi_holo), torch.cos(chi_holo)], dim=-1)
        torsions_holo_sincos = torch.cat([phi_psi_omega_sincos, chi_holo_sincos], dim=2)
        atom14_holo = self.fk_module(torsions_holo_sincos, rigids_holo, batch.aatype)

        # Geometry losses along path
        prev_R, prev_t = None, None
        prev_chi = None
        prev_C = None

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
            torsions_sincos = torch.cat([phi_psi_omega_sincos, chi_sincos], dim=2)
            atom14 = self.fk_module(torsions_sincos, rigids_t, batch.aatype)

            atom14_pos = atom14['atom14_pos']
            atom14_mask = atom14['atom14_mask'].bool()

            # Smoothness (between consecutive geometry steps)
            R_t, t_t = self._rigid_to_rt(rigids_t)
            if prev_R is not None:
                R_inv, t_inv = rigid_inverse(prev_R, prev_t)
                R_delta, t_delta = rigid_compose(R_inv, t_inv, R_t, t_t)
                xi_delta = se3_log(R_delta, t_delta)
                L_smooth = L_smooth + ((xi_delta ** 2).sum(dim=-1) * w_pow).sum() / (w_pow.sum() + 1e-8)

                d_chi = wrap_to_pi(chi_t - prev_chi)
                chi_smooth = ((d_chi ** 2) * chi_mask) * w_pow.unsqueeze(-1)
                chi_smooth_denom = (chi_mask * w_pow.unsqueeze(-1)).sum().clamp(min=1e-8)
                L_smooth = L_smooth + chi_smooth.sum() / chi_smooth_denom

            prev_R, prev_t = R_t, t_t
            prev_chi = chi_t

            # Clash (mask invalid atoms to avoid padded clashes)
            valid_atom = atom14_mask & batch.node_mask.unsqueeze(-1)
            atom14_pos_masked = atom14_pos.masked_fill(~valid_atom.unsqueeze(-1), 1e6)
            flat_atoms = atom14_pos_masked.reshape(atom14_pos.shape[0], -1, 3)
            L_clash = L_clash + clash_penalty(flat_atoms, clash_threshold=2.2)

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

            # Prior (late time)
            if stage1_chi is not None and t_val >= self.config.t_mid:
                d_chi_prior = wrap_to_pi(chi_t - stage1_chi)
                prior_term = ((d_chi_prior ** 2) * chi_mask) * w_pow.unsqueeze(-1)
                prior_denom = (chi_mask * w_pow.unsqueeze(-1)).sum().clamp(min=1e-8)
                L_prior = L_prior + prior_term.sum() / prior_denom

                R_t, t_t = self._rigid_to_rt(rigids_t)
                R1, t1 = self._rigid_to_rt(stage1_rigids)
                R_inv, t_inv = rigid_inverse(R_t, t_t)
                R_delta, t_delta = rigid_compose(R_inv, t_inv, R1, t1)
                xi_prior = se3_log(R_delta, t_delta)
                prior_rigid = ((xi_prior ** 2).sum(dim=-1) * w_pow).sum() / (w_pow.sum() + 1e-8)
                L_prior = L_prior + prior_rigid

        # Contact monotonicity
        for k in range(len(contact_scores) - 1):
            delta = contact_scores[k] - contact_scores[k + 1] - self.config.contact_eps
            L_contact = L_contact + torch.relu(delta).pow(2).mean()

        # Endpoint loss
        rigids_final = rigids_list[-1]
        chi_final = chi_list[-1]
        R_final, t_final = self._rigid_to_rt(rigids_final)
        R_holo, t_holo = self._rigid_to_rt(rigids_holo)
        R_inv, t_inv = rigid_inverse(R_final, t_final)
        R_delta, t_delta = rigid_compose(R_inv, t_inv, R_holo, t_holo)
        xi_end = se3_log(R_delta, t_delta)
        L_end_rigid = ((xi_end ** 2).sum(dim=-1) * w_pow).sum() / (w_pow.sum() + 1e-8)

        d_chi_end = wrap_to_pi(chi_final - batch.torsion_holo[..., 3:7])
        end_chi_term = ((d_chi_end ** 2) * chi_mask) * w_pow.unsqueeze(-1)
        end_chi_denom = (chi_mask * w_pow.unsqueeze(-1)).sum().clamp(min=1e-8)
        L_end_chi = end_chi_term.sum() / end_chi_denom

        # FAPE endpoint
        chi_sincos = torch.stack([torch.sin(chi_final), torch.cos(chi_final)], dim=-1)
        torsions_final_sincos = torch.cat([phi_psi_omega_sincos, chi_sincos], dim=2)
        atom14_final = self.fk_module(torsions_final_sincos, rigids_final, batch.aatype)

        pred_R = rigids_final.get_rots().get_rot_mats()
        pred_t = atom14_final['atom14_pos'][:, :, 1]
        true_R = rigids_holo.get_rots().get_rot_mats()
        true_t = atom14_holo['atom14_pos'][:, :, 1]

        L_end_fape = fape_loss(
            atom14_final['atom14_pos'],
            atom14_holo['atom14_pos'],
            (pred_R, pred_t),
            (true_R, true_t),
            w_res=None,
        )

        L_end = (self.config.w_end_chi * L_end_chi +
                 self.config.w_end_fape * L_end_fape +
                 L_end_rigid)

        total = (
            self.config.w_fm_chi * L_fm_chi +
            self.config.w_fm_rigid * L_fm_rigid +
            self.config.w_bg * L_bg +
            self.config.w_smooth * L_smooth +
            self.config.w_clash * L_clash +
            self.config.w_pep * L_pep +
            self.config.w_contact * L_contact +
            self.config.w_prior * L_prior +
            self.config.w_end * L_end
        )

        return {
            'total': total,
            'fm_chi': L_fm_chi,
            'fm_rigid': L_fm_rigid,
            'bg': L_bg,
            'smooth': L_smooth,
            'clash': L_clash,
            'pep': L_pep,
            'contact': L_contact,
            'prior': L_prior,
            'end': L_end,
        }

    def train_step(self, batch) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        batch = self._batch_to_device(batch)
        t = torch.rand(batch.esm.shape[0], device=self.device)

        if self.scaler is not None:
            with autocast():
                losses = self.compute_losses(batch, t)
                loss = losses['total']
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses = self.compute_losses(batch, t)
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
        batch.torsion_apo = batch.torsion_apo.to(self.device)
        batch.torsion_holo = batch.torsion_holo.to(self.device)
        batch.bb_mask = batch.bb_mask.to(self.device)
        batch.chi_mask = batch.chi_mask.to(self.device)
        batch.node_mask = batch.node_mask.to(self.device)
        batch.N_apo = batch.N_apo.to(self.device)
        batch.Ca_apo = batch.Ca_apo.to(self.device)
        batch.C_apo = batch.C_apo.to(self.device)
        batch.N_holo = batch.N_holo.to(self.device)
        batch.Ca_holo = batch.Ca_holo.to(self.device)
        batch.C_holo = batch.C_holo.to(self.device)
        batch.lig_points = batch.lig_points.to(self.device)
        batch.lig_types = batch.lig_types.to(self.device)
        batch.lig_mask = batch.lig_mask.to(self.device)
        batch.w_res = batch.w_res.to(self.device)
        if self.config.use_nma and batch.nma_features is None:
            raise ValueError("use_nma=True but batch.nma_features is None")
        if batch.nma_features is not None:
            batch.nma_features = batch.nma_features.to(self.device)
        batch.aatype = batch.aatype.to(self.device)
        return batch

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()

        epoch_losses = {
            'total': 0.0,
            'fm_chi': 0.0,
            'fm_rigid': 0.0,
            'bg': 0.0,
            'smooth': 0.0,
            'clash': 0.0,
            'pep': 0.0,
            'contact': 0.0,
            'prior': 0.0,
            'end': 0.0,
        }

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch:3d}', ncols=120, leave=True)
        for batch in pbar:
            step_losses = self.train_step(batch)
            for k in epoch_losses:
                epoch_losses[k] += step_losses[k]

            pbar.set_postfix({
                'loss': f"{step_losses['total']:.3f}",
                'fm': f"{step_losses['fm_chi'] + step_losses['fm_rigid']:.3f}",
            })

        n_batches = len(self.train_loader)
        return {k: v / n_batches for k, v in epoch_losses.items()}

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        val_losses = {
            'total': 0.0,
            'fm_chi': 0.0,
            'fm_rigid': 0.0,
            'bg': 0.0,
            'smooth': 0.0,
            'clash': 0.0,
            'pep': 0.0,
            'contact': 0.0,
            'prior': 0.0,
            'end': 0.0,
        }

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation', ncols=120, leave=False):
                batch = self._batch_to_device(batch)
                t = torch.rand(batch.esm.shape[0], device=self.device)
                losses = self.compute_losses(batch, t)
                for k in val_losses:
                    val_losses[k] += losses[k].item()

        n_batches = len(self.val_loader)
        return {k: v / n_batches for k, v in val_losses.items()}

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
        print("Start training - Stage-2")
        print(f"{'='*80}\n")

        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch

            train_losses = self.train_epoch()
            train_info = (
                f"Epoch {epoch:3d} | Loss: {train_losses['total']:.4f} "
                f"FM:{train_losses['fm_chi'] + train_losses['fm_rigid']:.2f} "
                f"Smooth:{train_losses['smooth']:.2f}"
            )
            print(train_info)

            if epoch % 1 == 0:
                val_results = self.validate()
                val_info = (
                    f" | Val Loss: {val_results['total']:.4f} "
                    f"FM:{val_results['fm_chi'] + val_results['fm_rigid']:.2f} "
                    f"Contact:{val_results['contact']:.3f}"
                )
                print(val_info)

                current_metric = val_results['total']
                if current_metric < self.best_val_metric:
                    self.best_val_metric = current_metric
                    self.patience_counter = 0
                    save_path = Path(self.config.save_dir) / 'best_model.pt'
                    self.save_checkpoint(str(save_path), verbose=False)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.early_stop_patience:
                        print("Early stopping triggered")
                        break

        final_path = Path(self.config.save_dir) / 'final_model.pt'
        self.save_checkpoint(str(final_path))
