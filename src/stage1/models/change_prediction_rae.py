"""
Change-Prediction RAE for learning ligand-induced conformational changes.

Architecture:
1. Frozen Encoder: (structure) → latent_z (pretrained, frozen)
2. Delta-z computation: delta_z = encode(holo) - encode(apo)
3. Ligand Predictor: (apo_structure, ligand) → predicted_delta_z
4. Decoder: (z_apo + predicted_delta_z) → holo_structure

Training objective:
- Primary: ||predicted_delta_z - delta_z||^2 (latent change prediction)
- Secondary: reconstruction loss (optional, for regularization)

This approach avoids chi1 dominance by operating in latent space.
"""

import sys
import os
from typing import Dict, Optional, Any, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

_current_file = Path(__file__).resolve()
_project_root = _current_file.parents[3]
flash_ipa_path = _project_root / 'vendor' / 'flash_ipa' / 'src'

if flash_ipa_path.exists() and str(flash_ipa_path) not in sys.path:
    sys.path.insert(0, str(flash_ipa_path))

from flash_ipa.rigid import Rigid, Rotation

from .adapter import ESMAdapter
from .delta_z_predictor import DeltaZPredictor
from .ipa import FlashIPAModule, FlashIPAModuleConfig
from .ligand_condition import LigandConditioner, LigandConditionerConfig
from .torsion_head import TorsionHead
from .fk_openfold import OpenFoldFK, create_openfold_fk, reorder_torsions_to_openfold
from ..modules.edge_embed import EdgeEmbedderAdapter, ProjectEdgeConfig
from ..data.residue_constants import restype_order


@dataclass
class ChangePredictionRAEConfig:
    """Configuration for Change-Prediction RAE."""
    # ESM Adapter
    esm_dim: int = 1280
    c_s: int = 384
    
    # EdgeEmbedder
    c_p: int = 128
    z_factor_rank: int = 2
    num_rbf: int = 16
    
    # FlashIPA (frozen encoder)
    c_hidden: int = 128
    no_heads: int = 8
    depth: int = 3
    no_qk_points: int = 8
    no_v_points: int = 12
    
    # LigandConditioner (for ligand predictor)
    d_lig: int = 64
    num_heads_cross: int = 8
    warmup_steps: int = 2000
    
    # Delta-z predictor
    delta_z_hidden: int = 256
    delta_z_layers: int = 3
    
    # Chi Head (decoder)
    torsion_hidden: int = 128
    chi_angles: int = 4
    
    # Loss weights
    lambda_latent: float = 1.0
    lambda_recon: float = 0.1
    
    # General
    dropout: float = 0.1


class ChangePredictionRAE(nn.Module):
    """
    Change-Prediction RAE for learning ligand-induced conformational changes.
    
    Key insight: Instead of predicting holo structure directly, predict the
    CHANGE in latent space (delta_z = z_holo - z_apo) from ligand information.
    """
    
    def __init__(self, config: ChangePredictionRAEConfig):
        super().__init__()
        self.config = config
        
        # 1. Frozen Encoder (ESM Adapter + EdgeEmbedder + IPA)
        self.esm_adapter = ESMAdapter(
            esm_dim=config.esm_dim,
            output_dim=config.c_s,
            dropout=config.dropout
        )
        
        edge_config = ProjectEdgeConfig(
            c_s=config.c_s,
            c_p=config.c_p,
            z_factor_rank=config.z_factor_rank,
            num_rbf=config.num_rbf,
        )
        self.edge_embedder = EdgeEmbedderAdapter(edge_config)
        
        ipa_config = FlashIPAModuleConfig(
            c_s=config.c_s,
            c_z=config.c_p,
            c_hidden=config.c_hidden,
            no_heads=config.no_heads,
            depth=config.depth,
            no_qk_points=config.no_qk_points,
            no_v_points=config.no_v_points,
            z_factor_rank=config.z_factor_rank,
            dropout=config.dropout,
        )
        self.ipa_module = FlashIPAModule(ipa_config)
        
        # 2. Ligand Conditioner (for ligand predictor)
        ligand_config = LigandConditionerConfig(
            c_s=config.c_s,
            d_lig=config.d_lig,
            num_heads=config.num_heads_cross,
            dropout=config.dropout,
            warmup_steps=config.warmup_steps,
        )
        self.ligand_conditioner = LigandConditioner(ligand_config)
        
        # 3. Delta-z Predictor
        self.delta_z_predictor = DeltaZPredictor(
            c_s=config.c_s,
            hidden=config.delta_z_hidden,
            n_layers=config.delta_z_layers,
            dropout=config.dropout,
        )
        
        # 4. Decoder (TorsionHead + FK)
        self.chi_head = TorsionHead(
            c_s=config.c_s,
            c_hidden=config.torsion_hidden,
            dropout=config.dropout,
            n_angles=config.chi_angles
        )
        
        self.fk_module = create_openfold_fk()
        
        self._freeze_encoder()
        
        print(f"✓ ChangePredictionRAE initialized")
        print(f"  - Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  - Trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def _freeze_encoder(self):
        """Freeze encoder parameters (ESM adapter, edge embedder, IPA)."""
        for param in self.esm_adapter.parameters():
            param.requires_grad = False
        for param in self.edge_embedder.parameters():
            param.requires_grad = False
        for param in self.ipa_module.parameters():
            param.requires_grad = False
        self._set_frozen_encoder_eval()

    def _set_frozen_encoder_eval(self):
        """Frozen encoder must be deterministic during training.

        Otherwise dropout in the frozen modules creates a synthetic
        apo-vs-holo delta target even when the deterministic latent states are
        identical.
        """
        self.esm_adapter.eval()
        self.edge_embedder.eval()
        self.ipa_module.eval()

    def train(self, mode: bool = True):
        """Keep the frozen encoder in eval mode while training trainable heads."""
        super().train(mode)
        self._set_frozen_encoder_eval()
        return self

    def load_stage1_modules(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefixes: Iterable[str] = (
            'esm_adapter',
            'edge_embedder',
            'ipa_module',
            'ligand_conditioner',
            'chi_head',
        ),
    ) -> Dict[str, Any]:
        """Load compatible modules from a Stage-1 checkpoint.

        This lets change-prediction experiments reuse a trained representation
        space instead of freezing a randomly initialized encoder.  Keys with a
        DDP ``module.`` prefix are accepted, and shape mismatches are skipped.
        """
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        own_state = self.state_dict()
        clean_state: Dict[str, torch.Tensor] = {}
        wanted = tuple(f'{p}.' for p in prefixes)
        skipped = []
        for key, value in state_dict.items():
            clean_key = key[len('module.'):] if key.startswith('module.') else key
            if not clean_key.startswith(wanted):
                continue
            if clean_key not in own_state:
                skipped.append((clean_key, 'missing'))
                continue
            if tuple(own_state[clean_key].shape) != tuple(value.shape):
                skipped.append((clean_key, f'shape {tuple(value.shape)} != {tuple(own_state[clean_key].shape)}'))
                continue
            clean_state[clean_key] = value

        incompatible = self.load_state_dict(clean_state, strict=False)
        return {
            'loaded': sorted(clean_state.keys()),
            'skipped': skipped,
            'missing_after_partial_load': list(incompatible.missing_keys),
            'unexpected_after_partial_load': list(incompatible.unexpected_keys),
        }
    
    def _sequence_to_aatype(self, sequences, max_len: int, device):
        """Convert sequences to aatype indices."""
        B = len(sequences)
        aatype = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                if j >= max_len:
                    break
                aatype[i, j] = restype_order.get(aa, 20)
        return aatype
    
    def _build_rigids_from_backbone(self, N, Ca, C, mask, eps=1e-6):
        """Build per-residue backbone frames from N/CA/C."""
        device = Ca.device
        default_e1 = torch.tensor([1.0, 0.0, 0.0], device=device)
        default_e2 = torch.tensor([0.0, 1.0, 0.0], device=device)
        default_e3 = torch.tensor([0.0, 0.0, 1.0], device=device)
        
        e1 = C - Ca
        e1_norm = torch.norm(e1, dim=-1, keepdim=True)
        e1_valid = e1_norm > eps
        e1_normalized = e1 / torch.clamp(e1_norm, min=eps)
        e1 = torch.where(e1_valid, e1_normalized, default_e1.expand_as(e1))
        
        u = N - Ca
        proj = (u * e1).sum(dim=-1, keepdim=True) * e1
        e2 = u - proj
        e2_norm = torch.norm(e2, dim=-1, keepdim=True)
        e2_valid = e2_norm > eps
        e2_normalized = e2 / torch.clamp(e2_norm, min=eps)
        e2 = torch.where(e2_valid, e2_normalized, default_e2.expand_as(e2))
        
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
        
        R = torch.where(torch.isnan(R), torch.eye(3, device=device).view(1, 1, 3, 3).expand_as(R), R)
        t = torch.where(torch.isnan(t), torch.zeros_like(t), t)
        
        rotation = Rotation(rot_mats=R)
        return Rigid(rots=rotation, trans=t)
    
    def encode(self, batch, structure_type='apo'):
        """
        Encode structure to latent space using frozen encoder.
        
        Args:
            batch: Stage1Batch
            structure_type: 'apo' or 'holo'
        
        Returns:
            z: [B, N, c_s] latent representation
            rigids: Rigid frames
        """
        B, N = batch.esm.shape[:2]
        
        # ESM Adapter
        s = self.esm_adapter(batch.esm)
        
        # Build rigids from backbone
        if structure_type == 'apo':
            rigids = self._build_rigids_from_backbone(
                batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
            )
        else:
            rigids = self._build_rigids_from_backbone(
                batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
            )
        
        # EdgeEmbedder
        Ca = batch.Ca_apo if structure_type == 'apo' else batch.Ca_holo
        edge_outputs = self.edge_embedder(s, Ca, batch.node_mask)
        z_f1 = edge_outputs['z_f1']
        z_f2 = edge_outputs['z_f2']
        
        # IPA (no ligand conditioning for encoder)
        z, rigids_updated = self.ipa_module(
            s, rigids, z_f1, z_f2, batch.node_mask,
            ligand_conditioner=None,
            lig_points=None,
            lig_types=None,
            protein_mask=batch.node_mask,
            ligand_mask=None,
            current_step=0,
        )
        
        return z, rigids_updated

    def predict_delta_z(
        self,
        batch,
        current_step: int = 0,
        gate_lambda: Optional[float] = None,
        return_conditioned: bool = False,
    ):
        """Predict latent change from ligand-conditioned apo features.

        ``gate_lambda`` is useful for diagnostics: passing 1.0 forces the ligand
        conditioner fully open and avoids confusing warmup behavior with true
        ligand insensitivity.
        """
        s = self.esm_adapter(batch.esm)
        conditioner_kwargs = {}
        if gate_lambda is None:
            conditioner_kwargs['current_step'] = current_step
        else:
            conditioner_kwargs['gate_lambda'] = float(gate_lambda)

        s_with_ligand = self.ligand_conditioner(
            s,
            batch.lig_points,
            batch.lig_types,
            batch.node_mask,
            batch.lig_mask,
            **conditioner_kwargs,
        )
        delta_z_pred = self.delta_z_predictor(s_with_ligand, batch.node_mask)
        if return_conditioned:
            return delta_z_pred, s_with_ligand
        return delta_z_pred
    
    def forward(self, batch, current_step: int = 0) -> Dict[str, torch.Tensor]:
        """
        Forward pass for change prediction.
        
        Returns:
            {
                'z_apo': [B, N, c_s] apo latent
                'z_holo': [B, N, c_s] holo latent
                'delta_z_true': [B, N, c_s] true change
                'delta_z_pred': [B, N, c_s] predicted change
                'z_holo_pred': [B, N, c_s] predicted holo latent
                'pred_chi': [B, N, 4, 2] predicted chi angles
                'rigids_apo': apo frames
                'rigids_holo_pred': predicted holo frames
            }
        """
        B, N = batch.esm.shape[:2]
        device = batch.esm.device
        
        # 1. Encode apo and holo (frozen encoder)
        with torch.no_grad():
            z_apo, rigids_apo = self.encode(batch, 'apo')
            z_holo, rigids_holo = self.encode(batch, 'holo')
        
        # 2. Compute true delta_z
        delta_z_true = z_holo - z_apo
        
        # 3-4. Ligand conditioning on apo and delta-z prediction
        delta_z_pred, s_with_ligand = self.predict_delta_z(
            batch,
            current_step=current_step,
            return_conditioned=True,
        )
        
        # 5. Predict holo latent
        z_holo_pred = z_apo + delta_z_pred
        
        # 6. Decode to structure (chi angles)
        pred_chi = self.chi_head(z_holo_pred)
        
        # 7. FK reconstruction
        aatype = self._sequence_to_aatype(batch.sequences, N, device)
        
        torsion_apo = batch.torsion_apo.to(device)
        phi_psi_omega = torsion_apo[:, :, :3]
        phi_psi_omega_sincos = torch.stack(
            [torch.sin(phi_psi_omega), torch.cos(phi_psi_omega)], dim=-1
        )
        torsions_sincos = torch.cat([phi_psi_omega_sincos, pred_chi], dim=2)
        torsions_sincos = reorder_torsions_to_openfold(torsions_sincos)
        
        atom14_result = self.fk_module(torsions_sincos, rigids_apo, aatype)
        
        return {
            'z_apo': z_apo,
            'z_holo': z_holo,
            'delta_z_true': delta_z_true,
            'delta_z_pred': delta_z_pred,
            'z_holo_pred': z_holo_pred,
            'pred_chi': pred_chi,
            's_with_ligand': s_with_ligand,
            'rigids_apo': rigids_apo,
            'atom14_pos': atom14_result['atom14_pos'],
            'atom14_mask': atom14_result['atom14_mask'],
        }


def create_change_prediction_rae(config: Optional[ChangePredictionRAEConfig] = None) -> ChangePredictionRAE:
    """Create Change-Prediction RAE model."""
    if config is None:
        config = ChangePredictionRAEConfig()
    return ChangePredictionRAE(config)
