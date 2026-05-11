"""
Stage-1 完整模型

架构：
ESM-2(冻结) → Adapter → IPA → LigandCond → ChiHead

Author: BINDRAE Team
Date: 2025-10-28
"""

import sys
import os
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass

import torch
import torch.nn as nn

# FlashIPA路径 (项目内 vendor 目录)
from pathlib import Path

_current_file = Path(__file__).resolve()
_project_root = _current_file.parents[3]  # 向上3级到项目根目录
flash_ipa_path = _project_root / 'vendor' / 'flash_ipa' / 'src'

# 统一使用 pathlib 进行路径检查和添加
if flash_ipa_path.exists() and str(flash_ipa_path) not in sys.path:
    sys.path.insert(0, str(flash_ipa_path))

from flash_ipa.rigid import Rigid, Rotation

from .adapter import ESMAdapter
from .ipa import FlashIPAModule, FlashIPAModuleConfig
from .ligand_condition import LigandConditioner, LigandConditionerConfig
from .torsion_head import CandidateChi1Scorer, ContactHead, Chi1RotamerHead, TorsionHead
from .fk_openfold import OpenFoldFK, create_openfold_fk, reorder_torsions_to_openfold
from ..modules.edge_embed import EdgeEmbedderAdapter, ProjectEdgeConfig
from ..data.residue_constants import restype_order


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class Stage1ModelConfig:
    """Stage-1模型配置"""
    # ESM Adapter
    esm_dim: int = 1280
    c_s: int = 384
    
    # EdgeEmbedder
    c_p: int = 128
    z_factor_rank: int = 2
    num_rbf: int = 16
    
    # FlashIPA
    c_hidden: int = 128
    no_heads: int = 8
    depth: int = 3
    no_qk_points: int = 8
    no_v_points: int = 12
    
    # LigandConditioner
    d_lig: int = 64
    num_heads_cross: int = 8
    warmup_steps: int = 2000
    # 增强配体编码器
    use_enhanced_ligand: bool = False
    enhanced_ligand_layers: int = 2
    enhanced_ligand_heads: int = 4
    ligand_num_rbf: int = 16
    
    # Chi Head
    torsion_hidden: int = 128
    chi_angles: int = 4
    use_pocket_chi1_expert: bool = False
    pocket_chi1_expert_hidden: int = 128
    pocket_chi1_expert_layers: int = 2
    pocket_chi1_gate_threshold: float = 0.5
    pocket_chi1_residual_scale: float = 0.25
    use_chi1_rotamer_posterior: bool = False
    chi1_rotamer_hidden: int = 128
    use_candidate_chi1_scorer: bool = False
    candidate_chi1_hidden: int = 128
    use_geometry_candidate_scorer: bool = False
    geometry_scorer_hidden: int = 64
    geometry_scorer_num_rbf: int = 16
    geometry_scorer_use_sgeo: bool = False
    geometry_scorer_sgeo_dim: int = 32
    geometry_scorer_use_typed_energy: bool = False
    geometry_scorer_lig_type_dim: int = 20
    geometry_scorer_typed_pair_dim: int = 64
    geometry_scorer_typed_cutoff: float = 6.0
    geometry_scorer_typed_init_scale: float = 0.1
    use_contact_posterior: bool = False
    contact_hidden: int = 128
    
    # Pocket Routing Adapter (early/mid-trunk ligand→pocket intervention)
    use_pocket_routing_adapter: bool = False
    pocket_routing_hidden: int = 128
    pocket_routing_layers: int = 2
    pocket_routing_gate_threshold: float = 0.3  # w_res threshold for pocket gating
    pocket_routing_residual_scale: float = 0.5  # scale of adapter residual
    
    # 通用
    dropout: float = 0.1
    
    @classmethod
    def small(cls) -> 'Stage1ModelConfig':
        """小型配置 (原始配置) - 约5M参数，depth=3，最稳定"""
        return cls()
    
    @classmethod
    def stable_wide(cls) -> 'Stage1ModelConfig':
        """稳定增强配置 - 约7M参数
        
        完全保持 small 的 IPA 配置，只增强输出部分。
        这是最稳定的增强方式。
        """
        return cls(
            # 完全保持 small 的 IPA 配置
            c_s=384,
            c_p=128,
            c_hidden=128,
            no_heads=8,
            depth=3,
            no_qk_points=8,
            no_v_points=12,
            # 只增强这些不影响 IPA 的部分
            d_lig=96,            # 64 → 96 (配体编码)
            num_heads_cross=12,  # 8 → 12 (交叉注意力)
            torsion_hidden=256,  # 128 → 256 (输出头)
        )
    
    @classmethod
    def medium(cls) -> 'Stage1ModelConfig':
        """中型配置 - 约10M参数，平衡深度与稳定性
        
        Note: headdim_eff = c_hidden + 36 + z_factor_rank*32 <= 256
            128 + 36 + 2*32 = 228 ✓
        """
        return cls(
            c_s=384,
            c_p=128,
            c_hidden=128,  # 保持128以满足FlashAttn限制
            no_heads=12,
            depth=5,  # 从8降到5，更稳定
            no_qk_points=8,
            no_v_points=12,
            torsion_hidden=192,
        )
    
    @classmethod
    def large(cls) -> 'Stage1ModelConfig':
        """大型配置 - 约40M参数，宽而深
        
        Note: headdim_eff = c_hidden + 36 + z_factor_rank*32 <= 256
            使用 z_factor_rank=1: 152 + 36 + 32 = 220 ✓
        """
        return cls(
            c_s=512,
            c_p=192,
            c_hidden=152,  # 限制以满足FlashAttn
            z_factor_rank=1,  # 降低以允许更大c_hidden
            no_heads=16,
            depth=8,
            no_qk_points=12,
            no_v_points=16,
            d_lig=96,
            num_heads_cross=12,
            torsion_hidden=256,
        )
    
    @classmethod
    def wide_shallow(cls) -> 'Stage1ModelConfig':
        """宽而浅配置 (RAE风格) - 约25M参数
        
        Note: headdim_eff = c_hidden + 36 + z_factor_rank*32 <= 256
            使用 z_factor_rank=1: 152 + 36 + 32 = 220 ✓
        """
        return cls(
            c_s=768,  # 2x 宽度
            c_p=256,
            c_hidden=152,  # 限制以满足FlashAttn
            z_factor_rank=1,  # 降低以允许更大c_hidden
            no_heads=12,
            depth=4,  # 保持较浅
            no_qk_points=8,
            no_v_points=12,
            d_lig=128,
            num_heads_cross=12,
            torsion_hidden=256,
        )
    
    @classmethod
    def enhanced_ligand(cls) -> 'Stage1ModelConfig':
        """增强配体编码器配置 - 基于 stable_wide + 增强配体编码
        
        核心改进：
        - 使用 EnhancedLigandEncoder (RBF距离 + 自注意力)
        - d_lig: 64 → 128 (更大配体表示)
        - 2层自注意力，4头
        
        预期效果: Chi1 +5-10%
        """
        return cls(
            # 保持 stable_wide 的 IPA 配置
            c_s=384,
            c_p=128,
            c_hidden=128,
            no_heads=8,
            depth=3,
            no_qk_points=8,
            no_v_points=12,
            # 增强配体编码器（核心改进）
            d_lig=128,  # 64 → 128
            num_heads_cross=12,
            use_enhanced_ligand=True,  # 启用增强编码器
            enhanced_ligand_layers=2,
            enhanced_ligand_heads=4,
            ligand_num_rbf=16,
            # 输出头
            torsion_hidden=256,
        )


class PocketRoutingAdapter(nn.Module):
    """Pocket-gated residual MLP applied between LigandConditioner and EdgeEmbedder.
    
    Selectively amplifies ligand-conditioned representations for pocket residues
    (w_res > threshold) while leaving non-pocket residues unchanged.
    """

    def __init__(self, c_s: int, hidden: int = 128, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        layers = [nn.LayerNorm(c_s)]
        in_dim = c_s
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, c_s))
        self.mlp = nn.Sequential(*layers)
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s: torch.Tensor, gate: torch.Tensor, scale: float = 0.5) -> torch.Tensor:
        """
        Args:
            s: [B, N, c_s] node representations (post ligand conditioning)
            gate: [B, N] continuous or binary pocket gate (0 for non-pocket)
            scale: residual scaling factor
        Returns:
            s + scale * gate * MLP(s)
        """
        delta = self.mlp(s)
        return s + scale * gate.unsqueeze(-1) * delta


class Stage1Model(nn.Module):
    """
    Stage-1 完整模型
    
    流程：
    ESM → Adapter → EdgeEmbed → IPA → LigandCond → ChiHead
    """
    
    def __init__(self, config: Stage1ModelConfig):
        """
        Args:
            config: Stage-1模型配置
        """
        super().__init__()
        
        self.config = config
        
        # 1. ESM Adapter
        self.esm_adapter = ESMAdapter(
            esm_dim=config.esm_dim,
            output_dim=config.c_s,
            dropout=config.dropout
        )
        
        # 2. EdgeEmbedder
        edge_config = ProjectEdgeConfig(
            c_s=config.c_s,
            c_p=config.c_p,
            z_factor_rank=config.z_factor_rank,
            num_rbf=config.num_rbf,
        )
        self.edge_embedder = EdgeEmbedderAdapter(edge_config)
        
        # 3. FlashIPA
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
        
        # 4. LigandConditioner
        ligand_config = LigandConditionerConfig(
            c_s=config.c_s,
            d_lig=config.d_lig,
            num_heads=config.num_heads_cross,
            dropout=config.dropout,
            warmup_steps=config.warmup_steps,
            # 增强编码器配置
            use_enhanced_encoder=config.use_enhanced_ligand,
            enhanced_num_layers=config.enhanced_ligand_layers,
            enhanced_num_heads=config.enhanced_ligand_heads,
            num_rbf=config.ligand_num_rbf,
        )
        self.ligand_conditioner = LigandConditioner(ligand_config)
        
        # 5. Chi Head (only chi1-4)
        self.chi_head = TorsionHead(
            c_s=config.c_s,
            c_hidden=config.torsion_hidden,
            dropout=config.dropout,
            n_angles=config.chi_angles
        )

        # 5b. Optional late pocket-only chi1 residual expert
        self.pocket_chi1_expert = None

        # 5b. Optional posterior heads for Stage-2 soft guidance experiments
        self.chi1_rotamer_head = None
        if config.use_chi1_rotamer_posterior:
            self.chi1_rotamer_head = Chi1RotamerHead(
                c_s=config.c_s,
                c_hidden=config.chi1_rotamer_hidden,
                dropout=config.dropout,
            )

        self.contact_head = None
        if config.use_contact_posterior:
            self.contact_head = ContactHead(
                c_s=config.c_s,
                c_hidden=config.contact_hidden,
                dropout=config.dropout,
            )

        self.candidate_chi1_scorer = None
        if config.use_candidate_chi1_scorer:
            self.candidate_chi1_scorer = CandidateChi1Scorer(
                c_s=config.c_s,
                c_hidden=config.candidate_chi1_hidden,
                dropout=config.dropout,
            )

        self.geometry_candidate_scorer = None
        if config.use_geometry_candidate_scorer:
            from .torsion_head import GeometryCandidateScorer
            self.geometry_candidate_scorer = GeometryCandidateScorer(
                c_hidden=config.geometry_scorer_hidden,
                num_rbf=config.geometry_scorer_num_rbf,
                dropout=config.dropout,
                use_sgeo=config.geometry_scorer_use_sgeo,
                c_s=config.c_s,
                sgeo_proj_dim=config.geometry_scorer_sgeo_dim,
                bounded_residual=getattr(config, 'geometry_scorer_bounded_residual', False),
                residual_max=getattr(config, 'geometry_scorer_residual_max', 5.0),
                residual_tau=getattr(config, 'geometry_scorer_residual_tau', 2.0),
                gate_norm=getattr(config, 'geometry_scorer_gate_norm', False),
                gate_clamp=getattr(config, 'geometry_scorer_gate_clamp', 6.0),
                gate_init_bias=getattr(config, 'geometry_scorer_gate_init_bias', 0.0),
                use_typed_energy=getattr(config, 'geometry_scorer_use_typed_energy', False),
                lig_type_dim=getattr(config, 'geometry_scorer_lig_type_dim', 20),
                typed_pair_dim=getattr(config, 'geometry_scorer_typed_pair_dim', 64),
                typed_cutoff=getattr(config, 'geometry_scorer_typed_cutoff', 6.0),
                typed_init_scale=getattr(config, 'geometry_scorer_typed_init_scale', 0.1),
            )

        # 5c. Optional pocket routing adapter (early/mid-trunk)
        self.pocket_routing_adapter = None
        if config.use_pocket_routing_adapter:
            self.pocket_routing_adapter = PocketRoutingAdapter(
                c_s=config.c_s,
                hidden=config.pocket_routing_hidden,
                n_layers=config.pocket_routing_layers,
                dropout=config.dropout,
            )
        if config.use_pocket_chi1_expert:
            expert_layers = [nn.LayerNorm(config.c_s)]
            in_dim = config.c_s
            hidden_dim = config.pocket_chi1_expert_hidden
            n_layers = max(int(config.pocket_chi1_expert_layers), 1)
            for _ in range(n_layers - 1):
                expert_layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ])
                in_dim = hidden_dim
            expert_layers.append(nn.Linear(in_dim, 2))
            self.pocket_chi1_expert = nn.Sequential(*expert_layers)
            nn.init.normal_(self.pocket_chi1_expert[-1].weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.pocket_chi1_expert[-1].bias)
        
        # 6. FK模块（扭转角→全原子坐标）
        self.fk_module = create_openfold_fk()
        
        print(f"✓ Stage1Model 初始化完成")
        print(f"  - 总参数量: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  - 包含FK模块（OpenFold式）")
    
    def _sequence_to_aatype(self, sequences: List[str], max_len: int, device) -> torch.Tensor:
        """
        将氨基酸序列转换为aatype索引
        
        Args:
            sequences: List[氨基酸序列（单字母）]
            max_len: 最大长度（用于padding）
            device: 设备
            
        Returns:
            aatype: [B, N] 残基类型索引(0-19, 20=UNK)
        """
        B = len(sequences)
        aatype = torch.zeros(B, max_len, dtype=torch.long, device=device)
        
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                if j >= max_len:
                    break
                aatype[i, j] = restype_order.get(aa, 20)  # 20=UNK
        
        return aatype
    
    def forward(self,
                batch: 'Stage1Batch',
                current_step: int = 0,
                geometry_scorer_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            batch: Stage1Batch数据
            current_step: 当前训练步数（用于warmup）
            geometry_scorer_kwargs: 透传给 GeometryCandidateScorer.forward 的额外参数
                （如 base_detach, gate_override, beta），仅在该 scorer 启用时生效

        Returns:
            {
                'pred_chi': [B, N, 4, 2] 预测的chi(sin, cos)
                's_final': [B, N, c_s] 最终节点表示
                'rigids_final': Rigid对象
            }
        """
        B, N = batch.esm.shape[:2]
        device = batch.esm.device
        
        # 1. ESM Adapter
        s = self.esm_adapter(batch.esm)  # [B, N, 384]
        
        # 2. 创建初始Rigid帧（从apo的N, CA, C）
        rigids = self._build_rigids_from_backbone(
            batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
        )
        
        # 3. LigandConditioner（在IPA前注入，符合理论）
        s_with_ligand = self.ligand_conditioner(
            s,
            batch.lig_points,
            batch.lig_types,
            batch.node_mask,
            batch.lig_mask,
            current_step=current_step
        )
        
        # 3b. Pocket Routing Adapter (pocket-gated residual before IPA)
        if self.pocket_routing_adapter is not None:
            pocket_gate = ((batch.w_res > self.config.pocket_routing_gate_threshold) & batch.node_mask.bool()).float()
            s_with_ligand = self.pocket_routing_adapter(
                s_with_ligand, pocket_gate, self.config.pocket_routing_residual_scale
            )
        
        # 4. EdgeEmbedder
        edge_outputs = self.edge_embedder(s_with_ligand, batch.Ca_apo, batch.node_mask)
        z_f1 = edge_outputs['z_f1']
        z_f2 = edge_outputs['z_f2']
        
        # 5. FlashIPA（几何分支）
        s_geo, rigids_updated = self.ipa_module(
            s_with_ligand,
            rigids,
            z_f1,
            z_f2,
            batch.node_mask,
            ligand_conditioner=self.ligand_conditioner,
            lig_points=batch.lig_points,
            lig_types=batch.lig_types,
            protein_mask=batch.node_mask,
            ligand_mask=batch.lig_mask,
            current_step=current_step,
        )
        
        # 6. TorsionHead（使用IPA输出）
        pred_chi = self.chi_head(s_geo)  # [B, N, 4, 2]

        chi1_rotamer_logits = None
        chi1_rotamer_probs = None
        if self.chi1_rotamer_head is not None:
            chi1_rotamer_logits = self.chi1_rotamer_head(s_geo)  # [B, N, 3]
            chi1_rotamer_probs = torch.softmax(chi1_rotamer_logits, dim=-1)

        contact_logits = None
        contact_probs = None
        if self.contact_head is not None:
            contact_logits = self.contact_head(s_geo)  # [B, N]
            contact_probs = torch.sigmoid(contact_logits)

        candidate_chi1_logits = None
        candidate_chi1_probs = None
        if self.candidate_chi1_scorer is not None:
            candidate_chi1_logits = self.candidate_chi1_scorer(s_geo)  # [B, N, 3]
            candidate_chi1_probs = torch.softmax(candidate_chi1_logits, dim=-1)

        geometry_chi1_logits = None
        geometry_chi1_probs = None
        geometry_chi1_base_logits = None
        geometry_chi1_residual_logits = None
        geometry_chi1_gate = None
        geometry_chi1_typed_energy = None
        if self.geometry_candidate_scorer is not None:
            aatype_for_geom = self._sequence_to_aatype(batch.sequences, N, device)
            scorer_kwargs = dict(geometry_scorer_kwargs) if geometry_scorer_kwargs else {}
            result = self.geometry_candidate_scorer(
                self.fk_module, rigids_updated, aatype_for_geom,
                batch.torsion_apo, batch.lig_points, batch.lig_types, batch.lig_mask, batch.node_mask,
                s_geo=s_geo if self.config.geometry_scorer_use_sgeo else None,
                return_decomposition=True,
                **scorer_kwargs,
            )
            (
                geometry_chi1_logits,
                geometry_chi1_base_logits,
                geometry_chi1_residual_logits,
                geometry_chi1_gate,
                geometry_chi1_typed_energy,
            ) = result
            geometry_chi1_probs = torch.softmax(geometry_chi1_logits, dim=-1)

        pocket_chi1_delta = None
        if self.pocket_chi1_expert is not None:
            pocket_gate = ((batch.w_res > self.config.pocket_chi1_gate_threshold) & batch.node_mask.bool()).float()
            pocket_chi1_delta = self.pocket_chi1_expert(s_geo) * pocket_gate.unsqueeze(-1)
            base_chi1 = pred_chi[:, :, :1, :]
            corrected_chi1 = base_chi1 + self.config.pocket_chi1_residual_scale * pocket_chi1_delta.unsqueeze(2)
            corrected_chi1 = nn.functional.normalize(corrected_chi1, p=2, dim=-1, eps=1e-4)
            pred_chi = torch.cat([corrected_chi1, pred_chi[:, :, 1:, :]], dim=2)
        
        # 7. FK重建全原子坐标
        # 将序列转换为aatype索引
        aatype = self._sequence_to_aatype(batch.sequences, N, device)

        # 组装完整torsions: 使用apo的phi/psi/omega，预测chi
        torsion_apo = batch.torsion_apo  # [B, N, 7] (phi/psi/omega/chi1-4)
        torsion_apo = torsion_apo.to(device)
        phi_psi_omega = torsion_apo[:, :, :3]
        phi_psi_omega_sincos = torch.stack(
            [torch.sin(phi_psi_omega), torch.cos(phi_psi_omega)], dim=-1
        )  # [B, N, 3, 2]
        torsions_sincos = torch.cat([phi_psi_omega_sincos, pred_chi], dim=2)  # [B,N,7,2]
        torsions_sincos = reorder_torsions_to_openfold(torsions_sincos)

        atom14_result = self.fk_module(torsions_sincos, rigids_updated, aatype)
        
        return {
            'pred_chi': pred_chi,
            'chi1_rotamer_logits': chi1_rotamer_logits,
            'chi1_rotamer_probs': chi1_rotamer_probs,
            'contact_logits': contact_logits,
            'contact_probs': contact_probs,
            'candidate_chi1_logits': candidate_chi1_logits,
            'candidate_chi1_probs': candidate_chi1_probs,
            'geometry_chi1_logits': geometry_chi1_logits,
            'geometry_chi1_probs': geometry_chi1_probs,
            'geometry_chi1_base_logits': geometry_chi1_base_logits,
            'geometry_chi1_residual_logits': geometry_chi1_residual_logits,
            'geometry_chi1_gate': geometry_chi1_gate,
            'geometry_chi1_typed_energy': geometry_chi1_typed_energy,
            'pocket_chi1_delta': pocket_chi1_delta,
            's_final': s_geo,  # IPA输出（已含配体信息）
            'rigids_final': rigids_updated,
            'atom14_pos': atom14_result['atom14_pos'],      # [B, N, 14, 3]
            'atom14_mask': atom14_result['atom14_mask'],    # [B, N, 14]
        }

    def _build_rigids_from_backbone(self,
                                    N: torch.Tensor,
                                    Ca: torch.Tensor,
                                    C: torch.Tensor,
                                    mask: torch.Tensor,
                                    eps: float = 1e-6) -> Rigid:
        """Build per-residue backbone frames from N/CA/C.
        
        修复：当向量接近零时，使用单位向量（而不是除以很小的 norm）
        """
        device = Ca.device
        default_e1 = torch.tensor([1.0, 0.0, 0.0], device=device)
        default_e2 = torch.tensor([0.0, 1.0, 0.0], device=device)
        default_e3 = torch.tensor([0.0, 0.0, 1.0], device=device)
        
        # e1: CA -> C
        e1 = C - Ca
        e1_norm = torch.norm(e1, dim=-1, keepdim=True)
        e1_valid = e1_norm > eps
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

        R = torch.stack([e1, e2, e3], dim=-1)  # [B, N, 3, 3]
        t = Ca

        # For padded residues, set identity rotation and zero translation
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
            eye = torch.eye(3, device=device).view(1, 1, 3, 3)
            R = torch.where(mask_expanded, R, eye)
            t = torch.where(mask.unsqueeze(-1), t, torch.zeros_like(t))

        # 最终检查：替换任何残留的 NaN
        R = torch.where(torch.isnan(R), torch.eye(3, device=device).view(1, 1, 3, 3).expand_as(R), R)
        t = torch.where(torch.isnan(t), torch.zeros_like(t), t)

        rotation = Rotation(rot_mats=R)
        return Rigid(rots=rotation, trans=t)


def create_stage1_model(config: Optional[Stage1ModelConfig] = None) -> Stage1Model:
    """
    创建Stage-1模型
    
    Args:
        config: 模型配置（可选，使用默认配置）
        
    Returns:
        Stage1Model实例
    """
    if config is None:
        config = Stage1ModelConfig()
    
    return Stage1Model(config)
