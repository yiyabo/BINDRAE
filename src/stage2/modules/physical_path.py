"""Projected physical refinement for endpoint-exact phase-warped paths.

The optimizer in this module is deliberately not a learned MD-replica
predictor.  It searches the normal bundle around an existing endpoint-exact
guide path for a small, smooth correction that improves differentiable
geometry and steric objectives while preserving the guide's contact timing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from flash_ipa.rigid import Rigid, Rotation

from src.stage1.models.fk_openfold import reorder_torsions_to_openfold

from .geometry import (
    compute_peptide_loss,
    residue_ligand_min_dist,
    soft_contact_from_dist,
    wrap_to_pi,
)
from .phase_residual import (
    endpoint_zero_envelope,
    project_block_tangent_normal,
    project_product_tangent_normal,
)
from .se3 import rigid_compose, se3_exp


@dataclass(frozen=True)
class PhysicalPathOptimizationConfig:
    """Hyperparameters for test-time projected normal refinement."""

    iterations: int = 32
    learning_rate: float = 5e-2
    envelope: str = "poly"
    projection_mode: str = "block"
    components: str = "translation"
    rotation_metric_scale: float = 1.0
    translation_metric_scale: float = 1.0
    chi_metric_scale: float = 1.0
    min_tangent_norm: float = 1e-4
    max_metric_norm: float = 1.5
    gradient_clip: float = 10.0
    protein_clash_distance: float = 2.0
    ligand_clash_distance: float = 2.2
    contact_distance: float = 6.0
    contact_temperature: float = 1.0
    pocket_threshold: float = 0.3
    max_clash_atoms: int = 512
    weight_peptide: float = 1.0
    weight_protein_clash: float = 1.0
    weight_ligand_clash: float = 1.0
    weight_contact_anchor: float = 0.25
    weight_distance_anchor: float = 1.0
    weight_residual_magnitude: float = 0.05
    weight_temporal_smoothness: float = 0.10
    normalize_physical_terms: bool = True
    optimizer: str = "adam"
    num_starts: int = 1
    route_seed_scale: float = 0.0
    route_seed_rank: int = 2
    route_seed_smoothing_steps: int = 2
    route_seed: int = 20260720
    frame_aggregation: str = "mean"
    frame_softmax_beta: float = 10.0
    line_search_steps: int = 8
    line_search_shrink: float = 0.5
    acceptance_tolerance: float = 1e-8

    def validate(self) -> None:
        if self.iterations < 0:
            raise ValueError("iterations must be >= 0")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0")
        if self.max_metric_norm < 0.0:
            raise ValueError("max_metric_norm must be >= 0")
        if self.gradient_clip < 0.0:
            raise ValueError("gradient_clip must be >= 0")
        if self.max_clash_atoms < 2:
            raise ValueError("max_clash_atoms must be >= 2")
        if self.projection_mode not in {"product", "block"}:
            raise ValueError("projection_mode must be 'product' or 'block'")
        if self.components not in {"translation", "rigid", "all"}:
            raise ValueError("components must be 'translation', 'rigid', or 'all'")
        if self.optimizer not in {"adam", "backtracking"}:
            raise ValueError("optimizer must be 'adam' or 'backtracking'")
        if self.num_starts <= 0:
            raise ValueError("num_starts must be positive")
        if self.num_starts > 1 and self.route_seed_scale <= 0.0:
            raise ValueError("route_seed_scale must be > 0 when num_starts > 1")
        if self.route_seed_scale < 0.0:
            raise ValueError("route_seed_scale must be >= 0")
        if self.route_seed_rank <= 0:
            raise ValueError("route_seed_rank must be positive")
        if self.route_seed_smoothing_steps < 0:
            raise ValueError("route_seed_smoothing_steps must be >= 0")
        if self.frame_aggregation not in {"mean", "max", "softmax"}:
            raise ValueError("frame_aggregation must be 'mean', 'max', or 'softmax'")
        if self.frame_softmax_beta <= 0.0:
            raise ValueError("frame_softmax_beta must be > 0")
        if self.line_search_steps <= 0:
            raise ValueError("line_search_steps must be positive")
        if not 0.0 < self.line_search_shrink < 1.0:
            raise ValueError("line_search_shrink must be in (0, 1)")
        if self.acceptance_tolerance < 0.0:
            raise ValueError("acceptance_tolerance must be >= 0")
        for name in (
            "weight_peptide",
            "weight_protein_clash",
            "weight_ligand_clash",
            "weight_contact_anchor",
            "weight_distance_anchor",
            "weight_residual_magnitude",
            "weight_temporal_smoothness",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0")


@dataclass
class PhysicalPathOptimizationResult:
    rigids: List[Rigid]
    chi: List[torch.Tensor]
    times: List[float]
    projected_rigid: torch.Tensor
    projected_chi: torch.Tensor
    applied_rigid: torch.Tensor
    applied_chi: torch.Tensor
    diagnostics: Dict[str, object]


def _rigid_to_rt(rigid: Rigid) -> Tuple[torch.Tensor, torch.Tensor]:
    return rigid.get_rots().get_rot_mats(), rigid.get_trans()


def _rt_to_rigid(rotation: torch.Tensor, translation: torch.Tensor) -> Rigid:
    return Rigid(rots=Rotation(rot_mats=rotation), trans=translation)


def _detach_rigid(rigid: Rigid) -> Rigid:
    rotation, translation = _rigid_to_rt(rigid)
    return _rt_to_rigid(rotation.detach(), translation.detach())


def _validate_path_shapes(
    base_rigids: Sequence[Rigid],
    base_chi: Sequence[torch.Tensor],
    times: Sequence[float],
    bridge_tangent_rigid: torch.Tensor,
    bridge_tangent_chi: torch.Tensor,
) -> Tuple[int, int, int]:
    if len(base_rigids) != len(base_chi) or len(base_rigids) != len(times):
        raise ValueError("base rigid, chi, and time paths must have equal length")
    if len(times) < 3:
        raise ValueError("physical normal refinement requires at least one interior frame")
    if abs(float(times[0])) > 1e-8 or abs(float(times[-1]) - 1.0) > 1e-8:
        raise ValueError("path times must include exact 0 and 1 endpoints")
    if any(float(b) <= float(a) for a, b in zip(times[:-1], times[1:])):
        raise ValueError("path times must be strictly increasing")

    batch_size, n_res, chi_dim = base_chi[0].shape
    n_interior = len(times) - 2
    expected_rigid = (n_interior, batch_size, n_res, 6)
    expected_chi = (n_interior, batch_size, n_res, chi_dim)
    if tuple(bridge_tangent_rigid.shape) != expected_rigid:
        raise ValueError(
            f"bridge_tangent_rigid shape={tuple(bridge_tangent_rigid.shape)}, "
            f"expected {expected_rigid}"
        )
    if tuple(bridge_tangent_chi.shape) != expected_chi:
        raise ValueError(
            f"bridge_tangent_chi shape={tuple(bridge_tangent_chi.shape)}, "
            f"expected {expected_chi}"
        )
    return n_interior, batch_size, n_res


def _smooth_chain_field(
    field: torch.Tensor,
    node_mask: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Diffuse a low-frequency route field along contiguous chain neighbors."""
    if steps <= 0 or field.shape[-2] <= 1:
        return field * node_mask.view(1, *node_mask.shape, 1).to(field.dtype)
    mask = node_mask.view(1, *node_mask.shape, 1).to(field.dtype)
    value = field * mask
    for _ in range(int(steps)):
        total = value.clone()
        count = mask.clone()
        total[..., 1:, :] = total[..., 1:, :] + value[..., :-1, :]
        count[..., 1:, :] = count[..., 1:, :] + mask[..., :-1, :]
        total[..., :-1, :] = total[..., :-1, :] + value[..., 1:, :]
        count[..., :-1, :] = count[..., :-1, :] + mask[..., 1:, :]
        value = total / count.clamp_min(1.0)
        value = value * mask
    return value


def _normalize_route_seed(
    rigid: torch.Tensor,
    chi: torch.Tensor,
    node_mask: torch.Tensor,
    chi_mask: torch.Tensor,
    config: PhysicalPathOptimizationConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize each batch item's raw route field in the product metric."""
    rigid_metric = torch.cat(
        [
            rigid[..., :3] / float(config.rotation_metric_scale),
            rigid[..., 3:] / float(config.translation_metric_scale),
        ],
        dim=-1,
    )
    chi_metric = chi / float(config.chi_metric_scale)
    node_weight = node_mask.float().unsqueeze(0).unsqueeze(-1)
    chi_weight = chi_mask.float().unsqueeze(0)
    rigid_energy = (rigid_metric.square() * node_weight).sum(dim=(0, 2, 3))
    chi_energy = (chi_metric.square() * chi_weight).sum(dim=(0, 2, 3))
    rigid_dims = 3 if config.components == "translation" else 6
    rigid_count = node_weight.sum(dim=(0, 2, 3)) * rigid_dims
    chi_count = (
        chi_weight.sum(dim=(0, 2, 3))
        if config.components == "all"
        else torch.zeros_like(rigid_count)
    )
    rms = torch.sqrt(
        (rigid_energy + chi_energy)
        / (rigid_count + chi_count).clamp_min(1.0)
    ).clamp_min(1e-8)
    scale = float(config.route_seed_scale) / rms
    return (
        rigid * scale.view(1, -1, 1, 1),
        chi * scale.view(1, -1, 1, 1),
    )


def _route_initializations(
    times: Sequence[float],
    base_rigids: Sequence[Rigid],
    bridge_tangent_rigid: torch.Tensor,
    bridge_tangent_chi: torch.Tensor,
    node_mask: torch.Tensor,
    chi_mask: torch.Tensor,
    config: PhysicalPathOptimizationConfig,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create deterministic, paired, path-global low-frequency route seeds."""
    zero_rigid = torch.zeros_like(bridge_tangent_rigid)
    zero_chi = torch.zeros_like(bridge_tangent_chi)
    if config.num_starts == 1:
        return [(zero_rigid, zero_chi)]

    n_interior, batch_size, n_res, _ = bridge_tangent_rigid.shape
    base_rotations = torch.stack(
        [_rigid_to_rt(rigid)[0] for rigid in base_rigids[1:-1]],
        dim=0,
    ).to(
        device=bridge_tangent_rigid.device,
        dtype=bridge_tangent_rigid.dtype,
    )
    if tuple(base_rotations.shape[:3]) != (n_interior, batch_size, n_res):
        raise ValueError("base path rotations do not match bridge tangent shapes")
    rank = int(config.route_seed_rank)
    cpu_generator = torch.Generator(device="cpu")
    seeds: List[Tuple[torch.Tensor, torch.Tensor]] = []
    interior_times = torch.as_tensor(
        list(times[1:-1]), dtype=torch.float32, device="cpu"
    )
    temporal = torch.stack(
        [torch.sin((index + 1) * math.pi * interior_times) for index in range(rank)],
        dim=-1,
    )
    temporal = temporal / temporal.square().mean(dim=0, keepdim=True).sqrt().clamp_min(
        1e-8
    )

    pair_count = (int(config.num_starts) + 1) // 2
    for pair_index in range(pair_count):
        cpu_generator.manual_seed(int(config.route_seed) + pair_index)
        spatial_rigid = torch.randn(
            rank, batch_size, n_res, 6, generator=cpu_generator
        ).to(device=bridge_tangent_rigid.device, dtype=bridge_tangent_rigid.dtype)
        spatial_chi = torch.randn(
            rank,
            batch_size,
            n_res,
            bridge_tangent_chi.shape[-1],
            generator=cpu_generator,
        ).to(device=bridge_tangent_chi.device, dtype=bridge_tangent_chi.dtype)
        spatial_rigid = _smooth_chain_field(
            spatial_rigid,
            node_mask,
            int(config.route_seed_smoothing_steps),
        )
        spatial_chi = _smooth_chain_field(
            spatial_chi,
            node_mask,
            int(config.route_seed_smoothing_steps),
        )
        temporal_device = temporal.to(
            device=bridge_tangent_rigid.device,
            dtype=bridge_tangent_rigid.dtype,
        )
        rigid_global = torch.einsum(
            "mr,rbnd->mbnd", temporal_device, spatial_rigid
        )
        rigid = torch.cat(
            [
                torch.matmul(
                    base_rotations.transpose(-1, -2),
                    rigid_global[..., :3].unsqueeze(-1),
                ).squeeze(-1),
                torch.matmul(
                    base_rotations.transpose(-1, -2),
                    rigid_global[..., 3:].unsqueeze(-1),
                ).squeeze(-1),
            ],
            dim=-1,
        )
        chi = torch.einsum("mr,rbnd->mbnd", temporal_device, spatial_chi)
        if config.components == "translation":
            rigid[..., :3] = 0.0
            chi.zero_()
        elif config.components == "rigid":
            chi.zero_()
        chi = chi * chi_mask.unsqueeze(0).to(chi.dtype)
        rigid = rigid * node_mask.unsqueeze(0).unsqueeze(-1).to(rigid.dtype)
        rigid, chi = _normalize_route_seed(
            rigid,
            chi,
            node_mask,
            chi_mask,
            config,
        )
        seeds.append((rigid, chi))
        if len(seeds) < int(config.num_starts):
            seeds.append((-rigid, -chi))
    return seeds[: int(config.num_starts)]


def apply_projected_normal_residual(
    base_rigids: Sequence[Rigid],
    base_chi: Sequence[torch.Tensor],
    times: Sequence[float],
    raw_rigid: torch.Tensor,
    raw_chi: torch.Tensor,
    bridge_tangent_rigid: torch.Tensor,
    bridge_tangent_chi: torch.Tensor,
    node_mask: torch.Tensor,
    chi_mask: torch.Tensor,
    config: PhysicalPathOptimizationConfig,
) -> Tuple[List[Rigid], List[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply an endpoint-zero product-normal residual to a guide path."""
    n_interior, batch_size, n_res = _validate_path_shapes(
        base_rigids,
        base_chi,
        times,
        bridge_tangent_rigid,
        bridge_tangent_chi,
    )
    if tuple(raw_rigid.shape) != (n_interior, batch_size, n_res, 6):
        raise ValueError("raw_rigid shape does not match the guide path")
    if tuple(raw_chi.shape) != tuple(bridge_tangent_chi.shape):
        raise ValueError("raw_chi shape does not match the guide path")

    path_rigids: List[Rigid] = [base_rigids[0]]
    path_chi: List[torch.Tensor] = [base_chi[0]]
    applied_rigid: List[torch.Tensor] = []
    applied_chi: List[torch.Tensor] = []
    normal_cos: List[torch.Tensor] = []
    active_fraction: List[torch.Tensor] = []

    for interior_idx in range(n_interior):
        residual_rigid = raw_rigid[interior_idx]
        residual_chi = raw_chi[interior_idx]
        if config.components == "translation":
            residual_rigid = torch.cat(
                [torch.zeros_like(residual_rigid[..., :3]), residual_rigid[..., 3:]],
                dim=-1,
            )
            residual_chi = torch.zeros_like(residual_chi)
        elif config.components == "rigid":
            residual_chi = torch.zeros_like(residual_chi)
        projection_fn = {
            "product": project_product_tangent_normal,
            "block": project_block_tangent_normal,
        }[config.projection_mode]
        projection = projection_fn(
            residual_rigid,
            residual_chi,
            bridge_tangent_rigid[interior_idx],
            bridge_tangent_chi[interior_idx],
            node_mask=node_mask,
            chi_mask=chi_mask,
            rotation_scale=config.rotation_metric_scale,
            translation_scale=config.translation_metric_scale,
            chi_scale=config.chi_metric_scale,
            min_tangent_norm=config.min_tangent_norm,
            max_metric_norm=config.max_metric_norm,
        )
        t_value = float(times[interior_idx + 1])
        envelope = endpoint_zero_envelope(
            raw_rigid.new_tensor(t_value), kind=config.envelope
        )
        rigid_residual = projection["projected_rigid"] * envelope
        chi_residual = projection["projected_chi"] * envelope
        base_rotation, base_translation = _rigid_to_rt(base_rigids[interior_idx + 1])
        residual_rotation, residual_translation = se3_exp(rigid_residual)
        path_rotation, path_translation = rigid_compose(
            base_rotation,
            base_translation,
            residual_rotation,
            residual_translation,
        )
        path_rigids.append(_rt_to_rigid(path_rotation, path_translation))
        path_chi.append(wrap_to_pi(base_chi[interior_idx + 1] + chi_residual))
        applied_rigid.append(rigid_residual)
        applied_chi.append(chi_residual)
        normal_cos.append(projection["projected_parallel_cos_abs"])
        active_fraction.append(projection["active_mask"].float().mean())

    path_rigids.append(base_rigids[-1])
    path_chi.append(base_chi[-1])
    stats = {
        "normal_parallel_cos_abs": torch.stack(normal_cos).mean(),
        "normal_active_fraction": torch.stack(active_fraction).mean(),
    }
    return (
        path_rigids,
        path_chi,
        torch.stack(applied_rigid),
        torch.stack(applied_chi),
        stats,
    )


def deterministic_nonbonded_clash_loss(
    atom14_pos: torch.Tensor,
    atom14_mask: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    threshold: float,
    max_atoms: int,
) -> torch.Tensor:
    """Deterministic sampled heavy-atom clash loss excluding nearby residues."""
    losses: List[torch.Tensor] = []
    batch_size, n_res, n_atoms, _ = atom14_pos.shape
    residue_ids = torch.arange(n_res, device=atom14_pos.device).view(n_res, 1)
    residue_ids = residue_ids.expand(n_res, n_atoms).reshape(-1)
    for batch_idx in range(batch_size):
        valid = (atom14_mask[batch_idx] & node_mask[batch_idx].unsqueeze(-1)).reshape(-1)
        valid_indices = valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() < 2:
            continue
        if valid_indices.numel() > int(max_atoms):
            pick = torch.linspace(
                0,
                valid_indices.numel() - 1,
                steps=int(max_atoms),
                device=valid_indices.device,
            ).round().long().unique()
            valid_indices = valid_indices[pick]
        coords = atom14_pos[batch_idx].reshape(-1, 3)[valid_indices]
        selected_residue_ids = residue_ids[valid_indices]
        distances = torch.cdist(coords.float(), coords.float())
        pair_mask = torch.triu(
            torch.ones_like(distances, dtype=torch.bool), diagonal=1
        )
        pair_mask = pair_mask & (
            (selected_residue_ids[:, None] - selected_residue_ids[None, :]).abs() > 1
        )
        if not pair_mask.any():
            continue
        penetration = torch.relu(distances.new_tensor(float(threshold)) - distances)
        losses.append(penetration[pair_mask].square().mean())
    if not losses:
        return atom14_pos.new_tensor(0.0)
    return torch.stack(losses).mean().to(atom14_pos.dtype)


def _interpolate_backbone_torsions(batch, t_value: float) -> torch.Tensor:
    t = min(max(float(t_value), 0.0), 1.0)
    progress = 3.0 * t * t - 2.0 * t * t * t
    delta = wrap_to_pi(batch.torsion_holo[..., :3] - batch.torsion_apo[..., :3])
    return wrap_to_pi(batch.torsion_apo[..., :3] + progress * delta)


def _decode_atom14(fk_module, batch, rigid: Rigid, chi: torch.Tensor, t_value: float):
    backbone = _interpolate_backbone_torsions(batch, t_value)
    torsions = torch.cat([backbone, chi], dim=-1)
    torsions_sincos = torch.stack([torch.sin(torsions), torch.cos(torsions)], dim=-1)
    return fk_module(
        reorder_torsions_to_openfold(torsions_sincos),
        rigid,
        batch.aatype,
    )


def _reference_ligand_profiles(
    fk_module,
    batch,
    base_rigids: Sequence[Rigid],
    base_chi: Sequence[torch.Tensor],
    times: Sequence[float],
    config: PhysicalPathOptimizationConfig,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    contacts: List[torch.Tensor] = []
    distances: List[torch.Tensor] = []
    with torch.no_grad():
        for rigid, chi, t_value in zip(base_rigids[1:-1], base_chi[1:-1], times[1:-1]):
            atom14 = _decode_atom14(fk_module, batch, rigid, chi, float(t_value))
            valid = atom14["atom14_mask"].bool() & batch.node_mask.unsqueeze(-1).bool()
            distance = residue_ligand_min_dist(
                atom14["atom14_pos"].float(),
                valid,
                batch.lig_points.float(),
                batch.lig_mask.bool(),
            )
            contacts.append(
                soft_contact_from_dist(
                    distance,
                    d_c=config.contact_distance,
                    tau=config.contact_temperature,
                ).detach()
            )
            distances.append(distance.detach())
    return contacts, distances


def _aggregate_frame_values(
    values: Sequence[torch.Tensor],
    config: PhysicalPathOptimizationConfig,
) -> torch.Tensor:
    if not values:
        raise ValueError("cannot aggregate an empty frame sequence")
    stacked = torch.stack(list(values))
    if config.frame_aggregation == "mean":
        return stacked.mean()
    if config.frame_aggregation == "max":
        return stacked.max()
    beta = float(config.frame_softmax_beta)
    return (
        torch.logsumexp(stacked * beta, dim=0) - math.log(stacked.shape[0])
    ) / beta


def _physical_terms(
    fk_module,
    batch,
    path_rigids: Sequence[Rigid],
    path_chi: Sequence[torch.Tensor],
    times: Sequence[float],
    applied_rigid: torch.Tensor,
    applied_chi: torch.Tensor,
    reference_contacts: Sequence[torch.Tensor],
    reference_distances: Sequence[torch.Tensor],
    config: PhysicalPathOptimizationConfig,
) -> Dict[str, torch.Tensor]:
    peptide_terms: List[torch.Tensor] = []
    protein_clash_terms: List[torch.Tensor] = []
    ligand_clash_terms: List[torch.Tensor] = []
    contact_terms: List[torch.Tensor] = []
    distance_terms: List[torch.Tensor] = []

    pocket_mask = (
        batch.node_mask.bool() & (batch.w_res > float(config.pocket_threshold))
    )
    contact_mask = torch.where(
        pocket_mask.any(dim=-1, keepdim=True),
        pocket_mask,
        batch.node_mask.bool(),
    )
    for frame_idx, (rigid, chi, t_value) in enumerate(
        zip(path_rigids[1:-1], path_chi[1:-1], times[1:-1])
    ):
        atom14 = _decode_atom14(fk_module, batch, rigid, chi, float(t_value))
        positions = atom14["atom14_pos"].float()
        valid = atom14["atom14_mask"].bool() & batch.node_mask.unsqueeze(-1).bool()
        peptide_terms.append(
            compute_peptide_loss(
                positions,
                valid,
                batch.node_mask.bool(),
                peptide_bond_mask=batch.peptide_bond_mask.bool(),
            )
        )
        protein_clash_terms.append(
            deterministic_nonbonded_clash_loss(
                positions,
                valid,
                batch.node_mask.bool(),
                threshold=config.protein_clash_distance,
                max_atoms=config.max_clash_atoms,
            )
        )
        ligand_distance = residue_ligand_min_dist(
            positions,
            valid,
            batch.lig_points.float(),
            batch.lig_mask.bool(),
        )
        valid_ligand_distance = batch.node_mask.bool() & torch.isfinite(ligand_distance)
        ligand_penetration = torch.relu(
            ligand_distance.new_tensor(float(config.ligand_clash_distance))
            - ligand_distance.clamp(max=50.0)
        )
        ligand_clash_terms.append(
            (ligand_penetration.square() * valid_ligand_distance.float()).sum()
            / valid_ligand_distance.float().sum().clamp(min=1.0)
        )
        contact = soft_contact_from_dist(
            ligand_distance,
            d_c=config.contact_distance,
            tau=config.contact_temperature,
        )
        contact_terms.append(
            ((contact - reference_contacts[frame_idx]).square() * contact_mask.float()).sum()
            / contact_mask.float().sum().clamp(min=1.0)
        )
        distance_error = F.smooth_l1_loss(
            ligand_distance.clamp(max=50.0),
            reference_distances[frame_idx].clamp(max=50.0),
            reduction="none",
            beta=0.5,
        )
        distance_terms.append(
            (distance_error * contact_mask.float()).sum()
            / contact_mask.float().sum().clamp(min=1.0)
        )

    def mean_or_zero(values: Sequence[torch.Tensor]) -> torch.Tensor:
        if values:
            return torch.stack(list(values)).mean()
        return applied_rigid.new_tensor(0.0)

    def max_or_zero(values: Sequence[torch.Tensor]) -> torch.Tensor:
        if values:
            return torch.stack(list(values)).max()
        return applied_rigid.new_tensor(0.0)

    def aggregate_or_zero(values: Sequence[torch.Tensor]) -> torch.Tensor:
        if values:
            return _aggregate_frame_values(values, config)
        return applied_rigid.new_tensor(0.0)

    rigid_metric = torch.cat(
        [
            applied_rigid[..., :3] / float(config.rotation_metric_scale),
            applied_rigid[..., 3:] / float(config.translation_metric_scale),
        ],
        dim=-1,
    )
    chi_metric = applied_chi / float(config.chi_metric_scale)
    node_weight = batch.node_mask.float().unsqueeze(0)
    chi_weight = batch.chi_mask.float().unsqueeze(0)
    residual_magnitude = (
        rigid_metric.square().sum(dim=-1) * node_weight
    ).sum() / node_weight.sum().clamp(min=1.0)
    residual_magnitude = residual_magnitude + (
        chi_metric.square() * chi_weight
    ).sum() / chi_weight.sum().clamp(min=1.0)

    zero_rigid = torch.zeros_like(applied_rigid[:1])
    zero_chi = torch.zeros_like(applied_chi[:1])
    padded_rigid = torch.cat([zero_rigid, rigid_metric, zero_rigid], dim=0)
    padded_chi = torch.cat([zero_chi, chi_metric, zero_chi], dim=0)
    rigid_delta = padded_rigid[1:] - padded_rigid[:-1]
    chi_delta = padded_chi[1:] - padded_chi[:-1]
    temporal_smoothness = (
        rigid_delta.square().sum(dim=-1) * node_weight
    ).sum() / (node_weight.sum().clamp(min=1.0) * rigid_delta.shape[0])
    temporal_smoothness = temporal_smoothness + (
        chi_delta.square() * chi_weight
    ).sum() / (chi_weight.sum().clamp(min=1.0) * chi_delta.shape[0])

    return {
        "peptide": aggregate_or_zero(peptide_terms),
        "protein_clash": aggregate_or_zero(protein_clash_terms),
        "ligand_clash": aggregate_or_zero(ligand_clash_terms),
        "contact_anchor": mean_or_zero(contact_terms),
        "distance_anchor": mean_or_zero(distance_terms),
        "residual_magnitude": residual_magnitude,
        "temporal_smoothness": temporal_smoothness,
        "peptide_mean": mean_or_zero(peptide_terms),
        "peptide_max": max_or_zero(peptide_terms),
        "protein_clash_mean": mean_or_zero(protein_clash_terms),
        "protein_clash_max": max_or_zero(protein_clash_terms),
        "ligand_clash_mean": mean_or_zero(ligand_clash_terms),
        "ligand_clash_max": max_or_zero(ligand_clash_terms),
    }


def _term_weights(config: PhysicalPathOptimizationConfig) -> Dict[str, float]:
    return {
        "peptide": float(config.weight_peptide),
        "protein_clash": float(config.weight_protein_clash),
        "ligand_clash": float(config.weight_ligand_clash),
        "contact_anchor": float(config.weight_contact_anchor),
        "distance_anchor": float(config.weight_distance_anchor),
        "residual_magnitude": float(config.weight_residual_magnitude),
        "temporal_smoothness": float(config.weight_temporal_smoothness),
    }


def _objective(
    terms: Dict[str, torch.Tensor],
    scales: Dict[str, torch.Tensor],
    config: PhysicalPathOptimizationConfig,
) -> torch.Tensor:
    total = next(iter(terms.values())).new_tensor(0.0)
    for name, weight in _term_weights(config).items():
        if weight <= 0.0:
            continue
        total = total + float(weight) * terms[name] / scales[name]
    return total


def _evaluate_physical_candidate(
    fk_module,
    batch,
    base_rigids: Sequence[Rigid],
    base_chi: Sequence[torch.Tensor],
    times: Sequence[float],
    raw_rigid: torch.Tensor,
    raw_chi: torch.Tensor,
    bridge_tangent_rigid: torch.Tensor,
    bridge_tangent_chi: torch.Tensor,
    reference_contacts: Sequence[torch.Tensor],
    reference_distances: Sequence[torch.Tensor],
    scales: Dict[str, torch.Tensor],
    config: PhysicalPathOptimizationConfig,
):
    path = apply_projected_normal_residual(
        base_rigids,
        base_chi,
        times,
        raw_rigid,
        raw_chi,
        bridge_tangent_rigid,
        bridge_tangent_chi,
        batch.node_mask,
        batch.chi_mask,
        config,
    )
    terms = _physical_terms(
        fk_module,
        batch,
        path[0],
        path[1],
        times,
        path[2],
        path[3],
        reference_contacts,
        reference_distances,
        config,
    )
    return path, terms, _objective(terms, scales, config)


def _clipped_gradients(
    objective: torch.Tensor,
    parameters: Sequence[torch.Tensor],
    gradient_clip: float,
) -> Tuple[List[torch.Tensor], float]:
    raw_gradients = torch.autograd.grad(
        objective,
        list(parameters),
        allow_unused=True,
    )
    gradients = [
        gradient if gradient is not None else torch.zeros_like(parameter)
        for gradient, parameter in zip(raw_gradients, parameters)
    ]
    norm = torch.sqrt(
        sum(gradient.detach().float().square().sum() for gradient in gradients)
    )
    norm_value = float(norm.item())
    if gradient_clip > 0.0 and norm_value > float(gradient_clip):
        factor = float(gradient_clip) / max(norm_value, 1e-12)
        gradients = [gradient * factor for gradient in gradients]
    return gradients, norm_value


def _optimize_single_projected_normal_path(
    fk_module,
    batch,
    base_rigids: Sequence[Rigid],
    base_chi: Sequence[torch.Tensor],
    times: Sequence[float],
    bridge_tangent_rigid: torch.Tensor,
    bridge_tangent_chi: torch.Tensor,
    config: PhysicalPathOptimizationConfig,
    initial_raw_rigid: Optional[torch.Tensor] = None,
    initial_raw_chi: Optional[torch.Tensor] = None,
) -> PhysicalPathOptimizationResult:
    """Optimize one route seed, with the unmodified guide as a safe fallback."""
    config.validate()
    base_rigids = [_detach_rigid(rigid) for rigid in base_rigids]
    base_chi = [chi.detach() for chi in base_chi]
    bridge_tangent_rigid = bridge_tangent_rigid.detach()
    bridge_tangent_chi = bridge_tangent_chi.detach()
    n_interior, batch_size, n_res = _validate_path_shapes(
        base_rigids,
        base_chi,
        times,
        bridge_tangent_rigid,
        bridge_tangent_chi,
    )
    device = base_chi[0].device
    dtype = base_chi[0].dtype
    zero_rigid = torch.zeros(
        n_interior, batch_size, n_res, 6, device=device, dtype=dtype
    )
    zero_chi = torch.zeros_like(bridge_tangent_chi, device=device, dtype=dtype)
    if initial_raw_rigid is None:
        initial_raw_rigid = zero_rigid
    if initial_raw_chi is None:
        initial_raw_chi = zero_chi
    if tuple(initial_raw_rigid.shape) != tuple(zero_rigid.shape):
        raise ValueError("initial_raw_rigid shape does not match the guide path")
    if tuple(initial_raw_chi.shape) != tuple(zero_chi.shape):
        raise ValueError("initial_raw_chi shape does not match the guide path")
    raw_rigid = torch.nn.Parameter(
        initial_raw_rigid.detach().to(device=device, dtype=dtype).clone()
    )
    raw_chi = torch.nn.Parameter(
        initial_raw_chi.detach().to(device=device, dtype=dtype).clone()
    )
    reference_contacts, reference_distances = _reference_ligand_profiles(
        fk_module, batch, base_rigids, base_chi, times, config
    )

    with torch.enable_grad():
        guide_path = apply_projected_normal_residual(
            base_rigids,
            base_chi,
            times,
            zero_rigid,
            zero_chi,
            bridge_tangent_rigid,
            bridge_tangent_chi,
            batch.node_mask,
            batch.chi_mask,
            config,
        )
        guide_terms = _physical_terms(
            fk_module,
            batch,
            guide_path[0],
            guide_path[1],
            times,
            guide_path[2],
            guide_path[3],
            reference_contacts,
            reference_distances,
            config,
        )
        scales: Dict[str, torch.Tensor] = {}
        for name, value in guide_terms.items():
            if config.normalize_physical_terms and name in {
                "peptide",
                "protein_clash",
                "ligand_clash",
            }:
                scales[name] = value.detach().clamp_min(1e-3)
            else:
                scales[name] = value.detach().new_tensor(1.0)
        guide_objective = _objective(guide_terms, scales, config).detach()
        with torch.no_grad():
            _, start_terms, start_objective_tensor = _evaluate_physical_candidate(
                fk_module,
                batch,
                base_rigids,
                base_chi,
                times,
                raw_rigid,
                raw_chi,
                bridge_tangent_rigid,
                bridge_tangent_chi,
                reference_contacts,
                reference_distances,
                scales,
                config,
            )
        start_objective = float(start_objective_tensor.item())
        best_objective = float(guide_objective.item())
        best_rigid = zero_rigid.detach().clone()
        best_chi = zero_chi.detach().clone()
        if math.isfinite(start_objective) and start_objective < best_objective:
            best_objective = start_objective
            best_rigid.copy_(raw_rigid.detach())
            best_chi.copy_(raw_chi.detach())

        completed_iterations = 0
        accepted_steps = 0
        rejected_steps = 0
        objective_evaluations = 2
        final_step_size = float(config.learning_rate)
        max_gradient_norm = 0.0

        if config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                [raw_rigid, raw_chi], lr=config.learning_rate
            )
            for iteration in range(int(config.iterations)):
                optimizer.zero_grad(set_to_none=True)
                _, _, objective = _evaluate_physical_candidate(
                    fk_module,
                    batch,
                    base_rigids,
                    base_chi,
                    times,
                    raw_rigid,
                    raw_chi,
                    bridge_tangent_rigid,
                    bridge_tangent_chi,
                    reference_contacts,
                    reference_distances,
                    scales,
                    config,
                )
                objective_evaluations += 1
                if not torch.isfinite(objective):
                    rejected_steps += 1
                    break
                objective_value = float(objective.detach().item())
                if objective_value < best_objective:
                    best_objective = objective_value
                    with torch.no_grad():
                        best_rigid.copy_(raw_rigid)
                        best_chi.copy_(raw_chi)
                objective.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [raw_rigid, raw_chi], float(config.gradient_clip)
                ) if config.gradient_clip > 0.0 else torch.sqrt(
                    sum(
                        parameter.grad.detach().float().square().sum()
                        for parameter in (raw_rigid, raw_chi)
                        if parameter.grad is not None
                    )
                )
                max_gradient_norm = max(
                    max_gradient_norm, float(gradient_norm.detach().item())
                )
                optimizer.step()
                accepted_steps += 1
                completed_iterations = iteration + 1

            with torch.no_grad():
                _, _, candidate_objective_tensor = _evaluate_physical_candidate(
                    fk_module,
                    batch,
                    base_rigids,
                    base_chi,
                    times,
                    raw_rigid,
                    raw_chi,
                    bridge_tangent_rigid,
                    bridge_tangent_chi,
                    reference_contacts,
                    reference_distances,
                    scales,
                    config,
                )
                objective_evaluations += 1
                candidate_objective = float(candidate_objective_tensor.item())
                if math.isfinite(candidate_objective) and candidate_objective < best_objective:
                    best_objective = candidate_objective
                    best_rigid.copy_(raw_rigid)
                    best_chi.copy_(raw_chi)
        else:
            current_rigid = raw_rigid.detach().clone().requires_grad_(True)
            current_chi = raw_chi.detach().clone().requires_grad_(True)
            current_objective = start_objective
            for iteration in range(int(config.iterations)):
                _, _, objective = _evaluate_physical_candidate(
                    fk_module,
                    batch,
                    base_rigids,
                    base_chi,
                    times,
                    current_rigid,
                    current_chi,
                    bridge_tangent_rigid,
                    bridge_tangent_chi,
                    reference_contacts,
                    reference_distances,
                    scales,
                    config,
                )
                objective_evaluations += 1
                if not torch.isfinite(objective):
                    rejected_steps += 1
                    break
                current_objective = float(objective.detach().item())
                gradients, gradient_norm = _clipped_gradients(
                    objective,
                    [current_rigid, current_chi],
                    float(config.gradient_clip),
                )
                max_gradient_norm = max(max_gradient_norm, gradient_norm)
                if not math.isfinite(gradient_norm) or gradient_norm <= 1e-12:
                    break

                step_size = float(config.learning_rate)
                accepted = False
                tolerance = float(config.acceptance_tolerance) * max(
                    1.0, abs(current_objective)
                )
                for _ in range(int(config.line_search_steps)):
                    proposed_rigid = (
                        current_rigid - step_size * gradients[0]
                    ).detach()
                    proposed_chi = (
                        current_chi - step_size * gradients[1]
                    ).detach()
                    with torch.no_grad():
                        _, _, proposed_objective_tensor = (
                            _evaluate_physical_candidate(
                                fk_module,
                                batch,
                                base_rigids,
                                base_chi,
                                times,
                                proposed_rigid,
                                proposed_chi,
                                bridge_tangent_rigid,
                                bridge_tangent_chi,
                                reference_contacts,
                                reference_distances,
                                scales,
                                config,
                            )
                        )
                    objective_evaluations += 1
                    proposed_objective = float(proposed_objective_tensor.item())
                    if (
                        math.isfinite(proposed_objective)
                        and proposed_objective <= current_objective - tolerance
                    ):
                        current_rigid = proposed_rigid.requires_grad_(True)
                        current_chi = proposed_chi.requires_grad_(True)
                        current_objective = proposed_objective
                        final_step_size = step_size
                        accepted = True
                        accepted_steps += 1
                        if proposed_objective < best_objective:
                            best_objective = proposed_objective
                            best_rigid.copy_(proposed_rigid)
                            best_chi.copy_(proposed_chi)
                        break
                    step_size *= float(config.line_search_shrink)
                completed_iterations = iteration + 1
                if not accepted:
                    rejected_steps += 1
                    break

        final_path = apply_projected_normal_residual(
            base_rigids,
            base_chi,
            times,
            best_rigid,
            best_chi,
            bridge_tangent_rigid,
            bridge_tangent_chi,
            batch.node_mask,
            batch.chi_mask,
            config,
        )
        final_terms = _physical_terms(
            fk_module,
            batch,
            final_path[0],
            final_path[1],
            times,
            final_path[2],
            final_path[3],
            reference_contacts,
            reference_distances,
            config,
        )

    detached_rigids = [_detach_rigid(rigid) for rigid in final_path[0]]
    detached_chi = [chi.detach() for chi in final_path[1]]
    applied_rigid = final_path[2].detach()
    applied_chi = final_path[3].detach()
    envelope = torch.stack(
        [
            endpoint_zero_envelope(
                applied_rigid.new_tensor(float(t_value)), kind=config.envelope
            )
            for t_value in times[1:-1]
        ],
        dim=0,
    ).view(n_interior, 1, 1, 1)
    projected_rigid = applied_rigid / envelope.clamp_min(1e-8)
    projected_chi = applied_chi / envelope.clamp_min(1e-8)
    diagnostics: Dict[str, object] = {
        "iterations": float(completed_iterations),
        "optimizer": config.optimizer,
        "guide_objective": float(guide_objective.item()),
        "initial_objective": float(guide_objective.item()),
        "start_objective": float(start_objective),
        "final_objective": float(best_objective),
        "objective_improvement": float(guide_objective.item() - best_objective),
        "accepted_steps": int(accepted_steps),
        "rejected_steps": int(rejected_steps),
        "objective_evaluations": int(objective_evaluations),
        "final_step_size": float(final_step_size),
        "max_gradient_norm": float(max_gradient_norm),
        "normal_parallel_cos_abs": float(
            final_path[4]["normal_parallel_cos_abs"].detach().item()
        ),
        "normal_active_fraction": float(
            final_path[4]["normal_active_fraction"].detach().item()
        ),
        "applied_rotation_rms": float(
            final_path[2][..., :3].detach().square().mean().sqrt().item()
        ),
        "applied_translation_rms": float(
            final_path[2][..., 3:].detach().square().mean().sqrt().item()
        ),
        "applied_chi_rms": float(
            final_path[3].detach().square().mean().sqrt().item()
        ),
    }
    for name in guide_terms:
        diagnostics[f"initial_{name}"] = float(guide_terms[name].detach().item())
        diagnostics[f"final_{name}"] = float(final_terms[name].detach().item())
    return PhysicalPathOptimizationResult(
        rigids=detached_rigids,
        chi=detached_chi,
        times=[float(value) for value in times],
        projected_rigid=projected_rigid,
        projected_chi=projected_chi,
        applied_rigid=applied_rigid,
        applied_chi=applied_chi,
        diagnostics=diagnostics,
    )


def optimize_projected_normal_path(
    fk_module,
    batch,
    base_rigids: Sequence[Rigid],
    base_chi: Sequence[torch.Tensor],
    times: Sequence[float],
    bridge_tangent_rigid: torch.Tensor,
    bridge_tangent_chi: torch.Tensor,
    config: PhysicalPathOptimizationConfig,
) -> PhysicalPathOptimizationResult:
    """Search paired global routes and return the best safe physical refinement."""
    config.validate()
    seeds = _route_initializations(
        times,
        base_rigids,
        bridge_tangent_rigid,
        bridge_tangent_chi,
        batch.node_mask.bool(),
        batch.chi_mask.bool(),
        config,
    )
    results: List[PhysicalPathOptimizationResult] = []
    for rigid_seed, chi_seed in seeds:
        results.append(
            _optimize_single_projected_normal_path(
                fk_module,
                batch,
                base_rigids,
                base_chi,
                times,
                bridge_tangent_rigid,
                bridge_tangent_chi,
                config,
                initial_raw_rigid=rigid_seed,
                initial_raw_chi=chi_seed,
            )
        )
    selected_index = min(
        range(len(results)),
        key=lambda index: float(results[index].diagnostics["final_objective"]),
    )
    selected = results[selected_index]
    diagnostics = dict(selected.diagnostics)
    diagnostics.update(
        {
            "num_starts": int(config.num_starts),
            "selected_start": int(selected_index),
            "route_seed": int(config.route_seed),
            "route_seed_scale": float(config.route_seed_scale),
            "route_seed_rank": int(config.route_seed_rank),
            "route_seed_smoothing_steps": int(config.route_seed_smoothing_steps),
            "frame_aggregation": config.frame_aggregation,
            "start_objectives": [
                float(result.diagnostics["start_objective"]) for result in results
            ],
            "start_final_objectives": [
                float(result.diagnostics["final_objective"]) for result in results
            ],
            "starts_improved_over_guide": int(
                sum(
                    float(result.diagnostics["final_objective"])
                    < float(result.diagnostics["guide_objective"])
                    - float(config.acceptance_tolerance)
                    for result in results
                )
            ),
            "total_objective_evaluations": int(
                sum(int(result.diagnostics["objective_evaluations"]) for result in results)
            ),
        }
    )
    return PhysicalPathOptimizationResult(
        rigids=selected.rigids,
        chi=selected.chi,
        times=selected.times,
        projected_rigid=selected.projected_rigid,
        projected_chi=selected.projected_chi,
        applied_rigid=selected.applied_rigid,
        applied_chi=selected.applied_chi,
        diagnostics=diagnostics,
    )
