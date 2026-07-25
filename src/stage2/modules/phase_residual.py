"""Geometry helpers for phase-warped, endpoint-zero path residuals."""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


PHASE_WARP_VARIANTS = {
    "residue_monotone",
    "global_monotone",
    "global_chain_monotone",
    "chain_nonmonotone",
    "residue_nonmonotone",
}


def _smooth_chain_values(
    values: torch.Tensor,
    node_mask: torch.Tensor,
    peptide_bond_mask: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Average residue values over valid peptide-chain neighbors."""
    if peptide_bond_mask.shape != (
        node_mask.shape[0],
        max(node_mask.shape[1] - 1, 0),
    ):
        raise ValueError(
            "peptide_bond_mask must have shape [batch, residues - 1]"
        )
    valid = node_mask.bool()
    edge = peptide_bond_mask.bool()
    if valid.shape[1] > 1:
        edge = edge & valid[:, :-1] & valid[:, 1:]
    mask = valid.to(dtype=values.dtype).unsqueeze(0)
    result = values * mask
    for _ in range(int(steps)):
        total = result * mask
        degree = mask.expand_as(result).clone()
        if result.shape[-1] > 1:
            edge_f = edge.to(dtype=result.dtype).unsqueeze(0)
            total[..., :-1] = total[..., :-1] + result[..., 1:] * edge_f
            total[..., 1:] = total[..., 1:] + result[..., :-1] * edge_f
            degree[..., :-1] = degree[..., :-1] + edge_f
            degree[..., 1:] = degree[..., 1:] + edge_f
        result = torch.where(
            mask.bool(),
            total / degree.clamp_min(1.0),
            torch.zeros_like(total),
        )
    return result


def phase_tau_from_logits(
    logits: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    variant: str = "residue_monotone",
    logit_scale: float = 1.0,
    rate_eps: float = 1e-3,
    rate_clip: float = 10.0,
    nonmonotone_max_offset: float = 0.5,
    peptide_bond_mask: Optional[torch.Tensor] = None,
    chain_residual_scale: float = 1.0,
    chain_smoothing_steps: int = 2,
) -> Dict[str, torch.Tensor]:
    """Convert matched per-residue logits into endpoint-fixed phase values.

    All variants consume the same ``[steps, batch, residues]`` logits. The
    global control averages valid residue logits before applying the same
    monotone construction. The chain-coupled controls add centered residue
    corrections after peptide-neighbor smoothing. The non-monotone controls
    predict bounded direct offsets from the uniform time grid, so they change
    the inductive bias without adding parameters.
    """
    if logits.ndim != 3:
        raise ValueError(
            "phase logits must have shape [steps, batch, residues], got "
            f"{tuple(logits.shape)}"
        )
    if node_mask.shape != logits.shape[1:]:
        raise ValueError(
            "node_mask must match the batch/residue dimensions of phase logits"
        )
    if variant not in PHASE_WARP_VARIANTS:
        raise ValueError(f"Unsupported phase warp variant: {variant}")
    if float(logit_scale) <= 0.0:
        raise ValueError("logit_scale must be > 0")
    if float(rate_eps) <= 0.0:
        raise ValueError("rate_eps must be > 0")
    if float(rate_clip) < 0.0:
        raise ValueError("rate_clip must be >= 0")
    if not 0.0 < float(nonmonotone_max_offset) <= 1.0:
        raise ValueError("nonmonotone_max_offset must be in (0, 1]")
    if float(chain_residual_scale) < 0.0:
        raise ValueError("chain_residual_scale must be >= 0")
    if int(chain_smoothing_steps) < 0:
        raise ValueError("chain_smoothing_steps must be >= 0")

    node_mask = node_mask.bool()
    mask_f = node_mask.to(dtype=logits.dtype)
    effective_logits = logits
    if variant in {"global_monotone", "global_chain_monotone"}:
        denominator = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        global_logits = (logits * mask_f.unsqueeze(0)).sum(
            dim=-1, keepdim=True
        ) / denominator.unsqueeze(0)
        effective_logits = global_logits.expand_as(logits)
        if variant == "global_chain_monotone":
            if peptide_bond_mask is None:
                raise ValueError(
                    "global_chain_monotone requires peptide_bond_mask"
                )
            residue_logits = _smooth_chain_values(
                logits - effective_logits,
                node_mask,
                peptide_bond_mask,
                int(chain_smoothing_steps),
            )
            residue_mean = (residue_logits * mask_f.unsqueeze(0)).sum(
                dim=-1, keepdim=True
            ) / denominator.unsqueeze(0)
            residue_logits = (residue_logits - residue_mean) * mask_f.unsqueeze(0)
            effective_logits = effective_logits + float(
                chain_residual_scale
            ) * torch.tanh(residue_logits)
    elif variant == "chain_nonmonotone":
        if peptide_bond_mask is None:
            raise ValueError("chain_nonmonotone requires peptide_bond_mask")
        denominator = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        global_logits = (logits * mask_f.unsqueeze(0)).sum(
            dim=-1, keepdim=True
        ) / denominator.unsqueeze(0)
        residue_logits = _smooth_chain_values(
            logits - global_logits,
            node_mask,
            peptide_bond_mask,
            int(chain_smoothing_steps),
        )
        residue_mean = (residue_logits * mask_f.unsqueeze(0)).sum(
            dim=-1, keepdim=True
        ) / denominator.unsqueeze(0)
        residue_logits = (residue_logits - residue_mean) * mask_f.unsqueeze(0)
        effective_logits = global_logits + float(
            chain_residual_scale
        ) * residue_logits

    if variant in {
        "residue_monotone",
        "global_monotone",
        "global_chain_monotone",
    }:
        rates = F.softplus(effective_logits * float(logit_scale)) + float(rate_eps)
        if float(rate_clip) > 0.0:
            rates = rates.clamp(max=float(rate_clip))
        rates = torch.where(node_mask.unsqueeze(0), rates, torch.ones_like(rates))
        cumulative = torch.cumsum(rates, dim=0)
        total_rate = cumulative[-1].clamp_min(float(rate_eps))
        tau = torch.cat(
            [
                torch.zeros_like(cumulative[:1]),
                (cumulative / total_rate.unsqueeze(0)).clamp(0.0, 1.0),
            ],
            dim=0,
        )
        return {
            "tau": tau,
            "interval_rate": rates,
            "effective_logits": effective_logits,
        }

    n_steps = logits.shape[0]
    grid = torch.linspace(
        0.0,
        1.0,
        steps=n_steps + 1,
        device=logits.device,
        dtype=logits.dtype,
    ).view(n_steps + 1, 1, 1)
    interior_grid = grid[1:]
    envelope = torch.sin(torch.pi * interior_grid)
    offsets = (
        float(nonmonotone_max_offset)
        * envelope
        * torch.tanh(effective_logits * float(logit_scale))
    )
    predicted = (interior_grid + offsets).clamp(0.0, 1.0)
    predicted = torch.cat(
        [predicted[:-1], torch.ones_like(predicted[-1:])], dim=0
    )
    tau = torch.cat([torch.zeros_like(predicted[:1]), predicted], dim=0)
    identity_tau = grid.expand_as(tau)
    tau = torch.where(node_mask.unsqueeze(0), tau, identity_tau)
    interval_rate = (tau[1:] - tau[:-1]) * float(n_steps)
    return {
        "tau": tau,
        "interval_rate": interval_rate,
        "effective_logits": effective_logits,
    }


def endpoint_zero_envelope(t: torch.Tensor, kind: str = "sin2") -> torch.Tensor:
    """Return a unit-peak envelope that is exactly zero at both endpoints."""
    t_clamped = t.float().clamp(0.0, 1.0)
    if kind == "poly":
        value = 4.0 * t_clamped * (1.0 - t_clamped)
    elif kind == "sin2":
        value = torch.sin(torch.pi * t_clamped).square()
    else:
        raise ValueError(f"Unsupported endpoint-zero envelope: {kind}")
    interior = (t_clamped > 0.0) & (t_clamped < 1.0)
    return torch.where(interior, value, torch.zeros_like(value))


def _validate_scale(name: str, value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _metric_coordinates(
    rigid: torch.Tensor,
    chi: torch.Tensor,
    *,
    rotation_scale: float,
    translation_scale: float,
    chi_scale: float,
) -> torch.Tensor:
    return torch.cat(
        [
            rigid[..., :3] / rotation_scale,
            rigid[..., 3:] / translation_scale,
            chi / chi_scale,
        ],
        dim=-1,
    )


def project_product_tangent_normal(
    residual_rigid: torch.Tensor,
    residual_chi: torch.Tensor,
    bridge_tangent_rigid: torch.Tensor,
    bridge_tangent_chi: torch.Tensor,
    *,
    node_mask: Optional[torch.Tensor] = None,
    chi_mask: Optional[torch.Tensor] = None,
    rotation_scale: float = 1.0,
    translation_scale: float = 1.0,
    chi_scale: float = 1.0,
    min_tangent_norm: float = 1e-4,
    max_metric_norm: float = 0.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Project a residual off the endpoint-bridge tangent.

    The inner product lives on the local product tangent space
    ``se(3) x R^k``. Characteristic scales make radians and Angstroms
    comparable without changing the returned tensors' native units.
    Residues whose endpoint displacement is effectively zero are disabled;
    an off-bridge path is not identifiable for those residues.
    """
    if residual_rigid.shape != bridge_tangent_rigid.shape:
        raise ValueError("residual_rigid and bridge_tangent_rigid must match")
    if residual_chi.shape != bridge_tangent_chi.shape:
        raise ValueError("residual_chi and bridge_tangent_chi must match")
    if residual_rigid.shape[:-1] != residual_chi.shape[:-1]:
        raise ValueError("rigid and chi tensors must share leading dimensions")
    if residual_rigid.shape[-1] != 6:
        raise ValueError("rigid tangents must have final dimension 6")

    rotation_scale = _validate_scale("rotation_scale", rotation_scale)
    translation_scale = _validate_scale("translation_scale", translation_scale)
    chi_scale = _validate_scale("chi_scale", chi_scale)
    min_tangent_norm = float(min_tangent_norm)
    if min_tangent_norm < 0.0:
        raise ValueError("min_tangent_norm must be >= 0")
    max_metric_norm = float(max_metric_norm)
    if max_metric_norm < 0.0:
        raise ValueError("max_metric_norm must be >= 0")

    if node_mask is None:
        node_mask = torch.ones(
            residual_rigid.shape[:-1],
            dtype=torch.bool,
            device=residual_rigid.device,
        )
    else:
        node_mask = node_mask.bool()
    if node_mask.shape != residual_rigid.shape[:-1]:
        raise ValueError("node_mask shape must match residual leading dimensions")

    if chi_mask is None:
        chi_mask = torch.ones_like(residual_chi, dtype=torch.bool)
    else:
        chi_mask = chi_mask.bool()
    if chi_mask.shape != residual_chi.shape:
        raise ValueError("chi_mask shape must match residual_chi")
    chi_mask = chi_mask & node_mask.unsqueeze(-1)

    rigid_mask = node_mask.unsqueeze(-1).to(dtype=residual_rigid.dtype)
    chi_mask_f = chi_mask.to(dtype=residual_chi.dtype)
    residual_rigid = residual_rigid * rigid_mask
    bridge_tangent_rigid = bridge_tangent_rigid * rigid_mask
    residual_chi = residual_chi * chi_mask_f
    bridge_tangent_chi = bridge_tangent_chi * chi_mask_f

    residual_metric = _metric_coordinates(
        residual_rigid,
        residual_chi,
        rotation_scale=rotation_scale,
        translation_scale=translation_scale,
        chi_scale=chi_scale,
    )
    tangent_metric = _metric_coordinates(
        bridge_tangent_rigid,
        bridge_tangent_chi,
        rotation_scale=rotation_scale,
        translation_scale=translation_scale,
        chi_scale=chi_scale,
    )

    tangent_norm_sq = tangent_metric.square().sum(dim=-1)
    residual_norm_sq = residual_metric.square().sum(dim=-1)
    dot = (residual_metric * tangent_metric).sum(dim=-1)
    active_mask = node_mask & (tangent_norm_sq >= min_tangent_norm**2)
    coefficient = torch.where(
        active_mask,
        dot / tangent_norm_sq.clamp_min(eps),
        torch.zeros_like(dot),
    )

    projected_rigid = residual_rigid - coefficient.unsqueeze(-1) * bridge_tangent_rigid
    projected_chi = residual_chi - coefficient.unsqueeze(-1) * bridge_tangent_chi
    active_f = active_mask.unsqueeze(-1).to(dtype=projected_rigid.dtype)
    projected_rigid = projected_rigid * active_f
    projected_chi = projected_chi * active_mask.unsqueeze(-1).to(projected_chi.dtype)

    projected_metric = _metric_coordinates(
        projected_rigid,
        projected_chi,
        rotation_scale=rotation_scale,
        translation_scale=translation_scale,
        chi_scale=chi_scale,
    )
    projected_norm_sq = projected_metric.square().sum(dim=-1)
    if max_metric_norm > 0.0:
        projected_norm = projected_norm_sq.clamp_min(eps).sqrt()
        cap_scale = (max_metric_norm / projected_norm).clamp(max=1.0)
        cap_scale = torch.where(active_mask, cap_scale, torch.zeros_like(cap_scale))
        projected_rigid = projected_rigid * cap_scale.unsqueeze(-1)
        projected_chi = projected_chi * cap_scale.unsqueeze(-1)
        projected_metric = _metric_coordinates(
            projected_rigid,
            projected_chi,
            rotation_scale=rotation_scale,
            translation_scale=translation_scale,
            chi_scale=chi_scale,
        )
        projected_norm_sq = projected_metric.square().sum(dim=-1)
    projected_dot = (projected_metric * tangent_metric).sum(dim=-1)
    raw_parallel_cos = dot.abs() / (
        residual_norm_sq.sqrt() * tangent_norm_sq.sqrt()
    ).clamp_min(eps)
    projected_parallel_cos = projected_dot.abs() / (
        projected_norm_sq.sqrt() * tangent_norm_sq.sqrt()
    ).clamp_min(eps)
    zeros = torch.zeros_like(raw_parallel_cos)

    return {
        "projected_rigid": projected_rigid,
        "projected_chi": projected_chi,
        "projection_coefficient": coefficient,
        "active_mask": active_mask,
        "tangent_metric_norm": tangent_norm_sq.sqrt(),
        "raw_residual_metric_norm": residual_norm_sq.sqrt(),
        "projected_residual_metric_norm": projected_norm_sq.sqrt(),
        "raw_parallel_cos_abs": torch.where(active_mask, raw_parallel_cos, zeros),
        "projected_parallel_cos_abs": torch.where(
            active_mask,
            projected_parallel_cos,
            zeros,
        ),
    }


def _project_metric_block(
    residual: torch.Tensor,
    tangent: torch.Tensor,
    *,
    element_mask: torch.Tensor,
    active_mask: torch.Tensor,
    scale: float,
    min_tangent_norm: float,
    eps: float,
) -> Dict[str, torch.Tensor]:
    """Project one product-manifold block without coupling other blocks."""
    mask_f = element_mask.to(dtype=residual.dtype)
    residual = residual * mask_f
    tangent = tangent * mask_f
    residual_metric = residual / scale
    tangent_metric = tangent / scale
    residual_norm_sq = residual_metric.square().sum(dim=-1)
    tangent_norm_sq = tangent_metric.square().sum(dim=-1)
    dot = (residual_metric * tangent_metric).sum(dim=-1)
    # The public threshold selects residues by total endpoint motion. Reusing
    # that comparatively large threshold inside each block would leave small
    # but nonzero block tangents unprojected. Within an active residue, project
    # every numerically defined block direction instead.
    block_tangent_floor = min(float(min_tangent_norm), 1e-6)
    tangent_active = active_mask & (
        tangent_norm_sq > block_tangent_floor**2
    )
    denominator_floor = torch.finfo(tangent_norm_sq.dtype).tiny
    coefficient = torch.where(
        tangent_active,
        dot / tangent_norm_sq.clamp_min(denominator_floor),
        torch.zeros_like(dot),
    )
    projected = residual - coefficient.unsqueeze(-1) * tangent
    projected = projected * active_mask.unsqueeze(-1).to(projected.dtype) * mask_f
    projected_metric = projected / scale
    projected_norm_sq = projected_metric.square().sum(dim=-1)
    projected_dot = (projected_metric * tangent_metric).sum(dim=-1)
    raw_parallel_cos = dot.abs() / (
        residual_norm_sq.sqrt() * tangent_norm_sq.sqrt()
    ).clamp_min(eps)
    projected_parallel_cos = projected_dot.abs() / (
        projected_norm_sq.sqrt() * tangent_norm_sq.sqrt()
    ).clamp_min(eps)
    zeros = torch.zeros_like(dot)
    return {
        "projected": projected,
        "coefficient": coefficient,
        "tangent_active_mask": tangent_active,
        "tangent_metric_norm_sq": tangent_norm_sq,
        "raw_residual_metric_norm_sq": residual_norm_sq,
        "projected_residual_metric_norm_sq": projected_norm_sq,
        "raw_parallel_cos_abs": torch.where(
            tangent_active, raw_parallel_cos, zeros
        ),
        "projected_parallel_cos_abs": torch.where(
            tangent_active, projected_parallel_cos, zeros
        ),
    }


def project_block_tangent_normal(
    residual_rigid: torch.Tensor,
    residual_chi: torch.Tensor,
    bridge_tangent_rigid: torch.Tensor,
    bridge_tangent_chi: torch.Tensor,
    *,
    node_mask: Optional[torch.Tensor] = None,
    chi_mask: Optional[torch.Tensor] = None,
    rotation_scale: float = 1.0,
    translation_scale: float = 1.0,
    chi_scale: float = 1.0,
    min_tangent_norm: float = 1e-4,
    max_metric_norm: float = 0.0,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Project rotation, translation, and chi blocks independently.

    A shared product-space projection can remove a chi-parallel component by
    introducing a compensating translation. This stricter decomposition gives
    each block its own projection coefficient, while preserving the original
    residue-level support contract. If one block has negligible endpoint
    motion but another block is active, that block's residual is already normal
    and is retained unchanged.
    """
    if residual_rigid.shape != bridge_tangent_rigid.shape:
        raise ValueError("residual_rigid and bridge_tangent_rigid must match")
    if residual_chi.shape != bridge_tangent_chi.shape:
        raise ValueError("residual_chi and bridge_tangent_chi must match")
    if residual_rigid.shape[:-1] != residual_chi.shape[:-1]:
        raise ValueError("rigid and chi tensors must share leading dimensions")
    if residual_rigid.shape[-1] != 6:
        raise ValueError("rigid tangents must have final dimension 6")

    rotation_scale = _validate_scale("rotation_scale", rotation_scale)
    translation_scale = _validate_scale("translation_scale", translation_scale)
    chi_scale = _validate_scale("chi_scale", chi_scale)
    min_tangent_norm = float(min_tangent_norm)
    if min_tangent_norm < 0.0:
        raise ValueError("min_tangent_norm must be >= 0")
    max_metric_norm = float(max_metric_norm)
    if max_metric_norm < 0.0:
        raise ValueError("max_metric_norm must be >= 0")

    if node_mask is None:
        node_mask = torch.ones(
            residual_rigid.shape[:-1],
            dtype=torch.bool,
            device=residual_rigid.device,
        )
    else:
        node_mask = node_mask.bool()
    if node_mask.shape != residual_rigid.shape[:-1]:
        raise ValueError("node_mask shape must match residual leading dimensions")

    if chi_mask is None:
        chi_mask = torch.ones_like(residual_chi, dtype=torch.bool)
    else:
        chi_mask = chi_mask.bool()
    if chi_mask.shape != residual_chi.shape:
        raise ValueError("chi_mask shape must match residual_chi")
    chi_mask = chi_mask & node_mask.unsqueeze(-1)

    rigid_mask = node_mask.unsqueeze(-1).expand_as(residual_rigid)
    rotation_mask = rigid_mask[..., :3]
    translation_mask = rigid_mask[..., 3:]
    tangent_metric = _metric_coordinates(
        bridge_tangent_rigid * rigid_mask.to(bridge_tangent_rigid.dtype),
        bridge_tangent_chi * chi_mask.to(bridge_tangent_chi.dtype),
        rotation_scale=rotation_scale,
        translation_scale=translation_scale,
        chi_scale=chi_scale,
    )
    tangent_norm_sq = tangent_metric.square().sum(dim=-1)
    active_mask = node_mask & (tangent_norm_sq >= min_tangent_norm**2)

    rotation = _project_metric_block(
        residual_rigid[..., :3],
        bridge_tangent_rigid[..., :3],
        element_mask=rotation_mask,
        active_mask=active_mask,
        scale=rotation_scale,
        min_tangent_norm=min_tangent_norm,
        eps=eps,
    )
    translation = _project_metric_block(
        residual_rigid[..., 3:],
        bridge_tangent_rigid[..., 3:],
        element_mask=translation_mask,
        active_mask=active_mask,
        scale=translation_scale,
        min_tangent_norm=min_tangent_norm,
        eps=eps,
    )
    chi = _project_metric_block(
        residual_chi,
        bridge_tangent_chi,
        element_mask=chi_mask,
        active_mask=active_mask,
        scale=chi_scale,
        min_tangent_norm=min_tangent_norm,
        eps=eps,
    )

    projected_rigid = torch.cat(
        [rotation["projected"], translation["projected"]], dim=-1
    )
    projected_chi = chi["projected"]
    residual_metric = _metric_coordinates(
        residual_rigid * rigid_mask.to(residual_rigid.dtype),
        residual_chi * chi_mask.to(residual_chi.dtype),
        rotation_scale=rotation_scale,
        translation_scale=translation_scale,
        chi_scale=chi_scale,
    )
    projected_metric = _metric_coordinates(
        projected_rigid,
        projected_chi,
        rotation_scale=rotation_scale,
        translation_scale=translation_scale,
        chi_scale=chi_scale,
    )
    residual_norm_sq = residual_metric.square().sum(dim=-1)
    projected_norm_sq = projected_metric.square().sum(dim=-1)
    if max_metric_norm > 0.0:
        projected_norm = projected_norm_sq.clamp_min(eps).sqrt()
        cap_scale = (max_metric_norm / projected_norm).clamp(max=1.0)
        cap_scale = torch.where(active_mask, cap_scale, torch.zeros_like(cap_scale))
        projected_rigid = projected_rigid * cap_scale.unsqueeze(-1)
        projected_chi = projected_chi * cap_scale.unsqueeze(-1)
        projected_metric = _metric_coordinates(
            projected_rigid,
            projected_chi,
            rotation_scale=rotation_scale,
            translation_scale=translation_scale,
            chi_scale=chi_scale,
        )
        projected_norm_sq = projected_metric.square().sum(dim=-1)

    raw_dot = (residual_metric * tangent_metric).sum(dim=-1)
    projected_dot = (projected_metric * tangent_metric).sum(dim=-1)
    raw_parallel_cos = raw_dot.abs() / (
        residual_norm_sq.sqrt() * tangent_norm_sq.sqrt()
    ).clamp_min(eps)
    projected_parallel_cos = projected_dot.abs() / (
        projected_norm_sq.sqrt() * tangent_norm_sq.sqrt()
    ).clamp_min(eps)
    zeros = torch.zeros_like(raw_dot)

    return {
        "projected_rigid": projected_rigid,
        "projected_chi": projected_chi,
        "projection_coefficient": torch.stack(
            [
                rotation["coefficient"],
                translation["coefficient"],
                chi["coefficient"],
            ],
            dim=-1,
        ),
        "rotation_projection_coefficient": rotation["coefficient"],
        "translation_projection_coefficient": translation["coefficient"],
        "chi_projection_coefficient": chi["coefficient"],
        "active_mask": active_mask,
        "rotation_tangent_active_mask": rotation["tangent_active_mask"],
        "translation_tangent_active_mask": translation["tangent_active_mask"],
        "chi_tangent_active_mask": chi["tangent_active_mask"],
        "tangent_metric_norm": tangent_norm_sq.sqrt(),
        "raw_residual_metric_norm": residual_norm_sq.sqrt(),
        "projected_residual_metric_norm": projected_norm_sq.sqrt(),
        "raw_parallel_cos_abs": torch.where(active_mask, raw_parallel_cos, zeros),
        "projected_parallel_cos_abs": torch.where(
            active_mask, projected_parallel_cos, zeros
        ),
        "rotation_raw_parallel_cos_abs": rotation["raw_parallel_cos_abs"],
        "translation_raw_parallel_cos_abs": translation["raw_parallel_cos_abs"],
        "chi_raw_parallel_cos_abs": chi["raw_parallel_cos_abs"],
        "rotation_projected_parallel_cos_abs": rotation[
            "projected_parallel_cos_abs"
        ],
        "translation_projected_parallel_cos_abs": translation[
            "projected_parallel_cos_abs"
        ],
        "chi_projected_parallel_cos_abs": chi["projected_parallel_cos_abs"],
    }
