"""Geometry helpers for phase-warped, endpoint-zero path residuals."""

from typing import Dict, Optional

import torch


def endpoint_zero_envelope(t: torch.Tensor, kind: str = "poly") -> torch.Tensor:
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
