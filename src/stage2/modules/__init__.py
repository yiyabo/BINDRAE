"""
Stage-2 helper modules.
"""

from .se3 import (
    so3_log,
    so3_exp,
    se3_log,
    se3_exp,
    rigid_inverse,
    rigid_compose,
)
from .time_embed import SinusoidalTimeEmbedding
from .geometry import (
    wrap_to_pi,
    compute_peptide_loss,
    compute_contact_score,
    compute_w_eff,
)
from .phase_residual import (
    PHASE_WARP_VARIANTS,
    endpoint_zero_envelope,
    phase_tau_from_logits,
    project_block_tangent_normal,
    project_product_tangent_normal,
)
from .chain_internal import project_peptide_frame_translations
from .low_rank_residual import GraphCoupledLowRankResidualDecoder
from .physical_path import (
    PhysicalPathOptimizationConfig,
    PhysicalPathOptimizationResult,
    apply_projected_normal_residual,
    deterministic_nonbonded_clash_loss,
    optimize_projected_normal_path,
)

__all__ = [
    "so3_log",
    "so3_exp",
    "se3_log",
    "se3_exp",
    "rigid_inverse",
    "rigid_compose",
    "SinusoidalTimeEmbedding",
    "wrap_to_pi",
    "compute_peptide_loss",
    "compute_contact_score",
    "compute_w_eff",
    "PHASE_WARP_VARIANTS",
    "endpoint_zero_envelope",
    "phase_tau_from_logits",
    "project_block_tangent_normal",
    "project_product_tangent_normal",
    "project_peptide_frame_translations",
    "GraphCoupledLowRankResidualDecoder",
    "PhysicalPathOptimizationConfig",
    "PhysicalPathOptimizationResult",
    "apply_projected_normal_residual",
    "deterministic_nonbonded_clash_loss",
    "optimize_projected_normal_path",
]
