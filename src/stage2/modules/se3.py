"""
SE(3) utilities for Stage-2 (torch).
"""

from typing import Tuple

import torch


_SE3_COEFF_SERIES_THRESH = 1e-2
_SE3_V_INV_SERIES_THRESH = 1e-1


def _skew(v: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix from vectors [..., 3]."""
    zero = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack(
        [
            torch.stack([zero, -vz, vy], dim=-1),
            torch.stack([vz, zero, -vx], dim=-1),
            torch.stack([-vy, vx, zero], dim=-1),
        ],
        dim=-2,
    )


def _se3_v_coefficients(theta: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stable A/B/C coefficients for the SE(3) V matrix.

    For float32, the closed-form expressions become numerically unstable well
    before theta reaches 1e-4 because (1 - cos(theta)) and (1 - sin(theta)/theta)
    suffer catastrophic cancellation. Use series expansions over a wider range.
    """
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2

    A_series = 1.0 - theta2 / 6.0 + theta4 / 120.0 - theta6 / 5040.0
    B_series = 0.5 - theta2 / 24.0 + theta4 / 720.0 - theta6 / 40320.0
    C_series = 1.0 / 6.0 - theta2 / 120.0 + theta4 / 5040.0 - theta6 / 362880.0

    safe_theta = theta.clamp(min=eps)
    safe_theta2 = theta2.clamp(min=eps)
    A_exact = torch.sin(theta) / safe_theta
    B_exact = (1.0 - torch.cos(theta)) / safe_theta2
    C_exact = (1.0 - A_exact) / safe_theta2

    use_series = theta < _SE3_COEFF_SERIES_THRESH
    A = torch.where(use_series, A_series, A_exact)
    B = torch.where(use_series, B_series, B_exact)
    C = torch.where(use_series, C_series, C_exact)
    return A, B, C


def _se3_v_inv_factor(theta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Stable scalar factor in V^{-1} = I - 0.5 K + factor * K^2.

    The exact expression is numerically unusable in float32 for small angles, so
    we switch to a series expansion long before the mathematical limit.
    """
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2

    factor_series = (
        1.0 / 12.0
        + theta2 / 720.0
        + theta4 / 30240.0
        + theta6 / 1209600.0
    )

    safe_theta2 = theta2.clamp(min=eps)
    safe_B = B.clamp(min=eps)
    factor_exact = (1.0 - A / (2.0 * safe_B)) / safe_theta2
    use_series = theta < _SE3_V_INV_SERIES_THRESH
    return torch.where(use_series, factor_series, factor_exact)


def so3_exp(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Exponential map from so(3) to SO(3)."""
    theta = torch.norm(omega, dim=-1, keepdim=True).clamp(min=eps)
    axis = omega / theta
    K = _skew(axis)
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    R = eye + sin_t * K + (1.0 - cos_t) * (K @ K)

    small = (theta.squeeze(-1) < 1e-4)[..., None, None]
    if small.any():
        K_omega = _skew(omega)
        R_small = eye + K_omega + 0.5 * (K_omega @ K_omega)
        R = torch.where(small, R_small, R)
    return R


def so3_log(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Log map from SO(3) to so(3), returns axis-angle vector."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta_raw = ((trace - 1.0) / 2.0).clamp(min=-1.0, max=1.0)
    cos_theta = cos_theta_raw.clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
    theta = torch.acos(cos_theta)

    omega_hat = 0.5 * (R - R.transpose(-2, -1))
    omega = torch.stack(
        [omega_hat[..., 2, 1], omega_hat[..., 0, 2], omega_hat[..., 1, 0]],
        dim=-1,
    )

    near_pi = (cos_theta_raw < -0.9999)
    small = (cos_theta_raw > 1.0 - 1e-6)

    # Standard case: 0 < theta < pi
    sin_theta = torch.sin(theta).clamp(min=eps)
    scale = torch.where(
        near_pi.squeeze(-1),
        torch.zeros_like(theta).squeeze(-1),
        (theta / sin_theta).squeeze(-1)
    )
    omega = omega * scale.unsqueeze(-1)

    # Near-pi case: use diagonal elements to extract axis
    if near_pi.any():
        mask = near_pi.squeeze(-1)
        diag0 = R[..., 0, 0]
        diag1 = R[..., 1, 1]
        diag2 = R[..., 2, 2]

        # Choose the largest diagonal for numerical stability
        max_diag = torch.stack([diag0, diag1, diag2], dim=-1)
        idx = max_diag.argmax(dim=-1)

        def extract_axis(i):
            denom = (2.0 * (R[..., i, i] + 1.0)).clamp(min=eps).sqrt()
            axis_i = torch.zeros_like(omega)
            if i == 0:
                axis_i[..., 0] = (R[..., 0, 0] + 1.0) / denom
                axis_i[..., 1] = R[..., 0, 1] / denom
                axis_i[..., 2] = R[..., 0, 2] / denom
            elif i == 1:
                axis_i[..., 0] = R[..., 1, 0] / denom
                axis_i[..., 1] = (R[..., 1, 1] + 1.0) / denom
                axis_i[..., 2] = R[..., 1, 2] / denom
            else:
                axis_i[..., 0] = R[..., 2, 0] / denom
                axis_i[..., 1] = R[..., 2, 1] / denom
                axis_i[..., 2] = (R[..., 2, 2] + 1.0) / denom
            return axis_i

        axis0 = extract_axis(0)
        axis1 = extract_axis(1)
        axis2 = extract_axis(2)

        axis = torch.where(
            (idx == 0).unsqueeze(-1), axis0,
            torch.where((idx == 1).unsqueeze(-1), axis1, axis2)
        )
        omega_pi = axis * torch.pi
        omega = torch.where(mask.unsqueeze(-1), omega_pi, omega)

    # Small angle case
    if small.any():
        mask = small.squeeze(-1)
        omega_small = torch.stack(
            [omega_hat[..., 2, 1], omega_hat[..., 0, 2], omega_hat[..., 1, 0]],
            dim=-1,
        )
        omega = torch.where(mask.unsqueeze(-1), omega_small, omega)

    return omega


def se3_exp(xi: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exponential map from se(3) to SE(3).
    xi: [..., 6] (omega, v)
    Returns (R, t).
    """
    omega, v = xi[..., :3], xi[..., 3:]
    theta = torch.norm(omega, dim=-1, keepdim=True)
    R = so3_exp(omega, eps=eps)

    # V matrix uses _skew(omega) (unnormalized), NOT _skew(omega/theta).
    # Standard formula: V = I + B*[ω]× + C*[ω]×²
    # where B = (1-cosθ)/θ², C = (1-sinθ/θ)/θ².
    K = _skew(omega)
    eye = torch.eye(3, device=xi.device, dtype=xi.dtype).expand_as(K)
    _, B, C = _se3_v_coefficients(theta, eps=eps)

    V = eye + B[..., None] * K + C[..., None] * (K @ K)
    t = torch.matmul(V, v.unsqueeze(-1)).squeeze(-1)
    return R, t


def se3_log(R: torch.Tensor, t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Log map from SE(3) to se(3) (right-trivialized).
    Returns xi: [..., 6] (omega, v).
    """
    omega = so3_log(R, eps=eps)
    theta = torch.norm(omega, dim=-1, keepdim=True)  # [..., 1]
    # V_inv uses _skew(omega) (unnormalized), matching the V matrix convention.
    K = _skew(omega)
    eye = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(K)
    A, B, _ = _se3_v_coefficients(theta, eps=eps)
    factor = _se3_v_inv_factor(theta, A, B, eps=eps)
    V_inv = eye - 0.5 * K + factor[..., None] * (K @ K)

    v = torch.matmul(V_inv, t.unsqueeze(-1)).squeeze(-1)
    return torch.cat([omega, v], dim=-1)


def rigid_compose(R1: torch.Tensor, t1: torch.Tensor,
                  R2: torch.Tensor, t2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose rigid transforms: (R1,t1) ∘ (R2,t2)."""
    R = R1 @ R2
    t = torch.matmul(R1, t2.unsqueeze(-1)).squeeze(-1) + t1
    return R, t


def rigid_inverse(R: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of rigid transform."""
    R_inv = R.transpose(-2, -1)
    t_inv = -torch.matmul(R_inv, t.unsqueeze(-1)).squeeze(-1)
    return R_inv, t_inv
