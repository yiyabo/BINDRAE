"""
SE(3) utilities for Stage-2 (torch).
"""

from typing import Tuple

import torch


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
    cos_theta = ((trace - 1.0) / 2.0).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)
    theta = torch.acos(cos_theta)

    omega_hat = 0.5 * (R - R.transpose(-2, -1))
    omega = torch.stack(
        [omega_hat[..., 2, 1], omega_hat[..., 0, 2], omega_hat[..., 1, 0]],
        dim=-1,
    )

    sin_theta = torch.sin(theta).clamp(min=eps)
    scale = theta / sin_theta
    omega = omega * scale.unsqueeze(-1)

    small = theta < 1e-4
    if small.any():
        omega = torch.where(small.unsqueeze(-1), omega, omega)
        omega = torch.where(small.unsqueeze(-1), torch.stack(
            [omega_hat[..., 2, 1], omega_hat[..., 0, 2], omega_hat[..., 1, 0]],
            dim=-1,
        ), omega)
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

    K = _skew(omega / theta.clamp(min=eps))
    eye = torch.eye(3, device=xi.device, dtype=xi.dtype).expand_as(K)
    theta2 = theta * theta

    A = torch.where(theta < 1e-4, 1.0 - theta2 / 6.0, torch.sin(theta) / theta)
    B = torch.where(theta < 1e-4, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / theta2.clamp(min=eps))
    C = torch.where(theta < 1e-4, 1.0 / 6.0 - theta2 / 120.0, (1.0 - A) / theta2.clamp(min=eps))

    V = eye + B[..., None] * K + C[..., None] * (K @ K)
    t = torch.matmul(V, v.unsqueeze(-1)).squeeze(-1)
    return R, t


def se3_log(R: torch.Tensor, t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Log map from SE(3) to se(3) (right-trivialized).
    Returns xi: [..., 6] (omega, v).
    """
    omega = so3_log(R, eps=eps)
    theta = torch.norm(omega, dim=-1, keepdim=True)
    K = _skew(omega / theta.clamp(min=eps))
    eye = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(K)
    theta2 = theta * theta

    A = torch.where(theta < 1e-4, 1.0 - theta2 / 6.0, torch.sin(theta) / theta)
    B = torch.where(theta < 1e-4, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / theta2.clamp(min=eps))
    C = torch.where(theta < 1e-4, 1.0 / 6.0 - theta2 / 120.0, (1.0 - A) / theta2.clamp(min=eps))

    V = eye + B[..., None] * K + C[..., None] * (K @ K)

    # V_inv ≈ I - 0.5 K + (1/theta^2)*(1 - A/(2B)) K^2
    B_safe = B.clamp(min=eps)
    factor = (1.0 - A / (2.0 * B_safe)) / theta2.clamp(min=eps)
    V_inv = eye - 0.5 * K + factor[..., None] * (K @ K)

    small = theta < 1e-4
    if small.any():
        V_inv_small = eye - 0.5 * K + (1.0 / 12.0) * (K @ K)
        V_inv = torch.where(small[..., None, None], V_inv_small, V_inv)

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

