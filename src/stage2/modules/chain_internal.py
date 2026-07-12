"""Chain-preserving interpolation of protein backbone internal coordinates."""

from typing import List, Tuple, Union

import torch

from .se3 import (
    rigid_compose,
    rigid_inverse,
    se3_exp,
    se3_log,
    so3_exp,
    so3_log,
)


BondTarget = Union[float, torch.Tensor]


def _normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / torch.linalg.norm(vector, dim=-1, keepdim=True).clamp_min(eps)


def _closest_peptide_direction(
    ca_to_c: torch.Tensor,
    n_to_ca: torch.Tensor,
    current_direction: torch.Tensor,
    angle_cacn: BondTarget,
    angle_cnca: BondTarget,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Find the closest C->N direction compatible with fixed residue frames."""
    u_left = _normalize(-ca_to_c, eps=eps)
    u_right = _normalize(n_to_ca, eps=eps)
    current_direction = _normalize(current_direction, eps=eps)

    cross = torch.cross(u_left, u_right, dim=-1)
    cross_norm = torch.linalg.norm(cross, dim=-1, keepdim=True)
    normal = cross / cross_norm.clamp_min(eps)
    cosine_between = (u_left * u_right).sum(dim=-1, keepdim=True)
    determinant = (1.0 - cosine_between.square()).clamp_min(1e-4)
    left_angle = torch.as_tensor(
        angle_cacn, dtype=ca_to_c.dtype, device=ca_to_c.device
    )
    right_angle = torch.as_tensor(
        angle_cnca, dtype=ca_to_c.dtype, device=ca_to_c.device
    )
    if left_angle.ndim == ca_to_c.ndim - 1:
        left_angle = left_angle.unsqueeze(-1)
    if right_angle.ndim == ca_to_c.ndim - 1:
        right_angle = right_angle.unsqueeze(-1)
    left_target = torch.cos(left_angle)
    right_target = -torch.cos(right_angle)
    left_coefficient = (
        left_target - cosine_between * right_target
    ) / determinant
    right_coefficient = (
        right_target - cosine_between * left_target
    ) / determinant
    in_plane = left_coefficient * u_left + right_coefficient * u_right
    in_plane_norm_sq = in_plane.square().sum(dim=-1, keepdim=True)
    normal_scale = torch.sqrt((1.0 - in_plane_norm_sq).clamp_min(0.0))
    candidate_positive = in_plane + normal_scale * normal
    candidate_negative = in_plane - normal_scale * normal
    choose_positive = (
        candidate_positive * current_direction
    ).sum(dim=-1, keepdim=True) >= (
        candidate_negative * current_direction
    ).sum(dim=-1, keepdim=True)
    feasible = (in_plane_norm_sq <= 1.0) & (cross_norm > 1e-4)
    feasible_direction = torch.where(
        choose_positive, candidate_positive, candidate_negative
    )
    fallback_direction = _normalize(in_plane, eps=eps)
    fallback_direction = torch.where(
        torch.linalg.norm(in_plane, dim=-1, keepdim=True) > 1e-4,
        fallback_direction,
        current_direction,
    )
    return _normalize(
        torch.where(feasible, feasible_direction, fallback_direction), eps=eps
    )


def project_peptide_frame_translations(
    atom14_pos: torch.Tensor,
    atom14_mask: torch.Tensor,
    node_mask: torch.Tensor,
    peptide_bond_mask: torch.Tensor,
    n_iterations: int = 8,
    relaxation: float = 0.75,
    anchor_strength: float = 0.02,
    bond_length: BondTarget = 1.33,
    angle_cacn: BondTarget = 2.035,
    angle_cnca: BondTarget = 2.124,
    max_translation: float = 2.0,
    activation_loss_threshold: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Project peptide geometry using bounded per-residue frame translations.

    The input path remains the global guide. Each Jacobi iteration only translates
    complete residue frames, so intra-residue geometry is unchanged and no
    N-terminal reconstruction error can accumulate along the full chain.
    """
    if atom14_pos.ndim != 4 or atom14_pos.shape[-2:] != (14, 3):
        raise ValueError("atom14_pos must have shape [B, N, 14, 3]")
    batch_size, n_residues = atom14_pos.shape[:2]
    if atom14_mask.shape != (batch_size, n_residues, 14):
        raise ValueError("atom14_mask must have shape [B, N, 14]")
    if node_mask.shape != (batch_size, n_residues):
        raise ValueError("node_mask must have shape [B, N]")
    if peptide_bond_mask.shape != (batch_size, max(n_residues - 1, 0)):
        raise ValueError("peptide_bond_mask must have shape [B, N-1]")
    if n_iterations < 0:
        raise ValueError("n_iterations must be non-negative")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must be in (0, 1]")
    if not 0.0 <= anchor_strength < 1.0:
        raise ValueError("anchor_strength must be in [0, 1)")
    if max_translation <= 0.0:
        raise ValueError("max_translation must be positive")
    if activation_loss_threshold < 0.0:
        raise ValueError("activation_loss_threshold must be non-negative")

    valid_bond = (
        peptide_bond_mask.bool()
        & node_mask[:, :-1].bool()
        & node_mask[:, 1:].bool()
        & atom14_mask[:, :-1, 1].bool()
        & atom14_mask[:, :-1, 2].bool()
        & atom14_mask[:, 1:, 0].bool()
        & atom14_mask[:, 1:, 1].bool()
    )
    translation = atom14_pos.new_zeros((batch_size, n_residues, 3))
    guide_n = atom14_pos[:, :, 0]
    guide_ca = atom14_pos[:, :, 1]
    guide_c = atom14_pos[:, :, 2]
    target_length = torch.as_tensor(
        bond_length, dtype=atom14_pos.dtype, device=atom14_pos.device
    )
    if target_length.shape == valid_bond.shape:
        target_length = target_length.unsqueeze(-1)
    target_cacn = torch.as_tensor(
        angle_cacn, dtype=atom14_pos.dtype, device=atom14_pos.device
    )
    target_cnca = torch.as_tensor(
        angle_cnca, dtype=atom14_pos.dtype, device=atom14_pos.device
    )
    if target_cacn.shape == valid_bond.shape:
        target_cacn = target_cacn.unsqueeze(-1)
    if target_cnca.shape == valid_bond.shape:
        target_cnca = target_cnca.unsqueeze(-1)

    for _ in range(n_iterations):
        n_coord = guide_n + translation
        ca_coord = guide_ca + translation
        c_coord = guide_c + translation
        current_vector = n_coord[:, 1:] - c_coord[:, :-1]
        target_direction = _closest_peptide_direction(
            c_coord[:, :-1] - ca_coord[:, :-1],
            ca_coord[:, 1:] - n_coord[:, 1:],
            current_vector,
            angle_cacn=target_cacn,
            angle_cnca=target_cnca,
            eps=eps,
        )
        error = target_length * target_direction - current_vector
        correction = 0.5 * float(relaxation) * error
        active_bond = valid_bond
        if activation_loss_threshold > 0.0:
            current_direction = _normalize(current_vector, eps=eps)
            left_direction = _normalize(
                ca_coord[:, :-1] - c_coord[:, :-1], eps=eps
            )
            right_direction = _normalize(
                ca_coord[:, 1:] - n_coord[:, 1:], eps=eps
            )
            current_cacn = torch.acos(
                (left_direction * current_direction)
                .sum(dim=-1, keepdim=True)
                .clamp(-1.0, 1.0)
            )
            current_cnca = torch.acos(
                (right_direction * -current_direction)
                .sum(dim=-1, keepdim=True)
                .clamp(-1.0, 1.0)
            )
            local_loss = (
                (torch.linalg.norm(current_vector, dim=-1, keepdim=True) - target_length).square()
                + 0.1 * (current_cacn - target_cacn).square()
                + 0.1 * (current_cnca - target_cnca).square()
            )
            active_bond = active_bond & (
                local_loss.squeeze(-1) > float(activation_loss_threshold)
            )
        correction = correction * active_bond.unsqueeze(-1).to(correction.dtype)

        update = torch.zeros_like(translation)
        update[:, :-1] = update[:, :-1] - correction
        update[:, 1:] = update[:, 1:] + correction
        degree = atom14_pos.new_zeros((batch_size, n_residues, 1))
        degree[:, :-1] = degree[:, :-1] + active_bond.unsqueeze(-1)
        degree[:, 1:] = degree[:, 1:] + active_bond.unsqueeze(-1)
        translation = translation + update / degree.clamp_min(1.0)
        if anchor_strength:
            translation = translation * (1.0 - float(anchor_strength))
        translation_norm = torch.linalg.norm(translation, dim=-1, keepdim=True)
        translation = translation * (
            float(max_translation) / translation_norm.clamp_min(eps)
        ).clamp(max=1.0)
        translation = translation * node_mask.unsqueeze(-1).to(translation.dtype)
    return translation


def _relative_rigids(
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    left_rotation, left_translation = rigid_inverse(
        rotation[:, :-1], translation[:, :-1]
    )
    return rigid_compose(
        left_rotation,
        left_translation,
        rotation[:, 1:],
        translation[:, 1:],
    )


def interpolate_relative_rigids(
    apo_rotation: torch.Tensor,
    apo_translation: torch.Tensor,
    holo_rotation: torch.Tensor,
    holo_translation: torch.Tensor,
    progress: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interpolate adjacent-residue transforms instead of absolute frames."""
    apo_rel_rotation, apo_rel_translation = _relative_rigids(
        apo_rotation, apo_translation
    )
    holo_rel_rotation, holo_rel_translation = _relative_rigids(
        holo_rotation, holo_translation
    )
    apo_rel_inverse_rotation, apo_rel_inverse_translation = rigid_inverse(
        apo_rel_rotation, apo_rel_translation
    )
    delta_rotation, delta_translation = rigid_compose(
        apo_rel_inverse_rotation,
        apo_rel_inverse_translation,
        holo_rel_rotation,
        holo_rel_translation,
    )
    delta_twist = se3_log(delta_rotation, delta_translation)
    increment_rotation, increment_translation = se3_exp(
        delta_twist * float(progress)
    )
    return rigid_compose(
        apo_rel_rotation,
        apo_rel_translation,
        increment_rotation,
        increment_translation,
    )


def project_anchored_pose_graph(
    guide_rotation: torch.Tensor,
    guide_translation: torch.Tensor,
    apo_rotation: torch.Tensor,
    apo_translation: torch.Tensor,
    holo_rotation: torch.Tensor,
    holo_translation: torch.Tensor,
    node_mask: torch.Tensor,
    peptide_bond_mask: torch.Tensor,
    progress: float,
    n_iterations: int = 20,
    learning_rate: float = 0.05,
    edge_weight: float = 1.0,
    anchor_weight: float = 0.1,
    rotation_metric_scale: float = 1.5,
    max_rotation: float = 0.5,
    max_translation: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Balance adjacent-frame consistency against an absolute SE(3) guide.

    This reference optimizer is intended for diagnostics. It uses an anchored
    pose-graph objective, so local corrections cannot accumulate freely along
    the entire chain as they do in a single-root NeRF reconstruction.
    """
    if guide_rotation.ndim != 4 or guide_rotation.shape[-2:] != (3, 3):
        raise ValueError("guide_rotation must have shape [B, N, 3, 3]")
    if guide_translation.shape != guide_rotation.shape[:-1]:
        raise ValueError("guide_translation must have shape [B, N, 3]")
    if node_mask.shape != guide_rotation.shape[:2]:
        raise ValueError("node_mask must have shape [B, N]")
    if peptide_bond_mask.shape != (
        guide_rotation.shape[0], max(guide_rotation.shape[1] - 1, 0)
    ):
        raise ValueError("peptide_bond_mask must have shape [B, N-1]")
    if n_iterations < 0:
        raise ValueError("n_iterations must be non-negative")
    if learning_rate <= 0.0 or edge_weight < 0.0 or anchor_weight < 0.0:
        raise ValueError("pose-graph optimization weights must be non-negative")
    if max_rotation <= 0.0 or max_translation <= 0.0:
        raise ValueError("pose-graph correction caps must be positive")

    target_rotation, target_translation = interpolate_relative_rigids(
        apo_rotation,
        apo_translation,
        holo_rotation,
        holo_translation,
        progress,
    )
    target_inverse_rotation, target_inverse_translation = rigid_inverse(
        target_rotation, target_translation
    )
    valid_node = node_mask.bool()
    valid_edge = (
        peptide_bond_mask.bool()
        & valid_node[:, :-1]
        & valid_node[:, 1:]
    )
    delta = guide_translation.new_zeros((*guide_translation.shape[:2], 6))
    first_moment = torch.zeros_like(delta)
    second_moment = torch.zeros_like(delta)

    with torch.enable_grad():
        for iteration in range(1, int(n_iterations) + 1):
            delta = delta.detach().requires_grad_(True)
            increment_rotation, increment_translation = se3_exp(delta)
            current_rotation, current_translation = rigid_compose(
                guide_rotation,
                guide_translation,
                increment_rotation,
                increment_translation,
            )
            relative_rotation, relative_translation = _relative_rigids(
                current_rotation, current_translation
            )
            error_rotation, error_translation = rigid_compose(
                target_inverse_rotation,
                target_inverse_translation,
                relative_rotation,
                relative_translation,
            )
            edge_twist = se3_log(error_rotation, error_translation)
            edge_metric = (
                rotation_metric_scale**2 * edge_twist[..., :3].square().sum(dim=-1)
                + edge_twist[..., 3:].square().sum(dim=-1)
            )
            anchor_metric = (
                rotation_metric_scale**2 * delta[..., :3].square().sum(dim=-1)
                + delta[..., 3:].square().sum(dim=-1)
            )
            loss = (
                float(edge_weight)
                * (edge_metric * valid_edge.to(edge_metric.dtype)).sum()
                + float(anchor_weight)
                * (anchor_metric * valid_node.to(anchor_metric.dtype)).sum()
            ) / max(guide_rotation.shape[0], 1)
            gradient = torch.autograd.grad(loss, delta)[0]
            gradient = gradient * valid_node.unsqueeze(-1).to(gradient.dtype)
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient.square()
            first_hat = first_moment / (1.0 - 0.9**iteration)
            second_hat = second_moment / (1.0 - 0.999**iteration)
            delta = delta - float(learning_rate) * first_hat / (
                torch.sqrt(second_hat) + 1e-8
            )
            rotation_norm = torch.linalg.norm(delta[..., :3], dim=-1, keepdim=True)
            translation_norm = torch.linalg.norm(
                delta[..., 3:], dim=-1, keepdim=True
            )
            delta = torch.cat(
                [
                    delta[..., :3]
                    * (float(max_rotation) / rotation_norm.clamp_min(1e-8)).clamp(max=1.0),
                    delta[..., 3:]
                    * (float(max_translation) / translation_norm.clamp_min(1e-8)).clamp(max=1.0),
                ],
                dim=-1,
            )
            delta = delta * valid_node.unsqueeze(-1).to(delta.dtype)

        increment_rotation, increment_translation = se3_exp(delta.detach())
        projected_rotation, projected_translation = rigid_compose(
            guide_rotation,
            guide_translation,
            increment_rotation,
            increment_translation,
        )
    return projected_rotation, projected_translation, delta.detach()


def backbone_frame(
    n_coord: torch.Tensor,
    ca_coord: torch.Tensor,
    c_coord: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a right-handed residue frame with CA as the origin."""
    e1 = _normalize(c_coord - ca_coord)
    n_direction = n_coord - ca_coord
    e2 = _normalize(n_direction - (n_direction * e1).sum(dim=-1, keepdim=True) * e1)
    e3 = _normalize(torch.cross(e1, e2, dim=-1))
    rotation = torch.stack([e1, e2, e3], dim=-1)
    return rotation, ca_coord


def place_atom_nerf(
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    bond_length: torch.Tensor,
    bond_angle: torch.Tensor,
    dihedral: torch.Tensor,
) -> torch.Tensor:
    """Place p4 from p1-p2-p3 and matching NeRF internal coordinates."""
    bc = _normalize(p3 - p2)
    normal = _normalize(torch.cross(p2 - p1, bc, dim=-1))
    tangent = torch.cross(normal, bc, dim=-1)
    direction = (
        torch.sin(bond_angle) * torch.cos(dihedral) * tangent
        + torch.sin(bond_angle) * torch.sin(dihedral) * normal
        - torch.cos(bond_angle) * bc
    )
    return p3 + bond_length.unsqueeze(-1) * direction


def extract_internal_coordinates(
    atom_chain: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract NeRF length, angle, and dihedral for atoms after the root triplet."""
    if atom_chain.ndim != 2 or atom_chain.shape[-1] != 3:
        raise ValueError("atom_chain must have shape [M, 3]")
    if atom_chain.shape[0] < 3:
        raise ValueError("atom_chain must contain at least three atoms")
    if atom_chain.shape[0] == 3:
        empty = atom_chain.new_zeros((0,))
        return empty, empty, empty

    p1 = atom_chain[:-3]
    p2 = atom_chain[1:-2]
    p3 = atom_chain[2:-1]
    p4 = atom_chain[3:]
    previous = _normalize(p2 - p3)
    current = _normalize(p4 - p3)
    cosine = (previous * current).sum(dim=-1).clamp(-1.0, 1.0)
    angles = torch.acos(cosine)
    lengths = torch.linalg.norm(p4 - p3, dim=-1)

    bc = _normalize(p3 - p2)
    normal = _normalize(torch.cross(p2 - p1, bc, dim=-1))
    tangent = torch.cross(normal, bc, dim=-1)
    direction = _normalize(p4 - p3)
    dihedrals = torch.atan2(
        (direction * normal).sum(dim=-1),
        (direction * tangent).sum(dim=-1),
    )
    return lengths, angles, dihedrals


def reconstruct_from_internal_coordinates(
    root_triplet: torch.Tensor,
    lengths: torch.Tensor,
    angles: torch.Tensor,
    dihedrals: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct an atom chain from a root triplet and NeRF coordinates."""
    if root_triplet.shape != (3, 3):
        raise ValueError("root_triplet must have shape [3, 3]")
    if lengths.shape != angles.shape or lengths.shape != dihedrals.shape:
        raise ValueError("lengths, angles, and dihedrals must have matching shapes")
    atoms: List[torch.Tensor] = [root_triplet[i] for i in range(3)]
    for idx in range(lengths.shape[0]):
        atoms.append(
            place_atom_nerf(
                atoms[-3],
                atoms[-2],
                atoms[-1],
                lengths[idx],
                angles[idx],
                dihedrals[idx],
            )
        )
    return torch.stack(atoms, dim=0)


def _interpolate_angle(start: torch.Tensor, end: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    delta = torch.atan2(torch.sin(end - start), torch.cos(end - start))
    return start + t * delta


def _interpolate_root_triplet(
    root_apo: torch.Tensor,
    root_holo: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    rotation_apo, translation_apo = backbone_frame(
        root_apo[0], root_apo[1], root_apo[2]
    )
    rotation_holo, translation_holo = backbone_frame(
        root_holo[0], root_holo[1], root_holo[2]
    )
    relative = rotation_apo.transpose(-1, -2) @ rotation_holo
    rotation_t = rotation_apo @ so3_exp(t * so3_log(relative))
    translation_t = (1.0 - t) * translation_apo + t * translation_holo
    local_apo = (root_apo - translation_apo) @ rotation_apo
    local_holo = (root_holo - translation_holo) @ rotation_holo
    local_t = (1.0 - t) * local_apo + t * local_holo
    return local_t @ rotation_t.transpose(-1, -2) + translation_t


def connected_segments(
    node_mask: torch.Tensor,
    peptide_bond_mask: torch.Tensor,
) -> List[Tuple[int, int]]:
    """Return inclusive-exclusive contiguous residue segments."""
    if node_mask.ndim != 1:
        raise ValueError("node_mask must have shape [N]")
    if peptide_bond_mask.shape != (max(node_mask.shape[0] - 1, 0),):
        raise ValueError("peptide_bond_mask must have shape [N-1]")
    segments: List[Tuple[int, int]] = []
    index = 0
    n_residues = node_mask.shape[0]
    while index < n_residues:
        if not bool(node_mask[index].item()):
            index += 1
            continue
        end = index + 1
        while (
            end < n_residues
            and bool(node_mask[end].item())
            and bool(peptide_bond_mask[end - 1].item())
        ):
            end += 1
        segments.append((index, end))
        index = end
    return segments


def interpolate_backbone_internal(
    n_apo: torch.Tensor,
    ca_apo: torch.Tensor,
    c_apo: torch.Tensor,
    n_holo: torch.Tensor,
    ca_holo: torch.Tensor,
    c_holo: torch.Tensor,
    node_mask: torch.Tensor,
    peptide_bond_mask: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interpolate N/CA/C through a chain-connected internal-coordinate bridge."""
    if n_apo.shape != ca_apo.shape or n_apo.shape != c_apo.shape:
        raise ValueError("apo backbone tensors must have matching shapes")
    if n_holo.shape != n_apo.shape or ca_holo.shape != n_apo.shape or c_holo.shape != n_apo.shape:
        raise ValueError("apo and holo backbone tensors must have matching shapes")
    if n_apo.ndim != 3 or n_apo.shape[-1] != 3:
        raise ValueError("backbone tensors must have shape [B, N, 3]")
    batch_size, n_residues, _ = n_apo.shape
    if node_mask.shape != (batch_size, n_residues):
        raise ValueError("node_mask must have shape [B, N]")
    if peptide_bond_mask.shape != (batch_size, max(n_residues - 1, 0)):
        raise ValueError("peptide_bond_mask must have shape [B, N-1]")
    if t.ndim == 0:
        progress = t.expand(batch_size, n_residues)
    elif t.shape == (batch_size,):
        progress = t[:, None].expand(batch_size, n_residues)
    elif t.shape == (batch_size, n_residues):
        progress = t
    else:
        raise ValueError("t must be scalar or have shape [B] or [B, N]")
    progress = progress.clamp(0.0, 1.0)

    n_out = (1.0 - progress[..., None]) * n_apo + progress[..., None] * n_holo
    ca_out = (1.0 - progress[..., None]) * ca_apo + progress[..., None] * ca_holo
    c_out = (1.0 - progress[..., None]) * c_apo + progress[..., None] * c_holo
    sample_outputs = []
    for batch_idx in range(batch_size):
        atoms_out = torch.stack(
            [n_out[batch_idx], ca_out[batch_idx], c_out[batch_idx]], dim=1
        )
        for start, end in connected_segments(
            node_mask[batch_idx], peptide_bond_mask[batch_idx]
        ):
            apo_chain = torch.stack(
                [
                    n_apo[batch_idx, start:end],
                    ca_apo[batch_idx, start:end],
                    c_apo[batch_idx, start:end],
                ],
                dim=1,
            ).reshape(-1, 3)
            holo_chain = torch.stack(
                [
                    n_holo[batch_idx, start:end],
                    ca_holo[batch_idx, start:end],
                    c_holo[batch_idx, start:end],
                ],
                dim=1,
            ).reshape(-1, 3)
            lengths_apo, angles_apo, dihedrals_apo = extract_internal_coordinates(apo_chain)
            lengths_holo, angles_holo, dihedrals_holo = extract_internal_coordinates(holo_chain)
            root_time = progress[batch_idx, start]
            parameter_time = progress[batch_idx, start + 1:end].repeat_interleave(3)
            root = _interpolate_root_triplet(
                apo_chain[:3], holo_chain[:3], root_time
            )
            lengths = (1.0 - parameter_time) * lengths_apo + parameter_time * lengths_holo
            angles = (1.0 - parameter_time) * angles_apo + parameter_time * angles_holo
            dihedrals = _interpolate_angle(
                dihedrals_apo, dihedrals_holo, parameter_time
            )
            rebuilt = reconstruct_from_internal_coordinates(
                root, lengths, angles, dihedrals
            ).reshape(end - start, 3, 3)
            atoms_out = torch.cat(
                [atoms_out[:start], rebuilt, atoms_out[end:]], dim=0
            )
        sample_outputs.append(atoms_out)
    backbone = torch.stack(sample_outputs, dim=0)
    n_result, ca_result, c_result = backbone[:, :, 0], backbone[:, :, 1], backbone[:, :, 2]
    start_mask = (progress <= 0.0).all(dim=1).view(batch_size, 1, 1)
    end_mask = (progress >= 1.0).all(dim=1).view(batch_size, 1, 1)
    n_result = torch.where(start_mask, n_apo, torch.where(end_mask, n_holo, n_result))
    ca_result = torch.where(start_mask, ca_apo, torch.where(end_mask, ca_holo, ca_result))
    c_result = torch.where(start_mask, c_apo, torch.where(end_mask, c_holo, c_result))
    return n_result, ca_result, c_result
