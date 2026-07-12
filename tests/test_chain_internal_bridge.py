import unittest

import torch

from src.stage2.modules.chain_internal import (
    extract_internal_coordinates,
    interpolate_backbone_internal,
    place_atom_nerf,
    project_anchored_pose_graph,
    project_peptide_frame_translations,
    reconstruct_from_internal_coordinates,
)


def _peptide_error(atom14):
    c_coord = atom14[:, :-1, 2]
    n_coord = atom14[:, 1:, 0]
    ca_left = atom14[:, :-1, 1]
    ca_right = atom14[:, 1:, 1]
    bond = (torch.linalg.norm(n_coord - c_coord, dim=-1) - 1.33).square()
    direction = torch.nn.functional.normalize(n_coord - c_coord, dim=-1)
    left = torch.nn.functional.normalize(ca_left - c_coord, dim=-1)
    right = torch.nn.functional.normalize(ca_right - n_coord, dim=-1)
    angle_left = torch.acos((left * direction).sum(dim=-1).clamp(-1.0, 1.0))
    angle_right = torch.acos((right * -direction).sum(dim=-1).clamp(-1.0, 1.0))
    return (
        bond
        + 0.1 * (angle_left - 2.035).square()
        + 0.1 * (angle_right - 2.124).square()
    ).mean()


def _chain(root, lengths, angles, dihedrals):
    atoms = [root[index] for index in range(3)]
    for length, angle, dihedral in zip(lengths, angles, dihedrals):
        atoms.append(place_atom_nerf(atoms[-3], atoms[-2], atoms[-1], length, angle, dihedral))
    return torch.stack(atoms).reshape(-1, 3, 3)


class ChainInternalBridgeTest(unittest.TestCase):
    def setUp(self):
        self.root = torch.tensor(
            [[0.0, 0.0, 0.0], [1.45, 0.0, 0.0], [2.0, 1.35, 0.0]],
            dtype=torch.float64,
        )
        self.lengths = torch.tensor([1.33, 1.46, 1.52] * 3, dtype=torch.float64)
        self.angles = torch.tensor([2.03, 2.12, 1.94] * 3, dtype=torch.float64)
        self.dihedrals = torch.tensor(
            [-0.8, 3.0, -1.1, 0.5, -3.0, 1.4, -1.5, 2.9, -0.4],
            dtype=torch.float64,
        )

    def test_internal_coordinates_round_trip(self):
        chain = _chain(self.root, self.lengths, self.angles, self.dihedrals)
        flat = chain.reshape(-1, 3)
        lengths, angles, dihedrals = extract_internal_coordinates(flat)
        rebuilt = reconstruct_from_internal_coordinates(flat[:3], lengths, angles, dihedrals)
        self.assertLess((rebuilt - flat).abs().max().item(), 1e-8)

    def test_bridge_has_exact_endpoints_and_connected_intermediate(self):
        apo = _chain(self.root, self.lengths, self.angles, self.dihedrals)
        holo_lengths = self.lengths * 1.01
        holo_angles = self.angles + 0.02
        holo_dihedrals = self.dihedrals + 0.3
        holo_root = self.root @ torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        ).T + torch.tensor([3.0, -2.0, 1.0], dtype=torch.float64)
        holo = _chain(holo_root, holo_lengths, holo_angles, holo_dihedrals)
        args = (
            apo[None, :, 0], apo[None, :, 1], apo[None, :, 2],
            holo[None, :, 0], holo[None, :, 1], holo[None, :, 2],
            torch.ones(1, apo.shape[0], dtype=torch.bool),
            torch.ones(1, apo.shape[0] - 1, dtype=torch.bool),
        )
        start = interpolate_backbone_internal(*args, torch.tensor(0.0, dtype=torch.float64))
        end = interpolate_backbone_internal(*args, torch.tensor(1.0, dtype=torch.float64))
        for actual, expected in zip(start, (apo[None, :, 0], apo[None, :, 1], apo[None, :, 2])):
            self.assertLess((actual - expected).abs().max().item(), 1e-7)
        for actual, expected in zip(end, (holo[None, :, 0], holo[None, :, 1], holo[None, :, 2])):
            self.assertLess((actual - expected).abs().max().item(), 1e-7)

        middle = interpolate_backbone_internal(*args, torch.tensor(0.5, dtype=torch.float64))
        c_to_n = torch.linalg.norm(middle[2][:, :-1] - middle[0][:, 1:], dim=-1)
        expected = 0.5 * (self.lengths[::3] + holo_lengths[::3])
        self.assertLess((c_to_n[0] - expected).abs().max().item(), 1e-7)

    def test_bridge_has_finite_time_gradient(self):
        apo = _chain(self.root, self.lengths, self.angles, self.dihedrals)
        holo = _chain(self.root + 0.2, self.lengths, self.angles, self.dihedrals + 0.2)
        time = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        n_coord, ca_coord, c_coord = interpolate_backbone_internal(
            apo[None, :, 0], apo[None, :, 1], apo[None, :, 2],
            holo[None, :, 0], holo[None, :, 1], holo[None, :, 2],
            torch.ones(1, apo.shape[0], dtype=torch.bool),
            torch.ones(1, apo.shape[0] - 1, dtype=torch.bool),
            time,
        )
        loss = n_coord.square().mean() + ca_coord.square().mean() + c_coord.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(time.grad).item())

    def test_asynchronous_phase_preserves_chain_connectivity(self):
        apo = _chain(self.root, self.lengths, self.angles, self.dihedrals)
        holo_lengths = self.lengths * 1.02
        holo = _chain(
            self.root + 0.5,
            holo_lengths,
            self.angles + 0.03,
            self.dihedrals + 0.4,
        )
        phase = torch.tensor([[0.0, 0.2, 0.8, 1.0]], dtype=torch.float64)
        n_coord, _, c_coord = interpolate_backbone_internal(
            apo[None, :, 0], apo[None, :, 1], apo[None, :, 2],
            holo[None, :, 0], holo[None, :, 1], holo[None, :, 2],
            torch.ones(1, apo.shape[0], dtype=torch.bool),
            torch.ones(1, apo.shape[0] - 1, dtype=torch.bool),
            phase,
        )
        c_to_n = torch.linalg.norm(c_coord[:, :-1] - n_coord[:, 1:], dim=-1)
        expected = (
            (1.0 - phase[:, 1:]) * self.lengths[::3]
            + phase[:, 1:] * holo_lengths[::3]
        )
        self.assertLess((c_to_n - expected).abs().max().item(), 1e-7)

    def test_local_projection_reduces_peptide_error_without_chain_rebuild(self):
        chain = _chain(self.root, self.lengths, self.angles, self.dihedrals)
        atom14 = torch.zeros(1, chain.shape[0], 14, 3, dtype=torch.float64)
        atom14[:, :, :3] = chain.unsqueeze(0)
        offsets = torch.tensor(
            [[0.0, 0.0, 0.0], [0.3, -0.2, 0.1], [-0.2, 0.25, -0.1], [0.2, 0.1, 0.2]],
            dtype=torch.float64,
        )
        atom14 = atom14 + offsets[None, :, None]
        atom14.requires_grad_(True)
        atom14_mask = torch.zeros(1, chain.shape[0], 14, dtype=torch.bool)
        atom14_mask[:, :, :3] = True
        node_mask = torch.ones(1, chain.shape[0], dtype=torch.bool)
        bond_mask = torch.ones(1, chain.shape[0] - 1, dtype=torch.bool)

        before = _peptide_error(atom14)
        translation = project_peptide_frame_translations(
            atom14,
            atom14_mask,
            node_mask,
            bond_mask,
            n_iterations=24,
            relaxation=0.75,
            anchor_strength=0.0,
        )
        projected = atom14 + translation[:, :, None]
        after = _peptide_error(projected)
        self.assertLess(after.item(), before.item() * 0.05)
        self.assertLess(translation.norm(dim=-1).max().item(), 1.0)
        after.backward()
        self.assertTrue(torch.isfinite(atom14.grad).all().item())

    def test_anchored_pose_graph_reduces_adjacent_frame_error(self):
        rotation = torch.eye(3, dtype=torch.float64).expand(1, 4, 3, 3).clone()
        endpoint_translation = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
              [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
        guide_translation = endpoint_translation + torch.tensor(
            [[[0.0, 0.0, 0.0], [0.2, 0.3, 0.0],
              [-0.2, -0.2, 0.1], [0.1, 0.25, -0.1]]],
            dtype=torch.float64,
        )
        target_relative = endpoint_translation[:, 1:] - endpoint_translation[:, :-1]
        before = (
            guide_translation[:, 1:]
            - guide_translation[:, :-1]
            - target_relative
        ).square().mean()
        _, projected_translation, delta = project_anchored_pose_graph(
            rotation,
            guide_translation,
            rotation,
            endpoint_translation,
            rotation,
            endpoint_translation,
            torch.ones(1, 4, dtype=torch.bool),
            torch.ones(1, 3, dtype=torch.bool),
            progress=0.5,
            n_iterations=40,
            learning_rate=0.05,
            anchor_weight=0.05,
            max_rotation=0.5,
            max_translation=1.0,
        )
        after = (
            projected_translation[:, 1:]
            - projected_translation[:, :-1]
            - target_relative
        ).square().mean()
        self.assertLess(after.item(), before.item() * 0.1)
        self.assertTrue(torch.isfinite(delta).all().item())
        self.assertLessEqual(delta[..., 3:].norm(dim=-1).max().item(), 1.0 + 1e-8)


if __name__ == "__main__":
    unittest.main()
