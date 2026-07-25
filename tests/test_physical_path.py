import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from flash_ipa.rigid import Rigid, Rotation

from src.stage2.modules.physical_path import (
    PhysicalPathOptimizationConfig,
    _route_initializations,
    apply_projected_normal_residual,
    deterministic_nonbonded_clash_loss,
    optimize_projected_normal_path,
)


def _rigid(translation: torch.Tensor) -> Rigid:
    batch_size, n_res, _ = translation.shape
    rotation = torch.eye(3).view(1, 1, 3, 3).expand(
        batch_size, n_res, 3, 3
    ).clone()
    return Rigid(rots=Rotation(rot_mats=rotation), trans=translation)


def _rigid_with_rotation(translation: torch.Tensor, rotation: torch.Tensor) -> Rigid:
    batch_size, n_res, _ = translation.shape
    rotations = rotation.view(1, 1, 3, 3).expand(
        batch_size, n_res, 3, 3
    ).clone()
    return Rigid(rots=Rotation(rot_mats=rotations), trans=translation)


class _PointFK(nn.Module):
    def forward(self, torsions_sincos, rigids, aatype):
        del torsions_sincos, aatype
        translation = rigids.get_trans()
        positions = translation.new_zeros((*translation.shape[:-1], 14, 3))
        positions[..., 0, :] = translation
        mask = torch.zeros(positions.shape[:-1], dtype=torch.bool)
        mask[..., 0] = True
        return {"atom14_pos": positions, "atom14_mask": mask}


class PhysicalPathTest(unittest.TestCase):
    def test_route_seeds_are_paired_and_global_frame_consistent(self):
        times = [0.0, 0.5, 1.0]
        identity = torch.eye(3)
        quarter_turn = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        translations = [
            torch.tensor([[[0.0, 0.0, 0.0]]]),
            torch.tensor([[[1.0, 0.0, 0.0]]]),
            torch.tensor([[[2.0, 0.0, 0.0]]]),
        ]
        identity_path = [
            _rigid_with_rotation(value, identity) for value in translations
        ]
        rotated_path = [
            _rigid_with_rotation(value, quarter_turn) for value in translations
        ]
        tangent_rigid = torch.zeros(1, 1, 1, 6)
        tangent_chi = torch.zeros(1, 1, 1, 4)
        node_mask = torch.ones(1, 1, dtype=torch.bool)
        chi_mask = torch.ones(1, 1, 4, dtype=torch.bool)
        config = PhysicalPathOptimizationConfig(
            iterations=0,
            components="translation",
            num_starts=2,
            route_seed_scale=0.3,
            route_seed_rank=1,
            route_seed_smoothing_steps=0,
            route_seed=11,
        )

        identity_seeds = _route_initializations(
            times,
            identity_path,
            tangent_rigid,
            tangent_chi,
            node_mask,
            chi_mask,
            config,
        )
        rotated_seeds = _route_initializations(
            times,
            rotated_path,
            tangent_rigid,
            tangent_chi,
            node_mask,
            chi_mask,
            config,
        )

        self.assertTrue(torch.equal(identity_seeds[1][0], -identity_seeds[0][0]))
        identity_global = identity_seeds[0][0][..., 3:]
        rotated_global = torch.matmul(
            quarter_turn.view(1, 1, 1, 3, 3),
            rotated_seeds[0][0][..., 3:].unsqueeze(-1),
        ).squeeze(-1)
        self.assertTrue(torch.allclose(identity_global, rotated_global, atol=1e-6))

    def test_projected_residual_is_normal_and_endpoint_exact(self):
        times = [0.0, 0.5, 1.0]
        base_rigids = [
            _rigid(torch.tensor([[[0.0, 0.0, 0.0]]])),
            _rigid(torch.tensor([[[0.5, 0.0, 0.0]]])),
            _rigid(torch.tensor([[[1.0, 0.0, 0.0]]])),
        ]
        base_chi = [torch.zeros(1, 1, 4) for _ in times]
        raw_rigid = torch.zeros(1, 1, 1, 6)
        raw_rigid[..., 3] = 1.0
        raw_rigid[..., 4] = 1.0
        raw_chi = torch.zeros(1, 1, 1, 4)
        tangent_rigid = torch.zeros_like(raw_rigid)
        tangent_rigid[..., 3] = 1.0
        tangent_chi = torch.zeros_like(raw_chi)
        config = PhysicalPathOptimizationConfig(
            iterations=0,
            max_metric_norm=0.0,
            projection_mode="block",
            components="translation",
        )

        path = apply_projected_normal_residual(
            base_rigids,
            base_chi,
            times,
            raw_rigid,
            raw_chi,
            tangent_rigid,
            tangent_chi,
            torch.ones(1, 1, dtype=torch.bool),
            torch.ones(1, 1, 4, dtype=torch.bool),
            config,
        )

        self.assertTrue(
            torch.equal(path[0][0].get_trans(), base_rigids[0].get_trans())
        )
        self.assertTrue(
            torch.equal(path[0][-1].get_trans(), base_rigids[-1].get_trans())
        )
        self.assertAlmostEqual(path[0][1].get_trans()[0, 0, 0].item(), 0.5, places=5)
        self.assertGreater(path[0][1].get_trans()[0, 0, 1].item(), 0.9)
        self.assertLess(path[4]["normal_parallel_cos_abs"].item(), 1e-6)

    def test_deterministic_clash_loss_detects_nonlocal_overlap(self):
        positions = torch.zeros(1, 3, 14, 3)
        mask = torch.zeros(1, 3, 14, dtype=torch.bool)
        mask[:, :, 0] = True
        positions[0, 0, 0] = torch.tensor([0.0, 0.0, 0.0])
        positions[0, 1, 0] = torch.tensor([4.0, 0.0, 0.0])
        positions[0, 2, 0] = torch.tensor([0.5, 0.0, 0.0])
        node_mask = torch.ones(1, 3, dtype=torch.bool)

        collided = deterministic_nonbonded_clash_loss(
            positions, mask, node_mask, threshold=2.0, max_atoms=32
        )
        positions[0, 2, 0] = torch.tensor([8.0, 0.0, 0.0])
        separated = deterministic_nonbonded_clash_loss(
            positions, mask, node_mask, threshold=2.0, max_atoms=32
        )
        self.assertGreater(collided.item(), 1.0)
        self.assertEqual(separated.item(), 0.0)

    def test_optimizer_moves_normal_to_reduce_ligand_collision(self):
        times = [0.0, 0.5, 1.0]
        base_rigids = [
            _rigid(torch.tensor([[[0.0, 0.0, 0.0]]])),
            _rigid(torch.tensor([[[1.0, 0.0, 0.0]]])),
            _rigid(torch.tensor([[[2.0, 0.0, 0.0]]])),
        ]
        base_chi = [torch.zeros(1, 1, 4) for _ in times]
        tangent_rigid = torch.zeros(1, 1, 1, 6)
        tangent_rigid[..., 3] = 2.0
        tangent_chi = torch.zeros(1, 1, 1, 4)
        batch = SimpleNamespace(
            torsion_apo=torch.zeros(1, 1, 7),
            torsion_holo=torch.zeros(1, 1, 7),
            aatype=torch.zeros(1, 1, dtype=torch.long),
            node_mask=torch.ones(1, 1, dtype=torch.bool),
            chi_mask=torch.ones(1, 1, 4, dtype=torch.bool),
            peptide_bond_mask=torch.zeros(1, 0, dtype=torch.bool),
            lig_points=torch.tensor([[[1.0, 0.15, 0.0]]]),
            lig_mask=torch.ones(1, 1, dtype=torch.bool),
            w_res=torch.ones(1, 1),
        )
        config = PhysicalPathOptimizationConfig(
            iterations=40,
            learning_rate=0.08,
            max_metric_norm=2.0,
            ligand_clash_distance=1.5,
            weight_peptide=0.0,
            weight_protein_clash=0.0,
            weight_ligand_clash=1.0,
            weight_contact_anchor=0.0,
            weight_distance_anchor=0.0,
            weight_residual_magnitude=0.001,
            weight_temporal_smoothness=0.001,
        )

        result = optimize_projected_normal_path(
            _PointFK(),
            batch,
            base_rigids,
            base_chi,
            times,
            tangent_rigid,
            tangent_chi,
            config,
        )

        self.assertTrue(
            torch.equal(result.rigids[0].get_trans(), base_rigids[0].get_trans())
        )
        self.assertTrue(
            torch.equal(result.rigids[-1].get_trans(), base_rigids[-1].get_trans())
        )
        self.assertLess(
            result.diagnostics["final_ligand_clash"],
            0.2 * result.diagnostics["initial_ligand_clash"],
        )
        self.assertGreater(
            abs(result.rigids[1].get_trans()[0, 0, 1].item()), 0.5
        )
        self.assertEqual(result.diagnostics["applied_rotation_rms"], 0.0)
        self.assertEqual(result.diagnostics["applied_chi_rms"], 0.0)
        self.assertEqual(tuple(result.projected_rigid.shape), (1, 1, 1, 6))
        self.assertEqual(tuple(result.projected_chi.shape), (1, 1, 1, 4))
        self.assertTrue(torch.equal(result.projected_rigid[..., :3], torch.zeros_like(result.projected_rigid[..., :3])))
        self.assertTrue(torch.equal(result.projected_chi, torch.zeros_like(result.projected_chi)))
        self.assertTrue(torch.equal(result.applied_rigid, result.projected_rigid))
        self.assertEqual(result.diagnostics["num_starts"], 1)
        self.assertLessEqual(
            result.diagnostics["final_objective"],
            result.diagnostics["guide_objective"],
        )

    def test_multistart_backtracking_escapes_symmetric_collision(self):
        times = [0.0, 0.5, 1.0]
        base_rigids = [
            _rigid(torch.tensor([[[0.0, 0.0, 0.0]]])),
            _rigid(torch.tensor([[[1.0, 0.0, 0.0]]])),
            _rigid(torch.tensor([[[2.0, 0.0, 0.0]]])),
        ]
        base_chi = [torch.zeros(1, 1, 4) for _ in times]
        tangent_rigid = torch.zeros(1, 1, 1, 6)
        tangent_rigid[..., 3] = 2.0
        tangent_chi = torch.zeros(1, 1, 1, 4)
        batch = SimpleNamespace(
            torsion_apo=torch.zeros(1, 1, 7),
            torsion_holo=torch.zeros(1, 1, 7),
            aatype=torch.zeros(1, 1, dtype=torch.long),
            node_mask=torch.ones(1, 1, dtype=torch.bool),
            chi_mask=torch.ones(1, 1, 4, dtype=torch.bool),
            peptide_bond_mask=torch.zeros(1, 0, dtype=torch.bool),
            lig_points=torch.tensor([[[1.0, 0.0, 0.0]]]),
            lig_mask=torch.ones(1, 1, dtype=torch.bool),
            w_res=torch.ones(1, 1),
        )
        config = PhysicalPathOptimizationConfig(
            iterations=20,
            learning_rate=0.2,
            max_metric_norm=2.0,
            ligand_clash_distance=1.5,
            weight_peptide=0.0,
            weight_protein_clash=0.0,
            weight_ligand_clash=1.0,
            weight_contact_anchor=0.0,
            weight_distance_anchor=0.0,
            weight_residual_magnitude=0.001,
            weight_temporal_smoothness=0.001,
            optimizer="backtracking",
            num_starts=2,
            route_seed_scale=0.35,
            route_seed_rank=1,
            route_seed_smoothing_steps=0,
            route_seed=7,
            frame_aggregation="max",
            line_search_steps=8,
        )

        result = optimize_projected_normal_path(
            _PointFK(),
            batch,
            base_rigids,
            base_chi,
            times,
            tangent_rigid,
            tangent_chi,
            config,
        )

        self.assertEqual(result.diagnostics["num_starts"], 2)
        self.assertEqual(len(result.diagnostics["start_objectives"]), 2)
        self.assertAlmostEqual(
            result.diagnostics["start_objectives"][0],
            result.diagnostics["start_objectives"][1],
            places=6,
        )
        self.assertLessEqual(
            result.diagnostics["final_objective"],
            result.diagnostics["guide_objective"],
        )
        self.assertLess(
            result.diagnostics["final_ligand_clash"],
            0.5 * result.diagnostics["initial_ligand_clash"],
        )
        self.assertGreater(result.diagnostics["accepted_steps"], 0)
        self.assertLess(result.diagnostics["normal_parallel_cos_abs"], 1e-6)
        self.assertTrue(
            torch.equal(result.rigids[0].get_trans(), base_rigids[0].get_trans())
        )
        self.assertTrue(
            torch.equal(result.rigids[-1].get_trans(), base_rigids[-1].get_trans())
        )
        off_axis = result.rigids[1].get_trans()[0, 0, 1:]
        self.assertGreater(torch.linalg.norm(off_axis).item(), 0.2)


if __name__ == "__main__":
    unittest.main()
