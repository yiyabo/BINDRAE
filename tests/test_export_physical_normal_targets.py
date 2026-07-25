import argparse
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from flash_ipa.rigid import Rigid, Rotation

from scripts.export_physical_normal_targets import build_cache_payload
from src.stage2.modules import PhysicalPathOptimizationResult


def _rigid(translation: torch.Tensor) -> Rigid:
    rotation = torch.eye(3).view(1, 1, 3, 3).expand(
        translation.shape[0], translation.shape[1], 3, 3
    )
    return Rigid(rots=Rotation(rot_mats=rotation), trans=translation)


class PhysicalTargetExportTest(unittest.TestCase):
    def test_payload_matches_phase_normal_cache_contract(self):
        args = argparse.Namespace(
            checkpoint="teacher.pt",
            physical_normal_envelope="poly",
            physical_normal_projection_mode="block",
            physical_normal_iterations=8,
            physical_normal_learning_rate=0.05,
            physical_normal_components="translation",
            physical_normal_max_metric_norm=1.0,
            physical_normal_gradient_clip=10.0,
            physical_normal_protein_clash_dist=2.0,
            physical_normal_ligand_clash_dist=2.2,
            physical_normal_max_clash_atoms=256,
            physical_normal_weight_peptide=1.0,
            physical_normal_weight_protein_clash=1.0,
            physical_normal_weight_ligand_clash=1.0,
            physical_normal_weight_contact_anchor=0.25,
            physical_normal_weight_distance_anchor=1.0,
            physical_normal_weight_residual=20.0,
            physical_normal_weight_temporal=0.2,
            phase_residual_bridge_mode=None,
            phase_residual_rotation_metric_scale=None,
            phase_residual_translation_metric_scale=None,
            phase_residual_chi_metric_scale=None,
        )
        config = SimpleNamespace(
            phase_residual_bridge_mode="cartesian_backbone",
            phase_residual_rotation_metric_scale=1.0,
            phase_residual_translation_metric_scale=1.0,
            phase_residual_chi_metric_scale=1.0,
        )
        batch = SimpleNamespace(
            node_mask=torch.tensor([[True, True]]),
            chi_mask=torch.ones(1, 2, 4, dtype=torch.bool),
        )
        zero = torch.zeros(2, 1, 2, 6)
        projected = zero.clone()
        projected[..., 4] = 0.2
        applied = projected * 0.75
        result = PhysicalPathOptimizationResult(
            rigids=[_rigid(torch.zeros(1, 2, 3)) for _ in range(4)],
            chi=[torch.zeros(1, 2, 4) for _ in range(4)],
            times=[0.0, 0.25, 0.75, 1.0],
            projected_rigid=projected,
            projected_chi=torch.zeros(2, 1, 2, 4),
            applied_rigid=applied,
            applied_chi=torch.zeros(2, 1, 2, 4),
            diagnostics={"objective_improvement": 1.0},
        )
        tau = torch.tensor(
            [
                [[0.0, 0.0]],
                [[0.2, 0.3]],
                [[0.7, 0.8]],
                [[1.0, 1.0]],
            ]
        )

        payload = build_cache_payload(
            args, config, "sample", 2, batch, result, tau
        )

        self.assertEqual(payload["schema_version"].item(), "md_phase_normal_v1")
        self.assertEqual(payload["source"].item(), "physical_normal_teacher_v1")
        self.assertEqual(payload["phase_target_mode"].item(), "learned_teacher")
        self.assertEqual(payload["normal_projection_mode"].item(), "block")
        self.assertEqual(payload["residual_trans"].shape, (2, 2, 3))
        self.assertTrue(np.allclose(payload["residual_trans"][..., 1], 0.2))
        self.assertTrue(np.all(payload["residual_valid_mask"]))


if __name__ == "__main__":
    unittest.main()
