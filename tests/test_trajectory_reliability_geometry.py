import math
import unittest

import torch

from scripts.evaluate_stage2_trajectory_reliability import peptide_geometry_metrics


def planar_dipeptide():
    positions = torch.zeros(1, 2, 14, 3)
    mask = torch.zeros(1, 2, 14, dtype=torch.bool)
    mask[:, :, :3] = True
    c0 = torch.tensor([0.0, 0.0, 0.0])
    n1 = torch.tensor([1.33, 0.0, 0.0])
    ca0 = 1.525 * torch.tensor([math.cos(2.035), math.sin(2.035), 0.0])
    ca1_direction = math.pi - 2.124
    ca1 = n1 + 1.46 * torch.tensor(
        [math.cos(ca1_direction), math.sin(ca1_direction), 0.0]
    )
    positions[0, 0, 1] = ca0
    positions[0, 0, 2] = c0
    positions[0, 1, 0] = n1
    positions[0, 1, 1] = ca1
    return positions, mask


class TrajectoryReliabilityGeometryTest(unittest.TestCase):
    def test_planar_ideal_dipeptide_has_near_zero_errors(self):
        positions, mask = planar_dipeptide()
        metrics = peptide_geometry_metrics(
            positions,
            mask,
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.bool),
        )
        self.assertLess(metrics["peptide_bond_mae_a"].item(), 1e-5)
        self.assertLess(metrics["peptide_angle_mae_rad"].item(), 1e-5)
        self.assertLess(metrics["peptide_omega_planarity_mae_rad"].item(), 1e-5)
        self.assertEqual(metrics["peptide_bond_violation_frac"].item(), 0.0)

    def test_distorted_bond_is_reported_as_violation(self):
        positions, mask = planar_dipeptide()
        positions[0, 1, 0, 0] = 2.0
        metrics = peptide_geometry_metrics(
            positions,
            mask,
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.bool),
        )
        self.assertGreater(metrics["peptide_bond_mae_a"].item(), 0.6)
        self.assertEqual(metrics["peptide_bond_violation_frac"].item(), 1.0)

    def test_non_peptide_neighbor_is_excluded(self):
        positions, mask = planar_dipeptide()
        positions[0, 1, 0, 0] = 20.0
        metrics = peptide_geometry_metrics(
            positions,
            mask,
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 1, dtype=torch.bool),
        )
        self.assertEqual(metrics["peptide_bond_count"].item(), 0.0)
        self.assertEqual(metrics["peptide_bond_mae_a"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
