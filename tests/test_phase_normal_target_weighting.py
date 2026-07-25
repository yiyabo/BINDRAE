import unittest

import torch

from src.stage2.training.trainer import _phase_normal_target_weights


class PhaseNormalTargetWeightingTest(unittest.TestCase):
    def test_uniform_preserves_base_weight(self):
        base = torch.tensor([[[1.0, 0.0]]])
        rigid = torch.zeros(1, 1, 2, 6)
        chi = torch.zeros(1, 1, 2, 4)
        chi_mask = torch.ones_like(chi, dtype=torch.bool)
        actual = _phase_normal_target_weights(
            base,
            rigid,
            chi,
            chi_mask,
            mode="uniform",
            magnitude_scale=0.05,
            magnitude_boost=9.0,
        )
        self.assertTrue(torch.equal(actual, base))

    def test_target_magnitude_boost_is_bounded_and_respects_mask(self):
        base = torch.tensor([[[1.0, 1.0, 0.0]]])
        rigid = torch.zeros(1, 1, 3, 6)
        rigid[0, 0, 0, 3] = 0.025
        rigid[0, 0, 1, 3] = 0.10
        rigid[0, 0, 2, 3] = 0.10
        chi = torch.zeros(1, 1, 3, 4)
        chi_mask = torch.ones_like(chi, dtype=torch.bool)
        actual = _phase_normal_target_weights(
            base,
            rigid,
            chi,
            chi_mask,
            mode="target_magnitude",
            magnitude_scale=0.05,
            magnitude_boost=9.0,
        )
        self.assertAlmostEqual(float(actual[0, 0, 0]), 5.5, places=5)
        self.assertAlmostEqual(float(actual[0, 0, 1]), 10.0, places=5)
        self.assertEqual(float(actual[0, 0, 2]), 0.0)

    def test_applied_path_uses_squared_time_envelope(self):
        base = torch.ones(3, 1, 1)
        rigid = torch.zeros(3, 1, 1, 6)
        chi = torch.zeros(3, 1, 1, 4)
        chi_mask = torch.ones_like(chi, dtype=torch.bool)
        actual = _phase_normal_target_weights(
            base,
            rigid,
            chi,
            chi_mask,
            mode="applied_path",
            magnitude_scale=0.05,
            magnitude_boost=0.0,
            time_envelope=torch.tensor([0.2, 1.0, 0.2]),
        )
        expected = torch.tensor([0.04, 1.0, 0.04]).view(3, 1, 1)
        self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
