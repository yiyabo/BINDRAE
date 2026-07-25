import unittest

import torch

from src.stage2.training.trainer import (
    _select_phase_warp_warm_start_state,
    _select_shared_trunk_warm_start_state,
)


class Stage2SharedTrunkWarmStartTest(unittest.TestCase):
    def test_loads_shared_weights_and_resets_path_heads(self):
        source = {
            "encoder.weight": torch.ones(2, 3),
            "residual_gate_mlp.0.weight": torch.ones(1, 3),
        }
        target = {
            "encoder.weight": torch.zeros(2, 3),
            "time_warp_head.0.weight": torch.zeros(1, 3),
            "residual_rotation_head.0.weight": torch.zeros(1, 3),
        }
        selected, reset_target, ignored_source = (
            _select_shared_trunk_warm_start_state(source, target)
        )
        self.assertEqual(set(selected), {"encoder.weight"})
        self.assertEqual(
            set(reset_target),
            {
                "time_warp_head.0.weight",
                "residual_rotation_head.0.weight",
            },
        )
        self.assertEqual(ignored_source, ["residual_gate_mlp.0.weight"])

    def test_rejects_missing_shared_weight(self):
        with self.assertRaisesRegex(RuntimeError, "missing source key encoder.weight"):
            _select_shared_trunk_warm_start_state(
                {}, {"encoder.weight": torch.zeros(2, 3)}
            )

    def test_rejects_shared_shape_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "shape mismatch encoder.weight"):
            _select_shared_trunk_warm_start_state(
                {"encoder.weight": torch.zeros(3, 2)},
                {"encoder.weight": torch.zeros(2, 3)},
            )

    def test_phase_warp_mode_preserves_phase_and_resets_residual(self):
        source = {
            "encoder.weight": torch.ones(2, 3),
            "time_warp_head.0.weight": torch.ones(1, 3),
            "residual_translation_head.0.weight": torch.ones(1, 3),
        }
        target = {
            "encoder.weight": torch.zeros(2, 3),
            "time_warp_head.0.weight": torch.zeros(1, 3),
            "residual_translation_head.0.weight": torch.zeros(1, 3),
        }
        selected, reset_target, ignored_source = (
            _select_phase_warp_warm_start_state(source, target)
        )
        self.assertEqual(
            set(selected), {"encoder.weight", "time_warp_head.0.weight"}
        )
        self.assertEqual(
            reset_target, ["residual_translation_head.0.weight"]
        )
        self.assertEqual(ignored_source, [])


if __name__ == "__main__":
    unittest.main()
