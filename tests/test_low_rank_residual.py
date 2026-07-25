import unittest

import torch

from src.stage2.modules.low_rank_residual import (
    GraphCoupledLowRankResidualDecoder,
)


class GraphCoupledLowRankResidualDecoderTest(unittest.TestCase):
    def _decoder(self, rank=2):
        return GraphCoupledLowRankResidualDecoder(
            residue_dim=12,
            time_dim=6,
            hidden_dim=16,
            rank=rank,
            active_blocks={"rotation", "translation", "chi"},
            rotation_gate_bias=-2.0,
            translation_gate_bias=-4.0,
            chi_gate_bias=-2.0,
        )

    def test_zero_initialized_coefficients_start_at_zero_residual(self):
        decoder = self._decoder()
        output = decoder(
            torch.randn(2, 5, 12),
            torch.randn(2, 6),
            torch.ones(2, 5, dtype=torch.bool),
            torch.ones(2, 5, 4, dtype=torch.bool),
        )
        self.assertEqual(output["residual_rotation"].abs().max().item(), 0.0)
        self.assertEqual(output["residual_translation"].abs().max().item(), 0.0)
        self.assertEqual(output["residual_chi"].abs().max().item(), 0.0)

    def test_padding_and_missing_chi_are_hard_masked(self):
        decoder = self._decoder()
        decoder.coefficient_heads["chi"][-1].bias.data.fill_(1.0)
        node_mask = torch.tensor([[True, True, False]])
        chi_mask = torch.tensor(
            [[[True, False, True, True], [True, True, True, True], [True] * 4]]
        )
        output = decoder(
            torch.randn(1, 3, 12),
            torch.randn(1, 6),
            node_mask,
            chi_mask,
        )
        self.assertEqual(output["residual_chi"][0, 0, 1].item(), 0.0)
        self.assertEqual(output["residual_chi"][0, 2].abs().sum().item(), 0.0)
        self.assertEqual(output["residual_gate"][0, 2].item(), 0.0)

    def test_fixed_spatial_basis_has_at_most_configured_temporal_rank(self):
        torch.manual_seed(7)
        decoder = self._decoder(rank=2)
        torch.nn.init.normal_(decoder.coefficient_heads["rotation"][-1].weight)
        features = torch.randn(1, 7, 12)
        node_mask = torch.ones(1, 7, dtype=torch.bool)
        chi_mask = torch.ones(1, 7, 4, dtype=torch.bool)
        paths = []
        for _ in range(6):
            output = decoder(
                features,
                torch.randn(1, 6),
                node_mask,
                chi_mask,
            )
            paths.append(output["residual_rotation"].reshape(-1))
        matrix = torch.stack(paths, dim=0)
        singular_values = torch.linalg.svdvals(matrix)
        numerical_rank = int((singular_values > singular_values[0] * 1e-5).sum())
        self.assertLessEqual(numerical_rank, 2)

    def test_inactive_block_is_exactly_zero(self):
        decoder = GraphCoupledLowRankResidualDecoder(
            residue_dim=8,
            time_dim=4,
            hidden_dim=12,
            rank=2,
            active_blocks={"rotation"},
            rotation_gate_bias=0.0,
            translation_gate_bias=0.0,
            chi_gate_bias=0.0,
        )
        decoder.coefficient_heads["translation"][-1].bias.data.fill_(3.0)
        output = decoder(
            torch.randn(1, 3, 8),
            torch.randn(1, 4),
            torch.ones(1, 3, dtype=torch.bool),
            torch.ones(1, 3, 4, dtype=torch.bool),
        )
        self.assertEqual(output["residual_translation"].abs().sum().item(), 0.0)
        self.assertEqual(output["residual_translation_gate"].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
