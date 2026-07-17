import unittest

import torch

from src.stage2.modules.phase_residual import (
    endpoint_zero_envelope,
    project_block_tangent_normal,
    project_product_tangent_normal,
)


class PhaseResidualProjectionTest(unittest.TestCase):
    def test_endpoint_envelope_is_exact(self):
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        envelope = endpoint_zero_envelope(t)
        self.assertEqual(envelope[0].item(), 0.0)
        self.assertEqual(envelope[-1].item(), 0.0)
        self.assertAlmostEqual(envelope[2].item(), 1.0, places=6)

    def test_default_envelope_has_zero_endpoint_slope(self):
        epsilon = 1e-4
        t = torch.tensor([0.0, epsilon, 1.0 - epsilon, 1.0])
        envelope = endpoint_zero_envelope(t)
        left_slope = (envelope[1] - envelope[0]) / epsilon
        right_slope = (envelope[-1] - envelope[-2]) / epsilon
        self.assertLess(abs(left_slope.item()), 2e-3)
        self.assertLess(abs(right_slope.item()), 2e-3)

    def test_projection_is_metric_orthogonal(self):
        torch.manual_seed(7)
        residual_rigid = torch.randn(2, 5, 6)
        residual_chi = torch.randn(2, 5, 4)
        tangent_rigid = torch.randn(2, 5, 6)
        tangent_chi = torch.randn(2, 5, 4)
        result = project_product_tangent_normal(
            residual_rigid,
            residual_chi,
            tangent_rigid,
            tangent_chi,
            rotation_scale=0.5,
            translation_scale=2.0,
            chi_scale=0.75,
        )

        projected = torch.cat(
            [
                result["projected_rigid"][..., :3] / 0.5,
                result["projected_rigid"][..., 3:] / 2.0,
                result["projected_chi"] / 0.75,
            ],
            dim=-1,
        )
        tangent = torch.cat(
            [
                tangent_rigid[..., :3] / 0.5,
                tangent_rigid[..., 3:] / 2.0,
                tangent_chi / 0.75,
            ],
            dim=-1,
        )
        dot = (projected * tangent).sum(dim=-1)
        self.assertLess(dot.abs().max().item(), 2e-5)

    def test_parallel_residual_is_removed(self):
        tangent_rigid = torch.tensor([[[0.2, -0.1, 0.3, 1.0, 2.0, -0.5]]])
        tangent_chi = torch.tensor([[[0.4, -0.2, 0.1, 0.0]]])
        result = project_product_tangent_normal(
            3.5 * tangent_rigid,
            3.5 * tangent_chi,
            tangent_rigid,
            tangent_chi,
        )
        self.assertLess(result["projected_rigid"].abs().max().item(), 2e-6)
        self.assertLess(result["projected_chi"].abs().max().item(), 2e-6)

    def test_near_zero_endpoint_motion_disables_residual(self):
        result = project_product_tangent_normal(
            torch.ones(1, 2, 6),
            torch.ones(1, 2, 4),
            torch.zeros(1, 2, 6),
            torch.zeros(1, 2, 4),
            min_tangent_norm=1e-3,
        )
        self.assertFalse(result["active_mask"].any().item())
        self.assertEqual(result["projected_rigid"].abs().sum().item(), 0.0)
        self.assertEqual(result["projected_chi"].abs().sum().item(), 0.0)

    def test_projection_has_finite_gradients(self):
        torch.manual_seed(11)
        residual_rigid = torch.randn(2, 3, 6, requires_grad=True)
        residual_chi = torch.randn(2, 3, 4, requires_grad=True)
        tangent_rigid = torch.randn(2, 3, 6)
        tangent_chi = torch.randn(2, 3, 4)
        result = project_product_tangent_normal(
            residual_rigid,
            residual_chi,
            tangent_rigid,
            tangent_chi,
        )
        loss = (
            result["projected_rigid"].square().mean()
            + result["projected_chi"].square().mean()
        )
        loss.backward()
        self.assertTrue(torch.isfinite(residual_rigid.grad).all().item())
        self.assertTrue(torch.isfinite(residual_chi.grad).all().item())

    def test_metric_norm_cap_preserves_orthogonality(self):
        torch.manual_seed(17)
        tangent_rigid = torch.randn(2, 4, 6)
        tangent_chi = torch.randn(2, 4, 4)
        result = project_product_tangent_normal(
            20.0 * torch.randn(2, 4, 6),
            20.0 * torch.randn(2, 4, 4),
            tangent_rigid,
            tangent_chi,
            max_metric_norm=0.5,
        )
        projected = torch.cat(
            [result["projected_rigid"], result["projected_chi"]],
            dim=-1,
        )
        tangent = torch.cat([tangent_rigid, tangent_chi], dim=-1)
        self.assertLessEqual(projected.norm(dim=-1).max().item(), 0.50001)
        self.assertLess((projected * tangent).sum(dim=-1).abs().max().item(), 2e-5)

    def test_block_projection_is_orthogonal_within_each_component(self):
        torch.manual_seed(19)
        residual_rigid = torch.randn(2, 5, 6)
        residual_chi = torch.randn(2, 5, 4)
        tangent_rigid = torch.randn(2, 5, 6)
        tangent_chi = torch.randn(2, 5, 4)
        result = project_block_tangent_normal(
            residual_rigid,
            residual_chi,
            tangent_rigid,
            tangent_chi,
            rotation_scale=0.5,
            translation_scale=2.0,
            chi_scale=0.75,
        )

        block_pairs = (
            (
                result["projected_rigid"][..., :3] / 0.5,
                tangent_rigid[..., :3] / 0.5,
            ),
            (
                result["projected_rigid"][..., 3:] / 2.0,
                tangent_rigid[..., 3:] / 2.0,
            ),
            (result["projected_chi"] / 0.75, tangent_chi / 0.75),
        )
        for projected, tangent in block_pairs:
            dot = (projected * tangent).sum(dim=-1)
            self.assertLess(dot.abs().max().item(), 2e-5)

    def test_block_projection_does_not_trade_chi_for_translation(self):
        tangent_rigid = torch.tensor([[[0.2, 0.0, 0.0, 1.0, 0.0, 0.0]]])
        tangent_chi = torch.tensor([[[0.5, 0.0, 0.0, 0.0]]])
        residual_rigid = torch.zeros_like(tangent_rigid)
        residual_chi = 3.0 * tangent_chi
        result = project_block_tangent_normal(
            residual_rigid,
            residual_chi,
            tangent_rigid,
            tangent_chi,
        )
        self.assertEqual(result["projected_rigid"].abs().sum().item(), 0.0)
        self.assertLess(result["projected_chi"].abs().max().item(), 1e-6)

    def test_zero_tangent_block_is_retained_when_residue_motion_is_active(self):
        tangent_rigid = torch.tensor([[[0.4, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        tangent_chi = torch.zeros(1, 1, 4)
        residual_rigid = torch.tensor([[[0.0, 0.2, 0.0, 1.5, -0.5, 0.25]]])
        residual_chi = torch.tensor([[[0.1, -0.2, 0.0, 0.0]]])
        result = project_block_tangent_normal(
            residual_rigid,
            residual_chi,
            tangent_rigid,
            tangent_chi,
        )
        torch.testing.assert_close(
            result["projected_rigid"][..., 3:], residual_rigid[..., 3:]
        )
        torch.testing.assert_close(result["projected_chi"], residual_chi)

    def test_block_projection_disables_residue_with_no_endpoint_motion(self):
        result = project_block_tangent_normal(
            torch.ones(1, 2, 6),
            torch.ones(1, 2, 4),
            torch.zeros(1, 2, 6),
            torch.zeros(1, 2, 4),
            min_tangent_norm=1e-3,
        )
        self.assertFalse(result["active_mask"].any().item())
        self.assertEqual(result["projected_rigid"].abs().sum().item(), 0.0)
        self.assertEqual(result["projected_chi"].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
