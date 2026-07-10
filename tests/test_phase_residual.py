import unittest

import torch

from src.stage2.modules.phase_residual import (
    endpoint_zero_envelope,
    project_product_tangent_normal,
)


class PhaseResidualProjectionTest(unittest.TestCase):
    def test_endpoint_envelope_is_exact(self):
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        envelope = endpoint_zero_envelope(t, kind="poly")
        self.assertEqual(envelope[0].item(), 0.0)
        self.assertEqual(envelope[-1].item(), 0.0)
        self.assertAlmostEqual(envelope[2].item(), 1.0, places=6)

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


if __name__ == "__main__":
    unittest.main()
