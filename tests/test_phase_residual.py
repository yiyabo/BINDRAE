import unittest

import torch

from src.stage2.modules.phase_residual import (
    endpoint_zero_envelope,
    phase_tau_from_logits,
    project_block_tangent_normal,
    project_product_tangent_normal,
)


class PhaseResidualProjectionTest(unittest.TestCase):
    def test_zero_logits_recover_identity_monotone_phase(self):
        logits = torch.zeros(4, 2, 3)
        node_mask = torch.tensor([[True, True, True], [True, True, False]])
        peptide_bond_mask = torch.tensor([[True, True], [True, False]])
        expected = torch.linspace(0.0, 1.0, 5).view(5, 1, 1)
        for variant in (
            "residue_monotone",
            "global_monotone",
            "global_chain_monotone",
            "chain_nonmonotone",
        ):
            result = phase_tau_from_logits(
                logits,
                node_mask,
                variant=variant,
                peptide_bond_mask=peptide_bond_mask,
            )
            torch.testing.assert_close(
                result["tau"], expected.expand_as(result["tau"])
            )

    def test_chain_monotone_is_endpoint_fixed_and_monotone(self):
        torch.manual_seed(23)
        logits = torch.randn(5, 2, 4)
        node_mask = torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        )
        peptide_bond_mask = torch.tensor(
            [[True, True, True], [True, True, False]]
        )
        tau = phase_tau_from_logits(
            logits,
            node_mask,
            variant="global_chain_monotone",
            peptide_bond_mask=peptide_bond_mask,
        )["tau"]
        self.assertTrue(torch.equal(tau[0], torch.zeros_like(tau[0])))
        self.assertTrue(torch.equal(tau[-1], torch.ones_like(tau[-1])))
        self.assertTrue((torch.diff(tau, dim=0) >= 0.0).all().item())

    def test_zero_chain_scale_exactly_recovers_global_monotone(self):
        torch.manual_seed(29)
        logits = torch.randn(5, 2, 4)
        node_mask = torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        )
        global_result = phase_tau_from_logits(
            logits,
            node_mask,
            variant="global_monotone",
        )
        chain_result = phase_tau_from_logits(
            logits,
            node_mask,
            variant="global_chain_monotone",
            peptide_bond_mask=torch.tensor(
                [[True, True, True], [True, True, False]]
            ),
            chain_residual_scale=0.0,
        )
        torch.testing.assert_close(chain_result["tau"], global_result["tau"])
        torch.testing.assert_close(
            chain_result["interval_rate"], global_result["interval_rate"]
        )

    def test_chain_smoothing_reduces_neighbor_phase_discontinuity(self):
        alternating = torch.tensor(
            [
                [-4.0, 4.0, -4.0, 4.0, -4.0],
                [4.0, -4.0, 4.0, -4.0, 4.0],
                [-4.0, 4.0, -4.0, 4.0, -4.0],
                [4.0, -4.0, 4.0, -4.0, 4.0],
            ]
        ).unsqueeze(1)
        node_mask = torch.ones(1, 5, dtype=torch.bool)
        peptide_bond_mask = torch.ones(1, 4, dtype=torch.bool)
        unsmoothed = phase_tau_from_logits(
            alternating,
            node_mask,
            variant="global_chain_monotone",
            peptide_bond_mask=peptide_bond_mask,
            chain_smoothing_steps=0,
        )["tau"]
        smoothed = phase_tau_from_logits(
            alternating,
            node_mask,
            variant="global_chain_monotone",
            peptide_bond_mask=peptide_bond_mask,
            chain_smoothing_steps=2,
        )["tau"]
        unsmoothed_jump = torch.diff(unsmoothed[1:-1, 0], dim=-1).abs().mean()
        smoothed_jump = torch.diff(smoothed[1:-1, 0], dim=-1).abs().mean()
        self.assertLess(smoothed_jump.item(), unsmoothed_jump.item())

    def test_chain_monotone_has_finite_gradients(self):
        logits = torch.randn(4, 2, 3, requires_grad=True)
        result = phase_tau_from_logits(
            logits,
            torch.ones(2, 3, dtype=torch.bool),
            variant="global_chain_monotone",
            peptide_bond_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        result["tau"][1:-1].square().mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_global_monotone_uses_one_phase_per_system(self):
        logits = torch.tensor(
            [
                [[-2.0, 1.0, 3.0], [2.0, -4.0, 100.0]],
                [[3.0, -1.0, 0.0], [-2.0, 5.0, -100.0]],
                [[0.0, 4.0, -2.0], [1.0, 1.0, 100.0]],
            ]
        )
        node_mask = torch.tensor([[True, True, True], [True, True, False]])
        tau = phase_tau_from_logits(
            logits,
            node_mask,
            variant="global_monotone",
        )["tau"]
        torch.testing.assert_close(tau[:, 0, 0], tau[:, 0, 2])
        torch.testing.assert_close(tau[:, 1, 0], tau[:, 1, 1])
        torch.testing.assert_close(
            tau[:, 1, 2], torch.linspace(0.0, 1.0, 4)
        )

    def test_nonmonotone_control_is_endpoint_fixed_but_can_backtrack(self):
        logits = torch.tensor([5.0, -5.0, 5.0, 0.0]).view(4, 1, 1)
        result = phase_tau_from_logits(
            logits,
            torch.ones(1, 1, dtype=torch.bool),
            variant="residue_nonmonotone",
            nonmonotone_max_offset=0.5,
        )
        tau = result["tau"][:, 0, 0]
        self.assertEqual(tau[0].item(), 0.0)
        self.assertEqual(tau[-1].item(), 1.0)
        self.assertTrue((torch.diff(tau) < 0.0).any().item())

    def test_chain_nonmonotone_is_endpoint_fixed_but_can_backtrack(self):
        logits = torch.tensor([5.0, -5.0, 5.0, 0.0]).view(4, 1, 1)
        result = phase_tau_from_logits(
            logits,
            torch.ones(1, 1, dtype=torch.bool),
            variant="chain_nonmonotone",
            nonmonotone_max_offset=0.5,
            peptide_bond_mask=torch.ones(1, 0, dtype=torch.bool),
        )
        tau = result["tau"][:, 0, 0]
        self.assertEqual(tau[0].item(), 0.0)
        self.assertEqual(tau[-1].item(), 1.0)
        self.assertTrue((torch.diff(tau) < 0.0).any().item())

    def test_chain_nonmonotone_steps_zero_recovers_residue_nonmonotone(self):
        torch.manual_seed(31)
        logits = torch.randn(4, 2, 4)
        node_mask = torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        )
        residue_result = phase_tau_from_logits(
            logits,
            node_mask,
            variant="residue_nonmonotone",
        )
        chain_result = phase_tau_from_logits(
            logits,
            node_mask,
            variant="chain_nonmonotone",
            peptide_bond_mask=torch.tensor(
                [[True, True, True], [True, True, False]]
            ),
            chain_residual_scale=1.0,
            chain_smoothing_steps=0,
        )
        torch.testing.assert_close(chain_result["tau"], residue_result["tau"])
        torch.testing.assert_close(
            chain_result["interval_rate"], residue_result["interval_rate"]
        )

    def test_chain_nonmonotone_smoothing_reduces_neighbor_phase_jumps(self):
        logits = torch.tensor(
            [
                [-4.0, 4.0, -4.0, 4.0, -4.0],
                [4.0, -4.0, 4.0, -4.0, 4.0],
                [-4.0, 4.0, -4.0, 4.0, -4.0],
                [4.0, -4.0, 4.0, -4.0, 4.0],
            ]
        ).unsqueeze(1)
        node_mask = torch.ones(1, 5, dtype=torch.bool)
        residue_tau = phase_tau_from_logits(
            logits,
            node_mask,
            variant="residue_nonmonotone",
        )["tau"]
        chain_tau = phase_tau_from_logits(
            logits,
            node_mask,
            variant="chain_nonmonotone",
            peptide_bond_mask=torch.ones(1, 4, dtype=torch.bool),
            chain_smoothing_steps=2,
        )["tau"]
        residue_jump = torch.diff(residue_tau[1:-1, 0], dim=-1).abs().mean()
        chain_jump = torch.diff(chain_tau[1:-1, 0], dim=-1).abs().mean()
        self.assertLess(chain_jump.item(), residue_jump.item())

    def test_chain_nonmonotone_has_finite_gradients(self):
        logits = torch.randn(4, 2, 3, requires_grad=True)
        result = phase_tau_from_logits(
            logits,
            torch.ones(2, 3, dtype=torch.bool),
            variant="chain_nonmonotone",
            peptide_bond_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        result["tau"][1:-1].square().mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_phase_controls_have_finite_gradients(self):
        logits = torch.randn(4, 2, 3, requires_grad=True)
        result = phase_tau_from_logits(
            logits,
            torch.ones(2, 3, dtype=torch.bool),
            variant="residue_nonmonotone",
        )
        result["tau"][1:-1].square().mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

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
