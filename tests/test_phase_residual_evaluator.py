import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from flash_ipa.rigid import Rigid, Rotation

from scripts.evaluate_stage2_transition_paths import (
    build_arg_parser,
    build_rigids_from_backbone,
    construct_path,
    phase_orthogonal_residual_path,
    postprocess_phase_tau_values,
    resolve_path_parameterization,
)


class _EvaluatorModel(nn.Module):
    def forward(self, chi, t, node_mask, **kwargs):
        batch_size, n_res, _ = chi.shape
        mask = node_mask.unsqueeze(-1).float()
        residual_trans = torch.zeros(batch_size, n_res, 3, device=chi.device)
        residual_trans[..., 1] = 0.2
        return {
            "time_warp_logits": torch.zeros(
                batch_size,
                n_res,
                1,
                device=chi.device,
            ),
            "residual_chi": torch.zeros(batch_size, n_res, 4, device=chi.device),
            "residual_rigid_rot": torch.zeros(
                batch_size,
                n_res,
                3,
                device=chi.device,
            ),
            "residual_rigid_trans": residual_trans * mask,
        }


def _rigid(translations):
    batch_size, n_res, _ = translations.shape
    rotations = torch.eye(3).view(1, 1, 3, 3).expand(
        batch_size,
        n_res,
        3,
        3,
    ).clone()
    return Rigid(rots=Rotation(rot_mats=rotations), trans=translations)


class PhaseResidualEvaluatorTest(unittest.TestCase):
    def test_cummax_phase_projection_preserves_endpoints(self):
        tau = [
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[0.4, 0.2]]),
            torch.tensor([[0.3, 0.6]]),
            torch.tensor([[1.0, 1.0]]),
        ]
        projected = torch.stack(postprocess_phase_tau_values(tau, "cummax"))
        self.assertTrue(bool((projected[1:] >= projected[:-1]).all()))
        torch.testing.assert_close(projected[0], torch.zeros_like(projected[0]))
        torch.testing.assert_close(projected[-1], torch.ones_like(projected[-1]))
        self.assertAlmostEqual(float(projected[2, 0, 0]), 0.4)

    def test_parser_accepts_per_sample_metrics_flag(self):
        args = build_arg_parser().parse_args(
            ["--checkpoint", "dummy.pt", "--include_per_sample_metrics"]
        )
        self.assertTrue(args.include_per_sample_metrics)

    def test_resolve_physical_normal_path_mode(self):
        args = SimpleNamespace(path_parameterization="phase_physical_normal_v1")
        config = SimpleNamespace(path_parameterization="flow")
        self.assertEqual(
            resolve_path_parameterization(args, config),
            "phase_physical_normal_v1",
        )

    def test_construct_path_replays_checkpoint_peptide_retraction(self):
        rigids_apo = _rigid(torch.zeros(1, 2, 3))
        rigids_holo = _rigid(torch.ones(1, 2, 3))
        chi = torch.zeros(1, 2, 4)
        phase_path = ([rigids_apo, rigids_holo], [chi, chi], [0.0, 1.0])
        projected_path = ([rigids_apo, rigids_holo], [chi, chi], [0.0, 1.0])
        args = SimpleNamespace(
            path_parameterization="checkpoint",
            phase_residual_tau_mode=None,
            phase_residual_bridge_mode=None,
            time_warp_logit_scale=None,
            time_warp_rate_eps=None,
            time_warp_rate_clip=None,
            phase_residual_envelope=None,
            phase_residual_scale=None,
            phase_residual_rotation_metric_scale=None,
            phase_residual_translation_metric_scale=None,
            phase_residual_chi_metric_scale=None,
            phase_residual_min_tangent_norm=None,
            phase_residual_max_metric_norm=None,
        )
        config = SimpleNamespace(
            path_parameterization="phase_orthogonal_residual_v1",
            phase_residual_peptide_retraction=True,
            phase_residual_peptide_retraction_iterations=6,
            phase_residual_peptide_retraction_relaxation=0.5,
            phase_residual_peptide_retraction_anchor_strength=0.1,
            phase_residual_peptide_retraction_max_translation=0.75,
            phase_residual_peptide_retraction_activation_loss_threshold=0.01,
        )
        fk_module = object()
        batch = SimpleNamespace()

        with patch(
            "scripts.evaluate_stage2_transition_paths.phase_orthogonal_residual_path",
            return_value=phase_path,
        ) as build_phase, patch(
            "scripts.evaluate_stage2_transition_paths.project_peptide_geometry_onto_path",
            return_value=projected_path,
        ) as retract_path:
            result = construct_path(
                args,
                config,
                object(),
                batch,
                rigids_apo,
                rigids_holo,
                n_steps=4,
                interaction_prior=None,
                esm_gate_context=None,
                integration_clips={},
                fk_module=fk_module,
            )

        build_phase.assert_called_once()
        retract_path.assert_called_once_with(
            batch,
            *phase_path,
            fk_module=fk_module,
            n_iterations=6,
            relaxation=0.5,
            anchor_strength=0.1,
            max_translation=0.75,
            activation_loss_threshold=0.01,
        )
        self.assertEqual(result[:3], projected_path)
        self.assertEqual(result[3], {})

    def test_evaluator_preserves_endpoints_and_applies_normal_residual(self):
        torsion_apo = torch.zeros(1, 2, 7)
        torsion_holo = torsion_apo.clone()
        torsion_holo[..., 3] = 0.3
        batch = SimpleNamespace(
            node_mask=torch.ones(1, 2, dtype=torch.bool),
            chi_mask=torch.ones(1, 2, 4, dtype=torch.bool),
            torsion_apo=torsion_apo,
            torsion_holo=torsion_holo,
            esm=torch.zeros(1, 2, 2),
            lig_points=torch.zeros(1, 1, 3),
            lig_types=torch.zeros(1, 1, 20),
            lig_mask=torch.ones(1, 1, dtype=torch.bool),
            w_res=torch.ones(1, 2),
            nma_features=None,
        )
        apo_trans = torch.zeros(1, 2, 3)
        holo_trans = torch.tensor([[[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]])
        rigids_apo = _rigid(apo_trans)
        rigids_holo = _rigid(holo_trans)
        rigids, chi, times = phase_orthogonal_residual_path(
            _EvaluatorModel(),
            batch,
            rigids_apo,
            rigids_holo,
            n_steps=4,
            interaction_prior=None,
            esm_gate_context=None,
            tau_mode="identity",
            warp_variant="residue_monotone",
            nonmonotone_max_offset=0.5,
            bridge_mode="se3_geodesic",
            logit_scale=1.0,
            rate_eps=1e-3,
            rate_clip=10.0,
            envelope_kind="poly",
            residual_scale=1.0,
            rotation_metric_scale=1.0,
            translation_metric_scale=1.0,
            chi_metric_scale=1.0,
            min_tangent_norm=1e-4,
            max_metric_norm=0.5,
        )
        self.assertEqual(times, [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertTrue(torch.equal(rigids[0].get_trans(), apo_trans))
        self.assertTrue(torch.equal(rigids[-1].get_trans(), holo_trans))
        self.assertTrue(torch.equal(chi[-1], torsion_holo[..., 3:7]))
        self.assertGreater(rigids[2].get_trans()[0, 0, 1].item(), 0.1)

    def test_cartesian_bridge_uses_interpolated_backbone_triplets(self):
        torsions = torch.zeros(1, 2, 7)
        ca_apo = torch.tensor([[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]]])
        n_apo = ca_apo + torch.tensor([-1.2, 0.5, 0.0])
        c_apo = ca_apo + torch.tensor([1.3, 0.0, 0.0])
        ca_holo = torch.tensor([[[0.0, 0.0, 0.0], [3.5, 0.8, 0.2]]])
        n_holo = ca_holo + torch.tensor([-1.1, 0.6, 0.1])
        c_holo = ca_holo + torch.tensor([1.25, -0.1, 0.0])
        node_mask = torch.ones(1, 2, dtype=torch.bool)
        batch = SimpleNamespace(
            node_mask=node_mask,
            peptide_bond_mask=torch.ones(1, 1, dtype=torch.bool),
            chi_mask=torch.ones(1, 2, 4, dtype=torch.bool),
            torsion_apo=torsions,
            torsion_holo=torsions.clone(),
            esm=torch.zeros(1, 2, 2),
            lig_points=torch.zeros(1, 1, 3),
            lig_types=torch.zeros(1, 1, 20),
            lig_mask=torch.ones(1, 1, dtype=torch.bool),
            w_res=torch.ones(1, 2),
            nma_features=None,
            N_apo=n_apo,
            Ca_apo=ca_apo,
            C_apo=c_apo,
            N_holo=n_holo,
            Ca_holo=ca_holo,
            C_holo=c_holo,
        )
        rigids_apo = build_rigids_from_backbone(
            n_apo, ca_apo, c_apo, node_mask
        )
        rigids_holo = build_rigids_from_backbone(
            n_holo, ca_holo, c_holo, node_mask
        )
        rigids, _, _ = phase_orthogonal_residual_path(
            _EvaluatorModel(),
            batch,
            rigids_apo,
            rigids_holo,
            n_steps=4,
            interaction_prior=None,
            esm_gate_context=None,
            tau_mode="identity",
            warp_variant="residue_monotone",
            nonmonotone_max_offset=0.5,
            bridge_mode="cartesian_backbone",
            logit_scale=1.0,
            rate_eps=1e-3,
            rate_clip=10.0,
            envelope_kind="poly",
            residual_scale=0.0,
            rotation_metric_scale=1.0,
            translation_metric_scale=1.0,
            chi_metric_scale=1.0,
            min_tangent_norm=1e-4,
            max_metric_norm=0.5,
        )
        self.assertTrue(
            torch.allclose(
                rigids[2].get_trans(), 0.5 * (ca_apo + ca_holo), atol=1e-6
            )
        )


if __name__ == "__main__":
    unittest.main()
