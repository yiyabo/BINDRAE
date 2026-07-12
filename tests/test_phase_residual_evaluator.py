import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from flash_ipa.rigid import Rigid, Rotation

from scripts.evaluate_stage2_transition_paths import (
    build_rigids_from_backbone,
    phase_orthogonal_residual_path,
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
