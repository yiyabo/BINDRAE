import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from flash_ipa.rigid import Rigid, Rotation

from src.stage2.training.trainer import Stage2Trainer


class _FakePhaseResidualModel(nn.Module):
    def forward(self, chi, t, node_mask, **kwargs):
        batch_size, n_res, _ = chi.shape
        device = chi.device
        residue_index = torch.arange(n_res, device=device).view(1, n_res, 1)
        time = t.view(batch_size, 1, 1)
        residual_rot = torch.zeros(batch_size, n_res, 3, device=device)
        residual_trans = torch.zeros(batch_size, n_res, 3, device=device)
        residual_trans[..., 1] = 0.25
        residual_chi = torch.zeros(batch_size, n_res, 4, device=device)
        residual_chi[..., 1] = 0.15
        mask = node_mask.unsqueeze(-1).float()
        return {
            "d_chi": residual_chi * 0.0,
            "d_rigid_rot": residual_rot * 0.0,
            "d_rigid_trans": residual_trans * 0.0,
            "gate": mask,
            "time_warp_logits": (residue_index * (time - 0.5)) * mask,
            "residual_chi": residual_chi * mask,
            "residual_rigid_rot": residual_rot * mask,
            "residual_rigid_trans": residual_trans * mask,
            "residual_gate": mask,
        }


class _ZeroPhaseResidualModel(_FakePhaseResidualModel):
    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        out["residual_chi"] = torch.zeros_like(out["residual_chi"])
        out["residual_rigid_rot"] = torch.zeros_like(out["residual_rigid_rot"])
        out["residual_rigid_trans"] = torch.zeros_like(out["residual_rigid_trans"])
        return out


def _rigid(translations):
    batch_size, n_res, _ = translations.shape
    rotations = torch.eye(3).view(1, 1, 3, 3).expand(
        batch_size,
        n_res,
        3,
        3,
    ).clone()
    return Rigid(rots=Rotation(rot_mats=rotations), trans=translations)


class PhaseResidualPathTest(unittest.TestCase):
    def test_scalar_distribution_summary_reports_tail(self):
        summary = Stage2Trainer._scalar_distribution_summary(
            {"pep_interior": [0.0, 1.0, 2.0, 100.0, float("nan")]}
        )
        self.assertAlmostEqual(summary["pep_interior_batch_p50"], 1.5)
        self.assertEqual(summary["pep_interior_batch_max"], 100.0)
        self.assertGreater(summary["pep_interior_batch_p95"], 80.0)

    def _trainer(self, tau_mode):
        trainer = Stage2Trainer.__new__(Stage2Trainer)
        trainer.device = torch.device("cpu")
        trainer.global_step = 0
        trainer.autocast_dtype = None
        trainer.model = _FakePhaseResidualModel()
        trainer.config = SimpleNamespace(
            n_integration_steps=4,
            phase_residual_tau_mode=tau_mode,
            time_warp_logit_scale=1.0,
            time_warp_rate_eps=1e-3,
            time_warp_rate_clip=10.0,
            phase_residual_rotation_metric_scale=1.0,
            phase_residual_translation_metric_scale=1.0,
            phase_residual_chi_metric_scale=1.0,
            phase_residual_min_tangent_norm=1e-4,
            phase_residual_max_metric_norm=0.0,
            phase_residual_envelope="poly",
            phase_residual_scale=1.0,
            bg_beta=1.5,
        )
        return trainer

    def _batch(self):
        batch_size, n_res = 1, 3
        torsion_apo = torch.zeros(batch_size, n_res, 7)
        torsion_holo = torsion_apo.clone()
        torsion_holo[..., 3] = torch.tensor([0.4, 0.2, 0.0])
        return SimpleNamespace(
            node_mask=torch.ones(batch_size, n_res, dtype=torch.bool),
            peptide_bond_mask=torch.ones(batch_size, n_res - 1, dtype=torch.bool),
            chi_mask=torch.ones(batch_size, n_res, 4, dtype=torch.bool),
            torsion_apo=torsion_apo,
            torsion_holo=torsion_holo,
            esm=torch.zeros(batch_size, n_res, 2),
            lig_points=torch.zeros(batch_size, 1, 3),
            lig_types=torch.zeros(batch_size, 1, 20),
            lig_mask=torch.ones(batch_size, 1, dtype=torch.bool),
            w_res=torch.tensor([[1.0, 0.5, 0.0]]),
            nma_features=None,
        )

    def test_identity_phase_path_has_exact_endpoints_and_off_bridge_motion(self):
        trainer = self._trainer("identity")
        batch = self._batch()
        apo_trans = torch.zeros(1, 3, 3)
        holo_trans = torch.tensor([[[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        rigids_apo = _rigid(apo_trans)
        rigids_holo = _rigid(holo_trans)

        rigids, chi, times = trainer.phase_orthogonal_residual_path(
            batch,
            rigids_apo,
            rigids_holo,
        )

        self.assertEqual(times, [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertTrue(torch.equal(rigids[0].get_trans(), apo_trans))
        self.assertTrue(torch.equal(rigids[-1].get_trans(), holo_trans))
        self.assertTrue(torch.equal(chi[0], batch.torsion_apo[..., 3:7]))
        self.assertTrue(torch.equal(chi[-1], batch.torsion_holo[..., 3:7]))
        self.assertGreater(rigids[2].get_trans()[0, 0, 1].item(), 0.1)
        self.assertEqual(rigids[2].get_trans()[0, 2, 1].item(), 0.0)
        self.assertLess(
            trainer._last_phase_residual_stats[
                "phase_residual_projected_parallel_cos"
            ].item(),
            1e-5,
        )

        regularization = trainer._phase_residual_regularization(batch)
        for value in regularization.values():
            self.assertTrue(torch.isfinite(value).item())

    def test_learned_phase_is_monotone_and_endpoint_exact(self):
        trainer = self._trainer("learned")
        batch = self._batch()
        rigids_apo = _rigid(torch.zeros(1, 3, 3))
        rigids_holo = _rigid(
            torch.tensor([[[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.2, 0.0, 0.0]]])
        )
        rigids, chi, times = trainer.phase_orthogonal_residual_path(
            batch,
            rigids_apo,
            rigids_holo,
        )
        self.assertEqual(times[0], 0.0)
        self.assertEqual(times[-1], 1.0)
        self.assertTrue(torch.equal(rigids[-1].get_trans(), rigids_holo.get_trans()))
        self.assertTrue(torch.equal(chi[-1], batch.torsion_holo[..., 3:7]))
        self.assertGreater(
            trainer._last_timewarp_stats["time_warp_tau_abs_mean"].item(),
            0.0,
        )

    def test_zero_residual_reduces_exactly_to_phase_bridge(self):
        trainer = self._trainer("identity")
        trainer.model = _ZeroPhaseResidualModel()
        batch = self._batch()
        rigids_apo = _rigid(torch.zeros(1, 3, 3))
        rigids_holo = _rigid(
            torch.tensor([[[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.2, 0.0, 0.0]]])
        )
        rigids, chi, times = trainer.phase_orthogonal_residual_path(
            batch,
            rigids_apo,
            rigids_holo,
        )
        for rigid_t, chi_t, time in zip(rigids, chi, times):
            expected_rigid, expected_chi = trainer._interpolate_endpoints(
                batch,
                rigids_apo,
                rigids_holo,
                time,
            )
            self.assertTrue(
                torch.allclose(rigid_t.get_trans(), expected_rigid.get_trans(), atol=1e-7)
            )
            self.assertTrue(torch.allclose(chi_t, expected_chi, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
