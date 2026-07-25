import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
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


class _RecordingPhaseResidualModel(_FakePhaseResidualModel):
    def __init__(self):
        super().__init__()
        self.rigids_seen = []

    def forward(self, *args, **kwargs):
        self.rigids_seen.append(kwargs["rigids"])
        return super().forward(*args, **kwargs)


class _BackboneOnlyFK(nn.Module):
    def forward(self, torsions_sincos, rigids, aatype):
        del torsions_sincos, aatype
        rotation = rigids.get_rots().get_rot_mats()
        translation = rigids.get_trans()
        local = translation.new_zeros((*translation.shape[:-1], 14, 3))
        local[..., 0, :] = local.new_tensor([-1.20, 0.50, 0.0])
        local[..., 2, :] = local.new_tensor([1.30, 0.0, 0.0])
        position = torch.einsum('bnij,bnaj->bnai', rotation, local)
        position = position + translation.unsqueeze(-2)
        mask = torch.zeros(position.shape[:-1], dtype=torch.bool, device=position.device)
        mask[..., :3] = True
        return {'atom14_pos': position, 'atom14_mask': mask}


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
            phase_warp_variant="residue_monotone",
            phase_nonmonotone_max_offset=0.5,
            phase_residual_bridge_mode="se3_geodesic",
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
            phase_residual_peptide_retraction=False,
            phase_residual_peptide_retraction_iterations=16,
            phase_residual_peptide_retraction_relaxation=0.75,
            phase_residual_peptide_retraction_anchor_strength=0.0,
            phase_residual_peptide_retraction_max_translation=1.0,
            phase_residual_peptide_retraction_activation_loss_threshold=0.0,
            pep_bond_len=1.33,
            pep_angle_cacn=2.035,
            pep_angle_cnca=2.124,
            bg_beta=1.5,
        )
        return trainer

    def _batch(self):
        batch_size, n_res = 1, 3
        torsion_apo = torch.zeros(batch_size, n_res, 7)
        torsion_holo = torsion_apo.clone()
        torsion_holo[..., 3] = torch.tensor([0.4, 0.2, 0.0])
        ca_apo = torch.tensor(
            [[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.6, 0.0, 0.0]]]
        )
        n_apo = ca_apo + torch.tensor([-1.2, 0.5, 0.0])
        c_apo = ca_apo + torch.tensor([1.3, 0.0, 0.0])
        ca_holo = torch.tensor(
            [[[0.0, 0.0, 0.0], [3.6, 0.6, 0.0], [7.0, 1.5, 0.2]]]
        )
        n_holo = ca_holo + torch.tensor([-1.1, 0.6, 0.1])
        c_holo = ca_holo + torch.tensor([1.25, -0.1, 0.0])
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
            N_apo=n_apo,
            Ca_apo=ca_apo,
            C_apo=c_apo,
            N_holo=n_holo,
            Ca_holo=ca_holo,
            C_holo=c_holo,
            aatype=torch.zeros(batch_size, n_res, dtype=torch.long),
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

    def test_learned_phase_head_uses_configured_cartesian_bridge(self):
        trainer = self._trainer("learned")
        trainer.config.phase_residual_bridge_mode = "cartesian_backbone"
        trainer.model = _RecordingPhaseResidualModel()
        batch = self._batch()
        rigids_apo = trainer._build_rigids_from_backbone(
            batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
        )
        rigids_holo = trainer._build_rigids_from_backbone(
            batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
        )

        trainer._phase_residual_tau_values(batch, rigids_apo, rigids_holo)

        first_midpoint = torch.full((1, 3), 0.125)
        expected_rigids, _ = trainer._phase_interpolate_endpoints_tensor(
            batch, rigids_apo, rigids_holo, first_midpoint
        )
        actual_rigids = trainer.model.rigids_seen[0]
        self.assertTrue(
            torch.allclose(
                actual_rigids.get_trans(), expected_rigids.get_trans(), atol=1e-7
            )
        )
        self.assertTrue(
            torch.allclose(
                actual_rigids.get_rots().get_rot_mats(),
                expected_rigids.get_rots().get_rot_mats(),
                atol=1e-7,
            )
        )

    def test_validation_replica_selection_is_stable(self):
        trainer = self._trainer("identity")
        trainer.current_epoch = 1
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = root / "sample__silver_r000.npz"
            second = root / "sample__silver_r001.npz"
            first.touch()
            second.touch()

            trainer._validation_mode = False
            self.assertEqual(
                trainer._replicated_supervision_cache_path(tmp_dir, "sample"),
                second,
            )
            trainer._validation_mode = True
            self.assertEqual(
                trainer._replicated_supervision_cache_path(tmp_dir, "sample"),
                first,
            )

    def test_normal_supervision_uses_target_phase_bridge(self):
        trainer = self._trainer("learned")
        trainer.config.phase_residual_bridge_mode = "cartesian_backbone"
        trainer.model = _RecordingPhaseResidualModel()
        batch = self._batch()
        rigids_apo = trainer._build_rigids_from_backbone(
            batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
        )
        rigids_holo = trainer._build_rigids_from_backbone(
            batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
        )
        target_tau = torch.full((2, 1, 3), 0.75)

        records = trainer._phase_normal_teacher_forced_records(
            batch,
            rigids_apo,
            rigids_holo,
            target_tau,
            [0.25, 0.5],
        )

        expected_rigids, _ = trainer._phase_interpolate_endpoints_tensor(
            batch, rigids_apo, rigids_holo, target_tau[0]
        )
        self.assertEqual(len(records), 2)
        self.assertTrue(
            torch.allclose(
                trainer.model.rigids_seen[0].get_trans(),
                expected_rigids.get_trans(),
                atol=1e-7,
            )
        )
        self.assertTrue(
            torch.allclose(
                trainer.model.rigids_seen[0].get_rots().get_rot_mats(),
                expected_rigids.get_rots().get_rot_mats(),
                atol=1e-7,
            )
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

    def test_zero_residual_scale_is_strict_warp_only(self):
        trainer = self._trainer("learned")
        trainer.config.phase_residual_scale = 0.0
        batch = self._batch()
        rigids_apo = _rigid(torch.zeros(1, 3, 3))
        rigids_holo = _rigid(
            torch.tensor([[[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.2, 0.0, 0.0]]])
        )
        rigids, chi, _ = trainer.phase_orthogonal_residual_path(
            batch,
            rigids_apo,
            rigids_holo,
        )
        for index, tau in enumerate(trainer._last_phase_tau_values):
            expected_rigid, expected_chi = trainer._phase_interpolate_endpoints_tensor(
                batch,
                rigids_apo,
                rigids_holo,
                tau,
            )
            self.assertTrue(
                torch.allclose(
                    rigids[index].get_trans(),
                    expected_rigid.get_trans(),
                    atol=1e-7,
                )
            )
            self.assertTrue(torch.allclose(chi[index], expected_chi, atol=1e-7))

    def test_cartesian_bridge_rebuilds_frames_from_backbone_triplets(self):
        trainer = self._trainer("identity")
        trainer.config.phase_residual_bridge_mode = "cartesian_backbone"
        trainer.model = _ZeroPhaseResidualModel()
        batch = self._batch()
        rigids_apo = trainer._build_rigids_from_backbone(
            batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
        )
        rigids_holo = trainer._build_rigids_from_backbone(
            batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
        )
        rigids, _, times = trainer.phase_orthogonal_residual_path(
            batch, rigids_apo, rigids_holo
        )
        self.assertEqual(times, [0.0, 0.25, 0.5, 0.75, 1.0])
        expected_mid_ca = 0.5 * (batch.Ca_apo + batch.Ca_holo)
        self.assertTrue(
            torch.allclose(rigids[2].get_trans(), expected_mid_ca, atol=1e-6)
        )
        self.assertTrue(torch.equal(rigids[0].get_trans(), batch.Ca_apo))
        self.assertTrue(torch.equal(rigids[-1].get_trans(), batch.Ca_holo))

    def test_peptide_retraction_is_bounded_and_preserves_endpoints(self):
        batch = self._batch()
        trainer = self._trainer("identity")
        trainer.config.phase_residual_bridge_mode = "cartesian_backbone"
        trainer.config.phase_residual_peptide_retraction = True
        trainer.fk_module = _BackboneOnlyFK()
        rigids_apo = trainer._build_rigids_from_backbone(
            batch.N_apo, batch.Ca_apo, batch.C_apo, batch.node_mask
        )
        rigids_holo = trainer._build_rigids_from_backbone(
            batch.N_holo, batch.Ca_holo, batch.C_holo, batch.node_mask
        )

        rigids, chi, times = trainer.phase_orthogonal_residual_path(
            batch, rigids_apo, rigids_holo
        )

        self.assertEqual(times, [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertTrue(torch.equal(rigids[0].get_trans(), batch.Ca_apo))
        self.assertTrue(torch.equal(rigids[-1].get_trans(), batch.Ca_holo))
        self.assertTrue(torch.equal(chi[-1], batch.torsion_holo[..., 3:7]))
        stats = trainer._last_phase_residual_stats
        self.assertGreater(stats['phase_peptide_retraction_mean'].item(), 0.0)
        self.assertLessEqual(
            stats['phase_peptide_retraction_max'].item(),
            trainer.config.phase_residual_peptide_retraction_max_translation + 1e-6,
        )
        self.assertGreater(stats['phase_peptide_retraction_active_frac'].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
