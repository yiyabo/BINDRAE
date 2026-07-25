import unittest
from types import SimpleNamespace

import numpy as np
import torch

from scripts.export_cached_phase_path_candidate import (
    _apply_cached_translation_residual,
)
from scripts.export_stage2_path_candidates import (
    _assert_endpoint_contract,
    path_product_state_arrays,
)
from scripts.optimize_path4_openmm_gate0 import _translated_product_state
from flash_ipa.rigid import Rigid, Rotation


class Stage2PathCandidateExportTest(unittest.TestCase):
    def test_serializes_exact_rigid_and_chi_product_state(self):
        rotations = []
        translations = []
        rigids = []
        chi = []
        for frame in range(3):
            rotation = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
            translation = torch.full((1, 2, 3), float(frame))
            angle = torch.full((1, 2, 4), 0.1 * frame)
            rotations.append(rotation[0].numpy())
            translations.append(translation[0].numpy())
            rigids.append(Rigid(rots=Rotation(rot_mats=rotation), trans=translation))
            chi.append(angle)

        observed_rotation, observed_translation, observed_chi = (
            path_product_state_arrays(rigids, chi, 2)
        )

        np.testing.assert_array_equal(observed_rotation, np.stack(rotations))
        np.testing.assert_array_equal(observed_translation, np.stack(translations))
        np.testing.assert_allclose(observed_chi[2], 0.2)

    def test_optimizer_translation_updates_frame_state_and_preserves_endpoints(self):
        translation = np.zeros((3, 2, 3), dtype=np.float32)
        correction = np.zeros_like(translation)
        correction[1, :, 0] = [0.25, -0.5]
        candidate = SimpleNamespace(
            has_product_state=True,
            rigid_translation_angstrom=translation,
        )

        updated = _translated_product_state(candidate, correction)

        np.testing.assert_array_equal(updated[[0, -1]], translation[[0, -1]])
        np.testing.assert_array_equal(updated[1], correction[1])

    def test_cached_translation_residual_composes_in_the_body_frame(self):
        rotation = torch.tensor(
            [[[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]]
        )
        base = Rigid(
            rots=Rotation(rot_mats=rotation),
            trans=torch.tensor([[[10.0, 20.0, 30.0]]]),
        )

        corrected = _apply_cached_translation_residual(
            base, torch.tensor([[[1.0, 0.0, 0.0]]])
        )

        torch.testing.assert_close(
            corrected.get_trans(), torch.tensor([[[10.0, 21.0, 30.0]]])
        )

    @staticmethod
    def _batch():
        n = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
        ca = torch.tensor([[[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
        c = torch.tensor([[[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
        return SimpleNamespace(
            N_apo=n,
            Ca_apo=ca,
            C_apo=c,
            N_holo=n + 1.0,
            Ca_holo=ca + 1.0,
            C_holo=c + 1.0,
            node_mask=torch.ones((1, 2), dtype=torch.bool),
            bb_mask=torch.ones((1, 2, 3), dtype=torch.bool),
        )

    @staticmethod
    def _positions(batch):
        positions = np.zeros((2, 2, 14, 3), dtype=np.float32)
        positions[0, :, 0] = batch.N_apo[0].numpy()
        positions[0, :, 1] = batch.Ca_apo[0].numpy()
        positions[0, :, 2] = batch.C_apo[0].numpy()
        positions[1, :, 0] = batch.N_holo[0].numpy()
        positions[1, :, 1] = batch.Ca_holo[0].numpy()
        positions[1, :, 2] = batch.C_holo[0].numpy()
        return positions

    def test_allows_bounded_fk_idealization_but_keeps_ca_exact(self):
        batch = self._batch()
        positions = self._positions(batch)
        positions[:, :, 0, 0] += 0.18

        ca_error, representation_error = _assert_endpoint_contract(
            positions, batch, 2
        )

        self.assertEqual(ca_error, 0.0)
        self.assertAlmostEqual(representation_error, 0.18, places=5)

    def test_rejects_frame_translation_drift(self):
        batch = self._batch()
        positions = self._positions(batch)
        positions[0, 0, 1, 0] += 0.01

        with self.assertRaisesRegex(ValueError, "frame-translation"):
            _assert_endpoint_contract(positions, batch, 2)

    def test_rejects_large_fk_representation_mismatch(self):
        batch = self._batch()
        positions = self._positions(batch)
        positions[1, 1, 2, 0] += 0.75

        with self.assertRaisesRegex(ValueError, "FK representation"):
            _assert_endpoint_contract(positions, batch, 2)


if __name__ == "__main__":
    unittest.main()
