import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.evaluate_path4_candidate_md_reference import (
    aggregate_records,
    candidate_state_tensors,
    validate_md_reference,
)


class Path4CandidateMdReferenceTest(unittest.TestCase):
    def test_candidate_state_requires_and_reconstructs_v3_product_state(self):
        missing = SimpleNamespace(has_product_state=False)
        with self.assertRaisesRegex(ValueError, "v3 candidate"):
            candidate_state_tensors(missing, torch.device("cpu"))

        candidate = SimpleNamespace(
            has_product_state=True,
            n_frames=3,
            rigid_rotation_matrix=np.broadcast_to(
                np.eye(3, dtype=np.float32), (3, 2, 3, 3)
            ).copy(),
            rigid_translation_angstrom=np.arange(18, dtype=np.float32).reshape(
                3, 2, 3
            ),
            chi_radians=np.zeros((3, 2, 4), dtype=np.float32),
        )

        rigids, chi = candidate_state_tensors(candidate, torch.device("cpu"))

        self.assertEqual(len(rigids), 3)
        torch.testing.assert_close(
            rigids[1].get_trans(),
            torch.as_tensor(candidate.rigid_translation_angstrom[1]).unsqueeze(0),
        )
        self.assertEqual(tuple(chi[0].shape), (1, 2, 4))

    def test_md_reference_identity_contract_and_metric_aggregate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample__silver_r0.npz"
            np.savez_compressed(
                path,
                schema_version=np.array("md_phase_normal_v1"),
                phase_target_mode=np.array("inferred"),
                sample_id=np.array("sample__silver_r0"),
                n_residues=np.array(2),
                residue_identity_hash=np.array("hash"),
            )
            batch = SimpleNamespace(
                n_residues=[2], residue_identity_hashes=["hash"]
            )
            with np.load(path, allow_pickle=False) as data:
                validate_md_reference(data, path, batch, "sample")

        records = [
            {
                "metrics": {
                    "md_path_product_rmse": 1.0,
                    "md_path_translation_mae_a": 2.0,
                    "md_path_rotation_mae_rad": 3.0,
                    "md_path_chi_mae_rad": 4.0,
                }
            },
            {
                "metrics": {
                    "md_path_product_rmse": 3.0,
                    "md_path_translation_mae_a": 4.0,
                    "md_path_rotation_mae_rad": 5.0,
                    "md_path_chi_mae_rad": 6.0,
                }
            },
        ]
        aggregate = aggregate_records(records)
        self.assertEqual(aggregate["system_macro"]["md_path_product_rmse"], 2.0)


if __name__ == "__main__":
    unittest.main()
