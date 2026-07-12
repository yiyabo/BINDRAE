import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.stage2.training.trainer import Stage2Trainer
from scripts.build_stage2_phase_teacher_subset import summarize_cache_file


class PhaseTeacherCacheTest(unittest.TestCase):
    def test_loads_interpolated_contact_event_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "sample_1.npz"
            np.savez_compressed(
                cache_path,
                schema_version=np.array("phase_teacher_v1"),
                sample_id=np.array("sample/1"),
                n_residues=np.array(2, dtype=np.int32),
                t_values=np.array([0.0, 0.5, 1.0], dtype=np.float32),
                tau_target=np.array(
                    [[0.0, 0.0], [0.2, 0.7], [1.0, 1.0]],
                    dtype=np.float32,
                ),
                phase_confidence=np.array(
                    [[0.0, 0.0], [0.8, 0.8], [0.0, 0.0]],
                    dtype=np.float32,
                ),
                node_mask=np.array([True, True]),
                pocket_mask=np.array([True, True]),
                active_mask=np.array([True, True]),
                approach_mask=np.array([True, False]),
                formed_contact_mask=np.array([False, False]),
                release_mask=np.array([False, True]),
            )

            trainer = Stage2Trainer.__new__(Stage2Trainer)
            trainer.device = torch.device("cpu")
            trainer.config = SimpleNamespace(
                phase_teacher_cache_dir=tmpdir,
                w_phase_teacher=1.0,
                phase_teacher_mask_mode="contact_event",
                phase_teacher_min_confidence=0.05,
                phase_teacher_missing_policy="error",
            )
            batch = SimpleNamespace(
                pdb_ids=["sample/1"],
                n_residues=torch.tensor([2]),
                node_mask=torch.tensor([[True, True]]),
            )

            targets = trainer._load_phase_teacher_targets(
                batch,
                [0.0, 0.25, 0.5, 0.75, 1.0],
            )

            self.assertIsNotNone(targets)
            torch.testing.assert_close(
                targets["tau"][:, 0, 0],
                torch.tensor([0.0, 0.1, 0.2, 0.6, 1.0]),
            )
            self.assertEqual(int(targets["mask"][:, 0, 0].sum()), 3)
            self.assertEqual(int(targets["mask"][:, 0, 1].sum()), 0)
            self.assertAlmostEqual(float(targets["source_grid_error"][0]), 0.25)

            summary = summarize_cache_file(
                cache_path,
                mask_mode="contact_event",
                min_confidence=0.05,
                n_model_steps=4,
            )
            self.assertEqual(summary["sample_id"], "sample/1")
            self.assertEqual(summary["selected_residues"], 1)
            self.assertGreater(summary["supervised_points"], 0)


if __name__ == "__main__":
    unittest.main()
