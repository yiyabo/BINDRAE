import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.residue_identity import (
    RESIDUE_ALIGNMENT_VERSION,
    residue_keys_to_array,
)
from src.stage2.datasets.backbone import (
    _load_torsions,
    align_by_residue_ids,
)


class Stage2ResidueAlignmentTest(unittest.TestCase):
    def test_backbone_alignment_uses_target_axis(self):
        target = [("A", 10, ""), ("A", 11, ""), ("A", 11, "A"), ("A", 12, "")]
        apo_keys = target
        holo_keys = [("B", 11, ""), ("B", 11, "A"), ("B", 12, "")]

        apo_base = np.arange(12, dtype=np.float32).reshape(4, 3)
        holo_base = np.asarray([[11, 0, 0], [12, 0, 0], [13, 0, 0]], dtype=np.float32)
        apo, holo, mask = align_by_residue_ids(
            (apo_base, apo_base + 100, apo_base + 200),
            apo_keys,
            (holo_base, holo_base + 100, holo_base + 200),
            holo_keys,
            target,
        )

        np.testing.assert_array_equal(mask, [False, True, True, True])
        np.testing.assert_array_equal(holo[0][1:], holo_base)
        np.testing.assert_array_equal(apo[0], apo_base)

    def test_torsions_are_scattered_and_masked_by_key(self):
        target = [("A", 10, ""), ("A", 11, ""), ("A", 11, "A"), ("A", 12, "")]
        source = [("B", 11, ""), ("B", 12, "")]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "torsion_holo.npz"
            np.savez_compressed(
                path,
                phi=np.asarray([1.0, 2.0], dtype=np.float32),
                psi=np.asarray([3.0, 4.0], dtype=np.float32),
                omega=np.asarray([5.0, 6.0], dtype=np.float32),
                chi=np.asarray([[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0]], dtype=np.float32),
                chi_mask=np.ones((2, 4), dtype=np.bool_),
                bb_mask=np.ones(2, dtype=np.bool_),
                residue_keys=residue_keys_to_array(source),
                residue_alignment_version=np.asarray(RESIDUE_ALIGNMENT_VERSION),
            )

            torsions = _load_torsions(path, target)

        np.testing.assert_array_equal(
            torsions["residue_present"],
            np.asarray([False, True, False, True]),
        )
        self.assertEqual(torsions["angles"][1, 0], 1.0)
        self.assertEqual(torsions["angles"][3, 6], 14.0)
        self.assertFalse(torsions["chi_mask"][0].any())
        self.assertTrue(torsions["chi_mask"][1].all())

    def test_legacy_torsion_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "legacy.npz"
            np.savez_compressed(
                path,
                phi=np.zeros(1),
                psi=np.zeros(1),
                omega=np.zeros(1),
                chi=np.zeros((1, 4)),
                chi_mask=np.ones((1, 4), dtype=np.bool_),
            )
            with self.assertRaisesRegex(ValueError, "no canonical residue_keys"):
                _load_torsions(path, [("A", 1, "")])


if __name__ == "__main__":
    unittest.main()
