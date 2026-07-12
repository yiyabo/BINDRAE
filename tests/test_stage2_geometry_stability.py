import unittest

import numpy as np
import torch

from src.stage2.datasets.dataset_stage2 import _build_peptide_bond_mask
from src.stage2.modules.geometry import compute_peptide_loss


class PeptideConnectivityTest(unittest.TestCase):
    def test_mask_excludes_chain_boundaries_and_coordinate_gaps(self):
        residue_keys = [("A", 1, ""), ("A", 2, ""), ("B", 1, ""), ("B", 2, "")]
        node_mask = np.ones(4, dtype=bool)
        n_apo = np.zeros((4, 3), dtype=np.float32)
        c_apo = np.zeros((4, 3), dtype=np.float32)
        n_holo = np.zeros((4, 3), dtype=np.float32)
        c_holo = np.zeros((4, 3), dtype=np.float32)
        c_apo[0, 0] = c_holo[0, 0] = 1.33
        c_apo[1, 0] = c_holo[1, 0] = 1.33
        c_apo[2, 0] = c_holo[2, 0] = 5.0

        mask = _build_peptide_bond_mask(
            residue_keys,
            node_mask,
            n_apo,
            c_apo,
            n_holo,
            c_holo,
        )
        np.testing.assert_array_equal(mask, np.asarray([True, False, False]))

    def test_loss_ignores_masked_large_gap(self):
        atom14_pos = torch.zeros(1, 2, 14, 3)
        atom14_mask = torch.zeros(1, 2, 14, dtype=torch.bool)
        atom14_mask[:, :, :3] = True
        node_mask = torch.ones(1, 2, dtype=torch.bool)
        atom14_pos[0, 0, 2, 0] = 100.0

        masked_loss = compute_peptide_loss(
            atom14_pos,
            atom14_mask,
            node_mask,
            peptide_bond_mask=torch.zeros(1, 1, dtype=torch.bool),
        )
        unmasked_loss = compute_peptide_loss(atom14_pos, atom14_mask, node_mask)

        self.assertEqual(masked_loss.item(), 0.0)
        self.assertGreater(unmasked_loss.item(), 1000.0)


if __name__ == "__main__":
    unittest.main()
