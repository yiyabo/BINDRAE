from __future__ import annotations

import unittest

import numpy as np

from src.data.ligand_copy_identification import (
    CopyIdentificationError,
    identify_ligand_atom_indices,
    kabsch_transform,
    match_copy_by_internal_geometry,
    sorted_internal_distances,
)


def _rigid(coords: np.ndarray, *, angle: float, shift: np.ndarray) -> np.ndarray:
    """Rotate about z and translate. Preserves every internal distance."""
    cos, sin = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return coords @ rotation.T + shift


#: An asymmetric five-atom body. Asymmetry matters: a symmetric body has
#: degenerate atom signatures and the correspondence step would be arbitrary.
_BODY = np.array(
    [
        [0.00, 0.00, 0.00],
        [1.52, 0.00, 0.00],
        [2.31, 1.28, 0.10],
        [1.05, 2.40, -0.35],
        [-0.42, 1.90, 0.62],
    ]
)


class SortedInternalDistancesTest(unittest.TestCase):
    def test_the_vector_is_invariant_under_a_rigid_transform(self):
        moved = _rigid(_BODY, angle=1.1, shift=np.array([120.0, -8.0, 33.0]))

        np.testing.assert_allclose(
            sorted_internal_distances(_BODY),
            sorted_internal_distances(moved),
            atol=1e-12,
        )

    def test_the_vector_is_invariant_under_atom_reordering(self):
        shuffled = _BODY[[3, 0, 4, 1, 2]]

        np.testing.assert_allclose(
            sorted_internal_distances(_BODY),
            sorted_internal_distances(shuffled),
            atol=1e-12,
        )

    def test_a_single_atom_has_no_internal_distances(self):
        self.assertEqual(sorted_internal_distances(np.zeros((1, 3))).size, 0)


class MatchCopyTest(unittest.TestCase):
    def test_the_transformed_copy_is_found_among_decoys(self):
        true_copy = _rigid(_BODY, angle=0.7, shift=np.array([40.0, 5.0, -2.0]))
        # Decoys are the same component in a different conformation, which is
        # what the real corpus contains -- not a different molecule.
        decoy_a = _rigid(_BODY * 1.02, angle=0.2, shift=np.array([-30.0, 0.0, 0.0]))
        # Perturb one atom, not all of them: adding a constant vector is a
        # translation, which leaves every internal distance untouched and would
        # make the decoy genuinely indistinguishable from the true copy.
        conformer = _BODY.copy()
        conformer[4] += np.array([0.0, 0.15, 0.0])
        decoy_b = _rigid(conformer, angle=2.0, shift=np.array([70.0, 0.0, 0.0]))

        match = match_copy_by_internal_geometry(_BODY, [decoy_a, true_copy, decoy_b])

        self.assertEqual(match.candidate_index, 1)
        self.assertLess(match.deviation_angstrom, 1e-9)
        self.assertEqual(match.candidate_count, 3)

    def test_storage_precision_noise_still_matches(self):
        """SDF keeps four decimals and the npy is float32; both show ~1e-4 A."""
        quantized = np.round(
            _rigid(_BODY, angle=0.4, shift=np.array([12.0, 0.0, 0.0])), 4
        ).astype(np.float32).astype(np.float64)

        match = match_copy_by_internal_geometry(_BODY, [quantized])

        self.assertEqual(match.candidate_index, 0)
        self.assertLess(match.deviation_angstrom, 1e-3)

    def test_a_small_absolute_margin_is_accepted_when_the_ratio_is_large(self):
        """The measured corpus case: best 1e-4 A, runner-up 0.028 A.

        An absolute-gap threshold was tried first and rejected this, which was a
        false alarm -- 0.028 A is 280x worse than 1e-4 A.
        """
        true_copy = np.round(_BODY, 4)
        near_decoy = _BODY + np.array([[0.0, 0.0, 0.0]] * 4 + [[0.0, 0.02, 0.0]])

        match = match_copy_by_internal_geometry(_BODY, [near_decoy, true_copy])

        self.assertEqual(match.candidate_index, 1)
        self.assertGreater(match.runner_up_ratio, 10.0)

    def test_no_matching_copy_raises_rather_than_returning_the_closest(self):
        with self.assertRaises(CopyIdentificationError) as caught:
            match_copy_by_internal_geometry(_BODY, [_BODY * 1.3])

        self.assertEqual(caught.exception.reason, "no_copy_matched_reference_geometry")

    def test_two_identical_copies_are_refused_as_ambiguous(self):
        """Genuinely duplicated conformers cannot be told apart; refuse."""
        copy_a = _rigid(_BODY, angle=0.3, shift=np.array([10.0, 0.0, 0.0]))
        copy_b = _rigid(_BODY, angle=1.9, shift=np.array([-60.0, 0.0, 0.0]))

        with self.assertRaises(CopyIdentificationError) as caught:
            match_copy_by_internal_geometry(_BODY, [copy_a, copy_b])

        self.assertEqual(caught.exception.reason, "copy_match_ambiguous")

    def test_a_different_atom_count_scores_as_infinite_not_skipped(self):
        truncated = _BODY[:3]
        true_copy = np.round(_BODY, 4)

        match = match_copy_by_internal_geometry(_BODY, [truncated, true_copy])

        self.assertEqual(match.candidate_index, 1)
        self.assertFalse(np.isfinite(match.runner_up_deviation_angstrom))

    def test_no_candidates_raises(self):
        with self.assertRaises(CopyIdentificationError) as caught:
            match_copy_by_internal_geometry(_BODY, [])

        self.assertEqual(caught.exception.reason, "no_candidates")


class KabschTest(unittest.TestCase):
    def test_a_rigid_transform_is_recovered_exactly(self):
        shift = np.array([13.0, -4.0, 7.5])
        target = _rigid(_BODY, angle=0.9, shift=shift)

        rotation, translation, rmsd = kabsch_transform(_BODY, target)

        self.assertLess(rmsd, 1e-10)
        np.testing.assert_allclose(_BODY @ rotation.T + translation, target, atol=1e-9)

    def test_a_reflection_is_not_introduced(self):
        """A proper rotation has determinant +1; a mirrored fit would be wrong."""
        target = _rigid(_BODY, angle=2.4, shift=np.zeros(3))

        rotation, _, _ = kabsch_transform(_BODY, target)

        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=9)

    def test_an_empty_atom_set_is_rejected(self):
        with self.assertRaises(ValueError):
            kabsch_transform(np.zeros((0, 3)), np.zeros((0, 3)))


class IdentifyLigandAtomIndicesTest(unittest.TestCase):
    def _sdf_with_copies(self, n_copies: int, true_slot: int):
        """Build an observed atom array holding ``n_copies`` of the body."""
        blocks, groups, cursor = [], [], 0
        for slot in range(n_copies):
            if slot == true_slot:
                coords = _rigid(_BODY, angle=0.55, shift=np.array([0.0, 0.0, 0.0]))
            else:
                # Same component, different conformation and position.
                coords = _rigid(
                    _BODY + np.array([[0.0, 0.0, 0.0]] * 4 + [[0.0, 0.03 * (slot + 1), 0.0]]),
                    angle=0.3 * slot,
                    shift=np.array([25.0 * (slot + 1), 0.0, 0.0]),
                )
            blocks.append(coords)
            groups.append(tuple(range(cursor, cursor + len(coords))))
            cursor += len(coords)
        return np.vstack(blocks), groups

    def test_the_true_copy_atoms_are_selected_from_fifteen(self):
        """`2hdr-A-4A3-506` held 15 copies of an 11-atom component."""
        observed, groups = self._sdf_with_copies(15, true_slot=9)

        selection = identify_ligand_atom_indices([_BODY], observed, groups)

        self.assertEqual(selection.atom_indices, groups[9])
        self.assertEqual(len(selection.atom_indices), 5)
        self.assertEqual(selection.observed_atom_count, 75)
        self.assertEqual(selection.reference_atom_count, 5)
        self.assertLess(selection.fit_rmsd_angstrom, 1e-9)

    def test_a_multi_residue_ligand_is_selected_whole(self):
        """An oligosaccharide's second unit is found via the recovered transform.

        This is why identification does not stop at the matched fragment: the
        remaining residues may share a fragment with it, or sit in their own.
        """
        unit_two = _BODY + np.array([1.44, 6.0, 0.0])
        transform = dict(angle=0.8, shift=np.array([15.0, -3.0, 2.0]))
        ligand = np.vstack([_rigid(_BODY, **transform), _rigid(unit_two, **transform)])
        decoy = _rigid(_BODY * 1.05, angle=0.1, shift=np.array([-40.0, 0.0, 0.0]))

        observed = np.vstack([decoy, ligand])
        groups = [tuple(range(0, 5)), tuple(range(5, 15))]

        selection = identify_ligand_atom_indices([_BODY, unit_two], observed, groups)

        self.assertEqual(selection.atom_indices, tuple(range(5, 15)))
        self.assertEqual(selection.reference_atom_count, 10)

    def test_a_reference_atom_with_no_counterpart_is_refused(self):
        observed, groups = self._sdf_with_copies(3, true_slot=1)
        orphan = np.vstack([_BODY, np.array([[500.0, 500.0, 500.0]])])

        with self.assertRaises(CopyIdentificationError) as caught:
            identify_ligand_atom_indices([_BODY, orphan[-1:]], observed, groups)

        self.assertEqual(caught.exception.reason, "reference_atom_unpaired")

    def test_an_empty_reference_is_refused(self):
        observed, groups = self._sdf_with_copies(2, true_slot=0)

        with self.assertRaises(CopyIdentificationError) as caught:
            identify_ligand_atom_indices([], observed, groups)

        self.assertEqual(caught.exception.reason, "no_reference")

    def test_diagnostics_report_the_margin(self):
        observed, groups = self._sdf_with_copies(4, true_slot=2)

        diagnostics = identify_ligand_atom_indices(
            [_BODY], observed, groups
        ).as_diagnostics()

        self.assertEqual(diagnostics["matched_candidate"], 2)
        self.assertEqual(diagnostics["candidate_count"], 4)
        self.assertEqual(diagnostics["selected_atoms"], 5)
        self.assertIn("runner_up_ratio", diagnostics)


if __name__ == "__main__":
    unittest.main()
