from __future__ import annotations

import unittest

import numpy as np

from src.data.ligand_residue_selection import (
    COVALENT_TOLERANCE,
    LigandResidueNotFound,
    ResidueView,
    parse_residue_number,
    residues_are_bonded,
    select_ligand_residues,
)


def _residue(
    resname: str,
    resnum: int,
    coords: list[list[float]],
    elements: list[str],
    *,
    chain_id: str = "A",
) -> ResidueView:
    return ResidueView(
        resname=resname,
        resnum=resnum,
        coords=np.array(coords, dtype=np.float64),
        elements=tuple(elements),
        chain_id=chain_id,
    )


def _translated(residue: ResidueView, offset: float, *, resnum: int) -> ResidueView:
    """The same component placed ``offset`` angstroms away along x."""
    return ResidueView(
        resname=residue.resname,
        resnum=resnum,
        coords=residue.coords + np.array([offset, 0.0, 0.0]),
        elements=residue.elements,
        chain_id=residue.chain_id,
    )


#: A three-atom stand-in for an organic component. Bond lengths are irrelevant
#: inside a residue -- selection never re-derives intra-residue bonding.
_LIGAND = _residue(
    "ATP", 501,
    [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
    ["C", "C", "O"],
)


class SelectLigandResiduesTest(unittest.TestCase):
    def test_a_distant_second_copy_is_discarded(self):
        """The production defect: another ATP elsewhere in the chain."""
        residues = [_LIGAND, _translated(_LIGAND, 70.0, resnum=602)]

        selection = select_ligand_residues(residues, resname="ATP", resnum=501)

        self.assertEqual(selection.selected_indices, (0,))
        self.assertEqual(selection.discarded_same_resname, (1,))
        self.assertTrue(selection.defect_present)
        self.assertEqual(selection.selected_atom_count, 3)
        self.assertAlmostEqual(
            selection.discarded_max_centroid_distance_angstrom, 70.0, places=3
        )

    def test_ten_scattered_copies_collapse_to_one(self):
        """`5nhd-C-XYP-506` had ten copies spread 76 A; only the seed survives."""
        residues = [_LIGAND] + [
            _translated(_LIGAND, 8.0 * (index + 1), resnum=600 + index)
            for index in range(9)
        ]

        selection = select_ligand_residues(residues, resname="ATP", resnum=501)

        self.assertEqual(selection.selected_indices, (0,))
        self.assertEqual(len(selection.discarded_same_resname), 9)
        self.assertEqual(selection.selected_atom_count, 3)

    def test_a_covalently_linked_disaccharide_is_kept_whole(self):
        """An oligosaccharide is one chemical entity written one unit per residue.

        Truncating it to the seed residue would be a different corruption of the
        same input, so the closure must cross the glycosidic bond.
        """
        first = _residue("BGC", 1, [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], ["C", "O"])
        # Glycosidic C-O: 1.44 A from the first residue's bridging oxygen.
        second = _residue("BGC", 2, [[2.94, 0.0, 0.0], [4.44, 0.0, 0.0]], ["C", "C"])
        residues = [first, second]

        selection = select_ligand_residues(residues, resname="BGC", resnum=1)

        self.assertEqual(selection.selected_indices, (0, 1))
        self.assertEqual(selection.covalent_partner_indices, (1,))
        self.assertEqual(selection.discarded_same_resname, ())
        self.assertFalse(selection.defect_present)
        self.assertEqual(selection.selected_atom_count, 4)

    def test_the_closure_is_transitive(self):
        """A trisaccharide reaches the third unit through the second."""
        units = [
            _residue("XYP", 1, [[0.0, 0.0, 0.0]], ["O"]),
            _residue("XYP", 2, [[1.44, 0.0, 0.0]], ["C"]),
            _residue("XYP", 3, [[2.88, 0.0, 0.0]], ["O"]),
        ]

        selection = select_ligand_residues(units, resname="XYP", resnum=1)

        self.assertEqual(selection.selected_indices, (0, 1, 2))
        self.assertEqual(selection.discarded_same_resname, ())

    def test_a_linked_chain_still_drops_an_unlinked_copy(self):
        """Closure and discard are not mutually exclusive.

        A genuine disaccharide plus a stray third unit far away: keep two, drop
        one. A single-residue filter would keep one; the old name-only filter
        would keep all three.
        """
        residues = [
            _residue("BGC", 1, [[0.0, 0.0, 0.0]], ["O"]),
            _residue("BGC", 2, [[1.44, 0.0, 0.0]], ["C"]),
            _residue("BGC", 3, [[60.0, 0.0, 0.0]], ["C"]),
        ]

        selection = select_ligand_residues(residues, resname="BGC", resnum=1)

        self.assertEqual(selection.selected_indices, (0, 1))
        self.assertEqual(selection.discarded_same_resname, (2,))
        self.assertTrue(selection.defect_present)

    def test_closure_can_be_disabled_for_callers_that_verify_their_result(self):
        """The closure over-reaches on some structures -- on `3hxy-A-MDN-443` it
        grew a 9-atom diphosphonate into 68 atoms. A caller that can check its
        answer against what was already written tries both and keeps the one
        that reproduces the deposited ligand."""
        residues = [
            _residue("BGC", 1, [[0.0, 0.0, 0.0]], ["O"]),
            _residue("BGC", 2, [[1.44, 0.0, 0.0]], ["C"]),
        ]

        with_closure = select_ligand_residues(residues, resname="BGC", resnum=1)
        seed_only = select_ligand_residues(
            residues, resname="BGC", resnum=1, covalent_closure=False
        )

        self.assertEqual(with_closure.selected_indices, (0, 1))
        self.assertEqual(seed_only.selected_indices, (0,))
        self.assertEqual(seed_only.covalent_partner_indices, ())
        self.assertEqual(seed_only.discarded_same_resname, (1,))

    def test_a_separate_metal_ion_is_not_part_of_the_ligand(self):
        """A site zinc is its own residue and is not ligand scaffold.

        A metal that belongs to the component -- heme iron -- lives inside the
        component's own residue, so excluding lone ions costs nothing there.
        """
        residues = [
            _residue("LIG", 301, [[0.0, 0.0, 0.0]], ["O"]),
            _residue("ZN", 401, [[2.05, 0.0, 0.0]], ["ZN"]),
        ]

        selection = select_ligand_residues(residues, resname="LIG", resnum=301)

        self.assertEqual(selection.selected_indices, (0,))
        self.assertEqual(selection.covalent_partner_indices, ())

    def test_missing_seed_raises_and_never_falls_back_to_name_only(self):
        """The silent fallback is the defect; absence must stay an exception."""
        residues = [_LIGAND, _translated(_LIGAND, 70.0, resnum=602)]

        with self.assertRaises(LigandResidueNotFound):
            select_ligand_residues(residues, resname="ATP", resnum=999)

    def test_seed_ambiguity_is_reported_not_silently_merged(self):
        residues = [
            ResidueView(
                resname="LIG", resnum=1, coords=np.zeros((1, 3)),
                elements=("C",), insertion_code=" ",
            ),
            ResidueView(
                resname="LIG", resnum=1, coords=np.array([[40.0, 0.0, 0.0]]),
                elements=("C",), insertion_code="B",
            ),
        ]

        selection = select_ligand_residues(residues, resname="LIG", resnum=1)

        self.assertTrue(selection.seed_ambiguous)
        self.assertEqual(selection.selected_indices, (0,))

    def test_an_insertion_code_can_be_requested_explicitly(self):
        residues = [
            ResidueView(
                resname="LIG", resnum=1, coords=np.zeros((1, 3)),
                elements=("C",), insertion_code=" ",
            ),
            ResidueView(
                resname="LIG", resnum=1, coords=np.array([[40.0, 0.0, 0.0]]),
                elements=("C",), insertion_code="B",
            ),
        ]

        selection = select_ligand_residues(
            residues, resname="LIG", resnum=1, insertion_code="B"
        )

        self.assertEqual(selection.selected_indices, (1,))
        self.assertFalse(selection.seed_ambiguous)

    def test_an_unknown_element_is_reported(self):
        residues = [
            _residue("LIG", 1, [[0.0, 0.0, 0.0]], ["C"]),
            _residue("XYZ", 2, [[1.4, 0.0, 0.0]], ["Xx"]),
        ]

        selection = select_ligand_residues(residues, resname="LIG", resnum=1)

        self.assertIn("XX", selection.unknown_elements)


class ResiduesAreBondedTest(unittest.TestCase):
    def test_radii_separate_a_disulfide_from_zinc_coordination(self):
        """Both sit near 2.05 A, which is why a flat cutoff cannot work.

        S-S 2.05 A is a covalent bond; Zn-O 2.05 A is coordination. Summed
        covalent radii put them on opposite sides of the ceiling, a single
        distance threshold cannot.
        """
        sulfur_left = _residue("CYX", 1, [[0.0, 0.0, 0.0]], ["S"])
        sulfur_right = _residue("CYX", 2, [[2.05, 0.0, 0.0]], ["S"])
        oxygen = _residue("LIG", 3, [[0.0, 0.0, 0.0]], ["O"])
        zinc = _residue("ZN", 4, [[2.05, 0.0, 0.0]], ["ZN"])

        self.assertTrue(residues_are_bonded(sulfur_left, sulfur_right))
        self.assertFalse(residues_are_bonded(oxygen, zinc))

    def test_a_typical_carbon_carbon_bond_is_bonded(self):
        left = _residue("A", 1, [[0.0, 0.0, 0.0]], ["C"])
        right = _residue("B", 2, [[1.54, 0.0, 0.0]], ["C"])

        self.assertTrue(residues_are_bonded(left, right))

    def test_a_van_der_waals_contact_is_not_bonded(self):
        left = _residue("A", 1, [[0.0, 0.0, 0.0]], ["C"])
        right = _residue("B", 2, [[3.6, 0.0, 0.0]], ["C"])

        self.assertFalse(residues_are_bonded(left, right))

    def test_an_empty_residue_bonds_to_nothing(self):
        empty = ResidueView(
            resname="A", resnum=1, coords=np.zeros((0, 3)), elements=()
        )
        other = _residue("B", 2, [[0.0, 0.0, 0.0]], ["C"])

        self.assertFalse(residues_are_bonded(empty, other))

    def test_tolerance_is_the_documented_default(self):
        self.assertAlmostEqual(COVALENT_TOLERANCE, 1.15)


class ParseResidueNumberTest(unittest.TestCase):
    def test_a_standard_entry_key_yields_the_residue_number(self):
        self.assertEqual(parse_residue_number("2hdr-A-4A3-506"), 506)

    def test_a_numeric_component_name_does_not_confuse_the_parse(self):
        self.assertEqual(parse_residue_number("2qje-D-Z8T-2"), 2)

    def test_a_non_integer_tail_returns_none_rather_than_guessing(self):
        self.assertIsNone(parse_residue_number("entry_17"))
        self.assertIsNone(parse_residue_number("1abc-A-LIG"))


class ResidueViewValidationTest(unittest.TestCase):
    def test_mismatched_element_count_is_rejected(self):
        with self.assertRaises(ValueError):
            ResidueView(
                resname="LIG", resnum=1,
                coords=np.zeros((2, 3)), elements=("C",),
            )

    def test_wrong_coordinate_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            ResidueView(
                resname="LIG", resnum=1,
                coords=np.zeros((2, 2)), elements=("C", "O"),
            )


if __name__ == "__main__":
    unittest.main()
