from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_apobind_holdout_leakage.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_apobind_holdout_leakage", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _entry(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "record": {"transition_id": f"apobind:{sample_id}:pilot"},
    }


def _metadata(
    sample_id: str,
    *,
    sequence: str,
    scaffold: str,
    endpoints: list[str],
    error: str | None = None,
) -> dict:
    return {
        "sample_id": sample_id,
        "sequence": sequence,
        "sequence_sha256": f"sha:{sample_id}",
        "scaffold": scaffold,
        "endpoint_pdb_ids": endpoints,
        "error": error,
    }


class AuditApobindHoldoutLeakageTest(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("Bio"),
        "Biopython not installed",
    )
    def test_filter_excludes_family_scaffold_endpoint_and_invalid_metadata(self):
        holdout = {
            "holdout": _metadata(
                "holdout",
                sequence="ACDEFGHIKLMNPQRSTVWY" * 5,
                scaffold="c1ccccc1",
                endpoints=["1AAA", "2BBB"],
            )
        }
        candidates = {
            "family": _metadata(
                "family",
                sequence="ACDEFGHIKLMNPQRSTVWY" * 5,
                scaffold="C1CCCCC1",
                endpoints=["3CCC", "4DDD"],
            ),
            "scaffold": _metadata(
                "scaffold",
                sequence="Y" * 100,
                scaffold="c1ccccc1",
                endpoints=["5EEE", "6FFF"],
            ),
            "endpoint": _metadata(
                "endpoint",
                sequence="W" * 100,
                scaffold="C1CCCC1",
                endpoints=["1AAA", "7GGG"],
            ),
            "invalid": _metadata(
                "invalid",
                sequence="V" * 100,
                scaffold="C1CCC1",
                endpoints=["8HHH", "9III"],
                error="ValueError: broken metadata",
            ),
            "clean": _metadata(
                "clean",
                sequence="M" * 100,
                scaffold="C1CC1",
                endpoints=["3JJJ", "4KKK"],
            ),
        }
        entries = [_entry(sample_id) for sample_id in candidates]

        retained, excluded = MODULE.filter_candidates(
            entries,
            candidates,
            holdout,
            identity_threshold=0.30,
            coverage_threshold=0.80,
            workers=1,
        )

        self.assertEqual([row["sample_id"] for row in retained], ["clean"])
        reasons = {row["sample_id"]: row["reasons"] for row in excluded}
        self.assertEqual(reasons["family"], ["holdout_protein_family"])
        self.assertEqual(reasons["scaffold"], ["holdout_ligand_scaffold"])
        self.assertEqual(reasons["endpoint"], ["holdout_endpoint_pdb"])
        self.assertEqual(reasons["invalid"], ["invalid_candidate_metadata"])

    def test_invalid_holdout_metadata_is_a_hard_failure(self):
        holdout = {
            "holdout": _metadata(
                "holdout",
                sequence="A" * 100,
                scaffold="c1ccccc1",
                endpoints=["1AAA", "2BBB"],
                error="FileNotFoundError: missing torsion",
            )
        }
        candidates = {
            "candidate": _metadata(
                "candidate",
                sequence="M" * 100,
                scaffold="C1CC1",
                endpoints=["3CCC", "4DDD"],
            )
        }

        with self.assertRaisesRegex(ValueError, "Holdout metadata is invalid"):
            MODULE.filter_candidates(
                [_entry("candidate")],
                candidates,
                holdout,
                identity_threshold=0.30,
                coverage_threshold=0.80,
                workers=1,
            )


if __name__ == "__main__":
    unittest.main()
