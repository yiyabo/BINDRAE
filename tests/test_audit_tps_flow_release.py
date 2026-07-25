import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.audit_tps_flow_release import (
    REQUIRED_REPOSITORY_PATHS,
    audit_files,
    audit_repository,
)


class TpsFlowReleaseAuditTest(unittest.TestCase):
    def test_repository_audit_separates_presence_from_reproducibility_gaps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative in REQUIRED_REPOSITORY_PATHS:
                path = root / relative
                if Path(relative).suffix:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("", encoding="utf-8")
                else:
                    path.mkdir(parents=True, exist_ok=True)
            (root / "tps_inference_adk_fixed.py").write_text(
                "import pyrosetta\nclose_idx=[]\nopen_idx=[]\n",
                encoding="utf-8",
            )

            report = audit_repository(root)
            self.assertTrue(report["required_paths_complete"])
            self.assertFalse(report["environment_lock_present"])
            self.assertFalse(report["adk_split_files_present"])
            self.assertTrue(report["adk_imports_pyrosetta"])
            self.assertTrue(report["adk_has_hardcoded_endpoint_indices"])

    def test_artifact_audit_checks_size_and_checksum(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            content = b"tps-flow-test-artifact"
            (root / "artifact.bin").write_bytes(content)
            expected = {
                "artifact.bin": (len(content), hashlib.md5(content).hexdigest())
            }
            report = audit_files(root, expected, checksum=True)
            self.assertTrue(report["complete"])
            self.assertTrue(report["files"]["artifact.bin"]["md5_ok"])


if __name__ == "__main__":
    unittest.main()
