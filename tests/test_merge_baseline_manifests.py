import json
from pathlib import Path

from scripts.merge_baseline_manifests import read_jsonl


def test_read_jsonl_ignores_blank_lines(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    path.write_text('\n{"sample_id": "a"}\n\n{"sample_id": "b"}\n')
    assert [row["sample_id"] for row in read_jsonl(path)] == ["a", "b"]
