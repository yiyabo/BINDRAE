#!/usr/bin/env python3
"""Audit a local TPS-Flow checkout and its official Zenodo artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional


UPSTREAM_REPOSITORY = "https://github.com/lfs119/TPS-Flow"
UPSTREAM_COMMIT = "fa94ad66bdf93e4bf0dbc0214aa77ec87edb1901"
WEIGHTS_RECORD = "https://doi.org/10.5281/zenodo.17731628"
DATA_RECORD = "https://doi.org/10.5281/zenodo.17555901"

WEIGHT_FILES = {
    "1hpv.ckpt": (283_319_089, "73e2218c3bde3a4b9b64d1bf135abffb"),
    "1brs.ckpt": (283_469_553, "58825e09293121bcd9730c943055ed8d"),
    "recon_energy.ckpt": (287_982_297, "74901dba1d8cb9566c9b5f2d16165eac"),
    "base.ckpt": (284_288_241, "0c0ba05ecadb1fa0f9ca34359ab47efc"),
    "adk.ckpt": (283_495_665, "3301721f72032cb287530c5697f5537f"),
}

DATA_FILES = {
    "md1.nc": (1_624_640_816, "511a2738c7d28c8d570479df0e207f6b"),
    "md_bin_10.rar": (8_003_796, "2294f94698c44361aedf2deb5fa103e4"),
    "md_bin_0414.rar": (299_251_693, "1135e1382645bc116f06061744e0fb78"),
}

REQUIRED_REPOSITORY_PATHS = (
    "README.md",
    "train.py",
    "tps_inference_apo.py",
    "tps_inference_1brs_fixed.py",
    "tps_inference_1hpv_fixed.py",
    "tps_inference_adk_fixed.py",
    "tps_flow/model",
    "tps_flow/transport",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_dir", default="reference/TPS-Flow")
    parser.add_argument("--weights_dir", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip_checksum", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_dir: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def audit_files(
    root: Optional[Path],
    expected: Dict[str, tuple[int, str]],
    *,
    checksum: bool,
) -> Dict[str, object]:
    if root is None:
        return {"configured": False, "complete": False, "files": {}}
    rows: Dict[str, object] = {}
    complete = True
    for name, (expected_size, expected_md5) in expected.items():
        path = root / name
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        size_ok = size == expected_size
        actual_md5 = md5sum(path) if exists and size_ok and checksum else None
        md5_ok = actual_md5 == expected_md5 if actual_md5 is not None else None
        valid = bool(exists and size_ok and (md5_ok is not False))
        complete = complete and valid
        rows[name] = {
            "path": str(path),
            "exists": exists,
            "size": size,
            "expected_size": expected_size,
            "size_ok": size_ok,
            "md5": actual_md5,
            "expected_md5": expected_md5,
            "md5_ok": md5_ok,
            "valid": valid,
        }
    return {"configured": True, "complete": complete, "files": rows}


def contains_text(path: Path, text: str) -> bool:
    return path.is_file() and text in path.read_text(errors="ignore")


def audit_repository(repo_dir: Path) -> Dict[str, object]:
    required = {
        name: (repo_dir / name).exists() for name in REQUIRED_REPOSITORY_PATHS
    }
    commit = git_commit(repo_dir) if repo_dir.is_dir() else None
    adk_script = repo_dir / "tps_inference_adk_fixed.py"
    return {
        "path": str(repo_dir),
        "exists": repo_dir.is_dir(),
        "required_paths": required,
        "required_paths_complete": all(required.values()),
        "commit": commit,
        "expected_commit": UPSTREAM_COMMIT,
        "commit_matches": commit == UPSTREAM_COMMIT if commit else None,
        "environment_lock_present": any(
            (repo_dir / name).is_file()
            for name in ("requirements.txt", "environment.yml", "pyproject.toml")
        ),
        "license_file_present": any(
            (repo_dir / name).is_file()
            for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")
        ),
        "adk_split_files_present": all(
            (repo_dir / "splits" / name).is_file()
            for name in ("adk_train.csv", "adk_val.csv")
        ),
        "adk_imports_pyrosetta": contains_text(adk_script, "pyrosetta"),
        "adk_has_hardcoded_endpoint_indices": contains_text(
            adk_script, "close_idx"
        ) and contains_text(adk_script, "open_idx"),
    }


def failed_checks(report: Dict[str, object]) -> Iterable[str]:
    repository = report["repository"]
    if not repository["required_paths_complete"]:
        yield "repository_required_paths"
    if repository["commit_matches"] is False:
        yield "repository_commit"
    for key in ("weights", "data"):
        section = report[key]
        if section["configured"] and not section["complete"]:
            yield key


def main() -> None:
    args = parse_args()
    report: Dict[str, object] = {
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "weights_record": WEIGHTS_RECORD,
        "data_record": DATA_RECORD,
        "repository": audit_repository(Path(args.repo_dir)),
        "weights": audit_files(
            Path(args.weights_dir) if args.weights_dir else None,
            WEIGHT_FILES,
            checksum=not args.skip_checksum,
        ),
        "data": audit_files(
            Path(args.data_dir) if args.data_dir else None,
            DATA_FILES,
            checksum=not args.skip_checksum,
        ),
    }
    failures = list(failed_checks(report))
    report["failed_checks"] = failures
    report["passed"] = not failures
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.strict and failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
