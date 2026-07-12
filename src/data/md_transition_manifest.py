"""Canonical manifest validation for trajectory-supervised BINDRAE data."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse


SCHEMA_VERSION = "bindrae_md_transition_v1"

EVIDENCE_TIERS = frozenset(
    {
        "gold_atomistic_transition",
        "silver_enhanced_sampling",
        "bronze_modeled_path",
        "context_equilibrium",
    }
)
RECORD_STATUSES = frozenset(
    {"candidate", "metadata_verified", "downloaded", "prepared", "rejected"}
)
SPLIT_NAMES = frozenset({"unassigned", "train", "val", "test"})


@dataclass(frozen=True)
class ManifestIssue:
    """One actionable manifest validation result."""

    severity: str
    code: str
    message: str
    line_number: Optional[int] = None
    transition_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "line_number": self.line_number,
            "transition_id": self.transition_id,
        }


def _mapping(record: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = record.get(key)
    return value if isinstance(value, Mapping) else {}


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_remote_path(value: str) -> bool:
    return urlparse(value).scheme in {"http", "https", "s3", "gs"}


def _resolve_local_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def _add_issue(
    issues: List[ManifestIssue],
    severity: str,
    code: str,
    message: str,
    *,
    line_number: Optional[int],
    transition_id: Optional[str],
) -> None:
    issues.append(
        ManifestIssue(
            severity=severity,
            code=code,
            message=message,
            line_number=line_number,
            transition_id=transition_id,
        )
    )


def load_transition_manifest(
    path: str | Path,
) -> Tuple[List[Dict[str, Any]], List[ManifestIssue]]:
    """Read JSONL records while retaining parse failures as audit issues."""

    manifest_path = Path(path)
    records: List[Dict[str, Any]] = []
    issues: List[ManifestIssue] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                _add_issue(
                    issues,
                    "error",
                    "invalid_json",
                    f"invalid JSON: {exc.msg}",
                    line_number=line_number,
                    transition_id=None,
                )
                continue
            if not isinstance(value, dict):
                _add_issue(
                    issues,
                    "error",
                    "record_not_object",
                    "each JSONL record must be an object",
                    line_number=line_number,
                    transition_id=None,
                )
                continue
            value["_manifest_line_number"] = line_number
            records.append(value)
    return records, issues


def validate_transition_record(
    record: Mapping[str, Any],
    *,
    line_number: Optional[int] = None,
    base_dir: str | Path | None = None,
    check_files: bool = False,
) -> List[ManifestIssue]:
    """Validate one source-neutral transition record.

    Candidate records may omit local files. Records marked ``prepared`` must
    provide enough endpoint and trajectory material for preprocessing.
    """

    issues: List[ManifestIssue] = []
    line_number = line_number or record.get("_manifest_line_number")
    transition_id_value = record.get("transition_id")
    transition_id = (
        str(transition_id_value).strip()
        if _nonempty_string(transition_id_value)
        else None
    )

    def issue(severity: str, code: str, message: str) -> None:
        _add_issue(
            issues,
            severity,
            code,
            message,
            line_number=line_number,
            transition_id=transition_id,
        )

    if record.get("schema_version") != SCHEMA_VERSION:
        issue(
            "error",
            "schema_version",
            f"schema_version must be {SCHEMA_VERSION!r}",
        )
    if transition_id is None:
        issue("error", "transition_id", "transition_id must be a non-empty string")
    if not _nonempty_string(record.get("ensemble_id")):
        issue("error", "ensemble_id", "ensemble_id must be a non-empty string")

    status = record.get("status")
    if status not in RECORD_STATUSES:
        issue("error", "status", f"status must be one of {sorted(RECORD_STATUSES)}")

    source = _mapping(record, "source")
    if not source:
        issue("error", "source", "source must be an object")
    else:
        if not _nonempty_string(source.get("name")):
            issue("error", "source_name", "source.name must be a non-empty string")
        if not _nonempty_string(source.get("record_url")):
            issue("warning", "source_url", "source.record_url is missing")
        if not _nonempty_string(source.get("license")):
            issue("warning", "source_license", "source.license is missing")

    evidence = _mapping(record, "evidence")
    tier = evidence.get("tier")
    if tier not in EVIDENCE_TIERS:
        issue(
            "error",
            "evidence_tier",
            f"evidence.tier must be one of {sorted(EVIDENCE_TIERS)}",
        )
    for key in (
        "contains_endpoint_transition",
        "biased_sampling",
        "physical_time_interpretable",
    ):
        if not isinstance(evidence.get(key), bool):
            issue("error", f"evidence_{key}", f"evidence.{key} must be boolean")

    usage = _mapping(record, "usage")
    for key in ("phase_supervision", "heldout_benchmark", "kinetics_claims"):
        if not isinstance(usage.get(key), bool):
            issue("error", f"usage_{key}", f"usage.{key} must be boolean")

    split = _mapping(record, "split")
    split_name = split.get("name")
    if split_name not in SPLIT_NAMES:
        issue("error", "split_name", f"split.name must be one of {sorted(SPLIT_NAMES)}")

    quality = _mapping(record, "quality")
    for key in ("endpoint_mapping_verified", "transition_verified"):
        if not isinstance(quality.get(key), bool):
            issue("error", f"quality_{key}", f"quality.{key} must be boolean")
    mapping_fraction = quality.get("residue_mapping_fraction")
    if mapping_fraction is not None:
        if not isinstance(mapping_fraction, (int, float)) or isinstance(
            mapping_fraction, bool
        ):
            issue(
                "error",
                "residue_mapping_fraction",
                "quality.residue_mapping_fraction must be numeric or null",
            )
        elif not 0.0 <= float(mapping_fraction) <= 1.0:
            issue(
                "error",
                "residue_mapping_fraction_range",
                "quality.residue_mapping_fraction must be within [0, 1]",
            )

    endpoints = _mapping(record, "endpoints")
    trajectory = _mapping(record, "trajectory")
    coordinate_paths = trajectory.get("coordinate_paths")
    if not isinstance(coordinate_paths, list):
        issue(
            "error",
            "coordinate_paths",
            "trajectory.coordinate_paths must be a list",
        )
        coordinate_paths = []
    elif any(not _nonempty_string(value) for value in coordinate_paths):
        issue(
            "error",
            "coordinate_path_value",
            "trajectory.coordinate_paths entries must be non-empty strings",
        )

    n_frames = trajectory.get("n_frames")
    if n_frames is not None and (
        not isinstance(n_frames, int) or isinstance(n_frames, bool) or n_frames < 1
    ):
        issue("error", "n_frames", "trajectory.n_frames must be a positive integer or null")

    prepared = status == "prepared"
    if prepared:
        if not coordinate_paths:
            issue(
                "error",
                "prepared_without_coordinates",
                "prepared records require trajectory.coordinate_paths",
            )
        if not isinstance(n_frames, int) or isinstance(n_frames, bool) or n_frames < 3:
            issue(
                "error",
                "prepared_frame_count",
                "prepared records require at least three trajectory frames",
            )
        for state in ("apo", "holo"):
            pdb_id = endpoints.get(f"{state}_pdb_id")
            structure_path = endpoints.get(f"{state}_structure_path")
            if not _nonempty_string(pdb_id) and not _nonempty_string(structure_path):
                issue(
                    "error",
                    f"prepared_{state}_endpoint",
                    f"prepared records require an {state} PDB id or structure path",
                )

    phase_supervision = usage.get("phase_supervision") is True
    heldout_benchmark = usage.get("heldout_benchmark") is True
    kinetics_claims = usage.get("kinetics_claims") is True
    transition_present = evidence.get("contains_endpoint_transition") is True
    transition_verified = quality.get("transition_verified") is True
    endpoint_mapping_verified = quality.get("endpoint_mapping_verified") is True

    if phase_supervision:
        if not prepared:
            issue(
                "error",
                "phase_supervision_not_prepared",
                "phase-supervision records must have status=prepared",
            )
        if split_name not in {"train", "val"}:
            issue(
                "error",
                "phase_supervision_split",
                "phase-supervision records must be assigned to train or val",
            )
        if not transition_present or not transition_verified:
            issue(
                "error",
                "phase_supervision_transition",
                "phase supervision requires a verified endpoint transition",
            )
        if not endpoint_mapping_verified:
            issue(
                "error",
                "phase_supervision_mapping",
                "phase supervision requires verified endpoint residue mapping",
            )

    if heldout_benchmark:
        if not prepared:
            issue(
                "error",
                "benchmark_not_prepared",
                "held-out benchmark records must have status=prepared",
            )
        if tier != "gold_atomistic_transition":
            issue(
                "error",
                "benchmark_tier",
                "held-out headline benchmarks require gold_atomistic_transition",
            )
        if split_name != "test":
            issue(
                "error",
                "benchmark_split",
                "held-out benchmark records must use split.name=test",
            )
        if phase_supervision:
            issue(
                "error",
                "benchmark_training_leakage",
                "held-out benchmark records cannot provide phase supervision",
            )
        if not transition_present or not transition_verified:
            issue(
                "error",
                "benchmark_transition",
                "held-out benchmarks require a verified endpoint transition",
            )
        if not endpoint_mapping_verified:
            issue(
                "error",
                "benchmark_mapping",
                "held-out benchmarks require verified endpoint residue mapping",
            )
        if not _nonempty_string(split.get("family_group")):
            issue(
                "error",
                "benchmark_family_group",
                "held-out benchmarks require split.family_group",
            )

    if kinetics_claims:
        if evidence.get("physical_time_interpretable") is not True:
            issue(
                "error",
                "kinetics_without_time",
                "kinetics claims require physical_time_interpretable=true",
            )
        if tier not in {"gold_atomistic_transition", "silver_enhanced_sampling"}:
            issue(
                "error",
                "kinetics_tier",
                "kinetics claims are not permitted for modeled/equilibrium-only paths",
            )
        if evidence.get("biased_sampling") is True and not _nonempty_string(
            evidence.get("kinetics_justification")
        ):
            issue(
                "error",
                "biased_kinetics_justification",
                "biased kinetics claims require evidence.kinetics_justification",
            )

    if tier == "context_equilibrium" and transition_present:
        issue(
            "warning",
            "equilibrium_transition_mismatch",
            "context_equilibrium records should be promoted after transition verification",
        )
    if tier == "bronze_modeled_path" and evidence.get(
        "physical_time_interpretable"
    ) is True:
        issue(
            "error",
            "modeled_path_physical_time",
            "bronze modeled paths cannot carry physical-time interpretation",
        )

    if check_files:
        root = Path(base_dir) if base_dir is not None else Path.cwd()
        file_fields: List[Tuple[str, str]] = []
        for key in ("apo_structure_path", "holo_structure_path"):
            value = endpoints.get(key)
            if _nonempty_string(value):
                file_fields.append((f"endpoints.{key}", str(value)))
        topology_path = trajectory.get("topology_path")
        if _nonempty_string(topology_path):
            file_fields.append(("trajectory.topology_path", str(topology_path)))
        for index, value in enumerate(coordinate_paths):
            if _nonempty_string(value):
                file_fields.append((f"trajectory.coordinate_paths[{index}]", str(value)))
        for field, value in file_fields:
            if not _is_remote_path(value) and not _resolve_local_path(value, root).is_file():
                issue("error", "missing_file", f"{field} does not exist: {value}")

    return issues


def phase_supervision_eligible(record: Mapping[str, Any]) -> bool:
    """Return whether a fully prepared record can supervise residue phase."""

    trajectory = _mapping(record, "trajectory")
    coordinate_paths = trajectory.get("coordinate_paths")
    n_frames = trajectory.get("n_frames")
    return (
        record.get("status") == "prepared"
        and isinstance(coordinate_paths, list)
        and bool(coordinate_paths)
        and isinstance(n_frames, int)
        and not isinstance(n_frames, bool)
        and n_frames >= 3
        and _mapping(record, "usage").get("phase_supervision") is True
        and _mapping(record, "split").get("name") in {"train", "val"}
        and _mapping(record, "evidence").get("contains_endpoint_transition") is True
        and _mapping(record, "quality").get("transition_verified") is True
        and _mapping(record, "quality").get("endpoint_mapping_verified") is True
    )


def heldout_benchmark_eligible(record: Mapping[str, Any]) -> bool:
    """Return whether a fully prepared record meets the gold benchmark gate."""

    trajectory = _mapping(record, "trajectory")
    coordinate_paths = trajectory.get("coordinate_paths")
    n_frames = trajectory.get("n_frames")
    return (
        record.get("status") == "prepared"
        and isinstance(coordinate_paths, list)
        and bool(coordinate_paths)
        and isinstance(n_frames, int)
        and not isinstance(n_frames, bool)
        and n_frames >= 3
        and _mapping(record, "usage").get("heldout_benchmark") is True
        and _mapping(record, "usage").get("phase_supervision") is False
        and _mapping(record, "split").get("name") == "test"
        and _mapping(record, "evidence").get("tier")
        == "gold_atomistic_transition"
        and _mapping(record, "evidence").get("contains_endpoint_transition") is True
        and _mapping(record, "quality").get("transition_verified") is True
        and _mapping(record, "quality").get("endpoint_mapping_verified") is True
        and _nonempty_string(_mapping(record, "split").get("family_group"))
    )


def audit_transition_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    initial_issues: Iterable[ManifestIssue] = (),
    base_dir: str | Path | None = None,
    check_files: bool = False,
) -> Tuple[List[ManifestIssue], Dict[str, Any]]:
    """Validate a complete manifest and summarize evidence/usage composition."""

    issues = list(initial_issues)
    seen_ids: Dict[str, int] = {}
    split_memberships: Dict[str, Dict[str, set[str]]] = {
        "ensemble_id": {},
        "family_group": {},
        "ligand_scaffold_group": {},
    }
    for index, record in enumerate(records, start=1):
        line_number = int(record.get("_manifest_line_number", index))
        transition_id = record.get("transition_id")
        if _nonempty_string(transition_id):
            normalized_id = str(transition_id).strip()
            if normalized_id in seen_ids:
                _add_issue(
                    issues,
                    "error",
                    "duplicate_transition_id",
                    f"duplicate transition_id; first seen on line {seen_ids[normalized_id]}",
                    line_number=line_number,
                    transition_id=normalized_id,
                )
            else:
                seen_ids[normalized_id] = line_number
        issues.extend(
            validate_transition_record(
                record,
                line_number=line_number,
                base_dir=base_dir,
                check_files=check_files,
            )
        )

        split = _mapping(record, "split")
        split_name = split.get("name")
        if split_name in {"train", "val", "test"}:
            membership_values = {
                "ensemble_id": record.get("ensemble_id"),
                "family_group": split.get("family_group"),
                "ligand_scaffold_group": split.get("ligand_scaffold_group"),
            }
            for membership_name, membership_value in membership_values.items():
                if _nonempty_string(membership_value):
                    normalized_value = str(membership_value).strip()
                    split_memberships[membership_name].setdefault(
                        normalized_value, set()
                    ).add(str(split_name))

    for membership_name, memberships in split_memberships.items():
        for membership_value, assigned_splits in memberships.items():
            if "test" in assigned_splits and assigned_splits - {"test"}:
                _add_issue(
                    issues,
                    "error",
                    f"{membership_name}_split_leakage",
                    (
                        f"{membership_name}={membership_value!r} crosses held-out "
                        f"splits: {sorted(assigned_splits)}"
                    ),
                    line_number=None,
                    transition_id=None,
                )

    severity_counts = Counter(issue.severity for issue in issues)
    status_counts = Counter(str(record.get("status", "missing")) for record in records)
    tier_counts = Counter(
        str(_mapping(record, "evidence").get("tier", "missing"))
        for record in records
    )
    split_counts = Counter(
        str(_mapping(record, "split").get("name", "missing")) for record in records
    )
    source_counts = Counter(
        str(_mapping(record, "source").get("name", "missing")) for record in records
    )
    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "num_records": len(records),
        "num_errors": severity_counts.get("error", 0),
        "num_warnings": severity_counts.get("warning", 0),
        "phase_supervision_eligible": sum(
            phase_supervision_eligible(record) for record in records
        ),
        "heldout_benchmark_eligible": sum(
            heldout_benchmark_eligible(record) for record in records
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "tier_counts": dict(sorted(tier_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
    }
    return issues, summary
