#!/usr/bin/env python3
"""Register a passed endpoint-equilibrium trajectory in the canonical MD manifest."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.md_transition_manifest import (  # noqa: E402
    load_transition_manifest,
    validate_transition_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--transition-id", required=True)
    parser.add_argument("--npt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replica-index", type=int, default=0)
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
    return parser.parse_args()


def load_one_candidate(path: Path, transition_id: str) -> Dict[str, Any]:
    records, issues = load_transition_manifest(path)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        raise ValueError(f"Candidate manifest contains {len(errors)} parse errors")
    matches = [record for record in records if record.get("transition_id") == transition_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one transition_id={transition_id!r}, found {len(matches)}")
    record = dict(matches[0])
    record.pop("_manifest_line_number", None)
    return record


def load_npt_metrics(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            row: Dict[str, Any] = dict(raw)
            for key in (
                "step",
                "time_ps",
                "temperature_k",
                "potential_kj_mol",
                "kinetic_kj_mol",
                "total_energy_kj_mol",
                "volume_nm3",
                "density_g_ml",
            ):
                row[key] = float(row[key])
            rows.append(row)
    if len(rows) < 3:
        raise ValueError(f"NPT metrics require at least three frames, found {len(rows)}")
    return rows


def infer_frame_interval_ps(rows: List[Mapping[str, Any]]) -> float:
    times = [float(row["time_ps"]) for row in rows]
    differences = [right - left for left, right in zip(times, times[1:]) if right > left]
    if not differences:
        raise ValueError("Could not infer a positive frame interval")
    return float(statistics.median(differences))


def build_context_record(
    candidate: Mapping[str, Any],
    *,
    npt_dir: Path,
    npt_report: Mapping[str, Any],
    metrics: List[Mapping[str, Any]],
    replica_index: int,
) -> Dict[str, Any]:
    if npt_report.get("status") != "npt_smoke_passed":
        raise ValueError(f"NPT report did not pass: {npt_report.get('status')!r}")
    record = copy.deepcopy(dict(candidate))
    base_transition_id = str(record["transition_id"]).removesuffix(":pilot")
    record["transition_id"] = f"{base_transition_id}:context-npt-{replica_index}"
    record["status"] = "prepared"
    record["evidence"] = {
        "tier": "context_equilibrium",
        "contains_endpoint_transition": False,
        "biased_sampling": False,
        "physical_time_interpretable": True,
    }
    record["trajectory"] = {
        "topology_path": str(npt_dir / "final_npt.pdb"),
        "coordinate_paths": [str(npt_dir / "npt_smoke.dcd")],
        "n_frames": len(metrics),
        "frame_interval_ps": infer_frame_interval_ps(metrics),
    }
    record["usage"] = {
        "phase_supervision": False,
        "heldout_benchmark": False,
        "kinetics_claims": False,
    }
    quality = dict(record.get("quality") or {})
    quality["transition_verified"] = False
    notes = list(quality.get("notes") or [])
    notes.append(
        "Passed restrained heating, short NVT, and short NPT endpoint-stability gates; "
        "this equilibrium replica contains no verified apo-holo transition."
    )
    quality["notes"] = notes
    record["quality"] = quality
    record["source_metadata"] = {
        **dict(record.get("source_metadata") or {}),
        "generated_protocol": "OpenMM ff14SB/TIP3P + OpenFF 2.2.1; staged minimization; NVT; NPT",
        "replica_index": replica_index,
        "npt_report": dict(npt_report.get("final") or {}),
    }
    return record


def main() -> None:
    args = parse_args()
    candidate = load_one_candidate(args.candidate_manifest, args.transition_id)
    report_path = args.npt_dir / "npt_report.json"
    metrics_path = args.npt_dir / "npt_metrics.csv"
    if not report_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"NPT output is incomplete: {args.npt_dir}")
    report = json.loads(report_path.read_text())
    metrics = load_npt_metrics(metrics_path)
    record = build_context_record(
        candidate,
        npt_dir=args.npt_dir,
        npt_report=report,
        metrics=metrics,
        replica_index=args.replica_index,
    )
    issues = validate_transition_record(record, base_dir=args.base_dir, check_files=True)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        details = "; ".join(f"{issue.code}: {issue.message}" for issue in errors[:10])
        raise ValueError(f"Generated record failed validation: {details}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "output": str(args.output),
        "transition_id": record["transition_id"],
        "status": record["status"],
        "tier": record["evidence"]["tier"],
        "n_frames": record["trajectory"]["n_frames"],
        "frame_interval_ps": record["trajectory"]["frame_interval_ps"],
        "phase_supervision": record["usage"]["phase_supervision"],
        "warnings": [issue.to_dict() for issue in issues if issue.severity == "warning"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
