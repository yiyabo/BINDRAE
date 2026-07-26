"""Test the premise behind the site-metal hypothesis for MD setup failures.

`scripts/prepare_md_pilot_system.py` repairs the holo endpoint with
``PDBFixer.removeHeterogens(keepWater=False)``, which discards every heterogen
including catalytic metal ions, and then re-adds only the ligand from its SDF.
The observed setup failures are concentrated in polyphosphate cofactors, which
are normally chelated by a divalent metal.

This script measures whether systems that failed OpenMM setup are enriched for a
metal ion in the ligand site, using the frozen panel manifest and the pipeline
outcomes that already exist on disk.

It is strictly read-only. It changes no preparation behaviour, no threshold, and
no manifest, and it creates no training data. A positive result would justify an
opt-in metal-retention policy; a negative result falsifies the hypothesis before
any pipeline change is written.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Element symbols treated as potentially structural/catalytic site metals.
# Alkali ions are included so the report can separate them from divalent ions
# rather than silently assuming which species matters.
METAL_ELEMENTS = frozenset(
    {
        "MG", "MN", "ZN", "CA", "FE", "NI", "CO", "CU", "CD", "HG",
        "NA", "K", "MO", "W", "V", "SR", "BA",
    }
)
DIVALENT_METALS = frozenset({"MG", "MN", "ZN", "CA", "FE", "NI", "CO", "CU", "CD"})
SKIP_RESIDUES = frozenset({"HOH", "WAT", "DOD"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", required=True)
    parser.add_argument(
        "--context-root",
        required=True,
        help="Context directory containing per-system pipeline_status.json files.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--site-distance-angstrom",
        type=float,
        default=5.0,
        help="A metal is 'in the site' when it is within this distance of a ligand heavy atom.",
    )
    return parser.parse_args()


def _read_records(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sample_id_from_paths(record: Mapping[str, Any]) -> str:
    endpoints = dict(record.get("endpoints") or {})
    holo = str(endpoints.get("holo_structure_path") or "")
    return Path(holo).parent.name if holo else ""


def _ligand_comp_id(record: Mapping[str, Any], sample_id: str) -> str:
    comp = (record.get("ligand") or {}).get("comp_id")
    if comp:
        return str(comp).strip().upper()
    parts = sample_id.split("-")
    return parts[2].strip().upper() if len(parts) >= 3 else ""


def _parse_hetatms(
    pdb_path: Path,
) -> Tuple[List[Tuple[str, str, str, Tuple[float, float, float]]], int]:
    """Return (resName, element, chain+seq, xyz) for HETATM rows, plus a row count."""
    rows: List[Tuple[str, str, str, Tuple[float, float, float]]] = []
    total = 0
    with pdb_path.open("r", errors="replace") as handle:
        for line in handle:
            if not line.startswith("HETATM"):
                continue
            total += 1
            res_name = line[17:20].strip().upper()
            if res_name in SKIP_RESIDUES:
                continue
            element = line[76:78].strip().upper()
            if not element:
                # Fall back to the atom name when the element column is absent.
                element = line[12:16].strip().upper()[:2]
            label = f"{line[21:22].strip()}{line[22:26].strip()}"
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            rows.append((res_name, element, label, xyz))
    return rows, total


def _min_distance(
    point: Tuple[float, float, float],
    cloud: Sequence[Tuple[float, float, float]],
) -> Optional[float]:
    if not cloud:
        return None
    best = math.inf
    px, py, pz = point
    for cx, cy, cz in cloud:
        distance = math.dist((px, py, pz), (cx, cy, cz))
        if distance < best:
            best = distance
    return best


def _outcomes(context_root: Path) -> Dict[str, Dict[str, str]]:
    found: Dict[str, Dict[str, str]] = {}
    for status_path in context_root.rglob("pipeline_status.json"):
        try:
            state = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sample_id = str(state.get("system_sample_id") or "")
        if sample_id:
            found[sample_id] = {
                "status": str(state.get("status") or ""),
                "failed_stage": str(state.get("failed_stage") or ""),
            }
    return found


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    panel = _read_records(Path(args.panel_manifest))
    outcomes = _outcomes(Path(args.context_root))
    records: List[Dict[str, Any]] = []

    for record in panel:
        sample_id = _sample_id_from_paths(record)
        comp_id = _ligand_comp_id(record, sample_id)
        holo_path = Path(str((record.get("endpoints") or {}).get("holo_structure_path") or ""))
        outcome = outcomes.get(sample_id, {"status": "unknown", "failed_stage": ""})
        row: Dict[str, Any] = {
            "sample_id": sample_id,
            "ligand_comp_id": comp_id,
            "status": outcome["status"],
            "failed_stage": outcome["failed_stage"],
            "holo_structure_path": str(holo_path),
        }
        if not holo_path.is_file():
            row["error"] = "missing_holo_structure"
            records.append(row)
            continue

        hetatms, hetatm_rows = _parse_hetatms(holo_path)
        ligand_cloud = [xyz for res_name, _, _, xyz in hetatms if res_name == comp_id]
        metals = [
            (res_name, element, label, xyz)
            for res_name, element, label, xyz in hetatms
            if element in METAL_ELEMENTS and res_name != comp_id
        ]
        metal_rows = []
        for res_name, element, label, xyz in metals:
            distance = _min_distance(xyz, ligand_cloud)
            metal_rows.append(
                {
                    "res_name": res_name,
                    "element": element,
                    "label": label,
                    "min_distance_to_ligand_angstrom": (
                        None if distance is None else round(distance, 3)
                    ),
                    "divalent": element in DIVALENT_METALS,
                }
            )
        metal_rows.sort(
            key=lambda item: (
                item["min_distance_to_ligand_angstrom"] is None,
                item["min_distance_to_ligand_angstrom"] or 0.0,
            )
        )
        in_site = [
            item
            for item in metal_rows
            if item["min_distance_to_ligand_angstrom"] is not None
            and item["min_distance_to_ligand_angstrom"] <= args.site_distance_angstrom
        ]
        row.update(
            hetatm_rows=hetatm_rows,
            ligand_atoms_found=len(ligand_cloud),
            metals=metal_rows,
            site_metals=in_site,
            has_site_metal=bool(in_site),
            has_divalent_site_metal=any(item["divalent"] for item in in_site),
            nearest_metal_angstrom=(
                metal_rows[0]["min_distance_to_ligand_angstrom"] if metal_rows else None
            ),
        )
        records.append(row)

    usable = [row for row in records if "error" not in row and row["ligand_atoms_found"] > 0]
    failed = [row for row in usable if row["status"] == "failed"]
    passed = [row for row in usable if row["status"] == "completed"]

    def rate(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        if not rows:
            return None
        return round(sum(bool(row[key]) for row in rows) / len(rows), 4)

    summary = {
        "site_distance_angstrom": args.site_distance_angstrom,
        "panel_systems": len(panel),
        "usable_systems": len(usable),
        "status_counts": dict(Counter(row["status"] for row in records)),
        "unresolved_ligand_systems": [
            row["sample_id"]
            for row in records
            if "error" not in row and row["ligand_atoms_found"] == 0
        ],
        "failed": {
            "n": len(failed),
            "with_site_metal": sum(row["has_site_metal"] for row in failed),
            "with_divalent_site_metal": sum(row["has_divalent_site_metal"] for row in failed),
            "site_metal_rate": rate(failed, "has_site_metal"),
            "divalent_site_metal_rate": rate(failed, "has_divalent_site_metal"),
        },
        "passed": {
            "n": len(passed),
            "with_site_metal": sum(row["has_site_metal"] for row in passed),
            "with_divalent_site_metal": sum(row["has_divalent_site_metal"] for row in passed),
            "site_metal_rate": rate(passed, "has_site_metal"),
            "divalent_site_metal_rate": rate(passed, "has_divalent_site_metal"),
        },
    }
    return {
        "schema_version": "bindrae_md_setup_site_metal_premise_v1",
        "claim_boundary": (
            "A read-only structural premise check on one 32-system engineering panel. "
            "It does not establish causation, does not change preparation behaviour, "
            "and is not an unbiased estimate for any larger pool."
        ),
        "panel_manifest": str(args.panel_manifest),
        "context_root": str(args.context_root),
        "summary": summary,
        "records": records,
    }


def main() -> None:
    args = parse_args()
    result = analyze(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    summary = result["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    print()
    print(f"{'sample_id':<24} {'lig':<5} {'status':<10} {'metals':<28} nearest_A")
    for row in result["records"]:
        if "error" in row:
            print(f"{row['sample_id']:<24} {row['ligand_comp_id']:<5} {row['status']:<10} {row['error']}")
            continue
        names = ",".join(
            f"{item['res_name']}@{item['min_distance_to_ligand_angstrom']}"
            for item in row["site_metals"]
        ) or "-"
        print(
            f"{row['sample_id']:<24} {row['ligand_comp_id']:<5} {row['status']:<10} "
            f"{names:<28} {row['nearest_metal_angstrom']}"
        )


if __name__ == "__main__":
    main()
