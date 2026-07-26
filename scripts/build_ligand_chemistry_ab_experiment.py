#!/usr/bin/env python3
"""Set up the paired legacy-vs-repaired ligand chemistry experiment.

The 2026-07-26 bond-order repair rewrote ``ligand.sdf`` for 315 samples, of
which 198 belong to the frozen consensus303 corpus and are of the *silent*
class: the connectivity-only SDF did not crash MD setup, it merely described a
different molecule.  Entire fused aromatic systems were simulated fully
saturated -- 2hdr-A-4A3-506 goes from 0 to 90 aromatic bonds on repair.

The open question is not whether the chemistry was wrong (it was) but whether
it changed the *paths* enough to invalidate the silver supervision.  This script
builds the two arms of that measurement:

  legacy    a shadow triplet tree whose ligand.sdf is the preserved
            ``ligand.legacy_connectivity_only.sdf``
  repaired  a shadow triplet tree whose ligand.sdf is the current one

Both arms are shadow trees so the two runs differ in exactly one file and take
an identical code path, and so no frozen artifact is touched or even opened for
writing.  Endpoints, coordinates and metadata are symlinked, never copied.

Seeds and protocol come from the source manifest unchanged, so the arms are
paired: same system, same seed, same everything but the ligand's bond orders.

Emits one candidate manifest per arm, ready for
``build_md_context_matrix.py --candidate-manifest``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMA_VERSION = "bindrae_ligand_chemistry_ab_experiment_v1"
LEGACY_SDF_NAME = "ligand.legacy_connectivity_only.sdf"

#: Files an arm's shadow sample directory needs. ligand.sdf is written per arm.
LINKED_FILES = ("apo.pdb", "holo.pdb", "ligand_coords.npy", "meta.json")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sample_id_of(record: dict[str, Any]) -> str:
    """Sample id as build_md_context_matrix derives it: the holo path's parent."""

    endpoints = record.get("endpoints") or {}
    holo = str(endpoints.get("holo_structure_path") or "")
    return Path(holo).parent.name if holo else ""


def select_systems(
    repair_records: Path,
    corpus: Path,
    *,
    count: int,
    require_silent: bool = True,
) -> list[str]:
    """Pick the systems whose chemistry changed most, within the frozen corpus.

    ``require_silent`` keeps only samples whose legacy SDF would *not* have
    crashed setup.  The crashing class needs no experiment: it has no legacy
    path to compare against, because it never produced one.
    """

    rows = {
        json.loads(line)["sample_id"]: json.loads(line)
        for line in Path(repair_records).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    wanted = [x.strip() for x in Path(corpus).read_text(encoding="utf-8").split() if x.strip()]
    picked = []
    for sample_id in wanted:
        row = rows.get(sample_id)
        if not row or row.get("status") != "repaired":
            continue
        if int(row.get("bond_orders_changed", 0)) <= 0:
            continue
        if require_silent and row["before"]["sentinel_hydrides"]:
            continue
        picked.append((int(row["bond_orders_changed"]), sample_id))
    picked.sort(reverse=True)
    # Two ligand copies in the same complex (2hdr-A-4A3-506 and -511) are
    # near-duplicate evidence, so keep one sample per deposited entry.
    seen_entries: set[str] = set()
    unique: list[str] = []
    for _, sample_id in picked:
        entry = sample_id.split("-")[0].lower()
        if entry in seen_entries:
            continue
        seen_entries.add(entry)
        unique.append(sample_id)
    return unique[:count]


def build_arm(
    *,
    arm: str,
    sample_ids: Sequence[str],
    triplet_root: Path,
    output_root: Path,
    records: dict[str, dict[str, Any]],
) -> Path:
    """Materialise one arm's shadow tree and candidate manifest."""

    arm_root = output_root / arm
    tree = arm_root / "samples"
    tree.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        source = triplet_root / sample_id
        target = tree / sample_id
        target.mkdir(parents=True, exist_ok=True)

        for name in LINKED_FILES:
            origin = source / name
            link = target / name
            if not origin.is_file():
                raise FileNotFoundError(f"{origin} is missing; cannot build {arm} arm")
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(origin.resolve())

        if arm == "legacy":
            ligand_source = source / LEGACY_SDF_NAME
            if not ligand_source.is_file():
                raise FileNotFoundError(
                    f"{ligand_source} is missing; the legacy arm needs the preserved "
                    "pre-repair SDF, so this sample cannot be compared"
                )
        else:
            ligand_source = source / "ligand.sdf"
        ligand_link = target / "ligand.sdf"
        if ligand_link.is_symlink() or ligand_link.exists():
            ligand_link.unlink()
        ligand_link.symlink_to(ligand_source.resolve())

        record = json.loads(json.dumps(records[sample_id]))  # deep copy
        endpoints = record["endpoints"]
        endpoints["apo_structure_path"] = str((target / "apo.pdb").resolve())
        endpoints["holo_structure_path"] = str((target / "holo.pdb").resolve())
        record["ligand_chemistry_arm"] = arm
        record["ligand_sdf_source"] = str(ligand_source.resolve())
        manifest_rows.append(record)

    manifest = arm_root / "candidate_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source-manifest", type=Path, required=True,
                        help="Manifest the frozen corpus was built from")
    parser.add_argument("--repair-records", type=Path, required=True)
    parser.add_argument("--corpus-list", type=Path, required=True,
                        help="consensus303 sample id list")
    parser.add_argument("--triplet-root", type=Path,
                        default=Path("processed_data/triplets/samples"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--sample", action="append", default=[],
                        help="Explicit sample id; repeatable. Overrides --count selection")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    manifest = load_manifest(args.source_manifest)
    records = {}
    for record in manifest:
        sample_id = sample_id_of(record)
        if sample_id:
            records[sample_id] = record

    if args.sample:
        sample_ids = list(dict.fromkeys(args.sample))
        missing = [s for s in sample_ids if s not in records]
        if missing:
            raise SystemExit(
                f"{len(missing)} explicitly requested sample(s) are absent from "
                f"{args.source_manifest}: {missing[:5]}"
            )
    else:
        # The frozen corpus was assembled across several selection rounds, so no
        # single manifest covers all 303. Rank by chemistry change, then take the
        # best systems this manifest can actually supply.
        ranked = select_systems(
            args.repair_records, args.corpus_list, count=10 ** 6
        )
        available = [s for s in ranked if s in records]
        if len(available) < args.count:
            raise SystemExit(
                f"Only {len(available)} eligible sample(s) are in {args.source_manifest}; "
                f"{args.count} requested"
            )
        sample_ids = available[: args.count]
        print(
            f"# ranked {len(ranked)} eligible systems, {len(available)} present in this "
            f"manifest, taking top {args.count}",
            file=sys.stderr,
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for arm in ("legacy", "repaired"):
        manifests[arm] = build_arm(
            arm=arm,
            sample_ids=sample_ids,
            triplet_root=args.triplet_root,
            output_root=args.output_root,
            records=records,
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": str(args.source_manifest),
        "triplet_root": str(args.triplet_root),
        "output_root": str(args.output_root),
        "systems": sample_ids,
        "arms": {arm: str(path) for arm, path in manifests.items()},
        "design": (
            "Paired: identical seeds and protocol from the source manifest; the arms "
            "differ in exactly one file per sample, ligand.sdf. Both arms are shadow "
            "trees, so no frozen artifact is written."
        ),
    }
    (args.output_root / "experiment.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
