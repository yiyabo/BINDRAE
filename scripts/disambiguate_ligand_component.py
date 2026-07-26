#!/usr/bin/env python3
"""Decide which CCD component an ambiguous observed ligand actually is.

Some components share a heavy-atom graph and differ only in bond orders and
formal charge -- NAI (NADH, neutral, puckered 1,4-dihydronicotinamide) and NAD
(NAD+, aromatic planar pyridinium) are the canonical pair.  A subgraph match
cannot separate them, so ``repair_triplet_ligand_bond_orders.py`` refuses to
guess and rejects the sample as ``resname_chemistry_ambiguous``.

Evidence, in the order it should be trusted:

  provenance   Which name refers to *this* sample at all.  AHoJ pairs a query
               entry with a different apo/holo entry, and the two deposited
               structures routinely hold related-but-distinct components: 1t26
               holds NAI while its holo 1t2d holds NAD, 4rc7 holds PL3 while
               4rc8 holds STE.  ``holo.pdb`` in the sample directory therefore
               contributes the *holo* component's name even when the ligand was
               extracted from the query structure.  Every ambiguity observed in
               the 2026-07-26 corpus was of this kind -- a provenance artifact,
               not a chemical question.

  coverage     Heavy atoms observed per copy divided by the component's atom
               count.  A candidate covered only partially is the wrong
               component.  This settled 1ivc (ST2 1.000 vs ST1 0.882) and 4rc7
               (PL3 1.000 vs STE 0.850).

  bond geometry  ADVISORY ONLY -- reported, never decisive.  Comparing observed
               bond lengths against the candidate's CCD *ideal* conformer was
               tried as a discriminator and **failed on real data**: it chose
               NAD over NAI for 1t26 and STE over PL3 for 4rc7, both contrary
               to the deposited entries and, for 4rc7, contrary to coverage.
               The idea looks sound -- NAD+ has an aromatic planar pyridinium
               where NADH has a puckered 1,4-dihydro ring -- and it separates
               cleanly when the observed geometry *is* one of the ideal
               conformers.  Against actual crystal coordinates the margin does
               not survive.  It is kept in the report because the numbers are
               informative, not because they decide anything.

Ring planarity is reported on the same advisory footing.

The authoritative check is not in this script: ask the deposited entries
(``https://data.rcsb.org/rest/v1/core/entry/<id>``, falling back to the
nonpolymer-entity records, which are needed whenever
``nonpolymer_bound_components`` is absent) which components ``meta.query_pdb``
and ``meta.holo_pdb`` actually contain.

The script never writes to a sample directory.  It emits a ranked verdict per
sample; applying a decision is a separate, explicit step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ligand_bond_orders import (  # noqa: E402
    LigandBondOrderError,
    load_ccd_template,
    read_observed_ligand,
    reconstruct_bond_orders,
)

SCHEMA_VERSION = "bindrae_ligand_component_disambiguation_v1"

#: Retained so the margin can still be reported, but bond geometry no longer
#: decides a verdict: see the module docstring for why it was demoted.
DEFAULT_DECISION_MARGIN_ANGSTROM = 0.02


def candidate_provenance(sample_dir: Path, resname: str) -> list[str]:
    """Where this resname came from -- meta and sample id outrank stray HETATMs."""

    sources: list[str] = []
    resname = resname.upper()
    meta_path = sample_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if str(meta.get("ligand_resname") or "").upper() == resname:
            sources.append("meta.ligand_resname")
        if str(meta.get("effective_ligand_resname") or "").upper() == resname:
            sources.append("meta.effective_ligand_resname")
    parts = sample_dir.name.split("-")
    if len(parts) >= 3 and parts[-2].upper() == resname:
        sources.append("sample_id")
    holo = sample_dir / "holo.pdb"
    if holo.is_file():
        for line in holo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("HETATM") and len(line) >= 20:
                if line[17:20].strip().upper() == resname:
                    sources.append("holo_pdb_hetatm")
                    break
    return sources


def _bond_lengths(molecule: Any) -> dict[tuple[int, int], float]:
    import numpy as np

    positions = np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64)
    lengths: dict[tuple[int, int], float] = {}
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        lengths[(min(i, j), max(i, j))] = float(
            np.linalg.norm(positions[i] - positions[j])
        )
    return lengths


def _plane_rms(positions: Any) -> float:
    """RMS deviation of points from their best-fit plane."""

    import numpy as np

    points = np.asarray(positions, dtype=np.float64)
    if len(points) < 4:
        return 0.0
    centred = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    normal = vh[-1]
    return float(np.sqrt(np.mean((centred @ normal) ** 2)))


def score_candidate(
    observed: Any,
    template: Any,
    *,
    discriminating_bonds: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Fit of the observed geometry to one candidate component."""

    import numpy as np
    from rdkit import Chem

    reconstructed, diagnostics = reconstruct_bond_orders(observed, template)
    # Multi-copy ligands are matched per copy, so take the correspondence the
    # reconstruction already computed instead of re-matching the whole molecule.
    match = diagnostics["atom_template_index"]
    observed_lengths = _bond_lengths(reconstructed)
    template_lengths = _bond_lengths(template)

    per_bond: dict[tuple[int, int], dict[str, float]] = {}
    for (i, j), observed_length in observed_lengths.items():
        if i >= len(match) or j >= len(match):
            continue
        key = (min(match[i], match[j]), max(match[i], match[j]))
        ideal = template_lengths.get(key)
        if ideal is None:
            continue
        bond = reconstructed.GetBondBetweenAtoms(i, j)
        per_bond[(i, j)] = {
            "observed": observed_length,
            "ideal": ideal,
            "delta": observed_length - ideal,
            "order": str(bond.GetBondType()),
        }

    def rmsd(keys: Sequence[tuple[int, int]]) -> float | None:
        values = [per_bond[k]["delta"] for k in keys if k in per_bond]
        if not values:
            return None
        return float(np.sqrt(np.mean(np.square(values))))

    all_bond_rmsd = rmsd(list(per_bond))
    discriminating_rmsd = (
        rmsd(sorted(discriminating_bonds)) if discriminating_bonds else None
    )

    observed_xyz = np.asarray(
        reconstructed.GetConformer().GetPositions(), dtype=np.float64
    )
    template_xyz = np.asarray(template.GetConformer().GetPositions(), dtype=np.float64)
    rings = []
    for ring in Chem.GetSymmSSSR(reconstructed):
        indices = list(ring)
        if len(indices) < 5:
            continue
        template_indices = [match[i] for i in indices if i < len(match)]
        if len(template_indices) != len(indices):
            continue
        rings.append(
            {
                "size": len(indices),
                "aromatic": all(
                    reconstructed.GetAtomWithIdx(i).GetIsAromatic() for i in indices
                ),
                "observed_plane_rms": _plane_rms(observed_xyz[indices]),
                "ideal_plane_rms": _plane_rms(template_xyz[template_indices]),
            }
        )

    template_atoms = int(template.GetNumAtoms())
    copies = int(diagnostics["component_copies"])
    return {
        "coverage": (
            (float(reconstructed.GetNumAtoms()) / copies) / template_atoms
            if template_atoms and copies else None
        ),
        "template_heavy_atoms": template_atoms,
        "formal_charge": int(Chem.GetFormalCharge(reconstructed)),
        "canonical_smiles": str(
            Chem.MolToSmiles(reconstructed, canonical=True, isomericSmiles=False)
        ),
        "aromatic_bonds": int(
            sum(1 for b in reconstructed.GetBonds() if b.GetIsAromatic())
        ),
        "component_copies": int(diagnostics["component_copies"]),
        "bond_length_rmsd_all_angstrom": all_bond_rmsd,
        "bond_length_rmsd_discriminating_angstrom": discriminating_rmsd,
        "rings": rings,
        "_bond_orders": {k: v["order"] for k, v in per_bond.items()},
    }


def discriminating_bond_set(scores: Sequence[dict[str, Any]]) -> set[tuple[int, int]]:
    """Bonds on which the candidates actually disagree."""

    if len(scores) < 2:
        return set()
    keys = set(scores[0]["_bond_orders"])
    for score in scores[1:]:
        keys &= set(score["_bond_orders"])
    return {
        key
        for key in keys
        if len({score["_bond_orders"][key] for score in scores}) > 1
    }


def disambiguate(
    sample_dir: Path,
    candidates: Sequence[str],
    *,
    ccd_dir: Path,
    decision_margin: float = DEFAULT_DECISION_MARGIN_ANGSTROM,
) -> dict[str, Any]:
    observed, _, _ = read_observed_ligand(sample_dir / "ligand.sdf")

    viable: list[tuple[str, Any]] = []
    failures: list[dict[str, str]] = []
    for resname in candidates:
        try:
            template, _ = load_ccd_template(resname, ccd_dir=ccd_dir)
            reconstruct_bond_orders(observed, template)
            viable.append((resname, template))
        except LigandBondOrderError as exc:
            failures.append({"resname": resname, "reason": exc.reason})

    # First pass to learn where the candidates disagree, second to score there.
    first = [score_candidate(observed, template) for _, template in viable]
    discriminating = discriminating_bond_set(first)
    scores = []
    for (resname, template) in viable:
        score = score_candidate(observed, template, discriminating_bonds=discriminating)
        score["resname"] = resname
        score["provenance"] = candidate_provenance(sample_dir, resname)
        score["named_by_sample"] = bool(
            {"meta.ligand_resname", "meta.effective_ligand_resname", "sample_id"}
            & set(score["provenance"])
        )
        score.pop("_bond_orders", None)
        scores.append(score)

    verdict: dict[str, Any] = {
        "decision": None,
        "basis": None,
        "discriminating_bonds": len(discriminating),
    }

    named = [s for s in scores if s["named_by_sample"]]
    if len(scores) == 1:
        verdict.update(decision=scores[0]["resname"], basis="only_viable_candidate")
    elif len(named) == 1:
        # The others are incidental HETATMs from the same file, not this residue.
        verdict.update(
            decision=named[0]["resname"],
            basis="only_candidate_named_by_meta_or_sample_id",
        )
    else:
        full = [s for s in scores if s.get("coverage") is not None and abs(s["coverage"] - 1.0) < 1e-9]
        partial = [s for s in scores if s.get("coverage") is not None and s["coverage"] < 1.0]
        if len(full) == 1 and partial:
            verdict.update(decision=full[0]["resname"], basis="full_component_coverage")
        else:
            verdict["basis"] = "needs_deposited_entry_lookup"
        ranked = sorted(
            (s for s in scores if s["bond_length_rmsd_discriminating_angstrom"] is not None),
            key=lambda s: s["bond_length_rmsd_discriminating_angstrom"],
        )
        if len(ranked) >= 2:
            # Advisory only. This margin chose wrongly on 2 of 5 real samples.
            verdict["advisory_geometry_pick"] = ranked[0]["resname"]
            verdict["advisory_margin_angstrom"] = (
                ranked[1]["bond_length_rmsd_discriminating_angstrom"]
                - ranked[0]["bond_length_rmsd_discriminating_angstrom"]
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_dir.name,
        "sample_dir": str(sample_dir),
        "candidates": candidates,
        "non_viable": failures,
        "scores": sorted(
            scores, key=lambda s: (not s["named_by_sample"], s["resname"])
        ),
        "verdict": verdict,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--triplet-root", type=Path, required=True)
    parser.add_argument("--ccd-dir", type=Path, required=True)
    parser.add_argument(
        "--repair-records", type=Path, default=None,
        help="records.jsonl from a repair run; ambiguous rows are read from it",
    )
    parser.add_argument(
        "--sample", action="append", default=[],
        help="Explicit sample id; repeatable. Candidates come from the repair records "
             "when available, otherwise from resname_candidates()",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--decision-margin", type=float, default=DEFAULT_DECISION_MARGIN_ANGSTROM
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    wanted: dict[str, list[str]] = {}
    if args.repair_records is not None:
        for line in Path(args.repair_records).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("reason") != "resname_chemistry_ambiguous":
                continue
            wanted[row["sample_id"]] = [
                entry["resname"] for entry in row.get("accepted_resnames", [])
            ]
    for sample_id in args.sample:
        wanted.setdefault(sample_id, [])

    if not wanted:
        print("No ambiguous samples to resolve.")
        return 0

    def fallback_candidates(sample_dir: Path) -> list[str]:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "repair_triplet_ligand_bond_orders",
            Path(__file__).resolve().parent / "repair_triplet_ligand_bond_orders.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.resname_candidates(sample_dir)

    results = []
    for sample_id, candidates in sorted(wanted.items()):
        sample_dir = args.triplet_root / sample_id
        if not candidates:
            candidates = fallback_candidates(sample_dir)
        results.append(
            disambiguate(
                sample_dir, candidates,
                ccd_dir=args.ccd_dir,
                decision_margin=args.decision_margin,
            )
        )

    for result in results:
        verdict = result["verdict"]
        print(f"\n{'=' * 72}\n{result['sample_id']}")
        print(f"{'=' * 72}")
        for score in result["scores"]:
            flag = "*" if score["named_by_sample"] else " "
            disc = score["bond_length_rmsd_discriminating_angstrom"]
            print(
                f" {flag} {score['resname']:<5} charge={score['formal_charge']:+d} "
                f"tmpl_atoms={score['template_heavy_atoms']:<4} "
                f"coverage={score['coverage']:.3f} "
                f"[advisory bondRMSD_disc={'n/a' if disc is None else f'{disc:.4f}'}]"
            )
            print(f"      provenance={score['provenance']}")
            for ring in score["rings"]:
                print(
                    f"      ring{ring['size']} aromatic={ring['aromatic']} "
                    f"planeRMS obs={ring['observed_plane_rms']:.3f} "
                    f"ideal={ring['ideal_plane_rms']:.3f}"
                )
        print(
            f" -> decision={verdict['decision']}  basis={verdict['basis']}"
            + (
                f"  margin={verdict['margin_angstrom']:.4f} A"
                if "margin_angstrom" in verdict
                else ""
            )
        )

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nReport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
