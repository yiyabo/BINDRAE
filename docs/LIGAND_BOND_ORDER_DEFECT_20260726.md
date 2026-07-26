# Ligand Bond-Order Defect and Repair

## Decision

Every `ligand.sdf` written by the AHoJ triplet exporter carried connectivity
only: all bonds order 1, no `M  CHG` record. On 2026-07-26 the defect was traced
to its root cause, a CCD-templated repair was implemented, and `ligand.sdf` was
rewritten for 315 of 335 samples covering the entire frozen consensus303 corpus
and the 32-system AHoJ mapping smoke panel.

**This changes the ligand chemistry underlying the frozen silver MD corpus.**
Endpoint geometry, `ligand_coords.npy`, and Stage-2 model inputs are unaffected.
No scientific threshold, split, cache, or selection metric was relaxed or
altered.

Whether the frozen silver paths must be regenerated is **not decided here**; the
paired measurement that informs that decision is described under
[Open Question](#open-question).

## Root Cause

The PDB format stores no bond orders. `prepare_ahojdb_triplets.py`
built the ligand with `Chem.MolFromPDBBlock`, which perceives connectivity by
proximity and writes every bond as order 1 with zero formal charge.

```text
SDF has connectivity only, no bond orders, no formal charges
  -> phosphorus carries four single bonds, valence 4 (P allows 3 or 5)
  -> RDKit completes the valence to 5 with an implicit hydrogen
  -> OpenFF parameterises a neutral P-H pseudo-phosphate
  -> solvated system reaches a non-finite initial energy
  -> prepare_md_pilot_system.py:328 aborts
```

Non-phosphate ligands do not abort. They are silently parameterised as fully
saturated, fully protonated molecules: in `1t26-A-NAI-416` the adenine ring was
read as a saturated `[C]([H])([H])` ring, and in the four systems selected for
the A/B experiment the entire fused aromatic system was simulated saturated
(0 aromatic bonds where the component declares 90 to 102).

The failing signature is `RuntimeError: Initial system energy or force is
non-finite` at `prepare_md_pilot_system.py:328`, which is where the AHoJ smoke
setup failures came from. Verification of the causal chain, at
`processed_data/triplets/samples/1t26-A-NAI-416/`:

```text
atoms=44  bonds=48  orders={1: 48}  MCHG=0
```

Connectivity is correct (NAD has 44 heavy atoms and 48 bonds). Only bond orders
and formal charges were missing.

### Hypotheses that were tested and refuted

Recorded so they are not re-investigated:

| Hypothesis | Refuted by |
|---|---|
| Site metals stripped by `removeHeterogens` | The failing systems' `holo.pdb` contains no metal at all |
| Ligand/protein steric overlap from a mismatched pose | Minimum ligand-protein distance is 1.2282 A; no pairs below 1.0 A |
| A CPU-platform artifact rather than real chemistry | CUDA (job 148733) and CPU (148734) fail on the identical line |

The metal-premise probe written for the first hypothesis is retained at
`scripts/archive/diagnostics/diagnose_md_setup_site_metal_premise.py`. It is
archived rather than active for two reasons: the hypothesis is dead, and the
probe matched HETATM records by the sample id's ligand code, which the
query/holo component drift described below makes unreliable.

## Repair

`src/data/ligand_bond_orders.py` matches the observed heavy-atom graph against
the PDB Chemical Component Dictionary entry and copies bond orders and formal
charges **onto** the observed molecule. It never rebuilds from the template, so
atom count, atom order, element sequence and coordinates are preserved by
construction and every index-aligned downstream tensor stays valid.

Matching is name-free (element-labelled connectivity), because `SDWriter` drops
the PDB atom names, so the same code serves both generation and repair.

Three defects surfaced across three dry-run rounds and are guarded by tests:

1. **Multi-copy ligands.** `extract_ligand_from_pdb` collects every HETATM
   residue in the chain sharing the resname, with no residue-number filter, so
   glycans arrive as one file holding up to 17 copies. Reconstruction matches
   per connected fragment; at generation time it matches per PDB residue, which
   also separates covalently linked oligosaccharides.
2. **The hydride sentinel was the wrong test.** A glutathione thiol
   legitimately carries `S-H` and the component says so. A fixed element
   blacklist rejected 6 correct samples. The defect signature is carrying *more*
   hydrogens than the matched template atom, not carrying any.
3. **Small ligand inside a large component.** G3H (10 heavy atoms) is a subgraph
   of NAD (44), so partial matching accepted it. A per-copy coverage floor of
   0.80 rules that out, matching the existing APObind preflight threshold.

### Reject ledger

| Outcome | Count |
|---|---|
| Repaired | 315 |
| `substructure_match_failed` | 12 |
| `unobserved_neighbor_on_sentinel_element` | 7 |
| `observed_heavy_atom_fraction_below_threshold` | 1 |
| **Total scoped** | **335** |

Of the 12 match failures, 9 are covalently linked oligosaccharides
(`BGC`/`XYP`/`BMA`; observed atoms equal `n x template - (n-1)`). A glycosidic
bond joins the units into one connected fragment and the written SDF no longer
carries residue numbers, so these are outside the repair's reach and require
regeneration from the source PDB. The generator handles them going forward via
`pdb_residue_groups`. The remaining 3 have matching atom counts but
non-matching graphs and have not been investigated.

The 7 sentinel rejections are partially observed phosphates. A phosphate missing
a terminal oxygen in the crystal cannot be rebuilt to correct chemistry, and
rejecting is the correct outcome.

### Component identity

Five samples were initially rejected as `resname_chemistry_ambiguous`. They were
not chemically ambiguous. AHoJ pairs a *query* entry with a different apo/holo
entry, and the two deposited structures hold related but distinct components:

| Sample | Query entry | Holo entry | Identity |
|---|---|---|---|
| `1ivc-A-ST2-471` | 1ivc: ST2 | 1ivd: ST1 | ST2 |
| `1t26-A-NAI-416` | 1t26: NAI | 1t2d: NAD | NAI |
| `1xqx-A-PCS-300` | 1xqx: PCS | 1xrl: PHK | PCS |
| `4rc7-A-PL3-1001` | 4rc7: PL3 | 4rc8: STE | PL3 |
| `5z7j-A-36J-301` | 5z7j: 36J | 5xo8: ZER | 36J |

`holo.pdb` in the sample directory therefore contributes the holo component's
name even though the ligand was extracted from the query structure: the holo
lookup filters `ligands.json` by the query's `target_ligand`, cannot match, and
falls back to `extract_ligand_from_pdb(query_pdb_path, ...)`. Candidates are now
tiered — `meta.json` and the sample id first, `holo.pdb` HETATM names only as a
rescue — and ambiguity is declared only within a tier.

Coverage independently confirms two of the five (`1ivc`: ST2 1.000 vs ST1 0.882;
`4rc7`: PL3 1.000 vs STE 0.850).

**A discriminator that was tried and failed.** Scoring observed bond lengths
against each candidate's CCD *ideal* conformer separates cleanly when the
observed geometry is itself an ideal conformer (0.0766 A margin on NAI/NAD), but
on real crystal coordinates it chose wrongly on 2 of 5 samples — NAD over NAI
for `1t26`, STE over PL3 for `4rc7`, the latter also contradicting coverage. It
is retained as advisory output in `scripts/disambiguate_ligand_component.py` and
must not be used to decide identity.

## Verification

Setup was re-run post-repair on the identical system, index, seed, platform and
minimization budget as a pre-repair run that failed:

| Run | System | Platform | Seed | Result |
|---|---|---|---|---|
| 148733 (pre-repair) | `1t26-A-NAI-416` | CUDA | 2026072503 | `RuntimeError: Initial system energy or force is non-finite`; no `preparation_report.json` |
| 148796 (post-repair) | `1t26-A-NAI-416` | CUDA | 2026072503 | `minimized_ready_for_dynamics` |
| 148795 (post-repair) | `4xcl-A-AGS-301` | CUDA | 2026072521 | `minimized_ready_for_dynamics` |

Post-repair ligand chemistry from `preparation_report.json`:

```text
1t26  NC(=O)C1=CN([C@@H]2O...)      1,4-dihydronicotinamide, aromatic adenine
4xcl  Nc1ncnc2c1ncn2...OP(=O)(O)O   aromatic adenine, P=O phosphates, no P-H
```

On-disk verification of the applied repair across 315 samples: backups present
315/315, strict re-parse 315/315, maximum coordinate deviation from
`ligand_coords.npy` 5.00e-05 A, remaining P-H hydrides 0, rejected samples left
byte-identical 20/20.

Offline unit tests: 23, in `tests/test_ligand_bond_orders.py` and
`tests/test_repair_triplet_ligand_bond_orders.py`.

## Impact on the Frozen Corpus

| | Count | Share of 303 |
|---|---:|---:|
| consensus303 systems whose bond orders changed | 209 | 69.0% |
| of which **silent** (legacy setup would have succeeded) | 198 | 65.3% |
| of which legacy setup would have crashed (P-H) | 14 | 4.6% |

**Affected:** the silver MD used to generate the phase supervision labels was
run with chemically wrong ligands — wrong charges, planarity and aromaticity.

**Not affected:** apo/holo endpoints come from crystal structures and never pass
through MD; ligand heavy-atom coordinates are crystallographic; Stage-2 consumes
ligand coordinates and atom/probe types, not bond orders. Inference paths and
the strict30 evaluation geometry are unchanged.

The accurate statement is: *the MD segment used to generate the phase
supervision labels had incorrect ligand electronic structure.* This does not
invalidate endpoints or model inputs.

## Open Question

Whether the 303-system silver corpus must be regenerated with corrected ligand
chemistry is undecided. A paired A/B measurement is running to inform it
(`processed_data/md_transition/ligand_chem_ab_20260726_v1`, jobs 148807 legacy /
148808 repaired).

Design: four silent-class systems from the frozen corpus, spanning the largest
chemistry changes and drawn from four distinct deposited entries. Both arms are
shadow triplet trees whose files are symlinks, so the arms differ in exactly one
file per sample — `ligand.sdf` — and no frozen artifact is opened for writing.
Seeds and protocol are taken unchanged from the source manifest, so the arms are
paired.

| System | Component | Bonds changed | Aromatic bonds |
|---|---|---:|---|
| `2hdr-A-4A3-511` | 4A3 | 60 | 0 -> 90 |
| `4i8x-A-6P3-401` | 6P3 | 56 | 0 -> 96 |
| `4whq-F-3N8-608` | 3N8 | 51 | 0 -> 102 |
| `1cde-A-DZF-225` | DZF | 44 | 0 -> 68 |

## Scope Not Covered

The repair covered 335 samples. `processed_data/triplets/samples/` holds 91,327
sample directories with 91,189 `ligand.sdf` files, and **the 1,769-pair
expansion pool draws from those existing directories**. They still carry the
defect. The generator fix applies only to newly generated triplets, so the
expansion pool needs the same repair pass before any scaled MD campaign; the
335-sample apply took 38 seconds, so the full sweep is a short CPU job.

## Artifacts

| Path | Contents |
|---|---|
| `src/data/ligand_bond_orders.py` | CCD-templated reconstruction |
| `scripts/repair_triplet_ligand_bond_orders.py` | `prefetch` and `repair` modes; dry run by default |
| `scripts/disambiguate_ligand_component.py` | Component identity evidence |
| `scripts/build_ligand_chemistry_ab_experiment.py` | Paired A/B arm construction |
| `scripts/slurm/repair_triplet_ligand_bond_orders_cpu.sh` | `MODE=dryrun\|apply` launcher |
| `processed_data/ccd_cache/` | 493 CCD reference components |
| `logs/ligand_bond_order_repair/` | Dry-run and apply reports, per-sample ledger, ambiguity verdicts |
| `<sample>/ligand.legacy_connectivity_only.sdf` | Pre-repair SDF, preserved for every repaired sample |

Every repaired sample retains its pre-repair SDF, so the change is reversible
and the A/B comparison is reproducible.
