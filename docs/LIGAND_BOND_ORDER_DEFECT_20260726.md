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
chemistry is **still undecided**. Two paired A/B measurements were run to inform
it and neither settles it; the n = 15 replication is both statistically
inconclusive and confounded by the extraction defect, as recorded at the end of
this section. The first measurement is
`processed_data/md_transition/ligand_chem_ab_20260726_v1` (jobs 148807 legacy /
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

### First result, n = 4

Both arms completed. Per system, replica 00:

| System | legacy | repaired |
|---|---|---|
| `2hdr-A-4A3-511` | prog 0.421, ligand RMSD **3.23 A**, fail | prog 0.376, ligand RMSD **0.89 A**, fail |
| `4i8x-A-6P3-401` | prog 0.591, pass | prog 0.597, pass |
| `4whq-F-3N8-608` | prog 0.677, pass | prog 0.542, pass |
| `1cde-A-DZF-225` | prog 0.544, pass | prog 0.477, fail |

```text
legacy    target_passed 6, failed_pull 2
repaired  target_passed 3, failed_pull 3, failed_target_export 2
```

Three readings, kept separate by how well each is supported:

1. **The chemistry changes the paths.** Same system, same seed, one file
   different, and progress moves by up to `0.135`. Corrected chemistry produces
   *different* silver paths, so the question of regenerating the corpus is real
   rather than academic.
2. **The repaired ligand behaves more physically.** The cleanest signal is
   `2hdr`: ligand heavy-atom RMSD falls from `3.23 A` to `0.89 A`. The legacy
   ligand, with zero aromatic bonds, was a floppy saturated molecule wandering
   the pocket; with 90 aromatic bonds the fused ring system is rigid and planar
   and stays put. This is the repair working as intended.
3. **The repaired arm passed fewer gates (3 vs 6). No conclusion is drawn.**
   n = 4, and the transitions sit near the `0.5` boundary where a small shift
   flips the verdict (`1cde`: 0.544 to 0.477). A rigid ligand plausibly makes
   the protein harder to pull, but that mechanism is unverified.

A 16-system replication is running (jobs 148893 legacy, 148894 repaired) because
n = 4 is too thin to decide whether to regenerate 303 systems of silver MD.

### Replication at n = 15: inconclusive, and confounded

The replication completed
(`processed_data/md_transition/ligand_chem_ab16_20260726_v1`). One system,
`2hdr-A-4A3-511`, has no legacy arm -- context task `148896_0` FAILED -- so 15 of
16 systems are paired.

```text
replicas   legacy 23/30      repaired 20/30
systems    legacy 13/15      repaired 11/15
```

**The pass-rate difference is not distinguishable from noise.** Only five systems
differ between the arms: `4b3u`, `6ekz`, `7jy1` and `7k8h` favour legacy and
`4qpd` favours repaired. A two-sided sign test on those five discordant pairs
gives `p = 0.375`. The `3 vs 6` split at n = 4 did not survive replication as a
signal; it survived only as a direction.

**The direction is nonetheless mechanistically coherent, and the ligand metric
points the other way.** Final ligand heavy-atom RMSD improves in 9 of 15 systems
and worsens in 6:

```text
improved  4b3u 0.95->0.41   5ibg 1.61->0.90   6ekz 1.69->1.03
          7k8h 1.29->1.04   1cde 2.05->1.33   and four smaller
worsened  4whq 0.99->1.96   4yu7 1.39->1.81   and four smaller
```

So corrected chemistry tends to place the ligand better while passing the pull
gate slightly less often, which is what a stiffer, more constrained ligand would
do: `target_progress_fraction` measures how far the *protein* moved, and a floppy
saturated ligand obstructs it less. Passing the gate more often is not the same
as producing a more physical path. 9-versus-6 is not a result either.

**The experiment is also confounded, by a defect discovered after it was
designed.** At least 10 of the 15 paired systems -- `1cde`, `1l2s`, `2c4j`,
`4b3u`, `4i8x`, `4qpd`, `4yu7`, `5ibg`, `6eg7`, `7k8h`, plus the unpaired `2hdr`
-- carry the multi-copy extraction defect recorded below. **Both arms therefore
hold the wrong atoms**, and the comparison is between two chemistry versions of a
molecule that should not be there. `4i8x` has 8 copies and `6eg7` has 8; `2hdr`
has 15.

**Verdict: the A/B does not answer whether the 303-system corpus must be
regenerated, and cannot be salvaged by adding systems.** Its ligands need the
extraction repair first. The cleaner successor is the 32-system smoke panel
re-run on repaired ligands (job 149011, output
`ahoj_mapping_smoke32x2_ligandfix_20260727_v1`), which corrects the atom set as
well as the chemistry and has a matched 21.9% baseline to compare against.

Recorded so the same confounded comparison is not rebuilt later.

## Full-Corpus Scan

The 335-sample apply was followed by a dry run over the whole corpus, because
the 1,769-pair expansion pool draws from these same directories and the
generator fix only helps newly generated triplets. 91,189 samples in 13 minutes
at 48 workers:

```text
would_repair                    82,485   (90.5%)
rejected                         8,704   ( 9.5%)
samples with bond-order changes 70,913   (77.8%)
bond orders changed          1,216,111
defect present before           23,194
resname_chemistry_ambiguous          0
```

**77.8% of the corpus carries wrong ligand chemistry**, confirming at scale the
72% seen in the frozen 303. The tiered resname resolution eliminated chemistry
ambiguity entirely at this scale.

Reaching that scale required three fixes to the tool itself, all of which are
invisible at 335 samples and are now in the code:

| Problem | Symptom at 90k | Fix |
|---|---|---|
| Enumeration cost | Hung before emitting a line | `os.scandir` and resname-from-directory-name; 3-4 metadata round trips per sample became ~1, and only 2 of 91,189 samples needed `meta.json` opened |
| Serial CCD download | 21,048 components at 0.4/s = 14.6 h | 16-thread pool, 7/s, completed in ~50 min with 0 failures |
| Cache torn by concurrent writers | Corrupt template on a race | Write to a temp file and `os.replace` |

### Metal coordination

The largest new reject class was `sanitize_failed_after_reconstruction`
(1,003, of which **944 are HEM**), failing on nitrogen valence. Copying CCD bond
orders onto a porphyrin leaves each pyrrole nitrogen with its two ring bonds
plus a bond to iron, which exceeds neutral nitrogen's valence under RDKit's
default model.

The fix re-types ligand-to-metal bonds as **dative**, which contributes nothing
to the donor's valence and is the correct model for coordination. A dative bond
must run donor to metal and RDKit bonds cannot be re-oriented in place, so they
are removed and re-added. Verified on 40 real failing samples: **40/40 now
repair**. It also recovered the two largest `ccd_template_parse_failed` groups,
`ICS` (Fe/Mo cofactor) and `OEX` (Mn/Ca oxygen-evolving cluster), whose CCD
reference files RDKit could not parse at all.

Losing all heme systems would have been a systematic coverage gap over
haemoglobin, cytochromes and peroxidases -- the same class of bias the
nucleotide cofactors would have caused.

## A Second, Deeper Defect: Wrong Ligand Extracted

The full-corpus scan surfaced a defect that is **independent of bond orders and
more fundamental**. The largest reject class, `substructure_match_failed`
(3,648), decomposes by what the observed graph actually contains:

```text
observed > template  2,543     ratios cluster at 1.8, 1.9, 2.8, 3.8, 4.7
equal                1,043
observed < template     62

top resnames: GLC(536) BGC(387) XYP(202) MAN(132)   sugars
              GLU(232) LYS(129) GLY(93) PRO(67)     standard amino acids
              CU1(251) AU(55)                       metal ions
```

Three distinct causes:

1. **Oligosaccharides (~1,250).** Non-integer ratios are the glycosidic
   signature `n x template - (n-1)`; two linked GLC give `2x12-1 = 23`, ratio
   1.9. Known boundary, described above.
2. **Standard amino acids (~700).** `GLU`, `LYS`, `GLY`, `PRO`, `PHE`, `ALA`
   are not ligands. `extract_ligand_from_pdb` collected peptide residues, and
   the 1.8/2.8 ratios show it collected *several linked residues* at once.
3. **Metal ions (~300).** `CU1`, `AU` are single-atom ions sharing the file
   with, or standing in for, the organic ligand.

For these samples `ligand.sdf` and `ligand_coords.npy` do not hold the ligand.
That means **the model's ligand conditioning input is wrong for them**, not
merely the MD chemistry -- a strictly larger blast radius than the bond-order
defect. Stage-2 consumes those coordinates directly.

This is not repaired here. It requires rewriting the extraction in
`prepare_ahojdb_triplets.py` to filter by residue identity and select a single
ligand residue, which changes `ligand_coords.npy` for the affected samples and
is a separate data change needing its own evaluation.

### The Defect Is Not Confined To The Reject Bucket

The section above located this defect inside `substructure_match_failed`, which
made it look like a 3,648-sample problem that the repair already refuses to touch.
**That framing was wrong and badly understated the scope.**

Querying `reconstruction.component_copies` across the whole v2 dry-run ledger:

```text
records with component_copies > 1     41,599 / 91,189  = 45.6%
```

These are samples where matching **succeeded**. The reconstruction found the CCD
template N separate times, one per disconnected fragment, reported
`component_copies = N`, and returned `status = would_repair`. They are not
rejected and they carry no warning. The bond-order repair would write a
chemically valid SDF for all N copies and move on.

So the reject bucket was never the population. It was the small subset where the
extra copies happened to be *linked* and so broke element-count matching. The
larger, silent subset is where they are *unlinked*, and those pass cleanly.

Top components by affected sample count:

```text
ADP 614  PO4 582  ATP 473  FE2 441  GAL 434  NAD 377  AMP 300  SO4 280
NAP 252  ANP 243  GLC 237  NAI 236  BGC 234  NDP 194  COA 188  GNP 176
```

`PO4`, `SO4` and `FE2` are crystallization additives and ions rather than
biological ligands, so for those samples the extracted "ligand" is wrong twice
over: multiple copies, of the wrong molecule.

### Confirmation That These Are Defects, Not Real Multi-Copy Binding

`component_copies > 1` has a legitimate reading: a pocket really can bind two
copies of a component. The discriminator is spatial. Real multi-copy binding is
clustered in one site; the extraction defect scatters copies across the chain.

Twenty affected samples were checked directly against `ligand.sdf` coordinates,
computing per-fragment centroids and the maximum pairwise centroid distance:

```text
max centroid spread > 15 A            20 / 20
median spread                         72.1 A
maximum spread                        195.0 A   (3vt2-A-IPT-601, 7 copies)
ligand_coords.npy atom count == SDF   20 / 20
```

Worked examples:

| Sample | Copies | Atoms in `.npy` | Template atoms | Spread |
|---|---:|---:|---:|---:|
| `2hdr-A-4A3-506` | 15 | 165 | 11 | 83.9 A |
| `1gu1-J-FA1-201` | 12 | 144 | 12 | 53.2 A |
| `4i8x-A-6P3-401` | 8 | 120 | 15 | 110.5 A |
| `3vt2-A-IPT-601` | 7 | 105 | 15 | 195.0 A |

No pocket is 195 A across. These are not multi-copy binding sites. And because
the `.npy` atom count equals the SDF atom count in every case, **Stage-2 receives
all of the scattered copies as its ligand conditioning input.**

### The Frozen Training Corpus Is Affected

Cross-referencing against the active Stage-2 split
(`stage2_oracle_motion_mdphase_consensus266_fam30scaf_blockv2`, seed 20260718):

| Split | Multi-copy systems | Share |
|---|---:|---:|
| train | 61 | ~28% |
| validation | 9 | ~35% |

The ID harvest recovered 220 train and 26 validation identifiers from the split
summaries, against filenames that say `train_208` and `val_24`, so the
denominators carry roughly +-12 of uncertainty. The share is ~28% either way.

Unlike the bond-order defect, this one is **silent**: training proceeds normally.
Roughly one system in four is simply conditioned on the wrong molecular input.

### Two Downstream Consequences

**1. An alternative explanation for the Stage-1 gate collapse.** A recorded
negative result holds that the ligand gate collapsed to `1.5e-5` and the ligand
residual contributed exactly zero, attributed at the time to the base-prior loss
overwhelming the gate gradient. If ~28% of ligand inputs are fragments scattered
up to 195 A from the pocket, a model that learns to zero the ligand pathway is
behaving correctly. This is a **live alternative hypothesis, not a demonstrated
cause**: it has not been shown that Stage-1 used an affected corpus, nor that
repairing extraction revives the gate. It is now testable, where it was
previously closed.

**2. It confounds the corpus-yield conclusion.** All three systems in the
budget probe that was used to argue the MD pull loss is physical carry this
defect, with 2, 3 and 10 copies spread 32-66 A. Spurious copies are parameterized
and solvated as real molecules and can sterically pin the loops that must move.
The withdrawal is recorded in `MD_CORPUS_YIELD_MEASUREMENT_20260726.md`.

### The Frozen Corpus Was Repaired, 2026-07-27

`scripts/repair_triplet_ligand_extraction.py` was applied to the same 335 samples
the bond-order repair covered (the frozen consensus corpus plus the smoke panel).
Report: `logs/ligand_extraction_repair/corpus303_apply_v1`.

```text
repaired                        99
single_fragment_not_in_scope   231
rejected                         5
ligand atoms  6,516 -> 1,819   (72% discarded)
```

Verification on disk, all 99: atom count matches the selection, `ligand_coords.npy`
and the SDF conformer agree to `5.00e-05 A`, both backups present, SDF re-parses.
Copies removed: median 3x, maximum 17x. The weakest identification still cleared
the gate with an 11x runner-up margin against a 10x requirement, and the largest
match deviation was `0.000127 A`.

**Subsetting preserved the bond-order repair**, which was the risk worth checking:
`1cde-A-DZF-225` went from 92 single and 44 double bonds across four copies to 23
and 11 for one, exactly a quarter of each.

A second bond-order pass over the repaired corpus is **not** needed. Re-running it
reports 159 samples with "changed" bonds, but on samples the extraction repair
never touched the before and after distributions are identical -- `1alw-A-ISA-11`
is `AROMATIC 6, DOUBLE 1, SINGLE 6` on both sides yet counts 6 changes. The metric
counts each aromatic bond as changed on a second pass, an artifact of kekule
versus aromatic representation round-tripping, not work to do. The seven samples
flagged `defect_present_before` are all sulfur-bearing (`GSH` twice, `ISA`, `SGC`,
`GTM`, `3SU`, `YIO`) and are the documented false positives of the element-based
sentinel indicator, which this module explicitly does not treat as a defect on its
own.

The five rejects fall in two classes, both left untouched:

| Sample | Atoms | Why |
|---|---:|---|
| `1ivc-A-ST2-471` | 30 | `copy_match_ambiguous`: two geometrically identical copies |
| `1ivd-A-ST1-471` | 34 | same |
| `5gnw-C-URA-301` | 64 | same, eight identical uracils |
| `1xnk-C-TWY-3` | 36 | no reference reproduced it; oligosaccharide boundary |
| `3azt-E-BGC-2` | 66 | same |

The ambiguous class is the one case where the alignment matrices would help:
identical conformers cannot be separated by internal geometry, only by position.
That path is available and was not needed for the other 99.

### The 326-System Corpus Is Being Regenerated, 2026-07-27

The open question above -- whether the silver corpus must be regenerated -- was
resolved by decision rather than by measurement, because neither A/B could answer
it and the measurement that would (path divergence between old and new ligands)
costs about as much as simply regenerating.

**Decision and its cost.** Regenerate all 326 systems on repaired ligands. The
corpus is expected to *shrink*, not grow: the A/B's direction, though not
significant, was that correct chemistry passes the pull gate slightly less often,
and a stiffer correctly-typed ligand obstructing the protein is a coherent
mechanism. At the A/B's -15% the corpus would land near 275.

That shrinkage is a correction, not a loss. A system that passes the gate only
because its ligand was wrong is a simulation of a molecule that is not there, and
it cannot support a ligand-conditioned induced-fit claim.

**The decision is reversible.** Output goes to a new root
(`processed_data/md_transition/corpus326_ligandfix_20260727_v2`); the existing
corpus's MD products are untouched, so both can be compared before either is
trained on. Only the *inputs* changed in place, and those have backups.

**Configuration, and why each value.**

| Setting | Value | Reason |
|---|---|---|
| Replicas | `0-4`, five attempted | The original corpus attempted five; `n_replicas` in its manifest is how many *passed*. Using the smoke panel's two would unfairly depress the new corpus |
| Context seed base | `60719000` | The original corpus's |
| Replica seed base | `60720000` | The original corpus's |
| Gates | unchanged | `progress >= 0.5`, mapping `0.95`, two-replica consensus |
| Manifest | real records, four sources merged | See below |

**Caveat that matters for any later comparison: the seeds are not paired
per system.** `build_md_context_matrix.py` assigns `seed = seed_base +
candidate_index`, and `candidate_index` follows manifest order. The regeneration
manifest is ordered by the system list; whether the original used the same order
is not established. The seed *bases* match, the per-system seeds may not.

This is harmless for the corpus-level question -- how many systems survive -- and
it is **not** harmless for a per-system path-divergence comparison, where a
different seed produces a different path even with an identical ligand. Anyone
measuring old-versus-new path RMSD must establish seed correspondence first, or
the ligand effect and the seed effect are confounded.

**A failed first attempt, recorded so it is not repeated.** Job 149163 was
launched with a hand-built minimal manifest carrying only the three fields
`build_md_context_matrix.py` reads: `transition_id`, `system_sample_id` and
`endpoints`. Nineteen of the first twenty-five tasks failed -- but at
`register_context`, not in the MD. Setup, NVT and NPT all returned 0. The
downstream `register_md_context_replica.py` validates against the full
`bindrae_md_transition_v1` schema and rejected the records for missing
`ensemble_id`, `source`, `split.name` and `quality.endpoint_mapping_verified`,
plus a wrong `schema_version` value. The array was cancelled, costing roughly
eight CPU-hours of MD that ran correctly and was then discarded.

The fix was **not** to hand-fill the missing fields.
`quality.endpoint_mapping_verified` is a claim that the endpoint residue mapping
was verified, and asserting it without performing the verification would be
fabricating provenance. The original transition manifests were located instead;
four of them merged cover all 326 systems with their real acquisition values
(`residue_mapping_fraction 0.967`, `status: metadata_verified`). The relaunch
(job 149209 -> 149210) uses those records unmodified.

**Operational lesson.** `scripts/audit_md_transition_manifest.py` already exists
and reports `num_errors` per manifest. Run it before submitting any corpus-scale
job. On the repaired manifest it returns `326 records, 0 errors, 0 warnings`; run
on the first attempt it would have caught the defect before 326 tasks were queued.

**What to read when it finishes.** Consensus and `smoke_state.json` are not
produced automatically; run `scripts/slurm/consolidate_ahoj_smoke_cpu.sh` against
the output root first. Then compare `consensus.consensus_systems` against the
existing corpus's **303 of 326**.

### Root Cause Is Shared With The Bond-Order Defect

These are not independent bugs. Both, along with `observed_sanitize_failed` (922,
over-valent oxygen and chlorine) and `ligand_coords_shape_mismatch` (64), follow
from one decision: the pipeline reads chemistry out of a coordinate file.
`Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)` infers bonds by
proximity and respects no residue boundary, and PDB carries no bond orders or
formal charges to begin with. Missing bond orders, collected same-resname
residues, distance-invented bonds and atom-count drift are four symptoms of
treating the ligand as coordinates rather than as a chemical object.

Ordering follows from this: **extraction must be fixed before bond orders are
applied at corpus scale**, because bond-order repair on a wrong atom set produces
a chemically valid molecule that is still the wrong molecule.

`ligand_coords_shape_mismatch` (64) is a third, unrelated pre-existing
inconsistency: the SDF and the `.npy` disagree on atom count. The repair
correctly refuses those rather than writing a mismatched pair.

### Reject classes not resolved

| Class | Count | Status |
|---|---:|---|
| `substructure_match_failed` | 3,648 | Wrong ligand extracted; see above |
| `observed_heavy_atom_fraction_below_threshold` | 2,179 | Crystal observed too little of the component; correct rejection |
| `observed_sanitize_failed` | 941 | The observed connectivity itself is illegal (oxygen with 3 bonds, chlorine with 1 that still over-valences). Proximity bond perception produced a wrong graph, which CCD reconstruction cannot repair because it presupposes correct connectivity |
| `unobserved_neighbor_on_sentinel_element` | 573 | Partially observed phosphate; correct rejection |
| `ccd_template_parse_failed` | 294 | Reduced by metal handling; 28 of 40 retested still fail for other reasons |
| `ligand_coords_shape_mismatch` | 64 | Pre-existing data inconsistency |

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
