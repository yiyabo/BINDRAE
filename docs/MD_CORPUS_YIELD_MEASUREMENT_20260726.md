# MD Corpus Yield Measurement

## Decision

The AHoJ mapping-aware smoke panel completed end to end for the first time on
2026-07-26. Its measured conversion from endpoint pair to accepted two-replica
consensus system is **7 / 32 = 21.9%**.

`PATH3_FINAL_EXPERIMENT_CONTRACT_20260725.md` requires "at least 3,000 novel
unique pairs before claiming that 2,000 accepted consensus systems are
operationally plausible", which implies a **67%** conversion. The measured rate
is roughly **one third** of that assumption.

A separate probe established that the dominant loss is **not** a sampling-budget
artifact. It did **not** establish that the loss is physical; see the correction
below.

No frozen threshold, split, or protocol was changed to obtain these numbers.

## Correction, 2026-07-26 (same day)

This document originally concluded that "the ~22% conversion is a real property
of RMSD-pull path generation on this corpus" and that the `0.5` progress gate
"must not be relaxed to raise yield". **That conclusion is withdrawn.** It rested
on a budget probe whose three systems all turn out to carry a corrupted ligand,
which is an uncontrolled confound the original analysis did not consider.

The measured funnel below is unchanged and remains valid as a measurement of
*this pipeline*. What is withdrawn is the interpretation that the loss is
irreducible. The affected sections are marked.

## Follow-up, 2026-07-27: the confound was tested and does not explain the loss

The withdrawal above prescribed one experiment: re-run the panel on repaired
ligands. That has now been done (job 149011, output
`ahoj_mapping_smoke32x2_ligandfix_20260727_v1`) on the same frozen 32-system
panel, same seeds, same protocol, same gates. Only the ligands changed -- bond
orders for the whole panel, and the atom set for 12 of the 32.

```text
end to end     baseline  7/32 = 21.9%      repaired  9/32 = 28.1%
```

**The gain is entirely at the context stage, and the pull stage did not move.**

| Stage | Baseline | Repaired |
|---|---|---|
| Context admitted | 22/32 = 68.8% | **29/32 = 90.6%** |
| Context failures | setup 10 | setup 1, nvt 1, npt 1 |
| Replica pull passed | 18/44 = 41% | 25/58 = 43% |
| Target export passed | 16 | 23 |
| Consensus systems | 7 | 9 |

The context recovery is the bond-order repair doing exactly what was predicted:
the ten setup crashes traced to illegal phosphorus valence are down to one.

**The steric-pinning hypothesis is not supported.** The three systems used in the
budget probe were re-run with their spurious copies removed. If distant copies
were pinning the loops, progress should rise:

| System | Ligand atoms | Baseline progress | Repaired |
|---|---|---|---|
| `6z85-D-HBI-302` | 170 -> 17 | 0.489, 0.449 | **0.397, 0.424** |
| `2hiz-A-LIJ-801` | 141 -> 47 | 0.405, 0.399 | **0.377, 0.377** |
| `1lol-B-XMP-2002` | 48 -> 24 | 0.471, 0.387 | **0.569, 0.585** (passed) |

Two of three got *worse*. Removing 153 atoms from `6z85` moved its progress down
by `0.09`. And across the repaired panel the 33 pull failures still cluster in
`0.303`-`0.457`, the same band as before, with none approaching the `0.5` gate.

**Consequence: the original conclusion is restored, on stronger evidence than it
originally had.** Two independent perturbations now fail to move pull progress --
tripling the sampling budget, and deleting the spurious ligand atoms. The loss at
the pull stage is a real property of RMSD-pull path generation on this corpus.

The withdrawal was still correct as a matter of method. The confound was real and
uncontrolled, and no claim was entitled to survive it untested. It was tested and
it did not survive; that is the withdrawal working, not a reversal of it.

## What Was Measured

`processed_data/md_transition/ahoj_mapping_smoke32x2_parallel_retry1_20260726_v1/smoke_state.json`
(schema `bindrae_ahoj_mapping_smoke_gpu33_v1`):

```text
context.outcome_counts      {"admitted": 22, "setup": 10}
finalization.outcome_counts {"failed_pull": 26, "failed_target_export": 2, "target_passed": 16}
consensus.consensus_systems 7
```

The funnel:

| Stage | In | Out | Retained |
|---|---:|---:|---:|
| Context (setup, NVT, NPT) | 32 | 22 | 68.8% |
| Replica pull, per replica | 44 | 16 | 36.4% |
| Target export | 18 | 16 | 88.9% |
| Two-replica consensus, per system | 22 | 7 | 31.8% |
| **End to end** | **32** | **7** | **21.9%** |

Consensus quality on the systems that survived is good: mean phase agreement
`0.9642`, mean residual agreement `0.8263`, `min_replicas=2`,
`min_support_fraction=0.5`. The problem is quantity, not quality.

The 10 context losses are the setup failures traced to the ligand bond-order
defect; see `LIGAND_BOND_ORDER_DEFECT_20260726.md`. Those are now repairable and
should be re-measured, which is the one part of this funnel expected to improve.

## The Loss Is Not Budgetary (Whether It Is Physical Is Undetermined)

Every one of the 24 replica pull failures violated the same gate:

```text
min_target_progress_fraction = 0.5     violated by 24/24
max_final_apo_ca_rmsd_angstrom = 1.0   also violated by 5
requires_apo_closer_than_holo          also violated by 2
```

and the violations were clustered just under the threshold, in `0.386`-`0.489`,
with none near zero. That pattern is consistent with either a budget that stops
the pull short or a system that cannot get further, so it was tested directly.

Three systems spanning the band were re-pulled at 2x and 3x the pulling budget,
with the same seeds and every other protocol parameter unchanged (jobs 148866,
148867):

| System | 10,000 steps | 20,000 steps | 30,000 steps |
|---|---:|---:|---:|
| `6z85-D-HBI-302` | 0.489 | 0.485 | **0.470** |
| `1lol-B-XMP-2002` | 0.471 | 0.470 | 0.498 |
| `2hiz-A-LIJ-801` | 0.405 | 0.412 | 0.436 |

Tripling the budget moves progress by at most `+0.031`, and `6z85` moves
*backwards*. Progress has saturated: under the `200,000 kJ/mol/nm^2` restraint
the structure reaches a force balance and stops approaching the target, so
additional sampling buys nothing. **A kinetic, budget-limited explanation is
therefore excluded.** That part of the probe stands.

### The Confound: All Three Probe Systems Have A Corrupted Ligand

The original reading of the saturation was "thermodynamic limit". That inference
does not survive checking what was actually in the simulated systems:

| Probe system | Ligand fragments | Ligand atoms | Max centroid spread |
|---|---:|---:|---:|
| `6z85-D-HBI-302` | 10 | 170 | 66 A |
| `1lol-B-XMP-2002` | 2 | 48 | 32 A |
| `2hiz-A-LIJ-801` | 3 | 141 | 65 A |

All three carry the multi-copy extraction defect recorded in
`LIGAND_BOND_ORDER_DEFECT_20260726.md`: `extract_ligand_from_pdb` collects every
same-resname HETATM residue in the chain with no residue-number filter, so the
"ligand" is the true ligand plus every other copy of the same component anywhere
in that chain.

Those spurious copies are not inert annotations. They are parameterized by
OpenFF and solvated as real molecules, sitting at surface sites tens of angstroms
from the pocket. A mechanism for the observed saturation follows directly:
**the spurious copies sterically pin the loops that have to move for the
apo-to-holo transition.** The pull reaches a force balance, exactly as measured,
but part of the opposing force comes from molecules that should not be in the
system at all.

**Consequence.** The probe distinguishes "budget-limited" from "force-balanced".
It does **not** distinguish "force-balanced by protein physics" from
"force-balanced by simulation artifact", because every system tested has the
artifact. Two explanations remain live and the data does not separate them.

The `0.5` gate therefore cannot be declared well-calibrated on this evidence, and
the ~22% conversion cannot be entered into downstream plans as a fixed physical
cost. It is the conversion rate *of a pipeline with a known defect in it*.

### What Would Settle It -- and did

Re-run the panel with the spurious copies removed, same seeds and same protocol.
If progress still saturates below `0.5`, the thermodynamic reading is restored.
If it does not, a substantial part of the 78% loss was self-inflicted.

This was run on 2026-07-27. Progress still saturates; see the follow-up section
at the top of this document. The thermodynamic reading is restored.

## Consequence for the Data-Scale Gate

At the measured rate, the number of novel pairs needed for 2,000 accepted
consensus systems is:

| Assumed conversion | Pairs needed for 2,000 systems |
|---|---:|
| 67% (contract assumption) | 3,000 |
| 28% (partial-tally estimate) | ~7,100 |
| **21.9% (measured)** | **~9,100** |

The current pool is **1,769** novel leakage-clean pairs. The PR #2 rescan is
estimated to recover 1,000-2,200, reaching **2,800-4,000**.

**Updated 2026-07-27 with the repaired-pipeline rate.** The conversion to use is
**28.1%**, measured on the same frozen panel after the ligand repair. The table
above is superseded by:

| Assumed conversion | Pairs needed for 2,000 systems |
|---|---:|
| 67% (contract assumption, never measured) | 3,000 |
| 21.9% (defective pipeline) | ~9,100 |
| **28.1% (repaired pipeline)** | **~7,100** |

The current pool is **1,769** novel leakage-clean pairs, and the PR #2 rescan is
estimated to reach **2,800-4,000**.

**2,000 accepted consensus systems remains out of reach, by roughly a factor of
1.8.** The repair moved the requirement from 9,100 pairs to 7,100 against a
ceiling of 4,000; it did not change the verdict. Reachable scale from a fully
rescanned pool is on the order of **800-1,100 accepted systems**.

This arithmetic no longer inherits the confound: `28.1%` is the conversion of a
pipeline whose ligand chemistry is correct throughout and whose ligand atom sets
are correct for every panel system that needed it.

The contract's **67%** was never measured and is not supported by anything. It
can now be replaced with a measured figure rather than merely deleted.

## Claim Boundary

The 32-system panel was **deliberately selected for difficulty**: its entry
condition is that apo and holo residue counts differ, because the whole point of
the mapping-aware route is to handle the mismatched-residue case that the old
equal-CA-count gate discarded. The repository's own wording for the analogous
APObind result applies here: this is a *stratified engineering smoke result, not
an unbiased estimate*.

So 21.9% is most defensible as a **lower bound** on an unbiased corpus, and the
true population rate is likely higher by an unknown amount.

Two supports were originally offered for treating the number as nonetheless
actionable. Only one survives:

1. **Stands, with a narrower reading.** The replica-level pull pass rate measured
   here, `11/27 = 41%` at the point of the partial tally, reproduces the `44%`
   borrowed from the APObind pilot, which was previously the weakest link in the
   arithmetic. But both runs used the same defective ligand extraction, so the
   agreement demonstrates that the number is *reproducible*, not that it is
   *correct*. Two measurements of the same broken pipeline agree with each other.
2. **Withdrawn.** "The budget probe shows the dominant loss is physical, so it
   does not shrink with effort or engineering." The probe cannot show this; see
   the correction above. Whether the loss shrinks with engineering is precisely
   the open question, and the engineering in question is repairing ligand
   extraction.

An unbiased estimate requires a randomly drawn panel on a repaired pipeline.
Neither the random draw nor the repair has been done.

## What Follows

Reordered by the correction above. Repairing extraction now precedes every
measurement, because every number in this document is a measurement of the
unrepaired pipeline.

1. **Repair ligand extraction.** Filter `extract_ligand_from_pdb` by residue
   number, keeping genuinely covalently linked partners. Until this is done, no
   yield number measured on this corpus can be interpreted.
2. **Re-run the budget probe on single-copy systems.** Same seeds, same protocol,
   `component_copies == 1` only. This is the experiment that separates protein
   physics from simulation artifact, and it is cheap.
3. **Re-measure the whole funnel** on the repaired pipeline. Both the 10 context
   losses (bond orders, cause known and fixed) and an unknown share of the 24 pull
   losses (extraction, cause known and not yet fixed) are expected to move.
4. **Only then re-baseline the data-scale gate** in the Path-3 contract. The
   `3,000` figure has no empirical basis, but neither does any replacement until
   step 3 produces a conversion rate from a pipeline without known defects.
5. Treat "2,000 accepted consensus systems" as an open planning question. It may
   be unreachable, as this document originally argued, or the argument may have
   been measuring a bug. Step 3 decides which.

## Artifacts

| Path | Contents |
|---|---|
| `processed_data/md_transition/ahoj_mapping_smoke32x2_parallel_retry1_20260726_v1/smoke_state.json` | The three headline counts |
| `.../context/collection/summary.json` | Context stage outcomes |
| `.../replicas/finalization/summary.json` | Replica and target-export outcomes |
| `.../replicas/pulls/*/replica_*/rmsd_pull_report.json` | Per-replica gates and final metrics |
| `processed_data/md_transition/pull_budget_probe_20260726_v1/` | 2x and 3x budget probe, jobs 148866 and 148867 |
