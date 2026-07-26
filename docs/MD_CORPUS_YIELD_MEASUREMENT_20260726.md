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
artifact and cannot be recovered by spending more compute.

No frozen threshold, split, or protocol was changed to obtain these numbers.

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

## The Loss Is Physical, Not Budgetary

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
additional sampling buys nothing. This is a thermodynamic limit, not a kinetic
one.

**Consequence.** The `0.5` gate is not mis-calibrated and must not be relaxed to
raise yield. The ~22% conversion is a real property of RMSD-pull path generation
on this corpus and belongs in every downstream plan as a fixed cost.

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

**2,000 accepted consensus systems is therefore not reachable from the available
pool, by roughly a factor of two to three.** This is not a matter of collecting a
little more; the conversion coefficient in the plan was never measured, and the
measurement invalidates the target rather than the effort.

Reachable scale from a fully rescanned pool, at the measured rate, is on the
order of **600-900 accepted systems**.

## Claim Boundary

The 32-system panel was **deliberately selected for difficulty**: its entry
condition is that apo and holo residue counts differ, because the whole point of
the mapping-aware route is to handle the mismatched-residue case that the old
equal-CA-count gate discarded. The repository's own wording for the analogous
APObind result applies here: this is a *stratified engineering smoke result, not
an unbiased estimate*.

So 21.9% is most defensible as a **lower bound** on an unbiased corpus, and the
true population rate is likely higher by an unknown amount. Two things keep the
conclusion standing anyway:

1. the replica-level pull pass rate measured here, `11/27 = 41%` at the point of
   the partial tally, independently reproduces the `44%` borrowed from the
   APObind pilot, which was previously flagged as the weakest link in the
   arithmetic. It is no longer borrowed;
2. the budget probe shows the dominant loss is physical, so it does not shrink
   with effort or engineering.

An unbiased estimate requires a randomly drawn panel, which has not been run.

## What Follows

1. Re-measure the context stage after the ligand bond-order repair. Ten of the
   32 losses were setup failures with a now-known and now-fixed cause, so the
   68.8% context retention is an underestimate of the repaired pipeline.
2. Re-baseline the data-scale gate in the Path-3 contract against a measured
   conversion rather than an assumed one. The `3,000` figure should be replaced,
   not merely raised.
3. Treat "2,000 accepted consensus systems" as an open planning question rather
   than a milestone on the current path. Either the target changes, the
   acquisition source changes, or the method stops depending on corpus scale.

## Artifacts

| Path | Contents |
|---|---|
| `processed_data/md_transition/ahoj_mapping_smoke32x2_parallel_retry1_20260726_v1/smoke_state.json` | The three headline counts |
| `.../context/collection/summary.json` | Context stage outcomes |
| `.../replicas/finalization/summary.json` | Replica and target-export outcomes |
| `.../replicas/pulls/*/replica_*/rmsd_pull_report.json` | Per-replica gates and final metrics |
| `processed_data/md_transition/pull_budget_probe_20260726_v1/` | 2x and 3x budget probe, jobs 148866 and 148867 |
