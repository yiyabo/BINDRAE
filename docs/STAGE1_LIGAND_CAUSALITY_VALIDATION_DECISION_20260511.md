# Stage-1 Ligand-Causality Validation Decision (2026-05-11)

## Decision

The validation run supports the same practical conclusion as the earlier
candidate-reranking review:

- Stage-1 candidate-reranking is not a paper-ready ligand-causal solution.
- The current translated-rerank checkpoint shows a strong candidate-level
  proximity/occupancy signal.
- The same checkpoint does not show stable chemistry/identity lift against
  no-ligand, scrambled-types, or batch-shuffled-ligand controls.
- The next Stage-1 direction should move to typed candidate
  rotamer-ligand interaction compatibility energy, not another continuation of
  the existing candidate-rerank objective.

This decision uses validation outputs generated on the validation split only.
No test split was used.

## Validation Runs

Dataset/contact audit:

- Job: `129140`
- Status: `COMPLETED`
- Exit code: `0:0`
- Runtime: `00:08:52`
- Output:
  `logs/stage1_diagnostics/ligcausal_validation_audit_20260511/ligand_sensitive_dataset_audit.json`

Full checkpoint diagnostics:

| checkpoint role | job | status | runtime | output tag |
| --- | ---: | --- | ---: | --- |
| baseline posterior/candidate | `129144` | `COMPLETED 0:0` | `00:15:38` | `ligcausal_validation_diag_baseline_postcand_20260511` |
| translated-rerank best | `129145` | `COMPLETED 0:0` | `00:15:43` | `ligcausal_validation_diag_translated_rerank_20260511` |
| shuffled-rerank best | `129146` | `COMPLETED 0:0` | `00:15:32` | `ligcausal_validation_diag_shuffled_rerank_20260511` |

Final summary report:

- `logs/stage1_diagnostics/ligcausal_validation_report_20260511/causality_validation_report.json`
- Recommendation after report fix:
  `proceed_to_typed_candidate_rotamer_ligand_interaction_energy`
- Failure pattern validated: `true`
- Paper-ready ligand-causal signal found: `false`

## Label-Layer Result

The label layer passed.

| audit item | value |
| --- | ---: |
| validation samples | `2814` |
| chi1-valid residues | `842429` |
| switch residues | `308969` |
| CA-contact switch residues | `26670` |
| pocket-switch residues | `13182` |
| raw atom14 availability | `0.0` |
| FK-derived atom14 available samples | `2768` |
| FK-derived atom14 availability fraction | `0.983653` |
| FK atom14 contacts at 4.5 A | `35780` |
| CA/FK contact overlap at 4.5 A | `12782` |
| translated-away FK contact fraction | `0.0` |

Interpretation:

- Raw `atom14_available=0` is a raw-data limitation, not a blocker for
  contact supervision.
- FK-derived atom14 contact is available for nearly all validation samples.
- CA-contact and FK atom14-contact overlap is non-empty and stable.
- The translated-away control correctly removes local contact.
- `batch_shuffled_ligand` remains a cross-sample ligand control, not a
  same-pocket chemical decoy.

## Diagnostic Result

The key diagnostic pattern is strongest in the translated-rerank checkpoint.
Its candidate-level translated-away lift is large, but stricter chemistry
controls remain near zero or unstable.

### Baseline checkpoint

Switch chi1 lift:

| control | correct - control |
| --- | ---: |
| no ligand | `+0.000036` |
| translated away | `+0.003059` |
| scrambled types | `+0.000110` |
| batch shuffled ligand | `+0.000422` |

Interpretation: no useful ligand-causal signal.

### Translated-rerank checkpoint

Switch chi1 lift:

| control | correct - control |
| --- | ---: |
| no ligand | `-0.000474` |
| translated away | `+0.006650` |
| scrambled types | `+0.000032` |
| batch shuffled ligand | `+0.000354` |

Candidate rotamer lift:

| subset | correct - translated away |
| --- | ---: |
| contact | `+0.132485` |
| contact-switch | `+0.065926` |
| pocket-switch | `+0.064978` |
| switch | `+0.071474` |

Strict candidate controls for the same checkpoint:

| control | contact | contact-switch | pocket-switch | switch |
| --- | ---: | ---: | ---: | ---: |
| no ligand | `+0.000946` | `-0.007227` | `-0.007837` | `-0.002390` |
| scrambled types | `+0.002037` | `+0.000705` | `-0.001902` | `-0.000068` |
| batch shuffled ligand | `+0.000218` | `+0.001410` | `+0.000000` | `+0.001098` |

Interpretation: strong proximity/occupancy signal, no stable chemistry signal.

### Shuffled-rerank checkpoint

Switch chi1 lift:

| control | correct - control |
| --- | ---: |
| no ligand | `+0.000062` |
| translated away | `+0.001555` |
| scrambled types | `-0.000383` |
| batch shuffled ligand | `+0.000283` |

Candidate rotamer lift:

| control | contact | contact-switch | pocket-switch | switch |
| --- | ---: | ---: | ---: | ---: |
| no ligand | `-0.000146` | `+0.000176` | `-0.000380` | `-0.001646` |
| translated away | `-0.011350` | `+0.001763` | `-0.000761` | `+0.009160` |
| scrambled types | `-0.002401` | `-0.002468` | `+0.000913` | `+0.000831` |
| batch shuffled ligand | `+0.000946` | `+0.006698` | `+0.001902` | `+0.001029` |

Interpretation: shuffled-rerank does not rescue ligand identity learning.

## Consequence

The current evidence is enough to stop extending candidate-rerank as the main
scientific route. The translated-rerank lane is useful as a diagnostic and as a
source of proximity-sensitive machinery, but it should not be promoted as
ligand-causal chemistry.

Stage-2 should continue to treat Stage-1 guidance as optional soft evidence
with zero-prior fallback until strict chemistry controls pass.

## Next Direction

Proceed to:

`typed_candidate_rotamer_ligand_interaction_compatibility_energy`

The next objective must make ligand atom/probe type information part of the
candidate rotamer score itself. A new model should be judged by lift over
no-ligand, scrambled-types, and batch-shuffled-ligand controls on
contact-switch and pocket-switch residues, not by translated-away lift alone.
