# Stage-1 Rotamer Oracle Data Audit

Date: 2026-06-17

## Purpose

Recent Stage-1 ligand-causality runs failed to produce validation lift over
no-ligand / decoy-ligand controls. This audit checks whether the validation
data itself contains a learnable ligand-geometric signal for chi1 endpoint
supervision.

The question is not whether a neural model can fit the labels. The question is
whether correct known-pose ligand geometry makes the holo chi1 rotamer
systematically better than apo or decoy controls on contact/switch residues.

## Runs

### Rotamer Oracle Signal

- Slurm job: `134492`
- Output:
  `logs/stage1_diagnostics/rotamer_oracle_signal_20260617_092710/rotamer_oracle_signal.json`
- Script: `scripts/audit_rotamer_oracle_signal.py`
- Launcher: `scripts/slurm/audit_rotamer_oracle_signal_2gpu.sh`

This run uses apo-backbone FK to generate the three canonical chi1 rotamer
candidates and scores each candidate with simple ligand geometry energies. It
then asks whether the holo bin ranks better with the correct ligand than with
translated or batch-shuffled ligand controls.

Key validation counts:

- `contact_switch_count`: 11,892 residues
- `pocket_switch_count`: 20,065 residues
- `switch_count`: 253,103 residues
- `contact_count`: 30,243 residues

Key results on contact-switch residues:

| Score | Correct holo-vs-other margin | Strict win rate | Lift vs translated | Lift vs shuffled |
|---|---:|---:|---:|---:|
| `shell35` | -0.2184 | 0.3283 | -0.2184 | -0.1624 |
| `shell45` | -0.2308 | 0.3258 | -0.2308 | -0.1693 |
| `contact` | -0.1365 | 0.3517 | -0.1365 | -0.0997 |
| `closest` | -0.0385 | 0.3512 | -0.0385 | -0.0281 |
| `anticlash` | -0.0228 | 0.0508 | -0.0228 | -0.0179 |

Key results on pocket-switch residues:

| Score | Correct holo-vs-other margin | Strict win rate | Lift vs translated | Lift vs shuffled |
|---|---:|---:|---:|---:|
| `shell35` | -0.1709 | 0.3141 | -0.1709 | -0.1248 |
| `contact` | -0.1235 | 0.3439 | -0.1235 | -0.0896 |
| `closest` | -0.0315 | 0.3433 | -0.0315 | -0.0224 |

Interpretation: simple ligand geometry does not rank the holo chi1 bin above
the other two canonical rotamers. The margins are negative on the exact subsets
where ligand-causal learning was expected to appear.

### Apo-vs-Holo Ligand Geometry

- Slurm job: `134500`
- Output:
  `logs/stage1_diagnostics/apo_holo_ligand_geometry_20260617_094059/apo_holo_ligand_geometry.json`
- Script: `scripts/audit_apo_holo_ligand_geometry.py`
- Launcher: `scripts/slurm/audit_apo_holo_ligand_geometry_2gpu.sh`

This run reconstructs apo and holo sidechains from stored torsions on the apo
backbone frame and directly compares their minimum sidechain-ligand distances.

Key results:

| Subset | Count | Holo - Apo min-dist mean | Holo closer rate | Clash rescue | Clash harm | Contact gain | Contact loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| contact-switch | 11,892 | +0.0202 A | 0.5025 | 0.1195 | 0.1231 | 0.1150 | 0.1391 |
| pocket-switch | 20,065 | +0.0482 A | 0.4980 | 0.0811 | 0.0773 | 0.1056 | 0.1342 |
| switch | 253,103 | -0.0120 A | 0.5019 | 0.0065 | 0.0062 | 0.0100 | 0.0131 |
| contact | 30,243 | +0.0148 A | 0.4997 | 0.0655 | 0.0635 | 0.0590 | 0.0664 |

Interpretation: holo chi1 labels are not systematically closer to the ligand
than apo chi1 labels on ligand-facing switch residues. Contact loss exceeds
contact gain on contact-switch and pocket-switch residues.

## Conclusion

The current Stage-1 chi1 endpoint formulation is not just hard for the model.
The validation labels do not expose a strong, monotonic known-pose ligand
geometry signal at the canonical chi1-bin level.

This explains why multiple training objectives can fit auxiliary losses while
failing correct-ligand lift: the supervision is dominated by base rotamer
statistics and noisy apo/holo differences, not by a clean ligand-causal
rotamer-selection target.

## Recommended Next Direction

Do not keep escalating deterministic chi1 endpoint training as the primary
Stage-1 objective.

A more defensible path is:

1. Use Stage-1 as a soft local posterior / proximity prior rather than a
   deterministic holo endpoint.
2. For ligand causality, train on pairwise residue-ligand interaction labels
   that are directly tied to geometry: contact persistence, clash relief,
   gained/lost local interactions, or ligand-facing residue masks.
3. Move conformational change generation into Stage-2, where apo-to-holo paths
   can model coupled changes and be evaluated with path/contact geometry.
4. If chi1 supervision is retained, gate it to a high-confidence subset where
   holo actually improves ligand geometry over apo and decoy controls.
