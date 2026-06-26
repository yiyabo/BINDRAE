# Stage-1 Typed Candidate Interaction Energy Plan (2026-05-11)

## Goal

Replace the current proximity-only candidate residual with a typed
rotamer-ligand compatibility energy that can distinguish correct ligand
chemistry from no-ligand, scrambled-types, and batch-shuffled-ligand controls.

This is still a Stage-1 known-pose induced-fit task. Ligand coordinates remain
aligned into the apo frame. The next phase should not introduce docking or
pose-search behavior.

## Current Problem

`GeometryCandidateScorer` already computes FK candidate atom14 geometry for
the three chi1 rotamer bins. It then scores candidates with:

```text
logits = base_prior + gate * ligand_residual(distance_rbf, min_dist, clash, optional s_geo)
```

This explains the validation result:

- translated-away controls are easy because local ligand density disappears;
- scrambled-types controls are hard because the current residual mostly sees
  geometry, not typed residue-ligand chemistry;
- batch-shuffled controls are not same-pocket chemical decoys, but they still
  show that the current objective does not reliably use ligand identity.

The next implementation should make the candidate residual explicitly depend on
typed side-chain atom and ligand token interactions.

## Proposed Model Object

Add a typed interaction energy branch inside `GeometryCandidateScorer`:

```text
typed_energy[i, k] =
  Pool_{a in chi1-dependent atom14 atoms, l in ligand tokens}
    PairMLP(residue atom type, ligand token type, distance RBF, direction features)

logits[i, k] =
  base_prior[i, k]
  + gate[i] * (distance_residual[i, k] + typed_energy[i, k])
```

Where:

- `i` is the residue index.
- `k` is the chi1 candidate bin.
- `a` is an FK-derived atom14 side-chain atom.
- `l` is a ligand heavy atom or probe token.
- ligand type features come from the existing `lig_types` tensor
  (`LIGAND_TYPE_DIM = 20`).
- residue-side features come from residue type and atom14 atom index/group.

The first version should keep the existing base prior and gate unchanged. It
should only add a typed residual term, so checkpoint compatibility and ablation
logic stay manageable.

## Minimal Implementation

### Code surfaces

Primary files:

- `src/stage1/models/torsion_head.py`
- `src/stage1/models/stage1_model.py`
- `src/stage1/training/config.py`
- `src/stage1/training/trainer.py`
- `src/stage1/modules/losses.py`
- `scripts/train_stage1.py`
- `scripts/diagnose_stage1_prior.py`

Slurm files:

- add a new launcher under `scripts/slurm/` rather than mutating the existing
  candidate-rerank launcher.

### Model changes

Add config flags:

```text
geometry_scorer_use_typed_energy: bool = false
geometry_scorer_typed_hidden: int = 128
geometry_scorer_typed_pair_dim: int = 64
geometry_scorer_typed_cutoff: float = 6.0
geometry_scorer_typed_pool: str = "softmin"
geometry_scorer_typed_max_pairs: int = 512
lambda_typed_candidate_energy: float = 0.0
typed_candidate_decoy_kind: str = "scrambled"
typed_candidate_contact_only: bool = true
typed_candidate_margin: float = 0.05
```

Extend `GeometryCandidateScorer.forward` to accept `lig_types` in addition to
`lig_points` and `lig_mask`.

The typed branch should:

1. reuse `compute_candidate_geometries`;
2. restrict to chi1-dependent atom14 atoms;
3. compute residue atom-ligand token distances under `typed_cutoff`;
4. build pair features from distance RBF, candidate atom identity, residue type,
   ligand token type, and optional direction vectors;
5. pool valid pair scores into one scalar per residue and candidate;
6. return the typed contribution in decomposition mode.

Keep a strict no-ligand behavior:

- no ligand means typed energy is exactly zero;
- translated-away should produce near-zero typed energy after cutoff masking;
- scrambled-types should keep geometry but change the typed pair features.

### Loss changes

Do not revive global candidate-rerank as the main objective. Add a narrow typed
candidate loss:

```text
L_typed =
  CE(correct_logits, holo_chi1_bin)
  + margin(correct_energy[holo], decoy_energy[holo])
  + noharm_on_apo_correct
  + zero_energy_on_non_contact
```

The first smoke version can use:

- `lambda_geometry_chi1` for the normal rotamer CE;
- `lambda_typed_candidate_energy` for correct-vs-decoy typed energy separation;
- contact-only mask on CA-contact switch residues;
- decoy kind `scrambled` first, then `shuffled`.

The loss must report separate metrics for:

- typed CE;
- typed correct-vs-scrambled margin;
- typed correct-vs-shuffled margin;
- non-contact zero penalty;
- apo-correct harm.

## Validation Plan

### Local checks

Run:

```bash
python -m py_compile \
  src/stage1/models/torsion_head.py \
  src/stage1/models/stage1_model.py \
  src/stage1/training/config.py \
  src/stage1/training/trainer.py \
  src/stage1/modules/losses.py \
  scripts/train_stage1.py \
  scripts/diagnose_stage1_prior.py
```

Run shell syntax checks for any new launcher:

```bash
bash -n scripts/slurm/<new_launcher>.sh
```

### Slurm smoke

Submit a small smoke first:

```text
MAX_BATCHES=20
TAG=typed_candidate_energy_smoke_20260511
```

Smoke must verify:

- model starts;
- typed branch receives gradients;
- no-ligand path produces zero typed energy;
- scrambled-types path changes typed pair features while preserving ligand
  coordinates;
- logs include typed loss components.

### Full validation

Only after smoke passes, run a short validation-budget training screen. The
selection metric should be strict-control lift, not global chi1 accuracy.

Primary go metrics:

| metric | target |
| --- | ---: |
| candidate `correct - scrambled_types`, contact-switch | `> +0.01` |
| candidate `correct - batch_shuffled_ligand`, contact-switch | `> +0.01` |
| candidate `correct - no_ligand`, contact-switch | `> +0.01` |
| apo-correct harm | no worse than baseline by more than `0.01` |
| non-contact typed energy | near zero |

Translated-away lift is a sanity check, not a success criterion.

## Expected Failure Modes

Shortcut through distance remains possible if typed energy is allowed to use
distance alone. The pair MLP should therefore receive ligand type features and
should be evaluated specifically against scrambled-types controls.

Overfitting to ligand type priors is possible. Batch-shuffled controls help
detect this, but they are not same-pocket chemical decoys. A later phase should
add same-pocket or scaffold-preserving decoys.

The typed branch may harm apo-correct residues. Keep no-harm and non-contact
zero penalties active from the first training screen.

## Stage-2 Interface Posture

Do not feed deterministic Stage-1 chi1 endpoints into Stage-2 from this branch.
Until strict chemistry controls pass, Stage-2 should only use:

- zero-prior fallback;
- ablation-only soft posterior features;
- optional entropy/confidence features after calibration checks.

## Implementation Order

1. Add typed branch plumbing and decomposition outputs.
2. Add typed energy loss and logging.
3. Add CLI/config flags.
4. Add a Slurm smoke launcher.
5. Run local compile and launcher syntax checks.
6. Submit a `MAX_BATCHES=20` smoke.
7. If smoke is clean, run a short screen and evaluate with the existing
   causality diagnostic report.
