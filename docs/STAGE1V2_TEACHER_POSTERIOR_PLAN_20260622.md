# Stage-1-v2 Teacher Posterior Plan

Date: 2026-06-22

## Decision

Stage-1-v2 should not be another deterministic holo-rotamer endpoint model.
The replacement role is a **teacher-distilled student**:

```text
external strong model / teacher ensemble / holo-truth upper bound
-> offline local ligand-causal posterior labels or teacher latents
-> compact in-repo Stage-1-v2 student posterior encoder
-> Stage-2 scalar posterior + latent guidance for apo-to-holo path generation
```

This keeps Stage-2 as the main contribution. Stage-1-v2 becomes the local
posterior interface that says which residues should approach, release, keep,
or avoid ligand contact, and with what confidence.

The external teacher is the source of high-quality Stage-1 signal. The student
is the reproducible, batchable, ablatable implementation used by BINDRAE during
Stage-2 training and inference.

## Teacher vs Student Contract

Do not collapse the terms:

- **Teacher**: external or oracle source such as DynamicBind, FlowDock, Boltz,
  Chai, AF3-like systems, MD/MISATO-like trajectories, or `holo_truth`. The
  teacher may be expensive, slow, or not fully under our control.
- **Student**: BINDRAE's compact `Stage1PosteriorV2`, trained to predict the
  teacher's local posterior from apo protein, sequence/ESM features, and the
  known ligand pose.
- **Stage-2 consumer**: the bridge flow model. It should consume the student
  outputs by default, with separate oracle/external-teacher modes only for
  upper-bound or diagnostic controls.

The main publishable method should be the student-guided Stage-2 model. Direct
online teacher use should be treated as an upper-bound baseline or as offline
label generation, not as the default inference path.

## Why This Is The Right Pivot

The current LC-PGBF v1 result already shows a small positive Stage-2 path
effect from a learned local interaction prior. The weak point is not the
two-stage idea; it is the quality and expressivity of the Stage-1 signal.

The next architecture therefore upgrades Stage-1 from a scalar local contact
channel to a richer posterior:

- apo sidechain-ligand minimum distance;
- teacher/holo sidechain-ligand minimum distance;
- signed distance delta;
- contact probability;
- approach / release / stable-contact / stable-noncontact classes;
- switch confidence;
- optional latent vector distilled from external teacher features.

## Teacher Sources

Use a staged teacher ladder:

| source | role | status |
|---|---|---|
| holo truth | oracle upper-bound labels and schema smoke | immediate |
| current interaction prior | continuity baseline | available |
| DynamicBind / FlowDock | practical external endpoint/pocket teachers; exact side-chain/switch quality must be stratified empirically | next |
| Boltz / Chai / AF3-like models | strong endpoint/reference teachers when accessible | later |
| MD/MISATO-like trajectories | path-level teacher if licensing/runtime permits | later |

The first implementation exports `teacher_source=holo_truth`. This is not the
final method, but it fixes the exact per-residue file format that external
teacher runs will reuse.

## Student Model Choice

The student should be a compact ligand-conditioned posterior encoder, not a
full holo decoder:

```text
input:
  apo backbone frames / apo chi
  ESM residue features
  known-pose ligand tokens
  residue masks / pocket weights

trunk:
  ESM adapter
  ligand cross-attention / ligand-conditioned residue update
  lightweight geometric residue encoder

outputs:
  contact probability
  teacher distance / signed distance delta
  approach probability
  release probability
  switch or active probability
  confidence
  optional per-residue latent z_post
```

The first implementation can use scalar heads only. `z_post` and Stage-2
representation alignment should be added after scalar posterior features pass
smoke tests and counterfactual controls.

## RAEv2/REPA Positioning

RAEv2's useful philosophy here is not image-specific, but it should remain an
analogy rather than the method name. BINDRAE is not implementing an image RAE
architecture or claiming that RAEv2 directly transfers to protein-ligand
induced-fit.

The transferable pieces are:

- use a strong pretrained/teacher representation rather than forcing a weak
  autoencoder to discover everything from scratch;
- aggregate multi-level local features into a compact latent;
- guide the generative model through representation alignment, not only by
  concatenating a scalar feature.

For BINDRAE this becomes:

```text
Stage-1-v2 posterior encoder:
  apo pocket geometry + ligand pose + sequence/ESM + teacher posterior labels
  -> per-residue posterior logits + per-residue latent z_post

Stage-2:
  consumes posterior scalars as existing prior features
  plus a REPA-like alignment loss between Stage-2 hidden state and z_post
```

The main BINDRAE claim should therefore be **teacher-distilled
posterior-guided bridge flow**. `z_post` alignment is optional REPA-style
latent guidance, not a prerequisite for the Stage-1-v2 scalar posterior.

## Benchmark Targets

Do not benchmark this as generic docking. The sharper benchmark is:

```text
known-pose ligand-conditioned induced-fit trajectory generation
```

Report three layers of evidence:

1. Endpoint quality:
   - pocket sidechain-ligand distance error;
   - pocket/backbone RMSD to holo;
   - clash and peptide-geometry validity.

2. Path quality:
   - residue transition endpoint error;
   - path MAE to apo-to-holo distance interpolation;
   - approach/release direction accuracy;
   - smoothness and steric validity across frames.

3. Coverage/ensemble quality:
   - DynamicFlow-style min RMSD to holo conformers;
   - cover ratio under a fixed holo-state RMSD threshold;
   - diversity without clash inflation.

## Immediate Implementation Steps

1. Export teacher posterior labels from holo truth on validation and training
   lanes.
2. Extract the teacher-posterior schema and validator into
   `src/stage1/posterior_v2/schema.py`.
3. Implement `TeacherPosteriorDataset` and a compact `Stage1PosteriorV2`
   student.
4. Train the student against `holo_truth` labels first, with no-ligand and
   shuffled-ligand controls.
5. Add external-teacher adapters that write the same schema.
6. Add Stage-2 student-posterior guidance:
   - scalar posterior features as current interaction-prior superset;
   - REPA-style hidden-state alignment to `z_post`;
   - zero/shuffled/oracle controls.
7. Run full-scale Stage-2 training with multi-seed transition-path evaluation.

## Strict Success Criteria

This line is worth continuing only if the Stage-1-v2 guided Stage-2 model
improves active-residue path metrics over no-prior and zero-prior controls
without worsening clash or background stability.

Endpoint-only gains are insufficient. A publishable claim needs path-level
transition improvements and external-teacher baselines.
