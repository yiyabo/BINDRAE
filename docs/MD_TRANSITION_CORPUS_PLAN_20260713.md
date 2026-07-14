# MD Transition Corpus Plan

Date: 2026-07-13

This document is the execution runbook for the trajectory-supervised APNB
track. The conference method and project-wide claims remain defined in
`BINDRAE_CONFERENCE_METHOD_BLUEPRINT_20260710.md` and
`CURRENT_PROJECT_STATUS_20260710.md`.

## Decision

The learned-phase model is retained as the highest-ceiling deterministic
method:

\[
X_i(t)=\operatorname{Exp}_{\gamma_i(\tau_i(t))}
\left[b(t)\Delta_i^\perp(t)\right],\qquad b(t)=t(1-t).
\]

Endpoint-only AHoJ-DB data can identify endpoint geometry and controlled
off-bridge correction, but it cannot identify residue event timing. The active
training hierarchy is therefore:

1. pretrain identity-phase normal residuals on the large endpoint corpus;
2. supervise monotone phase and off-bridge residuals on transition paths;
3. validate event order on held-out atomistic transition ensembles;
4. add a stochastic global path latent only after deterministic phase is
   demonstrably complementary to the residual branch.

No MD intermediate frame is available to the model at inference time.

## Evidence Tiers

Every trajectory record must carry one of the following evidence tiers. These
tiers must not be silently pooled in the headline benchmark.

| Tier | Definition | Permitted primary use |
|---|---|---|
| `gold_atomistic_transition` | Atomistic trajectory containing a verified endpoint-state transition under a documented protocol | Held-out path/event benchmark; phase supervision when not in test |
| `silver_enhanced_sampling` | Atomistic path generated with explicit bias or rare-event sampling | Phase/path supervision; downstream initialization studies |
| `bronze_modeled_path` | Coarse-grained, morphing, reconstructed, or optimization-derived transition path | Controlled pretraining and method diagnostics |
| `context_equilibrium` | Atomistic equilibrium trajectory without a verified endpoint transition | Flexibility, endpoint stability, and external physical checks |

`gold` denotes evidence quality for this task, not a claim that one finite MD
trajectory is the unique physical pathway. A biased trajectory must never be
used to make an absolute kinetics claim unless the sampling protocol and
reweighting make that claim separately defensible.

## Public Source Inventory

### Priority 0 controlled source: TransAtlas (availability blocked)

- Official search: <https://mmb.irbbarcelona.org/TransAtlas/search.php>
- Browse help: <https://mmb.irbbarcelona.org/TransAtlas/help.php?id=browseTab>
- Trajectory summary: <https://mmb.irbbarcelona.org/TransAtlas/help.php?id=summaryTab>
- Atomistic reconstruction: <https://mmb.irbbarcelona.org/TransAtlas/help.php?id=analysisTab>

TransAtlas is the closest large public collection to the endpoint-conditioned
task. It provides explicit endpoint-to-endpoint paths, 1000-frame raw and
100-frame reduced trajectories, endpoint identifiers, transition descriptors,
and formed/broken/transient contact annotations. The paths originate from
coarse-grained transition simulations; reconstructed atomistic intermediates
are useful initial structures but are not atomistic MD observations. Records
therefore enter as `bronze_modeled_path` until separately relaxed or simulated.

Availability check on 2026-07-13: the indexed help/search pages and FlexPortal
description remain visible to search engines, but the live `/TransAtlas/`
application returns HTTP 404. It is not currently a dependable automated
download source. Recovery through the maintainers, an institutional archive, or
an archived data dump should continue in parallel, but it must not block the MD
track.

### Priority 0 live source: GPCRmd

- Data download documentation:
  <https://gpcrmd-docs.readthedocs.io/en/latest/data-download.html>
- Primary resource paper:
  <https://www.nature.com/articles/s41592-020-0884-y>

GPCRmd exposes trajectories, topologies, coordinates, and protocol files, and
documents a batch API downloader. It is domain-specific and many simulations
sample a state rather than a complete apo-to-holo transition. Candidate records
remain `context_equilibrium` until an actual transition and both endpoint-state
matches are verified.

### Priority 1: ATLAS, MDRepo, and MDverse

- ATLAS: <https://www.dsimb.inserm.fr/ATLAS>
- ATLAS paper: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10767941/>
- MDRepo: <https://mdrepo.org/>
- MDRepo paper: <https://pmc.ncbi.nlm.nih.gov/articles/PMC11701643/>
- MDverse: <https://mdverse.github.io/>

ATLAS supplies standardized atomistic equilibrium trajectories. MDRepo is a
community trajectory repository with a command-line batch downloader, while
MDverse provides a reusable index of public MD datasets. These sources are
primarily discovery and physical-context resources. A record is promoted to
`gold_atomistic_transition` only after frame-level verification of a transition
between the required endpoint basins.

## Canonical Manifest Contract

The corpus is represented as JSONL. One line describes one trajectory replica;
`ensemble_id` groups replicas for the same endpoint pair. Paths are relative to
the manifest unless they are absolute.

```json
{
  "schema_version": "bindrae_md_transition_v1",
  "transition_id": "source:system:replica",
  "ensemble_id": "source:system",
  "status": "candidate",
  "source": {
    "name": "transatlas",
    "record_url": "https://...",
    "license": "verify-before-download"
  },
  "evidence": {
    "tier": "bronze_modeled_path",
    "contains_endpoint_transition": true,
    "biased_sampling": true,
    "physical_time_interpretable": false
  },
  "protein": {
    "uniprot_id": null,
    "chain_ids": ["A"],
    "sequence_sha256": null
  },
  "ligand": {
    "comp_id": null,
    "inchikey": null
  },
  "endpoints": {
    "apo_pdb_id": null,
    "holo_pdb_id": null,
    "apo_structure_path": null,
    "holo_structure_path": null
  },
  "trajectory": {
    "topology_path": null,
    "coordinate_paths": [],
    "n_frames": null,
    "frame_interval_ps": null
  },
  "usage": {
    "phase_supervision": false,
    "heldout_benchmark": false,
    "kinetics_claims": false
  },
  "split": {
    "name": "unassigned",
    "family_group": null,
    "ligand_scaffold_group": null
  },
  "quality": {
    "endpoint_mapping_verified": false,
    "residue_mapping_fraction": null,
    "transition_verified": false,
    "notes": []
  }
}
```

The manifest is intentionally source-neutral. Source-specific metadata is
preserved under an optional `source_metadata` object rather than changing the
canonical fields.

Audit candidate metadata without requiring downloads:

```bash
python scripts/audit_md_transition_manifest.py \
  --manifest processed_data/md_transitions/candidates.jsonl
```

After preprocessing, add `--check-files` so every declared local endpoint,
topology, and trajectory path is verified.

## Admission Gates

A trajectory may be used for phase supervision only if all of the following
hold:

1. provenance and simulation/path protocol are recorded;
2. both endpoint basins are defined and mapped to the same canonical residues;
3. at least three valid intermediate frames remain after preprocessing;
4. residue mapping coverage and missing-segment locations are reported;
5. the transition is verified at frame level rather than inferred from a title;
6. the record is assigned by protein-family and ligand-scaffold group before
   any model selection;
7. no trajectory or homologous replica from a held-out ensemble appears in
   training.

A record may enter the held-out headline benchmark only when it is
`gold_atomistic_transition`, `transition_verified=true`, and assigned to
`split.name=test`. Bronze and silver paths receive separate tables.

## Progress And Target Extraction

Raw frame index is not the learning target. Different simulations can traverse
the same structural path at different wall-clock rates.

For each accepted trajectory:

1. remove periodic-boundary artifacts and align the stable protein core;
2. map all frames onto the common apo/holo residue axis;
3. orient the path apo-to-holo and trim equilibrium dwell frames;
4. define global progress `s` by arc length, event-aligned progress, or a
   monotone combination of endpoint distance and contact events;
5. estimate residue phase `tau_i(s)` by monotone projection onto the endpoint
   bridge;
6. compute the product-manifold log residual from the bridge state to the
   observed frame and project it into the bridge-normal space;
7. mask endpoint neighborhoods where division by the endpoint-zero envelope is
   numerically ill-conditioned;
8. store confidence, projection error, residual norm, and event labels.

The decomposition is rejected for a residue/path when bridge projection is
ambiguous or the required residual exceeds the configured metric cap. Such
failures are data-quality evidence, not values to clip silently.

## Cluster-Generated Corpus

The planned 2-3 nodes and 24 A100 GPUs are sufficient for a serious curated MD
program, but GPU count does not remove the rare-event problem. The preferred
parallel layout is one independent system or replica per GPU, not one trajectory
spread over all 24 GPUs.

### Readiness audit (2026-07-13)

The cluster is software-ready for the one-system OpenMM pilot, but not yet
storage-ready for a production trajectory corpus:

- the isolated `BINDRAE-MD` environment is installed and validated on A100;
- the shared `/mnt/inaisfs` GPFS reports 400 TB total, 395 TB used, about 5.8 TB
  available, and 99% capacity utilization;
- the GPU partition exposes 22 nodes with 8 A100 GPUs per node;
- PLUMED/WESTPA and a production scratch/staging policy remain intentionally
  deferred until the one-system setup, minimization, and short-equilibration
  gates pass.

Create a separate versioned `BINDRAE-MD` environment instead of mutating the
training environment. Before the pilot, obtain a project scratch allocation or
document a safe node-local staging workflow. The 8-16-system pilot may fit in
the remaining space with aggressive output control, but a 50-300-system corpus
must not rely on an already 99%-full shared filesystem.

The initial environment is specified by `environment-md.yml`. It intentionally
uses an OpenMM-first stack and excludes PLUMED/WESTPA until the one-system smoke
passes. The environment pins the CUDA compatibility level to 12.4 and requests
the OpenFF base packages plus RDKit/AmberTools. A remote conda dry-run completed
successfully on 2026-07-13 and resolved OpenMM 8.2.0, OpenFF Toolkit 0.18.0,
AmberTools 24.8, and CUDA 12.4. The current dependency graph still pulls a CPU
PyTorch stack transitively through the force-field/OpenFF integration; this is a
known footprint cost, not a second training dependency. The environment was
installed successfully on 2026-07-13 and occupies approximately 4.7 GB. A
login-node validation passed all imports, AmberTools executable checks, and
OpenMM plugin discovery. Slurm job `141483` then completed on one A100 in 36
seconds: Reference, CPU, CUDA, and OpenCL all computed forces, and every
cross-platform force difference was within OpenMM tolerance. OpenMM documents
conda-forge CUDA installation and the
`openmm.testInstallation` check at
<https://docs.openmm.org/development/userguide/application/01_getting_started.html>;
OpenFF recommends an isolated conda-forge environment at
<https://docs.openforcefield.org/en/latest/install.html>.

### Pilot

1. Select 8-16 AHoJ systems spanning pocket opening, closure, side-chain gate,
   contact formation, and contact release.
2. Prepare and validate protein, ligand, solvent, ions, and force-field
   parameters before production allocation.
3. Run 3-5 independent replicas or path channels per system.
4. Compare ordinary endpoint equilibration with one explicit transition method
   such as string/path-CV or weighted ensemble.
5. Retain every failed transition attempt in the run manifest.

#### Current execution status

The endpoint screen is implemented by
`scripts/select_md_pilot_candidates.py` and its CPU Slurm wrapper. It checks
raw endpoint correspondence, sequence identity, ligand sanitization and
parameterization risk, aligned global and pocket motion, contact changes, and
candidate diversity. It writes a complete endpoint audit, ranked candidates,
selected IDs, and canonical transition-manifest records.

Job `141563` screened 8,000 deterministic samples in 70 seconds and found 481
initially eligible endpoint pairs. A stricter second pass, job `141565`,
excluded common cofactors and extreme endpoint mismatches; it found 319
eligible pairs and selected 16 across contact-switch, local-pocket,
domain-motion, and moderate-motion categories. The canonical manifest passed
with zero errors and zero warnings. These records remain
`context_equilibrium`: endpoint selection does not create transition evidence.

The first setup target is `6vba-A-QU4-901`: 223 protein residues, a single
31-heavy-atom organic ligand, 1.02 A aligned global CA RMSD, 0.69 A pocket CA
RMSD, and 11.49 A maximum local displacement. CPU jobs `141568`, `141573`, and
`141575` established the full protein-repair, OpenFF 2.2.1 ligand
parameterization, explicit-solvation, and minimization chain. They also exposed
that a raw per-atom force threshold is not a valid convergence gate for rigid
TIP3P water because constrained O/H forces cancel at the molecular level.

Job `141579` used deterministic solvent placement and staged minimization:
1,000 iterations with protein/ligand heavy-atom restraints followed by 1,000
unrestrained iterations. The resulting 35,144-atom system reached
`-569771.5 kJ/mol`; maximum residue/molecule net forces were 172.5 for protein,
128.4 for ligand, 119.5 for water, and 15.5 kJ/mol/nm for ions. This passes the
500 kJ/mol/nm setup gate. The delayed duplicate A100 job `141566` was canceled
rather than consuming a GPU for the same setup check.

Job `141603` then passed restrained heating and short NVT: final temperature was
292.5 K, protein CA RMSD was 0.716 A, and ligand heavy-atom RMSD was 0.958 A.
After fixing explicit periodic-box transfer between OpenMM contexts and making
ligand RMSD minimum-image aware, job `141615` passed short NPT at 302.1 K,
1.016 g/mL mean density, 0.006 volume coefficient of variation, 0.777 A protein
CA RMSD, and 1.121 A ligand RMSD.

The same setup/NVT/NPT lane was then expanded to `4tts-A-6DD-401`,
`2zd8-A-MER-401`, and `1daf-A-DSD-225`. All three passed system preparation and
NVT/NPT stability. Their final NPT protein CA RMSDs were 0.872, 0.669, and
0.700 A; ligand RMSDs were 0.935, 2.593, and 2.461 A; mean densities were
1.011, 1.020, and 1.018 g/mL. The four endpoint-equilibrium replicas are merged
under `processed_data/md_transition/context_manifests/ahoj_context_pilot4_npt0.jsonl`.
The combined manifest passes schema and file audits with zero errors and zero
warnings. Every record remains `context_equilibrium` with
`phase_supervision=false`.

The first global CA-RMSD collective-variable pull, job `141665`, was numerically
stable but correctly failed the transition gate: 5,000 kJ/mol/nm^2 was too weak
and apo CA RMSD improved only from 1.374 to 1.313 A. Job `141670` at 50,000
kJ/mol/nm^2 also remained more holo-like. Job `141671` fixed the pilot protocol
at 200,000 kJ/mol/nm^2, a 0.25 A target, 20 ps pull, and 4 ps endpoint hold. It
reached 0.628 A apo CA RMSD, crossed into the apo basin, and passed path and
atomistic audits: no severe heavy-atom clash, maximum peptide C-N distance
1.434 A, and 0.132 maximum heavy-bond relative deviation.

The same fixed protocol was applied without per-system retuning to three more
systems. `4tts-A-6DD-401` and `2zd8-A-MER-401` passed both the transition and
atomistic gates. `1daf-A-DSD-225` remained a recorded failure: its target-gap
progress was 0.420, and a longer 30 ps retry improved this only to 0.464, below
the declared 0.5 gate. The current fixed-protocol yield is therefore 3/4, not a
silently curated 100%.

The accepted paths are still biased atomistic sampling
(`silver_enhanced_sampling`), not kinetics evidence. Target extraction reverses
the generated holo-to-apo path, aligns the stable protein core, applies a
21-frame circular/coordinate low-pass filter, and retains only residues with
endpoint product-metric motion at least 0.5. A monotone dynamic program then
estimates `tau_i`; the product-manifold residual is projected normal to the
Cartesian-backbone bridge and divided by the endpoint-zero polynomial envelope
only where that operation is well conditioned. Phase and residual supervision
must each cover at least 5% of their eligible points.

Three systems passed this full target gate. Their canonical cache at
`processed_data/md_transition/phase_normal_cache_pilot3_20260713_v1` contains
303 frames, 796 mapped residues, and 2,111 valid normal-residual targets. The
per-system phase supervision densities are 5.10%, 7.85%, and 5.75%. These sparse
targets are appropriate for a low-weight training smoke, not a standalone
performance claim. Trainer support uses a dedicated `phase_normal_residual`
objective so MD head-coordinate targets are not confused with the historical
free-flow boundary-residual cache. One-GPU diagnostic job `141736` is the first
end-to-end training smoke; train and validation deliberately reuse the same
three systems and therefore measure pipeline health only.

Operationally, code-level trajectory-supervised training can start as soon as
the first silver paths pass those audits, without waiting for the entire corpus.
A realistic target is 1-2 days for a one-to-four-system training smoke, 3-7 days
for the first useful multi-system silver screen, and one-to-two weeks for the
8-16-system multi-replica pilot. Rare-event yield can lengthen this schedule;
gold held-out transition evidence remains a separate final-selection gate.

### Scale-out

Scale to 50-100 systems only after the pilot passes topology, endpoint-basin,
transition-yield, and storage gates. A later 100-300-system corpus is reasonable
with 24 GPUs if system preparation is automated and the selected rare-event
protocol produces usable transitions. Production jobs must use Slurm; no MD
production run belongs on the login node.

### Resource posture

- parallelize across systems and replicas first;
- keep equilibration, production, and analysis as separately resumable jobs;
- write wrapped/stripped analysis trajectories at controlled frame intervals;
- preserve raw checkpoints while avoiding unnecessary solvent-coordinate I/O;
- estimate storage before scaling, because trajectory I/O is more likely than
  host RAM to become the operational bottleneck;
- use unique run IDs and immutable protocol metadata.

### Infrastructure gate

Before allocating production GPUs:

1. create and lock the separate MD software environment;
2. submit `scripts/slurm/validate_md_environment_1gpu.sh` and preserve its JSON
   report;
3. run one protein-ligand setup/minimization/equilibration smoke;
4. verify GPU acceleration and checkpoint resume through Slurm;
5. measure trajectory bytes per nanosecond under the chosen output stride;
6. reserve storage for raw checkpoints, stripped trajectories, and analyses;
7. confirm ligand parameterization and license provenance for every pilot
   system.

## Experiment Gates

### Gate 1: public-data feasibility

- at least 20 endpoint-compatible transition paths;
- at least two independent sources or one source plus generated paths;
- canonical residue mapping coverage reported;
- source licenses and redistribution conditions recorded.

### Gate 2: target identifiability

- synthetic and held-out paths recover monotone `tau_i`;
- phase targets are stable to frame subsampling and progress alignment;
- phase and normal residual reconstruction error is below a declared threshold;
- event-order labels agree across replicate paths when the event is conserved.

### Gate 3: deterministic APNB

Under matched trunk, data, loss, and compute:

- warp-only improves event ordering over the synchronous bridge;
- residual-only improves off-bridge geometry/validity;
- full APNB improves both and uses a smaller residual than residual-only;
- full APNB does not collapse phase to identity;
- improvements transfer to held-out gold trajectories.

### Gate 4: stochastic extension

Only after Gate 3, train a global path latent on endpoint pairs with multiple
transition replicas. Report ensemble coverage and precision together; diversity
alone is not evidence of a better path model.

## Immediate Execution Queue

1. Complete the seven-system, 29-path low-weight phase/normal-residual training
   smoke and verify nonzero gradients, finite losses, replica rotation, and
   cache coverage.
2. Diagnose setup failures separately from transition failures. Do not weaken
   force, endpoint-crossing, mapping, or supervision-density gates to increase
   apparent yield.
3. Expand endpoint preparation and independent path replicas toward 20-50
   systems after the seven-system smoke confirms the training interface.
4. Export GPCRmd metadata through its public search/API surface and cross-match
   PDB/UniProt identifiers with AHoJ-DB.
5. Request or recover a TransAtlas metadata/data archive without blocking the
   live-source pipeline.
6. Inspect and download 20-50 high-quality matched candidates across available
   sources.
7. Promote only verified endpoint-crossing trajectories into phase supervision;
   keep context-equilibrium and failed pulls in separate manifests.
8. Run Gate 2 before formal learned-phase hyperparameter selection.

## Pilot Scale-Out Launch (2026-07-14)

The first scale-out batch is deliberately split into two auditable lanes rather
than launching the full 16-by-5 target as one opaque job set.

1. `silver_replica_pilot4_r1to4_20260714_v1` adds replicas 1-4 to the four
   systems with passed endpoint NPT contexts. The 16 matrix rows use unique
   seeds, resample Maxwell-Boltzmann velocities from the shared endpoint state,
   and lock the fixed pilot protocol: 500 pre-equilibration steps, 10,000 pull
   steps, 2,000 endpoint-hold steps, 100-step reporting,
   `k=200000 kJ/mol/nm^2`, and a `0.025 nm` target. Every row runs pull, path
   audit, atomistic audit, and phase-normal target export as a resumable
   pipeline. A failed transition remains a recorded outcome and does not block
   other replicas.
2. `context_pilot12_20260714_v1` prepares the remaining 12 systems from the
   strict 16-system selection. Each row independently runs setup/minimization,
   restrained-to-unrestrained NVT, NPT, and canonical context registration.
   Only a passed context may enter a later transition-replica matrix.

Slurm jobs `141911` and `141913` implement the first lane; job `141917`
implements the second. These are CPU arrays so they can exploit fragmented
cluster capacity while the one-A100 trajectory-supervised training smoke
remains queued. The resulting paths are still `silver_enhanced_sampling`, not
kinetics evidence or gold atomistic transitions.

Both lanes use `afterany` scientific continuations rather than `afterok`.
Failure to cross a transition or stability gate is an experimental outcome,
not an orchestration failure that should block unrelated systems. Replica
finalization records every outcome and assembles only targets that pass pull,
path, atomistic, and phase-normal target gates. Context continuation admits only
passed NPT contexts, then submits five independent pull replicas per admitted
new system and a second after-any finalization job. No training job is launched
automatically from these caches; model training remains a separate decision
after corpus quality and yield are inspected.

## Replica Pilot Outcome (2026-07-14)

The two replica lanes attempted 51 fixed-protocol paths. Twenty-nine paths
passed pull, path, atomistic, residue-mapping, phase-identifiability, and
normal-residual gates, for a path-level yield of 56.9%. These paths span seven
endpoint systems rather than 29 independent systems. Their immutable merged
cache is
`processed_data/md_transition/phase_normal_cache_silver29_20260714_v2` and
contains 2,929 frames, 6,055 residue instances, and 58,925 valid
normal-residual supervision points. Aggregate phase and residual supervision
densities are 15.5% and 15.4%, respectively.

The successful systems and replica counts are `1c3i` (5), `1qvt` (5), `2zd8`
(4), `3ef2` (5), `4tts` (1), `5hy8` (5), and `8czn` (4). `1daf`, `3x2h`, and
`7dw8` failed the fixed transition gate. Three physically accepted `6vba`
paths remained below the predeclared 5% target-identifiability threshold and
were not promoted. Five additional systems failed setup or minimization gates
and require chemistry/topology diagnosis rather than blind retries.

The initial `1qvt` target export failure was an identity-interface bug, not a
trajectory failure: the single-chain apo endpoint used chain `D`, while the
holo endpoint and prepared MD topology used chain `A`. All 186 numbered
residues and residue names matched. The exporter now follows the repository's
canonical identity contract: chain aliases are allowed only when apo, holo,
and MD topology are all single-chain; multi-chain systems retain strict chain
identity, duplicate aligned identities hard-fail, and residue-name mismatches
hard-fail. All five existing `1qvt` trajectories then passed every target gate
without rerunning MD.

The Stage-2 supervision loader now treats replicas as repeated observations of
one endpoint system. Direct single-cache files retain precedence. If only
`<sample>__silver_rXX.npz` files exist, training deterministically rotates over
sorted replicas by epoch, and phase and normal-residual loaders resolve the
same replica. This uses all accepted paths without pretending correlated
replicas are independent endpoint systems. Job `142152` is the first two-GPU,
two-epoch end-to-end smoke over the seven systems and 29-path cache; train and
validation deliberately reuse the same systems, so its purpose is pipeline
health rather than model selection or performance reporting.
