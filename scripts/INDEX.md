# Scripts Index

Top-level `scripts/` keeps active entrypoints and data-prep utilities. Older
helpers and one-off diagnostics are archived under `scripts/archive/`. Do not
start a new experiment from an archived script unless the current runbook says
to revive that lane.

## Current Operator Runbook

Read `../docs/PROJECT_HANDOFF_20260725.md` before launching or modifying an
experiment. The current Path-3 data-acquisition action is the bounded AHoJ
mapping-aware smoke, not a full-corpus or endpoint-only training run. The
direct GPU33 entrypoint is `run_ahoj_mapping_smoke_gpu33.sh`; it owns one
fresh output tag, persists its machine-readable state, and must not be mixed
with the older Slurm scale-out artifacts.

## Active Entry Points

- `train_stage1.py` - main Stage-1 training CLI.
- `train_stage2.py` - main Stage-2 training CLI; supports the endpoint-exact phase-normal path, OracleMotion features, ESM last-K fusion, and controlled legacy ablations.
- `train_stage1v2_posterior.py` - trains the Stage-1-v2 teacher-distilled posterior student.
- `audit_stage1v2_posterior.py` - audits a trained Stage-1-v2 posterior checkpoint with threshold, ranking, calibration, and ligand-control metrics.
- `export_stage1v2_posterior_cache.py` - exports per-sample Stage-1-v2 student posterior `.npz` caches for Stage-2 consumption.
- `export_oracle_motion_features.py` - exports OracleMotion-UB apo-to-holo motion features and direct oracle-apply audits.
- `export_stage2_teacher_residual_cache.py` - historical/free-flow teacher-residual exporter retained for reproducibility; it is not the active phase-normal path target.
- `validate_stage2_feature_subset.py` - load an exact Stage-2 subset through the production dataset path so prechecks catch feature-schema and canonical-residue mismatches before training.
- `build_stage2_leakage_clean_subset.py` - remove frozen validation/test protein-family and ligand-scaffold overlap before endpoint-trunk pretraining.
- `select_md_pilot_candidates.py` - screen and endpoint-pair-deduplicate MD
  acquisition candidates using explicit apo/holo sequence mapping and report
  requested-versus-available unique-pair capacity.
- `audit_md_candidate_capacity.py` - combine the frozen holdout leakage filter
  with historical apo/holo-pair exclusion, emit the truly novel acquisition
  manifest, and select a deterministic residue-mismatch stress smoke subset.
- `select_md_mapping_smoke_panel.py` - freeze a representative, motion-category
  balanced residue-mismatch panel from the leakage-clean AHoJ acquisition pool.
- `run_ahoj_mapping_smoke_gpu33.sh` - run the 32-system x 2-replica AHoJ
  mapping-aware OpenMM smoke directly on GPU33, with resumable context, pull,
  target-export, finalization, and consensus stages.
- `audit_external_endpoint_indexes.py` - normalize PSCDB, APObind, and CoDNaS-Q
  endpoint records, preserve source/site granularity, and report directional
  PDB-pair overlap without promoting PDB-only nonmatches to net-new systems.
- `select_apobind_structure_smoke.py` - reproduce the frozen strict APObind
  metadata proxy, deduplicate undirected endpoint pairs, and deterministically
  select a no-shared-endpoint cohort balanced over RMSD, resolution, and
  binding-site-size quartiles.
- `preflight_apobind_structure_smoke.py` - download selected RCSB mmCIF
  endpoints and conservatively audit declared chains, APObind site signatures,
  exact residue mapping, structural motion, unique site-local non-polymer
  ligand identity, and parseable CCD chemistry before any prepared-system or
  MD work.
- `export_apobind_structure_triplets.py` - resolve endpoint alternate locations,
  align holo protein and ligand into the apo frame, restore observed ligand
  bond orders from CCD chemistry, and emit source-neutral screening manifests.
- `audit_apobind_holdout_leakage.py` - filter exported APObind candidates against
  frozen validation/test protein families, ligand scaffolds, and exact endpoint
  PDB IDs without requiring candidate `torsion_apo.npz` files.
- `select_apobind_preparation_panel.py` - select a deterministic category-balanced
  max-min engineering panel from leakage-clean APObind candidates.
- `run_apobind_preparation_panel_gpu33.sh` - run the eight-system OpenMM
  preparation panel directly on private GPU33 workers and write per-system logs
  plus a machine-readable panel state; this is not a Slurm launcher.
- `build_apobind_preparation_rescue.py` - build the bounded APObind preparation
  rescue plan: one unchanged-threshold longer-minimization retry plus one
  deterministic leakage-clean, same-category replacement for a force-field
  unsupported panel member.
- `run_apobind_preparation_rescue_gpu33.sh` - run the two-system rescue plan
  directly on private GPU33 workers and write a machine-readable rescue state;
  prepared-ready outputs remain engineering inputs rather than accepted MD
  replicas or Path-3 labels.
- `freeze_apobind_prepared_panel.py` - reconcile the original APObind panel
  with its bounded retry/replacement state, verify every preparation artifact
  and the unchanged residue-force gate, and freeze a slot-aligned dynamics-smoke
  manifest only when the complete panel is resolved.
- `run_apobind_prepared_panel_dynamics_gpu33.sh` - run the frozen APObind panel
  through short NVT then NPT engineering gates directly on private GPU33 and
  record per-system failures plus a machine-readable aggregate state.
- `build_apobind_replica_pilot.py` - register a hash-locked, fully passed
  APObind NPT panel as endpoint contexts and build an immutable independent-seed
  replica matrix under the frozen global CA-RMSD pull protocol.
- `build_apobind_torsion_cache.py` - extract fresh apo/holo torsions for every
  system in a frozen APObind replica matrix, enforce the canonical residue-axis
  schema and pair-mapping gate, and write PDB/cache/identity hash evidence.
- `run_apobind_replica_pilot_gpu33.sh` - run the APObind 8-system by 2-replica
  silver-path matrix directly on private GPU33 after canonical-cache preflight,
  preserve all stage failures, finalize passed targets, and build consensus only
  for systems with two passed replicas.
- `run_apobind_target_recovery_gpu33.sh` - preserve the original pilot state and
  recover target export only for frozen matrix indices `2,3,6,7,10,11,13`, then
  write separate finalization, consensus, provenance, and recovery-state artifacts.
- `export_md_phase_normal_targets.py` - convert an audited silver atomistic path into monotone phase and bridge-normal residual targets with explicit identifiability gates.
- `export_md_phase_normal_matrix_entry.py` - re-export one replica from a merged collection, including explicit replica-matrix provenance and identity-phase targets for the residual-only matched ablation.
- `assemble_md_phase_normal_cache.py` - assemble only passed MD target directories into an immutable multi-system training cache.
- `merge_md_phase_normal_caches.py` - verify, deduplicate, and merge immutable MD phase-normal cache collections.
- `analyze_md_replica_consistency.py` - quantify phase/residual reproducibility across replicas and the deterministic explained-energy ceiling before choosing a stochastic path latent.
- `evaluate_md_phase_normal_oracle_ladder.py` - leave-one-replica-out zero,
  consensus, medoid, and route-oracle ladder for residual identifiability.
- `analyze_md_consensus_residual_structure.py` - measure temporal rank and
  spatial localization of deterministic consensus residual targets.
- `build_md_phase_normal_consensus_cache.py` - collapse repeated MD replicas
  into confidence-weighted, support-audited deterministic system targets.
- `split_md_phase_normal_systems.py` - create a deterministic system-disjoint train/validation split balanced over replica and supervision counts.
- `split_md_phase_normal_groups.py` - create audited train/validation/test splits whose joint components are disjoint by protein sequence family and Bemis-Murcko ligand scaffold.
- `audit_md_rmsd_pull.py` / `audit_md_atomistic_path.py` - path-progress and independent atomistic geometry gates for biased MD paths.
- `build_md_context_matrix.py` / `run_md_context_pipeline.py` - immutable,
  resumable setup-to-NPT context preparation for selected MD pilot systems.
- `build_md_replica_matrix.py` / `run_md_replica_pipeline.py` - immutable,
  independent-seed silver pull replicas with path/atomistic audits and target
  export.
- `collect_md_context_results.py` - collect only passed endpoint contexts while
  retaining failed-system outcomes.
- `finalize_md_replica_matrix.py` - summarize every replica outcome and assemble
  all passed phase-normal targets into one immutable cache.
- `validate_triplets_data.py` - validates Stage-1 triplet datasets.
- `audit_ligand_sensitive_dataset.py` - ligand-sensitive dataset audit.
- `diagnose_stage1_prior.py` - Stage-1 diagnostic entrypoint.
- `summarize_stage1_ligand_causality_validation.py` - validation report summarizer.
- `build_teacher_posterior_labels.py` - Stage-1-v2 teacher-posterior label exporter.

## Active Stage-2 Evaluation / Demo

- `evaluate_stage2_transition_paths.py` - residue-level transition path evaluator.
- `evaluate_stage2_md_reference_paths.py` - compare generated paths with all held-out audited MD replicas using system- and replica-macro metrics.
- `compare_stage2_md_reference_results.py` - align two MD-reference outputs and
  report direction-aware system-level paired bootstrap intervals and win rates.
- `evaluate_stage2_bridge_timewarp.py` - endpoint-exact bridge time-warp headroom evaluator for pure/oracle/projected-progress paths.
- `evaluate_stage2_trajectory_reliability.py` - internal trajectory reliability benchmark against cubic SE(3)+chi interpolation.
- `run_controlled_manifold_benchmark.py` - matched synchronous/warp/residual/full
  benchmark on controlled `SE(3) x T^2` paths with observed versus hidden route
  information; isolates phase-normal expressivity from route identifiability.
- `evaluate_stage2_path_critic.py` - Stage-2 path critic/evaluator helper.
- `generate_stage2_trajectories.py` - exports generated Stage-2 trajectory files for visualization.
- `run_ca_morph_baseline.py` - simple static/linear/smoothstep CA path baselines.
- `run_anm_baseline.py` - CA-ANM projection path baseline.
- `run_adaptive_anm_baseline.py` - adaptive CA-ANM path baseline.
- `run_ebdims2_baseline.py` - eBDIMS2 external endpoint-conditioned CA path baseline runner.
- `evaluate_ebdims2_ca_paths.py` - common CA path evaluator for eBDIMS2, ANM, adaptive ANM, and CA morph baselines.
- `audit_tps_flow_release.py` - pin and verify the public TPS-Flow checkout, weights, and data before reproduction.
- `build_path4_openmm_frame_reference.py` - build and preflight the frozen
  Path-3 all-atom per-frame cache shared by both Gate-0 scoring arms.
- `evaluate_path4_openmm_gate0.py` - independently relax and score one path
  from a contract-checked common all-atom frame-reference cache, with explicit
  per-frame minimization audit and relaxed-validity records.
- `optimize_path4_openmm_gate0.py` - non-learned multi-start OpenMM path
  optimizer with endpoint-exact fallback and force-call accounting.
- `summarize_path4_gate0_pairs.py` - strictly pair Path-3 and Path-4 OpenMM
  reports, enforce scorer/endpoint contract equality, recompute relaxed metrics
  on paired-valid frames, and report invalid-frame incidence and bootstrap
  intervals.
- `run_path4_gate0_dev_panel.py` - deterministically select a training-scope
  Gate-0 development panel, hard-exclude frozen evaluation/post-selection
  lists, isolate per-system failures and preflight rejections, and aggregate
  only successful primary pairs; `run_path4_gate0_dev_panel_gpu33.sh` is its
  first-panel direct private-node launcher,
  `run_path4_gate0_preflight_reserve_gpu33.sh` discovers prepared reserve
  candidates, and `run_path4_gate0_full_reserve7_gpu33.sh` runs the accepted
  reserve IDs through the full physical and Product-state comparison chain.

## Current Stage-2 Run Path

Read the current method and status before launching full-scale jobs:

- `../docs/BINDRAE_CONFERENCE_METHOD_BLUEPRINT_20260710.md`
- `../docs/CURRENT_PROJECT_STATUS_20260710.md`

Current primary training surface:

- `slurm/train_stage2_oracle_motion_ablation_4gpu.sh` - despite the historical filename, this is the active Stage-2 launcher and includes phase-normal, OracleMotion, ESM, REPA, and legacy ablation controls.

For final-scale runs, prefer a short `PRECHECK_ONLY=1` Slurm run before the
long training submission. Long jobs should use unique tags and should not
overwrite OracleMotion baseline or REPA comparison outputs.

## Data Preparation

- `prepare_casf2016.py`
- `prepare_ligands.py`
- `extract_pockets.py`
- `extract_torsions.py`
- `cache_esm2.py`
- `cache_ahojdb_esm2.py` - cache AHoJ-DB ESM2 features on the canonical torsion residue axis; apo-PDB parsing is a legacy fallback.
- `split_dataset.py`
- `verify_casf2016.py`
- `verify_ligand_consistency.py`

## Alternate Dataset Line

- `download_ahojdb_pdbs.py`
- `extract_ahojdb_torsions.py`
- `prepare_ahojdb_triplets.py`
- `cache_ahojdb_esm2.py`

## Archive

- `archive/legacy_tools/` - older setup helpers and test wrappers.
- `archive/diagnostics/` - one-off audits and lineage/debug scripts.
- `archive/20260627_pre_fullscale_stage2/` - exploratory Stage-1 and early
  Stage-2 entrypoints moved out of the main path before full-scale training.
- `slurm/archive/20260725_path4_pre_gate0/` - learned Path-4
  physical-normal-residual distillation is frozen until a documented Gate-0
  decision authorizes reopening it.
- `slurm/` - current cluster launchers; see `scripts/slurm/INDEX.md`.
