# Scripts Index

Top-level `scripts/` keeps active entrypoints and data-prep utilities. Older
helpers and one-off diagnostics are archived under `scripts/archive/`. Do not
start a new experiment from an archived script unless the current runbook says
to revive that lane.

## Active Entry Points

- `train_stage1.py` - main Stage-1 training CLI.
- `train_stage2.py` - main Stage-2 training CLI; supports the endpoint-exact phase-normal path, OracleMotion features, ESM last-K fusion, and controlled legacy ablations.
- `train_stage1v2_posterior.py` - trains the Stage-1-v2 teacher-distilled posterior student.
- `audit_stage1v2_posterior.py` - audits a trained Stage-1-v2 posterior checkpoint with threshold, ranking, calibration, and ligand-control metrics.
- `export_stage1v2_posterior_cache.py` - exports per-sample Stage-1-v2 student posterior `.npz` caches for Stage-2 consumption.
- `export_oracle_motion_features.py` - exports OracleMotion-UB apo-to-holo motion features and direct oracle-apply audits.
- `export_stage2_teacher_residual_cache.py` - historical/free-flow teacher-residual exporter retained for reproducibility; it is not the active phase-normal path target.
- `export_md_phase_normal_targets.py` - convert an audited silver atomistic path into monotone phase and bridge-normal residual targets with explicit identifiability gates.
- `assemble_md_phase_normal_cache.py` - assemble only passed MD target directories into an immutable multi-system training cache.
- `merge_md_phase_normal_caches.py` - verify, deduplicate, and merge immutable MD phase-normal cache collections.
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
- `evaluate_stage2_bridge_timewarp.py` - endpoint-exact bridge time-warp headroom evaluator for pure/oracle/projected-progress paths.
- `evaluate_stage2_trajectory_reliability.py` - internal trajectory reliability benchmark against cubic SE(3)+chi interpolation.
- `evaluate_stage2_path_critic.py` - Stage-2 path critic/evaluator helper.
- `generate_stage2_trajectories.py` - exports generated Stage-2 trajectory files for visualization.
- `run_ca_morph_baseline.py` - simple static/linear/smoothstep CA path baselines.
- `run_anm_baseline.py` - CA-ANM projection path baseline.
- `run_adaptive_anm_baseline.py` - adaptive CA-ANM path baseline.
- `run_ebdims2_baseline.py` - eBDIMS2 external endpoint-conditioned CA path baseline runner.
- `evaluate_ebdims2_ca_paths.py` - common CA path evaluator for eBDIMS2, ANM, adaptive ANM, and CA morph baselines.

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
- `slurm/` - current cluster launchers; see `scripts/slurm/INDEX.md`.
