# Slurm Launcher Index

Top-level launchers are the currently supported entrypoints. Superseded
experiment variants are archived under `scripts/slurm/archive/`. For new runs,
prefer the launchers listed here and treat archived files as reproducibility
records, not templates.

## Stage-1 Diagnostics

- `audit_ligand_sensitive_dataset.sh` - ligand-sensitive dataset and contact-label audit.
- `diagnose_stage1_prior.sh` - Stage-1 prior diagnostics.
- `build_teacher_posterior_labels_1gpu.sh` - Stage-1-v2 teacher-posterior label export smoke/full wrapper.
- `audit_stage1v2_posterior_1gpu.sh` - Stage-1-v2 best-checkpoint posterior audit wrapper.
- `export_stage1v2_posterior_cache_1gpu.sh` - Stage-1-v2 student posterior cache export wrapper.

## Stage-1 Training

- `train_stage1v2_posterior_4gpu.sh` - trains the teacher-distilled Stage-1-v2 posterior student.
- `cache_esm_lastk_1gpu.sh` - exports last-K ESM feature caches needed by ESM fusion runs.

## Stage-2

- `train_stage2_smoke.sh` - Stage-2 smoke.
- `train_stage2_ddp.sh` - Stage-2 DDP training.
- `export_oracle_motion_features_1gpu.sh` - OracleMotion-UB feature export and direct oracle-apply audit wrapper.
- `export_stage2_teacher_residual_cache_1gpu.sh` - exports free-flow teacher residual targets for endpoint-preserving boundary-residual Stage-2 students.
- `train_stage2_oracle_motion_smoke_1gpu.sh` - Stage-2 smoke with OracleMotion-UB matched/zero/shuffled controls.
- `train_stage2_oracle_motion_ablation_4gpu.sh` - active Stage-2 OracleMotion / ESM last-K / REPA ablation launcher; legacy filename, defaults to 2xA100 on the current cluster and supports matched plus REPA-target-shuffled controls.
- `submit_stage2_md_phase_normal_screen.sh` - submits the frozen-split Path-4 v2
  phase-only, identity-phase residual-only, and full phase-block identification
  screen with one shared data/compute contract; supports explicit independent
  or graph-coupled low-rank residual decoder selection and rank tagging.
- `submit_stage2_phase_block_v2_identification_screen.sh` - submits the Path-4
  v2 phase/rotation/rotation-chi/all block-identification screen, including
  train-scope, gate-bias, confidence-filter, and short-backfill controls.
- `submit_stage2_24k_resumable.sh` - fixed-tag submitter for the 24k OracleMotion / ESM7 / REPA matrix; reuses the interrupted run directories and keeps `AUTO_RESUME=1` so resubmission continues from `last_checkpoint.pt`.
- `evaluate_stage2_transition_paths_1gpu.sh` - residue-level transition evaluator.
- `evaluate_stage2_md_reference_1gpu.sh` - one-GPU held-out MD-reference path evaluator.
- `evaluate_stage2_path4_multistart_screen_1gpu.sh` - frozen-Path-3,
  multi-start projected-normal surrogate screen with endpoint-exact route seeds
  and backtracking acceptance; this is a precursor to, not a substitute for,
  the independent OpenMM Gate-0 validation.
- `evaluate_path4_openmm_gate0_1gpu.sh` - score one candidate from a required
  frozen all-atom frame-reference cache under the OpenMM Gate-0 contract,
  including the frozen post-minimization residue-force validity threshold.
- `run_path4_gate0_paired_rescore_1gpu.sh` - build one Path-3 all-atom frame
  cache, score Path-3 and an optimized candidate from that shared cache, and
  emit the strict paired engineering summary.
- `run_md_global_rmsd_pull_cpu.sh` - biased silver-path pilot; it is not a kinetics workflow.
- `select_md_pilot_candidates_cpu.sh` - CPU endpoint capacity scan with explicit
  apo/holo residue mapping, frozen mapping coverage, and unique-pair accounting.
- `audit_md_candidate_capacity_cpu.sh` - CPU audit of mapping-aware candidates
  after frozen validation/test family-scaffold exclusion and prior endpoint-pair
  removal; emits the novel pool and residue-mismatch smoke manifest.
- `run_md_context_pipeline_array_cpu.sh` - resumable setup, NVT, NPT, and
  context registration array for selected pilot systems.
- `launch_md_corpus_scaleout_after_selection_cpu.sh` - after candidate
  selection, builds a sample- and endpoint-pair-de-duplicated context matrix
  and chains context, collection, replica, and finalization jobs without
  changing scientific gates.
- `run_md_replica_pipeline_array_cpu.sh` - independent-seed silver pull,
  audit, and phase-normal target-export array.
- `continue_md_pilot_after_context_cpu.sh` - after-any continuation that
  collects passed contexts and submits five replicas per admitted system.
- `finalize_md_replica_matrix_cpu.sh` - after-any replica outcome summary and
  passed-target cache assembly.
- `audit_md_rmsd_pull_cpu.sh` / `audit_md_atomistic_path_cpu.sh` - path and atomistic admission audits.
- `export_md_phase_normal_targets_cpu.sh` - export audited phase/normal-residual targets on CPU.
- `export_md_phase_normal_matrix_cpu.sh` - array re-export of a merged MD collection under a selected phase reference.
- `assemble_md_phase_normal_tree_cpu.sh` - dependency-safe assembly of a complete re-export tree into one immutable cache.
- `evaluate_stage2_bridge_timewarp_1gpu.sh` - endpoint-exact bridge time-warp headroom evaluator before training a time-warp model.
- `submit_stage2_phase_specificity_screen.sh` - matched global/residue monotone and residue non-monotone phase controls on the frozen MD split.
- `submit_path3_final_training.sh` - guarded Path-3 final-budget submitter for
  the promoted candidate, matched phase controls, and phase/residual
  decomposition; preserves global batch 16 across 4x4 or 2x8 execution and
  defaults to plan-only mode.
- `evaluate_stage2_trajectory_reliability_1gpu.sh` - internal physical/contact trajectory reliability benchmark against cubic interpolation.
- `run_controlled_manifold_benchmark_cpu.sh` - CPU launcher for the controlled
  phase-normal identifiability benchmark; preferred because the small synthetic
  MLP does not require an A100.
- `run_controlled_manifold_benchmark_1gpu.sh` - optional one-GPU launcher for
  the same benchmark.
- `generate_stage2_trajectories_1gpu.sh` - trajectory export wrapper for visualization.
- `run_ca_morph_baseline_cpu.sh` - simple static/linear/smoothstep CA path baseline plus common CA evaluation.
- `run_anm_baseline_cpu.sh` - CA-ANM projection baseline plus common CA evaluation.
- `run_adaptive_anm_baseline_cpu.sh` - adaptive CA-ANM baseline plus common CA evaluation.
- `run_ebdims2_baseline_cpu.sh` - eBDIMS2 external path baseline plus common CA evaluation.
- `evaluate_ebdims2_ca_paths_cpu.sh` - evaluates already generated CA path baseline manifests.

## Full-Scale Submission Notes

- The active OracleMotion launcher defaults to 2 A100 through its SBATCH header.
  If requesting 4-6 GPUs, pass a matching Slurm resource request and set
  `NPROC_PER_NODE` to the same value.
- Run `PRECHECK_ONLY=1` before a long submission to validate cache shape,
  subset files, amino-acid typing, node masks, and ESM last-K availability.
- For REPA comparisons, compare `val_total_no_repa` plus endpoint/contact/path
  metrics; `val_repa` alone is only a target-fitting diagnostic.

## Archive

- `archive/legacy_stage1/` - older Stage-1 launchers kept for reproducibility, not active use.
- `archive/20260627_pre_fullscale_stage2/` - older exploratory launchers moved
  out of the main path before 60k/full-scale Stage-2 training.
- `archive/20260725_path4_pre_gate0/` - learned physical-normal residual
  distillation launcher retained as a reproducibility record; Gate-0 has not
  authorized its use.
