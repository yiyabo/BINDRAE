# Project Cleanup Log

Date: 2026-06-22

## Cleanup Phase 1

This cleanup follows the Stage-1-v2 pivot:

```text
Stage-1-v2 teacher posterior
-> Stage-2 guided bridge flow
-> known-pose induced-fit trajectory generation
```

The first cleanup phase is intentionally non-destructive:

- preserve all historical evidence;
- move superseded top-level documents into archive;
- keep active docs short and discoverable;
- classify scripts before moving or deleting code;
- remove generated cache noise only.

## Moved To Archive

Moved to:

```text
docs/archive/20260622_pre_stage1v2_pivot/
```

Files:

- `CHANGE_PREDICTION_RAE_FINAL_ASSESSMENT_20260616.md`
- `CHANGE_PREDICTION_RAE_RESCUE_NOTES_20260616.md`
- `RAEV2_PROTEIN_CONFORMATION_FEASIBILITY_ANALYSIS_20260612.md`
- `STAGE1_CANDIDATE_RERANKING_FINAL_DECISION_20260511.md`
- `STAGE1_LIGAND_CAUSALITY_FINAL_ASSESSMENT_20260612.md`
- `STAGE1_LIGAND_CAUSALITY_VALIDATION_DECISION_20260511.md`
- `STAGE1_ROTAMER_ORACLE_DATA_AUDIT_20260617.md`
- `STAGE1_SCREENING_WORKFLOW.md`
- `STAGE1_TYPED_CANDIDATE_INTERACTION_ENERGY_PLAN_20260511.md`

These files are still useful as historical context, but they are no longer the
first reading path for the project.

## Kept At Top Level

- `CURRENT_PROJECT_STATUS_20260622.md`
- `STAGE1V2_TEACHER_POSTERIOR_PLAN_20260622.md`
- `LC_PGBF_ARCHITECTURE_PLAN_20260618.md`
- `LC_PGBF_STAGE2_EXPERIMENT_RECORD_20260620.md`
- `STAGE1_STAGE2_PRIOR_INTERFACE_DECISION.md`
- `STAGE1_INTERACTION_PRIOR_PIVOT_20260617.md`

These define the current direction and the most relevant evidence chain.

## Remaining Cleanup Candidates

Docs:

- `docs/理论/` should be reviewed later and either translated into current
  design docs or moved under archive.
- `docs/RP/` is presentation material and already ignored by `.gitignore`.
- `docs/.spec-workflow/` appears to be workflow scaffolding and should stay out
  of the main reading path.

Scripts:

- many Stage-1 experimental launchers are still top-level under
  `scripts/slurm/`;
- keep them until Stage-1-v2 smoke and training launchers are stable;
- next cleanup should move clearly superseded launchers into
  `scripts/slurm/archive/20260622_pre_stage1v2_pivot/`.

Code:

- no source-code deletion was performed in this phase;
- source cleanup should wait until the Stage-1-v2 posterior encoder and Stage-2
  latent guidance path have working smoke tests.

## Cleanup Phase 2: OracleMotion Baseline Freeze

Date: 2026-06-25

The train12000/e10 OracleMotion Stage-2 upper-bound result is now treated as a
stable baseline, not as an open-ended reliability chase.

Added current source-of-truth docs:

- `docs/ORACLE_MOTION_BASELINE_SNAPSHOT_20260625.md`
- `docs/RAEV2_REPA_STAGE2_ENHANCEMENT_PLAN_20260625.md`

Policy update:

- do not launch more `steps=6/24` reliability, sample-level ranking, or
  visualization jobs by default;
- keep those evaluators available for later paper evidence and debugging;
- start the next code track from representation enhancement:
  ESM last-K fusion and REPA-style hidden-state alignment.

No source files were deleted or moved in this phase. The worktree contains many
active Stage-1 and Stage-2 experiment files, so destructive cleanup should wait
until the RAEv2/REPA enhancement smoke tests define the next stable interface.

## Cleanup Phase 3: Full-Scale Training Readiness

Date: 2026-06-26

The project is preparing to move from small same-budget representation
ablations to larger Stage-2 training and downstream comparison.

Added:

- `docs/FULL_SCALE_TRAINING_AND_EVALUATION_RUNBOOK_20260626.md`
- `docs/REPOSITORY_CLEANUP_POLICY_20260626.md`

Actions:

- documented the current full-scale training gate and evaluation route;
- documented current mainline files and archive candidates;
- documented the Stage-2 multi-forward DDP graph contract after the REPA smoke
  exposed unused-parameter failures;
- removed local generated noise covered by `.gitignore`, including Python
  `__pycache__`, `.pytest_cache`, and `.DS_Store` files.

Validation:

- local `py_compile` passed for Stage-2 model/trainer/config and
  `scripts/train_stage2.py`;
- remote `py_compile` passed in the `BINDRAE` conda environment;
- REPA DDP smoke job `137985` completed on 2 A100 with finite metrics;
- same-budget REPA matched and target-shuffled comparison jobs were submitted
  as `137986` and `137987`.

No source files, experiment launchers, logs, checkpoints, or datasets were
deleted in this phase.

## Cleanup Phase 4: Local Git Hygiene

Date: 2026-06-26

Actions:

- extended `.gitignore` for local agent/browser tool state:
  `.omo/` and `.playwright-mcp/`;
- extended `.gitignore` for common local Python/type-checker caches:
  `.mypy_cache/` and `.ruff_cache/`;
- removed generated local tool directories `.omo/` and `.playwright-mcp/`;
- rechecked that Python cache files, `.DS_Store`, and local test caches are not
  present after cleanup.

Scope boundary:

- no source files were deleted;
- no experiment launchers were deleted;
- no logs, checkpoints, processed data, or dataset artifacts were deleted;
- no files were staged or committed.

Next git cleanup step should be commit-boundary review, not broad deletion:

1. Stage-2 OracleMotion plus ESM/REPA implementation.
2. OracleMotion cache/export/evaluation utilities.
3. Current docs, indexes, runbooks, and cleanup records.
4. Stage-1-v2 posterior work, only if it is intended to remain active.

## Cleanup Phase 5: Commit Boundary Plan

Date: 2026-06-26

Actions:

- added `docs/GIT_STAGING_PLAN_20260626.md`;
- linked the staging plan from `docs/INDEX.md`;
- removed one ignored local backup file: `.qwen/settings.json.orig`;
- confirmed that the remaining untracked top-level items are project assets
  rather than generated tool cache.

Staging policy:

- use explicit file lists or `git add -p`;
- do not use `git add .` on this branch;
- keep Stage-2 OracleMotion/ESM/REPA, Stage-1-v2 posterior, documentation, and
  transitional Stage-1 experiments in separate commits.

No source files, experiment launchers, logs, checkpoints, or datasets were
deleted in this phase.
