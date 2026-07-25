# BINDRAE Project Handoff

Date: 2026-07-25

This is the operational handoff for active work. It does not replace the
frozen experimental contracts linked below. Read this file first when taking
over a task, then follow the lane-specific contract.

## One-Line Scope

```text
known apo + known holo + aligned ligand pose
  -> endpoint-conditioned residue-frame path reconstruction
```

The promoted model is Path-3 phase-only. It is not docking, apo-only holo
prediction, physical kinetics, or an MD trajectory generator.

## Current Lanes

| Lane | Status | Authoritative source | Handoff action |
|---|---|---|---|
| Path-3 anchor | Frozen 241-train / 30-validation / 32-original-test consensus corpus | `PATH3_FINAL_EXPERIMENT_CONTRACT_20260725.md` | Preserve as the learning-curve anchor; do not reselect on the frozen test. |
| AHoJ expansion | 1,769 leakage-clean, historically unattempted endpoint pairs; acquisition pool only | `PATH3_FINAL_EXPERIMENT_CONTRACT_20260725.md` | Finish the 32-system mapping-aware smoke before any larger AHoJ submission. |
| APObind pilot | Complete engineering pilot; 3 consensus targets | `PATH3_FINAL_EXPERIMENT_CONTRACT_20260725.md` | Keep as external-source quality-control evidence; do not merge into the Path-3 training manifest. |
| Path-4 OpenMM Gate-0 | Engineering contract implemented; Gate-0 unpassed | `PATH4_OPTIMIZER_GATE0_20260720.md` | Maintain the frozen scorer/reference contract; do not promote or scale this lane. |

## Active AHoJ Smoke

The current direct GPU33 run is a 32-system x 2-replica mapping-aware AHoJ
engineering smoke. It selects only candidates with unequal apo/holo residue
counts, exact common-residue mapping fraction at least 0.95, and deterministic
coverage across motion categories. It is not a final corpus run.

Remote output root:

```text
processed_data/md_transition/ahoj_mapping_smoke32x2_gpu33_20260725_v1/
```

Completion source of truth:

```text
processed_data/md_transition/ahoj_mapping_smoke32x2_gpu33_20260725_v1/smoke_state.json
```

Launcher log:

```text
logs/stage2/ahoj_mapping_smoke32x2_gpu33_20260725_v1/launcher.log
```

Execution sequence:

```text
freeze 32-system panel
  -> OpenMM setup/NVT/NPT context
  -> retain passed contexts only
  -> two independent fixed-protocol pull replicas
  -> pull/path/atomistic/mapping/target gates
  -> finalization and two-replica consensus
```

The only valid outputs are passed targets in systems with at least two accepted
replicas. Rejections remain recorded and are not training examples.

Relevant entrypoints:

- `scripts/select_md_mapping_smoke_panel.py`
- `scripts/run_ahoj_mapping_smoke_gpu33.sh`
- `scripts/run_md_context_pipeline.py`
- `scripts/run_md_replica_pipeline.py`
- `scripts/finalize_md_replica_matrix.py`

## Frozen Decisions

- Do not reopen direct instantaneous force-to-MD-normal-residual regression:
  `force_signal_supported=false`.
- Do not weaken the 0.95 exact common-residue mapping gate, frozen holdout
  leakage exclusions, physical gates, or consensus requirement.
- Endpoint-only records are not direct Path-3 supervision. The prior broad
  endpoint-only pretraining route is closed.
- The 1,769 AHoJ pairs do not establish that a 2,000-consensus-system campaign
  is feasible. The planning gate remains 3,000 novel pairs, a shortfall of
  1,231 pairs.
- Path-4 development summaries and CPU smoke are engineering evidence, not
  Gate-0 efficacy or CUDA scientific results.
- The learned physical-normal teacher-distillation launcher is archived at
  `scripts/slurm/archive/20260725_path4_pre_gate0/`; do not relaunch it before
  a documented Gate-0 reopening decision.
- The frozen test sets are not available for data-size selection, threshold
  tuning, or repeated model selection.

## APObind Boundary

The completed 8-system x 2-replica APObind pilot had 7 accepted replica targets
after cache recovery and 3 two-replica consensus systems. Its phase component is
stable, but residual-valid interior coverage is sparse and heterogeneous. The
three targets are immutable quality-control artifacts only; they are excluded
from the expansion training manifest and do not close the AHoJ planning-pool
shortfall.

## Path-4 Boundary

Path-4 compares the frozen Path-3 guide with a non-learned multi-start,
endpoint-exact OpenMM optimization path using a shared all-atom reference cache.
Reference topology preflight, cache identity, scorer contract, endpoint-energy
equality, and relaxed-frame validity are hard prerequisites. The available small
development panel is insufficient and Gate-0 remains unpassed.

## Worktree And Operations

The worktree intentionally contains a large uncommitted experimental batch.
Do not delete, move, reset, or bulk-stage files while handing off. In particular:

- never use `git add .`;
- keep future commits separated by Path-3 acquisition, Path-4 Gate-0, and docs;
- treat generated data and logs as remote artifacts, not code changes;
- use `gpu33pw` for bounded direct GPU33 work and Slurm only through current
  launchers for cluster work;
- do not turn the agent into a queue/watchdog poller.

## Next Delegable Tasks

1. Monitor the AHoJ smoke only through its state file or a bounded log check;
   on completion, report context, replica, target, and consensus counts with
   every rejection reason.
2. If the smoke passes without a systemic engineering failure, write a factual
   throughput/yield memo before deciding whether to submit more AHoJ systems.
3. Independently resolve the 1,231-pair AHoJ planning shortfall without relaxing
   mapping, leakage, or historical-attempt exclusions.
4. Do not start Path-3 500/1k/2k training until accepted consensus-system counts
   and a fresh family/scaffold-disjoint split exist.

## Required Reading By Lane

- Path-3 data and final evidence: `PATH3_FINAL_EXPERIMENT_CONTRACT_20260725.md`
- Path-4 physical optimizer: `PATH4_OPTIMIZER_GATE0_20260720.md`
- Method definition: `BINDRAE_CONFERENCE_METHOD_BLUEPRINT_20260710.md`
- Current implementation history: `CURRENT_PROJECT_STATUS_20260710.md`
- Script ownership: `../scripts/AGENTS.md` and `../scripts/INDEX.md`
