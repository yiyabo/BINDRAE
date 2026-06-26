# BINDRAE Agent Guide

## Project shape
BINDRAE is a two-stage protein-ligand induced-fit system for known ligand poses. Stage-1 learns a ligand-conditioned holo prior / pocket posterior from apo structure, sequence embeddings, and ligand tokens. Stage-2 learns an apo-to-holo conditional bridge flow on per-residue `SE(3)` frames and side-chain chi angles. The main implementation lives under `src/stage1/`, `src/stage2/`, and `scripts/`; `CLIProxyAPI/` is an independent nested Go project with its own `AGENTS.md`.

The main scientific contract is apo protein plus aligned ligand pose to holo endpoint/path. Do not reframe the repository as docking unless the task explicitly asks for pose-search work. The code and docs assume ligand coordinates are already in the apo frame or consistently aligned into it.

## Operating constraints
Never run Python training directly on the Beijing supercomputing login node. Use Slurm launchers through `sbatch`, with `/data/soft/slurm/24.11.4/bin` added to `PATH`, and activate the remote `BINDRAE` conda env first. Local edits can be syntax-checked here, but GPU training and large diagnostics belong on the cluster under `/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE`.

**GPU 资源使用原则**：默认优先使用 **2-3 张 A100** 做多 GPU 训练。当前集群经常有零散空闲卡，但单节点一次空出 4 张以上 A100 不稳定，直接申请 4+ 卡容易长时间 Pending。Slurm 训练脚本应优先请求 `--gres=gpu:A100:2` 或 `--gres=gpu:A100:3`，并用 `torchrun` 或 DDP 分布式训练。只有在确认单节点有连续空闲卡、需要更高吞吐的正式长训、或用户明确要求时，才申请 4-8 卡。快速 smoke test 或资源紧张时可以使用单卡。

When using SSH examples from `CLAUDE.md`, treat them as operational context and avoid putting credentials into new scripts or committed documentation. Prefer one-off shell commands for monitoring jobs and logs.

## Important directories
- `src/stage1/`: holo prior / ligand-conditioned pocket rotamer-contact modeling. See its local guide before changing losses, candidate scoring, or FK outputs.
- `src/stage2/`: conditional bridge flow using Stage-1 guidance, path integration, and path-level geometry losses. See its local guide before changing flow state or priors.
- `scripts/`: training, diagnostics, preprocessing, and Slurm wrappers. See its local guide before launching jobs or adding CLI flags.
- `docs/`: experiment rationale and design decisions. Start from `docs/CURRENT_PROJECT_STATUS_20260622.md`, then read the active Stage-1-v2 and Stage-2 records. Archived files are scientific context, not executable truth.
- `data/`, `processed_data/`, `logs/`, `checkpoints/`, `tmp/`: generated or large artifacts. Do not casually rewrite, commit, or recursively scan them unless the task is explicitly about data or results.
- `reference/` and `legacy/`: borrowed or historical code. Prefer wrapping or comparing against it instead of modifying it in-place.

`proposal/`, `RP/`, and large merged documents may contain presentation or historical material; use them for context only after checking fresher files under `docs/` and active code under `src/`.

## Code navigation priorities
For Stage-1 work, start from `src/stage1/training/trainer.py`, `src/stage1/models/stage1_model.py`, and `src/stage1/models/torsion_head.py`. For Stage-2 work, start from `src/stage2/training/trainer.py`, `src/stage2/models/torsion_flow.py`, and `src/stage2/modules/se3.py`. For experiment launch behavior, inspect the exact Slurm script under `scripts/slurm/` rather than assuming CLI defaults.

## Scientific assumptions to preserve
The default task is known-pose induced fit, not docking. Ligand coordinates are expected in the apo frame or aligned consistently to it. The structural state is per-residue backbone frames plus chi angles; do not introduce redundant backbone torsion state unless the design docs are intentionally being revised. FK and losses should remain global-SE(3)-consistent.

Angles live on `S^1`; use wrap-aware differences or sin/cos representations. Frame operations should remain consistent with the local SE(3) utilities instead of ad-hoc matrix math. Ligand effects should be evaluated on biologically relevant subsets such as contact, ligand-facing, switch, and pocket residues.

## Current Stage-1 direction
Stage-1-v2 should be a teacher-distilled local posterior encoder, not a hard holo-rotamer endpoint model. The current active plan is `docs/STAGE1V2_TEACHER_POSTERIOR_PLAN_20260622.md`: export local teacher posterior labels, train a compact posterior encoder, and use its scalar/latent guidance to improve Stage-2 path generation.

Older diagnostics showed that raw global chi1 accuracy can be dominated by a Dunbrack-like base prior, not ligand-causal learning. Keep evaluating ligand effects with lift-over-base, decoy/shuffled-ligand controls, contact/switch subsets, rescue/harm metrics, and calibrated local posterior quality.

## Stage-2 interface posture
Older notes may describe Stage-1 as a deterministic endpoint anchor. Newer diagnostics in `docs/STAGE1_STAGE2_PRIOR_INTERFACE_DECISION.md` recommend softer local posterior/contact information and caution against rigid priors. Keep Stage-2 prior changes compatible with zero-prior fallback and explicit ablations.

As of 2026-06-25, the strongest Stage-2 evidence is the frozen OracleMotion upper-bound baseline in `docs/ORACLE_MOTION_BASELINE_SNAPSHOT_20260625.md`. Treat it as the current stable starting point. Do not launch additional reliability step sweeps, sample ranking, or visualization jobs unless the user explicitly asks; the next default engineering track is `docs/RAEV2_REPA_STAGE2_ENHANCEMENT_PLAN_20260625.md` with ESM last-K fusion and REPA-style alignment.

## Validation habits
After Python edits, run targeted import or compile checks first, for example `python -m py_compile <changed files>`. For shell launchers, run `bash -n <script>`. For training changes, use smoke or diagnostic Slurm scripts before long runs, then inspect `logs/slurm/` and JSONL metrics under `logs/stage1/` or `logs/stage2/`.

Local syntax checks are acceptable in this checkout. Full dataset training, multi-GPU DDP, and long diagnostics should be submitted remotely. When a validation command fails because an optional dependency is unavailable locally, record that fact and still run static checks that do not require GPU or cluster-only data.

## Experiment hygiene
Use unique tags for new runs. Do not overwrite anchor checkpoints or logs. Selection metrics must match the scientific intent: ligand-lift metrics for ligand-causality experiments, path losses and endpoint/path quality for Stage-2, and smoke metrics only for pipeline health.

When resuming or branching, keep separate `save_dir` and `log_dir` unless intentionally continuing the same run. Do not use the test split for model selection; screening should use validation lanes and short budgets. The old Stage-1 screening workflow is archived under `docs/archive/20260622_pre_stage1v2_pivot/` and should be treated as historical context.

## Remote cluster quick facts
Remote project root: `/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE`. Slurm logs: `logs/slurm/`. Stage logs: `logs/stage1/`, `logs/stage2/`. Datasets are under `processed_data/triplets/`; some SDF files may be corrupt and may require ligand preprocessing rather than trainer workarounds.

Typical Slurm launchers set `PYTHONUNBUFFERED=1`, `OMP_NUM_THREADS=4`, one node, one task per node, and A100 GPUs. Monitor jobs with `squeue`, then inspect the matching Slurm `.out/.err` and metrics JSONL before drawing conclusions.
