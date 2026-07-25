# BINDRAE Agent Guide

## Project shape
BINDRAE is a protein-ligand conformational-path project for known ligand poses. The active paper track learns endpoint-conditioned apo-to-holo paths on per-residue `SE(3)` frames and side-chain chi angles. Its current method is an endpoint-exact asynchronous phase-normal bridge: a monotone residue phase controls progress along an analytic endpoint bridge, while a metric-orthogonal residual models off-bridge motion. Stage-1 is a future deployment track that will replace privileged endpoint-derived conditioning with an apo-and-ligand-conditioned posterior. The main implementation lives under `src/stage1/`, `src/stage2/`, and `scripts/`; `CLIProxyAPI/` is an independent nested Go project with its own `AGENTS.md`.

The active scientific contract is known apo and holo endpoints plus an aligned ligand pose to an ordered conformational path. Do not reframe the repository as docking, apo-only holo prediction, or physical MD generation unless the task explicitly changes scope. The code and docs assume ligand coordinates are already in the apo frame or consistently aligned into it.

## Operating constraints
Never run Python training directly on the Beijing supercomputing login node. Use Slurm launchers through `sbatch`, with `/data/soft/slurm/24.11.4/bin` added to `PATH`, and activate the remote `BINDRAE` conda env first. Local edits can be syntax-checked here, but GPU training and large diagnostics belong on the cluster under `/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE`.

**GPU 资源使用原则**：默认优先使用 **2-3 张 A100** 做多 GPU 训练。当前集群经常有零散空闲卡，但单节点一次空出 4 张以上 A100 不稳定，直接申请 4+ 卡容易长时间 Pending。Slurm 训练脚本应优先请求 `--gres=gpu:A100:2` 或 `--gres=gpu:A100:3`，并用 `torchrun` 或 DDP 分布式训练。只有在确认单节点有连续空闲卡、需要更高吞吐的正式长训、或用户明确要求时，才申请 4-8 卡。快速 smoke test 或资源紧张时可以使用单卡。

When using SSH examples from `CLAUDE.md`, treat them as operational context only. Never put credentials, private keys, or passwords into scripts, commits, docs, or chat. Prefer SSH host aliases from the local SSH config over raw user@IP:port strings in commands.

**Remote ops hygiene is mandatory** (see the full section below). High-frequency SSH/Slurm automation has repeatedly tripped API cyber-security policy blocks on legitimate BINDRAE work. Optimize for few, purposeful remote commands—not continuous cluster piloting inside the chat agent.

## Important directories
- `src/stage1/`: holo prior / ligand-conditioned pocket rotamer-contact modeling. See its local guide before changing losses, candidate scoring, or FK outputs.
- `src/stage2/`: conditional bridge flow using Stage-1 guidance, path integration, and path-level geometry losses. See its local guide before changing flow state or priors.
- `scripts/`: active training, diagnostics, preprocessing, and Slurm wrappers. See its local guide before launching jobs or adding CLI flags. Archived scripts under `scripts/archive/` and `scripts/slurm/archive/` are reproducibility records, not templates for new experiments.
- `docs/`: experiment rationale and design decisions. Start from `docs/BINDRAE_CONFERENCE_METHOD_BLUEPRINT_20260710.md` and `docs/CURRENT_PROJECT_STATUS_20260710.md`. Archived files are scientific context, not executable truth.
- `data/`, `processed_data/`, `logs/`, `checkpoints/`, `tmp/`: generated or large artifacts. Do not casually rewrite, commit, or recursively scan them unless the task is explicitly about data or results.
- `reference/` and `legacy/`: borrowed or historical code. Prefer wrapping or comparing against it instead of modifying it in-place.

`proposal/`, `RP/`, and large merged documents may contain presentation or historical material; use them for context only after checking fresher files under `docs/` and active code under `src/`.

## Code navigation priorities
For Stage-1 work, start from `src/stage1/training/trainer.py`, `src/stage1/models/stage1_model.py`, and `src/stage1/models/torsion_head.py`. For Stage-2 work, start from `src/stage2/training/trainer.py`, `src/stage2/models/torsion_flow.py`, `src/stage2/modules/phase_residual.py`, and `src/stage2/modules/se3.py`. For experiment launch behavior, inspect the exact Slurm script under `scripts/slurm/` rather than assuming CLI defaults.

## Scientific assumptions to preserve
The default task is known-pose induced fit, not docking. Ligand coordinates are expected in the apo frame or aligned consistently to it. The structural state is per-residue backbone frames plus chi angles; do not introduce redundant backbone torsion state unless the design docs are intentionally being revised. FK and losses should remain global-SE(3)-consistent.

Angles live on `S^1`; use wrap-aware differences or sin/cos representations. Frame operations should remain consistent with the local SE(3) utilities instead of ad-hoc matrix math. Ligand effects should be evaluated on biologically relevant subsets such as contact, ligand-facing, switch, and pocket residues.

## Current Stage-1 direction
Stage-1 is not the active paper bottleneck. It remains the future no-Oracle deployment track: predict a calibrated local motion/contact posterior and an endpoint distribution from apo structure, sequence, and ligand information, then condition the path model on those predictions. Do not restart the archived hard-holo or broad teacher-posterior sweeps unless the user explicitly resumes that track.

Older diagnostics showed that raw global chi1 accuracy can be dominated by a Dunbrack-like base prior, not ligand-causal learning. When Stage-1 resumes, keep evaluating ligand effects with lift-over-base, decoy/shuffled-ligand controls, contact/switch subsets, rescue/harm metrics, and calibrated local posterior quality.

## Stage-2 interface posture
The active parameterization is `phase_orthogonal_residual_v1`. Preserve its scientific decomposition:

- `tau_i(t)` is monotone and controls along-bridge progress;
- the rigid/chi residual is projected into the product-metric normal space;
- an endpoint-zero envelope and explicit endpoint insertion guarantee exact boundaries;
- residual magnitude, temporal, neighbor, and background terms keep the correction controlled.

The frozen OracleMotion snapshot in `docs/ORACLE_MOTION_BASELINE_SNAPSHOT_20260625.md` remains historical evidence, not the current architecture. OracleMotion is privileged endpoint-derived conditioning and contains no MD intermediate trajectory. ESM fusion and REPA are secondary representation studies, not headline novelty.

Before any full-scale run, complete the matched four-model screen: synchronous bridge, warp-only, residual-only, and full phase-normal bridge. Select using path/contact/event-order metrics and independent physical validity, not endpoint error alone. The proposed stochastic global path latent is documented but not implemented; do not present it as completed work.

## Validation habits
After Python edits, run targeted import or compile checks first, for example `python -m py_compile <changed files>`. For shell launchers, run `bash -n <script>`. For training changes, use smoke or diagnostic Slurm scripts before long runs, then inspect `logs/slurm/` and JSONL metrics under `logs/stage1/` or `logs/stage2/`.

Local syntax checks are acceptable in this checkout. Full dataset training, multi-GPU DDP, and long diagnostics should be submitted remotely. When a validation command fails because an optional dependency is unavailable locally, record that fact and still run static checks that do not require GPU or cluster-only data.

## Experiment hygiene
Use unique tags for new runs. Do not overwrite anchor checkpoints or logs. Selection metrics must match the scientific intent: ligand-lift metrics for ligand-causality experiments, path losses and endpoint/path quality for Stage-2, and smoke metrics only for pipeline health.

When resuming or branching, keep separate `save_dir` and `log_dir` unless intentionally continuing the same run. Do not use the test split for model selection; screening should use validation lanes and short budgets. The old Stage-1 screening workflow is archived under `docs/archive/20260622_pre_stage1v2_pivot/` and should be treated as historical context.

For multi-agent collaboration, do not use `git add .`. Keep commits split by
topic: Stage-2 training/evaluation code, Stage-1-v2 posterior code, documentation
and archive moves. If a file is already dirty and outside the current task,
inspect it before editing and preserve unrelated changes.

## Remote cluster quick facts
Remote project root: `/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE`. Slurm logs: `logs/slurm/`. Stage logs: `logs/stage1/`, `logs/stage2/`. Datasets are under `processed_data/triplets/`; some SDF files may be corrupt and may require ligand preprocessing rather than trainer workarounds.

Typical Slurm launchers set `PYTHONUNBUFFERED=1`, `OMP_NUM_THREADS=4`, one node, one task per node, and A100 GPUs. After `sbatch`, use a small number of purposeful `squeue`/`sacct` and log checks to confirm scheduling or diagnose a result. An occasional user-scoped queue overview is allowed when the active JobID is unknown. Do not enter a poll loop from the agent.

## Remote ops / anti false-positive cyber policy
Legitimate BINDRAE HPC work (SSH to the login host, `sbatch`, log tails, `rsync` of code) is allowed. What is **not** allowed is agent behavior that looks like automated intrusion or fleet control. Past sessions were blocked with `cyber_policy` / `该会话已被网络安全策略屏蔽` after dense SSH+Slurm loops—not because the science was illegal.

### Role split
- **Agent primary job**: edit local code/docs/tests; write or patch `scripts/slurm/*.sh`; run local `py_compile` / `bash -n` / unit tests.
- **Agent remote role**: perform bounded queue discovery, submit through launchers, and inspect job-scoped status/logs when that advances the task.
- **User or a cluster-side script** owns long waits, periodic monitoring, and overnight watchdog behavior.

### Operating guidelines
1. **Keep remote work bounded**: each user turn permits at most 10 total
   `ssh`/`scp`/`rsync` calls. Batch related checks and use only the calls needed
   for the current decision; the cap is not a target. Avoid a long stream of
   tiny remote calls.
2. **No poll loops**: never run `sleep N; ssh ... squeue/sacct/scontrol` in a loop. Re-check a job only when enough time has passed, its state may have changed, or the user explicitly asks.
3. **Queue discovery is allowed**: an occasional `squeue -u "$USER"` overview is acceptable when discovering active jobs. Prefer `squeue -j <JobID>`, `sacct -j <JobID>`, and matching logs once a JobID is known.
4. **Avoid heavy cluster enumeration**: do not dump all node details with `scontrol show nodes`, full `sinfo -N`, partition configuration, or accounting associations unless scheduling or cluster diagnosis actually requires it.
5. **Avoid lifecycle churn**: a normal sync -> submit -> initial status/log check is allowed. Do not repeatedly `scancel` -> edit -> `sbatch` -> poll -> `scancel` in one session.
6. **Batch file sync**: sync the changed file set together rather than pushing after every single-line edit.
7. **SSH posture**:
   - Prefer configured host aliases (for example `gpu33pw` / project aliases in `~/.ssh/config`).
   - Do **not** default to `StrictHostKeyChecking=no`.
   - Do **not** paste passwords, keys, or `sshpass` into commands.
   - Avoid embedding raw private IPs and nonstandard ports in every command when an alias exists.
8. **No login-node heavy compute**: no long `python`/`torchrun` training or full-dataset work on the login node; use `sbatch` (same as above).

### Preferred remote patterns
```bash
# Good: one sync + one submit + one status
rsync -azR path/to/changed_files... <host>:/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE/
ssh <host> 'cd /mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE && PATH=/data/soft/slurm/24.11.4/bin:$PATH sbatch scripts/slurm/<launcher>.sh'
ssh <host> 'PATH=/data/soft/slurm/24.11.4/bin:$PATH squeue -j <JobID> -o "%.18i %.12P %.20j %.8T %.10M %R"'

# Also acceptable when the active JobID is unknown
ssh <host> 'PATH=/data/soft/slurm/24.11.4/bin:$PATH squeue -u "$USER" -o "%.18i %.12P %.20j %.8T %.10M %R"'
```

```bash
# Bad: agent-as-watchdog (this pattern has triggered session blocks)
sleep 20; ssh ... squeue ...
sleep 20; ssh ... squeue ...
ssh ... 'scontrol show nodes -o'
ssh ... 'scancel ...'; sbatch ...'; squeue ...'
```

### When blocked or flagged
If the API returns `cyber_policy`, `flagged for possible cybersecurity risk`, or `该会话已被网络安全策略屏蔽` / `403` from the model gateway:
1. **Stop** further SSH/Slurm tool calls in that session.
2. **Do not fork** the blocked thread to “continue”; start a **new clean session**.
3. Summarize local progress and exact JobIDs/paths for the user; let them re-check the cluster or paste logs.
4. Resume with a smaller remote footprint (local edits first, single submit).

### What to tell the user instead of polling
After submit and an initial check, report the JobID, launcher path, log path pattern under `logs/slurm/`, and what “done” looks like. If the job is merely waiting, stop active monitoring and resume after a later user request or a meaningful state-change interval.
