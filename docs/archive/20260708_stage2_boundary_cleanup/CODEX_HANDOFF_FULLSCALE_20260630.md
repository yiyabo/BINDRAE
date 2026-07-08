# Codex Handoff: BINDRAE Full-Scale Stage-2 Training

## Project

BINDRAE is a two-stage protein-ligand induced-fit system. Stage-2 learns an apo-to-holo conditional bridge flow on per-residue SE(3) frames and side-chain chi angles, conditioned on OracleMotion features, ESM last-7 gated_residual fusion, and ligand tokens.

- Local repo: `/Users/apple/code/BINDRAE`
- Remote canonical worktree: `/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE_clean_stage2_repa`
- SSH alias: `gpu33pw`
- Slurm: `export PATH=/data/soft/slurm/24.11.4/bin:$PATH`
- Conda: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate BINDRAE`
- Python: `/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE/bin/python`
- Cluster nodes: `gpu34-gpu55` (Slurm-managed), `gpu33` (direct-login, non-Slurm)
- Do NOT run training on the login node. Use `sbatch` for all GPU work.

## Current Mainline Configuration (LOCKED)

```text
Stage-2 OracleMotion + ESM last-7 gated_residual + gate_bias=-2.0 + noREPA
24k train / val2048 / e10 / 4xA100 / batch_per_gpu=4 / global_batch=16
```

Best 24k result (gm2, completed):
```text
best epoch 9: val_total_no_repa = 5.1901, val_fm = 3.1623
checkpoint: checkpoints/stage2/stage2_stage2_24k_gm2_gated_m2_norepa_esm7sync_train24000_val2048_e10_bs4x4_20260630_054826/best_model.pt
```

12k comparison (val512, less strict):
```text
12k gm2 best: val_total_no_repa = 5.1625, val_fm = 3.3156
```

24k did not beat 12k on val_total_no_repa (val2048 is stricter), but 24k val_fm is significantly better (3.16 vs 3.32), and 24k showed no overfitting at e9 (still descending), while 12k overfit after epoch 6.

## Currently Running Jobs (as of 2026-06-30 23:20 CST)

### OracleMotion Full Cache Export (5 jobs)

4 training shards + 1 validation, all PENDING:

```text
138498  om_train_shard0  PENDING  1xA100  shard 0/4 of train_samples_full.txt
138499  om_train_shard1  PENDING  1xA100  shard 1/4
138500  om_train_shard2  PENDING  1xA100  shard 2/4
138501  om_train_shard3  PENDING  1xA100  shard 3/4
138502  om_val_full      PENDING  1xA100  full test_valid.txt (4297 samples)
```

All export to:
```text
logs/stage2_oracle_motion/oracle_motion_train_full_esm7sync_20260630/   (train, 4 shards)
logs/stage2_oracle_motion/oracle_motion_val_full_esm7sync_20260630/     (val)
```

Export config: `ESM_NUM_LAYERS=7`, `SKIP_BAD_SAMPLES=1`, `SKIP_DIRECT_APPLY=1`, `MAX_SAMPLES=0` (all samples).

Each shard writes its own `manifest.json` (shard-specific). After all 4 complete, manifests must be merged into one.

### Path Evaluator (1 job, may be completed)

```text
138497  s2_gm2_eval2048  RUNNING  gpu42  1xA100
```

Running `evaluate_stage2_transition_paths.py` on 2048 val samples with gm2 best_model.pt. Output at:
```text
logs/stage2/transition_eval/stage2_stage2_24k_gm2_gated_m2_norepa_esm7sync_train24000_val2048_e10_bs4x4_20260630_054826_maxb2048.json
```

## Completed Jobs

```text
138444  s2_24_single_resume   COMPLETED  e10 done  best val 5.2413
138445  s2_24_gm2_resume      COMPLETED  e10 done  best val 5.1901
138446  s2_24_gm15_resume     COMPLETED  e10 done  best val ~5.20 (check metrics)
138496  s2_gm2_path_eval3     COMPLETED  64-sample quick eval (pipeline verified)
```

## Code Changes Already Made (uncommitted locally, synced to remote)

### New files

- `scripts/slurm/submit_stage2_24k_resumable.sh` — fixed-tag submitter for 24k matrix, pins TAG/SAVE_DIR/LOG_DIR and sets AUTO_RESUME=1 per variant. Synced to remote.

### Modified files

- `scripts/slurm/evaluate_stage2_transition_paths_1gpu.sh` — added STAGE1V2_MODE/STAGE1V2_CACHE_DIR/STAGE1V2_FEATURES/STAGE1V2_FEATURE_SCALE params, made ROOT overridable. Synced to remote.
- `scripts/slurm/evaluate_stage2_trajectory_reliability_1gpu.sh` — made ROOT overridable. Synced to remote.
- `scripts/slurm/generate_stage2_trajectories_1gpu.sh` — made ROOT overridable. Synced to remote.
- `scripts/slurm/export_oracle_motion_features_1gpu.sh` — added SHARD_ID/NUM_SHARDS params, made ROOT overridable. Synced to remote.
- `scripts/export_oracle_motion_features.py` — added `--shard_id` and `--num_shards` args, skip non-shard samples in iter_batches_skip_bad. Synced to remote.
- `scripts/evaluate_stage2_transition_paths.py` — added `esm_gate_bias` and `esm_gate_context_dim` to `build_model_config_for_checkpoint`, added `esm_num_layers` to `create_stage2_dataloader` call. Synced to remote.
- `scripts/evaluate_stage2_trajectory_reliability.py` — added `esm_num_layers` to `create_stage2_dataloader` call. Synced to remote.
- `docs/FULL_SCALE_TRAINING_AND_EVALUATION_RUNBOOK_20260626.md` — updated baseline description, added resumable submitter references.
- `scripts/slurm/INDEX.md` — added submit_stage2_24k_resumable.sh entry.

### Uncommitted docs (pre-existing)

```text
M docs/COMPETITOR_DATASETS_METRICS_20260624.md
M docs/INDEX.md
?? docs/BINDRAE_POSITIONING_AND_BENCHMARK_STRATEGY_20260629.md
?? docs/STAGE2_GATED_REPRESENTATION_EXPERIMENTS_20260629.md
```

## Tasks for Codex

### Task 1: Monitor and Merge Full Cache Export

Check if the 5 export jobs (138498-138502) have completed:

```bash
ssh gpu33pw 'export PATH=/data/soft/slurm/24.11.4/bin:$PATH; sacct -j 138498,138499,138500,138501,138502 --format=JobID,JobName%28,State,Elapsed,ExitCode -P'
```

If completed, merge the 4 train shard manifests into one:

```python
# Merge logic: load all 4 shard manifest.json files, combine records,
# write merged manifest.json to the same output dir.
# Each shard manifest has "records" list and "bad_samples" list.
# Merge all records, merge all bad_samples, update counts.
# Save as manifest.json (overwrite or manifest_merged.json).
```

Also check val export (138502) completed and has manifest.json.

Expected output dirs:
```text
logs/stage2_oracle_motion/oracle_motion_train_full_esm7sync_20260630/manifest.json  (merged)
logs/stage2_oracle_motion/oracle_motion_val_full_esm7sync_20260630/manifest.json
```

Expected sample counts:
```text
train: ~73022 total in train_samples_full.txt, minus bad samples
val:   ~4297 total in test_valid.txt, minus bad samples
```

### Task 2: Generate Full-Scale Training Subset Files

Create ablation subset files for full-scale training:

```bash
# Train: use all valid samples from the merged manifest
# Val: use all valid samples from the val manifest
# Write to:
# processed_data/triplets/ablation_subsets/stage2_oracle_motion_esm7sync20260630_full_train.txt
# processed_data/triplets/ablation_subsets/stage2_oracle_motion_esm7sync20260630_full_val.txt
```

The subset files should contain one sample_id per line, matching the manifest records.

### Task 3: Submit Full-Scale Training

Add a `full` variant to `scripts/slurm/submit_stage2_24k_resumable.sh` (or create a new submitter) with:

```text
TRAIN_N=0 (or the actual count, e.g. 65000)
VAL_N=0 (or the actual val count, e.g. 4000)
MAX_EPOCHS=20  (24k was still descending at e10, so give more room)
BATCH_SIZE=4
NPROC_PER_NODE=4
AUTO_RESUME=1
ESM_FUSION_ENABLED=1
ESM_NUM_LAYERS=7
ESM_FUSION_MODE=gated_residual
ESM_GATE_BIAS=-2.0
ESM_GATE_CONTEXT_MODE=none
ESM_LAYER_ENTROPY_WEIGHT=0.02
REPA_ENABLED=0
REPA_WEIGHT=0.0
STAGE1V2_MODE=oracle_motion
STAGE1V2_TRAIN_CACHE_DIR=logs/stage2_oracle_motion/oracle_motion_train_full_esm7sync_20260630
STAGE1V2_VAL_CACHE_DIR=logs/stage2_oracle_motion/oracle_motion_val_full_esm7sync_20260630
TAG=stage2_full_gm2_gated_m2_norepa_esm7sync_trainfull_valfull_e20_bs4x4_20260630
SAVE_DIR=checkpoints/stage2/<TAG>
LOG_DIR=logs/stage2/<TAG>
```

Submit with 4xA100, `--time=18:00:00`, `--mem=300G`.

IMPORTANT: Use `AUTO_RESUME=1` and fixed TAG/SAVE_DIR so interrupted runs can resume.

### Task 4: Run Trajectory Reliability Evaluator

After full-scale training has a checkpoint (or use 24k gm2 as interim), submit:

```bash
sbatch --job-name=s2_traj_rely \
  --gres=gpu:A100:1 --cpus-per-task=4 --mem=80G --time=04:00:00 \
  --export=ALL,ROOT=<remote_root>,CHECKPOINT=<gm2_best_model.pt>,\
VALID_SAMPLES_FILE=ablation_subsets/stage2_oracle_motion_esm7sync20260629_val_2048_seed20260629.txt,\
STAGE1V2_MODE=oracle_motion,\
STAGE1V2_CACHE_DIR=logs/stage2_oracle_motion/oracle_motion_val2048_esm7sync_20260629,\
VAL_N=512,MAX_BATCHES=512,N_INTEGRATION_STEPS=12 \
  scripts/slurm/evaluate_stage2_trajectory_reliability_1gpu.sh
```

This compares BINDRAE paths against cubic SE(3)+chi interpolation baseline.

### Task 5: Commit Code Changes

When user approves, commit all code changes split by topic:

```text
1. Stage-2 evaluator fixes (esm_gate_bias, esm_num_layers in dataloader):
   scripts/evaluate_stage2_transition_paths.py
   scripts/evaluate_stage2_trajectory_reliability.py
   scripts/slurm/evaluate_stage2_transition_paths_1gpu.sh
   scripts/slurm/evaluate_stage2_trajectory_reliability_1gpu.sh
   scripts/slurm/generate_stage2_trajectories_1gpu.sh

2. OracleMotion export sharding:
   scripts/export_oracle_motion_features.py
   scripts/slurm/export_oracle_motion_features_1gpu.sh

3. Resumable training submitter:
   scripts/slurm/submit_stage2_24k_resumable.sh
   scripts/slurm/INDEX.md

4. Docs:
   docs/FULL_SCALE_TRAINING_AND_EVALUATION_RUNBOOK_20260626.md
   docs/BINDRAE_POSITIONING_AND_BENCHMARK_STRATEGY_20260629.md
   docs/STAGE2_GATED_REPRESENTATION_EXPERIMENTS_20260629.md
   docs/INDEX.md
   docs/COMPETITOR_DATASETS_METRICS_20260624.md
```

Do NOT use `git add .`. Stage files by topic.

## Key Constraints

1. Never run training/eval on the login node. Use `sbatch`.
2. Never set `AUTO_RESUME=0` for long training jobs.
3. Never use `git add .`. Split commits by topic.
4. Slurm jobs may be cancelled by system (UID 0) at ~5h10m. Use resumable submitter.
5. `gpu33` is non-Slurm, direct-login. `gpu34-gpu55` are Slurm-only.
6. Default to 2-4 A100. Only request 4+ when confirmed available.
7. Do not overwrite anchor checkpoints or logs.
8. Path evaluator requires `esm_num_layers` in dataloader call — this was a bug, now fixed.
9. Slurm `--export` with comma-separated values breaks if a value contains commas (e.g. STAGE1V2_FEATURES). Let the evaluator read from checkpoint config instead.
10. Paper strategy: endpoint-conditioned path reconstruction, NOT static endpoint prediction or docking. See `docs/BINDRAE_POSITIONING_AND_BENCHMARK_STRATEGY_20260629.md`.

## Key Files

```text
Training:
  scripts/train_stage2.py                         — training CLI
  scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh — main 4GPU launcher
  scripts/slurm/submit_stage2_24k_resumable.sh    — fixed-tag resumable submitter

Export:
  scripts/export_oracle_motion_features.py        — OracleMotion cache export (now with sharding)
  scripts/slurm/export_oracle_motion_features_1gpu.sh — export Slurm wrapper

Evaluation:
  scripts/evaluate_stage2_transition_paths.py     — path-level transition evaluator
  scripts/evaluate_stage2_trajectory_reliability.py — trajectory reliability vs interpolation
  scripts/generate_stage2_trajectories.py         — trajectory generation for visualization
  scripts/slurm/evaluate_stage2_transition_paths_1gpu.sh
  scripts/slurm/evaluate_stage2_trajectory_reliability_1gpu.sh
  scripts/slurm/generate_stage2_trajectories_1gpu.sh

Model/config:
  src/stage2/models/torsion_flow.py               — TorsionFlowNet, TorsionFlowNetConfig
  src/stage2/training/config.py                   — TrainingConfig
  src/stage2/training/trainer.py                  — CFM losses, Heun integration, validation
  src/stage2/datasets/dataset_stage2.py           — dataset, esm_num_layers
  src/stage2/datasets/esm_cache.py                — _esm_features_from_data, per_residue_layers
  src/stage1/models/adapter.py                    — ESMLayerFusionAdapter, gated_residual

Docs:
  docs/FULL_SCALE_TRAINING_AND_EVALUATION_RUNBOOK_20260626.md
  docs/BINDRAE_POSITIONING_AND_BENCHMARK_STRATEGY_20260629.md
  docs/STAGE2_GATED_REPRESENTATION_EXPERIMENTS_20260629.md
  docs/ORACLE_MOTION_BASELINE_SNAPSHOT_20260625.md
```

## Data Layout

```text
processed_data/triplets/
  samples/                         — 91327 sample dirs, each has esm.pt with per_residue_layers [N, 7, 1280]
  train_samples_full.txt           — 73022 train sample IDs
  test_valid.txt                   — 4297 val sample IDs
  ablation_subsets/                — subset files for screening

logs/stage2_oracle_motion/
  oracle_motion_train24000_esm7sync_20260629/     — 24k train cache (24k NPZ, manifest OK)
  oracle_motion_val2048_esm7sync_20260629/        — 2k val cache (2048 NPZ, manifest OK)
  oracle_motion_train_full_esm7sync_20260630/     — full train cache (4 shards, being exported)
  oracle_motion_val_full_esm7sync_20260630/       — full val cache (being exported)

checkpoints/stage2/
  stage2_stage2_24k_gm2_.../best_model.pt         — 24k gm2 best checkpoint (MAINLINE)
  stage2_stage2_24k_gm2_.../last_checkpoint.pt
  stage2_stage2_24k_single_.../best_model.pt
  stage2_stage2_24k_gm15_.../best_model.pt
```

## Quick Reference Commands

```bash
# Check queue
ssh gpu33pw 'export PATH=/data/soft/slurm/24.11.4/bin:$PATH; squeue -u zhaozc_criait -o "%.18i %.28j %.2t %.12M %.6D %R %b"'

# Check job status
ssh gpu33pw 'export PATH=/data/soft/slurm/24.11.4/bin:$PATH; sacct -j <JOBID> --format=JobID,JobName%28,State,Elapsed,ExitCode,NodeList%16 -P'

# Read metrics
ssh gpu33pw 'PY=/mnt/inaisfs/data/home/zhaozc_criait/miniconda3/envs/BINDRAE/bin/python; $PY - <<"PY"
import json, math
from pathlib import Path
root=Path("/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE_clean_stage2_repa")
# replace TAG with actual dir name
p=root/"logs/stage2/<TAG>/metrics.jsonl"
rows=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
def fm(r): return r.get("val_fm", r.get("val_fm_chi",0)+r.get("val_fm_rigid",0))
def v(r): return r.get("val_total_no_repa", math.nan)
for r in rows[-5:]:
    print("e%d val %.4f fm %.4f" % (r.get("epoch"), v(r), fm(r)))
best=min(rows, key=lambda r: v(r))
print("BEST e%d val %.4f fm %.4f" % (best.get("epoch"), v(best), fm(best)))
PY'

# Submit resumable 24k training
ssh gpu33pw 'ROOT=/mnt/inaisfs/data/home/zhaozc_criait/XinxiangWang/BINDRAE_clean_stage2_repa; cd "$ROOT"; scripts/slurm/submit_stage2_24k_resumable.sh <variant>'

# Syntax check
bash -n scripts/slurm/<script>.sh
python -m py_compile scripts/<script>.py
```
