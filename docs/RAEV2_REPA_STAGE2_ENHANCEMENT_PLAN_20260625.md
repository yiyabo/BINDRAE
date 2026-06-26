# RAEv2/REPA-Style Stage-2 Enhancement Plan

Date: 2026-06-25

## Goal

Improve the current OracleMotion-conditioned Stage-2 baseline without changing
the scientific contract:

```text
apo protein + aligned ligand pose + optional oracle/future Stage-1 motion prior
-> Stage-2 bridge flow
-> apo-to-holo induced-fit trajectory
```

The current baseline is frozen in:

```text
docs/ORACLE_MOTION_BASELINE_SNAPSHOT_20260625.md
```

## Non-Goals

- Do not reframe BINDRAE as docking.
- Do not replace OracleMotion with the current scalar/contact Stage-1-v2
  student for this track.
- Do not keep running more reliability/ranking/visualization jobs as the
  default next step.
- Do not make MD validation a training dependency; MD belongs to downstream
  physical validation after the learned path is technically stable.

## Current Architecture Facts

Current `TorsionFlowNet` path:

```text
ESM last-layer features
-> ESMAdapter
-> optional prior_residue_proj(OracleMotion/Stage-1 features)
-> LigandConditioner
-> EdgeEmbedder
-> FlashIPA
-> gate, chi velocity, rigid velocity heads
```

Current limitations:

- `ESMAdapter` is a simple single-layer projection from `[B, N, 1280]` to
  `[B, N, c_s]`.
- There is no last-K ESM layer fusion.
- There is no REPA-style auxiliary representation alignment loss.
- OracleMotion features currently enter as residue-level conditioning
  channels, not as a teacher representation for hidden-state alignment.

## Enhancement 1: ESM Last-K Fusion

Rationale:

RAEv2-style designs trust strong pretrained representations and avoid forcing a
small task model to rediscover all semantic structure from one compressed
feature. For BINDRAE, the analogous move is to let Stage-2 learn a weighted
fusion of multiple ESM layers, then project the fused residue representation
into the geometry trunk.

Implementation sketch:

1. Add a backward-compatible adapter, for example:

```text
src/stage1/models/adapter.py
  ESMAdapter              # current single-layer path
  ESMLayerFusionAdapter   # new optional last-K path
```

2. Accept either:

```text
[B, N, 1280]          # existing cache, old behavior
[B, N, K, 1280]       # last-K cache, new behavior
```

3. Add config fields:

```text
esm_fusion_enabled: bool = False
esm_num_layers: int = 1
esm_fusion_mode: str = "softmax_weighted"
esm_layer_dropout: float = 0.0
```

4. Preserve old checkpoints by defaulting to `esm_fusion_enabled=False`.

Dataset/cache note:

If current processed triplets only store last-layer ESM, the first code pass
should be backward-compatible and inert. A later cache/export pass can add
last-K ESM arrays without breaking existing experiments.

Acceptance checks:

- old checkpoints still load with single-layer ESM;
- zero/oracle/residue-shuffled controls still run;
- parameter increase is modest relative to the Stage-2 trunk;
- no hidden dependency on a future Stage-1 student.

## Enhancement 2: REPA-Style Representation Alignment

Rationale:

The OracleMotion baseline shows that aligned local motion information is
valuable. Instead of only concatenating that information as an input channel,
we can use it as a teacher signal for the Stage-2 hidden representation.

This is not endpoint leakage at inference if the alignment target is used only
as a training loss and disabled at inference. It is an upper-bound training
experiment first; later the target can be replaced by a learned Stage-1 motion
posterior.

Candidate alignment:

```text
Stage-2 hidden state:       s_geo after FlashIPA
Teacher target:            projection(oracle_motion_features)
Loss:                      cosine / normalized MSE on valid active/pocket residues
Controls:                  matched, zero, residue-shuffled
```

Implementation sketch:

1. In `src/stage2/models/torsion_flow.py`, optionally return an intermediate
   representation:

```text
outputs["repa_hidden"] = s_geo
```

2. Add a small projection head:

```text
stage2_hidden -> repa_dim
oracle_motion_features -> repa_dim
```

The oracle projection should be detached or frozen for the first pass.

3. In `src/stage2/training/trainer.py`, add an auxiliary loss:

```text
loss_repa = active_or_pocket_masked_alignment(repa_hidden, oracle_motion_target)
total_loss += repa_weight * loss_repa
```

4. Add config fields:

```text
repa_enabled: bool = False
repa_dim: int = 128
repa_weight: float = 0.0
repa_target: str = "oracle_motion"
repa_mask: str = "motion_active_or_pocket"
```

5. Keep the feature-channel path unchanged so ablations can isolate:

```text
OracleMotion input only
OracleMotion input + REPA alignment
REPA alignment with shuffled target
zero input + no REPA
```

Acceptance checks:

- with `repa_weight=0`, metrics match the frozen baseline within run noise;
- matched REPA improves or at least does not harm endpoint/contact/path metrics;
- shuffled REPA does not reproduce the matched gain;
- inference does not require oracle labels unless the selected feature mode
  explicitly uses oracle conditioning.

## Minimal Ablation Matrix After Implementation

Run small first, then scale only if the signal is clean:

| Mode | ESM fusion | REPA | Purpose |
| --- | --- | --- | --- |
| zero | off | off | frozen-capacity control |
| oracle matched | off | off | frozen baseline |
| oracle matched | on | off | last-K effect |
| oracle matched | off | on | REPA effect |
| oracle matched | on | on | combined effect |
| residue-shuffled | on | on | anti-leakage/control |

Do not start with a larger dataset until the small controlled matrix separates
matched from shuffled for the right reason.

## Code Touch Map

Likely files:

- `src/stage1/models/adapter.py`
  - add or generalize ESM fusion adapter.
- `src/stage2/models/torsion_flow.py`
  - instantiate fusion adapter;
  - optionally expose `repa_hidden`;
  - add projection heads only behind config flags.
- `src/stage2/training/config.py`
  - add ESM fusion and REPA config.
- `src/stage2/training/trainer.py`
  - compute REPA target and auxiliary loss;
  - log `train_repa_loss` and `val_repa_loss`.
- `scripts/train_stage2.py`
  - expose CLI flags.
- `scripts/slurm/`
  - add a new 2-3 GPU enhancement wrapper rather than overloading old
    baseline wrappers.

Optional later files:

- ESM cache/export utilities, if last-K features are not already stored.
- Stage-1 motion-posterior student, after the oracle-enhanced Stage-2 path is
  proven useful.

## Scientific Story If It Works

The story should be:

```text
OracleMotion proves the Stage-2 interface can use local motion priors.
ESM last-K fusion improves the protein representation feeding the bridge flow.
REPA-style alignment makes the bridge hidden state motion-aware instead of only
conditioning the final heads.
Future Stage-1 motion posterior distills oracle/external teacher motion into
an inference-time prior.
```

This is stronger than simply saying "we added more features": it creates a
clean path from upper-bound oracle conditioning to a learned, deployable motion
posterior.
