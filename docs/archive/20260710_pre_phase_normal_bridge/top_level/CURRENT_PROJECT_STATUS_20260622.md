# Current Project Status

Date: 2026-06-22

## One-Line Goal

BINDRAE targets known-pose protein-ligand induced-fit trajectory generation:

```text
apo protein + aligned ligand pose
-> ligand-causal local posterior
-> Stage-2 bridge flow
-> apo-to-holo pocket/path ensemble
```

This is not generic docking and not only endpoint holo prediction.

## Current Scientific Direction

The active direction is **teacher-distilled Stage-1-v2 posterior guidance for
Stage-2 bridge flow**.

Stage-1-v2 should be a local ligand-causal posterior encoder, not a hard holo
rotamer endpoint model. It should learn or distill:

- residue-ligand contact probability;
- apo-to-teacher distance delta;
- approach/release/switch classes;
- local confidence;
- optional posterior latent features for Stage-2 representation alignment.

Stage-2 remains the main generative model. Its job is to produce the
apo-to-holo path and intermediate conformations.

## Positioning Guardrails

Use RAEv2/REPA as a limited analogy, not as the method name. The transferable
idea is to avoid forcing a weak in-repo model to discover all local structure
signals from scratch when stronger teachers or oracle labels can provide a
better posterior target. BINDRAE is not implementing an image-style RAE
architecture.

The publishable method should be framed as:

```text
Teacher-Distilled Posterior-Guided Bridge Flow
```

Optional `z_post` alignment is REPA-style latent guidance and should be added
only after scalar posterior features pass counterfactual and Stage-2 ablation
tests.

## Current Evidence

LC-PGBF v1 already showed a small but clean positive signal:

- correct learned local prior slightly improves active-residue endpoint/path
  metrics over no-prior and zero-prior controls;
- the result is not strong enough for a paper claim by itself;
- it justifies upgrading the Stage-1 signal rather than abandoning the
  two-stage design.

The latest Stage-2 record is:

```text
docs/LC_PGBF_STAGE2_EXPERIMENT_RECORD_20260620.md
```

The active Stage-1-v2 plan is:

```text
docs/STAGE1V2_TEACHER_POSTERIOR_PLAN_20260622.md
```

2026-06-25 update: the strongest current evidence is now the train12000/e10
OracleMotion Stage-2 upper-bound baseline, frozen in:

```text
docs/ORACLE_MOTION_BASELINE_SNAPSHOT_20260625.md
```

This baseline shows a large matched-over-zero/residue-shuffled gain when
Stage-2 receives residue-aligned oracle motion features. The result is strong
enough to stop default reliability/ranking/visualization expansion and begin
representation enhancement:

```text
docs/RAEV2_REPA_STAGE2_ENHANCEMENT_PLAN_20260625.md
```

## Latest Stage-1-v2 Student State

The current student is trained in the `holo_truth` oracle-distillation lane,
so it is an upper-bound/proof-of-mechanism checkpoint, not yet the final
external-teacher deployment lane.

- Best checkpoint:
  `checkpoints/stage1v2/stage1v2_posterior_train4096_lcpgbf_cb20e_3gpu_fullcf_bs16_cs384_h384_20260622_222239/best_model.pt`
- Training job: `135622`; early-stopped at epoch 14, best epoch 6 by
  `posterior_selection_score`.
- Full val512 audit:
  `logs/stage1v2_audits/stage1v2_best_val512_audit_20260622_231033.json`
- Audit metrics:
  - `posterior_selection_score=0.6050`
  - `contact_balanced_acc=0.8376`, `contact_AUPRC=0.7678`
  - `active_balanced_acc=0.6015`, `active_AUPRC=0.4498`
  - `approach_balanced_acc=0.5843`, `release_balanced_acc=0.5830`
  - `pocket_delta_mae=2.5068`, `pocket_dist_mae=3.1536`
- Ligand-control lift remains strong for contact and small but positive for
  active/motion heads: no-lig contact lift `+0.3376`, active lift `+0.0181`;
  shuffled active lift `+0.0096`; translated active lift `+0.0142`.
- Student posterior caches:
  - train4096:
    `logs/stage1v2_student_posteriors/stage1v2_best_train4096_cache_20260622_231423/`
  - val512:
    `logs/stage1v2_student_posteriors/stage1v2_best_val512_cache_20260622_231033/`

Interpretation: the student is already useful as a soft pocket/contact prior
and a weak but real motion posterior. Stage-2 should consume scalar posterior
features first, with `none/zero/student/shuffled/oracle` controls, before
enabling any `z_post`/REPA-style latent alignment.

## Stage-2 Posterior Interface Status

Stage-2 now accepts multi-channel Stage-1-v2 posterior scalar features through
the existing soft-prior input path. The initial feature set is:

```text
contact_prob, active_prob, approach_prob, release_prob, confidence,
teacher_min_dist_pred_norm, signed_delta_dist_pred_norm
```

The supported modes are:

- `none`: no Stage-1-v2 feature input;
- `zero`: same 7-dimensional input capacity, all zeros;
- `student`: cached Stage-1-v2 student posterior;
- `student_shuffled`: mismatched cached student posterior control;
- `oracle_holo_truth`: same schema read from `holo_truth` labels;
- `external_teacher_cached`: reserved for external teacher caches.

Smoke status:

- `135638` passed `student` 1-GPU Stage-2 smoke.
- `135641`, `135642`, `135643` passed `zero`, `student_shuffled`, and
  `oracle_holo_truth` 1-GPU Stage-2 smokes.
- Short 2-GPU train512/val128 comparisons completed:
  - head-only scalar input: `135645` none, `135646` zero, `135647` student,
    plus `135658` student_shuffled and `135659` oracle_holo_truth.
  - feature-scale stress test: `135661` student scale4, `135662`
    student_shuffled scale4, `135663` oracle_holo_truth scale4.
  - trunk-injected scalar input: `135666` zero, `135667` student, `135668`
    student_shuffled, `135669` oracle_holo_truth.
  - posterior objective tests:
    `135671` zero, `135672` student, `135673` student_shuffled, `135674`
    oracle_holo_truth with `contact_active` loss reweighting and weak
    posterior contact guidance;
    `135675` zero and `135676` oracle_holo_truth with stronger contact
    guidance only.

Interpretation: the Stage-2 posterior feature path is technically live, but
head-only scalar input and simple feature scaling do not create a reliable
student-over-shuffled or oracle-over-zero signal. Trunk injection is healthier
(`student_trunk` and `oracle_trunk` slightly improve validation total loss
over `zero_trunk`), but contact-direction metrics remain noisy and do not yet
support a full-scale claim. Posterior-objective tests show that the new
`stage1v2_guidance` loss can distinguish correct from shuffled posteriors
(`shuffled` guidance loss is much larger), but even oracle contact guidance
only moves contact metrics weakly and does not improve endpoint/path quality.
Do not scale this line blindly; the current scalar posterior interface is not
yet a convincing Stage-2 movement prior.

## OracleMotion-UB Track

The next upper-bound question is stricter than oracle contact:

```text
If Stage-2 receives privileged per-residue apo->holo motion features, can the
Stage-1 -> Stage-2 interface improve generated endpoints and paths?
```

Current terminology and execution state:

- The current OracleMotion lane is **Stage-2 inference/evaluation with trained
  Stage-2 checkpoints**, not additional model training.
- `oracle_motion` is a privileged apo/holo-derived feature cache that
  temporarily stands in for the future Stage-1 motion-posterior student.
- The current Stage-1-v2 scalar/contact student is not the model being trusted
  in this lane. It remains useful historical/proof-of-mechanism context, but
  it is not yet the desired motion-posterior student.
- Stage-2 already has trained checkpoints for `zero`, matched
  `oracle_motion`, and `oracle_motion_residue_shuffled`; reliability audits
  load these checkpoints, integrate apo->holo trajectories, and compare them
  against holo endpoints and cubic SE(3)+chi interpolation.

Added entrypoints:

- `scripts/export_oracle_motion_features.py`
- `scripts/slurm/export_oracle_motion_features_1gpu.sh`
- `scripts/slurm/train_stage2_oracle_motion_smoke_1gpu.sh`
- `scripts/slurm/train_stage2_oracle_motion_ablation_4gpu.sh`

The exported schema is `bindrae_oracle_motion_v1`. It stores local frame
translation, rotation log, SE(3) log, raw and masked chi deltas, sin/cos chi
delta features, motion/contact masks, distance deltas, and a normalized
`oracle_motion_features` matrix for later Stage-2 conditioning.

Direct oracle-apply audits:

- Job `135678`, output:
  `logs/stage2_oracle_motion/oracle_motion_val16_rawchi_direct_20260623_020005/`
- Direct oracle application reached the expected ceiling:
  - frame translation error: `3.7e-6 A`
  - frame rotation error: `2.4e-8 rad`
  - chi error: `5.5e-8 rad`
  - sidechain atom14 position error: `3.9e-6 A`
  - ligand min-distance error: `2.7e-5 A`
  - holo-contact F1: `1.0`
- Job `135679`, val512 output:
  `logs/stage2_oracle_motion/oracle_motion_val512_direct_20260623_020114/`
- Val512 direct oracle application also reached ceiling:
  - frame translation error: `4.2e-6 A`
  - frame rotation error: `2.4e-8 rad`
  - chi error: `4.8e-8 rad`
  - sidechain atom14 position error: `4.4e-6 A`
  - ligand min-distance error: `1.3e-5 A`
  - contact F1: `1.0` on all valid, pocket, motion-active, holo-contact,
    and formed-contact subsets.

Important implementation note: direct oracle apply must use raw apo->holo
`delta_chi`, not only `chi_mask`-masked deltas. Some triplet torsion arrays
contain non-zero chi values where `chi_mask=false`, and the FK module can still
use those angles for atom14 reconstruction. The exporter therefore stores both
`delta_chi` and `delta_chi_masked`; the direct upper-bound audit uses raw
`delta_chi`.

Decision rule: if full matched oracle motion cannot beat zero and shuffled
oracle-motion controls on generated endpoint/path metrics, then the bottleneck
is Stage-2 conditioning or bridge dynamics rather than Stage-1 contact
posterior quality. If it succeeds, Stage-1 should move from contact posterior
toward motion-posterior targets.

Stage-2 engineering status:

- `stage1v2_posterior_feature_mode` now supports `oracle_motion`,
  `oracle_motion_residue_shuffled`, and `oracle_motion_sample_shuffled`.
- Stage-2 cache loading hard-fails on stale or mismatched `sample_id`,
  `n_residues`, and matched-residue `aatype`; cache masks may be stricter than
  Stage-2 `node_mask`, but cannot mark Stage-2-invalid residues as valid.
- Sample-shuffled controls require same-length donors; no implicit
  pad/truncate alignment is allowed.
- Validation logs now include unweighted endpoint metrics and sidechain-contact
  metrics aligned with the 4.5 A / 0.75 posterior-guidance convention.

Initial train512/val512 upper-bound result:

- Exported train512 OracleMotion cache with job `135680`:
  `logs/stage2_oracle_motion/oracle_motion_train512_direct_20260623_omv1/`.
  Direct oracle apply reached ceiling:
  frame translation `3.7e-6 A`, frame rotation `2.4e-8 rad`,
  chi `4.8e-8 rad`, sidechain atom14 `4.0e-6 A`,
  ligand min-distance `1.9e-5 A`, contact F1 `1.0`.
- Smoke jobs passed for matched (`135681`), zero (`135682`), and
  residue-shuffled (`135683`) OracleMotion feature modes.
- 4-GPU train512/val512, 3-epoch ablation:
  - zero `135684`:
    `logs/stage2/stage2_omotion_zero_train512_train512_val512_e3_bs2x4_20260623_025241/`;
  - matched `135686`:
    `logs/stage2/stage2_omotion_matched_train512_train512_val512_e3_bs2x4_20260623_025257/`;
  - residue-shuffled `135685`:
    `logs/stage2/stage2_omotion_res_shuffle_train512_train512_val512_e3_bs2x4_20260623_025257/`.
- Final epoch result favors matched OracleMotion over zero and
  residue-shuffled controls:
  - `val_total`: matched `12.6603`, zero `13.1469`,
    residue-shuffled `13.1525`;
  - `val_end`: matched `183.1509`, zero `186.5174`,
    residue-shuffled `186.3262`;
  - `val_end_chi_uw`: matched `2.0643`, zero `2.3224`,
    residue-shuffled `2.3224`;
  - sidechain holo-gap: matched `0.0620`, zero `0.0642`,
    residue-shuffled `0.0641`;
  - formed-contact sidechain recall: matched `0.1390`, zero `0.0204`,
    residue-shuffled `0.0247`.

Interpretation: the Stage-2 conditioning route can exploit privileged
per-residue motion information. This rescues the interface direction at the
upper-bound level, but it does not validate scalar contact posterior guidance
as sufficient. The next Stage-1 target should be motion-posterior distillation,
not more tuning of the scalar contact-only path.

Trajectory reliability audit:

- Added oracle-motion-aware trajectory export and reliability evaluators:
  `scripts/generate_stage2_trajectories.py` and
  `scripts/evaluate_stage2_trajectory_reliability.py`.
- Initial val64 reliability benchmark against endpoint-aware cubic
  SE(3)+chi interpolation shows a useful split:
  - matched OracleMotion beats zero/residue-shuffled learned paths on active
    endpoint distance, path distance MAE, direction accuracy, chi error, and
    formed/released contact recovery;
  - all learned paths are much smoother and have far lower peptide-geometry
    loss than independent cubic per-residue interpolation;
  - under the old integration clamps, learned paths still move too little and
    remain far from holo endpoint/contact recovery.
- Implementation consequence: Stage-2 path integration velocity clamps are now
  explicit config/CLI knobs (`integration_chi_clip`,
  `integration_rot_clip`, `integration_trans_clip`). Defaults preserve old
  behavior (`1.0`, `0.1`, `0.2`), but relaxed-clamp audits should be run before
  treating the current path endpoint deficit as a model-capacity conclusion.
- Relaxed-clamp evaluation of the old checkpoints (`chi=5.0`, `rot=1.0`,
  `trans=5.0`) only weakly improves the matched model and leaves zero/shuffled
  unchanged. On the same val64 subset, matched formed-contact recall improves
  from `0.043` to `0.085`, while active endpoint distance changes only from
  `1.703 A` to `1.699 A`. This suggests the old checkpoint's vector field was
  trained into a conservative regime; the next upper-bound check should retrain
  Stage-2 with relaxed integration clamps rather than only relaxing evaluation.
- Relaxed-integration 3-GPU train512/val512 retraining completed:
  - zero `135709`:
    `logs/stage2/stage2_omotion_relaxed_zero_3gpu_train512_val512_e3_bs2x3_20260623_092755/`;
  - matched `135710`:
    `logs/stage2/stage2_omotion_relaxed_matched_3gpu_train512_val512_e3_bs2x3_20260623_092756/`;
  - residue-shuffled `135711`:
    `logs/stage2/stage2_omotion_relaxed_res_shuffle_3gpu_train512_val512_e3_bs2x3_20260623_093830/`.
- Final epoch strongly favors matched OracleMotion: `val_total=12.3496`
  versus zero `13.1535` and residue-shuffled `13.1576`; `val_end=177.2053`
  versus zero `186.5210` and residue-shuffled `185.7162`;
  `val_end_chi_uw=1.8971` versus zero `2.3176` and residue-shuffled `2.3172`;
  formed-contact sidechain recall `0.1719` versus zero `0.0334` and
  residue-shuffled `0.0337`. This is the clearest evidence so far that
  privileged motion guidance improves Stage-2 endpoint/path training.
- Direct `gpu33:GPU2` val64 relaxed-checkpoint inference/evaluation completed
  after Slurm stayed pending. This was not training. It loaded the trained
  relaxed checkpoints and evaluated generated trajectories with
  `n_integration_steps=3`:
  - outputs:
    `logs/stage2/trajectory_reliability/omotion_relaxed_{zero,matched,res_shuffle}_3gpu_val64_reliability_steps3_direct_gpu33_gpu2_20260623.json`;
  - matched improves active endpoint distance (`1.599 A` versus zero
    `1.822 A` and residue-shuffled `1.821 A`), active path MAE (`0.814 A`
    versus `0.911 A`/`0.911 A`), active direction accuracy (`0.653` versus
    `0.494`/`0.514`), active chi error (`1.240 rad` versus `1.445`/`1.445`),
    and formed-contact recall (`0.170` versus `0.0`/`0.0`);
  - cubic interpolation reaches the endpoint by construction but has much
    worse peptide loss (`13.07`) than learned paths, so endpoint-only numbers
    are not sufficient for trajectory reliability.
- Larger direct val512 reliability audits completed, again as
  inference/evaluation only:
  - `gpu33:GPU2`, `n_integration_steps=6`:
    `logs/manual/stage2_reliability_val512_steps6_direct_gpu33_gpu2_20260623_200344.out`;
  - `gpu33:GPU7`, `n_integration_steps=12`:
    `logs/manual/stage2_reliability_val512_steps12_direct_gpu33_gpu7_20260624.out`.
- The `steps=12` val512 audit confirms the same matched OracleMotion signal at
  larger scale: active endpoint distance `1.840 A` versus zero `2.045 A` and
  residue-shuffled `2.045 A`; active path MAE `0.970 A` versus
  `1.057 A`/`1.056 A`; direction accuracy `0.641` versus `0.511`/`0.510`;
  active chi error `1.324 rad` versus `1.536`/`1.536`; formed-contact recall
  `0.182` versus `0.0027`/`0.0027`. Residue-shuffled remains essentially tied
  to zero, supporting the claim that residue-aligned motion information, not
  feature capacity alone, drives the gain.
- Matched paths have higher learned-path peptide loss than zero/shuffled
  (`0.017` versus about `0.0014`) because they actually perform contact
  rearrangement, but all learned paths remain far below endpoint-aware cubic
  interpolation (`17.44`). The current upper-bound claim should therefore be:
  OracleMotion improves ligand-induced movement/contact recovery while keeping
  path geometry much more reliable than naive endpoint interpolation.

Open control: `oracle_motion_sample_shuffled` is implemented but not run in the
full train512/val512 ablation because strict same-length donors expose
singleton-length samples in the current caches. Run it on an eligible same-length
subset or after exporting a larger train cache.

## Active Engineering Track

Current default tasks after the 2026-06-25 OracleMotion freeze:

1. Keep the train12000/e10 OracleMotion matched/zero/residue-shuffled results
   as the frozen Stage-2 upper-bound baseline.
2. Do not launch additional reliability step sweeps, sample ranking, or
   visualization by default; use them later for paper evidence or debugging.
3. Add RAEv2-style ESM last-K fusion as a backward-compatible optional Stage-2
   representation upgrade.
4. Add REPA-style hidden-state alignment from Stage-2 geometry representations
   to OracleMotion/future motion-posterior targets.
5. Preserve zero, matched, residue-shuffled, and no-prior controls for every
   representation change.
6. Return to Stage-1 motion-posterior distillation after the enhanced
   OracleMotion Stage-2 path is tested.

## Current Entrypoints

Main training:

- `scripts/train_stage1.py`
- `scripts/train_stage2.py`
- `scripts/train_interaction_prior.py`

Stage-1-v2 label, audit, and cache export:

- `scripts/build_teacher_posterior_labels.py`
- `scripts/slurm/build_teacher_posterior_labels_1gpu.sh`
- `scripts/audit_stage1v2_posterior.py`
- `scripts/export_stage1v2_posterior_cache.py`
- `scripts/slurm/audit_stage1v2_posterior_1gpu.sh`
- `scripts/slurm/export_stage1v2_posterior_cache_1gpu.sh`

Stage-2 path evaluation/export:

- `scripts/export_oracle_motion_features.py`
- `scripts/slurm/export_oracle_motion_features_1gpu.sh`
- `scripts/evaluate_stage2_transition_paths.py`
- `scripts/evaluate_stage2_trajectory_reliability.py`
- `scripts/generate_stage2_trajectories.py`
- `viewer/stage2_trajectory_viewer/`

## Cleanup Policy

Top-level docs should only contain current source-of-truth documents and active
experiment records. Historical analyses, failed branches, and superseded
consultation notes belong under `docs/archive/`.

Top-level scripts can remain until the active pipeline stabilizes, but their
role must be indexed as one of:

- active entrypoint;
- active diagnostic/evaluator;
- transitional experiment;
- legacy/archive candidate.

Do not delete experiment code needed to reproduce an existing result until the
replacement path has passed smoke tests.
