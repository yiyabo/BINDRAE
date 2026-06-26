# LC-PGBF Current Plan

Date: 2026-06-18
Updated: 2026-06-20

## Core Claim

LC-PGBF means:

```text
known-pose apo protein + aligned ligand pose
-> ligand-causal local Stage-1 posterior/contact guidance
-> Stage-2 apo-to-holo bridge flow
-> residue-level induced-fit path ensemble
```

The project is not docking and not generic holo ensemble generation. The claim is
also not that Stage-1 predicts a reliable hard holo endpoint. The useful claim is
narrower and stronger:

```text
soft ligand-causal Stage-1 guidance can improve Stage-2 path generation on
ligand-active residues.
```

## Current Experimental Status

The first clean evidence chain is now positive. The main checkpoint is:

```text
Stage-1 prior:
checkpoints/stage1/interaction_prior_stronger8g_label_enriched_bs64_local256_tc4.5_20260619_080838/best_model.pt

Stage-2 main:
checkpoints/stage2/stage2_lc_pgbf_newp_s1_wc2_featonly_train12000_val1200_e5_bs2x4_20260619_174421/best_model.pt
```

On the fixed 1.2k validation lane, the residue-level transition evaluator shows
that correct Stage-1 prior inference beats zero-channel, shuffled-prior, and
oracle-contact controls on the main active-residue path metrics:

| inference mode | active endpoint | active path MAE | active direction |
|---|---:|---:|---:|
| correct prior | 2.14785 | 1.11284 | 0.54414 |
| zero channel | 2.14929 | 1.11354 | 0.54252 |
| shuffled prior | 2.14859 | 1.11320 | 0.54346 |
| oracle contact | 2.14834 | 1.11312 | 0.54279 |

Full run record: `docs/LC_PGBF_STAGE2_EXPERIMENT_RECORD_20260620.md`.

## Interpretation

This is a small but clean effect. It supports the intended mechanism because:

- the same Stage-2 checkpoint is used for all four inference modes;
- zero-channel does not reproduce the gain;
- shuffled learned prior is weaker than correct learned prior;
- oracle-contact is not automatically better, so the scalar contact channel is
  not a universal shortcut.

The best current Stage-2 training setup is:

```text
interaction_prior_feature_mode=prior
interaction_prior_feature_scale=1
w_interaction_prior=0
w_contact=2
```

Do not promote `w_interaction_prior=0.1` to the main lane. It improved a few
direction-style metrics but worsened endpoint/path quality.

## Method Shape

Stage-1 should stay a local posterior/contact module. Good outputs include:

- residue-ligand contact or proximity probability;
- distance-bin or expected-distance features;
- ligand-facing and switch-residue confidence;
- decoy/shuffled-ligand lift diagnostics.

Stage-2 should consume these as soft per-residue features with explicit fallback
controls:

- `prior`: learned correct-ligand posterior;
- `zero`: same feature channel, filled with zero;
- `prior_shuffled`: learned prior with broken ligand-residue correspondence;
- `oracle_contact`: diagnostic upper/control channel, not the main method.

The immediate next evidence target is stability, not architectural expansion:

```text
repeat the main setup over 2-3 seeds and/or a larger training subset;
report residue-level transition metrics, not only global endpoint loss.
```

## Paper Posture

Keep the novelty statement narrow:

```text
BINDRAE generates known-pose ligand-causal induced-fit transition paths using a
soft local Stage-1 posterior to guide a Stage-2 bridge flow.
```

Do not claim:

- Stage-1 alone solves holo rotamer prediction;
- the generated path is real-time MD;
- scalar oracle contact is a perfect upper bound;
- endpoint loss alone proves path quality.

The current story is viable if the positive active-residue transition effect
replicates across seeds or larger-budget runs.
