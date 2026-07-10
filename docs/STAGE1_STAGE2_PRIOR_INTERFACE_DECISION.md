# Stage-1 / Stage-2 Prior Interface Decision

Date: 2026-04-26

> Status update (2026-07-10): retained as evidence for the future no-Oracle
> deployment track. Stage-1 is not the current paper bottleneck; the active
> method is the endpoint-conditioned phase-normal Stage-2 path.

## Executive Summary

Recent diagnostics indicate that the current BINDRAE prior interface should not continue as a deterministic Stage-1 holo decoder consumed as a Stage-2 attractor. Stage-1 E2 `epoch_018` is worse than apo χ1 carryover on pocket and ligand-facing subsets, while Stage-2 can use an oracle-quality χ signal weakly but does not benefit from the current E2 deterministic prior. The next architecture direction is therefore to reframe Stage-1 as a ligand-conditioned pocket rotamer/contact posterior with confidence, and to let Stage-2 consume soft features or energies rather than a single FK endpoint.

## Evidence

### Stage-1 Diagnostic Results

Checkpoint evaluated:

```text
checkpoints/stage1/opt8e2_20260411_122707/epoch_018.pt
```

Validation subset:

```text
256 samples / 64 batches
```

Pocket χ1 results:

| Condition | χ1 accuracy |
|---|---:|
| Apo carryover | 0.4781 |
| Stage-1 with correct ligand | 0.3899 |
| Stage-1 with no ligand | 0.3847 |
| Stage-1 with batch-shuffled ligand | 0.3873 |

Ligand-facing apo-CA subset:

| Condition | χ1 accuracy |
|---|---:|
| Apo carryover | 0.4611 |
| Stage-1 with correct ligand | 0.4017 |
| Stage-1 with no ligand | 0.3911 |
| Stage-1 with batch-shuffled ligand | 0.3932 |

Ligand-facing rescue/harmful-flip metrics:

| Metric | Value |
|---|---:|
| Rescue rate: apo wrong, Stage-1 correct | 0.2717 |
| Harmful flip rate: apo correct, Stage-1 wrong | 0.4463 |

Interpretation: Stage-1 has weak ligand causality and currently changes too many apo-correct sidechains into holo-wrong states. As a deterministic χ prior, it is net destructive relative to simply keeping apo χ1.

### Stage-2 Prior Isolation Results

The Stage-2 prior interface was decomposed into rigid and χ components.

Key findings:

1. Rigid/frame prior is strongly harmful.
2. χ-only input is safe but only useful when the χ signal is oracle-quality.
3. Current E2 χ input does not reliably beat no-prior under deterministic validation.

Representative deterministic validation runs (`val_t=0.5`, rigid prior off):

| Mode | Best validation loss |
|---|---:|
| No prior | 27.8001 |
| E2 χ input, pocket-local, scale 0.5 | 27.8616 |
| E2 χ input, pocket-local, scale 2.0 | 27.8488 |
| E2 χ input, pocket-local, scale 4.0 | 27.8334 |
| E2 χ input, pocket-local, scale 8.0 | 27.8349 |
| Oracle holo χ input, pocket-local, scale 8.0 | 27.7189 |

Interpretation: Stage-2 can consume high-quality χ guidance, but the current deterministic Stage-1 prediction is not high-quality enough to help. The prior loss and rigid prior should not be used as the main interface in their current form.

## Decision

Stop treating Stage-1 as a single deterministic holo-structure prior for Stage-2.

The next Stage-1 target should be:

```text
ligand-conditioned pocket rotamer/contact posterior
```

instead of:

```text
deterministic holo χ / FK decoder
```

Stage-2 should consume calibrated local uncertainty and compatibility information, not a single predicted endpoint.

## Recommended Next Architecture Direction

Stage-1 should emit local posterior-like information for pocket and ligand-facing residues:

```text
rotamer logits / probabilities
χ residual distribution around rotamer modes
residue-ligand contact or distance compatibility
confidence / expected utility q_i
```

Stage-2 should consume this as:

```text
soft input features
confidence-weighted local energy
selective guidance only where Stage-1 is confident
```

The current safe Stage-2 interface is χ-feature input only:

```text
use_stage1_rigid_prior = False
w_prior = 0
optional pocket-local stage1_chi input mask
stage1_chi_feature_scale tunable
```

This interface is useful for diagnostics, but it is not sufficient until Stage-1 provides better posterior/contact information.

## Practical Consequences

Do not prioritize further large-scale runs of:

- late pocket-only χ1 heads;
- deterministic Stage-1 routing adapters as the main solution;
- additional loss/ramp sweeps on the same deterministic χ decoder;
- rigid/frame prior losses in Stage-2;
- Stage-2 attraction to a single Stage-1 FK endpoint.

Prioritize:

1. Pocket rotamer classification or mixture-distribution heads.
2. Ligand-contact/contact-distance compatibility supervision.
3. Confidence calibration and selective-risk evaluation.
4. Stage-2 consumption of posterior/contact features or energies.
5. Continued oracle χ diagnostics as an upper-bound sanity check.

## Current Working Interpretation

BINDRAE's two-stage strategy remains valid, but the interface must change. The model should not attempt to challenge MD by asserting a single deterministic holo prior. A more defensible framing is a ligand-conditioned structural path generator guided by probabilistic pocket compatibility, with MD or experimental holo structures used for validation rather than direct replacement of atomistic dynamics.
