# TPS-Flow Reproduction and Comparison Contract

## Scope

TPS-Flow is a direct endpoint-conditioned transition-path comparator, but its
released checkpoints are system-specific. It must not be presented as a
universal inference-only baseline on the BINDRAE 326-system corpus.

Primary sources:

- paper: https://doi.org/10.1021/acs.jcim.6c00807
- code: https://github.com/lfs119/TPS-Flow
- supplementary information: https://doi.org/10.1021/acs.jcim.6c00807.s001
- model weights: https://doi.org/10.5281/zenodo.17731628
- released data: https://doi.org/10.5281/zenodo.17555901

The pinned upstream commit for reproduction is:

`fa94ad66bdf93e4bf0dbc0214aa77ec87edb1901`

## Comparison Lanes

### Public-system reproduction

Reproduce an official checkpoint on its matching public system. Start with ADK
because it is the clearest predefined closed-to-open endpoint path. Preserve
the official model, checkpoint, data split, frame count, and post-relaxation
setting. Compatibility fixes must live outside the upstream checkout or be
recorded as a minimal patch.

This lane compares TPS-Flow and BINDRAE on identical endpoints and reference
paths. It is a case-study comparison, not evidence of cross-system deployment.

### Cross-system generalization

BINDRAE keeps its frozen family/scaffold-disjoint split: 241 train systems, 30
validation systems, and 32 untouched test systems. TPS-Flow enters this table
only if it can be retrained under the same system-level split without using a
test-system trajectory or checkpoint. A system-specific official checkpoint is
not eligible for this lane.

## Reproduction Gates

1. Run `scripts/audit_tps_flow_release.py` and record the upstream commit.
2. Verify official Zenodo file sizes and MD5 checksums before inference.
3. Reproduce one official system without changing the generated path endpoints.
4. Record every compatibility patch, dependency version, random seed, and wall
   time.
5. Confirm the generated trajectory has the documented atom order and topology.
6. Reproduce at least one primary paper metric within a predeclared tolerance.
7. Convert the untouched generated trajectory to the common BINDRAE evaluator;
   endpoint insertion is forbidden for the endpoint-error measurement.

## Shared Metrics

Lower is better:

- progress-aligned Cartesian/path RMSE;
- translation, rotation, and chi path error when atom mapping permits;
- contact-event timing MAE;
- endpoint coordinate error;
- clash count or clash energy;
- peptide geometry error;
- wall time and GPU time per generated path.

Higher is better:

- residue event-order Spearman/Kendall correlation;
- pairwise lead-lag accuracy;
- transient-contact precision and recall;
- contact formation/release ordering accuracy;
- valid-path rate.

Endpoint error must be reported rather than hidden by appending the known holo
structure. Physical relaxation must be reported both before and after when the
official method exposes both outputs.

## Known Release Gaps

The public repository is a research snapshot rather than a packaged release:

- no committed dependency lock file;
- no standalone license file despite the README license statement;
- ADK split names in the README are absent from the published tree;
- the ADK inference script imports PyRosetta, which is absent from installation
  instructions;
- inference is system-specific and contains hard-coded endpoint indices;
- released checkpoints correspond to individual systems rather than a single
  cross-system model.

These gaps do not disqualify TPS-Flow. They require a recorded compatibility
layer before its numbers can be treated as reproduced.

## Decision Rule

TPS-Flow belongs in the public-system comparison as soon as official ADK
inference and one paper metric are reproduced. It belongs in the frozen
cross-system table only after matched retraining. Failure to make the released
ADK path run after documented compatibility fixes is reported as a
reproducibility result, not silently replaced by a reimplementation.
