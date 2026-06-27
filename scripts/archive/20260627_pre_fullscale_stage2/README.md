# Pre-Fullscale Stage-2 Script Archive

Date: 2026-06-27

This folder preserves exploratory or superseded Python entrypoints that were
useful during the Stage-1 and early Stage-2 investigation, but should not be
used as templates for new full-scale runs.

Current mainline work is under the top-level `scripts/` entrypoints:

- `train_stage2.py`
- `export_oracle_motion_features.py`
- `train_stage1v2_posterior.py`
- `build_teacher_posterior_labels.py`
- `export_stage1v2_posterior_cache.py`
- `audit_stage1v2_posterior.py`
- Stage-2 evaluators and trajectory exporters

Files here are retained for reproducibility only. If an archived idea is
revived, copy the needed behavior into a new, named mainline script rather than
running the archived entrypoint directly.
