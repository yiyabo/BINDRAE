# Stage-1 Screening Workflow

## Goal

Speed up idea validation for Stage-1 **without shrinking the main 8×A100 training scale** and without touching the test split.

This workflow is for:

- loss / mask / schedule screening
- delayed auxiliary supervision experiments
- shared-prefix branching from a common checkpoint

This workflow is **not** for final reporting. Final comparisons still use the normal full validation lane.

---

## Core principles

1. **Do not use the test split for model selection.**
2. **Keep the main training scale unchanged** (same model size, same 8 GPU regime).
3. **Shorten feedback loops by shortening budget**, not by weakening the final experiment design.
4. **Exploit common-prefix checkpoints** when candidate branches only diverge after some step.

---

## Recommended two-stage workflow

### Stage A — Screening lane

Use a short budget to answer:

- Is the candidate clearly worse?
- Is the candidate directionally promising?
- Is the candidate worth promotion to a longer run?

Recommended screening horizon:

- Immediate-effect experiments (e.g. plain E5a style): **5–6 epochs**
- Delayed/ramp experiments (e.g. E6 style): **8–12 epochs**

For delayed/ramp experiments, do **not** judge too early. The auxiliary signal may not have fully turned on.

### Stage B — Promotion lane

Only promote the best 1–2 candidates from Stage A.

Promotion criteria:

- Candidate shows a clearly better trend than the anchor run.
- Candidate is not merely a single-epoch spike.
- Candidate remains compatible with the current Stage-1 narrative.

---

## Shared-prefix branching

### When to use it

Use shared-prefix branching when multiple candidates are identical before a known point, for example:

- same base loss
- same schedule before `pchi1_start_step`
- same model / data / optimizer settings
- only different after auxiliary supervision starts to matter

### Why it helps

For E6-like runs, the early training phase is largely shared. Re-running that same prefix from step 0 for every candidate wastes wall-clock time.

### How to use it

1. Run a **prefix job** up to the shared divergence point.
2. Save checkpoint(s), ideally with `epoch_XXX.pt` and `latest_model.pt`.
3. Launch multiple branch jobs with:
   - `--resume_from <prefix checkpoint>`
   - new `save_dir`
   - new `log_dir`
4. Each branch keeps the model weights and step counter, but writes to its own experiment directory.

### Resume semantics in current code

Current Stage-1 trainer now supports `--resume_from`.

- **full-resume**: resume into the same run directory (restores optimizer/scheduler/best metric)
- **branch-resume**: resume from a checkpoint but write into a different `save_dir`; this keeps model weights, optimizer/scheduler state, and `global_step`, resets best metric / patience, and starts a fresh branch record

This makes shared-prefix branching operational.

---

## Validation policy

### Full validation remains the main judge

Use the normal full validation lane (`val_valid.txt`) for all promotion decisions.

### Optional dev-val lane

If validation becomes the main bottleneck, introduce a **fixed smaller dev-val file** derived from `val_valid.txt`.

Use dev-val only for:

- trend inspection
- fast pruning

Do **not** replace full validation with dev-val for final branch decisions.

---

## Kill criteria

### General rule

Pre-register kill criteria before launching a branch. Do not decide purely by feeling after the fact.

### For delayed/ramp pchi1 branches

Suggested practical rules:

1. **Before auxiliary turn-on**
   - Only use results for sanity checking, not final judgment.

2. **After auxiliary has clearly turned on**
   - If the branch remains materially below the current anchor for multiple validations, kill it.

3. **Around epoch 8–12**
   - If the branch still does not beat the best comparable screening baseline in a convincing way, kill it.

### Architecture escalation gate

Treat E6-type runs as the last serious objective/supervision experiments.

- If a candidate cannot push `pocket_chi1_acc` to roughly **0.373+** and sustain a meaningfully higher platform than E2, objective-only tuning is likely exhausted.
- At that point, shift attention to architecture.

---

## Naming convention

### Prefix runs

- `prefix_<family>_<timestamp>`

Example:

- `prefix_e6_20260415_120000`

### Branch runs

- `<family><variant>_from_<prefix_tag>_<timestamp>`

Examples:

- `e6a_from_prefix_e6_20260415_123000`
- `e6b_from_prefix_e6_20260415_124500`

This makes provenance obvious in logs and checkpoints.

---

## Suggested E6 screening pattern

### Prefix

Run a shared prefix that covers the common early regime.

For current E6-style experiments, a good operational anchor is:

- run until just before or around `pchi1_start_step`

### Current concrete prefix rule

For the current 8 GPU Stage-1 setup:

- total steps per epoch are roughly ~1800
- `pchi1_start_step = 8000` corresponds to a bit after epoch 4 begins

So the practical shared-prefix checkpoint should be:

- **`epoch_003.pt`** from a prefix run with `lambda_pchi1=0.0`

This keeps the branch resume point before auxiliary supervision starts to matter.

### Branches

From that prefix, branch candidates such as:

- `ligand_facing` vs `holo_pocket`
- shorter vs longer `pchi1_ramp_steps`
- same target lambda but different start timing

Keep all other knobs fixed.

### First concrete branch recommendation

The first actual branch after the prefix should be:

- **same mask**: `ligand_facing`
- **same target lambda**: `0.2`
- **same start step**: `8000`
- **gentler ramp**: `8000` steps instead of `4000`

Why this branch first:

- E5a showed directional improvement but weak magnitude.
- E5b showed that stronger immediate emphasis can hurt.
- E6a showed promising early signal, but the first post-activation validation dipped.

So the cleanest next branch is to test whether **the same hard mask and target lambda work better with a softer post-prefix curriculum**, instead of changing multiple variables at once.

Recommended branch name:

- `e6b_slowramp_from_<prefix_tag>_<timestamp>`

---

## Minimal operational checklist

Before launch:

- [ ] Candidate differs from anchor in exactly one intended way
- [ ] Save/log directories are unique
- [ ] `selection_metric=pocket_chi1_acc`
- [ ] Kill criteria written down in advance

After launch:

- [ ] Record first valid epoch
- [ ] Record best epoch and best `pocket_chi1_acc`
- [ ] Record whether gain is stable or spike-like
- [ ] Decide: promote / kill / escalate

---

## Current recommendation

Use this workflow immediately for future Stage-1 candidate testing.

For the current project phase:

- Keep the main lane at 8×A100
- Use screening budgets for faster feedback
- Use shared-prefix branching whenever candidate divergence happens late
- Do not use test split for model selection
