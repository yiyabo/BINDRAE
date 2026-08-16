# AHoJ Acquisition Filter Audit

Date: 2026-07-25

## Decision

The `1,231`-pair planning shortfall reported in
`PATH3_FINAL_EXPERIMENT_CONTRACT_20260725.md` is not a hard capacity limit of the
AHoJ index. It is dominated by two convenience filters that are not part of the
frozen acquisition contract. This audit measures that loss and adds opt-in
controls for both filters. It changes no default behaviour and no frozen gate.

## What Was Audited

The chain that produced the `1,769`-pair pool is:

```text
64,724 AHoJ records
  -> scripts/select_md_pilot_candidates.py   (mapping-aware screen)
  -> 2,470 unique endpoint pairs
  -> scripts/audit_md_candidate_capacity.py  (leakage + attempted-pair exclusion)
  -> 2,262 leakage-clean -> 1,769 novel
```

Four failure hypotheses were tested and rejected:

| Hypothesis | Verdict | Evidence |
|---|---|---|
| The scan only sampled a subset | Rejected | `scan_limit=0`, `screened=64724`; the full index was screened |
| `--select-count` truncated the pool | Rejected | `requested_select_count=3000`, `selected=2470`, `selection_shortfall=530`; the cap never bound |
| Dropped `HETATM` residues (`MSE`) inflate mapping rejections | Weak, not causal | `MSE` present in `5.0%` of `residue_mapping` rejections against a `1.7%` baseline (`n=60` each); at most a few hundred records |
| `pocket_too_small` is caused by unaligned ligands | Rejected | Nearest ligand-to-CA distance median `3.48 A` for rejected systems versus `3.50 A` for accepted ones |

The rejection ledger sums exactly to `64,724`, every exclusion carries a reason,
and the capacity audit cross-validates the leakage report against the clean list.
The pipeline is not silently dropping records. The residual
`pocket_too_small` population (`5,575` records) remains unexplained and is not
addressed here.

## Measured Capacity Loss

The mapping-aware screen accepted `9,859` records, which collapse to `2,470`
unique `(apo_pdb, holo_pdb)` pairs. The two largest non-frozen rejection classes
hold substantial unique-pair capacity that the current pool never sees:

| Rejection reason | Records | Unique pairs | Pairs absent from the accepted pool |
|---|---:|---:|---:|
| `ligand:multiple_unique_organic_fragments` | 8,613 | 1,790 | 1,347 |
| `excluded_ligand_resname` | 8,561 | 2,067 | 1,788 |
| Combined, deduplicated | | | **3,033** |

The blocklist rejections are concentrated in nucleotides and cofactors:
`GLC 1478`, `ADP 1025`, `NAD 1012`, `ATP 740`, `HEM 659`, `NAP 568`, `GDP 553`,
`AMP 378`.

## What Changed

Two opt-in controls, both defaulting to the existing frozen behaviour:

1. `describe_ligand(..., multi_fragment_policy=...)` in
   `src/data/md_pilot_selection.py`. The default `reject` is unchanged. The new
   `largest` policy keeps the heaviest organic fragment only when it is
   unambiguously primary: it must reach `fragment_dominance_ratio` times the
   heavy-atom count of the next fragment, or every other fragment must fall below
   `min_heavy_atoms`. Otherwise the sample is refused as
   `ambiguous_primary_fragment`. Discarded fragments are counted in the record
   and add a small `setup_penalty`, matching the existing treatment of
   non-organic fragments.
2. `--exclude-resname-preset` in `scripts/select_md_pilot_candidates.py`, with
   `frozen23` (default, the original 23-component blocklist), `glycosylation_only`
   (`NAG` only), and `none`. `select_md_pilot_candidates_cpu.sh` passes
   `EXCLUDE_RESNAME_PRESET`, `MULTI_FRAGMENT_POLICY`, and
   `FRAGMENT_DOMINANCE_RATIO` through, and the run summary records all three.

Ten unit tests pass in the remote `BINDRAE-MD` environment, including new
coverage for default rejection, successful disambiguation, refusal of two
comparable ligands, and invalid-policy handling.

## What Did Not Change

- the `0.95` exact common-residue mapping gate;
- the frozen holdout family/scaffold leakage exclusions;
- the historical attempted-pair exclusion;
- every physical gate, the two-replica consensus requirement, and the frozen
  train/validation/test splits;
- all existing defaults, so re-running the current launchers reproduces the
  `2,470`-pair scan byte-for-byte.

The ligand blocklist and the single-organic-fragment rule are convenience
chemistry filters. They are not among the frozen decisions, so relaxing them is
compatible with the contract; weakening mapping, leakage, or physical gates would
not be.

## Expected Recovery And Its Uncertainty

`3,033` is an upper bound, not a yield estimate. Those records were rejected
before reaching the downstream structural, pocket, and motion gates, so only a
fraction will survive a rescan. A rough projection using the observed stage-wise
pass rates gives roughly `1,000-2,200` recovered unique pairs, which would move
the pool from `1,769` to approximately `2,800-4,000` and plausibly close the
`3,000`-pair planning gate. This must be confirmed by an actual rescan, not
asserted from this bound.

Metal-bearing components such as `HEM` remain refused by the unchanged metal
gate regardless of preset, and `NAG` stays excluded under
`glycosylation_only` because glycosylation is not the induced-fit event of
interest.

## Rejected Alternative

Changing the deduplication key from `(apo_pdb, holo_pdb)` to
`(apo_pdb, holo_pdb, ligand)` would raise the accepted pool from `2,470` to
`6,604` unique units. This is **not** adopted. With both endpoints fixed, the
protein transition and the global CA-RMSD pull that generates the silver path are
the same regardless of which ligand is modelled, so those units are strongly
correlated observations rather than independent systems. Counting them would
inflate the system count in exactly the way the contract forbids for replicas.

## Next Steps

1. Re-run the mapping-aware scan under a new immutable tag with
   `MULTI_FRAGMENT_POLICY=largest` and `EXCLUDE_RESNAME_PRESET=glycosylation_only`,
   keeping every other threshold and the seed unchanged.
2. Re-run `audit_md_candidate_capacity.py` against the same frozen leakage report
   and attempted-pair manifests, and record the new shortfall.
3. Decide on scale-out only after the AHoJ `32 x 2` mapping-aware smoke reports a
   real yield, since the pair pool and the accepted-consensus-system yield are
   independent constraints.

## Claim Boundary

This audit is an acquisition-capacity engineering result. It does not create
training data, does not change any Path-3 model or split, and does not establish
that a 2,000-consensus-system campaign is feasible. The `3,033`-pair figure is a
measured upper bound on recoverable endpoint pairs, not accepted systems.
