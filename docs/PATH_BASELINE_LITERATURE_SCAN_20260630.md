# Path Baseline Literature Scan

Date: 2026-06-30

## Project-Facing Contract

The fair main benchmark for the current BINDRAE line is:

```text
known apo endpoint + known holo endpoint + ligand pose aligned in the apo frame
-> ligand-aware apo-to-holo structural transition path
```

This is not docking, not blind holo endpoint prediction, and not a claim about
physical kinetics. The primary question is whether BINDRAE reconstructs a
ligand-aware induced-fit structural path with better pocket contact switching,
side-chain transition behavior, and intermediate validity than endpoint-driven
path baselines.

The existing Stage-2 reliability evaluator is already a good anchor because it
reports:

- active endpoint distance error;
- active path MAE and direction accuracy;
- frame and chi endpoint errors;
- formed-contact recall, released-contact success, and stable-contact retention;
- path clash penalty, peptide loss, and smoothness;
- cubic `SE(3)` plus chi interpolation references.

External baselines should therefore be converted into the same path representation
or into atomistic/CG frames that can be reconstructed and measured by the same
ligand-pocket metrics.

## Recommended Comparison Set

### Tier 0: Required Internal And Cheap Controls

Run these first and keep them in every table.

| Baseline | Role | Scale |
| --- | --- | --- |
| Apo endpoint / holo oracle | boundary diagnostics | all evaluation samples |
| Linear Cartesian interpolation | naive endpoint-aware floor | all evaluation samples |
| `SE(3)` frame + chi interpolation | state-space interpolation floor | all evaluation samples |
| Cubic/geodesic `SE(3)` + chi interpolation | smoother endpoint-aware floor | all evaluation samples |
| No-ligand BINDRAE | ligand causal control | all matched checkpoints |
| Shuffled-ligand or shuffled-prior BINDRAE | anti-leakage/causal control | all matched checkpoints |
| Zero-prior / no-Stage1 BINDRAE | architecture capacity control | all matched checkpoints |

### Tier 1: Main External Path Baselines

These are the most defensible direct comparisons for the current paper.

| Method | Why it belongs | Suggested use |
| --- | --- | --- |
| ANMPathway | Classic two-endpoint ENM pathway method; directly tests whether BINDRAE is only a fancy morph/interpolation. | Full validation set or large subset if the server/code path permits automation. |
| eBDIMS | Established endpoint-conditioned coarse-grained Brownian/elastic pathway method. | Full validation set or large subset; use atomistic reconstruction/repacking if needed for side-chain metrics. |
| eBDIMS2 | Modern optimized eBDIMS successor for large-scale transition pathways and intermediates, with code available. | Representative subset first, then scale if automation is clean. |
| DeepPath | Closest deep-learning path generator: physics-guided/GAL, atomistic transition pathways between known states. | Must attempt reproducibility. If code/weights are usable, compare on a curated subset; otherwise report as paper-level/case-study context with access limitations. |
| TPS-Flow | 2026 endpoint-conditioned SE(3) flow trained on MD paths, with public code, system-specific checkpoints, and data. | Reproduce its public ADK/system cases first. It enters the cross-system table only after matched retraining. |

### Tier 2: Strong Candidate Path Baselines

These should be included in the related-work matrix and used if engineering cost
is reasonable.

| Method | Why it matters | Suggested use |
| --- | --- | --- |
| MinActionPath2 | 2024 web-server method for transition paths between two conformations using collective motion data. | Candidate server/scripted subset; direct PDF download was blocked by OUP/PMC challenge, but article is open access. |
| SIDE / Langevin bridge | Recent endpoint-conditioned stochastic bridge approach; compares against MinActionPath and EBDIMS. | Related work plus optional subset if code appears or implementation is practical. |
| ICONGENI | Internal-coordinate NMA-guided elastic network interpolation. | Optional classical baseline, especially for NMA/morphing reviewer concerns. |
| COMBAS / bidirectional relaxation-biased simulations | Physics-based two-state pathway construction that improves over generic targeted MD initial paths. | Small curated physics subset only. |
| Targeted MD / steered MD-lite | Natural endpoint-driven physical baseline. | Small curated high-quality panel, not full scale. |

### Tier 3: Not Main Path Baselines

These are important context but should not be the main comparison table unless
the task is explicitly changed.

| Family | Examples | Proper role |
| --- | --- | --- |
| Flexible endpoint/docking models | DynamicBind, FlowDock, NeuralPLexer | Endpoint or pipeline context, not ordered path baselines. |
| Static cofolding/foundation models | AlphaFold 3, Boltz, Chai-1 | Holo-like endpoint teachers/references, not direct transition-path competitors. |
| Ligand-agnostic ensemble generators | AlphaFlow, P2DFlow, EigenFold/Str2Str-like models | Best-of-K holo/pocket coverage and ensemble validity, with ligand input difference stated explicitly. |

## Battle Cards

### TPS-Flow

TPS-Flow is the closest released generative transition-path implementation in
the current scan. Its general SE(3), flow-matching, and endpoint-conditioning
ingredients are not BINDRAE novelty claims. The direct comparison instead tests
whether BINDRAE's endpoint-exact analytic bridge and learned phase deliver
better cross-system path reconstruction and event ordering.

The released checkpoints are system-specific (`base`, `recon_energy`, `1hpv`,
`1brs`, and `adk`). Therefore they support a public-system case-study table but
not inference-only evaluation on arbitrary AHoJ-DB pairs. Use
`TPS_FLOW_REPRODUCTION_PLAN_20260719.md` as the reproduction contract.

### DeepPath

DeepPath is the strongest AI-path competitor because it predicts atomistic
transition pathways between known protein states with a physics-guided active
learning loop. BINDRAE should not try to beat it only on global RMSD. The
fair pressure points are ligand-pocket contact trajectory, formed/released
contact switching, pocket chi trajectory, protein-ligand clashes, and runtime
under a known ligand pose.

Risk: the paper points to Hugging Face resources, but the checked Hugging Face
URL timed out locally on 2026-06-30. Treat it as a mandatory reproducibility
audit item before promising full-scale comparison.

### eBDIMS2 / eBDIMS

These are highly relevant because they generate endpoint-conditioned transition
pathways and intermediates efficiently. eBDIMS2 is especially credible because
it targets large systems and provides a modern codebase. BINDRAE's angle is
local ligand-aware induced-fit response: side-chain chi transitions, ligand
contacts, and pocket sterics rather than only global/domain motion.

Important implementation note: eBDIMS-family outputs are coarse-grained or
backbone-centric. For BINDRAE metrics, build a consistent reconstruction path:
map residue correspondence, reconstruct side chains with a deterministic
protocol, then evaluate ligand-contact and clash metrics against the fixed
aligned ligand.

### ANMPathway

ANMPathway is the classic two-endpoint elastic-network pathway baseline and
therefore belongs in the main table. It is the cleanest answer to the reviewer
question "is this just ENM-style morphing?" The win condition for BINDRAE is not
endpoint closure alone; it is better pocket-local contact switching, chi motion,
and fewer impossible intermediates around the ligand.

### MinActionPath2, SIDE, ICONGENI

These are useful to broaden the path-method battle beyond ANM/eBDIMS. They can
be used as optional path baselines or strong related work depending on code and
server automation. SIDE is very recent and directly positions itself against
MinActionPath and EBDIMS. ICONGENI is a good NMA/internal-coordinate morphing
baseline if reviewer pressure around NMA is high.

### TMD / SMD / COMBAS

Do not run these full scale. They are useful as physics-flavored validation on a
small curated subset, especially for cases with side-chain flips, contact
switches, and large induced-fit pocket changes. Report runtime honestly.

## Suggested Benchmark Progression

1. Freeze the BINDRAE path evaluator and external trajectory format.
2. Run Tier 0 controls on the exact validation split used for Stage-2 reliability.
3. Add ANMPathway and eBDIMS on a large automated subset.
4. Add eBDIMS2 on a representative subset, then expand if setup is stable.
5. Reproduce TPS-Flow on ADK or another released public system, then decide
   whether matched cross-system retraining is feasible.
6. Audit DeepPath code/weights and run a curated subset if reproducible.
7. Add MinActionPath2 / ICONGENI / SIDE only if automation is clean enough.
8. Run TMD/SMD-lite or COMBAS on a 20-50 case curated physics panel.

## Downloaded PDFs

Open or publicly available PDFs saved under:

```text
reference/papers/baseline_path_methods/
```

Files:

- `deeppath_chemical_science_2026.pdf`
- `ebdims2_natcomm_2026.pdf`
- `ebdims_natcomm_2016.pdf`
- `anm_pathway_ploscb_2014.pdf`
- `icongeni_plosone_2021.pdf`
- `side_langevin_bridge_arxiv_2025.pdf`
- `combas_jctc_2022.pdf`

Not downloaded:

- MinActionPath2 NAR 2024. OUP direct PDF returns a Cloudflare challenge and
  PMC PDF access returns a JavaScript/PoW preparation page from CLI. Keep the
  article link in the bibliography and download manually through a browser if
  needed.

## Source Links To Recheck

- DeepPath: https://pubs.rsc.org/en/content/articlehtml/2026/sc/d5sc08253f
- TPS-Flow paper: https://doi.org/10.1021/acs.jcim.6c00807
- TPS-Flow code: https://github.com/lfs119/TPS-Flow
- eBDIMS2 paper: https://www.nature.com/articles/s41467-026-69809-y
- eBDIMS2 code: https://github.com/domenicoscaramozzino/eBDIMS2
- eBDIMS paper: https://www.nature.com/articles/ncomms12575
- eBDIMS server/docs: https://ebdims.biophysics.se/docs/
- ANMPathway: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003521
- MinActionPath2: https://academic.oup.com/nar/article/52/W1/W256/7680623
- SIDE arXiv: https://arxiv.org/abs/2512.01903
- ICONGENI: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0258818
- COMBAS/JCTC: https://pubs.acs.org/doi/10.1021/acs.jctc.2c00390
