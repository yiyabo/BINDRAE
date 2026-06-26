# BINDRAE Stage-2 Path Viewer

Static browser viewer for exported Stage-2 apo-to-holo trajectory examples.

Default dataset:

```text
logs/stage2/generated_paths/lc_pgbf_seed44_correct_steps12_val4_20260622/manifest.json
```

Run from the repository root:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8765/viewer/stage2_trajectory_viewer/index.html
```

Use the `manifest` query parameter to inspect another exported trajectory run:

```text
http://127.0.0.1:8765/viewer/stage2_trajectory_viewer/index.html?manifest=/logs/stage2/generated_paths/<run>/manifest.json
```

The viewer expects each sample manifest entry to point to `trajectory_atom14.pdb`
and `trajectory.npz`, with sibling `apo_atom14.pdb`, `holo_atom14.pdb`,
`ligand_tokens.pdb`, and `summary.json`.
