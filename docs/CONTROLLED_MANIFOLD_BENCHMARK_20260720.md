# Controlled Product-Manifold Benchmark

Date: 2026-07-20

## Purpose

This benchmark tests the phase-normal decomposition independently of protein
encoders and silver MD noise. It asks two different questions:

1. Can an endpoint-exact phase plus normal model represent and learn
   asynchronous, obstacle-avoiding paths when the route is identifiable?
2. What should a deterministic residual do when the same endpoint-conditioned
   input admits exactly opposite routes?

It is a method identifiability experiment. It is not protein-path evidence and
does not turn generated paths into physical MD trajectories.

## Construction

Each sample is a multi-residue path on `SE(3) x T^2` with:

- known apo and holo endpoints;
- a strictly monotone residue phase
  `tau_i(t) = t + a_i t(1-t)`, with `|a_i| < 1`;
- an endpoint-zero normal detour with `sin^2(pi t)` envelope;
- translation, rotation, and chi residual directions orthogonal to the
  corresponding endpoint displacement;
- an obstacle centered on the synchronous translation bridge.

Every endpoint pair is duplicated with exactly opposite normal-route signs.
The endpoint-derived features and phase target are identical within each pair.

- `route_observed`: a one-bit route cue is included in the model input.
- `route_hidden`: the route cue is zero for both replicas, so the deterministic
  conditional-mean normal residual is exactly zero.

All four variants use the same MLP trunk and matched training budget:

1. synchronous bridge;
2. warp-only;
3. residual-only;
4. full phase plus normal residual.

## Run Contract

- Slurm job: `147041`, completed on `gpu38` using CPU only;
- model seeds: `7, 42, 137`;
- train: 256 endpoint pairs / 512 paths;
- validation: 64 endpoint pairs / 128 paths;
- test: 128 endpoint pairs / 256 paths;
- residues per system: 8;
- path intervals: 40;
- optimization: 300 epochs, validation-selected checkpoint per arm.

Result artifact:

```text
logs/stage2/controlled_manifold/controlled_manifold_3seed_cpu_20260720.json
```

## Results

All errors are lower-is-better. Event-order accuracy is higher-is-better.

| Route condition | Method | Product RMSE ↓ | Translation RMSE ↓ | Rotation RMSE ↓ | Chi RMSE ↓ | Event order ↑ | Residual RMS | Collision fraction ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| observed | synchronous | 0.636607 | 0.566187 | 0.202831 | 0.208711 | 0.5000 | 0.0000 | 1.0000 |
| observed | warp-only | 0.616695 | 0.545759 | 0.200175 | 0.205889 | 0.9962 | 0.0000 | 1.0000 |
| observed | residual-only | 0.158134 | 0.150840 | 0.032869 | 0.034254 | 0.5000 | 0.6207 | 0.0000 |
| observed | full | **0.007543** | **0.006881** | **0.002264** | **0.002090** | **0.9963** | 0.6208 | **0.0000** |
| hidden | synchronous | 0.636607 | 0.566187 | 0.202831 | 0.208711 | 0.5000 | 0.0000 | 1.0000 |
| hidden | warp-only | **0.616695** | **0.545759** | **0.200175** | **0.205889** | **0.9962** | 0.0000 | 1.0000 |
| hidden | residual-only | 0.636608 | 0.566188 | 0.202832 | 0.208711 | 0.5000 | 0.0011 | 1.0000 |
| hidden | full | 0.616697 | 0.545760 | 0.200176 | 0.205890 | 0.9961 | 0.0014 | 1.0000 |

The oracle mode has zero path error and zero collisions. Across all arms:

- endpoint maximum error: `2.38e-7`;
- maximum tangent-normal dot product: `4.58e-7`.

## Interpretation

When route information is observed, phase and normal residual are cleanly
complementary:

- phase recovers event order;
- the normal residual recovers the obstacle-avoiding spatial route;
- the full model recovers both.

When route information is hidden, the residual head predicts essentially zero
and the full model becomes warp-only. This is the mathematically correct
deterministic L2 solution, not an optimization failure.

Therefore Path-4 remains a sound architectural direction, but its protein
version should be reopened only with route-identifying inference information or
new evidence that endpoint-derived conditioning predicts a non-zero residual on
disjoint systems. Increasing residual capacity alone does not address the
identified failure mode.
