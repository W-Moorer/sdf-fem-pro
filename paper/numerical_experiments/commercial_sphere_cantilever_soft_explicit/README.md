# Commercial Sphere-Cantilever Soft Explicit Benchmark

This package records the short flexible-sphere/flexible-cantilever alignment
case used to check the SFC Lagrangian-SDF explicit contact path against an
independent Abaqus/Explicit reference.

## Model

| Quantity | Value |
|---|---:|
| Duration | `0.02 s` |
| Fixed explicit increment | `1.0e-5 s` |
| Output interval | `1.0e-3 s` |
| Beam element type | `C3D8R` |
| Sphere element type | `C3D4` |
| Beam Young's modulus | `2.5e8` |
| Sphere Young's modulus | `5.0e6` |
| Contact law | frictionless linear penalty |
| Penalty stiffness | `5.0e9` |
| Bulk viscosity | `0., 0.` |

The sphere and the cantilever are both flexible. The sphere is intentionally
softer than the cantilever. Abaqus files are external validation artifacts only;
the SFC core solver does not import Abaqus or read ODB files.

## Result Snapshot

| Metric | Value |
|---|---:|
| SFC sphere mean `u_z` L2 relative error vs Abaqus | `4.9157e-3` |
| SFC beam-tip mean `u_z` L2 relative error vs Abaqus | `2.5437e-2` |
| Abaqus/Explicit analysis wall time | `18.555 s` |
| SFC explicit Lagrangian-SDF solve wall time | `7.332 s` |
| Solve-time speedup, Abaqus analysis / SFC | `2.53x` |

The SFC run uses lumped-mass central difference and the structured top-patch
Lagrangian contact backend for this regular cantilever surface. This is a
validation fast path for the same current-surface contact geometry, not an
Abaqus dependency.

## Files

- `abaqus_input.inp`: generated Abaqus/Explicit reference input.
- `data/time_history.csv`: Abaqus and SFC displacement histories.
- `data/sfc_vs_abaqus_metrics.csv`: curve error metrics.
- `data/solver_timing.csv`: solver wall-time comparison.
- `figures/sphere_mean_uz.png`: sphere displacement curve.
- `figures/beam_tip_mean_uz.png`: cantilever tip displacement curve.
