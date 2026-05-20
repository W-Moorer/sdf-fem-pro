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
| SFC sphere mean `u_z` L2 relative error vs Abaqus | `4.9061e-3` |
| SFC beam-tip mean `u_z` L2 relative error vs Abaqus | `2.5440e-2` |
| Abaqus/Explicit analysis wall time | `18.555 s` |
| SFC explicit Lagrangian-SDF solve wall time | `7.417 s` |
| Solve-time speedup, Abaqus analysis / SFC | `2.50x` |

The SFC run uses lumped-mass central difference and the structured top-patch
Lagrangian contact backend for this regular cantilever surface. This is a
validation fast path for the same current-surface contact geometry, not an
Abaqus dependency.

## Stiff-Beam Optimization Check

The same script was also rerun on the original stiffer-beam input
(`E_beam=2.5e9`, `E_sphere=5.0e6`, `dt=1.0e-6 s`). Before the runner
optimization, this case took `36.47 s` in SFC versus `28.10 s` in
Abaqus/Explicit. After output thinning, vectorized contact scatter, and
persistent structured-contact workspace reuse, the SFC solve time is
`25.50 s`, giving a `1.10x` speedup over the Abaqus analysis wall time while
preserving the displacement-curve errors.

## Files

- `abaqus_input.inp`: generated Abaqus/Explicit reference input.
- `data/time_history.csv`: Abaqus and SFC displacement histories.
- `data/sfc_vs_abaqus_metrics.csv`: curve error metrics.
- `data/solver_timing.csv`: solver wall-time comparison.
- `data/stiff_beam_optimized_sfc_vs_abaqus_metrics.csv`: original stiff-beam
  optimization-check curve metrics.
- `data/stiff_beam_optimized_solver_timing.csv`: original stiff-beam
  optimization-check timing.
- `figures/sphere_mean_uz.png`: sphere displacement curve.
- `figures/beam_tip_mean_uz.png`: cantilever tip displacement curve.
