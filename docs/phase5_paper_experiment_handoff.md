# Phase-5 Paper Experiment Handoff

Status: Phase-5 paper experiment preparation is implemented and quick-mode
evidence generation passes. The runner prepares paper-facing CSV tables,
Markdown summaries, and PNG/PDF plots without adding new core physics. Friction,
self-contact, nonlinear FEM, GPU execution, and barrier contact remain out of
scope.

## Commands Run

```text
pytest -q
...................................................................      [100%]
67 passed in 5.06s
```

```text
python validation/run_phase4_paper_validation.py --quick --out-dir results/phase4
Wrote results\phase4\phase4_mesh_resolution.csv
Wrote results\phase4\phase4_contact_time_history.csv
Wrote results\phase4\phase4_performance_scaling.csv
Wrote results\phase4\phase4_claims.csv
Wrote results\phase4\phase4_summary.md
```

```text
python validation/run_phase5_paper_experiments.py --quick --out-dir results/phase5
Wrote results\phase5\phase5_metadata.csv
Wrote results\phase5\phase5_mesh_resolution.csv
Wrote results\phase5\phase5_contact_time_history.csv
Wrote results\phase5\phase5_performance_scaling.csv
Wrote results\phase5\phase5_material_space_sdf_baseline.csv
Wrote results\phase5\phase5_external_reference_validation.csv
Wrote results\phase5\phase5_contact_reference.csv
Wrote results\phase5\phase5_claims.csv
Wrote results\phase5\phase5_plots.csv
Wrote results\phase5\phase5_summary.md
```

The Phase-5 runner also wrote all requested PNG/PDF plots listed below.

## Output Files

Phase-5 tables and summaries:

- `results/phase5/phase5_metadata.csv`
- `results/phase5/phase5_mesh_resolution.csv`
- `results/phase5/phase5_contact_time_history.csv`
- `results/phase5/phase5_performance_scaling.csv`
- `results/phase5/phase5_material_space_sdf_baseline.csv`
- `results/phase5/phase5_external_reference_validation.csv`
- `results/phase5/phase5_contact_reference.csv`
- `results/phase5/phase5_claims.csv`
- `results/phase5/phase5_plots.csv`
- `results/phase5/phase5_summary.md`

Paper-prep plots:

- `results/phase5/phase5_mesh_resolution_trend.png`
- `results/phase5/phase5_mesh_resolution_trend.pdf`
- `results/phase5/phase5_contact_time_history.png`
- `results/phase5/phase5_contact_time_history.pdf`
- `results/phase5/phase5_performance_scaling.png`
- `results/phase5/phase5_performance_scaling.pdf`
- `results/phase5/phase5_speedup.png`
- `results/phase5/phase5_speedup.pdf`
- `results/phase5/phase5_material_space_sdf_error.png`
- `results/phase5/phase5_material_space_sdf_error.pdf`

## Performance Scaling

Quick mode uses four reduced surface sizes and two repeated timing runs per
timed row. The non-quick mode is implemented in the same runner with four
larger surface sizes `[4, 8, 12, 16]` and five repeats.

| Surface Resolution | Triangles | Query Points | Dynamic FEM-SDF Mean | Dynamic FEM-SDF Std | Speedup vs Brute Force | Claim Status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 8 | 4 | `2.663050e-03` | `2.863780e-05` | `1.038246` | quick evidence only |
| 3 | 18 | 9 | `5.811200e-03` | `3.030658e-04` | `2.150485` | quick evidence only |
| 4 | 32 | 16 | `9.915800e-03` | `1.548564e-04` | `3.697266` | quick evidence only |
| 5 | 50 | 25 | `1.543115e-02` | `1.171676e-04` | `5.689168` | quick evidence only |

The final acceleration claim is intentionally `not_supported` in quick mode.
The gate requires repeated non-quick timing, at least four surface sizes, enough
repeats, and dynamic FEM-SDF speedup greater than one.

## Material-Space SDF Baseline

The material-space SDF baseline is a frozen material-space plane SDF used only
for comparison. It is not the main method.

| Deformation | Baseline Rows | Max Gap Error | Max Normal Angle Error | Status |
| --- | ---: | ---: | ---: | --- |
| stretch | 20 | `5.217391e-03` | `0.000000e+00` | supported baseline comparison |
| shear | 20 | `2.708013e-03` | `3.580203e-01` | supported baseline comparison |

This supports the claim that baseline error data exists under stretch and
shear. It does not claim the frozen material-space baseline is the solver's
main SDF method.

## Reference Validation

Analytic linear-elastic reference:

| Resolution | Numerical Energy | Analytic Energy | Relative Error |
| ---: | ---: | ---: | ---: |
| 1 | `1.200000e-01` | `1.200000e-01` | `0.000000e+00` |
| 2 | `1.200000e-01` | `1.200000e-01` | `0.000000e+00` |
| 3 | `1.200000e-01` | `1.200000e-01` | `2.312965e-16` |

High-resolution brute-force closest-point projection contact reference:

- Quick-mode surface resolution: `5`
- Triangle count: `50`
- Reference rows: `25`
- Maximum reported gap error: `0.0`
- Maximum reported normal error: `0.0`

Abaqus is not used or required for these references.

## Claim Gates

| Claim | Status | Evidence |
| --- | --- | --- |
| Final acceleration claim | `not_supported` | `phase5_performance_scaling.csv`; gate value `quick,1.038245582836359,4,2` |
| Convergence-order claim | `not_supported` | `phase5_mesh_resolution.csv`; only two nonzero regression error points in quick mode |
| Material-space SDF comparison | `supported` | `phase5_material_space_sdf_baseline.csv`, 40 baseline rows |
| External/reference validation | `supported` | `phase5_external_reference_validation.csv` and `phase5_contact_reference.csv` |

## Supported Claims

- Phase-5 quick-mode evidence generation works end to end.
- Repeated timing fields `mean/std/min/max` are generated.
- Four surface sizes are exercised in quick mode; non-quick mode supports four
  larger sizes.
- Material-space SDF baseline comparison data exists for stretch and shear.
- Analytic linear-elastic and brute-force closest-point projection references
  are generated without Abaqus.
- PNG and PDF paper-prep figures are generated.

## Unsupported Claims

- Final paper acceleration claim is not supported by quick-mode evidence.
- Convergence order is not supported by the current quick-mode regression gate.
- Material-space SDF is not the main method.
- No claims are made for friction, self-contact, nonlinear FEM, GPU execution,
  barrier contact, or production BVH performance.

## Remaining Limitations

- Non-quick paper timing should be run on the target experiment machine before
  using any timing values in a paper.
- Quick-mode timings are useful for infrastructure and gate validation only.
- The SDF remains an oriented local current-surface distance, not a robust
  global sign method for arbitrary non-manifold geometry.
- The analytic reference currently covers small-strain linear elasticity only.
- The contact reference is deterministic and geometric; it is not an external
  experimental validation dataset.
