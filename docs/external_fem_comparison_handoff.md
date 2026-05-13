# External FEM Comparison Handoff

## Purpose

This handoff records the external open-source FEM comparison requested for
review. The comparison uses scikit-fem as an independent validation-only
reference. It does not modify `src/sfc` and does not make scikit-fem a core
solver dependency.

## Command

```bash
python validation/run_external_fem_comparison.py --out-dir results/external_fem
```

Quick smoke mode:

```bash
python validation/run_external_fem_comparison.py --quick --out-dir results/external_fem
```

## External Solver

- solver: scikit-fem
- version used in the generated evidence: 12.0.1
- role: open-source FEM reference for validation only

## Equivalent Model

Both solvers use the same model:

- linear TET4 elasticity
- structured cantilever/block mesh
- material: `E = 1.0e5`, `nu = 0.3`
- fixed boundary: all displacement components fixed at `x = 0`
- load: total `-1.0` force distributed over tip nodes in the `z` direction
- output: stiffness, displacement, strain, stress, von Mises stress, and
  element-centroid stress cloud

## Output Files

- `results/external_fem/external_fem_comparison.csv`
- `results/external_fem/external_fem_stress_cloud.csv`
- `results/external_fem/external_fem_plots.csv`
- `results/external_fem/external_fem_summary.md`
- `results/external_fem/external_fem_error_trends.png`
- `results/external_fem/external_fem_error_trends.pdf`
- `results/external_fem/external_fem_stress_cloud.png`
- `results/external_fem/external_fem_stress_cloud.pdf`
- `results/external_fem/external_fem_stress_3d.png`
- `results/external_fem/external_fem_stress_3d.pdf`

## Summary Metrics

| Resolution | Elements | Tip uz SFC | Tip uz scikit-fem | Displacement rel. error | Stress rel. error | VM rel. error | K rel. error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | -1.397105e-03 | -1.397105e-03 | 1.733362e-14 | 1.730092e-14 | 1.643458e-14 | 3.626491e-16 |
| 2 | 40 | -3.686769e-03 | -3.686769e-03 | 5.060386e-14 | 5.730537e-14 | 4.772254e-14 | 5.197130e-16 |
| 3 | 135 | -6.617482e-03 | -6.617482e-03 | 1.095487e-13 | 1.276845e-13 | 1.185186e-13 | 2.713819e-16 |
| 4 | 320 | -9.561298e-03 | -9.561298e-03 | 8.203661e-13 | 8.507136e-13 | 7.996486e-13 | 6.519882e-16 |

## 3D Boundary-Surface Stress Plot

The generated 3D boundary-surface stress figure compares:

1. SFC element-centroid von Mises stress.
2. scikit-fem element-centroid von Mises stress.
3. absolute von Mises error.

Each panel averages element von Mises stress to nodes and interpolates the
nodal values over subdivided boundary triangles, so the full body surface mesh
and a smoothed stress field are visible in the figure. The runner also writes a
projected side-surface plot for auxiliary inspection. Quantitative error gates
remain based on element stress values in `external_fem_comparison.csv`.

The maximum von Mises absolute error in the generated CSV is
`3.047944e-10`, which is numerical roundoff scale for the tested model.

## Supported Claim

The current SFC linear TET4 cantilever implementation agrees with an independent
open-source FEM implementation, scikit-fem, for stiffness assembly,
displacement, strain, stress, and von Mises stress on the tested equivalent
models.

## Unsupported Claims

- This is not Abaqus validation.
- This is not nonlinear FEM validation.
- This is not frictional or self-contact validation.
- This is not a comparison against production contact or BVH implementations.
- This does not validate arbitrary industrial geometries or non-manifold contact
  surfaces.
