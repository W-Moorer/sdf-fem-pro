# External Open-Source FEM Comparison

This validation compares the standalone SFC linear TET4 cantilever against scikit-fem on the same mesh, material, fixed boundary, and point load.

## Command

```bash
python validation/run_external_fem_comparison.py --out-dir results/external_fem
```

## External Solver

- solver: scikit-fem
- version: 12.0.1
- role: validation-only external open-source FEM reference

## Summary Metrics

| Resolution | Elements | Tip uz SFC | Tip uz scikit-fem | Displacement rel. error | Stress rel. error | VM rel. error | K rel. error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5 | -1.397105e-03 | -1.397105e-03 | 1.733362e-14 | 1.730092e-14 | 1.643458e-14 | 3.626491e-16 |
| 2 | 40 | -3.686769e-03 | -3.686769e-03 | 5.060386e-14 | 5.730537e-14 | 4.772254e-14 | 5.197130e-16 |
| 3 | 135 | -6.617482e-03 | -6.617482e-03 | 1.095487e-13 | 1.276845e-13 | 1.185186e-13 | 2.713819e-16 |
| 4 | 320 | -9.561298e-03 | -9.561298e-03 | 8.203661e-13 | 8.507136e-13 | 7.996486e-13 | 6.519882e-16 |

## Claim Gate

| Claim | Evidence | Gate value | Status |
| --- | --- | ---: | --- |
| SFC agrees with external open-source FEM for the linear TET4 cantilever | external_fem_comparison.csv::displacement_l2_rel_error | 8.203661e-13 | supported |
<!-- evidence csv=external_fem_comparison.csv field=displacement_l2_rel_error -->
| SFC stress recovery agrees with external-solver displacement field | external_fem_comparison.csv::stress_l2_rel_error | 8.507136e-13 | supported |
<!-- evidence csv=external_fem_comparison.csv field=stress_l2_rel_error -->

## Stress-Cloud Data

- rows: 500
- max von Mises absolute error: 3.047944e-10

## Plots

- `external_fem_error_trends.png` and `external_fem_error_trends.pdf`
- `external_fem_stress_cloud.png` and `external_fem_stress_cloud.pdf`
- `external_fem_stress_3d.png` and `external_fem_stress_3d.pdf`

## Interpretation

- This is an external open-source FEM comparison, not an internal refinement-only reference.
- The comparison uses identical linear TET4 mesh topology, material parameters, Dirichlet constraints, and nodal load.
- Stress and strain are compared as derived element quantities from each solver's displacement field on the same TET4 elements.
- Stress plots average element von Mises values to nodes and interpolate over subdivided boundary triangles for visualization.
- The result does not validate nonlinear FEM, friction, self-contact, or arbitrary industrial contact algorithms.
