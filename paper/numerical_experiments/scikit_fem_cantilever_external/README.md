# Locked Numerical Experiment: scikit-fem Cantilever External FEM Comparison

## Status

This folder freezes the paper-bound external open-source FEM comparison for a
3D TET4 cantilever beam. SFC is compared against scikit-fem on matching meshes,
material parameters, Dirichlet boundaries, and point loads.

This case proves that SFC's linear TET4 stiffness assembly, displacement solve,
strain recovery, stress recovery, and von Mises stress agree with an
independent open-source FEM implementation for the locked cantilever setup.

## Reproduce

```bash
python validation/run_external_fem_comparison.py --out-dir results/external_fem
```

## Primary Data and Figures

- `data/external_fem_comparison.csv`
- `figures/external_fem_error_trends.png`
- `figures/external_fem_error_trends.pdf`
- `figures/external_fem_stress_3d.png`
- `figures/external_fem_stress_3d.pdf`

## Stress Cloud Scheme

The paper-facing stress cloud uses the current engineering visualization
scheme:

1. element von Mises stress is averaged to nodes;
2. boundary triangles interpolate from nodal stress values;
3. triangles are subdivided for a smoother Matplotlib surface rendering.

This is not the older element-constant/blocky stress-coloring scheme and does
not use centroid-only contact or stress visualization.

## Locked Metrics

The locked external comparison row maxima are:

- max displacement L2 relative error: `8.203660706987664e-13`
- max stress L2 relative error: `8.507136039642287e-13`
- max von Mises L2 relative error: `7.996485535331023e-13`
- max stiffness Frobenius relative error: `6.519881706486751e-16`

The acceptance thresholds are versioned in `locked_thresholds.csv` and enforced
by `tests/test_locked_paper_experiments.py`.

## Supported Claim

For the locked linear TET4 cantilever benchmark, SFC agrees with scikit-fem to
near machine precision for displacement, stiffness, strain, stress, and von
Mises stress metrics.

## Boundaries

This case does not validate contact dynamics, nonlinear materials, friction, or
self-contact. The external comparison is a linear static FEM comparison.

