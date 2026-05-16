# Locked Numerical Experiment: TET4 Analytic Stress-Strain Patch

## Status

This folder freezes the paper-bound analytic patch evidence for TET4
stress-strain correctness. The case applies an affine uniaxial small-strain
deformation to TET4 meshes and compares recovered element strain and stress
against the analytic isotropic linear-elastic reference.

This case proves the element-level strain-displacement matrix, constitutive
law, and stress recovery for affine linear TET4 fields. It is not a contact or
dynamics validation case.

## Reproduce

```bash
python validation/run_phase7_physical_validation.py --out-dir results/phase7
```

## Primary Data and Figure

- `data/stress_strain_patch.csv`
- `figures/stress_strain_error.png`
- `figures/stress_strain_error.pdf`

## Locked Metrics

The analytic reference in `data/stress_strain_patch.csv` is:

- strain: `[0.001, 0, 0, 0, 0, 0]`
- stress: `[240, 80, 80, 0, 0, 0]`

Across four mesh resolutions, the locked maximum strain and stress L2 errors
are zero in the stored CSV.

The acceptance thresholds are versioned in `locked_thresholds.csv` and enforced
by `tests/test_locked_paper_experiments.py`.

## Supported Claim

For affine small-strain linear-elastic TET4 fields, SFC recovers the analytic
strain and stress to the locked numerical tolerance.

## Boundaries

This case does not validate contact, time integration, geometric nonlinearity,
or non-affine stress convergence.

