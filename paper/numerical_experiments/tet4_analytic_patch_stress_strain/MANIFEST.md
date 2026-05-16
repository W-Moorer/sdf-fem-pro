# Manifest

## Data

- `data/stress_strain_patch.csv`: analytic TET4 stress-strain patch metrics.
- `data/stress_strain_patch_cloud.csv`: element stress/strain cloud values
  for SFC, analytic reference, and absolute error.
- `data/phase7_claims.csv`: generated Phase-7 claim gates.

## Figures

- `figures/stress_strain_error.png` and `.pdf`: stress-strain error trend.
- `figures/stress_strain_patch_3d.png` and `.pdf`: 3D boundary-surface
  stress/strain cloud comparing SFC and analytic fields.

## Reproducibility

- `README.md`: locked command, scope, and claim boundary.
- `locked_thresholds.csv`: acceptance thresholds enforced by tests.
- `runner_summary.md`: copied Markdown summary from the validation runner.
