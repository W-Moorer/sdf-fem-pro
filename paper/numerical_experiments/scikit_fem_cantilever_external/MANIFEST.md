# Manifest

## Data

- `data/external_fem_comparison.csv`: primary SFC versus scikit-fem metrics.
- `data/external_fem_stress_cloud.csv`: stress-cloud export data.
- `data/external_fem_plots.csv`: generated plot registry.

## Figures

- `figures/external_fem_error_trends.png` and `.pdf`: numerical agreement
  trends over mesh resolution.
- `figures/external_fem_stress_3d.png` and `.pdf`: primary 3D nodal-averaged
  interpolated boundary stress cloud.
- `figures/external_fem_stress_cloud.png` and `.pdf`: secondary stress-cloud
  figure retained from the validation runner.

## Reproducibility

- `README.md`: locked command, stress-cloud scheme, and claim boundary.
- `locked_thresholds.csv`: acceptance thresholds enforced by tests.
- `runner_summary.md`: copied Markdown summary from the validation runner.

