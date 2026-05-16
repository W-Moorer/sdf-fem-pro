# Manifest

## Data

- `data/trajectory_comparison.csv`: primary locked trajectory metrics.
- `data/alignment_diagnostics.csv`: force-law and trajectory diagnostics.
- `data/claims.csv`: generated claim gates from the validation runner.
- `data/plots.csv`: generated plot registry.
- `data/external_solver_commands.csv`: CalculiX command metadata.

## Figures

- `figures/calculix_drop_z_cm.png` and `.pdf`: primary paper trajectory figure.
- `figures/calculix_drop_min_gap.png` and `.pdf`: contact gap history.
- `figures/calculix_drop_force_proxy.png` and `.pdf`: normal reaction/force history.
- `figures/calculix_drop_contact_energy.png` and `.pdf`: contact energy history.
- `figures/calculix_drop_total_energy.png` and `.pdf`: total energy diagnostic.

## Reproducibility

- `README.md`: locked command, claim boundary, and metrics.
- `locked_thresholds.csv`: acceptance thresholds enforced by tests.
- `runner_summary.md`: copied Markdown summary from the validation runner.
