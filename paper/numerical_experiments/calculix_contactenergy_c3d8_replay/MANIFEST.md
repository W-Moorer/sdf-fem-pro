# Manifest

## Data

- `data/calculix_contactenergy_replay.csv`: primary replay metrics.
- `data/calculix_contactenergy_claims.csv`: generated claim gates.
- `data/calculix_contactenergy_commands.csv`: CalculiX command metadata.
- `data/calculix_contactenergy_raw_cels.csv`: raw CalculiX CELS rows when
  available from the selected CalculiX output.
- `data/calculix_contactenergy_plots.csv`: generated visualization manifest.
- `data/calculix_contactenergy_stress_strain_cloud.csv`: C3D8 element-center
  strain and stress values post-processed from CalculiX final displacements.
- `data/contactenergy.inp`: copied official CalculiX input used for the run.
- `data/contactenergy.dat`: CalculiX output used for the replay.

## Figures

- `figures/calculix_contactenergy_stress_strain_3d.png`
- `figures/calculix_contactenergy_stress_strain_3d.pdf`
- `figures/calculix_contactenergy_contact_pressure_3d.png`
- `figures/calculix_contactenergy_contact_pressure_3d.pdf`

## Logs

- `logs/calculix_stdout.log`
- `logs/calculix_stderr.log`

## Reproducibility

- `README.md`: command, scope, and claim boundary.
- `locked_thresholds.csv`: acceptance thresholds enforced by tests.
- `runner_summary.md`: copied Markdown summary from the validation runner.
