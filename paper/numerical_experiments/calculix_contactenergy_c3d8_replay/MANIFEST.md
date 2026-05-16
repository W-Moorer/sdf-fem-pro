# Manifest

## Data

- `data/calculix_contactenergy_replay.csv`: primary replay metrics.
- `data/calculix_contactenergy_claims.csv`: generated claim gates.
- `data/calculix_contactenergy_commands.csv`: CalculiX command metadata.
- `data/calculix_contactenergy_raw_cels.csv`: raw CalculiX CELS rows when
  available from the selected CalculiX output.
- `data/contactenergy.inp`: copied official CalculiX input used for the run.
- `data/contactenergy.dat`: CalculiX output used for the replay.

## Logs

- `logs/calculix_stdout.log`
- `logs/calculix_stderr.log`

## Reproducibility

- `README.md`: command, scope, and claim boundary.
- `locked_thresholds.csv`: acceptance thresholds enforced by tests.
- `runner_summary.md`: copied Markdown summary from the validation runner.

