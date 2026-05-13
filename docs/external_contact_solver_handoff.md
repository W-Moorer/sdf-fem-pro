# External Contact Solver Comparison Handoff

## Scope

This validation adds an external open-source contact solver comparison using
SfePy 2025.3. SfePy runs its built-in two-body nonlinear penalty contact
example. The SFC code then replays the final SfePy deformed geometry and
evaluates current-surface dynamic SDF gaps on the contact interface.

This is external contact-solver evidence for the contact geometry layer. It is
not a claim that the current SFC package implements a full nonlinear contact
equilibrium solver.

## Commands

Full output:

```bash
python validation/run_external_contact_solver_comparison.py --out-dir results/external_contact_solver
```

Quick/CI output:

```bash
python validation/run_external_contact_solver_comparison.py --quick --out-dir results/external_contact_solver
```

Targeted test:

```bash
pytest -q tests/test_external_contact_solver_comparison.py
```

## Output Files

- `results/external_contact_solver/external_contact_solver_comparison.csv`
- `results/external_contact_solver/external_contact_solver_commands.csv`
- `results/external_contact_solver/external_contact_solver_claims.csv`
- `results/external_contact_solver/external_contact_solver_plots.csv`
- `results/external_contact_solver/external_contact_solver_summary.md`
- `results/external_contact_solver/external_contact_gap_comparison.png`
- `results/external_contact_solver/external_contact_gap_comparison.pdf`
- `results/external_contact_solver/external_contact_active_count.png`
- `results/external_contact_solver/external_contact_active_count.pdf`
- `results/external_contact_solver/sfepy_runs/`

## Metrics

| Approach | SfePy mean gap | SFC replay mean gap | Abs. difference | SfePy active | SFC active |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `0.000000e+00` | `0.000000e+00` | `1.000000e-04` | `1.000000e-04` | 0 | 0 |
| `4.000000e-02` | `-2.107687e-03` | `-2.581917e-03` | `4.742296e-04` | 2 | 1 |
| `6.000000e-02` | `-3.164172e-03` | `-3.876110e-03` | `7.119386e-04` | 2 | 1 |
| `8.000000e-02` | `-4.220656e-03` | `-5.170304e-03` | `9.496477e-04` | 2 | 1 |
| `1.000000e-01` | `-5.277141e-03` | `-6.464498e-03` | `1.187357e-03` | 2 | 1 |

## Claim Gates

| Claim | Evidence | Gate value | Status |
| --- | --- | ---: | --- |
| Contact state agreement | `external_contact_solver_comparison.csv::contact_state_agreement` | `true` | supported |
| Gap scale check | `external_contact_solver_comparison.csv::gap_mean_abs_difference` | `1.187357e-03` | supported |

## Interpretation

- SfePy provides the external nonlinear contact solve.
- SFC evaluates dynamic current-surface gap queries on the external solver's
  final deformed geometry.
- Both methods agree on contact activation for the tested prescribed approaches.
- Gap magnitudes follow the same trend; the maximum mean-gap difference is
  `1.187357e-03`.

## Limitations

- SfePy's exported gap is a contact-term surface average; SFC replay values are
  centroid closest-point projection gaps. They are comparable but not identical
  numerical quantities.
- The comparison does not prove that SFC has a full nonlinear contact
  equilibrium solver.
- The external case is SfePy's two-body penalty contact example, not Abaqus,
  CalculiX, IPC, or a commercial production contact code.
