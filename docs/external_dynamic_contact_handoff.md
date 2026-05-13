# External Dynamic Contact Handoff

## Scope

This validation replaces rigid-body external references with SfePy-based FEM
evidence. It contains two external SfePy layers:

- a transient elastodynamic run with mass matrix and SfePy Newmark integration
  over `T = 3 s`;
- SfePy two-body penalty-contact snapshot solves sampled along the same
  3-second approach path.

SFC outputs its own Newmark penalty-contact time history and also replays SfePy
dynamic/contact geometries with the current-surface dynamic SDF query.

This is not claimed as a single coupled SfePy transient nonlinear contact solve.

## Commands

Quick/CI:

```bash
python validation/run_external_dynamic_contact_comparison.py --quick --out-dir results/external_dynamic_contact
```

Full:

```bash
python validation/run_external_dynamic_contact_comparison.py --out-dir results/external_dynamic_contact
```

Targeted test:

```bash
pytest -q tests/test_external_dynamic_contact_comparison.py
```

## Output Files

- `results/external_dynamic_contact/external_dynamic_contact_time_history.csv`
- `results/external_dynamic_contact/external_dynamic_contact_comparison.csv`
- `results/external_dynamic_contact/external_dynamic_contact_claims.csv`
- `results/external_dynamic_contact/external_dynamic_contact_commands.csv`
- `results/external_dynamic_contact/external_dynamic_contact_plots.csv`
- `results/external_dynamic_contact/external_dynamic_contact_summary.md`
- `results/external_dynamic_contact/external_dynamic_contact_min_gap.png`
- `results/external_dynamic_contact/external_dynamic_contact_min_gap.pdf`
- `results/external_dynamic_contact/external_dynamic_contact_force_history.png`
- `results/external_dynamic_contact/external_dynamic_contact_force_history.pdf`
- `results/external_dynamic_contact/external_dynamic_contact_active_count.png`
- `results/external_dynamic_contact/external_dynamic_contact_active_count.pdf`

## Full-Mode Metrics

The non-quick run in `results/external_dynamic_contact` reports:

| Metric | Value | Status |
| --- | ---: | --- |
| total time | `3.0 s` | ok |
| SfePy transient elastodynamic rows | `301` | ok |
| SfePy contact snapshot rows | `13` | ok |
| SFC Newmark contact rows | `301` | ok |
| first SfePy contact snapshot time | `0.25 s` | ok |
| first SFC contact-snapshot replay time | `0.25 s` | ok |
| active-state agreement fraction | `1.0` | ok |
| max mean-gap absolute difference | `1.425066e-03` | ok |
| force proxy cosine similarity | `1.0` | ok |
| SFC contact force direction | `true` | ok |

## Supported Claims

- SfePy provides a validation-only transient elastodynamic reference with mass
  matrix and Newmark integration over `T = 3 s`.
- SfePy provides external two-body penalty-contact snapshot references along
  the same approach path.
- SFC current-surface dynamic SDF replay matches SfePy contact snapshot
  activation and gap scale for the tested path.
- SFC produces its own 3-second Newmark penalty-contact time history with an
  assembled mass matrix.

## Limitations

- The SfePy dynamic and SfePy contact layers are separate references.
- This does not prove a full SFC nonlinear contact equilibrium solver.
- Contact remains frictionless normal penalty contact.
- The contact force field is reported as a penalty-force proxy because SfePy
  exports contact gaps, not a directly matched nodal force vector in this
  validation path.
- The model is small and generated for deterministic validation; it is not an
  industrial-scale benchmark.
