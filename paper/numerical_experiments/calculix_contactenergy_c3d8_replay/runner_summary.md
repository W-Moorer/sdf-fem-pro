# CalculiX Contactenergy C3D8 Replay

This validation uses the official CalculiX `contactenergy.inp` C3D8
surface-to-surface contact-energy test as an external contact-law
reference. SFC does not solve the C3D8 mechanics here; it replays the
final CalculiX deformed geometry with the current-surface dynamic SDF
query and the same linear pressure-overclosure law.

## Command

```bash
python validation/run_calculix_contactenergy_replay.py --out-dir results/calculix_contactenergy_replay
```

## Metrics

| Metric | Value |
| --- | ---: |
| CalculiX total contact spring energy | 2.00001e-05 |
| SFC dynamic-SDF replay contact energy | 2.000009924032692e-05 |
| contact energy relative error | 3.798346405165345e-08 |
| CalculiX force reference magnitude | 4.0 |
| SFC dynamic-SDF replay force z | 3.999999848082693 |
| contact force relative error | 3.797932679461269e-08 |
| SFC min gap | -1.0000049999958627e-05 |

## Claims

| Claim | Status | Evidence |
| --- | --- | --- |
| calculix_contactenergy_c3d8_reference_available | true | calculix_contactenergy_replay.csv::status |
| dynamic_sdf_replays_c3d8_contact_energy | true | calculix_contactenergy_replay.csv::contact_energy_rel_error |
| dynamic_sdf_is_not_tet4_bound | true | calculix_contactenergy_replay.csv::dynamic_sdf_backend |

## Boundaries

- This is a C3D8 external contact-law/energy replay, not a TET4 trajectory
  equivalence claim.
- Dynamic SDF is evaluated on triangulated current C3D8 boundary faces.
- Friction, self-contact, hard contact, and nonlinear material behavior are
  outside this replay.

## External Solver

- solver: CalculiX
- version: This is Version 2.17
- completed: true
- command: `wsl --exec bash -lc "cd '/mnt/e/workspace/sdf-fem-pro/results/calculix_contactenergy_replay/calculix_run' && ccx contactenergy"`
