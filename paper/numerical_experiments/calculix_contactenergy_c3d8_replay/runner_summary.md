# CalculiX Contactenergy C3D8 Replay

This validation uses the official CalculiX `contactenergy.inp` C3D8
surface-to-surface contact-energy test as an external contact-law
reference. The primary SFC evidence replays the final CalculiX
deformed geometry with the current-surface dynamic SDF query and the
same linear pressure-overclosure law. A separate registered SFC C3D8
static backend solves the same input to produce direct field-error
plots.

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
| max post-processed von Mises stress | 100.0004565 |
| max post-processed engineering strain norm | 0.00047619265000000007 |
| max CalculiX displacement magnitude | 0.0001242862 |
| SFC validation C3D8 displacement L2 relative error | 0.37087960200929965 |
| SFC validation C3D8 stress L2 relative error | 0.007165960357991534 |
| SFC validation C3D8 von Mises L2 relative error | 0.0018828589182671497 |

## Figures

- `figures/calculix_contactenergy_stress_strain_3d.png`: C3D8
  stress, strain, and displacement clouds post-processed from the final
  CalculiX displacement field. The deformed shape is plotted with a
  labeled magnification factor so that the small static displacement is
  visible.
- `figures/calculix_contactenergy_contact_pressure_3d.png`: SFC
  dynamic-SDF replay contact pressure on the same current C3D8 boundary.
- `figures/calculix_contactenergy_error_metrics.png`: normalized
  CalculiX-vs-SFC replay values and log-scale relative errors for
  normal force and contact energy.
- `figures/calculix_contactenergy_sfc_c3d8_error_3d.png`: direct
  registered SFC C3D8 static solve compared with the CalculiX C3D8
  stress field on the same boundary.

## Claims

| Claim | Status | Evidence |
| --- | --- | --- |
| calculix_contactenergy_c3d8_reference_available | true | calculix_contactenergy_replay.csv::status |
| dynamic_sdf_replays_c3d8_contact_energy | true | calculix_contactenergy_replay.csv::contact_energy_rel_error |
| dynamic_sdf_is_not_tet4_bound | true | calculix_contactenergy_replay.csv::dynamic_sdf_backend |

## Boundaries

- This is a C3D8 external contact-law/energy replay, not a TET4 trajectory
  equivalence claim.
- The registered SFC C3D8 static backend is included for direct
  field-error plotting; nonlinear C3D8 trajectories are not claimed.
- Dynamic SDF is evaluated on triangulated current C3D8 boundary faces.
- Friction, self-contact, hard contact, and nonlinear material behavior are
  outside this replay.

## External Solver

- solver: CalculiX
- version: This is Version 2.17
- completed: true
- command: `wsl --exec bash -lc "cd '/mnt/e/workspace/sdf-fem-pro/results/calculix_contactenergy_replay/calculix_run' && ccx contactenergy"`
