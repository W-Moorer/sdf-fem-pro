# Persistent C3D4 F2F External Comparison Result

## Command

```bash
python validation/run_geometric_nonlinear_contact_validation.py --contact-mode persistent_calculix_c3d4_f2f --out-dir results/geometric_nonlinear_contact_validation_persistent_c3d4_f2f
```

The run completed successfully and wrote:

- `results/geometric_nonlinear_contact_validation_persistent_c3d4_f2f/geometric_contact_history.csv`
- `results/geometric_nonlinear_contact_validation_persistent_c3d4_f2f/geometric_contact_calculix_comparison.csv`
- `results/geometric_nonlinear_contact_validation_persistent_c3d4_f2f/geometric_contact_mesh_convergence.csv`
- `results/geometric_nonlinear_contact_validation_persistent_c3d4_f2f/geometric_contact_timestep_convergence.csv`
- `results/geometric_nonlinear_contact_validation_persistent_c3d4_f2f/geometric_contact_stress_clouds.csv`
- `results/geometric_nonlinear_contact_validation_persistent_c3d4_f2f/external_solver_commands.csv`
- `results/geometric_nonlinear_contact_validation_persistent_c3d4_f2f/geometric_contact_claims.csv`
- `results/geometric_nonlinear_contact_validation_persistent_c3d4_f2f/geometric_contact_validation_summary.md`

## Compared Results

Compared directories:

- old three-point quadrature: `results/geometric_nonlinear_contact_validation_full`
- strict non-persistent C3D4 F2F: `results/geometric_nonlinear_contact_validation_c3d4_f2f`
- persistent C3D4 F2F: `results/geometric_nonlinear_contact_validation_persistent_c3d4_f2f`

## Relative Error Comparison

| Resolution | Metric | Old three-point | C3D4 F2F | Persistent C3D4 F2F | Persistent vs old | Persistent vs C3D4 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| r1 | max penetration | `7.727284e-02` | `4.817436e-02` | `4.817436e-02` | `-2.909849e-02` | `0.000000e+00` |
| r1 | RF / normal force | `3.333271e-02` | `5.722847e-02` | `5.722847e-02` | `+2.389576e-02` | `0.000000e+00` |
| r1 | CELS / contact energy | `6.486660e-02` | `1.112276e-01` | `1.112276e-01` | `+4.636101e-02` | `0.000000e+00` |
| r2 | max penetration | `1.865451e-01` | `1.600986e-01` | `1.600986e-01` | `-2.644655e-02` | `0.000000e+00` |
| r2 | RF / normal force | `1.364248e-01` | `1.238871e-01` | `1.238871e-01` | `-1.253770e-02` | `0.000000e+00` |
| r2 | CELS / contact energy | `3.018645e-01` | `2.883060e-01` | `2.883060e-01` | `-1.355845e-02` | `0.000000e+00` |
| r3 | max penetration | `1.229794e-01` | `1.505151e-01` | `1.505151e-01` | `+2.753576e-02` | `0.000000e+00` |
| r3 | RF / normal force | `1.811129e-01` | `1.880409e-01` | `1.880409e-01` | `+6.928053e-03` | `0.000000e+00` |
| r3 | CELS / contact energy | `3.820071e-01` | `3.998027e-01` | `3.998027e-01` | `+1.779559e-02` | `0.000000e+00` |

Negative changes mean the persistent result is closer to CalculiX. Positive
changes mean it is farther from CalculiX.

## New Absolute Metrics

| Resolution | max pen. C | max pen. SFC | peak RF C | peak force SFC | peak CELS C | peak energy SFC | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r1 | `3.144397e-03` | `3.295876e-03` | `4.118874e+01` | `4.354591e+01` | `6.457809e-02` | `7.176096e-02` | `passed_scoped_gate` |
| r2 | `2.743927e-03` | `3.183225e-03` | `3.077931e+01` | `3.459247e+01` | `3.775696e-02` | `4.864252e-02` | `passed_scoped_gate` |
| r3 | `2.690467e-03` | `3.095423e-03` | `2.922631e+01` | `3.472205e+01` | `3.346637e-02` | `4.684631e-02` | `passed_scoped_gate` |

## CNUM / Active Count Check

| Resolution | CalculiX CNUM | old active samples | C3D4 active springs | persistent active springs | persistent generated springs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | `14` | `6` | `2` | `2` | `2` |
| r2 | `56` | `24` | `8` | `8` | `8` |
| r3 | `126` | `54` | `18` | `18` | `18` |

Persistent generated spring count did not move closer to CalculiX `CNUM` in
this run. The current lifecycle has zero release tolerance and no cutback retry
loop, so generated spring count matches the non-persistent active spring count.

## Cutback Diagnostic

| Resolution | max generated springs | max active springs | cutback recommendation count | reasons |
| ---: | ---: | ---: | ---: | --- |
| r1 | `2` | `2` | `38` | `active_set_oscillation`, `residual_growth`, `stable_or_improving` |
| r2 | `8` | `8` | `32` | `residual_growth`, `stable_or_improving` |
| r3 | `18` | `18` | `33` | `residual_growth`, `stable_or_improving` |

The diagnostics detect many steps where a cutback/retry would be recommended,
but the current runner only records the recommendation. It does not yet repeat
the time step with a smaller increment or freeze the previous contact spring set
inside that retry.

## Interpretation

The persistent mode did not improve the numerical comparison relative to the
non-persistent C3D4 F2F run. The metrics are identical to the previous strict
C3D4 F2F result:

- max penetration is improved versus the older three-point run for r1 and r2,
  but worse for r3;
- RF/normal force is improved versus the older three-point run only for r2;
- CELS/contact energy is improved versus the older three-point run only for r2;
- CNUM/generation count is not closer to CalculiX.

This is expected from the current implementation because persistent bookkeeping
is diagnostic unless it changes time-step control or spring retention. The
current lifecycle stores and reports spring state, but the force-producing set
remains the same as non-persistent C3D4 F2F for this benchmark.

## Next Required Change

To actually change the numerical trajectory, the runner needs an active cutback
loop:

1. detect `contact_cutback_recommended=true`;
2. reject the current step;
3. reduce `dt`;
4. freeze or reuse the previous generated contact spring set during retry;
5. accept the step only after active-set and residual diagnostics stabilize.

Only after this is implemented should CNUM/CELS/RF/max-penetration be expected
to move closer to CalculiX through persistent active-set behavior.
