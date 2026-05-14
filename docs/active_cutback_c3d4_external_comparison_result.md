# Active-Cutback C3D4 F2F External Comparison Result

## Command

```bash
python validation/run_geometric_nonlinear_contact_validation.py --contact-mode persistent_calculix_c3d4_f2f --out-dir results/geometric_nonlinear_contact_validation_active_cutback_c3d4_f2f
```

The run completed successfully and wrote:

- `results/geometric_nonlinear_contact_validation_active_cutback_c3d4_f2f/geometric_contact_history.csv`
- `results/geometric_nonlinear_contact_validation_active_cutback_c3d4_f2f/geometric_contact_calculix_comparison.csv`
- `results/geometric_nonlinear_contact_validation_active_cutback_c3d4_f2f/geometric_contact_mesh_convergence.csv`
- `results/geometric_nonlinear_contact_validation_active_cutback_c3d4_f2f/geometric_contact_timestep_convergence.csv`
- `results/geometric_nonlinear_contact_validation_active_cutback_c3d4_f2f/geometric_contact_stress_clouds.csv`
- `results/geometric_nonlinear_contact_validation_active_cutback_c3d4_f2f/external_solver_commands.csv`
- `results/geometric_nonlinear_contact_validation_active_cutback_c3d4_f2f/geometric_contact_claims.csv`
- `results/geometric_nonlinear_contact_validation_active_cutback_c3d4_f2f/geometric_contact_validation_summary.md`

## Compared Baselines

- old three-point quadrature: `results/geometric_nonlinear_contact_validation_full`
- non-persistent C3D4 F2F: `results/geometric_nonlinear_contact_validation_c3d4_f2f`
- passive persistent C3D4 F2F: `results/geometric_nonlinear_contact_validation_persistent_c3d4_f2f`
- active-cutback persistent C3D4 F2F: `results/geometric_nonlinear_contact_validation_active_cutback_c3d4_f2f`

## Relative Error Comparison

| Resolution | Metric | Old three-point | C3D4 F2F | Passive persistent | Active retry | Active vs passive | Active vs old |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | max penetration | `7.727284e-02` | `4.817436e-02` | `4.817436e-02` | `5.331147e-02` | `+5.137113e-03` | `-2.396137e-02` |
| r1 | RF / normal force | `3.333271e-02` | `5.722847e-02` | `5.722847e-02` | `6.247805e-02` | `+5.249580e-03` | `+2.914534e-02` |
| r1 | CELS / contact energy | `6.486660e-02` | `1.112276e-01` | `1.112276e-01` | `1.222185e-01` | `+1.099089e-02` | `+5.735190e-02` |
| r2 | max penetration | `1.865451e-01` | `1.600986e-01` | `1.600986e-01` | `1.707939e-01` | `+1.069532e-02` | `-1.575123e-02` |
| r2 | RF / normal force | `1.364248e-01` | `1.238871e-01` | `1.238871e-01` | `1.507992e-01` | `+2.691217e-02` | `+1.437447e-02` |
| r2 | CELS / contact energy | `3.018645e-01` | `2.883060e-01` | `2.883060e-01` | `3.215809e-01` | `+3.327494e-02` | `+1.971649e-02` |
| r3 | max penetration | `1.229794e-01` | `1.505151e-01` | `1.505151e-01` | `1.471759e-01` | `-3.339256e-03` | `+2.419651e-02` |
| r3 | RF / normal force | `1.811129e-01` | `1.880409e-01` | `1.880409e-01` | `1.485869e-01` | `-3.945403e-02` | `-3.252598e-02` |
| r3 | CELS / contact energy | `3.820071e-01` | `3.998027e-01` | `3.998027e-01` | `3.272825e-01` | `-7.252015e-02` | `-5.472456e-02` |

Negative change means the active retry result moved closer to CalculiX.
Positive change means it moved farther away.

## Active Retry Absolute Metrics

| Resolution | max pen. C | max pen. SFC | peak RF C | peak force SFC | peak CELS C | peak energy SFC | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| r1 | `3.144397e-03` | `3.312029e-03` | `4.118874e+01` | `4.376213e+01` | `6.457809e-02` | `7.247073e-02` | `passed_scoped_gate` |
| r2 | `2.743927e-03` | `3.212573e-03` | `3.077931e+01` | `3.542081e+01` | `3.775696e-02` | `4.989888e-02` | `passed_scoped_gate` |
| r3 | `2.690467e-03` | `3.086438e-03` | `2.922631e+01` | `3.356896e+01` | `3.346637e-02` | `4.441933e-02` | `passed_scoped_gate` |

## CNUM / Active Count Check

| Resolution | CalculiX CNUM | old active samples | C3D4 active springs | passive generated springs | active generated springs | active active samples |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | `14` | `6` | `2` | `2` | `2` | `2` |
| r2 | `56` | `24` | `8` | `8` | `8` | `8` |
| r3 | `126` | `54` | `18` | `18` | `18` | `18` |

CNUM did not improve. The active retry loop changes time-step acceptance and
state trajectory, but it does not change the number of generated slave-face
springs for this block-plane mesh.

## Cutback Summary

| Resolution | min accepted dt | max retry count | rows with retry | cutback-limited rows | max generated springs | max active samples |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | `1.250000e-04` | `4` | `65` | `0` | `2` | `2` |
| r2 | `1.250000e-04` | `4` | `76` | `0` | `8` | `8` |
| r3 | `1.250000e-04` | `4` | `83` | `0` | `18` | `18` |

The retry loop was active and reduced accepted time steps. No row hit the
cutback-limited guard.

## Interpretation

Active cutback retry changed the SFC trajectory, but it did not uniformly
improve the external CalculiX comparison.

Improvements:

- r1 and r2 max-penetration error remain better than the old three-point
  baseline.
- r3 RF and CELS errors improve relative to both old three-point and passive
  C3D4 F2F.
- r3 max-penetration error improves slightly relative to passive C3D4 F2F.

Regressions:

- r1 RF and CELS are farther from CalculiX than both old three-point and passive
  C3D4 F2F.
- r2 RF and CELS are farther from CalculiX than both old three-point and passive
  C3D4 F2F.
- CNUM is unchanged and remains far from CalculiX output.

## Conclusion

The active retry loop is functioning numerically, but the comparison result is
mixed. The remaining discrepancy is not just time-increment rejection. The next
source of mismatch is likely the definition and multiplicity of CalculiX contact
elements behind `CNUM`/`CELS`, plus the exact increment acceptance criteria and
contact-element retention rules.

Do not claim that active retry makes SFC equivalent to CalculiX. The supported
claim is narrower: active retry is implemented, it changes accepted time steps,
and it improves some r3 force/energy metrics, but it does not produce uniform
agreement improvement.
