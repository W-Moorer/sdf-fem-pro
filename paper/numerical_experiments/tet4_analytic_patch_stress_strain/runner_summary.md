# Phase-7 Physical Validation Summary

This validation adds stress/strain, standard FEM reference, contact force reference, and total-step acceleration evidence. It does not add new physics.

## Claims

| Claim | Status | Evidence | Gate value |
| --- | --- | --- | ---: |
| stress_strain_physical_correctness | supported | phase7_stress_strain.csv::max_stress_l2_error | 0.000000e+00 |
<!-- evidence csv=phase7_stress_strain.csv field=max_stress_l2_error -->
| contact_force_reference_correctness | supported | phase7_contact_reference.csv::normal_force_rel_error | 0.000000e+00 |
<!-- evidence csv=phase7_contact_reference.csv field=normal_force_rel_error -->
| total_step_acceleration_feasibility | supported | phase7_acceleration_feasibility.csv::speedup_vs_bruteforce | 4.034891e+00 |
<!-- evidence csv=phase7_acceleration_feasibility.csv field=speedup_vs_bruteforce -->

## Stress/Strain Patch

| Resolution | Elements | Max strain error | Max stress error |
| ---: | ---: | ---: | ---: |
| 1 | 5 | 0.000000e+00 | 0.000000e+00 |
| 2 | 40 | 0.000000e+00 | 0.000000e+00 |
| 3 | 135 | 0.000000e+00 | 0.000000e+00 |
| 4 | 320 | 0.000000e+00 | 0.000000e+00 |

## Cantilever Standard FEM Reference

| Resolution | Tip uz | Max von Mises | Tip rel. error to finest | Max VM rel. error to finest |
| ---: | ---: | ---: | ---: | ---: |
| 1 | -1.397105e-03 | 9.796707e+01 | 8.538791e-01 | 6.884284e-01 |
| 2 | -3.686769e-03 | 1.835477e+02 | 6.144070e-01 | 4.162502e-01 |
| 3 | -6.617482e-03 | 2.557485e+02 | 3.078887e-01 | 1.866248e-01 |
| 4 | -9.561298e-03 | 3.144287e+02 | 0.000000e+00 | 0.000000e+00 |

## Cantilever Stress Cloud

| Resolution | Stress samples | Min von Mises | Max von Mises |
| ---: | ---: | ---: | ---: |
| 1 | 5 | 1.280425e+01 | 9.796707e+01 |
| 2 | 40 | 8.043697e+00 | 1.835477e+02 |
| 3 | 135 | 1.454094e+01 | 2.557485e+02 |
| 4 | 320 | 2.024515e+01 | 3.144287e+02 |

## Contact Force-Displacement Reference

| Penetration | Brute force | Dynamic SDF | Relative error | Active counts |
| ---: | ---: | ---: | ---: | --- |
| 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0/0 |
| 2.500000e-03 | 8.000000e+02 | 8.000000e+02 | 0.000000e+00 | 64/64 |
| 5.000000e-03 | 1.600000e+03 | 1.600000e+03 | 0.000000e+00 | 64/64 |
| 1.000000e-02 | 3.200000e+03 | 3.200000e+03 | 0.000000e+00 | 64/64 |
| 2.000000e-02 | 6.400000e+03 | 6.400000e+03 | 0.000000e+00 | 64/64 |

## Acceleration Feasibility

| Resolution | Method | Stage | Mean seconds | Speedup | Status |
| ---: | --- | --- | ---: | ---: | --- |
| 4 | dynamic_fem_sdf_contact | contact_pipeline_only | 1.169060e-02 | 4.533725e+00 | speedup_supported |
| 4 | dynamic_fem_sdf_total_step | fem_assembly_linear_solve_plus_contact | 1.787493e-02 | 4.034891e+00 | speedup_supported |
| 8 | dynamic_fem_sdf_contact | contact_pipeline_only | 4.592957e-02 | 1.716508e+01 | speedup_supported |
| 8 | dynamic_fem_sdf_total_step | fem_assembly_linear_solve_plus_contact | 5.948010e-02 | 1.412017e+01 | speedup_supported |
| 12 | dynamic_fem_sdf_contact | contact_pipeline_only | 2.034044e-01 | 2.592048e+01 | speedup_supported |
| 12 | dynamic_fem_sdf_total_step | fem_assembly_linear_solve_plus_contact | 1.506099e-01 | 3.091095e+01 | speedup_supported |

## Plots

- `phase7_stress_strain_error.png` and `phase7_stress_strain_error.pdf`
- `phase7_cantilever_fem_reference.png` and `phase7_cantilever_fem_reference.pdf`
- `phase7_cantilever_stress_cloud.png` and `phase7_cantilever_stress_cloud.pdf`
- `phase7_cantilever_stress_3d.png` and `phase7_cantilever_stress_3d.pdf`
- `phase7_contact_force_displacement.png` and `phase7_contact_force_displacement.pdf`
- `phase7_acceleration_feasibility.png` and `phase7_acceleration_feasibility.pdf`

## Interpretation

- The stress/strain patch validates element recovery for affine small-strain linear elasticity.
- The cantilever study is a standard FEM mesh-refinement reference, not an external commercial-solver benchmark.
- The cantilever stress plots average element von Mises stress to nodes and interpolate over subdivided boundary triangles for visualization.
- The contact force-displacement curve validates dynamic SDF contact against brute-force closest-point projection contact.
- The acceleration table separates contact-only acceleration from total-step timing, which is the relevant claim boundary for paper text.
