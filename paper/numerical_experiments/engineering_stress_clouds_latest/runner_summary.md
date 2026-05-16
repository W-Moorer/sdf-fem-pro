# Phase-8 Engineering Validation Summary

This validation adds three paper-facing engineering cases without adding new core physics: a 3D cantilever beam external FEM comparison, a rigid flat indenter pressing an elastic block, and deformable-deformable block contact with action-reaction diagnostics.

## Reproduce

```bash
python validation/run_phase8_engineering_cases.py --out-dir results/phase8
```

For CI/reduced checks:

```bash
python validation/run_phase8_engineering_cases.py --quick --out-dir results/phase8
```

## Claims

| Claim | Status | Evidence | Gate value |
| --- | --- | --- | ---: |
| phase8_external_cantilever_fem_agreement | supported | phase8_cantilever_external.csv::stress_l2_rel_error | 8.069196e-13 |
<!-- evidence csv=phase8_cantilever_external.csv field=stress_l2_rel_error -->
| phase8_rigid_indenter_contact_response | supported | phase8_rigid_indenter_history.csv::normal_force_z | -2.880000e+04 |
<!-- evidence csv=phase8_rigid_indenter_history.csv field=normal_force_z -->
| phase8_deformable_deformable_action_reaction | supported | phase8_deformable_deformable_history.csv::action_reaction_imbalance | 0.000000e+00 |
<!-- evidence csv=phase8_deformable_deformable_history.csv field=action_reaction_imbalance -->

## 3D Cantilever External FEM

| Resolution | Elements | Disp. rel. err. | Stress rel. err. | K rel. err. |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 15 | 4.689785e-14 | 4.844585e-14 | 2.631252e-16 |
| 2 | 120 | 4.609882e-13 | 4.639006e-13 | 2.759517e-16 |
| 3 | 405 | 7.848542e-13 | 8.069196e-13 | 5.162660e-16 |

## Rigid Flat Indenter

| Indentation | Active contacts | Min gap | Max penetration | Normal force z | Max VM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000000e+00 | 0 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 6.000000e-03 | 16 | -6.000000e-03 | 6.000000e-03 | -7.200000e+03 | 9.928910e+03 |
| 1.200000e-02 | 16 | -1.200000e-02 | 1.200000e-02 | -1.440000e+04 | 1.985782e+04 |
| 1.800000e-02 | 16 | -1.800000e-02 | 1.800000e-02 | -2.160000e+04 | 2.978673e+04 |
| 2.400000e-02 | 16 | -2.400000e-02 | 2.400000e-02 | -2.880000e+04 | 3.971564e+04 |

## Deformable-Deformable Contact

| Approach | Active contacts | Min gap | Max penetration | Upper force z | Lower force z | Imbalance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000000e+00 | 0 | 3.500000e-02 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 2.500000e-02 | 0 | 1.000000e-02 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 4.000000e-02 | 36 | -5.000000e-03 | 5.000000e-03 | 1.080000e+04 | -1.080000e+04 | 0.000000e+00 |
| 5.500000e-02 | 36 | -2.000000e-02 | 2.000000e-02 | 4.320000e+04 | -4.320000e+04 | 0.000000e+00 |
| 7.000000e-02 | 36 | -3.500000e-02 | 3.500000e-02 | 7.560000e+04 | -7.560000e+04 | 0.000000e+00 |

## Plots

- `phase8_cantilever_external_stress_3d.png` and `phase8_cantilever_external_stress_3d.pdf`
- `phase8_rigid_indenter_force_history.png` and `phase8_rigid_indenter_force_history.pdf`
- `phase8_rigid_indenter_stress_3d.png` and `phase8_rigid_indenter_stress_3d.pdf`
- `phase8_deformable_deformable_history.png` and `phase8_deformable_deformable_history.pdf`
- `phase8_deformable_deformable_stress_3d.png` and `phase8_deformable_deformable_stress_3d.pdf`

## Limitations

- The rigid indenter is modeled as a prescribed flat rigid underside surface, not a curved Hertzian indenter.
- The contact cases are prescribed-displacement validation cases; they are not nonlinear contact equilibrium solves.
- Contact remains frictionless normal penalty contact, and the FEM model remains small-strain linear TET4.
- Stress plots use nodal averaging and subdivided boundary triangles for visualization; quantitative checks use CSV fields.
