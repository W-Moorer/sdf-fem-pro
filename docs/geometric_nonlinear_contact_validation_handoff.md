# Geometric Nonlinear Contact Validation Handoff

## Purpose

This handoff records the external contact evidence added after the no-contact
geometric nonlinear acceptance suite. The case is a gravity-driven deformable
block impacting a rigid plane. SFC uses a clean-room CalculiX-aligned StVK
geometric nonlinear implicit HHT/Newmark path with consistent TET4 mass,
material/geometric tangent, and penalty plane contact. CalculiX 2.17 is used
as the external dynamic contact reference.

This is a scoped validation. It does not claim source-level equivalence to
CalculiX surface-to-surface contact.

## Commands

Quick external run:

```bash
python validation/run_geometric_nonlinear_contact_validation.py --quick --out-dir results/geometric_nonlinear_contact_validation
```

Full r1/r2/r3 external run:

```bash
python validation/run_geometric_nonlinear_contact_validation.py --out-dir results/geometric_nonlinear_contact_validation_full
```

Local smoke run without CalculiX:

```bash
python validation/run_geometric_nonlinear_contact_validation.py --quick --skip-calculix --out-dir results/geometric_nonlinear_contact_validation_smoke
```

## Output Files

- `geometric_contact_history.csv`
- `geometric_contact_calculix_comparison.csv`
- `geometric_contact_mesh_convergence.csv`
- `geometric_contact_timestep_convergence.csv`
- `geometric_contact_stress_clouds.csv`
- `external_solver_commands.csv`
- `geometric_contact_claims.csv`
- `geometric_contact_validation_summary.md`

## Full Run Summary

CalculiX completed for r1, r2, and r3 with return code `0`.

| Resolution | first contact C/SFC | max pen. C | max pen. SFC | peak force C | peak force SFC | contact-zone VM C | contact-zone VM SFC | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `0.096 / 0.096` | `3.144397e-03` | `3.387373e-03` | `4.118874e+01` | `3.981581e+01` | `1.994642e+01` | `2.374152e+01` | `passed_scoped_gate` |
| 2 | `0.096 / 0.096` | `2.743927e-03` | `3.255793e-03` | `3.077931e+01` | `3.497837e+01` | `2.735979e+01` | `2.840869e+01` | `passed_scoped_gate` |
| 3 | `0.096 / 0.096` | `2.690467e-03` | `3.021339e-03` | `2.922631e+01` | `3.451957e+01` | `3.473394e+01` | `3.356425e+01` | `passed_scoped_gate` |

The scoped gate is intentionally loose. It confirms that the contact event,
force scale, penetration scale, and contact-zone stress scale are comparable.
It is not a high-precision contact equivalence claim.

## Mesh Trend

| Resolution | max penetration | peak normal force | peak contact energy | contact-zone max VM |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `3.387373e-03` | `3.981581e+01` | `6.038913e-02` | `2.374152e+01` |
| 2 | `3.255793e-03` | `3.497837e+01` | `4.915444e-02` | `2.840869e+01` |
| 3 | `3.021339e-03` | `3.451957e+01` | `4.625076e-02` | `3.356425e+01` |

This is a trend table only. No theoretical convergence order is claimed.

## Time-Step Trend

| dt | max penetration | peak normal force | peak contact energy | final z_cm |
| ---: | ---: | ---: | ---: | ---: |
| `2.0e-03` | `3.387373e-03` | `3.981581e+01` | `6.038913e-02` | `3.949636e-01` |
| `1.0e-03` | `3.583168e-03` | `3.963416e+01` | `5.987266e-02` | `3.949591e-01` |
| `5.0e-04` | `3.665087e-03` | `3.965632e+01` | `5.986156e-02` | `3.949543e-01` |

## ParaView Stress Clouds

The full run writes paired VTK stress clouds with a shared von Mises color scale
per resolution:

- `results/geometric_nonlinear_contact_validation_full/vtk_contact_stress_clouds/block_drop_r1/sfc_frame_0000.vtk`
- `results/geometric_nonlinear_contact_validation_full/vtk_contact_stress_clouds/block_drop_r1/calculix_frame_0000.vtk`
- `results/geometric_nonlinear_contact_validation_full/vtk_contact_stress_clouds/block_drop_r2/sfc_frame_0000.vtk`
- `results/geometric_nonlinear_contact_validation_full/vtk_contact_stress_clouds/block_drop_r2/calculix_frame_0000.vtk`
- `results/geometric_nonlinear_contact_validation_full/vtk_contact_stress_clouds/block_drop_r3/sfc_frame_0000.vtk`
- `results/geometric_nonlinear_contact_validation_full/vtk_contact_stress_clouds/block_drop_r3/calculix_frame_0000.vtk`

The shared color ranges are listed in
`geometric_contact_stress_clouds.csv`.

## Claim Gates

The full run reports:

| Claim | Supported |
| --- | --- |
| block-plane CalculiX comparison available | true |
| scoped block-plane contact acceptance | true |
| contact mesh convergence trend available | true |
| contact time-step convergence trend available | true |
| ParaView stress cloud comparison available | true |

## Limitations

- This validates the diagnostic clean-room CalculiX-aligned contact path, not
  a copied CalculiX implementation.
- The mechanics backend is separated from contact geometry so SFC dynamic-SDF
  contact queries can be swapped in; this external alignment runner uses the
  rigid-plane contact geometry for a controlled comparison.
- Contact remains normal penalty contact against a rigid plane.
- The scoped gate does not mean exact CalculiX surface-to-surface contact
  equivalence.
- Mesh and time-step results are trends only; no convergence order is claimed.
- Sphere-drop remains a harder follow-up case and is not used as the primary
  contact acceptance case here.
