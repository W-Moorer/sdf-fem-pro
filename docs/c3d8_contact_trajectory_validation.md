# C3D8 Contact Trajectory Validation

## Command

```bash
python validation/run_c3d8_contact_trajectory_validation.py --quick --out-dir results/c3d8_contact_trajectory_quick
```

## Output Package

- `results/c3d8_contact_trajectory_quick/c3d8_contact_trajectory.csv`
- `results/c3d8_contact_trajectory_quick/c3d8_contact_trajectory_summary.csv`
- `results/c3d8_contact_trajectory_quick/c3d8_contact_trajectory_stress_cloud.csv`
- `results/c3d8_contact_trajectory_quick/c3d8_contact_trajectory_claims.csv`
- `results/c3d8_contact_trajectory_quick/figures/c3d8_contact_trajectory_z_cm.png`
- `results/c3d8_contact_trajectory_quick/figures/c3d8_contact_trajectory_gap.png`
- `results/c3d8_contact_trajectory_quick/figures/c3d8_contact_trajectory_force_energy.png`
- `results/c3d8_contact_trajectory_quick/figures/c3d8_contact_trajectory_stress_cloud.png`

The same quick evidence is locked under:

```text
paper/numerical_experiments/c3d8_contact_trajectory_validation/
```

## Metrics

| Case | Rows | Active rows | Peak RF rel. error | Peak CELS rel. error | CDIS/gap rel. error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `block_plane_c3d8` | 69 | 22 | `1.435099e-4` | `2.403822e-4` | `1.032577e-4` |
| `block_block_c3d8` | 70 | 22 | `2.428073e-4` | `2.315966e-4` | `3.475215e-6` |

Both CalculiX runs completed with return code 0. The runner requested nodal
displacement/velocity, fixed-support `RF`, contact `CDIS/CSTR/CELS`, total
`CNUM`, and element stress/strain output. SFC then replayed the exported
current C3D8 configurations using triangulated C3D8 boundary faces and
`dynamic_surface_sdf`.

## Claim Boundary

Supported:

- The dynamic SDF query is not restricted to TET4; it can replay triangulated
  current C3D8 boundary surfaces.
- In the generated quick C3D8 block-plane and block-block cases, SFC replay
  matches CalculiX `CDIS`, fixed-support `RF`, and `CELS` to small relative
  error.

Not supported:

- Native SFC nonlinear C3D8 dynamics equivalence to CalculiX.
- A production C3D8 contact solver implementation.
- Friction, self-contact, barrier contact, or source-level CalculiX equivalence.
