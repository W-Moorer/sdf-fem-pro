# C3D8 Dynamic Contact Trajectory Validation

This validation generates C3D8 block-plane and block-block transient contact cases. CalculiX supplies the external trajectory; SFC replays the current deformed C3D8 boundary with dynamic SDF queries for gap, normal-force, contact-energy, and stress-cloud diagnostics.

## Commands

- `wsl --exec bash -lc "cd /mnt/e/workspace/sdf-fem-pro/results/c3d8_contact_trajectory_quick/calculix_runs/block_plane_c3d8_r1 && ccx block_plane_c3d8_r1"`
- `wsl --exec bash -lc "cd /mnt/e/workspace/sdf-fem-pro/results/c3d8_contact_trajectory_quick/calculix_runs/block_block_c3d8_r1 && ccx block_block_c3d8_r1"`

## Summary

| Case | r | Source | Rows | Active rows | Peak force rel. error | Peak CELS rel. error | Gap rel. error | Status |
|---|---:|---|---:|---:|---:|---:|---:|---|
| block_block_c3d8 | 1 | calculix_dat | 70 | 22 | 2.428073e-04 | 2.315966e-04 | 3.475215e-06 | external_replay |
| block_plane_c3d8 | 1 | calculix_dat | 69 | 22 | 1.435099e-04 | 2.403822e-04 | 1.032577e-04 | external_replay |

## Output Files

- `c3d8_contact_trajectory.csv`
- `c3d8_contact_trajectory_summary.csv`
- `c3d8_contact_trajectory_stress_cloud.csv`
- `c3d8_contact_trajectory_commands.csv`
- `c3d8_contact_trajectory_plots.csv`
- `c3d8_contact_trajectory_claims.csv`

## Scope

Supported claim: current-surface dynamic SDF can replay C3D8 deformed contact trajectories exported by CalculiX and compute comparable gap, force, energy, and stress-cloud diagnostics in the tested cases.

Unsupported claim: native SFC nonlinear C3D8 dynamics is exactly equivalent to CalculiX. The runner intentionally keeps CalculiX as the external trajectory source.

## Figures

- `figures\c3d8_contact_trajectory_z_cm.png`: C3D8 CalculiX trajectory center height
- `figures\c3d8_contact_trajectory_gap.png`: SFC dynamic-SDF gap and CalculiX CDIS where available
- `figures\c3d8_contact_trajectory_force_energy.png`: Normal reaction and contact spring energy
- `figures\c3d8_contact_trajectory_stress_cloud.png`: Final C3D8 stress cloud from CalculiX displacement/stress output
