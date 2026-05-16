# C3D8 Dynamic Contact Trajectory Validation

This locked numerical experiment contains the quick external CalculiX C3D8
trajectory replay evidence used by the manuscript draft.

## Reproduce

```bash
python validation/run_c3d8_contact_trajectory_validation.py --quick --out-dir results/c3d8_contact_trajectory_quick
```

The runner generates two transient dynamic C3D8 cases:

- `block_plane_c3d8`: elastic C3D8 block against a fixed rigid plane.
- `block_block_c3d8`: elastic C3D8 slave block against a fixed C3D8 master block.

CalculiX solves the trajectory. SFC replays each exported deformed C3D8
configuration with the current-surface dynamic SDF query and compares gap,
reaction force, contact energy, and stress-cloud diagnostics.

## Claim Boundary

Supported:

- The dynamic SDF query can consume current triangulated C3D8 boundary surfaces.
- On the generated quick cases, SFC replay of CalculiX-deformed C3D8 states
  matches CalculiX `CDIS`, fixed support `RF`, and `CELS` outputs to small
  relative error.

Unsupported:

- Native SFC nonlinear C3D8 dynamic trajectory equivalence.
- Friction, self-contact, barrier contact, or production BVH performance.
