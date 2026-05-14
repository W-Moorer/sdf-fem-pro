# Geometric Nonlinear VTK Handoff

## Purpose

This handoff adds a validation-only geometric nonlinear TET4 visualization
runner. It is intended to inspect stress/strain evolution in ParaView and to
separate center-of-mass motion from internal deformation modes.

This does not replace the current core linear SFC solver. The implementation is
an explicit total-Lagrangian diagnostic exporter.

## Implemented Model

- TET4 total-Lagrangian kinematics.
- St. Venant-Kirchhoff material.
- Green-Lagrange strain per element.
- Cauchy stress per element.
- von Mises stress per element.
- Conservative rigid-plane penalty contact.
- Velocity-Verlet time integration for an undamped diagnostic run.

## Command

```bash
python validation/run_geometric_nonlinear_vtk.py --resolution 2 --duration 0.5 --dt 0.00025 --frame-stride 20 --out-dir results/geometric_nonlinear_vtk
```

## Outputs

- `results/geometric_nonlinear_vtk/vtk/frame_0000.vtk`
- `results/geometric_nonlinear_vtk/vtk/frame_0001.vtk`
- ...
- `results/geometric_nonlinear_vtk/vtk/frame_0100.vtk`
- `results/geometric_nonlinear_vtk/geometric_nonlinear_history.csv`
- `results/geometric_nonlinear_vtk/geometric_nonlinear_frames.csv`
- `results/geometric_nonlinear_vtk/geometric_nonlinear_summary.md`

The VTK frame suffixes are continuous and zero padded. ParaView should load
them as a time series when opening `frame_0000.vtk`.

## VTK Fields

Point data:

- `displacement`
- `velocity`

Cell data:

- `von_mises`
- `green_lagrange_strain_norm`
- `green_lagrange_strain`
- `cauchy_stress`

## Latest Run Metrics

| Metric | Value |
| --- | ---: |
| frames written | 101 |
| first frame | `frame_0000.vtk` |
| last frame | `frame_0100.vtk` |
| history rows | 2001 |
| minimum gap | `-8.458748e-03` |
| maximum penetration | `8.458748e-03` |
| maximum von Mises stress | `4.358995e+01` |
| relative energy drift range | `-4.476440e-06` to `1.306697e-03` |

## Interpretation

The nonlinear VTK output should be used to inspect whether contact drives
internal deformation modes. If center-of-mass peaks decrease while total energy
is nearly conserved, the missing height is not necessarily numerical damping:
it can be stored in Green-Lagrange strain energy and internal vibration.

## Limitations

- This is not yet a production nonlinear implicit SFC solver.
- The contact target is a rigid horizontal plane only.
- The material is St. Venant-Kirchhoff, which is a simple geometric nonlinear
  diagnostic model, not a robust large-strain hyperelastic material for all
  deformations.
- No friction, self-contact, barrier contact, or GPU acceleration is included.
