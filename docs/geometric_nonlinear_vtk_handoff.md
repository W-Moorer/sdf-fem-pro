# Geometric Nonlinear VTK Handoff

## Purpose

This handoff adds a validation-only geometric nonlinear TET4 visualization
runner. It is intended to inspect stress/strain evolution in ParaView and to
separate center-of-mass motion from internal deformation modes. The default
case is a sphere-like deformable body dropped above a visible rigid plane, so
the initial separation is directly visible in `frame_0000.vtk`.

This does not replace the current core linear SFC solver. The implementation is
an explicit total-Lagrangian diagnostic exporter.

## Implemented Model

- TET4 total-Lagrangian kinematics.
- St. Venant-Kirchhoff material.
- Green-Lagrange strain per element.
- Cauchy stress per element.
- von Mises stress per element.
- Conservative rigid-plane penalty contact.
- CalculiX-style HHT/Newmark parameterization with Newton iterations for the
  implicit diagnostic path.
- Analytic material tangent and stress/geometric tangent for StVK TET4.
- Rigid-plane penalty contact tangent for the implicit residual.
- Velocity-Verlet remains available only as a cheaper visualization path.
- Mixed-cell VTK output containing the deformable TET4 body and a rigid plane
  quad. The deformable body has `object_id=1`; the plane has `object_id=0`.

## Command

```bash
python validation/run_geometric_nonlinear_vtk.py --case sphere_drop --integrator implicit_hht --hht-alpha 0 --resolution 1 --duration 0.2 --dt 0.001 --frame-stride 5 --initial-gap 0.08 --out-dir results/drop_sphere_geometric_nonlinear_vtk_implicit
```

## Outputs

- `results/drop_sphere_geometric_nonlinear_vtk_implicit/vtk/frame_0000.vtk`
- `results/drop_sphere_geometric_nonlinear_vtk_implicit/vtk/frame_0001.vtk`
- ...
- `results/drop_sphere_geometric_nonlinear_vtk_implicit/vtk/frame_0040.vtk`
- `results/drop_sphere_geometric_nonlinear_vtk_implicit/geometric_nonlinear_history.csv`
- `results/drop_sphere_geometric_nonlinear_vtk_implicit/geometric_nonlinear_frames.csv`
- `results/drop_sphere_geometric_nonlinear_vtk_implicit/geometric_nonlinear_summary.md`

The VTK frame suffixes are continuous and zero padded. ParaView should load
them as a time series when opening `frame_0000.vtk`.

## VTK Fields

Point data:

- `displacement`
- `velocity`

Cell data:

- `object_id`
- `von_mises`
- `green_lagrange_strain_norm`
- `green_lagrange_strain`
- `cauchy_stress`

## Latest Run Metrics

| Metric | Value |
| --- | ---: |
| integrator | `implicit_hht` |
| HHT alpha | `0` |
| frames written | 41 |
| first frame | `frame_0000.vtk` |
| last frame | `frame_0040.vtk` |
| history rows | 201 |
| initial visible gap | `8.000000e-02` |
| initial contact-quadrature gap | `8.520833e-02` |
| minimum gap | `-3.449636e-02` |
| maximum penetration | `3.449636e-02` |
| maximum von Mises stress | `6.990186e+01` |
| maximum Newton iterations | `3` |
| maximum Newton residual norm | `3.746808e-07` |
| relative energy drift range | `-7.633523e-04` to `6.602855e-16` |

## Additional Output

The cheaper explicit visualization path was regenerated with continuous frames:

- `results/drop_sphere_geometric_nonlinear_vtk/vtk/frame_0000.vtk` to
  `frame_0100.vtk`.
- `results/block_drop_geometric_nonlinear_vtk/vtk/frame_0000.vtk` to
  `frame_0100.vtk`.

The implicit block-drop diagnostic is available at:

- `results/block_drop_geometric_nonlinear_vtk_implicit/vtk/frame_0000.vtk` to
  `frame_0040.vtk`.

## Interpretation

The nonlinear VTK output should be used to inspect whether contact drives
internal deformation modes. If center-of-mass peaks decrease while total energy
is nearly conserved, the missing height is not necessarily numerical damping:
it can be stored in Green-Lagrange strain energy and internal vibration.

## Limitations

- This is not yet a production nonlinear implicit SFC solver.
- This is CalculiX-aligned at the method-structure level only: HHT/Newmark
  parameters, Newton iterations, material/geometric tangent, and contact
  tangent. It is not a source-level clone of CalculiX nonlinear contact.
- The contact target is a rigid horizontal plane only.
- The sphere is a deterministic sphere-like Delaunay TET4 mesh for visualization
  and diagnostics, not a high-quality production meshing workflow.
- The material is St. Venant-Kirchhoff, which is a simple geometric nonlinear
  diagnostic model, not a robust large-strain hyperelastic material for all
  deformations.
- No friction, self-contact, barrier contact, or GPU acceleration is included.
