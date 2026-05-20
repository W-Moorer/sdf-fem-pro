# Commercial Benchmark 2: Soft Sphere Drop Large-Deformation Case

This locked paper artifact stores the soft-material commercial-reference
comparison for a flexible C3D4 sphere dropped onto a rigid plane.

## Scope

- Commercial reference: Abaqus/Explicit 2024.
- SFC path: internally assembled TET4 FEM with Lagrangian-SDF rigid-plane
  contact and Abaqus-style lumped-mass central-difference integration.
- Contact: frictionless linear pressure-overclosure penalty.
- Damping: no bulk viscosity and zero Abaqus contact damping fraction.
- Core SFC does not read Abaqus ODB files or Abaqus-generated matrices.

## Parameters

| Quantity | Value |
| --- | ---: |
| Young's modulus | `5.0e6` |
| Poisson's ratio | `0.30` |
| Density | `1200` |
| Gravity | `9.81` |
| Contact penalty stiffness | `5.0e9` |
| Time step | `5.0e-6 s` |
| Total time | `1.0 s` |
| Output interval | `0.001 s` |

## Results

| Metric | Value |
| --- | ---: |
| Abaqus analysis wall time | `43.39265190018341 s` |
| Abaqus total wall time with VTK/energy extraction | `95.75483019975945 s` |
| SFC solve wall time | `23.48670110013336 s` |
| SFC speedup vs Abaqus analysis | `1.85x` |
| SFC speedup vs Abaqus total | `4.08x` |
| max center-z error | `6.664331569776957e-4 m` |
| RMS center-z error | `1.9611240341594274e-4 m` |
| max min-gap replay error | `1.228723401044542e-3 m` |
| Abaqus max P95 strain norm | `7.705068004982424e-2` |
| SFC max P95 strain norm | `2.2777315398360067e-2` |

## Files

- `abaqus_case_metrics.csv`: Abaqus runtime and model metadata.
- `abaqus_input.inp`: locked commercial-reference model input deck.
- `sfc_vs_abaqus_metrics.csv`: SFC/Abaqus error and timing metrics.
- `sfc_vs_abaqus_comparison.csv`: frame-by-frame comparison.
- `sfc_vs_abaqus_z_cm.png/pdf`: displacement trajectory.
- `sfc_vs_abaqus_p95_strain.png/pdf`: P95 strain history.
- `sfc_vs_abaqus_p95_von_mises.png/pdf`: P95 stress history.

## Claim Boundary

This case supports a large-deformation commercial-reference comparison for the
scoped frictionless normal-contact setting. It does not claim source-level
Abaqus equivalence, friction, self-contact, or general nonlinear-material
coverage.
