# Commercial Benchmark 1: Linear-Elastic Sphere Drop

This locked paper artifact stores the first commercial-reference comparison:
a flexible C3D4 sphere dropped onto a rigid plane using a no-damping,
frictionless, linear-penalty Abaqus/Explicit reference and an independent SFC
Lagrangian-SDF solve.

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
| Young's modulus | `5.0e7` |
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
| Abaqus analysis wall time | `54.84324390022084 s` |
| Abaqus total wall time with VTK/energy extraction | `109.08434669999406 s` |
| SFC solve wall time | `28.662549200002104 s` |
| SFC speedup vs Abaqus analysis | `1.91x` |
| SFC speedup vs Abaqus total | `3.81x` |
| max center-z error | `2.620857548957384e-4 m` |
| RMS center-z error | `9.513284595429342e-5 m` |
| max min-gap replay error | `8.251592139528283e-4 m` |
| Abaqus max P95 strain norm | `9.572005184641033e-3` |
| SFC max P95 strain norm | `5.001139661054739e-3` |

## Files

- `abaqus_case_metrics.csv`: Abaqus runtime and model metadata.
- `abaqus_input.inp`: locked commercial-reference model input deck.
- `sfc_vs_abaqus_metrics.csv`: SFC/Abaqus error and timing metrics.
- `sfc_vs_abaqus_comparison.csv`: frame-by-frame comparison.
- `sfc_vs_abaqus_z_cm.png/pdf`: displacement trajectory.
- `sfc_vs_abaqus_p95_von_mises.png/pdf`: P95 stress history.

## Claim Boundary

This case supports a commercial-reference trajectory and efficiency comparison
for a scoped frictionless normal-contact setting. It does not claim source-level
Abaqus equivalence, friction, self-contact, or general nonlinear-material
coverage.
