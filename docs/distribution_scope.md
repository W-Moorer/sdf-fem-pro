# Distribution Scope

## Package Identity

This project should be distributed as:

```text
SFC: a standalone dynamic SDF field backend for FEM contact research
```

It should not be described as:

```text
a universal production FEM contact solver
```

## Supported Scope

The supported research-backend scope is:

- construct a current-space dynamic narrow-band SDF field from a FEM boundary surface;
- query `phi` by interpolation;
- query the scalar derivative of the interpolated `phi` field;
- store closest-feature payloads per grid node;
- assemble frictionless normal penalty contact constraints;
- provide slave and master contact Jacobians;
- reject out-of-band field queries explicitly;
- report field accuracy, Eikonal residual, Jacobian finite-difference checks, and amortized crossover `Q*`;
- export validation CSV/VTU/PNG/PDF artifacts.

## Unsupported Scope

The current package does not claim support for:

- friction;
- self-contact;
- nonlinear FEM as the main method;
- GPU acceleration;
- barrier contact;
- POD, gappy POD, or neural SDF training;
- robust global SDF repair for arbitrary non-manifold geometry;
- production BVH superiority;
- source-level CalculiX equivalence;
- Abaqus-dependent core solving.

## External Solvers

External solver data, including CalculiX files, may be used only as optional validation or replay references. They are not required by:

- matrix assembly;
- force assembly;
- time integration;
- SDF field construction;
- contact query;
- contact Jacobian assembly.

Any future external solver importer should remain optional and outside the core runtime dependency set.

## Claim Gates

Paper and package claims should remain gated by evidence:

- True SDF field claim requires an actual `phi` grid and interpolation-only query path.
- Field accuracy claim requires phi error, normal error, scalar-gradient finite-difference mismatch, and Eikonal residual below stated thresholds.
- Contact Jacobian claim requires slave and master finite-difference tests.
- Acceleration claim requires:

```text
T_update + Q*T_field_query + Q_refine*T_refine < Q*T_projection_query
```

- External visual validation supports physical plausibility for the tested case only; it does not prove SDF acceleration or commercial-solver equivalence.

## Recommended Optional Interfaces

Future distribution work can add optional import/export extras:

- `sfc[mesh]` for Gmsh/VTK/OBJ/STL import/export;
- `sfc[visual]` for PyVista or richer plotting;
- `sfc[calculix]` for optional CalculiX `.inp`/`.frd` replay importers.

These extras should not become core dependencies.
