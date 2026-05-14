# Clean-Room CalculiX-Aligned Backend Handoff

## Scope

This update starts the requested separation between mechanics and contact
querying. The new backend in `src/sfc/fem/calculix_aligned.py` is a clean-room
implementation using public finite-element formulas. It does not copy CalculiX
source code and does not make CalculiX a runtime dependency.

The intended comparison boundary is:

- Mechanics: align modeling choices with the CalculiX validation setup where
  practical.
- Contact geometry: remain swappable so the project method can use the SFC
  dynamic FEM-induced SDF for gap and normal queries.

## Implemented Mechanics Alignment

- Positive-orientation TET4 reference data.
- Consistent C3D4/TET4 mass matrix.
- Total-Lagrangian St. Venant-Kirchhoff internal force.
- Material tangent and geometric tangent.
- HHT/Newmark implicit step with Newton iterations.
- Gravity loads assembled internally.
- Area-weighted triangle quadrature for slave-surface penalty contact.

## SDF Contact Geometry Boundary

`src/sfc/contact/sdf_geometry.py` adds
`DynamicSurfaceSDFContactGeometry`. It converts slave-face quadrature points
into contact samples by querying the current master surface with
`dynamic_surface_sdf`.

The adapter requires a `candidate_provider` supplied by the broad phase and
does not fall back to global all-face projection. This preserves the method
boundary: acceleration still depends on candidate search plus local projection,
not brute-force hidden inside the production contact query.

## Validation Evidence

Focused tests:

```bash
pytest -q tests/test_dynamic_sdf_contact_geometry.py tests/test_calculix_aligned_backend.py tests/test_dynamic_surface_sdf.py
```

Latest focused result:

```text
15 passed
```

The geometric nonlinear contact validation runner now uses the aligned backend.
The latest full external comparison is under
`results/geometric_nonlinear_contact_validation_full/`.

The controlled contact-query equivalence run is documented in
`docs/contact_query_equivalence_handoff.md`. In quick mode, the analytic plane
provider and dynamic-SDF plane provider match to roundoff with maximum field
absolute error `3.659295e-13`.

The prescribed deforming-master SDF reference run is documented in
`docs/deforming_master_sdf_reference_handoff.md`. In the latest full run,
spatial-hash dynamic SDF matched all-face projection with zero reported gap and
force errors and maximum normal angle error `2.980232e-08`.

## Current Full Contact Comparison

| Resolution | activation C/SFC | max pen. C | max pen. SFC | peak force C | peak force SFC | status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `0.096 / 0.096` | `3.144397e-03` | `3.387373e-03` | `4.118874e+01` | `3.981581e+01` | `passed_scoped_gate` |
| 2 | `0.096 / 0.096` | `2.743927e-03` | `3.255793e-03` | `3.077931e+01` | `3.497837e+01` | `passed_scoped_gate` |
| 3 | `0.096 / 0.096` | `2.690467e-03` | `3.021339e-03` | `2.922631e+01` | `3.451957e+01` | `passed_scoped_gate` |

Compared with the prior lumped-mass diagnostic path, the r1 SFC maximum
penetration dropped from about `4.49e-03` to `3.39e-03`, and the r1 peak normal
force moved from about `50.93` to `39.82`, much closer to the CalculiX values.

## Remaining Non-Equivalences

- This is not a source-level clone of CalculiX.
- The contact discretization is still SFC pressure-overclosure quadrature, not
  CalculiX persistent surface-to-surface active contact generation.
- The current external alignment runner uses rigid-plane contact; SDF contact
  geometry is implemented as a swappable adapter and should be used in the next
  controlled SDF-vs-plane/contact experiment.
- Exact equality of max penetration is not expected until contact enforcement,
  active set generation, stabilization, and output metric definitions are also
  matched.
