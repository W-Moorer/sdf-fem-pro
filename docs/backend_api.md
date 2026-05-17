# Dynamic SDF Field Backend API

This package is a research backend for dynamic SDF field contact queries. It is not a general-purpose production FEM contact solver.

## Core Field Construction

```python
from sfc.sdf import DynamicNarrowBandSDF

field = DynamicNarrowBandSDF.build(
    x_current,
    boundary_faces,
    spacing=0.05,
    band_radius=0.25,
)
```

Inputs:

- `x_current`: current FEM nodal positions with shape `(n_nodes, 3)`.
- `boundary_faces`: oriented triangular boundary faces with shape `(n_faces, 3)`.
- `spacing`: scalar or length-3 grid spacing.
- `band_radius`: current-space narrow-band radius.

Projection is used only inside `build(...)` to populate grid-node `phi` values and closest-feature payloads.

## Query API

```python
phi = field.query_phi(x)
grad = field.query_spatial_derivative_phi(x)
grad_alias = field.query_gradient(x)
gap, normal = field.query_gap_normal(x)
payload = field.query_payload(x)
geometric_normal = field.query_geometric_normal(x)
```

Semantics:

- `query_phi(x)` returns trilinear interpolation of the scalar `phi` grid.
- `query_spatial_derivative_phi(x)` returns `sum_l Phi_l grad_x w_l(x)`, the derivative of the scalar interpolation.
- `query_gradient(x)` is an alias for the scalar interpolation derivative.
- `query_gap_normal(x)` returns `(phi, normalize(query_spatial_derivative_phi(x)))`.
- `query_payload(x)` returns interpolated closest face ids, barycentric coordinates, and closest normals for master-side FEM sensitivity.
- `query_geometric_normal(x)` returns an interpolated closest-feature normal estimate for diagnostics/refinement; it is not the scalar gap derivative.

Out-of-band queries raise `ValueError`; they do not silently fall back to projection.

## Contact API

```python
from sfc.contact import (
    field_contact_constraint_from_sample,
    assemble_field_contact_jacobian,
    field_penalty_contact_response,
)

constraint = field_contact_constraint_from_sample(slave_x_current, sample, field)
J = assemble_field_contact_jacobian([constraint], n_total_dofs=n_dofs)
force, K = field_penalty_contact_response([constraint], stiffness=k, n_total_dofs=n_dofs)
```

The contact gap is positive for separation and negative for penetration:

```text
g = query_phi(x_slave)
```

The slave Jacobian uses the scalar-field derivative:

```text
dg/dx_slave = N_slave * grad_x phi_hat(x_slave)^T
```

The master Jacobian uses closest-feature grid payloads:

```text
dg/dx_master = - sum_l w_l N_l n_l^T
```

The implemented penalty law is frictionless normal penalty contact:

```text
E = 0.5 * k * max(-g, 0)^2
lambda = k * max(-g, 0)
f = J^T lambda
```

## Current Batch Status

The public API is point-wise. Batch wrappers are recommended for downstream users:

```python
field.query_phi_batch(points)
field.query_gap_normal_batch(points)
field.query_payload_batch(points)
```

These wrappers are not yet promoted as stable API in this phase.

## Visualization and Exchange

`validation/run_external_visual_validation.py` exports diagnostic VTU/CSV/PNG/PDF files for:

- displacement magnitude;
- von Mises stress;
- strain norm;
- contact pressure/gap;
- absolute error fields;
- SDF phi slice;
- narrow-band valid mask;
- contact samples colored by interpolated gap;
- scalar-gradient normals.

These outputs are for validation and paper figures, not a core runtime dependency.
