# Phase-12 Exact Required-Node SDF Update Handoff

## Goal

Reduce dynamic SDF update cost without lowering contact accuracy.

The implemented strategy is exact sparse field population:

```text
contact samples
-> collect every query point's 8 trilinear grid corners
-> populate only that required grid-node union
-> query by interpolation only
```

Every populated grid node still uses the same closest-feature projection kernel as the full-grid builder, so values used by contact interpolation are not approximated.

## Implementation

- `DynamicNarrowBandSDF.build_required_points(...)`
  - added in `src/sfc/sdf/dynamic_narrow_band_sdf.py`
  - builds the same grid layout as the full builder when given the same `origin`, `shape`, and `spacing`
  - fills only the exact interpolation corners touched by `query_points`
  - leaves unused nodes invalid
  - records `metadata["population_mode"] = "required_points"`
  - records `metadata["required_node_count"]`

- `validation/run_true_field_solver_timing.py`
  - now builds fields with `build_required_points(...)`
  - records `field_population_mode=required_points_exact`
  - records required/valid/full grid-node counts

## Accuracy Gate

Added test:

```text
tests/test_dynamic_narrow_band_sdf.py::test_required_point_sparse_field_matches_full_field_queries
```

It verifies, on a nonplanar surface, that sparse required-node fields match full-grid fields for:

- `query_phi`
- scalar derivative `query_gradient`
- `query_gap_normal`
- closest-feature payload face nodes
- barycentric coordinates
- closest normals
- interpolation-only query path after build

## Timing Improvement

Default-scale timing command:

```text
python validation/run_true_field_solver_timing.py --out-dir results/true_field_solver_timing_sparse
```

Results:

- accuracy rows passed: `16/16`
- total-step acceleration rows passed: `12/16`
- best total-step speedup, projection/field: `6.466150`
- minimum measured query crossover \(Q^\ast\): `849.274`

Previous full-grid timing, from `results/true_field_solver_timing`, had:

- field update: `55.436 s`
- best total-step speedup: `1.355541`
- minimum measured \(Q^\ast\): about `5567`

Exact required-node update now has:

- field update: `8.283-9.637 s`
- best total-step speedup: `6.466150`
- minimum measured \(Q^\ast\): `849.274`

High-query comparison:

| Case | Full-grid speedup | Required-node speedup | Full-grid Q* | Required-node Q* |
| --- | ---: | ---: | ---: | ---: |
| `static_linear_tet4_q8192` | 1.309 | 6.466 | 6190.8 | 999.6 |
| `dynamic_linear_tet4_q8192` | 1.288 | 6.451 | 6301.4 | 1004.1 |
| `static_linear_hex8_q8192` | 1.356 | 6.404 | 5977.6 | 1009.8 |
| `dynamic_linear_hex8_q8192` | 1.293 | 6.437 | 6273.4 | 1002.4 |

Accuracy remains at roundoff scale:

- max gap absolute error: `3.81639164714898e-17`
- max force L2 relative error: `1.72652557265331e-15`

## Tests

Targeted:

```text
pytest -q tests/test_dynamic_narrow_band_sdf.py::test_required_point_sparse_field_matches_full_field_queries tests/test_true_field_solver_timing.py
```

Result:

```text
3 passed in 65.61s (0:01:05)
```

Full suite:

```text
pytest -q
```

Result:

```text
267 passed in 561.76s (0:09:21)
```

## Claim Boundary

Supported:

- Exact sparse SDF update for contact-touched interpolation cells.
- No query-time projection fallback.
- Improved solver-level contact-dominated acceleration without reducing contact accuracy in the tested TET4/HEX8 linear static/dynamic cases.

Not supported:

- Approximate coarse SDF contact values.
- Nonlinear FEM as a main-method claim.
- Friction, self-contact, GPU, barrier contact, POD, neural SDF, or production BVH superiority.
