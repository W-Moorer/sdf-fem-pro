# Phase-11 Solver-Level Timing Handoff

## Scope

This phase adds solver-level, contact-dominated timing cases for the true dynamic narrow-band SDF field path. The goal is to support the paper claim that FEM-SDF is faster in contact-dominated workloads after the measured crossover query count, while preserving the existing method boundary.

The tested method path is now:

```text
DynamicNarrowBandSDF.build_required_points(...)
-> compute_field_contact_constraints(...)
-> force assembly
-> static solve or dynamic explicit update
```

`build_required_points(...)` is exact for the queried contact points: it populates the union of the eight trilinear grid corners touched by every slave query point, and every populated grid node still uses the same closest-feature projection kernel as the full-grid builder. The projection path is retained only as a spatial-hash projection timing reference.

## New Files

- `validation/run_true_field_solver_timing.py`
- `tests/test_true_field_solver_timing.py`
- `docs/phase11_solver_level_timing_handoff.md`

## Command

```text
python validation/run_true_field_solver_timing.py --out-dir results/true_field_solver_timing_sparse
```

Generated files:

- `results/true_field_solver_timing_sparse/solver_step_timing.csv`
- `results/true_field_solver_timing_sparse/solver_crossover.csv`
- `results/true_field_solver_timing_sparse/true_field_solver_timing_summary.md`

Test command:

```text
pytest -q tests/test_true_field_solver_timing.py
```

Result:

```text
2 passed in 60.24s
```

Full suite:

```text
pytest -q
```

Result:

```text
267 passed in 561.76s (0:09:21)
```

## Default-Scale Results

Summary:

- accuracy rows passed: `16/16`
- total-step acceleration rows passed: `12/16`
- best total-step speedup, projection/field: `6.466150`
- minimum measured query crossover \(Q^\ast\): `849.274`

Compared with the previous full-grid field update, the default-scale exact required-node update reduces the measured field update from `55.436 s` to about `8.283-9.637 s`, and lowers the \(Q^\ast\) range from about `5978-6301` to about `849-1026`.

The accelerated high-query rows improve substantially:

| Case | Full-grid speedup | Required-node speedup | Required-node field step (s) | Projection step (s) |
| --- | ---: | ---: | ---: | ---: |
| `static_linear_tet4_q8192` | 1.309 | 6.466 | 12.502 | 80.838 |
| `dynamic_linear_tet4_q8192` | 1.288 | 6.451 | 12.424 | 80.149 |
| `static_linear_hex8_q8192` | 1.356 | 6.404 | 12.504 | 80.077 |
| `dynamic_linear_hex8_q8192` | 1.293 | 6.437 | 12.385 | 79.722 |

Accuracy remained at machine precision across all rows:

- max gap absolute error: below `4e-17`
- force L2 relative error: below `2e-15`
- state norm absolute error: roundoff scale

## Claim Support

Supported:

- Native linear TET4 static contact accuracy with true-field contact.
- Native linear TET4 dynamic contact accuracy with true-field contact.
- Native linear HEX8 static contact accuracy with true-field contact.
- Native linear HEX8 dynamic contact accuracy with true-field contact.
- Solver-level contact-dominated acceleration after measured \(Q^\ast\).

Still not supported:

- Nonlinear FEM as a main-method claim.
- Friction.
- Self-contact.
- GPU acceleration.
- Barrier contact.
- Neural SDF, POD, or data-driven SDF.
- Production BVH superiority.
- Abaqus-dependent core solve.

## Paper Wording

The safe paper claim is:

```text
In contact-dominated linear TET4/HEX8 workloads, the dynamic SDF field reduces total contact-step time after the measured crossover query count while preserving gap, force, and step-state accuracy against the spatial-hash projection timing reference.
```

Do not write:

```text
The method is always faster.
The paper proves nonlinear FEM contact accuracy.
The implementation is a production nonlinear contact solver.
```
