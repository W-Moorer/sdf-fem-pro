# Contact Query Equivalence Handoff

## Purpose

This validation isolates the project-specific contact query from the mechanics
backend. Both runs use the same clean-room CalculiX-aligned StVK mechanics,
consistent mass, HHT/Newmark Newton stepping, and penalty contact assembly.
Only the contact geometry provider changes:

- `plane`: analytic rigid-plane gap.
- `dynamic_sdf_plane`: the same plane represented by two oriented master
  triangles and queried through the dynamic FEM-induced SDF API.

This is the controlled test needed before interpreting SDF contact differences
against an external solver.

## Command

```bash
python validation/run_contact_query_equivalence.py --quick --out-dir results/contact_query_equivalence
```

## Output Files

- `results/contact_query_equivalence/contact_query_equivalence_history.csv`
- `results/contact_query_equivalence/contact_query_equivalence_metrics.csv`
- `results/contact_query_equivalence/contact_query_equivalence_claims.csv`
- `results/contact_query_equivalence/contact_query_equivalence_summary.md`

## Latest Quick Result

| Resolution | activation abs. error | max field abs. error | max field rel. error | status |
| ---: | ---: | ---: | ---: | --- |
| 1 | `0.000000e+00` | `3.659295e-13` | `1.009385e-13` | `passed` |

The claim gate
`same_backend_dynamic_sdf_query_matches_analytic_plane` is supported in the
quick output.

## Latest Full r1/r2/r3 Result

Command:

```bash
python validation/run_contact_query_equivalence.py --out-dir results/contact_query_equivalence_full
```

| Resolution | activation abs. error | max field abs. error | max field rel. error | status |
| ---: | ---: | ---: | ---: | --- |
| 1 | `0.000000e+00` | `1.403322e-12` | `1.243158e-13` | `passed` |
| 2 | `0.000000e+00` | `4.050094e-13` | `5.581199e-14` | `passed` |
| 3 | `0.000000e+00` | `2.629008e-13` | `1.362188e-13` | `passed` |

## Interpretation

For a rigid plane represented exactly by master triangles, the dynamic-SDF
contact query reproduces the analytic plane-contact history to roundoff while
using the same mechanics backend. This means later differences in external
contact comparisons should not be attributed to merely replacing the analytic
plane gap with the dynamic SDF query in this controlled geometry.

## Limitations

- The master surface is a flat rigid plane, not a curved or deforming body.
- The result validates the contact-query substitution only for this controlled
  geometry.
- It does not claim equivalence to CalculiX surface-to-surface active contact
  generation.

The next deforming-master reference step is documented in
`docs/deforming_master_sdf_reference_handoff.md`.
