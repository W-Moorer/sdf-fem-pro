# SDF Plane Equivalence Diagnostic

Date: 2026-05-16

This note records the block-drop diagnostic that compares the analytic rigid
plane contact query against the dynamic FEM-induced SDF query at identical SFC
states.

## Command

```bash
python validation/run_sdf_plane_equivalence_diagnostic.py --duration 0.08 --dt 0.001 --resolution 1 --initial-velocity-z -2 --stride 1 --out-dir results/sdf_plane_equivalence_block_drop
```

## Outputs

- `results/sdf_plane_equivalence_block_drop/sdf_plane_stateless_query_equivalence.csv`
- `results/sdf_plane_equivalence_block_drop/sdf_plane_persistent_lifecycle_equivalence.csv`
- `results/sdf_plane_equivalence_block_drop/sdf_plane_equivalence_summary.csv`
- `results/sdf_plane_equivalence_block_drop/sdf_plane_equivalence_summary.md`

## Result

| Comparison | States | Status | Max gap error | Max normal error | Max projection error | Max force error | Active mismatches | Missing SDF samples |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stateless_query | 81 | pass | 6.938894e-18 | 5.806889e-12 | 7.794571e-16 | 1.776357e-15 | 0 | 0 |
| persistent_lifecycle | 81 | pass | 7.092417e-15 | 9.971568e-13 | 3.974003e-16 | 5.983991e-12 | 0 | 0 |

A direct SFC-only trajectory comparison over the same 0.08 s window also
matched to floating-point tolerance:

| Metric | Max absolute difference |
| --- | ---: |
| z_cm | 1.165734e-15 |
| v_cm_z | 7.749357e-14 |
| max_penetration | 2.360308e-15 |
| normal_force_proxy | 1.216804e-11 |
| contact_energy_proxy | 4.524159e-14 |
| total_mechanical_energy_proxy | 2.398082e-14 |
| active_contact_count | 0 |
| generated_contact_spring_count | 0 |

## Interpretation

For this block-plane trajectory, the triangulated dynamic-SDF plane query is
sample-equivalent to the analytic plane query. The SDF master surface has enough
coverage for the tested slave quadrature points, candidate search does not miss
samples, normals match the plane normal, and the persistent spring lifecycle
produces the same active set.

Therefore, if `persistent_dynamic_sdf_calculix_f2f` still diverges from
`persistent_calculix_c3d4_f2f` in a full trajectory, the likely source is not
SDF resolution or basic closest-point geometry. The next checks should focus on
stateful lifecycle call ordering inside Newton, accepted-step history updates,
or downstream mechanics/state update differences after the query layer.

For the tested block-plane case and 0.08 s time window, the full SFC trajectory
does not diverge after replacing the analytic plane query with the dynamic-SDF
query.
