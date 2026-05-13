# Phase-4 Paper Validation Handoff

Status: Phase-4 validation evidence infrastructure is implemented and the
reduced quick-mode evidence run passes. This is still evidence for the current
minimal standalone FEM-SDF solver only; it does not add or validate friction,
self-contact, nonlinear FEM, GPU execution, or barrier contact.

## Commands Run

```text
pytest -q
............................................................             [100%]
60 passed in 2.63s
```

```text
python validation/run_phase3_validation.py --quick --out-dir results/validation
Wrote results\validation\phase3_validation.csv
Wrote results\validation\phase3_validation.md
```

```text
python validation/run_phase4_paper_validation.py --quick --out-dir results/phase4
Wrote results\phase4\phase4_mesh_resolution.csv
Wrote results\phase4\phase4_contact_time_history.csv
Wrote results\phase4\phase4_performance_scaling.csv
Wrote results\phase4\phase4_claims.csv
Wrote results\phase4\phase4_summary.md
```

```text
python benchmarks/run_contact_benchmark.py --quick --out-dir results/benchmarks
Wrote results\benchmarks\contact_benchmark.csv
Wrote results\benchmarks\contact_benchmark.md
```

```text
python benchmarks/compare_bvh_vs_dynamic_sdf.py --quick --out-dir results/benchmarks
Wrote results\benchmarks\contact_method_comparison.csv
Wrote results\benchmarks\contact_method_comparison.md
```

## Output Files

Phase-4 validation:

- `results/phase4/phase4_mesh_resolution.csv`
- `results/phase4/phase4_contact_time_history.csv`
- `results/phase4/phase4_performance_scaling.csv`
- `results/phase4/phase4_claims.csv`
- `results/phase4/phase4_summary.md`

Supporting Phase-3 validation and benchmarks:

- `results/validation/phase3_validation.csv`
- `results/validation/phase3_validation.md`
- `results/benchmarks/contact_benchmark.csv`
- `results/benchmarks/contact_benchmark.md`
- `results/benchmarks/contact_method_comparison.csv`
- `results/benchmarks/contact_method_comparison.md`

Plots: none generated in this run. The benchmark scripts only write plots when
their optional plotting flag is used.

## Mesh-Resolution Metrics

| Case | Resolutions | Metric | Values | Status | Paper Claim Support |
| --- | --- | --- | --- | --- | --- |
| Uniaxial linear elastic patch | 1, 2, 3 | relative energy error | `0.0`, `0.0`, `2.312964634635743e-16` | pass | Supports reporting reduced mesh-resolution energy-error trends only. |
| Fixed-base gravity elastic block | 1, 2, 3 | mean tip z displacement | `-1.3631864126364321e-03`, `-2.2098042076471127e-03`, `-2.8491565879671483e-03` | pass | Supports reporting displacement trend relative to finest quick mesh. |
| Mesh-resolution gate | 3 resolutions | resolution count | `3` | pass | Supports trend reporting; no theoretical convergence order is claimed. |

The uniaxial case uses an affine displacement patch and reports energy error.
The gravity block reports displacement relative to the finest available
quick-mode mesh, with relative-to-finest values `0.521547387604605`,
`0.2244005763039541`, and `0.0`.

## Contact Time-History Metrics

| Case | Steps | Min Gap | Max Penetration | Max Active Contacts | Max Normal Force | Max Contact Energy | Max Imbalance | Status | Paper Claim Support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Elastic block against rigid plane | 6 | `-2.500000e-02` | `2.500000e-02` | 4 | `8.000000e+02` | `1.000000e+01` | `1.009742e-28` | pass | Supports normal penalty-contact time-history evidence for this reduced setup. |
| Deformable-deformable contact | 6 | `-2.000000e-02` | `2.000000e-02` | 2 | `2.400000e+02` | `2.400000e+00` | `0.000000e+00` | pass | Supports action-reaction evidence for the reduced deformable-deformable sample. |

Each row in `phase4_contact_time_history.csv` includes `time`, `min_gap`,
`max_penetration`, `active_contact_count`, `normal_force`, `contact_energy`,
and `action_reaction_imbalance`.

## Performance Scaling Metrics

| Surface Resolution | Triangles | Query Points | Spatial Hash + Projection Speedup | Dynamic FEM-SDF Full Pipeline Speedup | Status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 8 | 4 | `4.4284204723876694` | `1.039681077818867` | pass |
| 4 | 32 | 16 | `23.714083234551598` | `4.912133620037821` | pass |
| 6 | 72 | 36 | `43.33628250014982` | `9.509550485528907` | pass |

These speedups are computed against the quick-mode brute-force all-triangle
projection rows in `phase4_performance_scaling.csv`. Speedup is reported only
where measured `speedup_vs_bruteforce > 1`. Rows without speedup must remain
blank and must not support acceleration claims. The material-space SDF baseline
is explicitly marked `unavailable`.

The isolated timing stages reported are:

- `build_broad_phase`
- `candidate_query_only`
- `local_projection_only`
- `dynamic_sdf_query_only`
- `contact_detection_end_to_end`
- `full_contact_pipeline_end_to_end`

## Evidence Gates

Implemented gate tests in `tests/test_phase4_validation.py` verify:

- quick-mode Phase-4 scripts generate all expected CSV and Markdown outputs;
- uniaxial and gravity studies use at least three resolutions;
- contact history CSV files include all required per-step fields;
- compared performance labels do not reuse the same end-to-end workload;
- Markdown evidence markers reference existing CSV fields;
- convergence/trend support requires at least three resolutions;
- acceleration is not claimed when measured speedup is not greater than one.

## Claim Support

| Claim | Evidence | Status |
| --- | --- | --- |
| Mesh-resolution trends are reported without theoretical order claim. | `phase4_mesh_resolution.csv`, `phase4_claims.csv` | supported for reduced quick-mode evidence |
| Contact time histories report gap, penetration, active contact count, normal force, energy, and action-reaction imbalance. | `phase4_contact_time_history.csv`, `phase4_claims.csv` | supported for reduced quick-mode evidence |
| Acceleration is claimed only where measured speedup is greater than one. | `phase4_performance_scaling.csv`, `phase4_claims.csv` | supported as a reporting gate |
| Material-space SDF baseline comparison exists. | `phase4_performance_scaling.csv` marks it unavailable | not supported |
| Publication-grade convergence order is established. | no supporting evidence file | not supported |

## Remaining Risks

- The Phase-4 quick run is reduced and deterministic. Larger meshes and repeated
  timing runs are still needed before final paper tables.
- TET4 linear elasticity is the only FEM formulation validated here.
- Contact is penalty normal contact only; friction, self-contact, nonlinear FEM,
  GPU execution, and barrier contact remain out of scope.
- The SDF is still an oriented local surface distance induced by current FEM
  surface geometry, not a robust global sign method for arbitrary geometry.
- Performance scaling uses a uniform spatial hash, not a production BVH.
- Timing numbers are hardware- and load-dependent; they should be regenerated
  for paper experiments.

## Final Phase-4 Status

Phase-4 evidence generation passes in quick mode. The current code can support
limited paper-claim preparation for linear TET4 FEM-SDF penalty contact, but it
is not yet final paper-experiment evidence without larger repeated runs and
external or analytic reference comparisons.
