# Phase-6 Final Paper Experiment Handoff

Status: non-quick Phase-5 experiments were run and packaged under
`results/paper_final`. No `src/sfc` core solver files were changed. No core
physics features were added.

## Exact Commands Run

```text
pytest -q
```

Output saved to `results/paper_final/logs/pytest_q.log`:

```text
...................................................................      [100%]
67 passed in 6.15s
```

```text
python validation/run_phase5_paper_experiments.py --out-dir results/paper_final
```

Output saved to `results/paper_final/logs/phase5_nonquick.log`. The command
completed and wrote the Phase-5 CSV, Markdown, PNG, and PDF outputs under
`results/paper_final`.

Packaging output was saved to `results/paper_final/logs/package_outputs.log`.

## Output Layout

Root generated evidence:

- `results/paper_final/phase5_metadata.csv`
- `results/paper_final/phase5_mesh_resolution.csv`
- `results/paper_final/phase5_contact_time_history.csv`
- `results/paper_final/phase5_performance_scaling.csv`
- `results/paper_final/phase5_material_space_sdf_baseline.csv`
- `results/paper_final/phase5_external_reference_validation.csv`
- `results/paper_final/phase5_contact_reference.csv`
- `results/paper_final/phase5_claims.csv`
- `results/paper_final/phase5_plots.csv`
- `results/paper_final/phase5_summary.md`

Final table package:

- `results/paper_final/tables/phase5_mesh_resolution.csv`
- `results/paper_final/tables/phase5_contact_time_history.csv`
- `results/paper_final/tables/phase5_performance_scaling.csv`
- `results/paper_final/tables/phase5_material_space_sdf_baseline.csv`
- `results/paper_final/tables/phase5_external_reference_validation.csv`
- `results/paper_final/tables/phase5_contact_reference.csv`
- `results/paper_final/tables/phase5_claims.csv`
- `results/paper_final/tables/phase5_plots.csv`

Final figure package:

- `results/paper_final/figures/phase5_mesh_resolution_trend.png`
- `results/paper_final/figures/phase5_mesh_resolution_trend.pdf`
- `results/paper_final/figures/phase5_contact_time_history.png`
- `results/paper_final/figures/phase5_contact_time_history.pdf`
- `results/paper_final/figures/phase5_performance_scaling.png`
- `results/paper_final/figures/phase5_performance_scaling.pdf`
- `results/paper_final/figures/phase5_speedup.png`
- `results/paper_final/figures/phase5_speedup.pdf`
- `results/paper_final/figures/phase5_material_space_sdf_error.png`
- `results/paper_final/figures/phase5_material_space_sdf_error.pdf`

Logs and metadata:

- `results/paper_final/logs/pytest_q.log`
- `results/paper_final/logs/phase5_nonquick.log`
- `results/paper_final/logs/package_outputs.log`
- `results/paper_final/metadata/runtime_metadata.csv`
- `results/paper_final/metadata/commands_run.txt`
- `results/paper_final/metadata/git_branch.txt`
- `results/paper_final/metadata/git_commit.txt`
- `results/paper_final/metadata/git_status_short.txt`
- `results/paper_final/metadata/git_diff_stat.txt`
- `results/paper_final/metadata/package_manifest.csv`

## Runtime Metadata

| Key | Value |
| --- | --- |
| run mode | `full` |
| surface sizes | `4,8,12,16` |
| timing repeats | `5` |
| Python | `3.13.5` |
| NumPy | `2.1.3` |
| SciPy | `1.15.3` |
| Matplotlib | `3.10.0` |
| platform | `Windows 11 AMD64` |
| logical CPU count | `24` |
| timer | `time.perf_counter` |
| core physics scope | `linear_tet4_oriented_dynamic_sdf_penalty_contact` |

## Performance Timing

Mean, standard deviation, minimum, and maximum are over five non-quick repeats.

| Surface | Triangles | Method | Mean | Std | Min | Max | Speedup | Status |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 32 | brute_force_all_triangle_projection | `4.045286e-02` | `5.954157e-04` | `3.966840e-02` | `4.104340e-02` |  | not_applicable |
| 4 | 32 | spatial_hash_plus_local_projection | `2.119000e-03` | `1.570954e-04` | `2.028400e-03` | `2.396300e-03` | `1.909054e+01` | speedup_supported |
| 4 | 32 | dynamic_fem_induced_sdf_full_pipeline | `1.130544e-02` | `7.851541e-04` | `1.042840e-02` | `1.210790e-02` | `3.578177e+00` | speedup_supported |
| 8 | 128 | brute_force_all_triangle_projection | `6.215336e-01` | `1.277610e-02` | `6.063181e-01` | `6.389071e-01` |  | not_applicable |
| 8 | 128 | spatial_hash_plus_local_projection | `8.101340e-03` | `2.124966e-04` | `7.951800e-03` | `8.442700e-03` | `7.671986e+01` | speedup_supported |
| 8 | 128 | dynamic_fem_induced_sdf_full_pipeline | `4.146486e-02` | `5.557354e-04` | `4.071130e-02` | `4.211050e-02` | `1.498941e+01` | speedup_supported |
| 12 | 288 | brute_force_all_triangle_projection | `3.052786e+00` | `2.325243e-02` | `3.016994e+00` | `3.073171e+00` |  | not_applicable |
| 12 | 288 | spatial_hash_plus_local_projection | `1.996770e-02` | `2.742989e-03` | `1.849990e-02` | `2.486530e-02` | `1.528862e+02` | speedup_supported |
| 12 | 288 | dynamic_fem_induced_sdf_full_pipeline | `9.397796e-02` | `1.417190e-03` | `9.304040e-02` | `9.640510e-02` | `3.248406e+01` | speedup_supported |
| 16 | 512 | brute_force_all_triangle_projection | `9.631436e+00` | `1.106389e-01` | `9.489238e+00` | `9.731748e+00` |  | not_applicable |
| 16 | 512 | spatial_hash_plus_local_projection | `3.265508e-02` | `5.768641e-04` | `3.177440e-02` | `3.338620e-02` | `2.949445e+02` | speedup_supported |
| 16 | 512 | dynamic_fem_induced_sdf_full_pipeline | `1.687706e-01` | `1.058151e-03` | `1.671798e-01` | `1.699191e-01` | `5.706820e+01` | speedup_supported |

Material-space SDF baseline rows are present as error-comparison data, not as a
timed main-method row.

## Reference And Baseline Metrics

Mesh-resolution evidence:

- Uniaxial energy error over resolutions 1, 2, 3, 4:
  `0.0`, `0.0`, `2.312964634635743e-16`, `3.469446951953614e-16`.
- Fixed-base gravity displacement over resolutions 1, 2, 3, 4:
  `-1.3631864126364321e-03`, `-2.2098042076471127e-03`,
  `-2.8491565879671483e-03`, `-3.262463267437264e-03`.
- Gravity relative-to-finest values:
  `0.58216037...`, `0.32265775...`, `0.12668546...`, `0.0`.

Material-space SDF baseline:

| Deformation | Rows | Max Gap Error | Max Normal Angle Error |
| --- | ---: | ---: | ---: |
| stretch | 20 | `5.217391e-03` | `0.000000e+00` |
| shear | 20 | `2.708013e-03` | `3.580203e-01` |

Analytic linear-elastic reference:

- Resolutions: `1, 2, 3, 4`.
- Maximum relative energy error: `3.469446951953614e-16`.

High-resolution brute-force closest-point projection contact reference:

- Surface resolution: `12`.
- Triangle count: `288`.
- Reference rows: `144`.
- Maximum gap error: `0.0`.
- Maximum normal error: `0.0`.

## Supported Claims

The following claims are supported by `results/paper_final/phase5_claims.csv`:

- `final_acceleration_claim`: supported.
  Gate value: `full,3.5781765155076513,4,5`.
- `convergence_order_claim`: supported.
  Gate value: `3,1.3303529231660702,0.9237213511161891`.
- `material_space_baseline_comparison`: supported.
  Gate value: `baseline_row_count=40`.
- `external_reference_validation`: supported.
  Gate value: `reference_rows,contact_reference_rows=4,144`.

These claims are limited to the current Phase-5 experiment definitions and the
recorded runtime environment.

## Unsupported Claims

The following remain unsupported and should not be stated as paper results:

- Frictional contact.
- Self-contact.
- Nonlinear FEM or large-deformation material behavior.
- GPU acceleration.
- Barrier contact.
- Production BVH performance.
- Robust global SDF sign for arbitrary non-manifold geometry.
- Abaqus-based validation in the core solver.

## Limitations

- The contact broad phase is a uniform spatial hash, not a production BVH.
- The SDF remains an oriented local current-surface distance.
- The material-space SDF baseline is intentionally frozen and validation-only;
  it is not the main method.
- Timings are hardware and load dependent. The metadata in
  `results/paper_final/metadata/runtime_metadata.csv` must accompany any paper
  table derived from these numbers.
- The analytic reference covers small-strain linear elasticity only.
- No `src/sfc` code was changed during Phase 6.
