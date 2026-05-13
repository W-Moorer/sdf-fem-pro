# Phase-3 Validation Handoff

Status: Phase-3 validation infrastructure is implemented for deterministic
physical-correctness, convergence-status, and timing smoke evidence. The solver
remains a minimal standalone FEM-SDF prototype and is not paper-experiment
ready.

## Commands

```bash
python validation/run_phase3_validation.py --quick --out-dir results/validation
pytest -q
python benchmarks/run_contact_benchmark.py --quick --out-dir results/benchmarks
python benchmarks/compare_bvh_vs_dynamic_sdf.py --quick --out-dir results/benchmarks
```

## Outputs

Phase-3 validation:

- `results/validation/phase3_validation.csv`
- `results/validation/phase3_validation.md`

Benchmarks:

- `results/benchmarks/contact_benchmark.csv`
- `results/benchmarks/contact_benchmark.md`
- `results/benchmarks/contact_method_comparison.csv`
- `results/benchmarks/contact_method_comparison.md`

## Validation Metrics

Latest quick validation metrics:

| Case | Metric | Value | Status |
| --- | --- | ---: | --- |
| `uniaxial_linear_elastic_patch` | `relative_energy_error` | `0.000000e+00` | ok |
| `gravity_fixed_base_elastic_block` | `mean_tip_z_displacement` | `-1.363186e-03` | ok |
| `undamped_free_vibration_energy` | `relative_energy_drift` | `5.421011e-15` | ok |
| `deformable_block_against_rigid_plane` | `signed_gap` | `-2.000000e-02` | ok |
| `deformable_deformable_contact_action_reaction` | `net_force_imbalance` | `2.524355e-29` | ok |
| `gravity_fixed_base_elastic_block` | `convergence_rate` | unavailable | unavailable |
| `contact_pipeline_speedup` | `brute_force_over_dynamic_full_pipeline` | `8.848037e-01` | ok |

Quick mode intentionally runs only one gravity-block mesh resolution, so the
convergence rate is explicitly reported as unavailable rather than claimed.
The quick timing run reports `brute_force_over_dynamic_full_pipeline < 1`, so
it does not support an acceleration claim on this tiny CI case.

## Timing Evidence

The Phase-3 runner reports isolated contact timing rows for:

- broad-phase build/update
- candidate query only
- local projection only
- dynamic SDF query only
- contact Jacobian assembly only
- penalty force assembly only
- full contact pipeline end-to-end
- FEM assembly only
- linear solve only

It also reports the compared method workloads:

- `brute_force_all_triangle_projection_workload`
- `spatial_hash_candidate_only_workload`
- `spatial_hash_plus_local_projection_workload`
- `dynamic_fem_induced_sdf_full_pipeline_workload`

The test suite verifies that compared benchmark labels do not reuse the same
workload.

Latest full test output:

```text
pytest -q
.....................................................                    [100%]
53 passed in 1.65s
```

## Coverage

Phase-3 validation evidence covers:

- uniaxial linear elastic patch energy consistency
- fixed-base elastic block under gravity
- undamped free-vibration energy conservation smoke check
- deformable point/block sample against a rigid plane
- deformable-deformable penalty contact action-reaction
- required low-level checks for TET4 stiffness/mass, boundary faces, local
  projection, contact gap sign, contact Jacobian finite difference, and core
  Abaqus de-dependency

## Remaining Risks

- The validation suite is deterministic and CI-sized; it is not a paper-scale
  validation corpus.
- No nonlinear FEM, friction, self-contact, GPU path, barrier contact, or
  production BVH is implemented or validated.
- The SDF remains an oriented local surface distance for outward-oriented
  surfaces, not a robust global sign method for arbitrary geometry.
- Quick-mode convergence is unavailable by design because only one mesh
  resolution is run.
- Timing evidence is useful for infrastructure checks, but should not be used
  as a paper performance claim without larger meshes and repeated runs.
