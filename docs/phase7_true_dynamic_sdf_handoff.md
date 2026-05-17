# Phase-7 True Dynamic SDF Field Handoff

## 1. Code status

Exact `pytest -q` output:

```text
........................................................................ [ 28%]
........................................................................ [ 57%]
........................................................................ [ 85%]
....................................                                     [100%]
252 passed in 333.36s (0:05:33)
```

Exact validation command output:

```text
python validation/run_true_dynamic_sdf_field_validation.py --quick --out-dir results/true_sdf

True dynamic narrow-band SDF validation complete.
accuracy: results\true_sdf\field_accuracy.csv
gradient: results\true_sdf\field_gradient_eikonal.csv
jacobian: results\true_sdf\field_contact_jacobian.csv
force: results\true_sdf\field_contact_force.csv
timing: results\true_sdf\field_timing.csv
crossover: results\true_sdf\field_crossover.csv
summary: results\true_sdf\true_dynamic_sdf_summary.md
```

Changed implementation/manuscript files:

- `src/sfc/sdf/narrow_band_grid.py` added.
- `src/sfc/sdf/dynamic_narrow_band_sdf.py` added.
- `src/sfc/contact/field_contact.py` added.
- `tests/test_dynamic_narrow_band_sdf.py` added.
- `validation/run_true_dynamic_sdf_field_validation.py` added.
- `src/sfc/sdf/dynamic_surface_sdf.py` reframed as an internal projection kernel.
- `src/sfc/sdf/__init__.py` exports the field API.
- `src/sfc/contact/__init__.py` exports field-contact API.
- `src/sfc/contact/narrow_phase.py` and `src/sfc/contact/sdf_geometry.py` are marked legacy projection-backed paths.
- `paper/manuscript.md` and `paper/main.tex` were rewritten around the field method.
- `docs/phase7_true_dynamic_sdf_handoff.md` added for this handoff.

Core physics status:

- No `src/sfc/fem` or `src/sfc/analysis` files were changed.
- FEM element assembly, material models, constraints, and time integration were not changed.
- Contact geometry/SDF modules were changed to add the field path.
- Friction, self-contact, nonlinear FEM, GPU, barrier contact, POD, neural SDF, and Abaqus dependencies were avoided.

## 2. SDF field implementation

Main file paths and symbols:

- Narrow-band grid: `src/sfc/sdf/narrow_band_grid.py`
  - `NarrowBandGrid`
  - `GridInterpolation`
  - `GridPayloadInterpolation`
  - `compute_finite_difference_gradient`
  - `grid_spec_from_surface`
- Dynamic SDF field builder/query object: `src/sfc/sdf/dynamic_narrow_band_sdf.py`
  - `DynamicNarrowBandSDF.build(...)`
  - `DynamicNarrowBandSDF.query_phi(...)`
  - `DynamicNarrowBandSDF.query_gradient(...)`
  - `DynamicNarrowBandSDF.query_gap_normal(...)`
  - `DynamicNarrowBandSDF.query_payload(...)`
  - `DynamicNarrowBandSDF.refine_query_projection(...)`
  - `SDFUpdateStats`
  - `FieldQueryPayload`
- Projection kernel: `src/sfc/sdf/dynamic_surface_sdf.py`
  - `surface_projection_distance_kernel(...)`
  - legacy `dynamic_surface_sdf(...)`
- Field-based contact: `src/sfc/contact/field_contact.py`
  - `FieldContactConstraint`
  - `field_contact_constraint_from_sample(...)`
  - `field_contact_jacobian_entries(...)`
  - `field_contact_jacobian_row(...)`
  - `assemble_field_contact_jacobian(...)`
  - `field_penalty_contact_response(...)`

Projection usage:

- Projection is used during `DynamicNarrowBandSDF.build(...)` to populate grid-node `phi`, closest face id, barycentric coordinates, and closest normal.
- Projection is used by `DynamicNarrowBandSDF.refine_query_projection(...)` only when explicit refinement is requested.
- Projection is not called by `query_phi`, `query_gradient`, `query_gap_normal`, or `query_payload`.
- `tests/test_dynamic_narrow_band_sdf.py::test_field_query_path_does_not_call_projection` monkeypatches the projection primitive to fail and confirms field queries still pass.

## 3. Field data layout

`NarrowBandGrid` stores:

- `phi`: scalar signed-distance grid, shape `(nx, ny, nz)`.
- `gradient`: vector gradient grid, shape `(nx, ny, nz, 3)`.
- `closest_face_id`: closest boundary triangle id per grid node.
- `barycentric`: closest-point barycentric coordinates per grid node, shape `(nx, ny, nz, 3)`.
- `closest_normal`: closest/current-surface normal per grid node, shape `(nx, ny, nz, 3)`.
- `valid_mask`: narrow-band validity mask.
- `origin`, `spacing`, and `metadata`.

The current implementation stores surface normals as the default gradient mode and also provides finite-difference gradient support through `compute_finite_difference_gradient(...)` and `gradient_mode="finite_difference"`.

## 4. Field query API

- `query_phi(x)`: trilinearly interpolates `phi`.
- `query_gradient(x)`: trilinearly interpolates the stored gradient grid.
- `query_gap_normal(x)`: returns `(phi, normalized_gradient)`.
- `query_payload(x)`: returns interpolation weights, closest face ids, face node ids, barycentric payloads, and grid-node normals.
- `query_contact_jacobian(...)`: no monolithic method with this exact name was added; the equivalent API is `field_contact_constraint_from_sample(...)` followed by `field_contact_jacobian_row(...)` or `assemble_field_contact_jacobian(...)`.

All query methods reject out-of-grid or invalid narrow-band cells instead of silently falling back to projection.

## 5. Contact formulation

Sign convention:

- `g > 0`: separation.
- `g = 0`: contact surface.
- `g < 0`: penetration.

Field gap:

```text
g_i = phihat(x_i) = sum_l w_l(x_i) Phi_l
```

Normal:

```text
n_i = grad(phihat)(x_i) / ||grad(phihat)(x_i)||
grad(phihat)(x_i) = sum_l w_l(x_i) grad(Phi_l)
```

Slave Jacobian:

```text
dg_i / dx_b^A = N_b^A grad(phihat)(x_i)^T
```

Master payload sensitivity:

```text
dg_i / dx_a^B = - sum_l w_l(x_i) N_a^Gamma(xi_l*) n_l^T
```

Penalty force:

```text
E_c = 0.5 k < -g >_+^2
lambda = k < -g >_+
f_c = J^T lambda
K_c ~= k J^T J
```

## 6. Tests

Field-specific test file:

- `tests/test_dynamic_narrow_band_sdf.py::test_plane_sdf_exactness_and_payload`
- `tests/test_dynamic_narrow_band_sdf.py::test_interpolation_consistency`
- `tests/test_dynamic_narrow_band_sdf.py::test_finite_difference_gradient_consistency`
- `tests/test_dynamic_narrow_band_sdf.py::test_eikonal_residual_on_plane`
- `tests/test_dynamic_narrow_band_sdf.py::test_field_phi_and_normal_match_current_surface_projection`
- `tests/test_dynamic_narrow_band_sdf.py::test_out_of_band_query_raises`
- `tests/test_dynamic_narrow_band_sdf.py::test_field_query_path_does_not_call_projection`
- `tests/test_dynamic_narrow_band_sdf.py::test_field_contact_slave_jacobian_matches_finite_difference`
- `tests/test_dynamic_narrow_band_sdf.py::test_field_contact_master_jacobian_matches_finite_difference`
- `tests/test_dynamic_narrow_band_sdf.py::test_field_penalty_force_action_reaction`

Coverage mapping:

- Plane SDF exactness: `test_plane_sdf_exactness_and_payload`.
- Interpolation consistency: `test_interpolation_consistency`.
- Gradient finite-difference consistency: `test_finite_difference_gradient_consistency`.
- Eikonal residual: `test_eikonal_residual_on_plane`.
- Field phi vs current-surface distance error: `test_field_phi_and_normal_match_current_surface_projection`.
- Field normal vs current-surface normal error: `test_field_phi_and_normal_match_current_surface_projection`.
- Out-of-band behavior: `test_out_of_band_query_raises`.
- No-projection query path: `test_field_query_path_does_not_call_projection`.
- Slave Jacobian finite difference: `test_field_contact_slave_jacobian_matches_finite_difference`.
- Master Jacobian finite difference: `test_field_contact_master_jacobian_matches_finite_difference`.

## 7. Validation outputs

Generated files from `python validation/run_true_dynamic_sdf_field_validation.py --quick --out-dir results/true_sdf`:

- `results/true_sdf/field_accuracy.csv`
- `results/true_sdf/field_gradient_eikonal.csv`
- `results/true_sdf/field_contact_jacobian.csv`
- `results/true_sdf/field_contact_force.csv`
- `results/true_sdf/field_timing.csv`
- `results/true_sdf/field_crossover.csv`
- `results/true_sdf/true_dynamic_sdf_summary.md`

Accuracy and Jacobian metrics:

- Max phi absolute error: `3.469447e-17`.
- Max normal L2 error: `2.284410e-15`.
- Max Eikonal residual: `2.220446e-16`.
- Slave Jacobian max FD error: `2.875566e-11`.
- Master Jacobian max FD error: `1.462164e-11`.

Timing metrics from `field_timing.csv`:

- Field update time: `7.777860399801284` s.
- Field query time: `5.880774930119514e-05` s/query.
- Spatial-hash projection query time: `0.005500450750114396` s/query.
- Explicit refinement time: `0.00041890399530529975` s/query.
- Timed query count: `400`.
- Grid node count: `3200`.
- Valid grid node count: `1927`.

Crossover from `field_crossover.csv`:

- `Q* = 1430`.
- Denominator `T_projection - T_field = 0.005441643000813201` s/query.
- Gate inequality: `T_update + Q*T_field_query < Q*T_spatial_hash_projection`.
- Supported at `Q* + 1`: `true`.
- Refinement fraction in this quick gate: `0.0`.

## 8. Claim gates

- True SDF field exists: supported. Evidence: `field_timing.csv`, `NarrowBandGrid.phi`, `valid_mask`.
- Field query uses interpolation only: supported. Evidence: query implementation and no-projection monkeypatch test.
- Field accuracy claim: supported for the quick validation case. Evidence: phi error, normal error, and Eikonal residual below thresholds.
- Field contact Jacobian claim: supported for the quick validation case. Evidence: slave/master FD errors in `field_contact_jacobian.csv`.
- SDF acceleration over projection query after `Q*`: supported for the measured quick workload when `Q > 1430`.
- Optional refinement claim: implementation exists and timing is reported, but it is not part of the main interpolation-only claim gate and was not used in the quick acceleration inequality (`q_refine_fraction=0.0`).

## 9. Unsupported claims

The current handoff does not support claims of:

- Robust global SDF for arbitrary non-manifold geometry.
- Friction.
- Self-contact.
- Nonlinear FEM as the main method.
- GPU acceleration.
- Barrier contact.
- Superiority over production BVH or production collision pipelines.
- Neural SDF, POD, gappy POD, or data-driven SDF as the main method.
- Abaqus-dependent core solver behavior.

## 10. Manuscript impact

Rewritten sections:

- Title:
  - `paper/main.tex` title changed to dynamic narrow-band SDF.
  - `paper/manuscript.md` title changed to dynamic narrow-band SDF.
- Abstract:
  - Reframed from projection-induced local surface distance to dynamic narrow-band SDF construction and interpolation query.
- Introduction:
  - Reframed projection as a spatial-hash field-population kernel.
  - Main contact query is now trilinear field interpolation.
- Method:
  - Replaced "dynamic surface distance" with dynamic narrow-band SDF field construction.
  - Added grid payload and field-contact sensitivity equations.
- Experiments:
  - Added primary true dynamic SDF field validation and claim gates.
  - Older CalculiX/SfePy material is marked supplementary/contextual.
- Limitations:
  - Added narrow-band validity limitation.
  - Removed forward-looking unsupported expansion claims from the conclusion framing.
