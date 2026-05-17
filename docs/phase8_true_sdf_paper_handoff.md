# Phase-8 True SDF Paper Handoff

## 1. Code and Command Status

Commands run:

```bash
pytest -q
python validation/run_true_dynamic_sdf_field_validation.py --out-dir results/true_sdf_final
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`pytest -q` output:

```text
257 passed in 342.88s (0:05:42)
```

The non-quick validation completed and regenerated `results/true_sdf_final`.
`paper/main.pdf` was regenerated from `paper/main.tex`.
The final LaTeX pass completed without unresolved-reference or overfull-box warnings.

Core physics scope:

- The Phase-8 work extends validation and manuscript reporting around the existing dynamic narrow-band SDF field.
- No friction, self-contact, nonlinear FEM main method, GPU path, barrier contact, POD, neural SDF, or Abaqus dependency was added.
- Projection remains limited to SDF field construction, explicit refinement, and timing reference measurements.

## 2. Implementation Status

Main implementation files:

- `src/sfc/sdf/narrow_band_grid.py`
  - `NarrowBandGrid`
  - trilinear interpolation of scalar, vector, and payload grids
  - analytic spatial derivative of scalar trilinear interpolation
  - finite-difference support for stored field gradients
- `src/sfc/sdf/dynamic_narrow_band_sdf.py`
  - `DynamicNarrowBandSDF`
  - `build(...)`
  - `query_phi(...)`
  - `query_spatial_derivative_phi(...)`
  - `query_gradient(...)`
  - `query_geometric_normal(...)`
  - `query_gap_normal(...)`
  - `query_payload(...)`
- `src/sfc/contact/field_contact.py`
  - field-based gap, scalar-derivative normal, Jacobian, and penalty force routines

Projection usage:

- Used during `DynamicNarrowBandSDF.build(...)` to populate grid-node distance and closest-feature payloads.
- Used only if explicit refinement is requested.
- Not used by field query APIs; queries use grid interpolation only.
- The slave Jacobian uses `query_spatial_derivative_phi(...)`, i.e. `sum_l Phi_l grad_x w_l`, not interpolation of stored normals.

## 3. Field Data Layout

The SDF field stores:

- `phi` grid.
- `gradient` grid, with finite-difference support. This is stored field data and diagnostic support; the variational contact derivative is computed from the scalar `phi` interpolation.
- `closest_face_id` per valid grid node.
- `barycentric` coordinates per valid grid node.
- `closest_normal` per valid grid node.
- `valid_mask` / narrow-band mask.
- `origin`, `spacing`, and grid shape metadata.

## 4. Result Files

Generated in `results/true_sdf_final`:

- `field_accuracy.csv`
- `field_accuracy_samples.csv`
- `field_eikonal.csv`
- `field_gradient_eikonal.csv`
- `field_contact_jacobian.csv`
- `field_contact_force.csv`
- `field_timing_scaling.csv`
- `field_timing.csv`
- `field_crossover_scaling.csv`
- `field_crossover.csv`
- `field_speedup_vs_query_count.csv`
- `material_space_vs_dynamic_field.csv`
- `true_dynamic_sdf_summary.md`

Generated figures:

- `figures/field_phi_error_vs_spacing.png`
- `figures/field_normal_error_vs_spacing.png`
- `figures/field_eikonal_residual_vs_spacing.png`
- `figures/field_contact_jacobian_fd_error.png`
- `figures/field_timing_components.png`
- `figures/field_speedup_vs_query_count.png`
- `figures/field_crossover_qstar.png`
- `figures/material_space_vs_dynamic_field_error.png`

## 5. Field Accuracy and Eikonal Results

Non-quick validation used a deterministic nonplanar triangulated surface with 512 boundary triangles, three grid spacings, and 72 query samples per spacing.

| spacing | grid nodes | valid nodes | max phi error | max normal error | max Eikonal residual |
|---:|---:|---:|---:|---:|---:|
| 0.100 | 1792 | 876 | `1.835893e-03` | `5.751473e-02` | `6.784055e-03` |
| 0.075 | 3969 | 2174 | `1.132281e-03` | `4.769023e-02` | `4.546733e-03` |
| 0.050 | 12493 | 7826 | `5.114950e-04` | `3.600289e-02` | `4.743522e-03` |

Worst reported values:

- Max phi absolute error: `1.835893e-03`.
- Max normal L2 error: `5.751473e-02`.
- Max Eikonal residual: `6.784055e-03`.
- Max scalar-gradient finite-difference mismatch: `4.932319e-11`.

## 6. Field Contact Results

`field_contact_jacobian.csv`:

- Gap: `-0.13`.
- Normal: `(0, 0, 1)`.
- Slave Jacobian FD max error: `2.875566e-11`.
- Master Jacobian FD max error: `1.462164e-11`.
- Status: `passed`.

Additional strict consistency tests:

- Nonplanar `query_phi` finite differences match `query_spatial_derivative_phi`.
- `query_gradient` aliases the scalar-field derivative, not interpolated closest normals.
- Nonplanar slave Jacobian finite differences match the scalar field gap derivative.
- Normal from scalar derivative has bounded error against projection normals on nonplanar surfaces.

`field_contact_force.csv`:

- Penetration: `0.13`.
- Normal force: `13.0`.
- Slave force: `(0, 0, 13)`.
- Master force: `(0, 0, -13)`.
- Action-reaction residual: `0.0`.

## 7. Timing and Crossover

| spacing | update s | field query s/query | projection query s/query | refinement s/query | Q* |
|---:|---:|---:|---:|---:|---:|
| 0.100 | `13.998359` | `5.708600e-05` | `1.762912e-02` | `5.952888e-04` | 797 |
| 0.075 | `28.491003` | `5.207117e-05` | `1.692433e-02` | `5.167875e-04` | 1689 |
| 0.050 | `73.523470` | `5.026517e-05` | `1.626798e-02` | `4.684375e-04` | 4534 |

The acceleration claim is supported only beyond the measured crossover:

```text
T_update + Q*T_field_query + Q_refine*T_refine < Q*T_spatial_hash_projection
```

The Phase-8 runs used `Q_refine = 0` for the crossover table.

## 8. Material-Space Baseline Check

The dynamic field remains accurate after current-space surface updates, while the material-space field accumulates deformation error:

| case | dynamic phi error | dynamic normal error | material phi error | material normal error |
|---|---:|---:|---:|---:|
| tilt_x | `8.326673e-17` | `1.523921e-15` | `9.783015e-02` | `1.193580e-01` |
| shear_xy | `5.551115e-17` | `3.351421e-15` | `4.265482e-02` | `9.962740e-02` |

## 9. Claim Gates

Supported:

- True dynamic narrow-band SDF field exists: supported by actual `phi` grid and field payload grids.
- Field query uses interpolation only: supported by implementation and the no-projection query-path test.
- Scalar gap-gradient consistency: supported by nonplanar finite-difference tests.
- Field accuracy claim: supported for the tested deterministic nonplanar surfaces under the reported thresholds.
- Field contact Jacobian claim: supported by slave/master finite-difference tests.
- SDF acceleration claim: supported only when `Q > Q*` for the measured settings.
- Optional refinement claim: supported only as optional refinement, not as the main query path.

Not supported:

- Robust global SDF for arbitrary non-manifold geometry.
- Friction.
- Self-contact.
- Nonlinear FEM as the main method.
- GPU acceleration.
- Barrier contact.
- Production BVH superiority.
- Abaqus-dependent core solver behavior.

## 10. Manuscript Status

Updated manuscript files:

- `paper/manuscript.md`
- `paper/main.tex`

Rewritten sections:

- Title.
- Abstract.
- Introduction.
- Related Work.
- FEM-induced dynamic narrow-band SDF field method.
- Field-interpolated contact formulation.
- Algorithm and cost model.
- Experiments.
- Claim gates.
- Limitations.

Manuscript direction:

- The main method is now the current-space dynamic narrow-band SDF field.
- The scalar gap derivative is now defined as the derivative of trilinear `phi` interpolation, and the optional interpolated closest-feature normal is separated as a geometric estimate.
- Related Work and first-pass formal references were added.
- Projection is described as the field construction kernel and optional refinement only.
- CalculiX, C3D8, Phase-9, and external-solver replay material are demoted to supplementary/contextual status.

## 11. Remaining Risks

- The non-quick validation is deterministic and paper-facing, but more curved closed-surface cases would be needed before claiming broad geometry robustness.
- Timing is Python-prototype and hardware dependent.
- Narrow-band invalid-cell behavior is rejection, not global SDF repair.
- The current method is frictionless normal penalty contact with linear FEM focus.
