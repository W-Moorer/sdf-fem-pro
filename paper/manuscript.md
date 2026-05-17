# A FEM-Induced Dynamic Narrow-Band Signed Distance Field with Closest-Feature Sensitivities for Finite-Element Contact

## Abstract

Signed distance fields provide efficient geometric queries for contact, but static SDFs are not directly suitable for deformable finite-element bodies whose surfaces evolve with nodal degrees of freedom. This paper presents a training-free dynamic narrow-band SDF field for FEM contact. At each time step, the current FEM boundary is used to populate a narrow-band signed distance field in the deformed configuration. In addition to distance values, each grid node stores closest-feature payloads, including face identity, barycentric coordinates, and current surface normals. These payloads allow the interpolated SDF field to provide contact gaps, scalar-field normals, and FEM-consistent master-side contact sensitivities. Contact queries are performed by field interpolation; closest-point projection is used only during field construction or optional refinement. We derive the field-based contact Jacobian using the analytic derivative of the scalar trilinear SDF interpolation, assemble frictionless penalty contact forces, and analyze the amortized cost relation between SDF field update and repeated contact queries. Validation reports field accuracy, Eikonal residuals, slave/master Jacobian finite-difference errors, and the crossover query count beyond which the dynamic SDF field outperforms projection-based contact queries.

## 1. Introduction

Contact in deformable finite-element simulation requires repeated evaluation of gaps, normals, contact sensitivities, and force contributions on evolving surfaces. Static signed distance fields are attractive for contact because they replace repeated geometric search with local field queries, but a static field does not remain a current-configuration distance field when a finite-element body deforms.

Existing dynamic SDF approaches often use deformation snapshots, reduced bases, learned fields, or other data-driven approximations. Those methods can be effective, but they introduce offline data requirements and may decouple the distance representation from the actual finite-element state. This work instead constructs the SDF directly from the current FEM boundary at each update.

The proposed method builds a current-space narrow-band SDF field from the current boundary surface. A spatial-hash accelerated closest-point projection kernel is used only to populate grid nodes. Once the field exists, contact queries interpolate the field and its closest-feature payload. The method therefore keeps the current-surface geometric consistency of projection methods while making the paper-facing contact query a true field interpolation.

The contributions are:

1. A training-free FEM-induced dynamic narrow-band SDF field built from the current finite-element boundary.
2. Closest-feature grid payloads that store face identity, barycentric coordinates, and normals for FEM-consistent master-side sensitivities.
3. A variationally consistent scalar-field derivative for the slave Jacobian.
4. A field-interpolated contact formulation for gap, normal, master Jacobian, and frictionless penalty force.
5. An amortized cost model with measured crossover counts showing when field updates pay off over repeated projection queries.

## 2. Related Work

Mesh-based finite-element contact commonly defines normal gap functions through closest-point or mortar projections on the current surface \cite{wriggers2006computational,laursen2002computational}. These formulations provide a direct route to contact sensitivities, but repeated projection at many slave samples can dominate contact-query cost. Our method keeps projection as an internal grid-population kernel and moves repeated queries to a current-space SDF field.

Distance fields and level sets provide a geometric representation in which proximity and normals are queried from a scalar field \cite{osher2003level,frisken2000adaptively,jones2006distance}. They have also been used in collision and contact pipelines in graphics and simulation \cite{bridson2002robust}. In contrast to static distance fields, the field here is rebuilt from the current finite-element boundary and stores closest-feature payloads for master-side sensitivities.

Dynamic collision systems often use spatial subdivision, hashing, or bounding hierarchies to accelerate broad-phase search \cite{teschner2003optimized}. We use spatial hashing only inside field construction; the paper-facing contact query is interpolation of the narrow-band field. Learned implicit fields such as DeepSDF \cite{park2019deepsdf} demonstrate the value of continuous SDF representations, but our setting deliberately avoids snapshots, reduced bases, and neural training so that the distance field remains induced by the current FEM state.

## 3. FEM-Induced Dynamic Narrow-Band SDF Field

Let the current master boundary be a triangulated finite-element surface

\[
\Gamma_h(\mathbf{q}) =
\bigcup_e \mathbf{s}_e(\xi,\eta,\mathbf{q}),
\qquad
\mathbf{s}_e(\xi,\eta,\mathbf{q})
=
\sum_{a\in I_e}N_a^\Gamma(\xi,\eta)\mathbf{x}_a .
\]

At each field update, a Cartesian narrow band \(\mathcal{G}_\rho(\Gamma_h)\) is constructed around the current surface. For each grid node \(\mathbf{y}_\ell\), the field stores

\[
\left(
\Phi_\ell,\,
\mathbf{d}_\ell,\,
f_\ell^\ast,\,
\boldsymbol{\xi}_\ell^\ast,\,
\mathbf{n}_\ell,\,
m_\ell
\right),
\]

where \(\Phi_\ell=\phi_h(\mathbf{y}_\ell,\mathbf{q})\), \(\mathbf{d}_\ell\) is an optional stored gradient or normal estimate, \(f_\ell^\ast\) is the closest surface triangle, \(\boldsymbol{\xi}_\ell^\ast\) are barycentric coordinates on that triangle, \(\mathbf{n}_\ell\) is the current closest-feature normal, and \(m_\ell\) is the narrow-band validity mask. The contact derivative is not defined by interpolating \(\mathbf{d}_\ell\); it is defined as the derivative of the scalar interpolation of \(\Phi_\ell\).

Closest-point projection is used here as a field construction kernel. It is not the main contact query. The implementation rejects out-of-band or invalid interpolation cells rather than silently falling back to projection.

## 4. Field-Interpolated Contact Formulation

For a slave sample point

\[
\mathbf{x}_i^A =
\sum_{b\in I_A} N_b^A \mathbf{x}_b^A ,
\]

the contact gap is the trilinearly interpolated SDF:

\[
g_i =
\hat{\phi}_B(\mathbf{x}_i^A)
=
\sum_{\ell\in\mathcal{C}(\mathbf{x}_i^A)}
w_\ell(\mathbf{x}_i^A)\Phi_\ell .
\]

The spatial derivative used by the slave Jacobian is the analytic derivative of this scalar interpolation:

\[
\nabla_x\hat{\phi}_B(\mathbf{x}_i^A)
=
\sum_{\ell\in\mathcal{C}(\mathbf{x}_i^A)}
\Phi_\ell\,\nabla_x w_\ell(\mathbf{x}_i^A) .
\]

The contact normal is the normalized scalar-field derivative:

\[
\mathbf{n}_i =
\frac{\nabla_x\hat{\phi}_B(\mathbf{x}_i^A)}
{\|\nabla_x\hat{\phi}_B(\mathbf{x}_i^A)\|}.
\]

An interpolated closest-feature normal \(\tilde{\mathbf{n}}_i=\mathrm{normalize}(\sum_\ell w_\ell \mathbf{n}_\ell)\) is retained only as a geometric normal estimate for diagnostics or explicit refinement; it is not the derivative of the scalar gap.

The sign convention is positive for separation and negative for penetration.

The slave Jacobian is

\[
\frac{\partial g_i}{\partial \mathbf{x}_b^A}
=
N_b^A \nabla_x\hat{\phi}_B(\mathbf{x}_i^A)^T .
\]

The master Jacobian is assembled from the interpolated closest-feature payload:

\[
\frac{\partial g_i}{\partial \mathbf{x}_a^B}
=
-
\sum_{\ell\in\mathcal{C}(\mathbf{x}_i^A)}
w_\ell(\mathbf{x}_i^A)
N_a^\Gamma(\boldsymbol{\xi}_\ell^\ast)
\mathbf{n}_\ell^T .
\]

For frictionless normal penalty contact,

\[
E_c =
\frac{1}{2}k\langle -g_i\rangle_+^2,
\qquad
\lambda_i =
k\langle -g_i\rangle_+,
\qquad
\mathbf{f}_{c,i} =
\mathbf{J}_i^T\lambda_i .
\]

A Gauss-Newton contact stiffness approximation is assembled as \(k\mathbf{J}_i^T\mathbf{J}_i\).

## 5. Algorithm and Cost Model

One update/query cycle is:

1. Extract the current FEM boundary triangles.
2. Build \(\mathcal{G}_\rho(\Gamma_h)\).
3. Populate grid-node \(\Phi_\ell\), optional stored gradient, closest face id, barycentric payload, closest normal, and validity mask using the projection kernel.
4. Evaluate repeated contact samples by field interpolation.
5. Differentiate the scalar interpolation for slave sensitivities.
6. Assemble field-based contact Jacobians and penalty forces.

The field is beneficial only when the update cost is amortized over enough queries. With optional refinement count \(Q_r\), the measured gate is

\[
T_\mathrm{update}
+
Q T_\mathrm{field}
+
Q_r T_\mathrm{refine}
<
Q T_\mathrm{projection}.
\]

Without refinement,

\[
Q^\ast =
\frac{T_\mathrm{update}}
{T_\mathrm{projection}-T_\mathrm{field}} .
\]

The paper therefore claims acceleration only for query counts exceeding the measured crossover \(Q^\ast\), not for every possible workload.

## 6. Implementation

The field implementation is in `src/sfc/sdf/narrow_band_grid.py` and `src/sfc/sdf/dynamic_narrow_band_sdf.py`. The query API includes `query_phi` and `query_spatial_derivative_phi`; `query_gradient` is the derivative of the scalar \(\phi\) interpolation, while `query_geometric_normal` exposes the optional interpolated closest-feature normal. The field-contact implementation is in `src/sfc/contact/field_contact.py`. The older projection path in `src/sfc/sdf/dynamic_surface_sdf.py` is retained as `surface_projection_distance_kernel` for field construction, validation references, and explicitly requested refinement.

The core finite-element mechanics remain internally assembled and independent of Abaqus-generated matrices, force files, `.odb` files, and spreadsheet outputs. No friction, self-contact, nonlinear FEM main method, GPU path, barrier contact, POD, neural SDF, or Abaqus dependency is introduced.

## 7. Experiments

All Phase-8 paper-facing true-SDF outputs are generated by:

```bash
python validation/run_true_dynamic_sdf_field_validation.py --out-dir results/true_sdf_final
```

The experiment uses a deterministic nonplanar triangulated current surface with 512 triangles and three grid spacings. The query path is interpolation-only; projection appears only in field construction and reference/error measurement. All reported gradient and Eikonal quantities use \(\sum_\ell\Phi_\ell\nabla_x w_\ell\), the derivative of the scalar gap interpolation.

### 7.1 Field Accuracy and Eikonal Residual

This experiment evaluates whether the rebuilt current-space field remains a usable local SDF as grid spacing changes. The reference distance and normal are computed by projecting each sample to the current triangulated surface, while the method evaluates the same samples by field interpolation. The table shows that the scalar gap error decreases as the spacing is refined; the normal error remains bounded because it is computed from the piecewise trilinear scalar derivative on a triangulated nonplanar surface.

| Spacing | Grid nodes | Valid nodes | Max \(\phi\) error | Max normal error | Max Eikonal residual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.100 | 1792 | 876 | \(1.835893\times10^{-3}\) | \(5.751473\times10^{-2}\) | \(6.784055\times10^{-3}\) |
| 0.075 | 3969 | 2174 | \(1.132281\times10^{-3}\) | \(4.769023\times10^{-2}\) | \(4.546733\times10^{-3}\) |
| 0.050 | 12493 | 7826 | \(5.114950\times10^{-4}\) | \(3.600289\times10^{-2}\) | \(4.743522\times10^{-3}\) |

Figures:

- `results/true_sdf_final/figures/field_phi_error_vs_spacing.png`
- `results/true_sdf_final/figures/field_normal_error_vs_spacing.png`
- `results/true_sdf_final/figures/field_eikonal_residual_vs_spacing.png`

### 7.2 Field Contact Jacobian and Force

This experiment checks whether the field-contact Jacobian is actually the derivative of the field gap used in the penalty energy. The slave block is compared against finite differences of \(\hat{\phi}(\mathbf{x})\), and the master block is compared against finite differences after rebuilding the field under master nodal perturbations. The small errors verify both the scalar-interpolation derivative used on the slave side and the closest-feature payload sensitivity used on the master side.

The slave and master Jacobian finite-difference checks pass:

| Case | Gap | Normal | Slave FD error | Master FD error | Status |
| --- | ---: | --- | ---: | ---: | --- |
| single slave / large triangle | \(-1.30\times10^{-1}\) | \((0,0,1)\) | \(2.875566\times10^{-11}\) | \(1.462164\times10^{-11}\) | passed |

The field penalty force check reports action-reaction balance to numerical precision and nonzero contact stiffness for the penetrating sample.

Figure:

- `results/true_sdf_final/figures/field_contact_jacobian_fd_error.png`

### 7.3 Material-Space SDF Baseline

This experiment isolates why a current-space field is needed for deformable FEM contact. The material-space baseline keeps a frozen reference-plane SDF while the current surface is tilted or sheared. The dynamic field is rebuilt from the current surface. The material-space field accumulates gap and normal errors because it no longer represents the deformed boundary; the rebuilt dynamic field avoids this mismatch.

| Case | Dynamic max \(\phi\) error | Dynamic max normal error | Material max \(\phi\) error | Material max normal error |
| --- | ---: | ---: | ---: | ---: |
| tilt_x | \(8.326673\times10^{-17}\) | \(1.523921\times10^{-15}\) | \(9.783015\times10^{-2}\) | \(1.193580\times10^{-1}\) |
| shear_xy | \(5.551115\times10^{-17}\) | \(3.351421\times10^{-15}\) | \(4.265482\times10^{-2}\) | \(9.962740\times10^{-2}\) |

Figure:

- `results/true_sdf_final/figures/material_space_vs_dynamic_field_error.png`

### 7.4 Amortized Timing and Crossover

The dynamic field is not claimed to be faster for small query counts. Its update cost is paid once per rebuild, while the interpolation query cost is paid per contact sample. The table reports the measured crossover \(Q^\ast\): finer spacing improves field resolution but increases update cost, so \(Q^\ast\) rises from hundreds to several thousand queries.

| Spacing | \(T_\mathrm{update}\) | \(T_\mathrm{field}\) | \(T_\mathrm{projection}\) | \(T_\mathrm{refine}\) | \(Q^\ast\) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.100 | 13.998359 | \(5.708600\times10^{-5}\) | \(1.762912\times10^{-2}\) | \(5.952888\times10^{-4}\) | 797 |
| 0.075 | 28.491003 | \(5.207117\times10^{-5}\) | \(1.692433\times10^{-2}\) | \(5.167875\times10^{-4}\) | 1689 |
| 0.050 | 73.523470 | \(5.026517\times10^{-5}\) | \(1.626798\times10^{-2}\) | \(4.684375\times10^{-4}\) | 4534 |

The field acceleration claim is supported only for \(Q>Q^\ast\) under the measured inequality. Optional refinement is implemented and timed, but the main acceleration gate uses \(Q_r=0\).

Figures:

- `results/true_sdf_final/figures/field_timing_components.png`
- `results/true_sdf_final/figures/field_speedup_vs_query_count.png` reports projection total time divided by SDF-field total time and marks each measured \(Q^\ast\).
- `results/true_sdf_final/figures/field_crossover_qstar.png`

### 7.5 External Visual FEM Validation

The preceding experiments verify the dynamic SDF field itself: field accuracy, Eikonal residuals, field-contact Jacobians, and the amortized query crossover. To make the resulting FEM/contact fields visually auditable, we include a three-dimensional boundary-surface visual comparison against a generated reference field. This comparison is not used to prove the SDF acceleration claim; it checks whether displacement, stress, strain, gap, and contact-pressure fields are physically interpretable in the tested configuration. CalculiX-style external data can be substituted in the same artifact format, but no external solver is required by the core SDF field construction or contact query.

| Metric | Value |
| --- | ---: |
| Displacement L2 relative error | \(1.404040\times10^{-2}\) |
| von Mises L2 relative error | \(2.699310\times10^{-2}\) |
| Strain-norm L2 relative error | \(2.104498\times10^{-2}\) |
| Gap \(L_\infty\) absolute error | \(4.000000\times10^{-4}\) |
| Pressure L2 relative error | \(9.654852\times10^{-3}\) |
| Contact-force relative error | \(8.775744\times10^{-3}\) |

Figures:

- `results/external_visual/figures/external_visual_fields.png` renders SFC, generated reference, and absolute error on the deformed 3D boundary surface.
- `results/external_visual/figures/dynamic_sdf_field_visualization.png` renders the current FEM boundary, SDF phi slice, validity slice, and contact-pressure surface in 3D.

## 8. Claim Gates

Supported for the Phase-8 experiments:

- True dynamic narrow-band SDF field exists.
- Field queries use interpolation only.
- Field accuracy passes the configured thresholds.
- Slave and master field-contact Jacobians pass finite-difference checks.
- SDF acceleration is supported after the measured \(Q^\ast\) crossover.
- External visual validation supports physical auditability for the tested generated reference case, not SDF acceleration or external-solver equivalence.

Not supported:

- Robust global SDF for arbitrary non-manifold geometry.
- Friction.
- Self-contact.
- Nonlinear FEM as the main method.
- GPU acceleration.
- Barrier contact.
- Superiority over production BVH.
- Abaqus-dependent core solver behavior.

## 9. Supplementary Context

Older CalculiX, C3D8, Phase-9, and external-solver replay results are not the main evidence for this method. They may be retained as supplementary context for external deformed-state checks, but the main paper claims are now field construction, field interpolation, field-contact Jacobians, penalty force assembly, and amortized SDF update/query cost.

## 10. Limitations

The field is valid only in the constructed narrow band. Sign handling assumes consistently oriented current surfaces in the tested cases. The current paper-facing implementation targets internally assembled linear finite-element mechanics and frictionless normal penalty contact. Timing results are Python prototype measurements and should not be interpreted as production BVH or hardware acceleration claims.

## 11. Evidence Package

Primary Phase-8 evidence:

- `results/true_sdf_final/field_accuracy.csv`
- `results/true_sdf_final/field_eikonal.csv`
- `results/true_sdf_final/field_contact_jacobian.csv`
- `results/true_sdf_final/field_contact_force.csv`
- `results/true_sdf_final/field_timing_scaling.csv`
- `results/true_sdf_final/field_crossover_scaling.csv`
- `results/true_sdf_final/material_space_vs_dynamic_field.csv`
- `results/true_sdf_final/true_dynamic_sdf_summary.md`

## Bibliography Status

The manuscript now uses first-pass formal references for FEM contact, level sets, distance fields, spatial hashing, cloth/contact simulation, and learned SDFs. Before submission, the bibliography should still be checked against the target venue's style and expanded with the closest SDF-contact papers for the final related-work positioning.
