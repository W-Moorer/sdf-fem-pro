# Phase-10 FEM-SDF Contact Experiment Design and Paper Storyline Review

This document uses the `research-paper-writing` skill workflow: clarify the paper story, map each major claim to evidence, and identify missing experiments before submission.

## 1. Target Paper Claim

The intended paper should prove a scoped claim:

> A FEM-induced dynamic narrow-band SDF can compute deformable-body contact accurately from the current finite-element surface, while reducing repeated contact-query cost in contact-dominated workloads after the measured crossover query count.

This is stronger than a pure SDF-field paper. It requires not only field accuracy and Jacobian tests, but also solver-level contact evidence: displacement, stress/strain, contact pressure, penetration/gap, reaction force, contact energy, and timing across representative flexible-body contact cases.

## 2. Current Storyline Assessment

The current manuscript is correctly centered on the dynamic SDF field:

- current FEM boundary -> dynamic narrow-band SDF field;
- projection only as field construction/refinement kernel;
- interpolation-only field query;
- scalar-field derivative for the slave Jacobian;
- closest-feature payload sensitivity for the master Jacobian;
- amortized cost model and measured \(Q^\ast\).

This is the right innovation core. However, it is still narrower than the target flexible-contact claim. The current evidence proves the SDF/contact-query layer well, but it does not yet fully prove solver-level accuracy across static/dynamic, linear/nonlinear, and multiple element families. Nonlinear FEM should remain an unsupported or future claim unless dedicated implementation and validation are added.

## 3. Innovation Point Analysis

| Innovation | Why it matters | Required evidence | Current status |
| --- | --- | --- | --- |
| Dynamic current-space narrow-band SDF field | Avoids stale material-space SDF under deformation. | Phi error, normal error, Eikonal residual, out-of-band behavior. | Supported |
| Closest-feature payload at grid nodes | Gives master-side FEM sensitivity from field interpolation. | Master Jacobian finite-difference tests. | Supported |
| Variational scalar-field slave derivative | Makes contact force consistent with scalar gap energy. | Nonplanar `query_phi` finite-difference gradient tests. | Supported |
| Field-interpolated contact force/Jacobian | Turns SDF field into FEM contact operator. | Slave/master Jacobian FD, action-reaction, force tests. | Supported at query/operator level |
| Solver-level flexible-contact accuracy | Shows FEM-SDF produces credible deformation/contact fields. | Static/dynamic contact examples with displacement/stress/pressure/gap/force comparisons. | Needs stronger experiments |
| Contact-dominated efficiency | Shows update cost is amortized over many contact samples. | \(T_\mathrm{update}\), \(T_\mathrm{field}\), \(T_\mathrm{projection}\), total step time, \(Q^\ast\). | Partially supported; needs solver-level cases |

## 4. Recommended Numerical Experiment Matrix

### E1. Field Construction and Query Accuracy

Purpose: prove the SDF field itself is valid.

Cases:

- analytic plane;
- tilted/sheared finite-element surface;
- curved triangulated surface;
- mesh refinement and grid-spacing study.

Metrics:

- max/mean \(\phi\) error against closest-surface oracle;
- normal error;
- Eikonal residual;
- scalar-gradient finite-difference consistency;
- out-of-band rejection rate.

Status: mostly implemented. Add one curved closed surface before broad geometry language.

### E2. Static Linear Contact, TET4

Purpose: prove solver-level accuracy for the first core target.

Case:

- deformable TET4 block pressed against a rigid plane or rigid indenter;
- optional second case: deformable-deformable block/block compression.

Reference:

- analytic patch/linear-elastic sanity checks where possible;
- optional external CalculiX/scikit-fem reference as validation artifact only.

Metrics:

- displacement relative error;
- von Mises stress/strain-norm error;
- contact pressure error;
- active contact area/active-set IoU;
- maximum penetration;
- reaction force and contact energy error.

Figures:

- 3D deformed displacement cloud;
- 3D von Mises cloud;
- 3D contact pressure/gap cloud;
- SFC/reference/error triptych.

### E3. Static Linear Contact, HEX8

Purpose: show the SDF field is not tied to TET4 boundary topology.

Case:

- C3D8/HEX8 block-plane compression or contact-energy replay;
- triangulate HEX8 boundary quads only for SDF field construction/query.

Metrics:

- same as E2;
- additional comparison of boundary quad triangulation sensitivity.

Status: current 3D visual validation is a start, but it is generated-reference only. A real external HEX8 static case would strengthen the paper.

### E4. Dynamic Linear Contact, TET4

Purpose: prove contact history accuracy.

Case:

- TET4 block drop/impact on rigid plane;
- prescribed initial velocity;
- frictionless normal penalty contact.

Metrics:

- first contact time;
- maximum penetration;
- peak normal reaction;
- contact energy history;
- center-of-mass trajectory;
- momentum/action-reaction balance;
- displacement/stress/contact-pressure snapshots.

Figures:

- time histories;
- 3D snapshots at pre-contact, peak contact, rebound;
- gap/pressure cloud.

### E5. Dynamic Linear Contact, HEX8

Purpose: show dynamic field contact works with another element family.

Case:

- HEX8 block impact or squeeze;
- same penalty/contact law as E4.

Metrics:

- same as E4;
- element-family comparison table.

### E6. Contact-Dominated Efficiency Scaling

Purpose: prove the practical efficiency claim.

Cases:

- many slave samples on a dense contact patch;
- sweep query count \(Q\);
- sweep grid spacing \(h\);
- compare field total cost against projection-query total cost.

Cost gate:

\[
T_\mathrm{update}
+ Q T_\mathrm{field}
+ Q_r T_\mathrm{refine}
<
Q T_\mathrm{projection}.
\]

Report:

- \(T_\mathrm{update}\);
- \(T_\mathrm{field}\);
- \(T_\mathrm{projection}\);
- optional \(T_\mathrm{refine}\);
- \(Q^\ast\);
- total step time including FEM assembly/solve/contact for contact-dominated cases.

Important wording:

- do not claim the SDF field is always faster;
- claim faster only after measured crossover and in contact-dominated workloads.

### E7. Nonlinear Static Contact

Purpose: possible future extension, not current supported claim.

Case:

- large indentation or large deformation block compression;
- nonlinear material/geometric formulation.

Status:

- Do not include as a main claim until nonlinear FEM assembly, consistent tangent/contact coupling, and validation are implemented.
- If used before implementation, it can only be an external replay/visualization context, not a core solver claim.

### E8. Nonlinear Dynamic Contact

Purpose: possible future high-end validation.

Case:

- large-deformation impact with nonlinear material/time integration.

Status:

- Out of current scope under the present constraints.
- Requires new solver capability and validation gates before publication claims.

## 5. Experiment Section Structure

The paper should organize experiments by claim strength:

1. SDF field accuracy and scalar-gradient consistency.
2. Field-contact Jacobian and penalty-force verification.
3. Static linear flexible-contact accuracy, TET4 and HEX8.
4. Dynamic linear flexible-contact accuracy, TET4 and HEX8.
5. Material-space SDF failure under deformation.
6. Contact-dominated amortized performance and \(Q^\ast\).
7. External visual FEM validation.
8. Limitations and unsupported nonlinear/friction/self-contact claims.

This order keeps the method proof before solver demonstrations and keeps external visual validation from replacing the SDF evidence.

## 6. Claim-Evidence Gate

| Paper claim | Minimum evidence required |
| --- | --- |
| Dynamic SDF field exists | Phi grid, payload grids, validity mask, interpolation-only query test |
| Field accuracy | Phi/normal/Eikonal thresholds on multiple surfaces |
| Contact operator correctness | Slave/master Jacobian FD, penalty force/action-reaction |
| Flexible-contact result accuracy | Static/dynamic displacement, stress, strain, pressure, force, energy comparisons |
| Multi-element applicability | TET4 and HEX8 cases with boundary extraction and SDF rebuild |
| Efficiency advantage | Measured \(Q^\ast\) and total step speedup for \(Q>Q^\ast\) |
| Nonlinear static/dynamic FEM | Not supported until nonlinear solver evidence is added |

## 7. Figure Font Requirement

All paper-facing generated figures should use Times-style English fonts:

- Matplotlib `font.family = serif`;
- `font.serif = Times New Roman, Times, Nimbus Roman, DejaVu Serif`;
- `mathtext.fontset = stix`;
- `pdf.fonttype = 42`;
- `ps.fonttype = 42`.

The current validation plotting scripts were updated to apply this style before figure generation.

## 8. Self-Review

- Contribution: strong if the paper stays focused on current-space dynamic SDF field plus FEM sensitivities.
- Writing clarity: the manuscript should state that solver-level flexible-contact accuracy is a separate validation layer from SDF-field accuracy.
- Experimental strength: adequate for SDF field claims, incomplete for broad flexible-contact claims.
- Evaluation completeness: needs static/dynamic solver-level cases with TET4 and HEX8 before broad claims.
- Method design soundness: the dynamic SDF field is coherent; nonlinear FEM must not be claimed without implementation and validation.
