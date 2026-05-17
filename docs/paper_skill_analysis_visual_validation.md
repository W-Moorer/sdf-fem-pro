# Paper Skill Analysis: Visual Validation and Claim Support

This analysis follows the `research-paper-writing` skill, using its paper-review checklist: every major manuscript claim must be technically correct and explicitly supported by evidence.

## Mini Outline

- Main story: the paper is now a dynamic narrow-band SDF field paper, not a projection-query contact paper.
- Visual issue: the external FEM visual figure must show 3D rendered stress/strain fields, not 2D projected point clouds.
- Claim separation: SDF advantages are proven by field accuracy, Jacobian finite differences, interpolation-only query, and amortized timing; visual FEM fields support physical auditability only.
- Submission risk: generated-reference visual validation is useful, but it should not be described as CalculiX equivalence unless real CalculiX data are used for that figure.

## Claim-Evidence Map

| Claim | Evidence | Status |
| --- | --- | --- |
| A true dynamic narrow-band SDF field exists. | `DynamicNarrowBandSDF`, `NarrowBandGrid`, phi grid, payload grids, validity mask, and Phase-8 validation outputs. | Supported |
| Contact query uses field interpolation, not projection. | Field query APIs and monkeypatch/no-projection query tests. | Supported |
| Slave Jacobian is the derivative of the scalar interpolated gap. | `query_spatial_derivative_phi(x)` and nonplanar finite-difference gradient tests. | Supported |
| Master Jacobian uses closest-feature payload sensitivity. | Payload grids plus slave/master finite-difference Jacobian tests. | Supported |
| SDF acceleration is amortized after crossover. | `field_timing_scaling.csv`, `field_crossover_scaling.csv`, and measured \(Q^\ast\). | Supported only for \(Q>Q^\ast\) |
| Stress/strain/contact fields are visually plausible. | 3D boundary-surface visual validation artifacts and external visual metrics. | Supported as physical audit only |
| The method is equivalent to CalculiX or a production FEM contact solver. | No source-level external solver equivalence experiment; no friction/self-contact/nonlinear FEM main method. | Unsupported |

## Reviewer-Style Findings

1. The old 2D point-cloud external figure was a real presentation defect. It made stress and strain look like sampled points rather than FEM field clouds. The figure should be a deformed 3D boundary-surface rendering with matched SFC/reference color scales and separate absolute-error color scales.

2. The SDF visualization figure should also avoid point-cloud panels. The paper-facing view should render the current FEM boundary, SDF slice, narrow-band validity slice, and contact-pressure surface in 3D so that it looks like a field representation rather than sampled dots.

3. The manuscript must preserve the distinction between method evidence and physical visualization. Field accuracy, Eikonal residuals, Jacobian finite-difference tests, and \(Q^\ast\) support the SDF method claims. The external visual figure supports physical interpretability of displacement, stress, strain, gap, and pressure fields.

4. The current external reference is generated inside the validation script. This is acceptable for a deterministic visual-audit artifact, but the paper should not imply source-level CalculiX equivalence unless a real CalculiX dataset is used for that plotted comparison.

5. The paper should keep the unsupported-claims paragraph visible. It protects the contribution from overclaiming: no friction, self-contact, nonlinear FEM main method, GPU, barrier contact, neural/POD SDF, production BVH superiority, or Abaqus-dependent core solver.

## Self-Review Checklist

- Contribution: pass. The paper's contribution is now field construction plus closest-feature sensitivities for FEM contact.
- Writing clarity: mostly pass. The visual validation wording should explicitly say "3D boundary-surface rendering" to avoid ambiguity.
- Experimental strength: pass for scoped SDF claims; needs real external cases for stronger engineering validation.
- Evaluation completeness: pass for Phase-8 method gates; external visual validation remains diagnostic.
- Method design soundness: pass under stated scope; unsupported claims must remain excluded.

## Recommended Manuscript Position

The external visual validation subsection should say that the figure renders deformed 3D boundary surfaces and is included after the SDF field experiments. It should not dominate the main evidence chain and should not replace the field accuracy, Jacobian, or \(Q^\ast\) tables.
