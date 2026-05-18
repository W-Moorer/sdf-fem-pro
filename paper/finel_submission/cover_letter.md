Dear Editor-in-Chief,

We are pleased to submit the manuscript entitled "A FEM-Induced Dynamic Narrow-Band Signed Distance Field with Closest-Feature Sensitivities for Finite-Element Contact" for consideration in *Finite Elements in Analysis and Design*.

The manuscript presents a finite-element contact methodology in which a dynamic narrow-band signed distance field is rebuilt from the current finite-element boundary. Closest-point projection is used only as an internal grid-population kernel, while repeated contact evaluations use field interpolation. In addition to signed distance values, each grid node stores closest-feature payloads, including face identity, barycentric coordinates, and current normals, enabling master-side finite-element contact sensitivities. The contact formulation uses the analytic derivative of the scalar trilinear SDF interpolation for the slave Jacobian and supports both preserved node-to-surface sampling and surface-to-surface quadrature backends.

We believe the manuscript fits the scope of *Finite Elements in Analysis and Design* because it focuses on a computational methodology for finite-element contact, its computer implementation, and numerical demonstrations on practical contact problems. The validation includes SDF field accuracy, Eikonal residuals, finite-difference checks of slave and master contact Jacobians, independent CalculiX native-contact comparisons, three-dimensional displacement/stress/strain/contact-field visualizations, and timing studies that report measured crossover counts for contact-dominated workloads.

The manuscript does not claim a production contact solver, friction, self-contact, barrier contact, GPU acceleration, neural SDFs, or superiority over production BVH implementations. Its scoped contribution is a training-free current-space dynamic SDF field construction and interpolation-based FEM contact backend with closest-feature sensitivities and measured amortized performance.

This manuscript has not been published previously and is not under consideration elsewhere. All authors have approved the manuscript for submission. We declare no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

Sincerely,

[Corresponding author name]
[Affiliation]
[Email]
