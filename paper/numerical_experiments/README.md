# Locked Paper Numerical Experiments

This directory contains versioned evidence artifacts that are intended to be
used directly by the manuscript. Each subdirectory freezes a specific claim
boundary, input/output data, figures, and acceptance thresholds.

## Cases

- `commercial_sphere_drop_linear_elastic/`: first commercial-software
  benchmark for Abaqus/Explicit 2024 vs independent SFC Lagrangian-SDF
  explicit dynamics. The case uses a no-damping, frictionless linear-penalty
  flexible sphere drop and locks displacement/stress curves, error metrics,
  and solve-time comparisons.
- `commercial_sphere_drop_soft_large_deformation/`: matching commercial
  benchmark with the same geometry and contact law but a ten-times lower
  Young's modulus. This creates a visibly softer impact response and stores
  the corresponding displacement, P95 strain/stress curves, errors, and
  timing metrics.
- `block_drop_dynamic_sdf_calculix_1s/`: scoped dynamic contact trajectory
  comparison between SFC dynamic-SDF contact and CalculiX for the locked
  block-plane drop case.
- `tet4_analytic_patch_stress_strain/`: analytic affine TET4 stress-strain
  patch test. This verifies element strain and stress recovery against a
  closed-form small-strain linear-elastic reference and includes a 3D
  stress/strain cloud comparing SFC and analytic fields.
- `scikit_fem_cantilever_external/`: external open-source FEM comparison
  against scikit-fem for a 3D cantilever beam on matching TET4 meshes,
  boundary conditions, material parameters, and loads.
- `calculix_contactenergy_c3d8_replay/`: official CalculiX C3D8
  contact-energy test replayed with SFC current-surface dynamic SDF on
  triangulated C3D8 boundary faces. This folder also contains 3D stress/strain
  and contact-pressure figures for the final deformed C3D8 state, plus a
  registered SFC C3D8 static solve for direct stress/error plotting. This
  is contact-law/energy evidence, not a TET4 trajectory-equivalence claim.
- `calculix_official_contact_examples/`: conservative catalog and smoke-run
  package for selected official CalculiX contact examples. It records
  `contactenergy.inp`, `contact1/contact3/contact6`, `scheibe2f2f.inp.gz`,
  and `ball.inp.gz`, and uses claim gates to separate direct paper references
  from law-alignment checks and future engineering candidates.
- `calculix_official_law_sdf_replay/`: official CalculiX C3D8 contact-law
  replay package. It instruments `contactenergy.inp` and
  `contact1/contact3/contact6` with contact-output requests, compares SFC
  current-surface dynamic-SDF gaps against CalculiX `CDIS`, compares replayed
  pressure against `CSTR`, and claim-gates `CELS` energy equivalence.
- `calculix_scheibe_ball_sdf_validation/`: official `scheibe2f2f` FRD
  replay plus official `ball` dynamic-drop native trajectory probe. The
  `scheibe2f2f` row parses CalculiX `DISP/CONTACT` FRD data and validates SFC
  deformed-surface SDF gap/pressure against `COPEN/CPRESS`. The `ball` row
  runs native SFC C3D8 dynamics and compares against CalculiX output where
  available, but its trajectory-equivalence claim is gated off because the
  locked CalculiX run fails near first contact and does not provide a complete
  active-contact history.

## Claim Discipline

These artifacts support only their scoped claims. They do not by themselves
claim friction, self-contact, production BVH performance, arbitrary nonlinear
contact equivalence, or robust global signed distance on arbitrary geometry.
