# Locked Paper Numerical Experiments

This directory contains versioned evidence artifacts that are intended to be
used directly by the manuscript. Each subdirectory freezes a specific claim
boundary, input/output data, figures, and acceptance thresholds.

## Cases

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

## Claim Discipline

These artifacts support only their scoped claims. They do not by themselves
claim friction, self-contact, production BVH performance, arbitrary nonlinear
contact equivalence, or robust global signed distance on arbitrary geometry.
