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
  closed-form small-strain linear-elastic reference.
- `scikit_fem_cantilever_external/`: external open-source FEM comparison
  against scikit-fem for a 3D cantilever beam on matching TET4 meshes,
  boundary conditions, material parameters, and loads.
- `engineering_stress_clouds_latest/`: paper-facing engineering stress-cloud
  exports for external cantilever, rigid indenter contact, and
  deformable-deformable contact. These figures use the current nodal-averaged
  and subdivided boundary-surface stress interpolation scheme.

## Claim Discipline

These artifacts support only their scoped claims. They do not by themselves
claim friction, self-contact, production BVH performance, arbitrary nonlinear
contact equivalence, or robust global signed distance on arbitrary geometry.

