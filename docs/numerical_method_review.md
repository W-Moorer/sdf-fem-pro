# Numerical-Methods Review After Acceptance Fixes

## P0 Blockers

None identified for first-stage prototype acceptance.

## P1 Major Issues Fixed

- Boundary orientation and SDF sign:
  - `src/sfc/mesh/topology.py::orient_tet4_connectivity`
  - `src/sfc/mesh/boundary.py::extract_boundary_faces`
  - `tests/test_mesh_topology.py`
  - `tests/test_mesh_boundary.py`
  - `tests/test_dynamic_surface_sdf.py`
  - Negative TET4 input is repaired and boundary faces can be oriented outward
    with coordinates.

- Contact force direction under reversed element input:
  - `tests/test_penalty_contact.py::test_contact_force_remains_repulsive_after_reversed_master_tet_input`
  - Verifies repulsive force direction after reversed master TET4 connectivity.

- Benchmark label/workload mismatch:
  - `benchmarks/run_contact_benchmark.py`
  - `tests/test_benchmarks.py`
  - Compared methods now use distinct workload labels and isolated timing rows.

- Public brute-force API exposure:
  - `src/sfc/sdf/__init__.py`
  - `src/sfc/sdf/dynamic_surface_sdf.py`
  - Slow all-face search is private/reference-only and not exported from
    `sfc.sdf`.

## P1 Major Issues Remaining

- Global robust closed-surface sign is not implemented for arbitrary geometry.
  Current support is an oriented local surface distance, validated on
  outward-oriented closed TET4 surfaces.
- No nonlinear FEM or large-deformation update is implemented.
- No friction, self-contact, active-set contact, or augmented/Lagrange contact
  formulation is implemented.
- Benchmark data is still CI-scale microbenchmark evidence, not paper-scale
  performance validation.

## P2 Minor Issues Fixed

- Strong no-legacy policy scan added for core source:
  - `tests/test_no_legacy_imports.py`
  - Checks legacy prototype import, legacy data filenames, `loadmat`,
    `read_excel`, `.odb`, `.xlsx`, and absolute Windows paths.

- Rigid mode testing strengthened:
  - `tests/test_tet4.py`
  - Adds infinitesimal rigid rotations and stiffness eigenvalue PSD check.

- Contact Jacobian testing strengthened:
  - `tests/test_penalty_contact.py`
  - Adds rotated triangle and multi-node slave finite-difference checks.

- README updated:
  - `README.md`
  - Now reflects implemented modules, install command, examples, benchmarks, and
    current limitations.

## Suggested Next Fixes

- Add manifold/orientation validation utilities for imported or generated
  surfaces.
- Add analytical convergence tests for TET4 linear elasticity.
- Add larger deterministic contact examples with known qualitative behavior.
- Add a real BVH or robust acceleration structure if large surface benchmarks
  become a paper goal.
- Add physical validation cases against analytical solutions or isolated
  external references.
