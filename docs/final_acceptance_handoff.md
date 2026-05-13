# Final Acceptance Handoff Report

Status: first-stage acceptance fixes applied. The code remains a minimal
standalone FEM-SDF solver prototype, not a paper-experiment-ready solver.

## Repository Status

- Branch: `main`
- Latest commit: `4364954`
- Uncommitted changes: yes
- Tracked diff: `README.md` modified.
- New/untracked implementation areas: `src/sfc/`, `tests/`, `benchmarks/`,
  `examples/`, `docs/`, `pyproject.toml`.
- Generated benchmark outputs are under ignored `results/benchmarks/`.

## Abaqus De-Dependency Status

- Core source under `src/sfc` has no references to Abaqus, legacy prototype
  imports, `GLOBALM.mat`, `GLOBALK.mat`, `Force*.mat`, `.odb`, `.xlsx`,
  `loadmat`, `read_excel`, `abaqus_workspace`, or absolute Windows paths.
- Strong policy test: `tests/test_no_legacy_imports.py`.
- Remaining Abaqus files/data are isolated in `legacy_abaqus_prototype/` and
  documented only in `docs/legacy_audit.md`.

## Acceptance Fixes Applied

- TET4 orientation repair:
  - `src/sfc/mesh/topology.py::orient_tet4_connectivity`
  - `VolumeMesh` now automatically reorders negatively oriented TET4 elements.
- Outward boundary faces:
  - `src/sfc/mesh/boundary.py::extract_boundary_faces(elements, X=None)`
  - With coordinates supplied, boundary faces are oriented outward using the
    adjacent element centroid.
- SDF sign robustness for the supported closed-TET4 case:
  - Tests now prove inside/outside sign is unchanged after reversed input
    connectivity is repaired.
  - Tests cover face-, edge-, and vertex-near sign cases.
- Public API cleanup:
  - `slow_reference_dynamic_surface_sdf` is no longer exported from `sfc.sdf`.
  - Production `dynamic_surface_sdf` requires supplied candidate faces.
- Numerical tests strengthened:
  - Rigid rotations and stiffness eigenvalue positive-semidefinite check.
  - Rotated-triangle and multi-node slave contact Jacobian finite differences.
  - Reversed-master-tet contact force remains repulsive.
- Benchmark validity improved:
  - Distinct workload names for compared methods.
  - Isolated timing stages for broad-phase build, candidate query, projection,
    SDF query, Jacobian assembly, penalty force, full contact pipeline, FEM
    assembly, and linear solve.

## Implemented Modules

- Mesh: `VolumeMesh`, orientation repair, boundary face extraction.
- FEM: isotropic linear elasticity, TET4 volume/gradients/B/stiffness/mass,
  sparse global stiffness/mass/gravity assembly.
- Time integration: Dirichlet utilities and linear Newmark-beta step.
- SDF/projection: point-triangle closest projection and oriented local dynamic
  surface distance over broad-phase candidates.
- Contact: spatial hash broad phase, narrow-phase constraints, contact
  Jacobian, penalty force, Gauss-Newton contact stiffness.
- Examples/benchmarks: deterministic two-tet example and quick CI benchmark.

## Validation Evidence

Latest full test run:

```text
pytest -q
.....................................................                    [100%]
53 passed in 1.65s
```

Editable install and example:

```text
python -m pip install -e ".[dev]"
Successfully installed sfc-0.1.0

python examples/two_tet_patch.py
Final z displacements: [ 0.         -0.00058139 -0.00058139 -0.0001689  -0.0001689 ]
```

Quick benchmark commands completed:

```text
python benchmarks/run_contact_benchmark.py --quick --out-dir results/benchmarks
python benchmarks/compare_bvh_vs_dynamic_sdf.py --quick --out-dir results/benchmarks
```

Outputs:

- `results/benchmarks/contact_benchmark.csv`
- `results/benchmarks/contact_benchmark.md`
- `results/benchmarks/contact_method_comparison.csv`
- `results/benchmarks/contact_method_comparison.md`

## Remaining Limitations

- Nonlinear FEM is not implemented.
- Friction is not implemented.
- Self-contact is not implemented.
- Only TET4 is supported.
- Only penalty contact is supported.
- Broad phase is a uniform spatial hash, not a production BVH.
- The SDF query is an oriented local surface distance. It is validated for
  outward-oriented closed TET4 surfaces, but not a robust global sign method for
  arbitrary non-manifold or open surfaces.
- Benchmarks are still microbenchmarks and should not yet be used as paper
  performance evidence without larger validation cases.

## Final Claim

Current status: **minimal standalone FEM-SDF solver prototype with first-stage
acceptance fixes applied**.

It passes first-stage acceptance for de-Abaqus and minimal independent solver
construction. It is still not research-grade or paper-experiment ready.
