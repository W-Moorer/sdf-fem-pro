# AGENTS.md

## Project goal

This project implements a standalone finite-element dynamic signed-distance-field contact solver for deformable bodies.

The code must not depend on Abaqus for matrix assembly, force assembly, time integration, or contact evaluation. Abaqus files may only be supported as optional importers or external validation references.

## Non-negotiable constraints

- Do not read or require `GLOBALM.mat`, `GLOBALK.mat`, `Force*.mat`, Abaqus `.odb`, or Abaqus-generated `.xlsx` files in the core solver.
- Do not hard-code Windows paths or absolute user paths.
- Do not make Abaqus a required dependency.
- Do not use data-driven SDF training, POD, gappy POD, or neural SDF as the main method.
- The main method is FEM-induced dynamic SDF based on the current finite-element surface geometry.
- Any legacy Abaqus-based code must stay under `legacy_abaqus_prototype/` and must not be imported by the core package.

## Abaqus theory alignment

Abaqus may be used as an external theory and validation reference, but never as a core dependency.

- New solver features should first align with the corresponding Abaqus theoretical formulation and output semantics, not with case-specific parameter tuning.
- When reproducing Abaqus-style behavior, document the theory being matched, such as implicit dynamics/HHT or generalized-alpha integration, automatic increment and cutback logic, nonlinear iteration convergence, contact enforcement, surface-to-surface constraint averaging, `*MPC, BEAM` RP kinematics, and RP reaction extraction by equivalent virtual work.
- Contact stiffness, smoothing, damping, or stabilization parameters must not be tuned only to match one benchmark curve. Prefer theory-based or mesh/material/contact-law-derived quantities that remain valid across cases.
- External Abaqus decks, `.inp` files, logs, and postprocessed metrics may be kept under validation or benchmark directories only. They must not be imported by `src/sfc` and must not be required for normal package execution.
- When Abaqus and SFC disagree, investigate formulation and output-measure definitions before changing numerical parameters. Typical checks include integration scheme, time increment control, contact law, master/slave or surface-to-surface semantics, MPC/RP reaction convention, stress/strain output location, and force/energy sign convention.

## Numerical method

The core formulation is:

1. Each deformable body has a volume mesh, material model, current nodal positions, velocities, and finite-element state.
2. FEM assembly must be performed internally.
3. The main first target is TET4 linear elasticity.
4. The solver must assemble mass matrix, stiffness matrix, internal force, external force, and Dirichlet constraints internally.
5. Contact gap is defined as:

   g_i = phi_B(x_i^A(q_A), q_B)

   where phi_B is the dynamic signed distance induced by the current FEM surface of body B.

6. The contact Jacobian must satisfy:

   dg/dx_A = N_A n^T

   dg/dx_B = -N_B n^T

7. SDF or contact-tube broad phase may be used for candidate detection, but final gap, normal, and Jacobian must be computed from the current FEM surface by local projection or normal-tube inversion.

## Testing expectations

Every implemented module must include tests.

Required tests:

- TET4 stiffness symmetry and positive semidefiniteness under rigid modes.
- TET4 mass matrix total mass conservation.
- Boundary face extraction correctness.
- Closest point projection on triangle.
- Contact gap sign convention.
- Contact Jacobian finite-difference check.
- No core module imports Abaqus-specific code.

When code changes are made, run pytest for the affected module or feature area before reporting task completion. At minimum, run the tests that cover the module you changed, for example:

pytest -q

Full-suite `pytest -q` is required only when the change is broad, cross-cutting, or affects shared solver behavior. If no code was changed, test execution is not required; mention that tests were skipped because the change was documentation-only or analysis-only.

## Coding style

- Use Python 3.11+.
- Use NumPy and SciPy sparse matrices.
- Prefer small pure functions with docstrings.
- Add type hints where practical.
- Keep examples deterministic.
- Do not introduce large dependencies without explicit justification.
