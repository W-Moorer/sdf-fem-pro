# Legacy Audit

Date: 2026-05-12

Scope: this audit covers the repository as found locally, with source concentrated in `legacy_abaqus_prototype/contact_sdf/` and no implemented core package under `sdf_fem_contact/` beyond an empty `results/` directory.

## Executive Summary

The current implementation is an Abaqus-seeded prototype, not a standalone FEM-SDF solver. The main driver reads an Abaqus `.inp`, imports global mass and stiffness matrices from `GLOBALM.mat` and `GLOBALK.mat`, assembles external loads from `Force1/F*.mat`, exports intermediate Abaqus-style `.inp` files, converts those to OBJ, builds grid SDFs, predicts deformed SDFs from reference SDFs, detects contact by grid SDF sampling, and solves a dynamic system using the imported matrices.

The reusable value is mostly conceptual: mesh parsing, boundary-face extraction, generalized-alpha integration structure, barycentric/HEX8 shape-function utilities, SDF grid querying, broad-phase candidate filtering, and contact force assembly patterns. The core solver must still be rewritten around native mesh data, internal TET4 mass/stiffness/internal-force assembly, explicit boundary conditions, current-surface projection contact, and analytically checkable contact Jacobians.

At audit time the legacy tree also contains generated/reference data: `GLOBALM.mat`, `GLOBALK.mat`, 1534 files under `Force1/`, and 17 Abaqus-exported `.xlsx` files under `excelfile/`. The `inpfile/` and `objfile/` directories are present but empty.

## Direct Legacy Dependencies

| Dependency | Modules / data | Evidence | Core disposition |
| --- | --- | --- | --- |
| Abaqus `.inp` input | `Abaqusread.py`, `main.py`, `mesh_processor.py`, `pre_processing.py`, `FEM.py` export path | `AbaqusRead` opens the input file at `Abaqusread.py:24-26`; `main.py` passes `inp_file` into `AbaqusRead` at `main.py:32`; the default script uses `E:/abaqus_workspace/Job-3.inp` at `main.py:123`; `mesh_processor.py:4` converts Abaqus INP to OBJ. | Keep only as an optional importer/export validation path. Do not import it from core solver modules. |
| `GLOBALM.mat` | `FEM.py`, `main.py` | `FEMSystem.__init__` calls `_load_mass_matrix` at `FEM.py:35`; `_load_mass_matrix` uses `scipy.io.loadmat` at `FEM.py:279`; the script config sets `mass_mat = "GLOBALM.mat"` at `main.py:124`. | Must be replaced by internal TET4 mass assembly. |
| `GLOBALK.mat` | `FEM.py`, `main.py` | `FEMSystem.__init__` calls `_load_stiffness_matrix` at `FEM.py:39`; `_load_stiffness_matrix` uses `loadmat` at `FEM.py:289`; the script config sets `stiffness_mat = "GLOBALK.mat"` at `main.py:125`. | Must be replaced by internal TET4 stiffness/internal-force assembly. |
| `Force*.mat` / `Force1/` | `FEM.py`, `main.py`, data tree | `_load_element_force` reads `F{global_elem_id}.mat` at `FEM.py:299-300`; `assemble_global_force_vector` enumerates `F*.mat` at `FEM.py:321-345`; script config sets `force_folders = ["Force1"]` at `main.py:126`. | Must be replaced by native external force/load APIs. |
| Abaqus `.xlsx` validation outputs | `post_processing.py`, `huitu*.py`, `test1.py`, `excelfile/` | `post_processing.py:397-401` reads node history from Excel; `post_processing.py:462-484` overlays Abaqus curves; `huitu.py:61-63`, `huitu.py:366-374`, `test1.py:49-106` do the same. | Keep only in validation/report scripts outside core. |
| Abaqus `.odb` | None found | No source references to `.odb` were found. | No action except preserve the no-ODB core rule. |
| Hard-coded paths | `main.py`, `Abaqusread.py` demo | `main.py:123` uses `E:/abaqus_workspace/Job-3.inp`; `Abaqusread.py:303` uses `E:/abaqus_workspace/Job-1.inp`. | Remove from runnable entry points; examples should use relative test assets. |

## Module-by-Module Audit

### `legacy_abaqus_prototype/contact_sdf/main.py`

This is the legacy orchestration script. It constructs an `AbaqusRead` parser, translates `Part-2`, creates `FEMSystem` from `.mat` matrices and force folders, exports reference geometry, computes grid SDFs, builds a contact detector, solves dynamics, and saves `sim_result_data.npz`.

Reusable concept: the high-level pipeline order is useful as a migration checklist: mesh -> body state -> surface geometry -> contact detector -> time integrator -> saved validation output.

Must be rewritten: the entry point depends directly on Abaqus `.inp`, `GLOBALM.mat`, `GLOBALK.mat`, and `Force1/`. It also assumes `Part-1` and `Part-2`, a hard-coded Windows path, a relative `objfile` output, and a fixed time span `np.linspace(0, 0.6, 10000)` at `main.py:79`.

### `legacy_abaqus_prototype/contact_sdf/Abaqusread.py`

This is an Abaqus INP parser. It identifies `*Part`, parses `*Node`, parses `*Element`, and special-cases the node set `Set-BC`.

Reusable concept: optional import of part/node/element data from Abaqus `.inp`.

Must be rewritten or quarantined as optional IO: the parser should not become a core mesh type. It assumes Abaqus syntax, treats only `Set-BC` as a boundary set (`Abaqusread.py:179-214`), and assigns that set to `Part1` or the first part (`Abaqusread.py:236-243`). The demo also hard-codes an absolute Windows path at `Abaqusread.py:303`.

### `legacy_abaqus_prototype/contact_sdf/FEM.py`

This file contains the dynamic solver facade, but it is not an internal FEM assembly implementation. It imports `GLOBALM.mat` and `GLOBALK.mat`, assembles external force from `Force*.mat`, computes damping as Rayleigh damping, contains a generalized-alpha integrator, provides element shape-function helpers for contact mapping, and exports deformed Abaqus `.inp`.

Reusable concepts:

- Part DOF offset bookkeeping (`FEM.py:15-34`).
- Generalized-alpha Newton iteration structure (`FEM.py:66-274`).
- TET barycentric coordinate utility (`FEM.py:360-374`).
- HEX8 shape functions and local-coordinate Newton solve (`FEM.py:375-421`).
- Mapping a contact point to weighted global DOFs (`FEM.py:489-519`).

Must be rewritten:

- Matrix assembly is external: `self.M = self._load_mass_matrix(...)` and `self.K = self._load_stiffness_matrix(...)` at `FEM.py:35-39`.
- Loads are external `.mat` files: `_load_element_force` at `FEM.py:299-319` and `assemble_global_force_vector` at `FEM.py:321-349`.
- There is no native TET4 constitutive model, no element stiffness/mass integration, and no native Dirichlet constraint application.
- `solve()` uses imported `M`, `K`, `F_gravity` and residual `F_gravity + F_contact - C v - K u` (`FEM.py:533-602`).
- Contact tangent stiffness is computed but not actually included in the Jacobian; the relevant code is commented out at `FEM.py:609-619`.
- Exports create `.inp` and `.obj` in fixed relative directories (`FEM.py:635-650`, `FEM.py:652-704`).

### `legacy_abaqus_prototype/contact_sdf/mesh_processor.py`

This file converts Abaqus INP volume elements to OBJ surface triangles and parses OBJ files. Boundary faces are detected by a sorted-node face dictionary: first occurrence adds a face, second occurrence removes it as internal (`mesh_processor.py:55-63`). It supports C3D8 and C3D10 in this converter (`mesh_processor.py:29-52`) and flips faces using the element center direction (`mesh_processor.py:97-116`).

Reusable concept: boundary face extraction by counting element faces, surface triangulation, OBJ parsing for visualization.

Must be rewritten for core: the core should extract boundary faces directly from native mesh connectivity, including TET4 first. It should not rely on writing `.inp`, converting to `.obj`, or Abaqus element names. The current C3D10 path uses only corner nodes for faces.

### `legacy_abaqus_prototype/contact_sdf/gradient_sdf.py`

This module builds dense and sparse/hashed grid SDFs from OBJ meshes using libigl if available. It stores SDF, gradients, grid metadata, nearest points, and query helpers. The sign convention is explicitly inside negative and outside positive (`gradient_sdf.py:638`, `gradient_sdf.py:675`, `gradient_sdf.py:717`).

Reusable concepts:

- SDF sign convention and gradient storage.
- Dense grid and hashed narrow-band storage.
- Batch query structure and optional broad-phase SDF acceleration.

Must be rewritten or reduced for core:

- It depends on OBJ input and libigl for SDF construction (`gradient_sdf.py:38-42`, `gradient_sdf.py:624-736`).
- A grid SDF can be a broad phase, but final contact gap/normal/Jacobian must come from current FEM surface projection, not from a precomputed grid.
- The fallback branch appears to return `C_grid` without defining it if `igl.signed_distance` is unavailable but the AABB/FWN fallback is used (`gradient_sdf.py:695`, `gradient_sdf.py:730`, `gradient_sdf.py:736`).

### `legacy_abaqus_prototype/contact_sdf/sdf_predict.py`

This module predicts deformed SDF grids from a reference SDF and surface displacement samples. It has an inverse-map predictor, `phi_pred(x) = phi0(x - u(x))`, and a Taylor predictor, `phi_pred = phi0 - n dot u`, with displacements interpolated by KDTree from reference vertices (`sdf_predict.py:394-470`, `sdf_predict.py:476-528`).

Reusable concept: fast interpolation and approximation ideas may be useful for visualization or a broad phase.

Must be rewritten for the main method: this is not the required dynamic SDF induced by the current FEM surface geometry. It is an approximate reference-grid prediction and does not provide the final local projection, contact normal, or contact Jacobian required by the project goal.

### `legacy_abaqus_prototype/contact_sdf/contact_test.py`

This file defines `ContactDetectorSurface`. It samples one body's surface triangles, evaluates the other body's SDF by trilinear interpolation, normalizes the SDF gradient into a contact normal, and returns contact pairs when `phi_other <= sdf_tol` (`contact_test.py:226-519`).

Reusable concepts:

- Vectorized trilinear value/gradient interpolation (`contact_test.py:14-219`).
- AABB broad-phase filtering (`contact_test.py:380-433`).
- Barycentric sampling of triangle faces (`contact_test.py:285-318`, `contact_test.py:445-466`).
- Candidate deduplication by position (`contact_test.py:320-347`).

Must be rewritten for core final contact:

- Detection is one-sided by `sample_mode` (`contact_test.py:392-411`).
- Final gap is the grid value, not a projection onto the current triangle surface.
- Final normal is a grid gradient, not necessarily the local surface normal from a closest-point or normal-tube solve.
- It returns `elem_id` for the sampled side but not a closest feature and shape functions for the opposing surface. The solver later tries to find an element containing the same point, which is not a robust closest-point contact model.

### `legacy_abaqus_prototype/contact_sdf/contact_force.py` and `contact_force_nvg.py`

`contact_force.py` implements a normal penalty/damping force and an approximate contact stiffness. `contact_force_nvg.py` adds a friction-cone projection and a local Nesterov solve, then assembles global contact forces and an approximate tangent.

Reusable concepts:

- Local-to-global contact force assembly using contact-point shape-function weights.
- Friction cone projection code (`contact_force_nvg.py:15-53`) may be conceptually useful.
- The local optimization framing `N * gamma + d = 0` subject to a cone is documented in comments (`contact_force_nvg.py:71-79`).

Must be rewritten for core:

- The gap sign convention is inconsistent between the no-friction and friction versions. `contact_force.py` uses `g = -p["phi0"]`; `contact_force_nvg.py` uses `gap_current = p["phi0"] - du_local[0]` and treats negative as penetration (`contact_force_nvg.py:184-205`).
- Tangent stiffness signs are approximate and not verified against the required contact Jacobian.
- Hard-coded contact parameters are embedded in `FEM.solve()`: `kn=4e6`, `cn=7500`, `kt=1e7`, `mu=0.5` (`FEM.py:522-596`).
- The computed contact tangent is not included in the Newton Jacobian in `FEM.py`.

### `legacy_abaqus_prototype/contact_sdf/pre_processing.py`

This file glues together parser geometry, OBJ conversion, static SDF generation, predicted dynamic SDFs, and the contact detector.

Reusable concepts:

- Vertex-to-node nearest-neighbor mapping (`pre_processing.py:12-27`).
- Face-to-volume-element mapping using shared node incidence (`pre_processing.py:29-53`).
- Contact-detector closure shape as an integration callback (`pre_processing.py:205-361`).

Must be rewritten:

- It is tightly coupled to Abaqus parser data, exported OBJ paths, reference SDF grids, and SDF predictors.
- It assumes named parts `Part-1` and `Part-2`, relative output directories, and a hard-coded Part-2 y-bound extension to at least `-1` (`pre_processing.py:147-166`).
- The required core detector should derive current surface geometry directly from the body mesh and current generalized coordinates.

### `legacy_abaqus_prototype/contact_sdf/post_processing.py`, `huitu*.py`, `test.py`, `test1.py`

These are visualization and validation scripts. They load `sim_result_data.npz`, generate plots/renders, and compare to Abaqus Excel exports.

Reusable concept: validation plots and deterministic result packaging may be useful later, but they should live outside the core package.

Must not be imported by core: they depend on `.npz` layout from the legacy driver, PyVista/Matplotlib/Pandas, and Abaqus `.xlsx` comparison files.

## Implicit Equations and Assumptions

### Structural Dynamics

The current residual uses the equation

```text
M a = F_gravity + F_contact - C v - K u
```

where `M`, `K`, and `F_gravity` are imported, not assembled. Damping is Rayleigh damping:

```text
C = alpha M + beta K
```

with both coefficients defaulting to zero in `FEM.py:52` and `FEM.py:353-354`.

The generalized-alpha integrator assumes `rho = 0.7` and computes

```text
alpha_m = (2 rho - 1) / (rho + 1)
alpha_f = rho / (rho + 1)
gamma = 0.5 + alpha_f - alpha_m
beta = 0.25 (gamma + 0.5)^2
```

at `FEM.py:87-94`. Step-size bounds are `hmin = 1e-6`, `hmax = 1e-3` (`FEM.py:101-103`).

### Imported Matrix Assumptions

The code assumes the imported matrices already match the parser-derived DOF count (`FEM.py:46-49`). It also implicitly assumes node IDs are 1-based and contiguous inside each part because DOF indices are computed as

```text
offset + (node_id - 1) * 3 + component
```

This appears throughout `FEM.py`, `pre_processing.py`, and plotting scripts. Non-contiguous node labels would break this mapping.

### Boundary Conditions

The parser reads a special `Set-BC`, but the dynamic and static solves do not apply native Dirichlet constraints. The imported Abaqus matrices may have constraints baked in, or the current simulations may be unconstrained. A standalone solver must make this explicit with a constraint API and tests.

### SDF Sign and Contact Gap

The SDF convention in `gradient_sdf.py` is outside positive, inside negative. The detector flags contact when

```text
phi_other(x) <= sdf_tol
```

and sets `normal = grad(phi_other) / ||grad(phi_other)||` (`contact_test.py:478-506`).

The force code then uses at least two gap conventions:

```text
g = -phi0                       # positive penetration in contact_force.py
gap_current = phi0 - du_n        # negative penetration in contact_force_nvg.py
```

The standalone solver should normalize this to the project definition:

```text
g_i = phi_B(x_i^A(q_A), q_B)
```

where positive means separation if the SDF remains outside-positive, negative means penetration, and contact forces/constraints use one consistent sign convention.

### Dynamic SDF Approximation

The legacy "dynamic SDF" is not recomputed from the current FEM surface. It predicts grid SDFs from a reference SDF:

```text
inverse map: phi(q, x) ~= phi0(x - u(x))
Taylor map:  phi(q, x) ~= phi0(x) - n(x) dot u(x)
```

with `u(x)` interpolated from nearest reference vertices by KDTree. This can be a broad-phase approximation, but it cannot be the final gap/normal/Jacobian method for the standalone solver.

### Contact Kinematics and Force Assembly

The solver stores an active contact pair with:

- sampled contact point `x`;
- SDF normal `n`;
- SDF value `phi0`;
- hit-side DOF indices and shape weights;
- optional other-side DOF indices and shape weights;
- a reference displacement snapshot for within-step incremental motion.

For force assembly, weighted point velocities and displacements are formed from nodal DOFs, local normal/tangent components are computed, and nodal forces are scattered by shape weights. This is useful scaffolding, but the core formulation must build these quantities from a robust projection pair and must expose the Jacobian:

```text
dg/dx_A = N_A n^T
dg/dx_B = -N_B n^T
```

with finite-difference tests.

### Geometry and Contact Candidate Assumptions

The legacy contact detector assumes two named parts and samples only one side by default (`sample_mode="2in1"`). Surface candidates are generated from triangle vertices and strict interior barycentric points, not from closest-point pairs. Candidate identity is deduplicated by rounded world coordinates with `eps=1e-4`. This is not sufficient for a production-quality finite-element contact formulation.

## Reuse vs Rewrite Summary

| Module | Reuse conceptually | Rewrite required |
| --- | --- | --- |
| `Abaqusread.py` | Optional `.inp` import reference. | Yes, as an optional importer only. |
| `FEM.py` | Integrator structure, DOF offsets, shape-function utilities. | Yes, replace matrix/load import with internal TET4 assembly and constraints. |
| `mesh_processor.py` | Boundary-face extraction and triangulation idea. | Yes, operate directly on native meshes. |
| `gradient_sdf.py` | SDF sign convention, grid/hash query ideas for broad phase. | Yes for core contact; final query must use current-surface projection. |
| `sdf_predict.py` | Fast interpolation ideas for visualization/broad phase. | Yes; not valid as main dynamic SDF method. |
| `contact_test.py` | AABB filtering, batched sampling, trilinear query mechanics. | Yes; final gap, normal, and Jacobian must be projection-based. |
| `contact_force.py` | Basic penalty scattering pattern. | Yes; sign/tangent must be redesigned and tested. |
| `contact_force_nvg.py` | Friction-cone projection concept. | Yes; integrate with consistent gap/Jacobian and solver tangent. |
| `pre_processing.py` | Mapping concepts. | Yes; current file is glue for the legacy pipeline. |
| `post_processing.py`, `huitu*.py`, `test*.py` | Validation and plotting ideas. | Keep outside core; do not import from core. |
| `main.py` | Pipeline outline. | Yes; replace with standalone examples using core APIs. |

## Proposed Migration Map

### Phase 0: Quarantine and Package Boundaries

- Keep all current files under `legacy_abaqus_prototype/`.
- Create real core modules under `sdf_fem_contact/`.
- Add a test that imports every core module and asserts no imports from `legacy_abaqus_prototype`, `Abaqusread`, Abaqus `.xlsx`, `GLOBALM.mat`, `GLOBALK.mat`, or `Force*.mat`.
- Move any future Abaqus importer to an optional namespace such as `sdf_fem_contact.io.abaqus_inp`.

### Phase 1: Native Mesh and Body Model

- Define native mesh containers: nodes, TET4 elements, boundary faces, node/element sets, and part/body IDs.
- Implement TET4 boundary-face extraction by sorted face counts.
- Define body state: reference positions, current positions, velocities, material, density, constraints, and external loads.
- Add tests for boundary face extraction and node/element indexing with non-contiguous labels.

### Phase 2: Internal TET4 FEM Assembly

- Implement TET4 linear elastic element stiffness with constant strain-displacement matrix.
- Implement consistent or lumped TET4 mass matrix with total mass conservation.
- Assemble global sparse `M`, `K`, internal force, external force, and Dirichlet constraints using NumPy/SciPy sparse matrices.
- Add tests required by `AGENTS.md`: stiffness symmetry, positive semidefiniteness under rigid modes, mass conservation, and no Abaqus imports.

### Phase 3: Current-Surface Contact Geometry

- Build boundary triangle data directly from current nodal positions.
- Implement closest-point projection on triangle with barycentric coordinates and feature classification.
- Implement broad phase separately: AABB, spatial hash, BVH, or optional SDF/contact tube.
- For final contact, compute gap, normal, closest point, active surface element, and shape weights from the current FEM surface.
- Add tests for closest triangle projection, gap sign convention, and contact Jacobian finite differences.

### Phase 4: Contact Residual and Jacobian

- Implement contact constraints with one consistent sign convention:

```text
g = phi_B(x_A(q_A), q_B)
```

- For a contact point on body A against body B, assemble Jacobian blocks:

```text
J_A = N_A n^T
J_B = -N_B n^T
```

- Start with frictionless normal contact. Add penalty or complementarity formulation after the geometry/Jacobian tests pass.
- Only then port friction-cone ideas from `contact_force_nvg.py`, with independent tests.

### Phase 5: Time Integration

- Port the generalized-alpha/Newmark structure into a solver module that consumes native assembled matrices/residuals.
- Apply Dirichlet constraints consistently to residuals, tangents, velocities, and accelerations.
- Include contact residual and tangent in Newton iterations instead of computing but ignoring contact stiffness.
- Add deterministic small examples: free vibration, constrained TET4 patch, and two-body contact.

### Phase 6: Validation and Legacy Parity

- Keep Abaqus `.inp` support only as optional import for mesh/sets, not as a source of matrices or forces.
- Keep Abaqus `.xlsx` comparisons in validation scripts only.
- Use legacy outputs as external reference plots, but do not require them to run tests or examples.
- Replace `main.py` with standalone examples that assemble all matrices and loads internally.

## Immediate Risks to Address First

1. Core package is currently empty, so the first implementation should establish clean package boundaries and tests before porting logic.
2. The legacy solver does not apply explicit Dirichlet constraints; this must be designed before dynamics validation.
3. The legacy dynamic SDF is a reference-grid prediction, not a current-surface projection. The contact geometry should be implemented from scratch.
4. Contact gap signs are inconsistent across force modules. Pick one convention and test it before adding friction.
5. Current DOF indexing assumes contiguous 1-based node IDs. Native core data structures should decouple external labels from internal array indices.
6. Contact tangent stiffness is currently not used by Newton. The standalone solver needs residual/tangent consistency from the start.
