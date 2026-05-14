# Drop-Impact External Reference Solver Selection

## Decision

Use **CalculiX/ccx** as the first-choice external reference solver for the
next dynamic impact validation case.

Use **FEBio** only as a fallback if CalculiX cannot be installed or cannot run
the selected model robustly.

Do not use SfePy as the primary external dynamic-contact reference for this
case. The installed SfePy workflow is useful for linear elastodynamics and
quasi-static contact snapshots, but it is not a clean single-solver transient
dynamic contact reference for a ball-drop comparison in this repository.

## Local Availability Check

CalculiX is available in the current WSL Ubuntu environment:

- command: `wsl --exec bash -lc "ccx -v"`
- detected version: `This is Version 2.17`
- package source observed locally: Ubuntu `calculix-ccx`

The validation runner still uses explicit solver discovery and will skip/fail
cleanly if WSL or `ccx` is unavailable on another machine. FEBio remains a
fallback candidate but has not been used for this evidence.

## Why CalculiX First

CalculiX is the better first target for this specific validation because:

- It uses Abaqus-like `.inp` files, so a Python validation runner can generate
  a deterministic model without adding a large Python dependency.
- It supports direct integration dynamic analysis through `*DYNAMIC`.
- It supports penalty contact through `*CONTACT PAIR`.
- It has contact damping documented for implicit dynamic calculations.
- Its output files are text/binary formats that can be postprocessed into CSV
  time histories for center-of-mass displacement, velocity proxy, contact
  force proxy, minimum gap, and energy-like diagnostics.

References:

- CalculiX `*DYNAMIC` manual:
  https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node192.html
- CalculiX direct integration dynamics:
  https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node131.html
- CalculiX `*CONTACT PAIR` manual:
  https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node177.html
- CalculiX contact damping note:
  https://www.feacluster.com/CalculiX/ccx_2.18/doc/ccx/node233.html

## Why FEBio Is Fallback

FEBio is a strong candidate for rigid-wall contact and has a clean XML input
format, but it is less aligned with the current minimal linear TET4 FEM-SDF
prototype as a first external reference:

- FEBio contact is commonly configured through augmented-Lagrangian style
  contact settings, while the current SFC method uses normal penalty contact.
- Rigid-wall contact is available, but matching all solver parameters and
  extracting directly comparable nodal force histories may require more
  solver-specific setup.
- FEBio remains useful if CalculiX contact convergence is poor or if its output
  proves easier to parse for the final sphere/plane model.

References:

- FEBio contact overview:
  https://www.help.febio.org/FEBioStudio/FEBioStudio_1-5-Section-8.2.html
- FEBio rigid wall interface:
  https://help.febio.org/docs/FEBioUser-4-1/UM41-Subsection-3.14.4.html

## Proposed Clean Validation Case

Case name:

`elastic_sphere_drop_rigid_plane`

Physical setup:

- A small linear-elastic TET4 sphere or sphere-like discretized elastic body.
- Initial height above a rigid plane.
- Initial downward velocity, or gravity-driven drop from a known height.
- Frictionless normal contact only.
- No self-contact, no friction, no nonlinear material, no GPU, no barrier
  contact.

Implemented current version:

- Use an initial downward velocity instead of a long gravity-only fall. This
  shortens runtime and makes first impact time deterministic.
- Use both a sphere-like TET4 body and a structured TET4 block body.
- The sphere-like body now uses a finer, regularized latitude-ring point set
  with a small bottom contact patch. Quick `r1` has more than 100 nodes and
  dozens of bottom/near-bottom boundary triangles, so CalculiX face-based
  contact has real surface area and integration points near first impact.
- The sphere-like case uses a higher linear contact stiffness than the block
  case to limit penetration while retaining CalculiX convergence in the short
  impact window.
- Use the same nodal coordinates and TET4 connectivity for SFC and CalculiX.
- Use face-based CalculiX slave surfaces and a fixed shell master plane.
- Use the shell `SPOS` master surface as the effective contact plane. The
  generated shell thickness is `0.01`, so the contact plane is `floor_z +
  0.005`, not the shell mid-surface at `floor_z`.
- Use matching SFC boundary-face area weights for smooth penalty contact.
- Use explicit `*DYNAMIC,DIRECT,ALPHA=-0.05` in the generated CalculiX input
  and the same HHT-alpha parameters in the SFC validation runner.
- Use a CalculiX-aligned SFC validation mode with consistent mass, smooth
  overclosure law, contact tangent, Newton iterations, and substepping.

Implemented runner:

```bash
python validation/run_calculix_drop_impact_comparison.py --quick --out-dir results/calculix_drop_impact
```

Long 1-second diagnostic runner:

```bash
python validation/run_calculix_drop_impact_comparison.py --quick --duration 1.0 --dt 0.005 --output-frequency 10 --calculix-auto-step --out-dir results/calculix_drop_impact_1s
```

The long command keeps the same two quick-mode validation geometries but
overrides the total time to `1.0 s`, uses a coarser nominal output increment for
inspection, and allows CalculiX adaptive time stepping. It is a diagnostic
visualization/comparison run, not a replacement for the short-window acceptance
gate.

High-resolution sphere diagnostic:

```bash
python validation/run_calculix_drop_impact_comparison.py --case sphere_like_drop --resolution 2 --duration 1.0 --dt 0.001 --output-frequency 10 --calculix-auto-step --calculix-timeout-seconds 600 --out-dir results/calculix_sphere_r2_dt001
```

This reruns the drop-sphere case on a finer validation mesh. Resolution `2`
contains 257 nodes, 890 TET4 elements, 476 boundary faces, and 116 near-bottom
contact faces. Because the model has an initial downward velocity, rebound
above the starting height is not automatically rejected. The physicality gate
uses the ballistic height bound `z0 + v0^2 / (2g)`, which is `0.748288` for the
latest `r2` run. The latest `r2` run stays below that bound in both CalculiX
and SFC, but CalculiX still times out at `0.678506 s` and reports a large
energy-balance error in its log, so this high-resolution strong-impact sphere
remains diagnostic only.

Gentle external-contact reference candidate:

```bash
python validation/run_calculix_drop_impact_comparison.py --quick --case block_drop --duration 1.0 --dt 0.002 --output-frequency 1 --initial-velocity-z 0.0 --gravity 9.81 --contact-stiffness 5000 --out-dir results/calculix_gentle_contact_reference
```

This isolates a drop-from-rest block/plane impact with a conservative
physicality gate: the post-contact rebound height must not exceed the starting
height. It is intended as the next external reference attempt after the
strong-impact long run showed incomplete CalculiX convergence and nonphysical
rebound behavior.

Generated outputs:

- `results/calculix_drop_impact/calculix_drop_time_history.csv`
- `results/calculix_drop_impact/calculix_drop_metrics.csv`
- `results/calculix_drop_impact/calculix_drop_summary.md`
- `results/calculix_drop_impact/calculix_drop_z_cm.png`
- `results/calculix_drop_impact/calculix_drop_min_gap.png`
- `results/calculix_drop_impact/calculix_drop_force_proxy.png`
- `results/calculix_drop_impact/calculix_drop_contact_energy.png`

Quick mode currently runs:

- `sphere_like_drop`, resolution `1`
- `block_drop`, resolution `1`

Non-quick mode supports multi-resolution dynamic comparison:

- `sphere_like_drop`, resolutions `1, 2, 3`
- `block_drop`, resolutions `1, 2, 3`

## Required Time-History Outputs

Both SFC and the external solver should output:

- `time`
- TET4 reference-volume-weighted center-of-mass height `z_cm`
- TET4 reference-volume-weighted center-of-mass velocity proxy `v_cm_z`
- minimum gap to the effective CalculiX `SPOS` contact plane `min_gap`
- maximum penetration `max_penetration`
- active contact count or active contact area proxy
- normal contact force proxy
- kinetic energy
- strain/internal energy if available
- contact penalty energy if available
- gravitational potential and total mechanical energy proxy for SFC
- rebound height after first impact

## Comparison Metrics

The comparison should report:

- first impact time error;
- peak normal force relative difference;
- rebound height relative difference;
- rebound physicality check: post-contact center-of-mass height must remain
  within the ballistic height bound implied by the initial downward velocity
  and gravity. For drop-from-rest cases this reduces to not exceeding the
  initial height;
- minimum gap / maximum penetration trend;
- center-of-mass trajectory error;
- qualitative agreement of contact activation windows;
- runtime of SFC contact detection/force assembly versus brute-force
  projection.

## Acceptance Gates

Do not make a paper claim unless all of these are true:

- CalculiX or FEBio executable is detected and its version/command is logged.
- The external solver completes the same generated model.
- SFC and the external model use the same geometry, material constants, initial
  conditions, and contact normal convention.
- CSV and Markdown outputs are generated.
- The generated Markdown claim markers reference backing CSV fields.
- The comparison explicitly states whether the external solver is CalculiX or
  FEBio.
- CalculiX force and contact-energy curves are compared only when the runner
  parses nonzero `RF` / `CONTACT PRINT` evidence from the `.dat` output.
- If the external solver fails or is unavailable, the validation status is
  `not_available`, not `supported`.

## Implementation Plan

1. Done: added `validation/run_calculix_drop_impact_comparison.py`.
2. Done: added deterministic sphere-like and structured block TET4 mesh
   generation for validation only.
3. Done: added CalculiX `.inp` writer for the generated sphere/plane case.
4. Done: added WSL `ccx` solver discovery.
5. Done: added parser for CalculiX `.dat` displacement, fixed-floor RF total,
   contact spring energy, and contact element count output.
6. Done: added SFC time-history runner using assembled consistent `M`, `K`,
   HHT-alpha implicit dynamics, smooth overclosure penalty, slave area
   weighting, contact tangent, Newton iterations, and substepping.
7. Done: added plots for center-of-mass height, RF/normal force comparison,
   minimum gap, and contact energy.
8. Done: added quick-mode tests that skip when CalculiX is unavailable and
   validate generated outputs, contact activation, claim gates, and Markdown
   evidence markers.

## Current Status

Status: **implemented and quick-mode evidence generated**.

External solver selected: **CalculiX/ccx primary, FEBio fallback**.

Current supported claim:

- CalculiX and the CalculiX-aligned SFC validation mode activate contact at the
  same time scale on the generated gentle 1-second `block_drop` face-to-face
  impact model. The latest gentle run reported identical first-contact times
  of `9.60e-02 s`, and both CalculiX and SFC kept the post-contact rebound
  center-of-mass height below the starting height.

Current unsupported claims:

- These are still reduced validation models, not high-fidelity production
  sphere-impact benchmarks.
- The earlier `sphere_like_drop` resolution `0` and Cartesian-Delaunay
  resolution `1` cases were not valid contact-force references: the bottom was
  effectively point-like and CalculiX contact activated only after large
  penetration. The current sphere-like case uses a regular bottom contact patch
  and a higher penalty stiffness, and is limited to a short initial-impact
  window to avoid late persistent-contact convergence issues in CalculiX 2.17.
- The 1-second strong-impact diagnostic is not accepted as validation evidence:
  the high-resolution `sphere_like_drop r2` run stays below the ballistic
  rebound-height bound but still times out in CalculiX and reports a large
  CalculiX energy-balance error. It is therefore a mesh-quality diagnostic, not
  a validated external reference.
- CalculiX RF is parsed from the fixed master plane and is more direct than the
  previous acceleration proxy, but it remains an external-force total and must
  be interpreted with CalculiX's RF caveats.
- The comparison does not validate friction, self-contact, nonlinear material
  response, or production contact algorithms.

## CalculiX Alignment Notes

The SFC validation mode now follows the CalculiX penalty-contact structure more
closely without copying GPL source code:

- source review used the public CalculiX code paths `dyna.c`, `dynamics.f`,
  `shape4tet.f`, `calcmass.f`, `contactprints.f`, `springforc_f2f.f`, and the
  public *DYNAMIC documentation;
- C3D4/TET4 linear element geometry follows the same constant-gradient
  four-node tetrahedral formulation;
- implicit dynamics uses the documented HHT alpha method, with alpha fixed to
  the CalculiX default `-0.05` through `*DYNAMIC,DIRECT,ALPHA=-0.05`;
- the SFC alignment mode uses a consistent mass matrix, not the earlier lumped
  mass matrix;
- face-based slave surface instead of only a slave node set;
- three quadrature points per boundary triangle, area-scaled force, and
  shape-function distribution to slave face nodes, matching the structure of
  CalculiX `springforc_f2f.f` more closely than the previous nodal-area or
  face-centroid approximation;
- CalculiX-unavailable contact outputs are left blank and excluded from force
  or energy comparisons instead of being replaced by zero-valued proxy curves;
- smooth overclosure activation inspired by CalculiX's linear
  pressure-overclosure regularization;
- contact tangent included in the HHT effective solve;
- Newton iterations inside each time step;
- optional substepping, enabled by default in this validation runner.

This is still not a line-by-line reimplementation of CalculiX. The comparison
is intentionally restricted to generated C3D4/TET4 linear dynamics with a fixed
direct time increment. The contact gap and force remain the SFC dynamic-SDF
validation method, because that is the variable under study.

The earlier `sphere_like_drop` resolution `0` and Cartesian-Delaunay
resolution `1` cases were removed from the active comparison suite because the
bottom contact was effectively vertex-like. In those configurations CalculiX
2.17 either produced no useful RF/contact-energy output or activated contact
only after visible geometric penetration. The current regularized sphere-like
mesh uses a small bottom contact patch so the external face-based contact model
has enough bottom facets and integration points to trigger at first impact.
