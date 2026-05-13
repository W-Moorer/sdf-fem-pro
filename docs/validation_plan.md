# Validation Plan

## Scope

The standalone solver is validated in layers: mesh topology, TET4 element
kernels, global assembly, time integration, local projection, dynamic
surface-SDF contact, broad phase, and penalty contact force/Jacobian assembly.
Phase 3 focuses on validation infrastructure for the current standalone
prototype. Phase 4 adds reduced paper-scale evidence gates for mesh-resolution
studies, contact histories, and contact performance scaling. Phase 5 prepares
final paper experiment evidence by adding repeated timing statistics,
paper-figure generation, material-space SDF baseline comparisons, and
reference-validation gates. Phase 7 adds physical-comparison evidence for
stress/strain recovery, standard FEM displacement/stress trends, contact
force-displacement references, and total-step acceleration feasibility. The
external FEM comparison adds a validation-only scikit-fem reference solve for
the same linear TET4 cantilever model. None of these phases adds new core
physics. Phase 8 adds more engineering-style paper examples: a 3D beam
external FEM comparison, a prescribed rigid flat indenter pressing an elastic
block, and deformable-deformable block contact with action-reaction and 3D
stress-cloud evidence. These remain linear TET4, oriented current-surface
distance, and frictionless normal penalty contact validations. The external
contact solver comparison adds a SfePy two-body nonlinear penalty contact solve
and replays its final deformed contact state with SFC current-surface gap
queries.

## Phase-3 Validation Case Table

| Case | Expected quantities | Current status | Limitations |
| --- | --- | --- | --- |
| TET4 element stiffness | Element stiffness is symmetric, rigid translations and infinitesimal rotations have near-zero strain energy, and element stiffness is positive semidefinite up to numerical tolerance. | Covered by pytest checks in the TET4 test layer. | Linear TET4 elasticity only; no nonlinear material or large-deformation verification. |
| TET4 consistent mass | Element/global mass integrates to the expected total mass for density times volume. | Covered by pytest checks in the TET4 mass test layer. | No modal or dynamic convergence study yet. |
| Boundary face extraction | Single-tet mesh exposes four boundary faces; shared interior faces are removed; coordinate-aware extraction orients closed-tet boundaries outward. | Covered by mesh boundary and topology tests. | No full imported-surface manifold validation utility yet. |
| Triangle closest-point projection | Projection returns correct closest point, barycentric coordinates, distance, and region behavior for face, edge, and vertex cases. | Covered by local projection tests. | Triangle degeneracy handling is not a broad production geometry validation suite. |
| Dynamic surface-SDF sign convention | For outward-oriented closed TET4 surfaces, outside samples have positive gap and inside samples have negative gap. | Covered by dynamic surface-SDF tests, including orientation-repair cases. | Current method is an oriented local surface distance, not a robust global winding/sign algorithm for arbitrary open or non-manifold surfaces. |
| Contact gap and Jacobian | Contact gap follows `g = phi_B(x_A, q_B)`; finite differences agree with slave and master Jacobian blocks on axis-aligned, rotated, and multi-node samples. | Covered by penalty contact Jacobian tests. | Penalty contact only; no friction, active-set, augmented Lagrangian, or self-contact validation. |
| Contact force direction | Penalty force is repulsive for penetration, including a reversed-master-tet input after orientation repair. | Covered by penalty contact tests. | Qualitative force-direction check only; no impact benchmark or energy audit yet. |
| Core Abaqus de-dependency | Core source must not import legacy Abaqus prototype code or reference required Abaqus files, `.mat`, `.odb`, `.xlsx`, `loadmat`, `read_excel`, or absolute Windows paths. | Covered by the no-legacy policy test. | Legacy files remain quarantined for reference/importer work outside the core package. |
| Quick benchmark output generation | Benchmark scripts produce deterministic CSV and Markdown summaries for CI-scale timing smoke checks. | Covered by benchmark smoke tests and manual commands when run. | Microbenchmark evidence only; not paper-scale performance validation. |
| Two-tet deterministic example | Example runs without Abaqus dependencies and reports deterministic final displacement values. | Manually runnable; latest accepted output is recorded in the handoff docs. | Demonstration scale only; not a physical validation benchmark. |
| Phase-3 uniaxial patch | Affine uniaxial displacement energy matches the analytic linear-elastic energy for the assembled TET4 patch. | Implemented in `validation/run_phase3_validation.py`. | Energy consistency smoke check, not a full patch-test suite over many meshes. |
| Phase-3 fixed-base gravity block | Mean free-end vertical displacement is downward and static residual on free DOFs is small. | Implemented in `validation/run_phase3_validation.py`. | Quick mode uses one mesh; convergence rate is unavailable by design. |
| Phase-3 free vibration | Undamped Newmark average-acceleration energy drift is small over a short deterministic run. | Implemented in `validation/run_phase3_validation.py`. | Short linear smoke check only. |
| Phase-3 rigid-plane contact | Penetrating deformable sample has negative gap and repulsive penalty force. | Implemented in `validation/run_phase3_validation.py`. | Rigid plane is represented by one triangle; no friction or impact solve. |
| Phase-3 deformable-deformable contact | Penalty contact force resultant is action-reaction balanced between bodies. | Implemented in `validation/run_phase3_validation.py`. | Small sample-level force assembly check only. |
| Phase-3 contact timing evidence | Compares brute-force projection, spatial hash candidate query, spatial hash plus local projection, and dynamic FEM-induced SDF full pipeline with distinct workload labels. | Implemented in `validation/run_phase3_validation.py`. | CI-scale timings only; not paper performance evidence. |

## Phase-4 Paper-Scale Validation Case Table

| Case | Expected quantities | Current status | Limitations |
| --- | --- | --- | --- |
| Phase-4 uniaxial mesh-resolution study | At least three structured TET4 resolutions with affine uniaxial strain energy error reported per resolution. | Implemented in `validation/run_phase4_paper_validation.py`; quick mode uses three resolutions. | Reports error trends only; no theoretical convergence order is claimed. |
| Phase-4 gravity block mesh-resolution study | At least three structured TET4 resolutions with fixed-base gravity displacement reported against the finest available run. | Implemented in `validation/run_phase4_paper_validation.py`; quick mode uses three resolutions. | Linear static gravity check only; no nonlinear or experimental reference is included. |
| Phase-4 rigid-plane contact time history | Per-step time, minimum gap, maximum penetration, active contact count, normal force, contact energy, and action-reaction imbalance. | Implemented in `validation/run_phase4_paper_validation.py`. | Rigid plane is a triangle surface surrogate; no friction, impact restitution, or barrier method. |
| Phase-4 deformable-deformable contact time history | Per-step gap, penetration, active contact count, force, penalty energy, and action-reaction imbalance for two deformable surface samples. | Implemented in `validation/run_phase4_paper_validation.py`. | Sample-level penalty contact history; no self-contact or large-deformation dynamics. |
| Phase-4 contact performance scaling | At least three surface sizes comparing brute-force projection, spatial hash plus local projection, and dynamic FEM-induced SDF full pipeline with isolated timing rows. | Implemented in `validation/run_phase4_paper_validation.py`. | Quick mode is deterministic and small; material-space SDF baseline is explicitly unavailable. |
| Phase-4 evidence gates | Markdown claims must reference backing CSV fields; convergence trends require at least three resolutions; acceleration is reported only when measured speedup is greater than one. | Covered by `tests/test_phase4_validation.py`. | These are policy gates, not substitutes for large repeated experiments. |

## Phase-5 Paper Experiment Preparation Table

| Case | Expected quantities | Current status | Limitations |
| --- | --- | --- | --- |
| Phase-5 repeated performance scaling | At least four surface sizes with repeated timing `mean/std/min/max`, problem-size metadata, and conditional speedup rows. | Implemented in `validation/run_phase5_paper_experiments.py`; quick mode uses four reduced sizes and repeated runs. | Quick mode cannot support the final acceleration claim; full non-quick timing must be regenerated before final paper tables. |
| Phase-5 material-space SDF baseline | Frozen material-space SDF baseline compared against current-surface dynamic SDF under stretch and shear, with gap and normal-angle errors. | Implemented as validation-only baseline output, not as the main method. | Baseline is intentionally simple and labeled; it is not a replacement for FEM-induced dynamic SDF. |
| Phase-5 analytic linear elasticity reference | Affine uniaxial linear-elastic energy compared to analytic small-strain value over multiple TET4 resolutions. | Implemented in `phase5_external_reference_validation.csv`. | Linear small-strain analytic reference only. |
| Phase-5 brute-force CPP contact reference | Spatial-hash candidate dynamic SDF compared against all-face brute-force closest-point projection on a higher-resolution surface. | Implemented in `phase5_contact_reference.csv`. | Reference is deterministic and geometry-local; not an external experiment. |
| Phase-5 paper figure generation | PNG and PDF plots for mesh trend, contact history, performance scaling, speedup, and material-space SDF error comparison. | Implemented by the Phase-5 runner. | Plot styling is utilitarian and intended for review/preparation. |
| Phase-5 claim gates | Final acceleration claim requires repeated non-quick timing with speedup evidence; convergence-order claim requires supported regression; material-space comparison claim requires baseline data. | Covered by `tests/test_phase5_paper_experiments.py`. | Gates prevent unsupported claims but do not create missing physics evidence. |

## Phase-7 Physical Validation Table

| Case | Expected quantities | Current status | Limitations |
| --- | --- | --- | --- |
| Analytic stress/strain patch | Element strain and stress recovered from the TET4 `B` matrix match analytic affine small-strain linear elasticity. | Implemented in `validation/run_phase7_physical_validation.py`; covered by `tests/test_phase7_physical_validation.py`. | Affine linear patch only; not nonlinear stress validation. |
| Standard FEM cantilever reference | Tip displacement, von Mises stress metrics, nodal-averaged interpolated 3D boundary-surface stress plots, and projected side-surface stress plots are reported over mesh refinement and compared to the finest internal TET4 FEM reference. | Implemented in Phase 7. | This is a standard internal FEM reference, not Abaqus or an external commercial solver; smoothed stress plots are visualization only. |
| Contact force-displacement reference | Dynamic SDF penalty contact force-displacement curve is compared against brute-force all-triangle closest-point projection contact. | Implemented in Phase 7. | Rigid-plane normal penalty contact only; no friction or self-contact. |
| Acceleration feasibility | Reports contact-only speedup and total-step timing including FEM assembly, linear solve, and contact work. | Implemented in Phase 7. | Timing is still hardware-dependent and should be reported with metadata. |
| Phase-7 claim gates | Stress/strain correctness, contact force reference correctness, and total-step acceleration feasibility must have backing CSV evidence. | Covered by Phase-7 tests. | Gates do not validate unsupported physics such as nonlinear FEM or friction. |

## External Open-Source FEM Comparison Table

| Case | Expected quantities | Current status | Limitations |
| --- | --- | --- | --- |
| scikit-fem cantilever reference | Same structured TET4 mesh, material constants, fixed boundary, and nodal tip load are solved by SFC and scikit-fem. Stiffness, displacement, strain, stress, von Mises stress, and projected stress-cloud errors are reported. | Implemented in `validation/run_external_fem_comparison.py`; covered by `tests/test_external_fem_comparison.py`. | External reference is linear TET4 scikit-fem only; not Abaqus, CalculiX, FEniCS, nonlinear FEM, or an industrial contact solver. |

## Phase-8 Engineering Validation Table

| Case | Expected quantities | Current status | Limitations |
| --- | --- | --- | --- |
| 3D cantilever beam external FEM | Beam displacement, stiffness, strain, stress, and von Mises stress are compared against scikit-fem on matching 3D structured TET4 beam meshes; 3D boundary-surface stress plots show SFC, external FEM, and absolute error. | Implemented in `validation/run_phase8_engineering_cases.py`; covered by `tests/test_phase8_engineering_cases.py`. | Linear TET4 beam only; not nonlinear bending, experimental data, or commercial FEM. |
| Rigid flat indenter pressing elastic block | Prescribed indentation, min gap, max penetration, active contact count, normal force, contact energy, action-reaction imbalance, block displacement, and 3D stress cloud are reported. | Implemented in Phase 8. | Validation-only prescribed-displacement contact load; the indenter is a flat rigid underside, not a Hertzian curved punch or nonlinear contact solve. |
| Deformable-deformable block contact | Prescribed approach, min gap, penetration, active contact count, upper/lower normal forces, action-reaction imbalance, contact energy, and two-body 3D stress cloud are reported. | Implemented in Phase 8. | Frictionless normal penalty contact only; no self-contact, friction, nonlinear equilibrium iteration, or large-deformation material response. |
| Phase-8 claim gates | External FEM agreement, rigid-indenter response direction/monotonicity, and two-body action-reaction balance must have backing CSV evidence. | Covered by Phase-8 tests. | Gates support the tested engineering cases only and do not broaden the method's physics scope. |

## External Contact Solver Comparison Table

| Case | Expected quantities | Current status | Limitations |
| --- | --- | --- | --- |
| SfePy two-body contact replay | SfePy solves its built-in two-body nonlinear penalty contact example. SFC replays the final deformed geometry and reports current-surface gap sign, gap magnitude, active contact count, and agreement with SfePy exported contact gaps. | Implemented in `validation/run_external_contact_solver_comparison.py`; covered by `tests/test_external_contact_solver_comparison.py` when SfePy is available. | This validates contact state/gap replay against an external contact solve; it is not a full SFC nonlinear contact equilibrium comparison. |
| External contact claim gates | Contact state agreement and gap-scale checks must have backing CSV evidence before paper text can cite the external contact solver comparison. | Implemented in `external_contact_solver_claims.csv`. | SfePy gap is a contact-term surface average; SFC gap is a centroid closest-point projection replay. |
| No-contact dynamic integrator benchmark | Constant-acceleration free fall and an undamped harmonic oscillator are compared against analytic solutions. Displacement, velocity, and total mechanical energy drift are reported before any contact evidence is considered. | Implemented in `validation/run_dynamic_integrator_benchmark.py`; covered by `tests/test_dynamic_integrator_benchmark.py`. | This validates only the no-contact Newmark average-acceleration path; HHT/contact dynamics remain separate evidence. |
| Gentle CalculiX contact reference | A low-speed `block_drop` case can be run for 3 seconds with no gravity and configurable contact stiffness to avoid the strong-impact convergence failures seen in the earlier diagnostic. | Supported by options in `validation/run_calculix_drop_impact_comparison.py` (`--case`, `--initial-velocity-z`, `--gravity`, `--contact-stiffness`). | This is a candidate external reference; it should not support a paper claim unless CalculiX completes and the generated claim gate is supported. |

## Required Numerical Checks

- TET4 volume, mass conservation, stiffness symmetry, and rigid translation
  null-energy tests.
- Boundary face extraction on one-tet and shared-face two-tet meshes.
- Closest point projection on triangle face, edge, and vertex regions.
- Dynamic surface local sign convention: outward-oriented closed TET4 surfaces
  are outside positive and inside negative.
- Boundary orientation invariance under reversed TET4 input.
- Contact gap Jacobian finite-difference check on axis-aligned and rotated
  triangles, including multi-node slave samples and master perturbations.
- TET4 rigid translations, infinitesimal rigid rotations, and stiffness
  positive semidefiniteness.
- No core imports or references to legacy solver inputs such as `.mat`, `.odb`,
  `.xlsx`, `loadmat`, `read_excel`, or absolute Windows paths.

## Benchmark Measurements

The benchmark scripts measure:

- FEM assembly time: global `K`, `M`, and gravity vector.
- Broad phase time: padded triangle AABB construction, hash insertion, and point
  candidate queries.
- Broad phase build/update time: padded triangle AABB construction and hash
  insertion.
- Candidate query time: point queries against a prebuilt spatial hash.
- Local projection time: closest point projection over precomputed candidates.
- Dynamic SDF query time: oriented local SDF queries over precomputed candidates.
- Contact Jacobian assembly time from precomputed constraints.
- Penalty force assembly time from precomputed constraints and Jacobian.
- Full contact pipeline end-to-end time.
- Linear solve time: deterministic sparse linear solve on a preassembled small
  FEM patch.

## Method Comparison

`benchmarks/compare_bvh_vs_dynamic_sdf.py` compares:

- Brute-force all-triangle projection.
- Spatial-hash candidate query only.
- Spatial-hash candidates plus local projection.
- Dynamic FEM-induced SDF full detection pipeline on current surface geometry.
- Material-space SDF baseline, marked unavailable until implemented.

The current broad phase is a uniform spatial hash rather than a BVH. A future
BVH implementation should use the same output schema so historical CSV files
remain comparable.

## Outputs

Both benchmark scripts write:

- CSV tables for automated comparisons.
- Markdown summaries for review.
- Optional matplotlib plots when `--plot` is provided and matplotlib is
  installed.

Default outputs go to `results/benchmarks/`.

## Expected Generated Artifacts

When Phase-3 validation commands are run with `--out-dir results/benchmarks`,
the expected generated files are:

- `results/benchmarks/contact_benchmark.csv`
- `results/benchmarks/contact_benchmark.md`
- `results/benchmarks/contact_method_comparison.csv`
- `results/benchmarks/contact_method_comparison.md`

When the Phase-3 validation runner is run with `--out-dir results/validation`,
the expected generated files are:

- `results/validation/phase3_validation.csv`
- `results/validation/phase3_validation.md`

When the Phase-4 validation runner is run with `--out-dir results/phase4`,
the expected generated files are:

- `results/phase4/phase4_mesh_resolution.csv`
- `results/phase4/phase4_contact_time_history.csv`
- `results/phase4/phase4_performance_scaling.csv`
- `results/phase4/phase4_claims.csv`
- `results/phase4/phase4_summary.md`

When the Phase-5 paper experiment runner is run with
`--out-dir results/phase5`, the expected generated files are:

- `results/phase5/phase5_metadata.csv`
- `results/phase5/phase5_mesh_resolution.csv`
- `results/phase5/phase5_contact_time_history.csv`
- `results/phase5/phase5_performance_scaling.csv`
- `results/phase5/phase5_material_space_sdf_baseline.csv`
- `results/phase5/phase5_external_reference_validation.csv`
- `results/phase5/phase5_contact_reference.csv`
- `results/phase5/phase5_claims.csv`
- `results/phase5/phase5_plots.csv`
- `results/phase5/phase5_summary.md`
- `results/phase5/phase5_mesh_resolution_trend.png`
- `results/phase5/phase5_mesh_resolution_trend.pdf`
- `results/phase5/phase5_contact_time_history.png`
- `results/phase5/phase5_contact_time_history.pdf`
- `results/phase5/phase5_performance_scaling.png`
- `results/phase5/phase5_performance_scaling.pdf`
- `results/phase5/phase5_speedup.png`
- `results/phase5/phase5_speedup.pdf`
- `results/phase5/phase5_material_space_sdf_error.png`
- `results/phase5/phase5_material_space_sdf_error.pdf`

When the Phase-7 physical validation runner is run with
`--out-dir results/phase7`, the expected generated files are:

- `results/phase7/phase7_stress_strain.csv`
- `results/phase7/phase7_cantilever_reference.csv`
- `results/phase7/phase7_cantilever_stress_cloud.csv`
- `results/phase7/phase7_contact_reference.csv`
- `results/phase7/phase7_acceleration_feasibility.csv`
- `results/phase7/phase7_claims.csv`
- `results/phase7/phase7_plots.csv`
- `results/phase7/phase7_summary.md`
- `results/phase7/phase7_stress_strain_error.png`
- `results/phase7/phase7_stress_strain_error.pdf`
- `results/phase7/phase7_cantilever_fem_reference.png`
- `results/phase7/phase7_cantilever_fem_reference.pdf`
- `results/phase7/phase7_cantilever_stress_cloud.png`
- `results/phase7/phase7_cantilever_stress_cloud.pdf`
- `results/phase7/phase7_cantilever_stress_3d.png`
- `results/phase7/phase7_cantilever_stress_3d.pdf`
- `results/phase7/phase7_contact_force_displacement.png`
- `results/phase7/phase7_contact_force_displacement.pdf`
- `results/phase7/phase7_acceleration_feasibility.png`
- `results/phase7/phase7_acceleration_feasibility.pdf`

When the external FEM comparison runner is run with
`--out-dir results/external_fem`, the expected generated files are:

- `results/external_fem/external_fem_comparison.csv`
- `results/external_fem/external_fem_stress_cloud.csv`
- `results/external_fem/external_fem_plots.csv`
- `results/external_fem/external_fem_summary.md`
- `results/external_fem/external_fem_error_trends.png`
- `results/external_fem/external_fem_error_trends.pdf`
- `results/external_fem/external_fem_stress_cloud.png`
- `results/external_fem/external_fem_stress_cloud.pdf`
- `results/external_fem/external_fem_stress_3d.png`
- `results/external_fem/external_fem_stress_3d.pdf`

These files are generated evidence, not source-of-truth documentation. Missing
files should be reported as pending generated outputs rather than treated as a
method failure unless the corresponding command fails.

## CI Smoke Benchmark

The quick benchmark path must stay deterministic and small:

```bash
python benchmarks/run_contact_benchmark.py --quick
python benchmarks/compare_bvh_vs_dynamic_sdf.py --quick
```

The pytest suite includes a smoke test that runs the quick contact benchmark in
a temporary directory and checks that CSV and Markdown outputs are produced.

## Phase-3 Limitations

- Validation currently supports a minimal standalone FEM-SDF prototype only.
- No paper-experiment-ready claim is made.
- No nonlinear FEM, frictional contact, self-contact, or production BVH is
  validated.
- The dynamic SDF validation is local and surface-induced from current FEM
  geometry; it is not a neural, POD, or data-trained SDF method.
- Abaqus data may be used later only as optional external validation reference,
  not as a core solver dependency.

## Phase-4 Limitations

- Phase 4 provides paper-scale evidence infrastructure and reduced quick-mode
  evidence, not final publication-grade repeated experiments.
- Mesh-resolution results report trends and relative-to-finest displacement
  differences only; theoretical convergence order is not claimed.
- Contact histories exercise penalty normal contact only.
- Performance scaling reports speedup only for measured rows with
  `speedup_vs_bruteforce > 1`; rows without speedup are explicitly marked as
  non-acceleration evidence.
- Material-space SDF baseline remains unavailable unless implemented later as a
  separate, clearly labeled comparison.

## Phase-5 Limitations

- Phase 5 is paper experiment preparation, not a substitute for final repeated
  production runs on the target paper hardware.
- Quick mode is a CI-scale run and cannot support the final acceleration claim.
- Convergence-order claims remain blocked unless the regression gate has at
  least three nonzero error points and passes its quality threshold.
- Material-space SDF baseline data is validation-only and clearly separated
  from the main dynamic FEM-induced SDF method.
- External Abaqus validation is not used; Abaqus remains optional and outside
  the core solver.

## Phase-7 Limitations

- The stress/strain verification covers affine small-strain linear elasticity.
- The cantilever comparison uses the finest internal TET4 run as a standard FEM
  reference, not an external solver.
- Contact force-displacement validation covers frictionless rigid-plane normal
  penalty contact.
- Total-step acceleration feasibility is reported separately from contact-only
  acceleration; paper text must not conflate the two.

## External FEM Comparison Limitations

- The external comparison uses scikit-fem as a validation-only open-source FEM
  reference and does not add scikit-fem to `src/sfc`.
- The compared model is a linear TET4 cantilever with matching mesh, material,
  fixed boundary, and nodal tip load.
- Stress and strain are compared as element quantities derived from each
  solver's displacement field on the same TET4 mesh.
- Stress plots are nodal-averaged interpolated visualizations; quantitative
  comparison gates use element stress data.
- This does not validate nonlinear FEM, frictional contact, self-contact,
  barrier contact, or production BVH/contact performance.
