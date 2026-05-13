# Phase-7 Physical Validation Handoff

## Purpose

Phase 7 adds physical-comparison evidence requested for manuscript review. It
does not add new core physics. The validation focuses on three missing
comparison categories:

1. Stress/strain recovery against an analytic small-strain linear-elastic
   patch.
2. Standard FEM displacement/stress trends on a fixed-base cantilever block,
   using the finest internal TET4 run as the reference.
3. Contact force-displacement response against brute-force all-triangle
   closest-point projection contact.

It also separates contact-only acceleration from total-step timing that includes
FEM assembly, linear solve, and contact evaluation.

## Commands

```bash
python validation/run_phase7_physical_validation.py --out-dir results/phase7
```

The CI-scale smoke path is:

```bash
python validation/run_phase7_physical_validation.py --quick --out-dir results/phase7
```

The full test command for acceptance is:

```bash
pytest -q
```

## Output Files

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

## Comparison 1: Analytic Stress/Strain Patch

The affine uniaxial patch uses the displacement field `u_x = 0.001 x`,
`u_y = u_z = 0`. For `E = 2.0e5` and `nu = 0.25`, the analytic strain and
stress are:

- strain: `[0.001, 0, 0, 0, 0, 0]`
- stress: `[240, 80, 80, 0, 0, 0]`

| Resolution | Elements | Max strain error | Max stress error | Status |
| ---: | ---: | ---: | ---: | --- |
| 1 | 5 | 0.000000e+00 | 0.000000e+00 | ok |
| 2 | 40 | 0.000000e+00 | 0.000000e+00 | ok |
| 3 | 135 | 0.000000e+00 | 0.000000e+00 | ok |
| 4 | 320 | 0.000000e+00 | 0.000000e+00 | ok |

Interpretation: this validates the TET4 strain recovery and linear-elastic
stress computation for affine small-strain states. It does not validate
nonlinear stress, large deformation, or plasticity.

## Comparison 2: Standard FEM Cantilever Reference

The fixed-base block is solved with the same internally assembled linear TET4
FEM system over multiple mesh resolutions. The finest run is used as the
internal reference. In addition to scalar displacement and stress metrics, the
runner writes element-centroid von Mises stress data, a 3D boundary-surface
stress plot, and a projected x-z side-surface stress plot. For visualization,
the plot averages element von Mises stress to nodes, interpolates over
subdivided boundary triangles, and uses a shared color scale across resolutions.
The quantitative values remain the element stress metrics in the CSV tables.

| Resolution | Nodes | Elements | Tip displacement z | Max von Mises | Tip rel. error to finest | Max VM rel. error to finest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 5 | -1.397105e-03 | 9.796707e+01 | 8.538791e-01 | 6.884284e-01 |
| 2 | 27 | 40 | -3.686769e-03 | 1.835477e+02 | 6.144070e-01 | 4.162502e-01 |
| 3 | 64 | 135 | -6.617482e-03 | 2.557485e+02 | 3.078887e-01 | 1.866248e-01 |
| 4 | 125 | 320 | -9.561298e-03 | 3.144287e+02 | 0.000000e+00 | 0.000000e+00 |

Interpretation: this provides a conventional FEM displacement/stress trend and
shows that the code reports stress and strain quantities, not only contact
geometry. This is not an external Abaqus, CalculiX, FEniCS, or commercial-solver
comparison. The 3D boundary-surface stress plot is intended as the primary
paper-facing visual because it shows the full body boundary mesh and smoothed
stress distribution instead of only reporting scalar summary curves.

## Comparison 3: Contact Force-Displacement Reference

The dynamic FEM-induced SDF contact response is compared against brute-force
all-triangle closest-point projection contact on the same rigid-plane surface.
Both methods use the same penalty law, so this isolates the geometric contact
query.

| Penetration | Brute-force normal force | Dynamic SDF normal force | Relative error | Active counts |
| ---: | ---: | ---: | ---: | --- |
| 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | 0/0 |
| 2.500000e-03 | 8.000000e+02 | 8.000000e+02 | 0.000000e+00 | 64/64 |
| 5.000000e-03 | 1.600000e+03 | 1.600000e+03 | 0.000000e+00 | 64/64 |
| 1.000000e-02 | 3.200000e+03 | 3.200000e+03 | 0.000000e+00 | 64/64 |
| 2.000000e-02 | 6.400000e+03 | 6.400000e+03 | 0.000000e+00 | 64/64 |

Interpretation: for this rigid-plane normal penalty case, the accelerated
dynamic SDF pipeline reproduces the brute-force closest-point projection contact
force-displacement curve exactly to reported precision.

## Acceleration Feasibility

The Phase-7 timing table separates two claims:

- `contact_pipeline_only`: contact geometry and contact force pipeline only.
- `fem_assembly_linear_solve_plus_contact`: a small total-step timing including
  FEM assembly, a deterministic sparse linear solve, and contact work.

| Surface resolution | Triangles | Query points | Stage | Dynamic mean seconds | Speedup vs brute force | Status |
| ---: | ---: | ---: | --- | ---: | ---: | --- |
| 4 | 32 | 16 | contact pipeline only | 1.105893e-02 | 4.497381e+00 | speedup_supported |
| 4 | 32 | 16 | total step | 1.719027e-02 | 3.244476e+00 | speedup_supported |
| 8 | 128 | 64 | contact pipeline only | 4.234420e-02 | 1.554752e+01 | speedup_supported |
| 8 | 128 | 64 | total step | 4.900783e-02 | 1.325408e+01 | speedup_supported |
| 12 | 288 | 144 | contact pipeline only | 9.536470e-02 | 3.372208e+01 | speedup_supported |
| 12 | 288 | 144 | total step | 1.096421e-01 | 3.062775e+01 | speedup_supported |

Interpretation: the acceleration is no longer only a raw closest-point query
claim. In these tested cases, the dynamic SDF path remains faster even when FEM
assembly and linear solve are included in the timed step. This is still a
prototype timing comparison against brute-force all-triangle projection, not a
claim against production BVH, IPC, GPU collision detection, or commercial
solvers.

## Claim Gates

| Claim | Evidence | Gate value | Status |
| --- | --- | ---: | --- |
| Stress/strain physical correctness | `phase7_stress_strain.csv::max_stress_l2_error` | 0.000000e+00 | supported |
| Contact force reference correctness | `phase7_contact_reference.csv::normal_force_rel_error` | 0.000000e+00 | supported |
| Total-step acceleration feasibility | `phase7_acceleration_feasibility.csv::speedup_vs_bruteforce` | 3.244476e+00 | supported |

## Remaining Boundaries

- The stress/strain comparison is analytic and affine; it is not a nonlinear or
  experimental material validation.
- The cantilever reference is an internal standard TET4 FEM refinement study; it
  is not yet an external solver comparison.
- The contact force comparison validates frictionless normal penalty contact
  against brute-force CPP on a rigid plane.
- The acceleration evidence is against brute-force all-triangle projection. It
  should not be stated as superiority over optimized BVH, IPC, GPU, or
  commercial contact implementations.
- No friction, self-contact, nonlinear FEM, barrier contact, or GPU acceleration
  is implemented in this phase.
