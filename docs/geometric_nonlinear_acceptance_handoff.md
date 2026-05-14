# Geometric Nonlinear Acceptance Handoff

## Purpose

This handoff records the geometric nonlinear acceptance suite that must pass
before using the block-plane contact result as nonlinear evidence.

The suite is validation-only. It does not claim that the production `src/sfc`
solver is now a full CalculiX-equivalent nonlinear contact solver.

## Implemented Checks

1. No-contact finite-deformation StVK patch test.
2. No-contact large-displacement cantilever comparison against CalculiX.
3. Newton residual and correction table for the SFC nonlinear static solve.
4. Mesh-resolution trend for the no-contact cantilever at r1, r2, and r3.
5. Time-step trend for block-plane contact at dt, dt/2, and dt/4.
6. Block-plane contact acceptance only after no-contact evidence is generated.

## Commands

```bash
python validation/run_geometric_nonlinear_acceptance.py --quick --out-dir results/geometric_nonlinear_acceptance
```

Full r1/r2/r3 CalculiX comparison:

```bash
python validation/run_geometric_nonlinear_acceptance.py --out-dir results/geometric_nonlinear_acceptance_full
```

Local smoke run without CalculiX:

```bash
python validation/run_geometric_nonlinear_acceptance.py --quick --skip-calculix --out-dir results/geometric_nonlinear_acceptance_smoke
```

## Output Files

- `results/geometric_nonlinear_acceptance/stvk_patch.csv`
- `results/geometric_nonlinear_acceptance/cantilever_calculix_comparison.csv`
- `results/geometric_nonlinear_acceptance/newton_convergence.csv`
- `results/geometric_nonlinear_acceptance/mesh_convergence.csv`
- `results/geometric_nonlinear_acceptance/timestep_convergence.csv`
- `results/geometric_nonlinear_acceptance/block_plane_contact_acceptance.csv`
- `results/geometric_nonlinear_acceptance/geometric_nonlinear_acceptance_claims.csv`
- `results/geometric_nonlinear_acceptance/geometric_nonlinear_acceptance_summary.md`

## Latest Full Results

CalculiX version: `2.17`.

The no-contact cantilever comparison reports:

| Resolution | Tip disp. rel. error | Reaction rel. error | Energy rel. error | Max VM rel. error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `3.251012e-08` | `7.262801e-12` | `2.735019e-06` | `2.049159e-07` |
| 2 | `1.506273e-08` | `5.412337e-14` | `1.384917e-03` | `3.455325e-08` |
| 3 | `1.427034e-08` | `8.326673e-16` | `6.186730e-03` | `1.787182e-07` |

The block-plane contact time-step trend reports:

| dt | final z_cm | min gap | energy drift range |
| ---: | ---: | ---: | ---: |
| `2.0e-03` | `3.944292e-01` | `-2.625506e-03` | `4.406466e-05` |
| `1.0e-03` | `3.944332e-01` | `-2.636771e-03` | `4.376907e-05` |
| `5.0e-04` | `3.944342e-01` | `-2.639230e-03` | `4.369970e-05` |

The claim gate CSV reports:

| Claim | Supported |
| --- | --- |
| analytic finite-deformation StVK patch | true |
| no-contact large-displacement CalculiX comparison | true |
| mesh convergence trend available | true |
| Newton convergence table available | true |
| time-step convergence trend available | true |
| contact checked after no-contact gates | true |

## Interpretation

This is enough to say that the current validation implementation has passed the
first geometric nonlinear acceptance gate for the scoped StVK TET4 diagnostic:

- the StVK finite-strain element response matches an analytic affine reference;
- the no-contact nonlinear cantilever matches CalculiX for tip displacement,
  fixed reaction, energy estimate, and max stress on r1/r2/r3;
- Newton iteration tables are generated;
- r1/r2/r3 mesh trend and dt/dt/2/dt/4 time-step trend files exist;
- block-plane contact is not treated as accepted until these no-contact gates
  are generated.

## Limitations

- The material model is St. Venant-Kirchhoff.
- CalculiX is used as a no-contact large-displacement reference only in this
  acceptance suite.
- Mesh/time-step outputs report trends only; no theoretical convergence order
  is claimed.
- Contact is still rigid-plane penalty contact, not a full CalculiX
  surface-to-surface contact clone.
