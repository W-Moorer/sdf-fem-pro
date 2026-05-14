# CalculiX-Aligned Backend Validation Handoff

## Purpose

This validation checks backend mechanics calculations independently of contact
detection. It exercises the clean-room backend in `src/sfc/fem/calculix_aligned.py`
before using it for SDF/contact comparisons.

The validation covers:

- consistent TET4 mass;
- affine StVK strain/stress/energy;
- material plus geometric tangent finite-difference agreement;
- no-contact HHT/Newmark freefall against analytic constant acceleration;
- no-contact nonlinear cantilever Newton convergence;
- optional CalculiX static cantilever comparison when `ccx` is available.

## Commands

Quick internal run:

```bash
python validation/run_calculix_aligned_backend_validation.py --quick --skip-calculix --out-dir results/calculix_aligned_backend_validation
```

Full internal r1/r2/r3 run:

```bash
python validation/run_calculix_aligned_backend_validation.py --skip-calculix --out-dir results/calculix_aligned_backend_validation_full
```

Quick external CalculiX run:

```bash
python validation/run_calculix_aligned_backend_validation.py --quick --out-dir results/calculix_aligned_backend_validation_external_quick
```

## Output Files

- `backend_mass_matrix.csv`
- `backend_stvk_patch.csv`
- `backend_tangent_fd.csv`
- `backend_freefall_dynamics.csv`
- `backend_cantilever_newton.csv`
- `backend_newton_iterations.csv`
- `backend_external_commands.csv`
- `backend_alignment_claims.csv`
- `backend_alignment_summary.md`

## Latest Full Internal Result

All backend claim gates are supported:

- `consistent_mass_backend_check`
- `stvk_affine_patch_backend_check`
- `stvk_tangent_finite_difference_backend_check`
- `hht_newmark_freefall_backend_check`
- `no_contact_newton_cantilever_backend_check`

Selected metrics:

| Check | Value |
| --- | ---: |
| r3 mass component max abs. error | `2.220446e-16` |
| freefall `dt=0.001` max z error | `4.440892e-16` |
| freefall `dt=0.001` energy range | `8.881784e-16` |
| tangent finite-difference relative error | `1.080587e-10` |
| r3 cantilever final residual norm | passed gate |

## Latest Quick CalculiX Comparison

The optional quick external run completed with CalculiX 2.17 for the r1
no-contact cantilever.

| Metric | Relative Error |
| --- | ---: |
| tip displacement z | `3.251012e-08` |
| fixed reaction z | `3.705369e-14` |
| max von Mises stress | `2.049336e-07` |

This supports the backend mechanics implementation for the scoped no-contact
static cantilever case. It does not validate CalculiX contact equivalence.

## Limitations

- This is a clean-room implementation using public finite-element formulas, not
  copied CalculiX source.
- The constitutive model is St. Venant-Kirchhoff in the validation backend.
- Optional CalculiX comparison is no-contact static cantilever only.
- Contact detection, active contact generation, and SDF query behavior are
  validated by separate runners.
