# CalculiX HHT Residual/Tangent Diagnostics

## Scope

This note documents the current residual/tangent alignment check for the
validation-only geometric nonlinear contact comparison.  It does not claim that
SFC copies CalculiX internals.  The goal is to isolate whether the remaining
RF/CELS/max-penetration differences are caused by the nonlinear dynamic
residual, contact tangent sign convention, previous static residual update, or
CalculiX contact-stabilization branches.

## CalculiX Source-Level Reading

The relevant local CalculiX source files are under `/tmp/sfc_calculix_source/src`.

- `dynamics.f` parses `*DYNAMIC`; the default HHT parameter is `alpha = -0.05`.
- `nonlingeo.c` sets `beta = (1 - alpha)^2 / 4` and `gamma = 0.5 - alpha`.
- `calcresidual.c` forms the Newton right-hand side for implicit dynamics as
  the negative of the SFC residual convention:

  `b = (1 + alpha)(f_ext - f_int) - alpha(f_ext_ini - f_int_ini) - M a`.

  SFC writes the same equation as:

  `R = M a + (1 + alpha)(f_int - f_ext) - alpha(f_int_ini - f_ext_ini)`,

  and solves `K_eff du = -R`.

- `nonlingeo.c` stores the previous accepted force vectors by copying `f` into
  `fini` and `fext` into `fextini` after convergence.  This matches the SFC
  `previous_static_residual` update when `previous_static_residual =
  f_int - f_ext - f_contact`.
- `springstiff_f2f.f` uses a hard-linear overclosure branch with contact
  stiffness proportional to slave area and `1 / kscale`.  The resulting normal
  derivative contributes the residual tangent with the same sign as SFC's
  derivative of `-f_contact`.
- `checkconvergence.c` can restore scaled contact stiffnesses when `kscale > 1`
  and contains contact energy/adaption branches for dynamic contact.  In the
  current quick validation run, the generated CalculiX input uses
  `*DYNAMIC, DIRECT`; stdout shows no kscale restore and no contact energy
  stabilization branch.

## New Diagnostic Output

The geometric nonlinear contact runner now writes:

- `geometric_contact_hht_residual_tangent.csv`

The CSV checks:

- HHT dynamic residual norm at the accepted SFC state.
- Directional finite-difference error for the effective HHT tangent.
- Directional finite-difference error for the static tangent.
- Directional finite-difference error for the contact residual tangent
  convention.
- Absolute and relative error between the accepted `previous_static_residual`
  and a recomputed static residual at the accepted state.

The quick external run was:

```bash
python validation/run_geometric_nonlinear_contact_validation.py --quick --out-dir results/geometric_contact_residual_tangent_check
```

Key row:

| Quantity | Value |
| --- | ---: |
| Probe time | `9.600000e-02` |
| Active contact count | `2` |
| Max penetration | `1.823857e-04` |
| HHT dynamic residual norm | `3.988149e-12` |
| Effective tangent FD relative error | `5.985430e-10` |
| Static tangent FD relative error | `1.771855e-10` |
| Contact residual tangent FD relative error | `4.065636e-12` |
| Previous-static update relative error | `9.079823e-12` |

## Current Interpretation

The SFC nonlinear dynamic residual/tangent is internally consistent for the
tested active-contact state.  The finite-difference errors are far below the
validation threshold, and the previous static residual update matches the
accepted end-of-step static residual.

The quick CalculiX run reports:

| CalculiX stdout diagnostic | Value |
| --- | ---: |
| Increment count | `60` |
| Max attempt | `1` |
| Minimum increment size | `2.000000e-03` |
| Cutback attempt count | `0` |
| kscale restore count | `0` |
| Contact energy stabilization count | `0` |
| Forced increment-size count | `0` |
| Max contact spring elements | `14` |

Therefore, for this run, the remaining RF/CELS/max-penetration differences are
not explained by an SFC HHT sign error, contact tangent sign error, stale
previous static residual, active retry/cutback, or a visible CalculiX
stabilization branch.  The remaining difference is more likely in the exact
trajectory-level contact-element lifecycle and output-definition details:
generated versus force-producing contact records, contact element persistence,
clearance evaluation at the beginning of the increment, and CalculiX internal
contact element state updates.

## Commands

```bash
pytest -q tests/test_geometric_nonlinear_contact_validation.py
python validation/run_geometric_nonlinear_contact_validation.py --quick --out-dir results/geometric_contact_residual_tangent_check
```

## Limitations

- The source reading is used only for clean-room alignment and diagnostics.
- The runner still does not copy CalculiX contact internals.
- The diagnostic probes one deterministic active-contact state in quick mode.
- Exact CalculiX surface-to-surface contact equivalence remains unsupported.
