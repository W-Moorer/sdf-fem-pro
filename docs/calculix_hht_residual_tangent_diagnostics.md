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

  `R = M a - (1 + alpha)B + alpha B_ini`,

  where `B = f_ext - f_int`.  This is the negative of the CalculiX right-hand
  side, and SFC solves `K_eff du = -R`.

- `nonlingeo.c` stores the previous accepted force vectors by copying `f` into
  `fini` and `fext` into `fextini` after convergence.  SFC now stores the
  accepted history in the same sign, `previous_static_residual = fextini -
  fini`.  Contact spring forces enter `fini` through their internal spring
  sign rather than as a separate external load.
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
- `geometric_contact_one_step_state_diagnostics.csv`
- `geometric_contact_hht_state_definition_diagnostics.csv`

The CSV checks:

- HHT dynamic residual norm at the accepted SFC state.
- Directional finite-difference error for the effective HHT tangent.
- Directional finite-difference error for the static tangent.
- Directional finite-difference error for the contact residual tangent
  convention.
- Absolute and relative error between the accepted `previous_static_residual`
  (`fextini - fini`) and the negative of the recomputed SFC static residual at
  the accepted state.

The one-step/state-definition CSV files additionally check:

- SFC internal force, contact force, mass term, effective residual, and tangent
  norms evaluated on CalculiX displacement states.
- One SFC Newmark/HHT step launched from the same CalculiX state and compared
  with the next CalculiX displacement state.
- CalculiX source-level state definitions for `beta`, `gamma`, prediction,
  acceleration update, and residual assembly.
- The initial acceleration regularization used by CalculiX at the start of a
  dynamic step.
- Whether six-decimal `.dat` displacement output can plausibly explain the
  acceleration/residual mismatch after reconstruction.

The quick external run was:

```bash
python validation/run_geometric_nonlinear_contact_validation.py --quick --out-dir results/geometric_contact_residual_tangent_check
```

Key residual/tangent row:

| Quantity | Value |
| --- | ---: |
| Probe time | `9.600000e-02` |
| Active contact count | `14` |
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

The new CalculiX-state one-step diagnostics narrow the trajectory mismatch
further:

- Pre-contact CalculiX displacement states satisfy the SFC HHT residual to
  roughly machine precision in the quick block-plane run.
- At contact activation, the SFC contact force evaluated on the CalculiX
  displacement state agrees with CalculiX RF within about one percent.
- The effective HHT residual evaluated on the CalculiX contact states becomes
  large, and the one-step update mismatch is dominated by the reconstructed
  acceleration/mass term.
- The `.dat` displacement precision estimate is far too small to explain the
  observed acceleration difference in the current quick run.
- The initial acceleration regularization difference between SFC's pure-mass
  initialization and CalculiX's small stiffness-regularized solve is negligible
  for the checked free-fall start.

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
previous static residual, active retry/cutback, visible CalculiX stabilization
branch, initial acceleration initialization, mass scaling, or `.dat`
displacement print precision.  The strongest current evidence points to a
contact-phase effective residual/acceleration mismatch after activation: the
same current geometry gives a similar normal contact force, but the coupled
Newmark/HHT state no longer satisfies both solvers' effective dynamic balance.

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
