# CalculiX Mechanics And Increment Acceptance Diagnostics

This note records the source-level mechanics/increment-acceptance findings used
to interpret SFC/CalculiX dynamic-contact trajectory differences.  The local
CalculiX source is inspected under `/tmp/sfc_calculix_source/src` as an
external reference only.

## Source-Level Findings

| Source file | Behavior relevant to alignment |
| --- | --- |
| `nonlingeo.c` | Sets HHT/Newmark parameters for `*DYNAMIC`: `beta=(1-alpha)^2/4`, `gamma=0.5-alpha`; runs the nonlinear increment/iteration loop; calls `prediction`, residual assembly, and `checkconvergence`. |
| `prediction.c` | Builds the dynamic predictor from old displacement, velocity, and acceleration using the same beta/gamma parameters. |
| `calcresidual.c` | For implicit dynamics, forms the residual with `(1+alpha)(fext-fint)-alpha(fext_n-fint_n)-M*a` plus damping when active.  SFC uses the negative residual convention but checks the same HHT evaluation point by finite differences. |
| `checkconvergence.c` | Accepts increments using residual ratios, correction ratios, contact-element changes, energy/contact stabilization, and DIRECT-vs-automatic increment rules.  This is stricter and more stateful than SFC's current relative-correction Newton stop. |

## DIRECT Increment Interpretation

The generated external input uses:

```text
*step, nlgeom, inc=1000000
*dynamic, direct, alpha=-0.05
```

For this mode, CalculiX keeps fixed user increments.  In the current quick
external run, stdout shows `max attempt = 1` and zero cutback attempts.  Lines
containing `no convergence` are per-iteration messages inside a still-accepted
increment, not failed increments.  A typical accepted increment can print:

```text
iteration 1
no convergence
iteration 2
convergence
```

Therefore, the important comparison is not simply whether stdout contains
`no convergence`; it is whether each increment eventually prints `convergence`
without a retry/cutback.

## New Evidence File

`validation/run_geometric_nonlinear_contact_validation.py` now writes:

```text
geometric_contact_mechanics_increment_acceptance.csv
```

The file records:

- CalculiX `*DYNAMIC, DIRECT` and `NLGEOM` flags parsed from the generated input.
- CalculiX alpha plus derived HHT beta/gamma.
- SFC alpha plus derived HHT beta/gamma.
- CalculiX increment count, convergence count, total Newton iterations, maximum
  iterations per increment, and mean iterations per increment from stdout.
- CalculiX cutback attempts and forced-increment-size messages.
- SFC accepted step count, maximum Newton iterations, mean Newton iterations,
  maximum Newton residual norm, and rows reaching the current iteration limit.
- A diagnostic label distinguishing fixed direct increments from automatic or
  cutback-driven acceptance.

## Current Interpretation

The contact-query diagnostics show that replaying the SFC clean-room contact
law on CalculiX displacements reduces the CNUM mismatch.  The next difference
class is mechanics/increment acceptance:

- CalculiX uses a multi-criterion `checkconvergence.c` acceptance rule.
- SFC currently accepts Newton iterations by relative correction tolerance or
  reaching its configured iteration loop.
- CalculiX stdout should be interpreted as accepted fixed increments if
  `convergence_count == increment_count`, `max_attempt == 1`, and cutback count
  is zero.

This is not an exact CalculiX equivalence claim.  It is a diagnostic boundary:
if mechanics/increment acceptance remains different after contact-query replay
is aligned, native SFC and CalculiX trajectories can still diverge even when the
contact force law is close on a prescribed trajectory.
