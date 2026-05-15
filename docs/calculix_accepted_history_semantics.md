# CalculiX Accepted History Semantics

This note records the clean-room source reading used for the geometric nonlinear
contact alignment diagnostics.  It does not copy CalculiX implementation code.

## Source-Level Semantics Checked

Local source path: `/tmp/sfc_calculix_source/src`.

Relevant files:

- `nonlingeo.c`
- `prediction.c`
- `resultsini.c`
- `iniparll.c`
- `calcresidual.c`
- `res2parll.c`

Observed semantics:

- At the start of a new accepted increment, CalculiX snapshots the previous
  converged state into `vini`, `fini`, and, for implicit dynamics, `veini`,
  `accini`, `fextini`, and `cvini`.
- `prediction.c` predicts displacement and velocity from the previous accepted
  velocity and acceleration, then resets the acceleration accumulator used by
  Newton corrections.
- `resultsini.c` / `iniparll.c` treat the Newton solution vector as an
  acceleration increment: displacement, velocity, and acceleration are updated
  by the Newmark `beta dt^2`, `gamma dt`, and unit factors.
- `calcresidual.c` forms the implicit dynamic RHS from the current force
  balance, the saved `fextini - fini` history, and the mass times current
  acceleration.
- On cutback, `nonlingeo.c` restores the saved state vectors instead of
  accepting the trial state.

## SFC Mapping

The validation backend maps this to:

- `previous_static_residual = fextini - fini`
- contact spring force enters `fini` as an internal spring contribution
- `hht_step()` returns the accepted `fext - fint` balance for the next step

The accepted history is now built from the same accepted contact evaluation that
produces `StepDiagnostics`.  This avoids a second stateful contact sampling call
at the accepted-step boundary, which could otherwise advance a persistent
contact lifecycle while constructing `fextini - fini`.

## Remaining Difference

This change aligns the accepted-history save boundary.  It does not imply exact
CalculiX contact equivalence.  Remaining external differences should be checked
against:

- CalculiX contact element lifecycle and clearance history;
- exact printed RF/CELS definitions;
- accepted contact spring energy history;
- any contact stabilization branch not exposed by the current diagnostics.
