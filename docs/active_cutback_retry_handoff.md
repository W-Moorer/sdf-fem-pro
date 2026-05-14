# Active Cutback Retry Handoff

## Purpose

This update changes the validation-only CalculiX-style contact alignment path
from passive cutback diagnostics to an active trial-step retry loop.

It remains a clean-room implementation. The local WSL CalculiX source was used
to understand the structure:

- `dyna.c` decreases the time increment and recomputes loads when an increment
  is too large;
- `checkconvergence.c` treats residual growth and contact-element active-set
  patterns as convergence/cutback signals;
- `gencontelem_f2f.f` considers previous contact elements during cutback
  handling.

No CalculiX source code was copied.

## Implemented Behavior

In `validation/run_geometric_nonlinear_contact_validation.py`, persistent
contact modes now use:

1. take a rollback snapshot of the contact lifecycle;
2. take a rollback snapshot of the contact convergence heuristic;
3. run a trial HHT/Newmark step;
4. evaluate generated contact count and residual trend;
5. if cutback is recommended, reject the trial;
6. restore lifecycle and heuristic state;
7. halve `dt`;
8. mark the next attempt as a cutback retry;
9. accept when the diagnostic stabilizes or retry/minimum-step guards are hit.

The history CSV now includes:

- `accepted_dt`
- `nominal_dt`
- `cutback_retry_count`
- `cutback_limited`
- `generated_contact_spring_count`
- `contact_cutback_recommended`
- `contact_convergence_reason`

## Implementation Files

- `validation/calculix_f2f_contact.py`
  - lifecycle snapshot/restore;
  - persistent cutback retry flag;
  - residual-growth threshold guard.
- `validation/run_geometric_nonlinear_contact_validation.py`
  - active trial-step rollback/retry loop;
  - variable accepted `dt`;
  - cutback retry fields in history output.
- `tests/test_calculix_f2f_contact.py`
  - lifecycle rollback test.
- `tests/test_geometric_nonlinear_contact_validation.py`
  - active retry fields and reduced accepted `dt` check.

## Smoke Command

```bash
python validation/run_geometric_nonlinear_contact_validation.py --quick --skip-calculix --contact-mode persistent_calculix_c3d4_f2f --out-dir results/geometric_nonlinear_contact_validation_active_cutback_smoke
```

Smoke output:

| Metric | Value |
| --- | ---: |
| SFC history rows | `101` |
| minimum accepted dt | `1.250000e-04` |
| max retry count | `4` |
| rows with retry count > 0 | `52` |
| cutback-limited rows | `0` |
| max generated contact springs | `2` |
| max active contact samples | `2` |

This confirms that cutback recommendations now participate in time integration
by reducing accepted time steps.

## Remaining Limits

- The retry rule is a clean-room halving strategy, not a full CalculiX automatic
  increment-control clone.
- The persistent spring lifecycle remains a validation alignment layer, not a
  copied CalculiX contact-element implementation.
- The external CalculiX comparison should be rerun after this change before
  judging whether RF/CELS/CNUM/max-penetration agreement improved.
