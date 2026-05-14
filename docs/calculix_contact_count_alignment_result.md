# CalculiX Contact Count Alignment Result

## Scope

This update addresses the next contact-enforcement alignment issue after active
cutback retry: separating SFC force-producing contact spring records from the
quantity CalculiX prints as `CNUM`.

No core solver code under `src/sfc` was modified. The change is limited to the
validation-only CalculiX alignment layer.

## Implementation

Changed files:

- `validation/calculix_f2f_contact.py`
- `validation/run_geometric_nonlinear_contact_validation.py`
- `tests/test_calculix_f2f_contact.py`
- `tests/test_geometric_nonlinear_contact_validation.py`
- `docs/validation_plan.md`

Main changes:

- Added `CalculixF2FContactSpring.calculix_cnum_weight`.
- Added lifecycle-level `calculix_contact_element_count`.
- Added `calculix_equivalent_contact_element_count(...)`.
- Added `calculix_equivalent_contact_count` and `sfc_floor_rf_z_equivalent`
  fields to SFC geometric contact history rows.
- Added `max_cnum_calculix`, `max_cnum_equivalent_sfc`, and
  `max_cnum_abs_error` to the external comparison CSV.

For the scoped block-plane model, one force-producing C3D4 slave face spring
against the S4 rigid-plane master contributes:

```text
nopes + nopem = 3 + 4 = 7
```

This matches the local CalculiX source convention for the generated contact
element storage used by the printed contact count.

## Command

```bash
python validation/run_geometric_nonlinear_contact_validation.py --contact-mode persistent_calculix_c3d4_f2f --out-dir results/geometric_nonlinear_contact_validation_cnum_aligned
```

## Output

- `results/geometric_nonlinear_contact_validation_cnum_aligned/geometric_contact_history.csv`
- `results/geometric_nonlinear_contact_validation_cnum_aligned/geometric_contact_calculix_comparison.csv`
- `results/geometric_nonlinear_contact_validation_cnum_aligned/geometric_contact_validation_summary.md`

## Result

| Resolution | CalculiX max CNUM | SFC CNUM-equivalent | Abs. error |
| ---: | ---: | ---: | ---: |
| r1 | `14` | `14` | `0` |
| r2 | `56` | `56` | `0` |
| r3 | `126` | `126` | `0` |

The underlying SFC force-producing spring record counts remain:

| Resolution | SFC force spring records | SFC CNUM-equivalent |
| ---: | ---: | ---: |
| r1 | `2` | `14` |
| r2 | `8` | `56` |
| r3 | `18` | `126` |

## RF / CELS / Penetration

This change intentionally does not alter the force law, contact tangent, or
accepted state trajectory. Therefore the RF, CELS, and max-penetration
differences are expected to match the previous active-cutback run:

| Resolution | Max penetration rel. error | Peak RF rel. error | Peak CELS rel. error |
| ---: | ---: | ---: | ---: |
| r1 | `5.331147e-02` | `6.247805e-02` | `1.222185e-01` |
| r2 | `1.707939e-01` | `1.507992e-01` | `3.215809e-01` |
| r3 | `1.471759e-01` | `1.485869e-01` | `3.272825e-01` |

## Interpretation

This fixes the CNUM definition mismatch. It does not claim that SFC contact
enforcement is source-level equivalent to CalculiX.

The remaining numerical differences are now narrower:

- RF still differs because the accepted SFC trajectory and CalculiX trajectory
  are not identical after contact activation.
- CELS still differs because it follows the same trajectory/contact-energy
  difference.
- Max penetration still differs for the same reason.
- Contact area weighting is explicitly current-face-area weighted in SFC and is
  covered by tests.

The next alignment target should be force/energy trajectory differences, not
CNUM. That means inspecting contact normal update, shell master offset,
pressure-stiffness scaling, and CalculiX increment acceptance around active
contact.

## Tests

Targeted tests:

```text
pytest -q tests/test_calculix_f2f_contact.py tests/test_geometric_nonlinear_contact_validation.py
14 passed
```

Full-suite status is recorded in the final task handoff.
