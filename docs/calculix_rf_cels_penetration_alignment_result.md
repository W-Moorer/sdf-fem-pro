# CalculiX RF/CELS/Penetration Alignment Result

## Scope

This update continues the CalculiX contact-enforcement alignment after the
`CNUM` definition was fixed. The goal is to separate remaining RF, CELS, and
max-penetration differences into:

- contact normal / shell master offset;
- pressure-stiffness and area weighting;
- CalculiX contact output definitions;
- increment acceptance after contact activation.

No core solver code under `src/sfc` was modified.

## Implementation

Changed validation behavior:

- Added `geometric_contact_alignment_diagnostics.csv`.
- Replayed the SFC hard-linear area-weighted contact law on CalculiX
  displacement fields.
- Parsed CalculiX stdout for increment attempts, increment size, contact spring
  element counts, no-convergence messages, and contact stiffness restore events.
- Added `--cutback-policy`:
  - `calculix_direct`: fixed direct increments matching the generated
    CalculiX `*DYNAMIC, DIRECT` input.
  - `active_retry`: previous SFC active-set retry/cutback diagnostic mode.
- Changed the geometric contact validation default to
  `--cutback-policy calculix_direct`.

## Commands

Active-retry diagnostic:

```bash
python validation/run_geometric_nonlinear_contact_validation.py --contact-mode persistent_calculix_c3d4_f2f --cutback-policy active_retry --out-dir results/geometric_nonlinear_contact_validation_alignment_diagnostics
```

CalculiX-direct alignment run:

```bash
python validation/run_geometric_nonlinear_contact_validation.py --contact-mode persistent_calculix_c3d4_f2f --cutback-policy calculix_direct --out-dir results/geometric_nonlinear_contact_validation_calculix_direct
```

## Main Finding

Replaying the current SFC hard-linear area-weighted contact law on the
CalculiX displacement fields matches CalculiX peak RF and peak CELS closely:

| Resolution | Peak RF replay rel. error | Peak CELS replay rel. error |
| ---: | ---: | ---: |
| r1 | `1.730241e-03` | `4.503437e-03` |
| r2 | `6.427561e-03` | `5.888181e-03` |
| r3 | `3.229489e-03` | `1.833210e-03` |

This means the dominant remaining RF/CELS/max-penetration differences are not
caused by the pressure law, current-face area weighting, shell master offset,
or contact output definition. They are driven mainly by the post-activation
trajectory difference between SFC and CalculiX.

## Increment Acceptance

CalculiX stdout for this generated input reports:

| Resolution | Max CalculiX attempt | Min CalculiX increment | kscale restores |
| ---: | ---: | ---: | ---: |
| r1 | `1` | `2.000000e-03` | `0` |
| r2 | `1` | `2.000000e-03` | `0` |
| r3 | `1` | `2.000000e-03` | `0` |

The previous active-retry SFC mode reduced accepted steps to `1.25e-04` in
many rows. That does not match the generated CalculiX `*DYNAMIC, DIRECT`
input. The default validation policy is therefore now `calculix_direct`.

## Active Retry vs CalculiX Direct

Relative errors for active retry vs fixed direct increments:

| Resolution | Metric | Active retry | CalculiX direct | Change |
| ---: | --- | ---: | ---: | ---: |
| r1 | max penetration | `5.331147e-02` | `4.817436e-02` | improved |
| r1 | peak RF | `6.247805e-02` | `5.722847e-02` | improved |
| r1 | peak CELS | `1.222185e-01` | `1.112276e-01` | improved |
| r2 | max penetration | `1.707939e-01` | `1.600986e-01` | improved |
| r2 | peak RF | `1.507992e-01` | `1.238871e-01` | improved |
| r2 | peak CELS | `3.215809e-01` | `2.883060e-01` | improved |
| r3 | max penetration | `1.471759e-01` | `1.505151e-01` | worsened |
| r3 | peak RF | `1.485869e-01` | `1.880409e-01` | worsened |
| r3 | peak CELS | `3.272825e-01` | `3.998027e-01` | worsened |

The direct policy is more faithful to the CalculiX input, but it is not a
monotone numerical improvement across all mesh resolutions.

## Interpretation

The current evidence supports:

- shell master offset is applied consistently (`0.005`);
- pressure stiffness is not being rescaled by `kscale` in this run;
- CNUM-equivalent counting is aligned;
- peak RF/CELS output definitions are aligned when replayed on the same
  CalculiX geometry;
- remaining differences are primarily trajectory differences after contact
  activation.

The next likely sources are:

- residual/tangent convention differences inside the nonlinear dynamic step;
- exact HHT residual evaluation point and static residual history;
- contact normal and master projection update details for non-planar cases;
- CalculiX internal contact stabilization branches not reproduced in the
  clean-room validation layer.

## Tests

Targeted tests:

```text
pytest -q tests/test_geometric_nonlinear_contact_validation.py
7 passed
```

Full-suite status is recorded in the final task handoff.
