# CalculiX C3D4 F2F External Comparison Result

## Current Command

```bash
python validation/run_geometric_nonlinear_contact_validation.py --quick --contact-mode persistent_calculix_c3d4_f2f --cutback-policy calculix_direct --out-dir results/calculix_rank_aware_dominant_quick
```

The latest quick alignment run completed with CalculiX available and wrote the
standard geometric nonlinear contact validation CSV/Markdown artifacts under:

```text
results/calculix_rank_aware_dominant_quick/
```

The full Python test suite also passed:

```text
189 passed in 183.98s
```

## Source-Level Alignment Updates

The current validation backend is now aligned with the following CalculiX C3D4
behaviors:

- C3D4 direct-dynamics mass uses the one-point centroid integration form
  `rho * V / 16 * 1 1^T`, matching the observed `e_c3d.f` path. This mass is
  translationally correct but rank deficient.
- Initial acceleration uses the CalculiX-style regularized solve
  `M + beta * (dt / 10)^2 * (1 + alpha) * K`, matching the `nonlingeo.c`
  initialization path.
- Persistent face-to-face contact springs store projection, normal, clearance,
  and lifecycle state instead of rebuilding all contact samples from scratch.
- Contact spring area is taken from the increment-start slave face area, not
  from the current Newton iterate.
- CONTACT PRINT rows are mapped through the active quadrature subset after
  released springs are compacted in CalculiX output.
- CNUM comparison uses CalculiX-equivalent generated spring count instead of
  raw SFC active sample count.
- CDIS-based gap comparison is restricted to active-contact times; geometric
  surface-centroid gap is reported separately.
- One-step diagnostics are rank-aware: raw nodal acceleration differences are
  recorded, but C3D4 one-point mass nullspace acceleration is not treated as a
  unique force-balance variable.

## Latest Quick Metrics

| Metric | CalculiX | SFC | Relative / absolute error |
| --- | ---: | ---: | ---: |
| contact activation time | `0.096` | `0.09600000000000007` | `6.94e-17` abs |
| max penetration | `2.855968e-3` | `2.855378e-3` | `2.07e-4` rel |
| peak normal force / RF | `3.221065e1` | `3.223240e1` | `6.75e-4` rel |
| peak contact energy / CELS | `4.003109e-2` | `4.008497e-2` | `1.35e-3` rel |
| max CNUM | `14` | `14` | `0` abs |
| center-of-mass z L2 | - | - | `3.98e-6` rel |
| active CDIS/min-gap L2 | - | - | `8.87e-4` rel |
| surface-centroid min-gap L2 | - | - | `4.72e-3` rel |
| contact-zone max von Mises | `3.790412e1` | `3.789591e1` | `2.17e-4` rel |

## Lifecycle Check

The generated contact lifecycle sequence is now aligned in the quick case:

```text
CalculiX CNUM: 0->14->12->14->12
SFC CNUM-equivalent: 0->14->12->14->12
Replay CNUM-equivalent: 0->14->12->14->12
```

Force and energy peak times also match within floating-point roundoff:

```text
force peak: CalculiX 0.112, SFC 0.11200000000000009
energy peak: CalculiX 0.112, SFC 0.11200000000000009
```

## Remaining Differences

The remaining quick-mode differences are small but not zero:

- force-law replay mean relative error: about `6.62e-3`;
- force-law replay max relative error: about `9.46e-3`;
- CELS replay mean relative error: about `1.02e-2`;
- CELS replay max relative error: about `1.37e-2`;
- one-step HHT effective residual at CalculiX states remains about
  `2e-3` to `4e-3` relative during active contact.

The dominant meaningful one-step difference is now reported as static residual
or contact/static-force balance. The raw acceleration difference is still
stored, but it is explicitly labeled:

```text
rank_deficient_mass_nullspace_not_a_unique_balance_variable
```

This is expected because CalculiX C3D4 one-point mass is rank deficient. A raw
nodal acceleration vector is not uniquely determined by `M a = r`; only the
mass-projected term `M a` is directly comparable.

## Current Interpretation

The large earlier mismatch was primarily caused by backend and diagnostic
non-equivalence, not by the dynamic SDF contact query itself. The main corrected
items were:

1. standard TET4 consistent mass vs. CalculiX one-point C3D4 mass;
2. pre-contact sample-gap convention vs. active CDIS gap comparison;
3. current-area spring weights vs. increment-start spring areas;
4. fixed raw CONTACT PRINT row mapping vs. active-subset row compaction;
5. treating rank-deficient acceleration nullspace as a physical dominant error.

The current quick case is therefore close enough for scoped CalculiX-aligned
validation of the clean-room backend, but it is still not a proof of bitwise or
source-level identity with all private CalculiX branches. The residual
sub-percent differences likely come from CONTACT PRINT/RF/CELS output
definitions, exact residual assembly details, and CalculiX internal contact
history/stabilization branches that are not fully exposed in `.dat` output.

