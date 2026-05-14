# Contact Enforcement Diagnostics Handoff

## Purpose

This diagnostic decomposes the remaining SFC/CalculiX block-plane contact
differences into separate enforcement terms:

- active-set definition;
- contact area weighting;
- pressure-overclosure law;
- contact tangent sign convention;
- CalculiX `RF` / `CONTACT PRINT` output definitions.

It does not change the core solver. It is an evidence layer for explaining why
SFC and CalculiX contact histories can differ even after backend mechanics and
SDF queries are aligned.

## Commands

Internal quick diagnostics:

```bash
python validation/run_contact_enforcement_diagnostics.py --quick --skip-calculix --out-dir results/contact_enforcement_diagnostics
```

External quick diagnostics with CalculiX output parsing:

```bash
python validation/run_contact_enforcement_diagnostics.py --quick --out-dir results/contact_enforcement_diagnostics_external_quick
```

## Output Files

- `contact_enforcement_active_set.csv`
- `contact_enforcement_area_weighting.csv`
- `contact_enforcement_pressure_law.csv`
- `contact_enforcement_tangent_fd.csv`
- `contact_enforcement_calculix_outputs.csv`
- `contact_enforcement_claims.csv`
- `contact_enforcement_summary.md`

## Key Internal Diagnostics

For the same prescribed block-plane penetration state:

| Definition | Active Count | Max Penetration |
| --- | ---: | ---: |
| surface node gap | `4` | `4.000000e-03` |
| face centroid gap | `2` | `4.000000e-03` |
| three-point face quadrature gap | `6` | `4.000000e-03` |
| smooth pressure active lambda | `6` | `4.000000e-03` |

This shows why active counts are not directly comparable unless the active-set
definition is the same.

## Area Weighting

| Definition | Normal Force | Contact Energy |
| --- | ---: | ---: |
| three-point quadrature smooth area-weighted | `4.209625e+01` | `7.670835e-02` |
| face-centroid hard linear area-weighted | `5.120000e+01` | `1.024000e-01` |
| three-point quadrature hard linear area-weighted | `5.120000e+01` | `1.024000e-01` |
| surface-node hard linear unweighted | `3.200000e+02` | `6.400000e-01` |

The unweighted nodal variant is much larger. This supports keeping face-area
weighting in the SFC diagnostic contact path.

## Pressure-Overclosure Law

The smooth non-attractive pressure law lowers force and energy near zero
overclosure compared with hard linear penalty. Example values with
`epsilon = 2.5e-03` and stiffness `2.0e04`:

| Overclosure | Smooth Lambda | Hard Lambda |
| ---: | ---: | ---: |
| `6.250000e-04` | `7.224739e+00` | `1.250000e+01` |
| `2.500000e-03` | `3.750000e+01` | `5.000000e+01` |
| `5.000000e-03` | `8.524164e+01` | `1.000000e+02` |
| `1.000000e-02` | `1.844042e+02` | `2.000000e+02` |

This explains part of the force/energy difference against a pure hard-linear
penalty reference.

## Contact Tangent

The finite-difference tangent check passes:

| Metric | Value |
| --- | ---: |
| force derivative norm | `2.040738e+03` |
| residual tangent action norm | `2.040738e+03` |
| relative error | `3.264947e-10` |

The tangent uses the residual convention:

```text
K_contact ~= -d(f_contact)/du
```

## CalculiX Output Definition

The external quick diagnostic parsed CalculiX 2.17 output. The parser sees:

- fixed-floor `RF` total reaction;
- `CONTACT PRINT` contact spring energy;
- `CONTACT PRINT` contact element count.

Maximum parsed values in the quick run:

| Field | Maximum |
| --- | ---: |
| contact count | `14` |
| normal force from floor RF | `3.221065e+01` |
| contact spring energy | `4.003109e-02` |

These are output definitions, not one-to-one equivalents of SFC quadrature
sample count or SFC assembled penalty force.

## Current Interpretation

The remaining SFC/CalculiX contact differences should be interpreted through
three concrete mismatches:

1. **Active set:** CalculiX contact-element count, SFC quadrature active count,
   face-centroid count, and node active count are different quantities.
2. **Area weighting:** unweighted nodal force is not comparable; face-area
   weighting is required.
3. **Pressure law:** the smooth non-attractive law intentionally differs from
   hard linear penalty near zero overclosure.

The contact tangent itself is internally consistent by finite difference.

## Limitations

- This diagnostic explains enforcement terms; it does not prove exact CalculiX
  surface-to-surface contact equivalence.
- CalculiX active contact generation remains persistent and solver-specific.
- The SFC diagnostic law is still a penalty normal contact law without
  friction, self-contact, augmented Lagrangian, or barrier enforcement.
