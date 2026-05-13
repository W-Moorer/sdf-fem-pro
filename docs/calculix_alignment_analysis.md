# CalculiX Alignment Analysis

## Scope

This note records why the earlier dynamic contact plots did not align and what
was changed in the validation-only runner. It does not change the core `src/sfc`
solver.

CalculiX source inspected:

- repository: <https://github.com/Dhondtguido/CalculiX>
- local checkout commit used for inspection: `9033050`
- files checked: `src/springforc_f2f.f`, `src/dynamics.f`, `src/dyna.c`,
  `src/contactprints.f`

## Root Causes

### 1. Wrong Reported Contact Plane

The generated CalculiX model uses:

```text
*shell section, elset=efloor, material=gummi
0.01
*surface, name=floor, type=element
efloor, SPOS
```

The master surface is therefore the shell `SPOS` side, not the shell
mid-surface. With thickness `0.01`, the effective contact plane is:

```text
z_contact = floor_z + 0.005
```

The previous validation plots reported gaps to `floor_z`, which shifted the
gap and penetration diagnostics by `0.005`.

### 2. Nodal-Area Contact Was Not CalculiX Face-To-Face Contact

CalculiX `springforc_f2f.f` evaluates slave-face contact at face integration
points. For linear pressure-overclosure it computes the normal spring force
from the integration-point clearance and `springarea`. The relevant structure
is:

```text
clear = (x_slave_ip - x_master_ip) dot n_master
stiff(1) = -springarea(1) * pressure_stiffness * clear / kscale
force_ip = -stiff(1) * n_master
```

The previous SFC validation runner used per-node accumulated surface areas. That
gave a different discretization, different active contact count, and different
force distribution.

### 3. Contact Count Is Not A Comparable Physical Quantity

CalculiX reports contact elements/integration output. The SFC runner reports
active face-centroid quadrature points. These counts are diagnostic only and
should not be used as a paper-level matching metric.

### 4. Strong Impact Is Still A Bad Reference Case

The original strong sphere/block diagnostic still should not be used as
external validation evidence. CalculiX returned nonzero status in that case,
large overclosure occurred before termination, and the result is dominated by
nonlinear contact convergence rather than by the SFC dynamic SDF formulation.

## Fixes Implemented

- Added explicit constants for the CalculiX shell thickness and `SPOS` contact
  plane.
- Changed gap and penetration diagnostics to use `floor_z + 0.005`.
- Changed the SFC validation contact from nodal area weighting to one
  centroid integration point per boundary triangle.
- Scaled contact force and tangent by current triangle area.
- Distributed the face quadrature force to the three slave face nodes with
  linear triangle weights.
- Added tests proving the runner uses the CalculiX `SPOS` plane and face-based
  contact integration.

## Current Gentle Reference Result

Command:

```bash
python validation/run_calculix_drop_impact_comparison.py --quick --case block_drop --duration 1.0 --dt 0.002 --output-frequency 1 --initial-velocity-z 0.0 --gravity 9.81 --contact-stiffness 5000 --out-dir results/calculix_gentle_contact_reference
```

Result summary:

| Metric | Value |
| --- | ---: |
| CalculiX return code | `0` |
| CalculiX completed | `true` |
| first contact time, CalculiX | `9.60e-02 s` |
| first contact time, SFC | `9.60e-02 s` |
| first contact time absolute error | `0.00e+00 s` |
| z center-of-mass relative error | `1.054328e-02` |
| minimum-gap L-infinity error | `1.446052e-02` |
| CalculiX max overclosure | `1.088870e-02` |
| SFC max overclosure | `1.003876e-02` |
| max rebound z after contact, CalculiX | `4.448849e-01` |
| max rebound z after contact, SFC | `4.439993e-01` |
| physical rebound height claim | `supported` |

The latest strong-impact sphere diagnostic is intentionally not accepted:
`sphere_like_drop r2` uses a finer mesh and `dt=0.001`, and both CalculiX and
SFC stay below the ballistic rebound-height bound implied by the initial
downward velocity. However, CalculiX still times out before 1 second and its
log reports a large energy-balance error, so this is treated as a rejected
contact-dynamics setup rather than evidence for the method.

## Remaining Differences

- CalculiX uses its full nonlinear contact search, contact element generation,
  and solver convergence logic. The SFC validation runner uses a simplified
  face-centroid plane contact approximation.
- CalculiX and SFC still have different active contact counts because the
  reported quantities are not the same.
- HHT-alpha dynamics is numerically dissipative and should not be described as
  energy-conserving contact dynamics.
- The strong impact case remains unsupported until a more robust external
  contact setup is designed.
