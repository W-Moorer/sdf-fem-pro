# Phase-9 Full Contact Validation

This locked package records the claim-gated validation matrix for C3D8 contact
evidence.

## Reproduce

```bash
python validation/run_phase9_full_contact_validation.py --quick --out-dir results/phase9_full_contact_validation
```

## Scope

The intended short-term multi-element scope is C3D4/TET4 and C3D8. C3D10 is
not included because no C3D10 backend is registered.

## Claim Gate Meaning

- `external_correctness` is allowed only when both a native SFC result and a
  CalculiX comparison exist.
- `trajectory_equivalence` is allowed only when a native SFC trajectory is
  compared against a CalculiX trajectory.
- `efficiency` is allowed only when that case has timing/acceleration evidence.

The current package supports C3D8 linear static contact correctness. It also
contains C3D8 dynamic external replay evidence, but the dynamic replay is not
allowed to become a native trajectory-equivalence claim.
