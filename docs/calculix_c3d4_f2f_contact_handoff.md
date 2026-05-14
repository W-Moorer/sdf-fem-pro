# CalculiX C3D4 F2F Contact Alignment Handoff

## Purpose

This change adds a validation-only contact mode that narrows the remaining SFC
vs. CalculiX difference before swapping in the dynamic SDF query. It does not
copy CalculiX source and does not change the production core contact method.

## Implemented Files

- `validation/calculix_f2f_contact.py`
  - `CalculixF2FContactSpring`
  - `CalculixC3D4FaceToFacePlaneContactGeometry`
  - `CalculixC3D4FaceToFaceSDFContactGeometry`
  - `active_contact_spring_count`
- `validation/run_geometric_nonlinear_contact_validation.py`
  - default SFC contact mode changed to `calculix_c3d4_f2f`;
  - added `--contact-mode calculix_c3d4_f2f`;
  - added `--contact-mode dynamic_sdf_calculix_f2f`;
  - retained older `plane` and `dynamic_sdf_plane` three-point quadrature modes.
- `tests/test_calculix_f2f_contact.py`
- `tests/test_geometric_nonlinear_contact_validation.py`

## Discretization

The strict validation mode uses:

- one integration point per linear C3D4/TET4 triangular slave face;
- centroid slave shape weights `(1/3, 1/3, 1/3)`;
- current triangle area as contact spring area;
- hard linear pressure-overclosure:

```text
lambda = k * area * max(-clearance, 0)
energy = 0.5 * k * area * max(-clearance, 0)^2
```

The analytic-plane and dynamic-SDF variants share the same enforcement. Only
the source of `clearance` and `normal` changes.

## Commands

Strict CalculiX-style SFC mode:

```bash
python validation/run_geometric_nonlinear_contact_validation.py --quick --skip-calculix --contact-mode calculix_c3d4_f2f --out-dir results/geometric_nonlinear_contact_validation_strict_smoke
```

Same enforcement with dynamic SDF plane query:

```bash
python validation/run_geometric_nonlinear_contact_validation.py --quick --skip-calculix --contact-mode dynamic_sdf_calculix_f2f --out-dir results/geometric_nonlinear_contact_validation_sdf_strict_smoke
```

Older non-strict three-point quadrature mode:

```bash
python validation/run_geometric_nonlinear_contact_validation.py --quick --skip-calculix --contact-mode plane --out-dir results/geometric_nonlinear_contact_validation_three_point_smoke
```

## What This Fixes

- Removes the previous three-point slave-face quadrature mismatch for the
  strict external CalculiX comparison path.
- Removes the smooth overclosure law from the strict external comparison path.
- Provides a contact-spring-element style diagnostic record with slave face,
  master face, shape weights, normal, spring area, clearance, and active flag.
- Keeps dynamic SDF isolated as a contact-query provider rather than changing
  the mechanics/contact enforcement backend.

## Remaining Non-Equivalences

- The mode is a clean-room implementation, not copied CalculiX source.
- It does not yet reproduce every CalculiX persistent active-set branch,
  cutback rule, or contact convergence heuristic.
- For deformable-deformable contact, the current backend still consumes
  slave-side `ContactSample` objects; a full master/slave spring element with
  master DOFs is a separate follow-up.
- CalculiX output definitions (`CNUM`, `CELS`, fixed-floor `RF`) still must be
  compared only against matching strict-mode diagnostics.

## Validation Status

The added tests check that:

- strict C3D4 F2F mode uses one centroid sample per slave triangle;
- strict mode differs from the older three-point quadrature sample count;
- strict dynamic-SDF plane query matches the strict analytic-plane query for a
  flat master surface;
- the geometric nonlinear contact validation runner defaults to
  `calculix_c3d4_f2f`.
