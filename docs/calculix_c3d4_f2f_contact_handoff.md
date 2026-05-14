# CalculiX C3D4 F2F Contact Alignment Handoff

## Purpose

This change adds a validation-only contact mode that narrows the remaining SFC
vs. CalculiX difference before swapping in the dynamic SDF query. It does not
copy CalculiX source and does not change the production core contact method.

## Implemented Files

- `validation/calculix_f2f_contact.py`
  - `CalculixF2FContactSpring`
  - `CalculixF2FContactLifecycle`
  - `CalculixContactConvergenceHeuristic`
  - `CalculixC3D4FaceToFacePlaneContactGeometry`
  - `CalculixC3D4FaceToFaceSDFContactGeometry`
  - `PersistentCalculixC3D4FaceToFacePlaneContactGeometry`
  - `PersistentCalculixC3D4FaceToFaceSDFContactGeometry`
  - `assemble_deformable_f2f_contact_response`
  - `active_contact_spring_count`
- `validation/run_geometric_nonlinear_contact_validation.py`
  - default SFC contact mode changed to `persistent_calculix_c3d4_f2f`;
  - added `--contact-mode persistent_calculix_c3d4_f2f`;
  - added `--contact-mode calculix_c3d4_f2f`;
  - added `--contact-mode persistent_dynamic_sdf_calculix_f2f`;
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

## Persistent Spring Lifecycle

The persistent modes maintain generated contact springs by slave face. Each
update records whether a spring was:

- `generated`
- `persisted`
- `cutback_persisted`
- `released`
- `inactive`
- `lost_candidate_released`

The runner now reports both force-producing active sample count and generated
contact spring count where available. This separates CalculiX-style contact
element bookkeeping from negative-clearance force activation.

## Contact Convergence / Cutback Diagnostic

`CalculixContactConvergenceHeuristic` records active-count and residual trends.
It recommends a cutback/retry when residuals grow or active-set counts
oscillate. This is a clean-room diagnostic heuristic, not a copied CalculiX
`checkconvergence.c` implementation.

## Deformable-Deformable Master DOF Assembly

`assemble_deformable_f2f_contact_response` assembles the master/slave spring
row:

```text
dg/dx_slave = N_slave n^T
dg/dx_master = -N_master n^T
f = J^T lambda
K ~= k area J^T J
```

The added test verifies action-reaction balance between slave and master DOFs.

## Commands

Strict CalculiX-style SFC mode:

```bash
python validation/run_geometric_nonlinear_contact_validation.py --quick --skip-calculix --contact-mode calculix_c3d4_f2f --out-dir results/geometric_nonlinear_contact_validation_strict_smoke
```

Persistent strict CalculiX-style SFC mode:

```bash
python validation/run_geometric_nonlinear_contact_validation.py --quick --skip-calculix --contact-mode persistent_calculix_c3d4_f2f --out-dir results/geometric_nonlinear_contact_validation_persistent_strict_smoke
```

Same enforcement with dynamic SDF plane query:

```bash
python validation/run_geometric_nonlinear_contact_validation.py --quick --skip-calculix --contact-mode dynamic_sdf_calculix_f2f --out-dir results/geometric_nonlinear_contact_validation_sdf_strict_smoke
```

Persistent same enforcement with dynamic SDF plane query:

```bash
python validation/run_geometric_nonlinear_contact_validation.py --quick --skip-calculix --contact-mode persistent_dynamic_sdf_calculix_f2f --out-dir results/geometric_nonlinear_contact_validation_persistent_sdf_strict_smoke
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
- Adds persistent spring lifecycle bookkeeping and generated spring count.
- Adds a cutback/contact convergence diagnostic based on residual and active-set
  trends.
- Adds deformable-deformable master DOF force and tangent assembly.
- Keeps dynamic SDF isolated as a contact-query provider rather than changing
  the mechanics/contact enforcement backend.

## Remaining Non-Equivalences

- The mode is a clean-room implementation, not copied CalculiX source.
- Persistent/cutback behavior is a clean-room approximation and still does not
  claim source-level CalculiX equivalence.
- The geometric nonlinear block-plane runner still uses rigid master contact;
  deformable-deformable master DOF assembly is implemented and unit tested as a
  validation primitive, not yet a full two-body dynamic runner.
- CalculiX output definitions (`CNUM`, `CELS`, fixed-floor `RF`) still must be
  compared only against matching strict-mode diagnostics.

## Validation Status

The added tests check that:

- strict C3D4 F2F mode uses one centroid sample per slave triangle;
- strict mode differs from the older three-point quadrature sample count;
- strict dynamic-SDF plane query matches the strict analytic-plane query for a
  flat master surface;
- the geometric nonlinear contact validation runner defaults to
  `persistent_calculix_c3d4_f2f`;
- persistent lifecycle generation, persistence, and release work as expected;
- cutback diagnostics flag active-set oscillation;
- deformable-deformable master/slave contact force is action-reaction balanced.
