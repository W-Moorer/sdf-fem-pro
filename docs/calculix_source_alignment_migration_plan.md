# CalculiX Source Alignment Migration Plan

This document records the local-source analysis for aligning SFC validation
behavior with CalculiX while preserving the project boundary: CalculiX remains
an external reference, and SFC uses clean-room Python implementations instead of
copying GPLv2 Fortran/C code into the core solver.

## Licensing Boundary

The local CalculiX source under `/tmp/sfc_calculix_source` is GPLv2.  Directly
copying or translating its implementation into this repository would create a
license-coupled derivative work.  The migration path used here is therefore:

1. Read the local source to understand behavior.
2. Implement independent clean-room Python logic for the validation layer.
3. Verify behavior using CSV comparisons and finite-difference checks.
4. Keep CalculiX execution optional and external.

## CalculiX Behavior To Match

| CalculiX source | Behavior | SFC migration status |
| --- | --- | --- |
| `precontact.c` | Updates contact penetration data, performs first-increment contact-node adjustment, and prepares slave integration points. | Partially represented through deterministic generated validation geometry; exact node-adjustment branches are not migrated. |
| `gencontelem_f2f.f` | Searches a master triangle, stores master local coordinates and normal, computes clearance, creates `ESPRNGC` contact spring elements, and applies dynamic/static/cutback lifecycle rules. | C3D4 one-point slave-face spring generation, hard-linear pressure law, CNUM-equivalent counting, and basic persistence are implemented in `validation/calculix_f2f_contact.py`. Full private active-set/cutback branches remain diagnostic-only. |
| `springforc_f2f.f` | Reconstructs current master projection from stored local coordinates, computes clearance along stored normal, computes force and contact spring energy. | Plane contact matches the same scalar law. Persistent dynamic-SDF F2F mode now reuses the stored master face, barycentric weights, and normal, and recomputes clearance from current coordinates. |
| `springstiff_f2f.f` | Uses the same clearance and normal to assemble contact tangent. | Contact tangent sign convention is finite-difference checked in `geometric_contact_hht_residual_tangent.csv`. |
| `printout.f` / `printoutelem.f` | `CELS` is total contact spring energy; `CNUM` is generated contact element count. | Output-definition diagnostics are implemented in `geometric_contact_lifecycle_output_diagnostics.csv`. |
| `printoutcontact.f` | `CF/CFN/CFS` are surface-force totals using stored normals, contact pressure, and spring area. | Current comparison uses floor `RF` and SFC normal-force proxy; the definitions are documented as comparable but not source-identical. |
| `nonlingeo.c`, `prediction.c`, `calcresidual.c`, `checkconvergence.c` | Defines HHT beta/gamma, dynamic predictor, residual assembly, and increment acceptance with residual/correction/contact/energy criteria. | HHT residual/tangent is finite-difference checked; mechanics/increment acceptance diagnostics are written to `geometric_contact_mechanics_increment_acceptance.csv`; `hht_step()` now exposes an optional clean-room `calculix_multicriteria` acceptance policy. |

## Current Alignment Result

The latest quick diagnostic writes:

- `results/geometric_contact_lifecycle_output_check/geometric_contact_lifecycle_output_diagnostics.csv`
- `results/geometric_contact_lifecycle_output_check/geometric_contact_alignment_diagnostics.csv`
- `results/geometric_contact_lifecycle_output_check/geometric_contact_hht_residual_tangent.csv`
- `results/geometric_contact_lifecycle_output_check/geometric_contact_mechanics_increment_acceptance.csv`

The important interpretation is:

- HHT residual/tangent and contact tangent signs are finite-difference
  consistent on the sampled SFC trajectory.
- The contact force-law replay on the CalculiX displacement trajectory is much
  closer than the native SFC time-history comparison.
- Remaining `RF/CELS/max penetration` differences are therefore mainly caused
  by trajectory divergence plus contact spring lifecycle/clearance-output
  definition differences, not by a simple force sign error.

## Next Migration Steps

The next clean-room implementation steps, in priority order, are:

1. Extend the current stored-master-projection behavior from rigid/fixed master
   validation surfaces to fully coupled deformable master bodies.
2. Separate three counts in every contact row:
   generated spring records, force-producing negative-clearance springs, and
   CalculiX `CNUM`-equivalent generated contact-element units.
3. Add a per-time-row replay table on the CalculiX displacement trajectory for
   `CNUM`, force, `CELS`, max penetration, and clearance.
4. Only after those gates pass, compare native SFC and CalculiX trajectories
   again.  If they still diverge, inspect mechanics/increment acceptance rather
   than contact-query definitions.
5. Use the optional clean-room `calculix_multicriteria` acceptance policy in
   controlled validation runs, compare its trajectory against the default
   relative-correction policy, and only promote it if it improves external
   alignment without weakening no-contact verification.

## Claim Boundary

The target is not to claim that SFC is a Python copy of CalculiX.  The target is
to isolate the project innovation:

```text
Keep mechanics, time integration, and validation contact definitions as close
as practical to CalculiX, while replacing only the contact query with the SFC
dynamic FEM-induced SDF.
```

Exact bitwise or source-level equivalence is not a supported claim unless the
project deliberately adopts CalculiX as a linked external backend or accepts
GPLv2 derivative-code obligations.
