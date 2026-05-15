# CalculiX Contact Lifecycle And Output Definitions

This note records the source-level findings used to interpret the remaining
SFC/CalculiX contact differences.  The CalculiX source inspected locally is
under `/tmp/sfc_calculix_source/src`; it is used only as an external validation
reference and is not imported by the core `src/sfc` package.

## Source Files Inspected

| Topic | CalculiX source path | Relevant behavior |
| --- | --- | --- |
| Contact pre-processing | `/tmp/sfc_calculix_source/src/precontact.c` | Updates contact penetration data, adjusts initial contact nodes on the first increment, and constructs slave integration points. |
| Contact element generation | `/tmp/sfc_calculix_source/src/contact.c`, `/tmp/sfc_calculix_source/src/gencontelem_f2f.f` | Dispatches face-to-face contact and creates `ESPRNGC` contact spring elements. |
| Contact force | `/tmp/sfc_calculix_source/src/springforc_f2f.f` | Recomputes current clearance from the stored master projection and normal, then computes contact force and contact spring energy. |
| Contact tangent | `/tmp/sfc_calculix_source/src/springstiff_f2f.f` | Uses the same stored projection/normal and hard-linear pressure-overclosure tangent for the contact spring. |
| Contact output request parsing | `/tmp/sfc_calculix_source/src/contactprints.f` | Accepts `CSTR`, `CDIS`, `CNUM`, `CELS`, `CF`, `CFN`, and `CFS`; requesting `CELS` enables contact energy output. |
| Contact output totals | `/tmp/sfc_calculix_source/src/printout.f`, `/tmp/sfc_calculix_source/src/printoutelem.f`, `/tmp/sfc_calculix_source/src/printoutcontact.f` | Prints total contact spring energy, total contact element count, and contact surface force summaries. |

## Lifecycle Interpretation

For the C3D4 face-to-face case, CalculiX does not simply evaluate an independent
penalty sample at every time row.  It generates contact spring elements:

- The generated element type is `ESPRNGC`.
- A linear TET4 slave face uses one slave face integration point in the current
  strict validation setup.
- The generated contact element stores slave nodes, master nodes, slave face
  identity, and the slave integration point identity.
- During the initial contact search, CalculiX stores the master local projection
  coordinates in `pmastsurf(1:2)` and the contact normal in `pmastsurf(4:6)`.
- During later Newton iterations, CalculiX reuses the stored master local
  projection and normal rather than performing an unconstrained new closest
  point search for every residual evaluation.
- In dynamic analysis, positive clearance can drop a contact spring unless tied
  or persistent-contact branches keep it active for convergence control.

This explains why matching the force law alone is not enough.  The SFC
diagnostic has to track the generated contact-spring lifecycle, not only active
penetrating samples.

## Clearance Update

For the F2F spring force and stiffness routines, CalculiX evaluates the current
clearance as:

```text
clear = (current_slave_ip - current_master_projection) dot stored_normal
```

where the current master projection is reconstructed from the stored master
local coordinates and current master nodal positions.  The stored normal is
read from `pmastsurf(4:6)`.  Initial static overclosure correction uses the
stored `springarea(2)` term and the load-step ramp.  This is distinct from a
fresh dynamic-SDF closest-point query at every row, where the closest face,
barycentric coordinates, and normal may change continuously.

## CNUM, CELS, And RF Definitions

The external comparison now treats these outputs as definition-sensitive:

- `CNUM` is the number of generated CalculiX contact spring elements printed by
  the contact output path.  It is not the same thing as the number of currently
  force-producing SFC samples.  The validation runner therefore records an SFC
  `calculix_equivalent_contact_count` when the validation-only F2F mode can
  map a generated spring to CalculiX CNUM units.
- `CELS` is CalculiX total contact spring energy printed by `printout.f`.  SFC
  reports the corresponding penalty spring energy assembled over its generated
  validation springs.  Differences can remain if the generated spring lifecycle
  or clearance update differs.
- `RF` is read from the floor node-set reaction-force output in the generated
  CalculiX input.  It is a fixed-floor total reaction, while the SFC diagnostic
  reports assembled normal penalty force.  The signs and surfaces are aligned
  in the post-processor, but the definitions are not source-identical.

## Current Diagnostic Evidence

`validation/run_geometric_nonlinear_contact_validation.py` now writes:

- `geometric_contact_lifecycle_output_diagnostics.csv`
- `geometric_contact_alignment_diagnostics.csv`
- `geometric_contact_hht_residual_tangent.csv`
- `geometric_contact_element_clearance_lifecycle_audit.csv`
- `geometric_contact_element_clearance_lifecycle_audit_summary.csv`

The lifecycle CSV records:

- CalculiX `CNUM` sequence.
- SFC CalculiX-equivalent `CNUM` sequence.
- A replay `CNUM` sequence computed by applying the clean-room force law to
  the CalculiX displacement trajectory.
- First mismatch time.
- Maximum absolute `CNUM` mismatch.
- Whether CalculiX release/reactivation was observed.
- Peak-time differences for RF and CELS proxies.
- A diagnosis string separating external-unavailable cases from true
  lifecycle/output-trajectory differences.

The per-contact-element audit is more granular.  The generated CalculiX input
requests non-totals `CDIS`, `CSTR`, and `CELS` contact print blocks and keeps
totals-only `CNUM` for the existing aggregate contact-count comparison.  The parser maps
each CalculiX row by `slave element + slave face` and writes, per time step:

- CalculiX active contact elements from `.dat` `CDIS/CSTR/CELS` rows.
- SFC native generated springs on the same slave element+face keys.
- CalculiX-displacement replay springs using the same SFC hard-linear
  area-weighted force law on CalculiX nodal displacements.
- Per-face clearance, normal pressure/force proxy, and contact energy.
- If this CalculiX build does not emit per-element `CELS` rows despite the
  request, the audit keeps the raw `calculix_cels_energy` field blank and fills
  `calculix_contact_energy` as an explicitly labeled
  `derived_from_cdis_cstr` proxy:
  `0.5 * |normal pressure| * current face area * max(-clearance, 0)`.
- Linf replay errors for clearance, force, and energy where CalculiX per-face
  output exists.

The current key finding is that the remaining RF/CELS/max-penetration
differences are consistent with contact spring lifecycle, trajectory, and
output-definition differences.  The replay sequence is important: if the
clean-room contact law replayed on the CalculiX displacement trajectory matches
CalculiX substantially better than the native SFC trajectory, the mismatch is
trajectory-dominated rather than a pure contact-output-definition error.
Earlier HHT residual/tangent diagnostics show that the SFC effective tangent,
contact tangent sign convention, and previous-static residual update are
finite-difference consistent for the sampled trajectory.

The validation-only persistent dynamic-SDF F2F mode also now preserves the
generated master face, barycentric weights, and normal for an existing spring,
then recomputes clearance from current nodal coordinates.  This mirrors the
CalculiX `pmastsurf` usage for the portion of the lifecycle that is safe to
clean-room migrate without copying source.

## Consequence For Method Claims

This evidence should not be described as exact CalculiX contact equivalence.
The scoped claim is narrower:

```text
SFC now has validation diagnostics that identify whether remaining differences
come from force-law/tangent errors, dynamic residual evaluation, contact spring
lifecycle differences, clearance update differences, or output-definition
differences.
```

The production method remains the FEM-induced dynamic-SDF contact query.  The
CalculiX-aligned F2F layer is a validation-only bridge used to isolate which
parts of the external solver comparison are comparable.
