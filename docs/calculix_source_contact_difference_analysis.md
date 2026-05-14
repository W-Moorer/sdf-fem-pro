# CalculiX Source Contact Difference Analysis

## Scope

This note answers where the current SFC/CalculiX contact-validation differences
come from after reading the local WSL CalculiX source. It does not change core
physics or the `src/sfc` solver.

Local CalculiX source inspected:

- checkout: `/tmp/sfc_calculix_source`
- commit: `9033050`
- files:
  - `/tmp/sfc_calculix_source/src/gencontelem_f2f.f`
  - `/tmp/sfc_calculix_source/src/springforc_f2f.f`
  - `/tmp/sfc_calculix_source/src/contactprints.f`
  - `/tmp/sfc_calculix_source/src/printout.f`
  - `/tmp/sfc_calculix_source/src/checkconvergence.c`
  - `/tmp/sfc_calculix_source/src/dynamics.f`
  - `/tmp/sfc_calculix_source/src/dyna.c`

Project code and evidence inspected:

- `src/sfc/fem/calculix_aligned.py`
- `src/sfc/contact/sdf_geometry.py`
- `validation/run_calculix_drop_impact_comparison.py`
- `validation/run_geometric_nonlinear_contact_validation.py`
- `validation/run_contact_enforcement_diagnostics.py`
- `docs/calculix_aligned_backend_validation_handoff.md`
- `docs/contact_query_equivalence_handoff.md`
- `docs/deforming_master_sdf_reference_handoff.md`
- `docs/contact_enforcement_diagnostics_handoff.md`
- `docs/geometric_nonlinear_contact_validation_handoff.md`

## Short Answer

The remaining difference is not primarily caused by the SDF query, and it is not
currently explained by the bulk FEM backend. The strongest evidence is:

- no-contact CalculiX cantilever comparison already matches at very small
  relative errors for displacement, reaction, and von Mises stress;
- analytic-plane contact and dynamic-SDF plane contact match to roundoff when
  the same SFC backend is used;
- dynamic SDF broad-phase/local projection matches brute-force projection on
  the deforming-master reference.

The current mismatch is mainly in **contact enforcement semantics**:

1. CalculiX generates contact spring elements and manages a persistent active
   set through increment/iteration logic.
2. SFC validation contact directly samples slave faces and assembles penalty
   samples without reproducing CalculiX's exact contact-element generation.
3. For C3D4/TET4 faces, CalculiX uses one slave-face integration point per
   linear triangular face in the inspected path, while some SFC validation paths
   use three triangle quadrature samples.
4. CalculiX `CNUM`, `CELS`, and floor `RF` outputs are not the same quantities
   as SFC active quadrature count, SFC contact energy, or summed SFC penalty
   force.

Therefore it is not valid to say that the only difference is the SDF contact
query. At present the comparison changes both the query and parts of the
contact enforcement discretization.

## CalculiX Contact Path From Source

### Contact Element Generation

In `gencontelem_f2f.f`, CalculiX loops over active contact ties and slave
faces, then evaluates slave-face integration points and searches/project them
onto the master surface.

Relevant source facts:

- `gencontelem_f2f.f:183-193` loops over slave faces.
- `gencontelem_f2f.f:215-222` sets C3D10 faces to `mint2d=3`, but C3D4/TET4
  faces to `mint2d=1` and `nopes=3`.
- `gencontelem_f2f.f:300-326` evaluates one integration point using
  `pslavsurf`, `shape3tri`, and the slave-face shape functions for the C3D4
  case.
- `gencontelem_f2f.f:319-403` searches the master triangulation from the slave
  integration point.
- `gencontelem_f2f.f:491-552` projects to the master face, stores the master
  local coordinates and normal, and computes clearance
  `clear = (x_slave_ip - x_master_projection) dot n_master`.
- `gencontelem_f2f.f:554-560` in dynamic calculation drops positive-clearance
  contacts for non-tied contact.
- `gencontelem_f2f.f:674-716` appends an `ESPRNGC` contact spring element with
  master nodes, slave nodes, integration-point id, and slave face id.

This differs from the current SFC validation contact, which does not create an
equivalent `ESPRNGC` element list and does not reproduce CalculiX's persistent
contact-element update logic.

### Pressure-Overclosure Force

In `springforc_f2f.f`, CalculiX evaluates the contact spring force for the
generated contact element:

- `springforc_f2f.f:151-157` reads the stored master normal and computes the
  clearance along that normal.
- `springforc_f2f.f:186-201` for linear/tied overclosure computes
  `stiff(1) = -springarea(1) * pressure_stiffness * clear / kscale` and contact
  spring energy `-stiff(1) * clear / 2`.
- `springforc_f2f.f:248-253` applies the normal force to the slave part of the
  contact element as `fnl(i,nopep) = -stiff(1) * xn(i)` and reports pressure as
  `stiff(1) / springarea(1)`.

SFC has two relevant validation implementations:

- `src/sfc/fem/calculix_aligned.py::assemble_contact_response` uses a hard
  linear penalty law `lambda = stiffness * area * penetration`.
- `validation/run_calculix_drop_impact_comparison.py::_calculix_aligned_plane_contact`
  uses a smooth non-attractive overclosure transition for diagnostic stability,
  so it is intentionally not identical to CalculiX hard linear overclosure near
  zero penetration.

### Contact Output Definitions

CalculiX output labels are also solver-specific:

- `contactprints.f:62-68` recognizes `CSTR`, `CDIS`, `CNUM`, `CELS`, `CF`,
  `CFN`, and `CFS`.
- `contactprints.f:192` turns on contact energy output when `CELS` is requested.
- `printout.f:623-629` writes `total contact spring energy`.
- `printout.f:630-636` writes `total number of contact elements`, which is
  `ne - ne0 + 1` in that output path.
- `printoutcontact.f:76-116` loops over contact elements, not SFC quadrature
  samples.

This means CalculiX `CNUM` should not be compared as if it were the number of
SFC active quadrature samples.

### Dynamics and Convergence Logic

The input runner writes:

```text
*dynamic, direct, alpha=-0.05
```

CalculiX source details:

- `dynamics.f:73-74` defaults to `alpha=-0.05` and `alpha(2)=0.5`.
- `dynamics.f:106-113` clamps HHT alpha to `[-1/3, 0]`.
- `dyna.c` contains time increment control/cutback behavior, not just a fixed
  one-shot Newmark update.
- `checkconvergence.c:567-585` explicitly uses contact-element stabilization
  patterns in convergence/divergence handling.

SFC's clean-room backend implements the matching HHT/Newmark parameter
convention, but its Newton/cutback/contact convergence control is not a
source-level copy of CalculiX.

## Current Numerical Difference

From `results/geometric_nonlinear_contact_validation_full/geometric_contact_calculix_comparison.csv`:

| Resolution | Max penetration C | Max penetration SFC | Penetration rel. diff. | Peak force C | Peak force SFC | Force rel. diff. | Peak contact energy C | Peak contact energy SFC |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r1 | `3.144397e-03` | `3.387373e-03` | `7.73e-02` | `4.118874e+01` | `3.981581e+01` | `3.33e-02` | `6.457809e-02` | `6.038913e-02` |
| r2 | `2.743927e-03` | `3.255793e-03` | `1.87e-01` | `3.077931e+01` | `3.497837e+01` | `1.36e-01` | `3.775696e-02` | `4.915444e-02` |
| r3 | `2.690467e-03` | `3.021339e-03` | `1.23e-01` | `2.922631e+01` | `3.451957e+01` | `1.81e-01` | `3.346637e-02` | `4.625076e-02` |

The contact activation time matches in these rows (`0.096 s` for both), so the
first-order event timing is aligned. The remaining differences show up after
contact is active, which points to contact enforcement, not collision detection.

## Difference Breakdown

### 1. Bulk Mechanics Backend

Status: mostly aligned for scoped no-contact checks.

Evidence:

- no-contact cantilever relative errors in the latest quick CalculiX comparison:
  - tip displacement z: `3.251012e-08`;
  - fixed reaction z: `3.705369e-14`;
  - max von Mises stress: `2.049336e-07`.
- StVK tangent finite-difference relative error: `1.080587e-10`.
- no-contact freefall `dt=0.001` max z error: `4.440892e-16`.

Interpretation: the bulk backend is not the first suspect for the block-plane
contact mismatch.

### 2. SDF Query

Status: isolated SDF query is aligned with the reference projections.

Evidence:

- analytic plane contact and dynamic-SDF plane contact match with max field
  errors near roundoff in `docs/contact_query_equivalence_handoff.md`;
- deforming-master SDF broad-phase/local-projection reference reports zero gap
  and force error against brute-force projection in the tested configuration.

Interpretation: the SDF closest-point query is not currently the dominant source
of the CalculiX mismatch.

### 3. Slave-Surface Integration

Status: not exactly CalculiX-equivalent.

CalculiX C3D4 face-to-face contact uses `mint2d=1` for a linear triangular
TET4 face in `gencontelem_f2f.f:219-222`. SFC currently has validation paths
using three triangle quadrature samples. Three-point quadrature can be a better
surface integration approximation, but it is not the same discretization as the
inspected CalculiX C3D4 path.

This can change:

- active contact count;
- which parts of a deforming face activate first;
- integrated normal force;
- contact energy;
- stress concentration near contact.

### 4. Active Set and Persistence

Status: not CalculiX-equivalent.

CalculiX builds contact spring elements and carries previous active-contact
information through increment/cutback/iteration logic. The source explicitly
counts previous and actual contact elements in `gencontelem_f2f.f:185-193` and
has persistent/contact-stabilization behavior in `gencontelem_f2f.f:587-666`
and `checkconvergence.c:567-585`.

SFC currently assembles contact directly from the current samples. It does not
yet reproduce CalculiX's persistent active set, contact-element freezing, or
contact-specific cutback convergence behavior.

### 5. Pressure-Overclosure Law

Status: partially aligned, depending on the validation path.

The geometric nonlinear backend path uses a hard linear penalty law, closer to
`springforc_f2f.f:191-200`. The older drop-impact diagnostic path uses a smooth
non-attractive law around zero penetration. That smooth law intentionally lowers
force near zero overclosure and is not equivalent to CalculiX linear
pressure-overclosure.

From `docs/contact_enforcement_diagnostics_handoff.md`, for the same prescribed
penetration:

- three-point smooth area-weighted force: `4.209625e+01`;
- three-point hard-linear area-weighted force: `5.120000e+01`;
- unweighted nodal force: `3.200000e+02`.

This confirms that the pressure law and area weighting can move force and
energy substantially even before considering CalculiX's active-set logic.

### 6. Tangent and Newton/Cutback Behavior

Status: SFC tangent is internally consistent, but not a full CalculiX contact
solver clone.

SFC contact tangent finite-difference check passes with relative error
`3.264947e-10`, using the residual convention `K_contact ~= -d(f_contact)/du`.
However, CalculiX has contact-element generation, active-set stabilization, and
cutback/convergence control around contact. SFC's validation backend uses a
clean-room Newton solve and fixed settings.

For rigid-plane contact with constant normal, the current SFC tangent captures
the dominant normal penalty stiffness. For general dynamic SDF contact, exact
CalculiX-level equivalence would also need careful treatment of evolving
projection coordinates, active-set changes, normal changes, and area/shape
derivatives.

### 7. Output Definitions

Status: not one-to-one comparable.

CalculiX floor `RF` is the fixed-floor total reaction. SFC reports summed
penalty force. CalculiX `CELS` is total contact spring energy over CalculiX
contact elements. SFC contact energy is integrated over SFC samples. CalculiX
`CNUM` is contact-element count; SFC active count is active contact samples.

These are useful diagnostics, but they should not be treated as identical
physical observables unless the contact discretization is also made identical.

## Why SFC Max Penetration Can Be Higher Than CalculiX

The observed higher SFC penetration after activation is consistent with the
following combined effects:

1. SFC is not using the same CalculiX contact-element active set and persistence.
   CalculiX can keep or regenerate contact spring elements through its increment
   logic, while SFC samples only the current sample set.
2. SFC's quadrature/contact sample locations differ from the inspected CalculiX
   C3D4 face-contact path.
3. The compared force/energy histories are not computed from identical
   contact-element definitions.
4. SFC's Newton/cutback/contact convergence logic is simpler than CalculiX's.

Because activation time matches and no-contact mechanics/SDF-query isolation
tests pass, the difference should be localized to enforcement and solver
semantics after contact is active.

## What Is Currently Safe To Claim

Safe:

- SFC bulk mechanics backend is aligned with CalculiX for scoped no-contact
  checks.
- SFC dynamic SDF query matches analytic/brute-force closest-point references
  in controlled tests.
- SFC block-plane contact has comparable activation time, penetration scale,
  normal-force scale, and stress scale to CalculiX in the current scoped
  validation.

Not safe:

- "Only the contact query differs from CalculiX."
- "SFC contact enforcement is CalculiX-equivalent."
- "SFC active contact count is directly comparable to CalculiX CNUM."
- "Current SFC contact force/energy should match CalculiX exactly."

## Required Next Steps For Strict CalculiX-Except-SDF Alignment

Implementation status: strict and persistent C3D4 F2F validation modes have now
been added in `validation/calculix_f2f_contact.py` and wired into
`validation/run_geometric_nonlinear_contact_validation.py` as
`--contact-mode persistent_calculix_c3d4_f2f`,
`--contact-mode calculix_c3d4_f2f`,
`--contact-mode persistent_dynamic_sdf_calculix_f2f`, and
`--contact-mode dynamic_sdf_calculix_f2f`.

1. Add a dedicated `calculix_c3d4_f2f` comparison mode:
   - one integration point per C3D4 triangular face, matching
     `gencontelem_f2f.f:219-222`;
   - hard linear overclosure, matching `springforc_f2f.f:191-200`;
   - CalculiX SPOS contact plane offset preserved for shell master surfaces.

   Status: implemented for rigid-plane and dynamic-SDF plane validation
   queries.

2. Reproduce contact spring-element state for the comparison runner:
   - store master face id, master local coordinates, master normal, slave face
     id, and integration-point id;
   - update/drop contacts with the same positive-clearance rule for the dynamic
     case;
   - report SFC contact count using a CalculiX-like contact-element definition.

   Status: implemented as diagnostic `CalculixF2FContactSpring` records plus a
   clean-room `CalculixF2FContactLifecycle`. The lifecycle stores generated
   springs, reports generated spring count separately from negative-clearance
   force samples, and supports generated/persisted/released event diagnostics.
   This still does not claim source-level CalculiX branch equivalence.

3. Add contact convergence/cutback diagnostics:
   - track active contact count and residual trends;
   - flag residual growth and active-set oscillation as cutback/retry
     candidates.

   Status: implemented as `CalculixContactConvergenceHeuristic` and reported by
   the geometric nonlinear contact validation runner.

4. Add deformable-deformable master DOF spring assembly:
   - assemble `dg/dx_slave = N_slave n^T`;
   - assemble `dg/dx_master = -N_master n^T`;
   - assemble `f = J^T lambda` and `K ~= k area J^T J`.

   Status: implemented as `assemble_deformable_f2f_contact_response` with
   action-reaction tests. A full two-body dynamic runner remains a follow-up.

5. Split validation outputs:
   - one strict CalculiX-discretization mode for external solver comparison;
   - one higher-order SFC quadrature mode for the proposed method's own accuracy
     study.

6. Compare only fields with matching definitions:
   - center-of-mass height and velocity;
   - geometry-based min gap and max penetration using the same sample points;
   - summed normal reaction using the same force sign and surface;
   - contact spring energy using the same sample/contact-element definition;
   - stress cloud at the same physical time and same color scale.

7. Keep SDF as the isolated variable only after the enforcement mode is matched:
   - run analytic-plane contact and dynamic-SDF-plane contact through the same
     `calculix_c3d4_f2f` enforcement path;
   - then compare against CalculiX.

## Conclusion

The current project has already separated the major causes:

- bulk FEM backend: aligned for scoped no-contact tests;
- SDF query: aligned against analytic/brute-force references;
- remaining discrepancy: CalculiX contact enforcement, active-set generation,
  contact-element state, quadrature definition, output definitions, and
  contact-specific convergence logic.

The next implementation step should not be another SDF change. It should be a
strict CalculiX C3D4 face-to-face contact enforcement mode for validation, with
SDF swapped in only after that enforcement path is matched.
