# CalculiX Function-Level Equivalence Map

This document tracks the clean-room function-level equivalence work used to
isolate SFC/CalculiX contact trajectory differences.  The goal is behavioral
equivalence for targeted functions, not copying CalculiX source code into the
BSD-licensed SFC core.

## Implemented Segments

| CalculiX function area | SFC equivalent | What is matched | Tests |
| --- | --- | --- | --- |
| `prediction.c` implicit dynamic branch | `sfc.fem.calculix_aligned.calculix_dynamic_predictor` | HHT/Newmark displacement and velocity predictor and reset of the acceleration accumulator before Newton iterations. | `tests/test_calculix_aligned_backend.py::test_calculix_dynamic_predictor_matches_prediction_state_update` |
| `iniparll.c` implicit dynamic update | `sfc.fem.calculix_aligned.calculix_apply_acceleration_increment` | Newton solution interpreted as acceleration increment; displacement update by `beta dt^2`, velocity update by `gamma dt`, and accumulated acceleration update. | `tests/test_calculix_aligned_backend.py::test_calculix_acceleration_increment_updates_displacement_velocity_acceleration` |
| `calcresidual.c` implicit dynamic residual | `sfc.fem.calculix_aligned.calculix_hht_effective_residual` | CalculiX RHS sign `b = (1 + alpha)(fext - fint) - alpha(fextini - fini) - M a`, represented in SFC as the negative residual `R`. | `tests/test_calculix_aligned_backend.py::test_calculix_hht_residual_and_tangent_helpers_match_solver_form` |
| `nonlingeo.c` HHT effective matrix assembly | `sfc.fem.calculix_aligned.calculix_hht_effective_tangent` | Effective tangent `M / (beta dt^2) + (1 + alpha) K_static`; contact tangent is included when present in `K_static`. | `tests/test_calculix_aligned_backend.py::test_calculix_hht_residual_and_tangent_helpers_match_solver_form` |
| `springforc_f2f.f` / `springstiff_f2f.f` hard-linear branch | `validation.calculix_f2f_contact.calculix_hard_linear_spring_law` | No tensile force for positive clearance; area-weighted linear compression force; `kscale` pressure-stiffness scaling; matching contact energy and tangent scale. | `tests/test_calculix_f2f_contact.py::test_calculix_hard_linear_spring_law_has_no_tension_and_area_weighted_compression` |
| `gencontelem_f2f.f` generated spring lifecycle | `validation.calculix_f2f_contact.calculix_contact_lifecycle_decision` and `CalculixF2FContactLifecycle.update` | Separates generated spring state from force-active penetration; reports generated, persisted, cutback-persisted, released, inactive, and reactivated transitions; keeps generated-key memory for release/reactivation diagnostics. | `tests/test_calculix_f2f_contact.py::test_lifecycle_decision_reports_generated_persisted_cutback_and_released_states`, `tests/test_calculix_f2f_contact.py::test_persistent_lifecycle_reports_reactivation_after_release` |
| Accepted `fini/fextini` history | `sfc.fem.calculix_aligned.static_force_state` and `hht_step` returned history | Accepted history is stored as `fextini - fini`; contact spring force enters `fini` as an internal spring contribution. | `tests/test_calculix_aligned_backend.py::test_initial_state_returns_calculix_fextini_minus_fini_history`, `tests/test_calculix_aligned_backend.py::test_hht_step_saves_accepted_contact_internal_force_history` |

## Current Boundary

The function-level map is scoped to the SFC validation backend.  It should not
be described as a source-level CalculiX clone.  Remaining gaps still include:

- full `gencontelem_f2f` branch coverage beyond the explicit generated,
  persisted, cutback-persisted, released, inactive, and reactivated transition
  states now covered;
- complete `checkconvergence` multi-criterion acceptance and stabilization;
- every `kscale` restoration branch;
- exact output-side definitions for all contact print variables;
- CalculiX global matrix storage and sparse solver details.

## Next Segments

The next sections to implement should be:

1. Increment acceptance: residual/correction/contact-change criteria as
   standalone functions with test vectors.
2. Contact stabilization and `kscale`: explicit pressure-stiffness scaling
   state and restoration behavior.
3. Output definitions: CNUM/CELS/RF calculations from the same generated spring
   records used by force assembly.
