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
| `nonlingeo.c` accepted-state snapshot boundary | `sfc.fem.calculix_aligned._snapshot_contact_state` and `_restore_contact_state` | Diagnostic previous-energy probes in `hht_step` no longer mutate persistent contact spring lifecycle before the new increment starts, matching CalculiX's separation between saved `neini/fini/fextini/veini/accini` state and trial increment evaluations. | `tests/test_calculix_aligned_backend.py::test_contact_state_snapshot_restores_lifecycle_and_cutback_flag` |
| `springforc_f2f.f` / `springstiff_f2f.f` hard-linear branch | `validation.calculix_f2f_contact.calculix_hard_linear_spring_law` | No tensile force for positive clearance; area-weighted linear compression force; `kscale` pressure-stiffness scaling; matching contact energy and tangent scale. | `tests/test_calculix_f2f_contact.py::test_calculix_hard_linear_spring_law_has_no_tension_and_area_weighted_compression` |
| `gencontelem_f2f.f` generated spring lifecycle | `validation.calculix_f2f_contact.calculix_contact_lifecycle_decision` and `CalculixF2FContactLifecycle.update` | Separates generated spring state from force-active penetration; reports generated, persisted, cutback-persisted, released, inactive, reactivated, no-master-candidate, lost-candidate-release, and lost-candidate-cutback-persist transitions; keeps generated-key memory for release/reactivation diagnostics. | `tests/test_calculix_f2f_contact.py::test_lifecycle_decision_reports_generated_persisted_cutback_and_released_states`, `tests/test_calculix_f2f_contact.py::test_persistent_lifecycle_reports_reactivation_after_release`, `tests/test_calculix_f2f_contact.py::test_lifecycle_can_keep_lost_candidate_during_cutback` |
| `gencontelem_f2f.f` master-face miss (`isol = 0`) | `CalculixC3D4FaceToFaceSDFContactGeometry.contact_springs` plus lifecycle lost-candidate decision | Empty broad-phase candidate lists are treated as no detected master face instead of a failed SDF query; persistent SDF contact can keep a previously generated spring during a cutback retry. | `tests/test_calculix_f2f_contact.py::test_sdf_geometry_skips_empty_candidate_queries_like_no_master_face`, `tests/test_calculix_f2f_contact.py::test_persistent_sdf_geometry_keeps_lost_candidate_on_cutback` |
| `gencontelem_f2f.f` positive-clearance contact-type branch | `calculix_contact_lifecycle_decision(..., allow_positive_clearance_generation=True)` | Models the targeted branch where a detected master face may still generate a spring at positive clearance for special contact types, while the hard-linear law remains force-inactive in tension. | `tests/test_calculix_f2f_contact.py::test_lifecycle_decision_can_generate_positive_clearance_contact_type`, `tests/test_calculix_f2f_contact.py::test_lifecycle_can_keep_positive_clearance_springs_without_force`, `tests/test_calculix_f2f_contact.py::test_persistent_plane_geometry_can_generate_positive_clearance_without_force` |
| `gencontelem_f2f.f` static initial overclosure ramp | `validation.calculix_f2f_contact.calculix_static_clearance_ramp` | Stores an initial negative-clearance offset analogous to `springarea(2)`, closes small positive initial gaps below `1 / pressure_stiffness`, and applies the load-step ramp through `reltime`. | `tests/test_calculix_f2f_contact.py::test_static_clearance_ramp_stores_initial_overclosure_offset`, `tests/test_calculix_f2f_contact.py::test_static_clearance_ramp_closes_small_positive_initial_gap`, `tests/test_calculix_f2f_contact.py::test_static_clearance_ramp_reuses_previous_offset_after_initialization` |
| `gencontelem_f2f.f` static `iloop=1/2` generation branch | `validation.calculix_f2f_contact.calculix_static_iloop_generation_decision` | Models the scoped `xstateini` history logic: regular penetration generation, previous-increment `iprev` counting, cutback inclusion of previous contact, `iloop=2` previous-contact replay, no-master release, aleatoric removal, and positive-clearance contact-type generation. | `tests/test_calculix_f2f_contact.py::test_static_iloop_generation_uses_penetration_without_cutback`, `tests/test_calculix_f2f_contact.py::test_static_iloop_generation_counts_previous_increment_contact`, `tests/test_calculix_f2f_contact.py::test_static_iloop_cutback_keeps_previous_positive_clearance_contact`, `tests/test_calculix_f2f_contact.py::test_static_iloop2_uses_previous_state_only`, `tests/test_calculix_f2f_contact.py::test_static_iloop_generation_models_no_master_aleatoric_and_contact_type_branches` |
| Accepted `fini/fextini` history | `sfc.fem.calculix_aligned.static_force_state`, `_static_force_state_from_responses`, and `hht_step` returned history | Accepted history is stored as `fextini - fini`; contact spring force enters `fini` as an internal spring contribution; accepted-step history is built from the same converged contact evaluation used for diagnostics, avoiding an extra stateful contact sampling call at the accepted boundary. | `tests/test_calculix_aligned_backend.py::test_initial_state_returns_calculix_fextini_minus_fini_history`, `tests/test_calculix_aligned_backend.py::test_hht_step_saves_accepted_contact_internal_force_history`, `tests/test_calculix_aligned_backend.py::test_hht_step_saves_history_from_accepted_contact_evaluation_without_resampling` |

## Current Boundary

The function-level map is scoped to the SFC validation backend.  It should not
be described as a source-level CalculiX clone.  Remaining gaps still include:

- full `gencontelem_f2f` branch coverage beyond the explicit generated,
  persisted, cutback-persisted, released, inactive, reactivated, no-master,
  lost-candidate, positive-clearance generation, static clearance-ramp, and
  static `iloop=1/2` `xstateini` generation states now covered;
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
