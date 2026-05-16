# Geometric Nonlinear Contact Validation

This validation compares the clean-room CalculiX-aligned SFC StVK geometric nonlinear block-plane contact path against CalculiX dynamic contact output.

SFC contact mode: `persistent_dynamic_sdf_calculix_f2f`.
SFC cutback policy: `calculix_direct`.

The strict `calculix_c3d4_f2f` modes use seven slave-face integration springs per C3D4 boundary triangle, stored master projection/normal clearance, and a hard linear pressure-overclosure law for CalculiX-style validation. The older `plane` and `dynamic_sdf_plane` modes use three triangle quadrature samples and are not strict CalculiX C3D4 contact discretizations.

## Outputs

- `geometric_contact_history.csv`
- `geometric_contact_calculix_comparison.csv`
- `geometric_contact_alignment_diagnostics.csv`
- `geometric_contact_hht_residual_tangent.csv`
- `geometric_contact_lifecycle_output_diagnostics.csv`
- `geometric_contact_mechanics_increment_acceptance.csv`
- `geometric_contact_one_step_state_diagnostics.csv`
- `geometric_contact_hht_state_definition_diagnostics.csv`
- `geometric_contact_element_clearance_lifecycle_audit.csv`
- `geometric_contact_element_clearance_lifecycle_audit_summary.csv`
- `geometric_contact_nodal_state_diagnostics.csv`
- `geometric_contact_floor_rf_distribution.csv`
- `geometric_contact_raw_contact_print_rows.csv`
- `geometric_contact_first_contact_spring_comparison.csv`
- `geometric_contact_true_velocity_comparison.csv`
- `geometric_contact_calculix_internal_exportability.csv`
- `geometric_contact_mesh_convergence.csv`
- `geometric_contact_timestep_convergence.csv`
- `geometric_contact_stress_clouds.csv`
- `geometric_contact_plots.csv`
- `geometric_contact_claims.csv`

## Claim Gates

| Claim | Supported | Evidence |
| --- | --- | --- |
| block_plane_geometric_contact_calculix_comparison_available | true | `geometric_contact_calculix_comparison.csv` |
| block_plane_geometric_contact_scoped_acceptance | true | `geometric_contact_calculix_comparison.csv` |
| calculix_contact_force_law_replay_diagnostics_available | true | `geometric_contact_alignment_diagnostics.csv` |
| hht_residual_tangent_trajectory_diagnostics_available | false | `geometric_contact_hht_residual_tangent.csv` |
| calculix_contact_lifecycle_output_diagnostics_available | true | `geometric_contact_lifecycle_output_diagnostics.csv` |
| calculix_mechanics_increment_acceptance_diagnostics_available | true | `geometric_contact_mechanics_increment_acceptance.csv` |
| calculix_state_one_step_mechanics_diagnostics_available | false | `geometric_contact_one_step_state_diagnostics.csv` |
| hht_newmark_state_definition_precision_diagnostics_available | true | `geometric_contact_hht_state_definition_diagnostics.csv` |
| calculix_per_contact_element_clearance_lifecycle_audit_available | false | `geometric_contact_element_clearance_lifecycle_audit_summary.csv` |
| calculix_nodal_velocity_acceleration_output_available | false | `geometric_contact_nodal_state_diagnostics.csv` |
| calculix_floor_node_rf_distribution_available | true | `geometric_contact_floor_rf_distribution.csv` |
| calculix_raw_contact_print_rows_preserved | true | `geometric_contact_raw_contact_print_rows.csv` |
| first_contact_per_spring_comparison_available | true | `geometric_contact_first_contact_spring_comparison.csv` |
| calculix_sfc_true_velocity_comparison_available | true | `geometric_contact_true_velocity_comparison.csv` |
| contact_mesh_convergence_trend_available | false | `geometric_contact_mesh_convergence.csv` |
| contact_timestep_convergence_trend_available | false | `geometric_contact_timestep_convergence.csv` |
| paraview_stress_cloud_comparison_available | false | `geometric_contact_stress_clouds.csv` |

## Contact Comparison

| Resolution | Status | z_cm L2 rel. | max pen. rel. | peak force rel. | CNUM abs. | contact-zone VM rel. |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | passed_scoped_gate | 2.601493e-04 | 2.316794e-02 | 9.816751e-03 | 0.000000e+00 | 3.937603e-03 |

## Alignment Diagnostics

| Resolution | Diagnosis | peak RF replay rel. | peak CELS replay rel. | force-law max rel. | kscale restores | max attempt | min CalculiX dt | min SFC dt |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | trajectory_difference_dominant | 1.985082e-03 | 2.605404e-03 | 1.229597e-02 | 0.000000e+00 | 1.000000e+00 | 1.000000e-03 | 1.000000e-03 |

## HHT Residual/Tangent Diagnostics

| Resolution | Contact mode | time | active | HHT tangent FD rel. | contact tangent FD rel. | previous-static update rel. |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |

## Lifecycle And Output Diagnostics

| Resolution | Diagnosis | CalculiX CNUM sequence | SFC CNUM sequence | Replay-on-CalculiX CNUM sequence | SFC CNUM max abs. | Replay CNUM max abs. | RF peak time error | CELS peak time error |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | contact_lifecycle_trajectory_difference_observed | `0->14->12->14->12->14->12->14->12->8->14->8->2->0->6->14->12->8->12->14->8->12->14->12->14->8->6->0->2->6->8->14->12->14->8->6->8->12->14->12->8->12->14->0` | `0->14->12->14->12->14->12->14->12->8->14->8->6->2->0->6->14->12->8->12->14->12->8->12->14->12->14->12->6->2->0->6->8->14->12->14->12->8->6->12->14->12->14->2->0` | `0->14->12->14->12->14->12->14->12->8->14->8->2->0->6->14->12->8->12->14->8->12->14->12->14->8->6->0->2->6->8->14->12->14->8->6->8->12->14->12->8->12->14->0` | 4.000000e+00 | 0.000000e+00 | 3.330669e-16 | 3.330669e-16 |

## Mechanics And Increment Acceptance Diagnostics

| Resolution | Diagnosis | Direct | CalculiX inc. | CalculiX max iter/inc. | SFC max iter | max ram/qam | max cam/uam | max contact change | steps w/change | energy stab. | final rejected |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | direct_fixed_increment_accepted_after_internal_newton | true | 1.000000e+03 | 4.000000e+00 | 5.000000e+00 | 2.821183e-08 | 9.110850e-09 | 1.400000e+01 | 2.100000e+01 | 2.000000e+00 | 0.000000e+00 |

## One-Step CalculiX-State Diagnostics

This table starts SFC from CalculiX displacement states. CalculiX does not print full global internal force vectors or tangent matrices in the parsed `.dat`, so the internal-force and tangent entries are SFC decompositions evaluated on CalculiX kinematics rather than direct CalculiX vectors.

| Resolution | time n+1 | Diagnosis | Dominant term | disp. rel. | accel. rel. | force-balance accel rel. | residual rel. | CalculiX RF | SFC-on-CalculiX force | SFC predicted force |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

## HHT/Newmark State Definition Diagnostics

| Resolution | Diagnosis | beta | gamma | initial accel rel. | mass rel. | .dat accel uncertainty | precision/diff | precontact residual rel. | active residual rel. |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | curve_only_mode_skipped_heavy_hht_state_diagnostics |  |  |  |  |  |  |  |  |

## Per-Contact-Element Clearance/Lifecycle Audit

| Resolution | time | CalculiX active | SFC native active | Replay active | per-contact output | clearance replay Linf | force replay Linf | energy replay Linf | Diagnosis |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |

## Nodal State And Raw Contact Output Diagnostics

| Output | Rows | Status |
| --- | ---: | --- |
| CalculiX nodal V/A state | 0 | unavailable |
| floor per-node RF distribution | 4000 | available |
| raw CDIS/CSTR/CELS rows | 5448 | available |
| first-contact per-spring comparison | 84 | available |
| direct CalculiX/SFC true velocity comparison | 1000 | available |

## Direct True Velocity Comparison

| Resolution | max nodal V rel. | max Vz rel. | max v_cm_z abs. | first contact V rel. | Diagnosis |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2.973926e-01 | 4.753449e-01 | 2.114516e-02 | 1.130447e-02 | velocity_trajectory_difference |

CalculiX internal arrays `clearini`, `pmastsurf`, `xstateini`, and `springarea` are not standard `.dat` outputs. The exportability table records source-path hints and whether instrumentation/debug export is required.

| Internal array | standard .dat | contact print | debug export | source path hint |
| --- | --- | --- | --- | --- |
| clearini | false | partially_cdis_after_printout_not_initial_array | true | `/tmp/sfc_calculix_source/src/nonlingeo.c; /tmp/sfc_calculix_source/src/gencontelem_f2f.f` |
| pmastsurf | false | false | true | `/tmp/sfc_calculix_source/src/nonlingeo.c; /tmp/sfc_calculix_source/src/gencontelem_f2f.f` |
| xstateini | false | false | true | `/tmp/sfc_calculix_source/src/nonlingeo.c` |
| springarea | false | indirectly_derivable_from_CSTR_and_force_law_when_clearance_is_available | true | `/tmp/sfc_calculix_source/src/gencontelem_f2f.f; /tmp/sfc_calculix_source/src/springforc_f2f.f` |

## Trend Rows

- Mesh convergence rows: `0`
- Time-step convergence rows: `0`
- VTK stress cloud rows: `0`
- Curve plot rows: `5`

## Curve Plots

- `calculix_drop_z_cm.png` and `calculix_drop_z_cm.pdf`
- `calculix_drop_min_gap.png` and `calculix_drop_min_gap.pdf`
- `calculix_drop_force_proxy.png` and `calculix_drop_force_proxy.pdf`
- `calculix_drop_contact_energy.png` and `calculix_drop_contact_energy.pdf`
- `calculix_drop_total_energy.png` and `calculix_drop_total_energy.pdf`

## Limitations

- This validates the diagnostic clean-room StVK TET4 contact path, not a copied CalculiX implementation.
- The mechanics backend is swappable with SFC dynamic-SDF contact geometry; this runner uses rigid-plane contact for external alignment.
- The strict mode still does not copy CalculiX source and does not reproduce every persistent active-set/cutback branch of CalculiX.
- Passing the scoped gate is not a source-level CalculiX contact equivalence claim.
- If CalculiX does not export RF/contact-energy data, the affected force/energy comparison fields remain blank.
- Mesh/time-step trend files do not claim theoretical convergence order.
