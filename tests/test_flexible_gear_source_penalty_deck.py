from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.prepare_flexible_gear_source_penalty_deck import prepare_source_penalty_deck_text
from validation.run_flexible_gear_source_penalty_abaqus import write_source_penalty_summary
from validation.run_flexible_gear_full_lagrangian_sdf_comparison import (
    constraint_region_contact_law_gate_metrics,
    constraint_region_tangent_gate_metrics,
    full_gear_evidence_ladder_rows,
    full_gear_entry_gate_metrics,
    load_patch_prerequisite_summary,
    path_tracking_gate_metrics,
    run_full_gear,
    source_active_set_line_search_gate_metrics,
    source_convergence_gate_metrics,
)


def test_source_convergence_gate_requires_converged_accepted_steps() -> None:
    summary = {
        "sfc_increment_count": 2,
        "source_accepted_increment_count": 2,
        "source_step_converged_count": 2,
        "source_unstable_accepted_count": 0,
        "source_unconverged_accepted_count": 0,
        "source_trial_gate_passed": 1,
        "source_accepted_contact_response_reuse_count": 1,
        "source_accepted_contact_response_requery_count": 1,
        "source_constraint_region_tangent_solve_count": 1,
        "source_constraint_region_tangent_active_rows_sum": 4,
        "final_active_contact_samples": 4,
        "source_final_time": 2.0e-5,
        "sfc_duration": 2.0e-5,
    }
    history = [
        {"time": 1.0e-5, "source_increment_accepted": 1},
        {"time": 2.0e-5, "source_increment_accepted": 1},
    ]

    gate = source_convergence_gate_metrics(summary, history)

    assert int(gate["source_convergence_gate_passed"]) == 1
    assert int(gate["source_convergence_tangent_required"]) == 1
    assert int(gate["source_convergence_tangent_used_when_required"]) == 1

    failed = dict(summary)
    failed["source_step_converged_count"] = 1
    failed_gate = source_convergence_gate_metrics(failed, history)

    assert int(failed_gate["source_convergence_gate_passed"]) == 0
    assert int(failed_gate["source_convergence_converged_count_matches_accepted"]) == 0


def test_source_active_set_line_search_gate_requires_stable_accepted_tracking() -> None:
    summary = {
        "source_active_set_line_search": True,
        "source_contact_active_set_stability": True,
        "source_line_search_trial_count": 4,
        "source_line_search_unstable_count": 0,
        "source_accepted_tracking_commit_count": 2,
        "final_active_contact_samples": 2,
    }
    history = [
        {
            "active_contact_region_count": 2,
            "contact_active_area": 1.0,
            "contact_region_normal_force": 5.0,
            "source_line_search_trial_count": 2,
            "source_line_search_reduced_count": 1,
            "source_line_search_stable_count": 1,
            "source_line_search_unstable_count": 0,
            "source_line_search_last_alpha": 0.5,
            "source_accepted_tracking_committed": 2,
        }
    ]

    gate = source_active_set_line_search_gate_metrics(summary, history)

    assert int(gate["source_active_set_line_search_gate_passed"]) == 1
    assert int(gate["source_active_set_line_search_row_commit_ok"]) == 1

    unstable_history = [dict(history[0])]
    unstable_history[0]["source_line_search_unstable_count"] = 1
    failed_unstable = source_active_set_line_search_gate_metrics(summary, unstable_history)

    assert int(failed_unstable["source_active_set_line_search_gate_passed"]) == 0
    assert int(failed_unstable["source_active_set_line_search_row_unstable_ok"]) == 0

    missing_commit = dict(summary)
    missing_commit["source_accepted_tracking_commit_count"] = 0
    failed_commit = source_active_set_line_search_gate_metrics(missing_commit, history)

    assert int(failed_commit["source_active_set_line_search_gate_passed"]) == 0
    assert int(failed_commit["source_active_set_line_search_summary_commit_ok"]) == 0


def test_constraint_region_tangent_gate_requires_consistent_active_rows() -> None:
    summary = {
        "source_constraint_region_tangent_solve_count": 2,
        "source_constraint_region_tangent_active_rows_sum": 4,
        "source_constraint_region_tangent_j_nnz_sum": 18,
        "source_constraint_region_tangent_scale_sum": 12.0,
    }
    history = [
        {
            "active_contact_region_count": 2,
            "contact_active_area": 3.0,
            "contact_region_normal_force": 5.0,
            "contact_tangent_source": "constraint_region_arrays",
            "contact_tangent_gap_jacobian_source": "constraint_region_fixed_payload",
            "contact_tangent_pressure_derivative": "linear_penalty_active_set",
            "contact_tangent_sign_convention": "d(-contact_force)/du",
            "contact_tangent_fixed_active_set": 1,
            "contact_tangent_active_region_count": 2,
            "contact_tangent_scale_sum": 6.0,
            "source_constraint_region_tangent_solve_count": 1,
        }
    ]

    gate = constraint_region_tangent_gate_metrics(summary, history)

    assert int(gate["constraint_region_tangent_gate_passed"]) == 1
    assert int(gate["constraint_region_tangent_row_active_count_ok"]) == 1
    assert int(gate["constraint_region_tangent_row_sign_convention_ok"]) == 1

    sample_tangent = [dict(history[0])]
    sample_tangent[0]["contact_tangent_source"] = "sample_arrays"
    failed = constraint_region_tangent_gate_metrics(summary, sample_tangent)

    assert int(failed["constraint_region_tangent_gate_passed"]) == 0
    assert int(failed["constraint_region_tangent_row_source_ok"]) == 0

    wrong_sign = [dict(history[0])]
    wrong_sign[0]["contact_tangent_sign_convention"] = "d(contact_force)/du"
    failed_sign = constraint_region_tangent_gate_metrics(summary, wrong_sign)

    assert int(failed_sign["constraint_region_tangent_gate_passed"]) == 0
    assert int(failed_sign["constraint_region_tangent_row_sign_convention_ok"]) == 0


def test_constraint_region_contact_law_gate_rejects_independent_sample_penalty() -> None:
    history = [
        {
            "active_contact_region_count": 2,
            "contact_active_area": 1.0,
            "contact_region_normal_force": 5.0,
            "contact_constraint_law_source": "slave_node_region_constraint",
            "contact_constraint_open_closed_source": "signed_area_average_region_gap",
            "contact_constraint_active_status_source": "aggregated_region_gap",
            "contact_constraint_normal_source": "area_average_region_normal",
            "contact_constraint_region_area_source": "slave_shape_tributary_area",
            "contact_constraint_force_distribution": "region_area_slave_shape_master_payload",
            "contact_constraint_secondary_node_regions_present": 1,
            "contact_constraint_master_payload_present": 1,
            "contact_constraint_area_positive": 1,
            "contact_constraint_independent_quadrature_penalty_disabled": 1,
            "contact_constraint_raw_sample_count": 8,
            "contact_constraint_region_count": 2,
        }
    ]

    gate = constraint_region_contact_law_gate_metrics(history)

    assert int(gate["constraint_region_contact_law_gate_passed"]) == 1
    assert float(gate["constraint_region_contact_law_max_raw_to_region_ratio"]) == 4.0

    independent_sample = [dict(history[0])]
    independent_sample[0]["contact_constraint_law_source"] = "none"
    independent_sample[0]["contact_constraint_independent_quadrature_penalty_disabled"] = 0
    failed = constraint_region_contact_law_gate_metrics(independent_sample)

    assert int(failed["constraint_region_contact_law_gate_passed"]) == 0
    assert int(failed["constraint_region_contact_law_row_law_ok"]) == 0
    assert int(failed["constraint_region_contact_law_no_independent_quadrature_penalty"]) == 0


def test_path_tracking_gate_requires_accepted_state_continuity() -> None:
    history = [
        {
            "active_contact_region_count": 2,
            "contact_active_area": 1.0,
            "contact_region_normal_force": 1.0,
            "contact_path_cache_hit_fraction": 0.0,
            "contact_path_cache_match_fraction": 0.0,
            "contact_master_face_switch_fraction": 0.0,
            "contact_active_region_jaccard": 0.0,
            "contact_active_region_persistence_fraction": 0.0,
            "contact_master_barycentric_drift_max": 0.0,
        },
        {
            "active_contact_region_count": 2,
            "contact_active_area": 1.0,
            "contact_region_normal_force": 1.0,
            "contact_path_cache_hit_fraction": 1.0,
            "contact_path_cache_match_fraction": 0.5,
            "contact_master_face_switch_fraction": 0.25,
            "contact_active_region_jaccard": 0.75,
            "contact_active_region_persistence_fraction": 1.0,
            "contact_master_barycentric_drift_max": 0.1,
        },
    ]

    gate = path_tracking_gate_metrics(history, min_cache_match_fraction=0.25)

    assert int(gate["path_tracking_gate_passed"]) == 1
    assert int(gate["path_tracking_continuity_observable"]) == 1

    random_jump = [dict(row) for row in history]
    random_jump[1]["contact_path_cache_hit_fraction"] = 0.25
    random_jump[1]["contact_master_face_switch_fraction"] = 1.0
    failed = path_tracking_gate_metrics(random_jump, min_cache_match_fraction=0.25)

    assert int(failed["path_tracking_gate_passed"]) == 0
    assert int(failed["path_tracking_cache_hit_gate_passed"]) == 0
    assert int(failed["path_tracking_face_switch_gate_passed"]) == 0


def test_full_gear_entry_gate_requires_region_totals_before_clouds() -> None:
    summary = {
        "tooth_patch_region_gate_passed": 1,
        "tooth_patch_ready_for_cropped_gear_patch": 1,
        "cropped_patch_gate_passed": 1,
        "cropped_patch_pressure_stress_gate_passed": 1,
        "cropped_patch_path_tracking_gate_passed": 1,
        "cropped_patch_active_region_continuity_gate_passed": 1,
        "source_trial_gate_passed": 1,
        "source_convergence_gate_passed": 1,
        "source_active_set_line_search_gate_passed": 1,
        "constraint_region_contact_law_gate_passed": 1,
        "constraint_region_tangent_gate_passed": 1,
        "contact_total_gate_passed": 1,
        "path_tracking_gate_passed": 1,
        "path_tracking_continuity_observable": 1,
        "contact_total_nodal_cpress_deferred": 1,
        "contact_total_no_nodal_priority_columns": 1,
        "contact_total_active_contact_present": 1,
        "contact_total_path_columns_present": 1,
        "source_convergence_final_time_matches_duration": 1,
        "source_convergence_no_unstable_or_unconverged_accepted": 1,
        "source_accepted_increment_count": 10,
        "sfc_increment_count": 10,
    }

    gate = full_gear_entry_gate_metrics(summary)

    assert int(gate["full_gear_entry_gate_passed"]) == 1
    assert int(gate["full_gear_entry_patch_ladder_gate_passed"]) == 1
    assert int(gate["full_gear_entry_source_active_set_line_search_gate_passed"]) == 1
    assert int(gate["full_gear_entry_contact_window_gate_passed"]) == 1
    assert int(gate["full_gear_entry_ready_for_nodal_contact_outputs"]) == 1
    assert int(gate["full_gear_entry_ready_for_strict_sync_window"]) == 1

    missing_patch = dict(summary)
    missing_patch["tooth_patch_region_gate_passed"] = 0
    failed_patch = full_gear_entry_gate_metrics(missing_patch)

    assert int(failed_patch["full_gear_entry_gate_passed"]) == 0
    assert int(failed_patch["full_gear_entry_patch_ladder_gate_passed"]) == 0

    missing_totals = dict(summary)
    missing_totals["contact_total_gate_passed"] = 0
    failed = full_gear_entry_gate_metrics(missing_totals)

    assert int(failed["full_gear_entry_gate_passed"]) == 0
    assert int(failed["full_gear_entry_ready_for_stress_cloud_comparison"]) == 0

    missing_path = dict(summary)
    missing_path["path_tracking_gate_passed"] = 0
    failed_path = full_gear_entry_gate_metrics(missing_path)

    assert int(failed_path["full_gear_entry_gate_passed"]) == 0
    assert int(failed_path["full_gear_entry_path_tracking_gate_passed"]) == 0

    unobservable_path = dict(summary)
    unobservable_path["path_tracking_continuity_observable"] = 0
    failed_observable = full_gear_entry_gate_metrics(unobservable_path)

    assert int(failed_observable["full_gear_entry_gate_passed"]) == 0
    assert int(failed_observable["full_gear_entry_path_tracking_continuity_observable"]) == 0

    missing_law = dict(summary)
    missing_law["constraint_region_contact_law_gate_passed"] = 0
    failed_law = full_gear_entry_gate_metrics(missing_law)

    assert int(failed_law["full_gear_entry_gate_passed"]) == 0
    assert int(failed_law["full_gear_entry_constraint_region_contact_law_gate_passed"]) == 0

    missing_line_search = dict(summary)
    missing_line_search["source_active_set_line_search_gate_passed"] = 0
    failed_line_search = full_gear_entry_gate_metrics(missing_line_search)

    assert int(failed_line_search["full_gear_entry_gate_passed"]) == 0
    assert int(failed_line_search["full_gear_entry_source_active_set_line_search_gate_passed"]) == 0


def test_full_gear_entry_gate_blocks_contact_outputs_without_active_contact() -> None:
    summary = {
        "tooth_patch_region_gate_passed": 1,
        "tooth_patch_ready_for_cropped_gear_patch": 1,
        "cropped_patch_gate_passed": 1,
        "cropped_patch_pressure_stress_gate_passed": 1,
        "cropped_patch_path_tracking_gate_passed": 1,
        "cropped_patch_active_region_continuity_gate_passed": 1,
        "source_trial_gate_passed": 1,
        "source_convergence_gate_passed": 1,
        "source_active_set_line_search_gate_passed": 1,
        "constraint_region_contact_law_gate_passed": 1,
        "constraint_region_tangent_gate_passed": 1,
        "contact_total_gate_passed": 1,
        "path_tracking_gate_passed": 1,
        "contact_total_nodal_cpress_deferred": 1,
        "contact_total_no_nodal_priority_columns": 1,
        "contact_total_active_contact_present": 0,
        "contact_total_path_columns_present": 1,
        "source_convergence_final_time_matches_duration": 1,
        "source_convergence_no_unstable_or_unconverged_accepted": 1,
        "source_accepted_increment_count": 200,
        "sfc_increment_count": 200,
    }

    gate = full_gear_entry_gate_metrics(summary)

    assert int(gate["full_gear_entry_gate_passed"]) == 1
    assert int(gate["full_gear_entry_ready_for_strict_sync_window"]) == 1
    assert int(gate["full_gear_entry_contact_window_gate_passed"]) == 0
    assert int(gate["full_gear_entry_ready_for_nodal_contact_outputs"]) == 0
    assert int(gate["full_gear_entry_ready_for_stress_cloud_comparison"]) == 0


def test_full_gear_patch_prerequisite_summary_loader_feeds_entry_gate(tmp_path: Path) -> None:
    tooth_summary = tmp_path / "tooth_patch_region_summary.csv"
    tooth_summary.write_text(
        "\n".join(
            [
                "tooth_patch_region_gate_passed,tooth_patch_ready_for_cropped_gear_patch,"
                "tooth_patch_path_tracking_gate_passed,unrelated_case_metric",
                "1,1,1,999",
            ]
        ),
        encoding="utf-8",
    )
    cropped_summary = tmp_path / "cropped_patch_region_summary.csv"
    cropped_summary.write_text(
        "\n".join(
            [
                "cropped_patch_gate_passed,cropped_patch_pressure_stress_gate_passed,"
                "cropped_patch_path_tracking_gate_passed,cropped_patch_active_region_continuity_gate_passed",
                "1,1,1,1",
            ]
        ),
        encoding="utf-8",
    )

    patch_evidence = load_patch_prerequisite_summary([tooth_summary, cropped_summary])

    assert int(patch_evidence["full_gear_patch_prerequisite_summary_count"]) == 2
    assert "unrelated_case_metric" not in patch_evidence
    assert int(patch_evidence["tooth_patch_region_gate_passed"]) == 1
    assert int(patch_evidence["cropped_patch_gate_passed"]) == 1

    summary = {
        **patch_evidence,
        "source_trial_gate_passed": 1,
        "source_convergence_gate_passed": 1,
        "source_active_set_line_search_gate_passed": 1,
        "constraint_region_contact_law_gate_passed": 1,
        "constraint_region_tangent_gate_passed": 1,
        "contact_total_gate_passed": 1,
        "path_tracking_gate_passed": 1,
        "path_tracking_continuity_observable": 1,
        "contact_total_nodal_cpress_deferred": 1,
        "contact_total_no_nodal_priority_columns": 1,
        "contact_total_active_contact_present": 1,
        "contact_total_path_columns_present": 1,
        "source_convergence_final_time_matches_duration": 1,
        "source_convergence_no_unstable_or_unconverged_accepted": 1,
        "source_accepted_increment_count": 10,
        "sfc_increment_count": 10,
    }

    gate = full_gear_entry_gate_metrics(summary)

    assert int(gate["full_gear_entry_gate_passed"]) == 1
    assert int(gate["full_gear_entry_patch_ladder_gate_passed"]) == 1


def test_full_gear_entry_gate_reports_short_strict_sync_window() -> None:
    summary = {
        "tooth_patch_region_gate_passed": 1,
        "tooth_patch_ready_for_cropped_gear_patch": 1,
        "cropped_patch_gate_passed": 1,
        "cropped_patch_pressure_stress_gate_passed": 1,
        "cropped_patch_path_tracking_gate_passed": 1,
        "cropped_patch_active_region_continuity_gate_passed": 1,
        "source_trial_gate_passed": 1,
        "source_convergence_gate_passed": 1,
        "source_active_set_line_search_gate_passed": 1,
        "constraint_region_contact_law_gate_passed": 1,
        "constraint_region_tangent_gate_passed": 1,
        "contact_total_gate_passed": 1,
        "contact_total_nodal_cpress_deferred": 1,
        "contact_total_no_nodal_priority_columns": 1,
        "contact_total_active_contact_present": 0,
        "contact_total_path_columns_present": 0,
        "source_convergence_final_time_matches_duration": 1,
        "source_convergence_no_unstable_or_unconverged_accepted": 1,
        "source_accepted_increment_count": 1,
        "sfc_increment_count": 1,
    }

    gate = full_gear_entry_gate_metrics(summary)

    assert int(gate["full_gear_entry_gate_passed"]) == 1
    assert int(gate["full_gear_entry_strict_sync_min_steps_met"]) == 0
    assert int(gate["full_gear_entry_ready_for_strict_sync_window"]) == 0
    assert int(gate["full_gear_entry_contact_window_gate_passed"]) == 0
    assert int(gate["full_gear_entry_ready_for_nodal_contact_outputs"]) == 0


def test_full_gear_evidence_ladder_blocks_clouds_until_strict_sync() -> None:
    summary = {
        "full_gear_entry_patch_ladder_gate_passed": 1,
        "source_convergence_gate_passed": 1,
        "source_active_set_line_search_gate_passed": 1,
        "constraint_region_contact_law_gate_passed": 1,
        "constraint_region_tangent_gate_passed": 1,
        "path_tracking_gate_passed": 1,
        "contact_total_gate_passed": 1,
        "full_gear_entry_gate_passed": 1,
        "full_gear_entry_contact_window_gate_passed": 1,
        "full_gear_entry_ready_for_strict_sync_window": 0,
        "contact_total_priority_metrics": "sfc_contact_total_priority_metrics.csv",
        "sfc_vtk_manifest": "sfc_manifest.csv",
        "abaqus_vtk_manifest": "abaqus_manifest.csv",
        "animation_metric_errors": "sfc_vs_abaqus_vtk_metric_errors.csv",
        "history_metric_errors": "sfc_vs_abaqus_history_metric_errors.csv",
    }

    rows = {row["evidence_stage"]: row for row in full_gear_evidence_ladder_rows(summary)}

    assert int(rows["patch_ladder_prerequisites"]["paper_evidence_allowed"]) == 1
    assert int(rows["region_contact_totals"]["paper_evidence_allowed"]) == 1
    assert int(rows["nodal_cpress_copen"]["paper_evidence_allowed"]) == 1
    assert int(rows["stress_strain_clouds"]["paper_evidence_allowed"]) == 0
    assert rows["stress_strain_clouds"]["blocking_reason"] == "strict_sync_gate_or_animation_metric_missing"


def test_full_gear_evidence_ladder_allows_clouds_after_strict_sync() -> None:
    summary = {
        "full_gear_entry_patch_ladder_gate_passed": 1,
        "source_convergence_gate_passed": 1,
        "source_active_set_line_search_gate_passed": 1,
        "constraint_region_contact_law_gate_passed": 1,
        "constraint_region_tangent_gate_passed": 1,
        "path_tracking_gate_passed": 1,
        "contact_total_gate_passed": 1,
        "full_gear_entry_gate_passed": 1,
        "full_gear_entry_contact_window_gate_passed": 1,
        "full_gear_entry_ready_for_strict_sync_window": 1,
        "contact_total_priority_metrics": "sfc_contact_total_priority_metrics.csv",
        "sfc_vtk_manifest": "sfc_manifest.csv",
        "abaqus_vtk_manifest": "abaqus_manifest.csv",
        "animation_metric_errors": "sfc_vs_abaqus_vtk_metric_errors.csv",
        "history_metric_errors": "sfc_vs_abaqus_history_metric_errors.csv",
    }

    rows = {row["evidence_stage"]: row for row in full_gear_evidence_ladder_rows(summary)}

    assert int(rows["patch_ladder_prerequisites"]["paper_evidence_allowed"]) == 1
    assert int(rows["history_vs_abaqus_manifest_totals"]["paper_evidence_allowed"]) == 1
    assert int(rows["nodal_cpress_copen"]["paper_evidence_allowed"]) == 1
    assert int(rows["stress_strain_clouds"]["paper_evidence_allowed"]) == 1


def test_prepare_source_penalty_deck_uses_standard_linear_penalty_and_strided_output() -> None:
    source = "\n".join(
        [
            "*Heading",
            "*Surface Interaction, name=IntProp-1",
            "1.,",
            "*Friction",
            "0.,",
            "*Surface Behavior, pressure-overclosure=HARD",
            "** INTERACTIONS",
            "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE",
            "_PickedSurf16, _PickedSurf15",
            "*Step, name=Step-1, nlgeom=YES, inc=2000",
            "*Dynamic",
            "1e-05,0.05,1e-10,5e-05",
            "*Output, field, variable=PRESELECT",
            "*Output, history, variable=PRESELECT",
            "*End Step",
        ]
    )

    text = prepare_source_penalty_deck_text(
        source,
        pressure_stiffness=5.0e9,
        frame_stride=5,
        dt=1.0e-5,
        duration=1.0e-4,
    )

    assert "*Surface Behavior, pressure-overclosure=LINEAR" in text
    assert "5.000000000000e+09" in text
    assert "pressure-overclosure=HARD" not in text
    assert "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE" in text
    assert "mechanical constraint=PENALTY" not in text
    assert "1.000000000000e-05,1.000000000000e-04,1.000000000000e-05,1.000000000000e-05" in text
    assert "*Output, field, variable=PRESELECT, frequency=5" in text
    assert "*Output, history, variable=PRESELECT, frequency=5" in text


def test_prepare_source_penalty_deck_can_preserve_automatic_increment_bounds() -> None:
    source = "\n".join(
        [
            "*Surface Behavior, pressure-overclosure=HARD",
            "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE",
            "S2, S1",
            "*Dynamic",
            "1e-05,0.05,1e-10,5e-05",
        ]
    )

    text = prepare_source_penalty_deck_text(
        source,
        pressure_stiffness=5.0e9,
        frame_stride=5,
        dt=1.0e-5,
        duration=1.0e-4,
        fixed_increment=False,
    )

    assert "1.000000000000e-05,1.000000000000e-04,1e-10,5e-05" in text


def test_prepare_source_penalty_deck_can_require_source_timing() -> None:
    source = "\n".join(
        [
            "*Surface Behavior, pressure-overclosure=HARD",
            "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE",
            "S2, S1",
            "*Dynamic",
            "1e-05,0.05,1e-10,5e-05",
        ]
    )

    text = prepare_source_penalty_deck_text(
        source,
        pressure_stiffness=5.0e9,
        frame_stride=2,
        dt=1.0e-5,
        duration=0.05,
        require_source_timing=True,
    )

    assert "1.000000000000e-05,5.000000000000e-02,1.000000000000e-05,1.000000000000e-05" in text


def test_prepare_source_penalty_deck_rejects_short_duration_when_source_timing_required() -> None:
    source = "\n".join(
        [
            "*Surface Behavior, pressure-overclosure=HARD",
            "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE",
            "S2, S1",
            "*Dynamic",
            "1e-05,0.05,1e-10,5e-05",
        ]
    )

    import pytest

    with pytest.raises(ValueError, match="requested duration"):
        prepare_source_penalty_deck_text(
            source,
            pressure_stiffness=5.0e9,
            frame_stride=2,
            dt=1.0e-5,
            duration=0.002,
            require_source_timing=True,
        )


def test_prepare_source_penalty_deck_replaces_existing_linear_data() -> None:
    source = "\n".join(
        [
            "*Surface Behavior, pressure-overclosure=LINEAR",
            "1.0e3",
            "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE, mechanical constraint=PENALTY",
            "S2, S1",
        ]
    )

    text = prepare_source_penalty_deck_text(source, pressure_stiffness=7.0e9, frame_stride=2)

    assert "7.000000000000e+09" in text
    assert "1.0e3" not in text
    assert text.count("mechanical constraint=PENALTY") == 1


def test_prepare_source_penalty_deck_can_write_explicit_style_contact_pair_parameter() -> None:
    source = "\n".join(
        [
            "*Surface Behavior, pressure-overclosure=HARD",
            "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE",
            "S2, S1",
        ]
    )

    text = prepare_source_penalty_deck_text(
        source,
        pressure_stiffness=5.0e9,
        frame_stride=1,
        contact_pair_penalty_parameter=True,
    )

    assert "*Contact Pair, interaction=IntProp-1, type=SURFACE TO SURFACE, mechanical constraint=PENALTY" in text


def test_source_penalty_summary_states_external_validation_only(tmp_path: Path) -> None:
    path = tmp_path / "summary.md"

    write_source_penalty_summary(
        path,
        {
            "pressure_stiffness": 5.0e9,
            "vtk_frame_stride": 5,
            "abaqus_analysis_wall_seconds": 1.5,
            "abaqus_reported_wall_seconds": 1.25,
            "abaqus_vtk_manifest": "abaqus_manifest.csv",
            "abaqus_vtk_frame_count": 3,
            "abaqus_vtk_export_frame_stride": 1,
        },
    )

    text = path.read_text(encoding="utf-8")
    assert "validation-only" in text
    assert "frictionless linear penalty pressure-overclosure" in text
    assert "VTK export frame stride: `1`" in text


def test_full_gear_runner_exposes_hht_alpha_parameter(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_parse(_source: Path):
        class Model:
            np = __import__("numpy")
            young = 1.0
            poisson = 0.25
            density = 1.0
            dynamic_duration = 1.0
            dynamic_initial_dt = 0.1
            dynamic_min_dt = 0.1
            dynamic_max_dt = 0.1
            contact_pressure_overclosure = "LINEAR"
            gear1_angular_velocity_z = 2.0
            gear2_torque_z = 3.0
            rp1 = np.asarray([0.0, 0.0, 0.0], dtype=float)
            rp2 = np.asarray([0.0, 0.0, 0.0], dtype=float)
            gear1 = type("Gear", (), {"nodes": np.asarray([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=float)})()
            gear2 = type("Gear", (), {"nodes": np.asarray([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 1.0]], dtype=float)})()
            gear1_contact_faces = np.asarray([[0, 1, 2]], dtype=int)
            gear2_contact_faces = np.asarray([[0, 1, 2]], dtype=int)

        return Model()

    def fake_build(*_args, **_kwargs):
        captured["effective_active_faces_per_body"] = int(_kwargs["active_faces_per_body"])
        captured["active_patch_sweep_radius"] = float(_kwargs["active_patch_sweep_radius"])

        class Patch:
            nodes = __import__("numpy").zeros((1, 3))
            elements = __import__("numpy").zeros((0, 4), dtype=int)
            contact_faces = __import__("numpy").zeros((0, 3), dtype=int)
            support_nodes = __import__("numpy").zeros((1,), dtype=int)
            rp = __import__("numpy").zeros(3)

        class Pair:
            gear1 = Patch()
            gear2 = Patch()
            initial_patch_gap = 0.0

        return Pair()

    def fake_solve(*_args, **kwargs):
        captured["hht_alpha"] = float(kwargs["hht_alpha"])
        captured["tet4_mass_kind"] = str(kwargs["tet4_mass_kind"])
        captured["history_frame_stride"] = int(kwargs["history_frame_stride"])
        captured["source_stress_postprocess"] = str(kwargs["source_stress_postprocess"])
        captured["source_checkpoint_stride"] = int(kwargs["source_checkpoint_stride"])
        return (
            [
                {
                    "time": 0.1,
                    "source_line_search_trial_count": 0,
                    "source_line_search_reduced_count": 0,
                    "source_line_search_stable_count": 0,
                    "source_line_search_unstable_count": 0,
                    "source_line_search_last_alpha": 1.0,
                    "source_accepted_tracking_committed": 0,
                }
            ],
            {
                "status": "completed",
                "nodes": 2,
                "elements": 0,
                "gear1_contact_faces": 0,
                "gear2_contact_faces": 0,
                "gear1_support_nodes": 1,
                "gear2_support_nodes": 1,
                "sfc_wall_seconds": 0.0,
                "source_increment_trial_count": 1,
                "source_rejected_trial_count": 0,
                "source_accepted_increment_count": 1,
                "source_step_converged_count": 1,
                "source_unstable_accepted_count": 0,
                "source_unconverged_accepted_count": 0,
                "source_accepted_contact_response_reuse_count": 1,
                "source_accepted_contact_response_requery_count": 0,
                "source_active_set_line_search": True,
                "source_contact_active_set_stability": True,
                "source_line_search_trial_count": 0,
                "source_line_search_reduced_count": 0,
                "source_line_search_stable_count": 0,
                "source_line_search_unstable_count": 0,
                "source_accepted_tracking_commit_count": 0,
                "source_constraint_region_tangent_solve_count": 0,
                "source_constraint_region_tangent_active_rows_sum": 0,
                "tooth_patch_region_gate_passed": 1,
                "tooth_patch_ready_for_cropped_gear_patch": 1,
                "cropped_patch_gate_passed": 1,
                "cropped_patch_pressure_stress_gate_passed": 1,
                "cropped_patch_path_tracking_gate_passed": 1,
                "cropped_patch_active_region_continuity_gate_passed": 1,
                "source_final_time": 0.1,
                "_source_increment_trial_rows": [
                    {
                        "source_trial_index": 1,
                        "source_trial_end_time": 0.1,
                        "source_trial_accepted": 1,
                        "source_trial_retry_required": 0,
                    }
                ],
            },
        )

    monkeypatch.setattr("validation.run_flexible_gear_full_lagrangian_sdf_comparison.parse_gear_input", fake_parse)
    monkeypatch.setattr("validation.run_flexible_gear_full_lagrangian_sdf_comparison.build_full_active_pair", fake_build)
    monkeypatch.setattr("validation.run_flexible_gear_full_lagrangian_sdf_comparison.solve_sfc_source_drive_pair", fake_solve)
    monkeypatch.setattr(
        "validation.run_flexible_gear_full_lagrangian_sdf_comparison._write_abaqus_alignment_deck",
        lambda *_args, **_kwargs: None,
    )

    _history, summary = run_full_gear(
        source=tmp_path / "dummy.inp",
        out_dir=tmp_path,
        active_faces_per_body=16,
        active_patch_radius_factor=1.0,
        duration=0.1,
        dt=0.1,
        target_overclosure=0.0,
        rotation_rate_z=0.0,
        pressure_stiffness=1.0,
        hard_max_iterations=1,
        contact_mode="penalty",
        run_abaqus=False,
        drive_mode="source_inp",
        hht_alpha=-0.05,
        tet4_mass_kind="consistent",
        history_frame_stride=3,
        source_stress_postprocess="finite_stvk_visual",
        source_dynamic_contact_window=True,
        source_checkpoint_path=tmp_path / "checkpoint.npz",
        resume_source_checkpoint=True,
        source_checkpoint_stride=7,
    )

    assert captured["hht_alpha"] == -0.05
    assert captured["tet4_mass_kind"] == "consistent"
    assert captured["history_frame_stride"] == 3
    assert captured["source_stress_postprocess"] == "finite_stvk_visual"
    assert captured["source_checkpoint_stride"] == 7
    assert captured["effective_active_faces_per_body"] == 16
    assert captured["active_patch_sweep_radius"] > 0.0
    assert int(summary["requested_active_faces_per_body"]) == 16
    assert int(summary["effective_active_faces_per_body"]) == 16
    assert int(summary["source_dynamic_contact_window"]) == 1
    assert int(summary["source_dynamic_contact_window_applied"]) == 1
    assert float(summary["source_dynamic_contact_window_sweep_radius"]) > 0.0
    assert "source_increment_trials" in summary
    assert "source_increment_trial_gate" in summary
    assert "source_convergence_gate" in summary
    assert "source_active_set_line_search_gate" in summary
    assert "constraint_region_contact_law_gate" in summary
    assert "constraint_region_tangent_gate" in summary
    assert "path_tracking_gate" in summary
    assert "contact_total_gate" in summary
    assert "full_gear_entry_gate" in summary
    assert "full_gear_evidence_ladder" in summary
    assert int(summary["source_trial_gate_passed"]) == 1
    assert int(summary["source_convergence_gate_passed"]) == 1
    assert int(summary["source_active_set_line_search_gate_passed"]) == 1
    assert int(summary["constraint_region_contact_law_gate_passed"]) == 1
    assert int(summary["constraint_region_tangent_gate_passed"]) == 1
    assert int(summary["path_tracking_gate_passed"]) == 1
    assert int(summary["contact_total_gate_passed"]) == 1
    assert int(summary["full_gear_entry_gate_passed"]) == 1
    assert int(summary["full_gear_entry_ready_for_strict_sync_window"]) == 0
    trial_csv = tmp_path / "sfc_source_increment_trials.csv"
    trial_gate_csv = tmp_path / "sfc_source_increment_trial_gate.csv"
    convergence_gate_csv = tmp_path / "sfc_source_convergence_gate.csv"
    line_search_gate_csv = tmp_path / "sfc_source_active_set_line_search_gate.csv"
    contact_law_gate_csv = tmp_path / "sfc_constraint_region_contact_law_gate.csv"
    tangent_gate_csv = tmp_path / "sfc_constraint_region_tangent_gate.csv"
    path_tracking_gate_csv = tmp_path / "sfc_path_tracking_gate.csv"
    contact_total_gate_csv = tmp_path / "sfc_contact_total_gate.csv"
    entry_gate_csv = tmp_path / "sfc_full_gear_entry_gate.csv"
    evidence_ladder_csv = tmp_path / "sfc_full_gear_evidence_ladder.csv"
    assert trial_csv.exists()
    assert trial_gate_csv.exists()
    assert convergence_gate_csv.exists()
    assert line_search_gate_csv.exists()
    assert contact_law_gate_csv.exists()
    assert tangent_gate_csv.exists()
    assert path_tracking_gate_csv.exists()
    assert contact_total_gate_csv.exists()
    assert entry_gate_csv.exists()
    assert evidence_ladder_csv.exists()
    assert "source_trial_accepted" in trial_csv.read_text(encoding="utf-8")
    assert "source_trial_gate_passed" in trial_gate_csv.read_text(encoding="utf-8")
    assert "source_convergence_gate_passed" in convergence_gate_csv.read_text(encoding="utf-8")
    assert "source_active_set_line_search_gate_passed" in line_search_gate_csv.read_text(encoding="utf-8")
    assert "constraint_region_contact_law_gate_passed" in contact_law_gate_csv.read_text(encoding="utf-8")
    assert "constraint_region_tangent_gate_passed" in tangent_gate_csv.read_text(encoding="utf-8")
    assert "path_tracking_gate_passed" in path_tracking_gate_csv.read_text(encoding="utf-8")
    assert "contact_total_gate_passed" in contact_total_gate_csv.read_text(encoding="utf-8")
    assert "full_gear_entry_gate_passed" in entry_gate_csv.read_text(encoding="utf-8")
    evidence_text = evidence_ladder_csv.read_text(encoding="utf-8")
    assert "region_contact_totals" in evidence_text
    assert "stress_strain_clouds" in evidence_text
