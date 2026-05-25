from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.prepare_flexible_gear_source_penalty_deck import prepare_source_penalty_deck_text
from validation.run_flexible_gear_source_penalty_abaqus import write_source_penalty_summary
from validation.run_flexible_gear_full_lagrangian_sdf_comparison import (
    full_gear_evidence_ladder_rows,
    full_gear_entry_gate_metrics,
    run_full_gear,
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


def test_full_gear_entry_gate_requires_region_totals_before_clouds() -> None:
    summary = {
        "source_trial_gate_passed": 1,
        "source_convergence_gate_passed": 1,
        "contact_total_gate_passed": 1,
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
    assert int(gate["full_gear_entry_ready_for_nodal_contact_outputs"]) == 1
    assert int(gate["full_gear_entry_ready_for_strict_sync_window"]) == 1

    missing_totals = dict(summary)
    missing_totals["contact_total_gate_passed"] = 0
    failed = full_gear_entry_gate_metrics(missing_totals)

    assert int(failed["full_gear_entry_gate_passed"]) == 0
    assert int(failed["full_gear_entry_ready_for_stress_cloud_comparison"]) == 0


def test_full_gear_entry_gate_reports_short_strict_sync_window() -> None:
    summary = {
        "source_trial_gate_passed": 1,
        "source_convergence_gate_passed": 1,
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


def test_full_gear_evidence_ladder_blocks_clouds_until_strict_sync() -> None:
    summary = {
        "source_convergence_gate_passed": 1,
        "contact_total_gate_passed": 1,
        "full_gear_entry_gate_passed": 1,
        "full_gear_entry_ready_for_strict_sync_window": 0,
        "contact_total_priority_metrics": "sfc_contact_total_priority_metrics.csv",
        "sfc_vtk_manifest": "sfc_manifest.csv",
        "abaqus_vtk_manifest": "abaqus_manifest.csv",
        "animation_metric_errors": "sfc_vs_abaqus_vtk_metric_errors.csv",
        "history_metric_errors": "sfc_vs_abaqus_history_metric_errors.csv",
    }

    rows = {row["evidence_stage"]: row for row in full_gear_evidence_ladder_rows(summary)}

    assert int(rows["region_contact_totals"]["paper_evidence_allowed"]) == 1
    assert int(rows["nodal_cpress_copen"]["paper_evidence_allowed"]) == 1
    assert int(rows["stress_strain_clouds"]["paper_evidence_allowed"]) == 0
    assert rows["stress_strain_clouds"]["blocking_reason"] == "strict_sync_gate_or_animation_metric_missing"


def test_full_gear_evidence_ladder_allows_clouds_after_strict_sync() -> None:
    summary = {
        "source_convergence_gate_passed": 1,
        "contact_total_gate_passed": 1,
        "full_gear_entry_gate_passed": 1,
        "full_gear_entry_ready_for_strict_sync_window": 1,
        "contact_total_priority_metrics": "sfc_contact_total_priority_metrics.csv",
        "sfc_vtk_manifest": "sfc_manifest.csv",
        "abaqus_vtk_manifest": "abaqus_manifest.csv",
        "animation_metric_errors": "sfc_vs_abaqus_vtk_metric_errors.csv",
        "history_metric_errors": "sfc_vs_abaqus_history_metric_errors.csv",
    }

    rows = {row["evidence_stage"]: row for row in full_gear_evidence_ladder_rows(summary)}

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

        return Model()

    def fake_build(*_args, **_kwargs):
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
            [{"time": 0.1}],
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
                "source_constraint_region_tangent_solve_count": 0,
                "source_constraint_region_tangent_active_rows_sum": 0,
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
        active_faces_per_body=0,
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
        source_checkpoint_path=tmp_path / "checkpoint.npz",
        resume_source_checkpoint=True,
        source_checkpoint_stride=7,
    )

    assert captured["hht_alpha"] == -0.05
    assert captured["tet4_mass_kind"] == "consistent"
    assert captured["history_frame_stride"] == 3
    assert captured["source_stress_postprocess"] == "finite_stvk_visual"
    assert captured["source_checkpoint_stride"] == 7
    assert "source_increment_trials" in summary
    assert "source_increment_trial_gate" in summary
    assert "source_convergence_gate" in summary
    assert "contact_total_gate" in summary
    assert "full_gear_entry_gate" in summary
    assert "full_gear_evidence_ladder" in summary
    assert int(summary["source_trial_gate_passed"]) == 1
    assert int(summary["source_convergence_gate_passed"]) == 1
    assert int(summary["contact_total_gate_passed"]) == 1
    assert int(summary["full_gear_entry_gate_passed"]) == 1
    assert int(summary["full_gear_entry_ready_for_strict_sync_window"]) == 0
    trial_csv = tmp_path / "sfc_source_increment_trials.csv"
    trial_gate_csv = tmp_path / "sfc_source_increment_trial_gate.csv"
    convergence_gate_csv = tmp_path / "sfc_source_convergence_gate.csv"
    contact_total_gate_csv = tmp_path / "sfc_contact_total_gate.csv"
    entry_gate_csv = tmp_path / "sfc_full_gear_entry_gate.csv"
    evidence_ladder_csv = tmp_path / "sfc_full_gear_evidence_ladder.csv"
    assert trial_csv.exists()
    assert trial_gate_csv.exists()
    assert convergence_gate_csv.exists()
    assert contact_total_gate_csv.exists()
    assert entry_gate_csv.exists()
    assert evidence_ladder_csv.exists()
    assert "source_trial_accepted" in trial_csv.read_text(encoding="utf-8")
    assert "source_trial_gate_passed" in trial_gate_csv.read_text(encoding="utf-8")
    assert "source_convergence_gate_passed" in convergence_gate_csv.read_text(encoding="utf-8")
    assert "contact_total_gate_passed" in contact_total_gate_csv.read_text(encoding="utf-8")
    assert "full_gear_entry_gate_passed" in entry_gate_csv.read_text(encoding="utf-8")
    evidence_text = evidence_ladder_csv.read_text(encoding="utf-8")
    assert "region_contact_totals" in evidence_text
    assert "stress_strain_clouds" in evidence_text
