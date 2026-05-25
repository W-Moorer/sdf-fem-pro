from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_two_block_sliding_region_validation import run_validation, two_block_sliding_region_gate_metrics


def test_two_block_sliding_region_validation_writes_totals_before_clouds(tmp_path: Path) -> None:
    rows, summary = run_validation(out_dir=tmp_path, resolution=1, steps=5, penetration=0.01, lateral_shift=0.10)

    assert len(rows) == 5
    assert summary["status"] == "completed"
    assert int(summary["constraint_regions_final"]) > 0
    assert float(summary["normal_force_final"]) >= 0.0
    assert int(summary["path_tracking_gate_passed"]) == 1
    assert int(summary["flat_punch_prerequisite_gate_passed"]) == 1
    assert int(summary["two_block_sliding_region_gate_passed"]) == 1
    assert int(summary["two_block_ready_for_cropped_gear_patch"]) == 1
    assert int(summary["two_block_nodal_cpress_deferred"]) == 1
    assert int(summary["two_block_constraint_region_law_gate_passed"]) == 1
    assert int(summary["two_block_force_consistency_gate_passed"]) == 1
    assert int(summary["two_block_virtual_work_consistency_gate_passed"]) == 1
    assert int(summary["contact_total_gate_passed"]) == 1
    assert int(summary["contact_total_secondary_pressure_recovery_from_region"]) == 1
    assert float(summary["active_region_jaccard_min"]) >= float(summary["path_tracking_min_active_region_jaccard_threshold"])
    assert float(summary["path_cache_hit_fraction_min_after_first"]) >= float(summary["path_tracking_min_cache_hit_threshold"])
    assert float(summary["path_cache_match_fraction_min_after_first"]) >= float(summary["path_tracking_min_cache_match_threshold"])
    assert float(summary["master_face_switch_fraction_max"]) <= float(summary["path_tracking_max_face_switch_threshold"])
    assert float(summary["master_barycentric_drift_max"]) <= float(summary["path_tracking_max_barycentric_drift_threshold"])

    totals = tmp_path / "two_block_sliding_contact_totals.csv"
    continuity = tmp_path / "two_block_sliding_region_continuity.csv"
    tracking = tmp_path / "two_block_sliding_path_tracking.csv"
    gate = tmp_path / "two_block_sliding_region_gate.csv"
    report = tmp_path / "two_block_sliding_region_summary.md"
    assert totals.exists()
    assert continuity.exists()
    assert tracking.exists()
    assert gate.exists()
    assert report.exists()

    with totals.open(newline="", encoding="utf-8") as handle:
        total_rows = list(csv.DictReader(handle))
    assert "normal_force" in total_rows[0]
    assert "contact_virtual_work" in total_rows[0]
    assert "contact_region_energy" in total_rows[0]
    assert "contact_active_area" in total_rows[0]
    assert "contact_active_region_jaccard" in total_rows[0]
    assert "contact_master_face_switch_fraction" in total_rows[0]
    assert "contact_path_cache_hit_fraction" in total_rows[0]
    assert "contact_master_barycentric_drift_max" in total_rows[0]
    assert "nodal_cpress_deferred" in total_rows[0]
    assert "contact_constraint_open_closed_source" in total_rows[0]
    assert "contact_constraint_force_distribution" in total_rows[0]
    assert total_rows[0]["contact_secondary_pressure_recovery_source"] == "constraint_region"
    assert "max_contact_secondary_pressure_nodeavg" not in total_rows[0]

    with continuity.open(newline="", encoding="utf-8") as handle:
        continuity_rows = list(csv.DictReader(handle))
    assert len(continuity_rows) == 5
    assert "contact_active_region_persistence_fraction" in continuity_rows[0]
    assert "contact_active_region_jaccard" in continuity_rows[0]

    with tracking.open(newline="", encoding="utf-8") as handle:
        tracking_rows = list(csv.DictReader(handle))
    assert len(tracking_rows) == 5
    assert "contact_master_face_switch_count" in tracking_rows[0]
    assert "contact_path_cache_match_fraction" in tracking_rows[0]
    assert "contact_master_barycentric_drift_mean" in tracking_rows[0]
    assert float(summary["path_cache_hit_fraction_min_after_first"]) > 0.0

    report_text = report.read_text(encoding="utf-8")
    assert "accepted-state path tracking gate: PASS" in report_text
    assert "flat-punch prerequisite gate: PASS" in report_text
    assert "two-block sliding region gate: PASS" in report_text
    assert "two_block_sliding_region_gate_passed" in gate.read_text(encoding="utf-8")


def test_two_block_gate_requires_flat_punch_and_region_law() -> None:
    row = {
        "normal_force": 10.0,
        "contact_region_normal_force": 10.0,
        "contact_virtual_work": 4.0,
        "contact_region_virtual_work": 4.0,
        "contact_region_energy": 2.0,
        "nodal_cpress_deferred": 1,
        "active_contact_region_count": 2,
        "contact_active_area": 1.0,
        "contact_path_cache_hit_fraction": 1.0,
        "path_tracking_min_cache_hit_threshold": 0.999,
        "contact_path_cache_match_fraction": 1.0,
        "contact_master_face_switch_fraction": 0.0,
        "contact_active_region_jaccard": 1.0,
        "contact_active_region_persistence_fraction": 1.0,
        "contact_master_barycentric_drift_max": 0.0,
        "contact_constraint_law_source": "slave_node_region_constraint",
        "contact_constraint_open_closed_source": "signed_area_average_region_gap",
        "contact_constraint_normal_source": "area_average_region_normal",
        "contact_constraint_force_distribution": "region_area_slave_shape_master_payload",
        "contact_constraint_independent_quadrature_penalty_disabled": 1,
        "contact_secondary_pressure_recovery_source": "constraint_region",
    }
    rows = [dict(row), dict(row)]

    passed = two_block_sliding_region_gate_metrics(
        rows,
        flat_punch_summary={"flat_punch_rf_penetration_gate_passed": 1},
    )

    assert int(passed["two_block_sliding_region_gate_passed"]) == 1

    missing_flat = two_block_sliding_region_gate_metrics(
        rows,
        flat_punch_summary={"flat_punch_rf_penetration_gate_passed": 0},
    )
    assert int(missing_flat["two_block_sliding_region_gate_passed"]) == 0
    assert int(missing_flat["flat_punch_prerequisite_gate_passed"]) == 0

    wrong_law_rows = [dict(row), dict(row)]
    wrong_law_rows[1]["contact_constraint_law_source"] = "independent_quadrature_penalty"
    wrong_law = two_block_sliding_region_gate_metrics(
        wrong_law_rows,
        flat_punch_summary={"flat_punch_rf_penetration_gate_passed": 1},
    )

    assert int(wrong_law["two_block_sliding_region_gate_passed"]) == 0
    assert int(wrong_law["two_block_constraint_region_law_gate_passed"]) == 0
