from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_tooth_patch_region_constraint_validation import (  # noqa: E402
    run_validation,
    tooth_patch_region_gate_metrics,
)


def test_tooth_patch_region_constraint_gate_writes_totals_before_clouds(tmp_path: Path) -> None:
    rows, summary = run_validation(
        out_dir=tmp_path,
        steps=3,
        lateral_shift=0.04,
        run_two_block_prerequisite=True,
    )

    assert len(rows) == 3
    assert summary["status"] == "completed"
    assert int(summary["two_block_sliding_prerequisite_gate_passed"]) == 1
    assert int(summary["tooth_patch_region_gate_passed"]) == 1
    assert int(summary["tooth_patch_ready_for_cropped_gear_patch"]) == 1
    assert int(summary["tooth_patch_constraint_region_law_gate_passed"]) == 1
    assert int(summary["tooth_patch_q4_master_payload_gate_passed"]) == 1
    assert int(summary["tooth_patch_path_tracking_gate_passed"]) == 1
    assert float(summary["tooth_patch_path_cache_hit_fraction_min_after_first"]) >= 0.999
    assert float(summary["tooth_patch_path_cache_match_fraction_min_after_first"]) >= 0.999

    totals = tmp_path / "tooth_patch_region_totals.csv"
    tracking = tmp_path / "tooth_patch_region_path_tracking.csv"
    gate = tmp_path / "tooth_patch_region_gate.csv"
    report = tmp_path / "tooth_patch_region_summary.md"
    assert totals.exists()
    assert tracking.exists()
    assert gate.exists()
    assert report.exists()

    with totals.open(newline="", encoding="utf-8") as handle:
        total_rows = list(csv.DictReader(handle))
    assert "nodal_cpress_deferred" in total_rows[0]
    assert "contact_region_normal_force" in total_rows[0]
    assert "contact_constraint_open_closed_source" in total_rows[0]
    assert "q4_master_payload_width" in total_rows[0]
    assert int(total_rows[0]["q4_master_payload_width"]) == 4
    assert "max_contact_secondary_pressure_nodeavg" not in total_rows[0]

    report_text = report.read_text(encoding="utf-8")
    assert "tooth-patch region gate: PASS" in report_text
    assert "Q4 master payload gate: PASS" in report_text


def test_tooth_patch_region_gate_rejects_missing_prerequisite_or_q4_payload() -> None:
    row = {
        "normal_force": 10.0,
        "contact_region_normal_force": 10.0,
        "contact_virtual_work": 4.0,
        "contact_region_virtual_work": 4.0,
        "contact_region_energy": 2.0,
        "nodal_cpress_deferred": 1,
        "active_contact_region_count": 2,
        "contact_active_area": 1.0,
        "q4_master_payload_width": 4,
        "contact_path_cache_hit_fraction": 1.0,
        "contact_path_cache_match_fraction": 1.0,
        "contact_master_face_switch_fraction": 0.0,
        "contact_active_region_jaccard": 1.0,
        "contact_master_barycentric_drift_max": 0.0,
        "contact_constraint_law_source": "slave_node_region_constraint",
        "contact_constraint_open_closed_source": "signed_area_average_region_gap",
        "contact_constraint_normal_source": "area_average_region_normal",
        "contact_constraint_force_distribution": "region_area_slave_shape_master_payload",
        "contact_constraint_independent_quadrature_penalty_disabled": 1,
    }
    rows = [dict(row), dict(row)]

    passed = tooth_patch_region_gate_metrics(
        rows,
        two_block_summary={"two_block_sliding_region_gate_passed": 1},
    )
    assert int(passed["tooth_patch_region_gate_passed"]) == 1

    missing_prerequisite = tooth_patch_region_gate_metrics(
        rows,
        two_block_summary={"two_block_sliding_region_gate_passed": 0},
    )
    assert int(missing_prerequisite["tooth_patch_region_gate_passed"]) == 0
    assert int(missing_prerequisite["two_block_sliding_prerequisite_gate_passed"]) == 0

    wrong_payload_rows = [dict(row), dict(row)]
    wrong_payload_rows[1]["q4_master_payload_width"] = 3
    wrong_payload = tooth_patch_region_gate_metrics(
        wrong_payload_rows,
        two_block_summary={"two_block_sliding_region_gate_passed": 1},
    )
    assert int(wrong_payload["tooth_patch_region_gate_passed"]) == 0
    assert int(wrong_payload["tooth_patch_q4_master_payload_gate_passed"]) == 0
