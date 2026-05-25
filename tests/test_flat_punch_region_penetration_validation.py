from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_flat_punch_region_penetration_validation import flat_punch_rf_penetration_gate_metrics, run_validation


def test_flat_punch_region_penetration_matches_linear_pressure_law(tmp_path: Path) -> None:
    rows, summary = run_validation(
        out_dir=tmp_path,
        resolution=1,
        penetrations=(0.0, 0.005, 0.01),
        pressure_stiffness=6.0e4,
    )

    assert len(rows) == 3
    assert summary["status"] == "completed"
    assert float(summary["active_area_final"]) > 0.0
    assert float(summary["max_normal_force_rel_error"]) < 1.0e-12
    assert float(summary["max_contact_energy_rel_error"]) < 1.0e-12
    assert int(summary["flat_punch_rf_penetration_gate_passed"]) == 1
    assert int(summary["flat_punch_ready_for_two_block_sliding"]) == 1
    assert int(summary["flat_punch_nodal_cpress_deferred"]) == 1
    assert int(summary["flat_punch_constraint_region_law_gate_passed"]) == 1

    totals = tmp_path / "flat_punch_rf_penetration.csv"
    gate_csv = tmp_path / "flat_punch_rf_penetration_gate.csv"
    summary_csv = tmp_path / "flat_punch_region_summary.csv"
    report = tmp_path / "flat_punch_region_summary.md"
    assert totals.exists()
    assert gate_csv.exists()
    assert summary_csv.exists()
    assert report.exists()

    with totals.open(newline="", encoding="utf-8") as handle:
        total_rows = list(csv.DictReader(handle))
    assert "normal_force" in total_rows[0]
    assert "expected_normal_force" in total_rows[0]
    assert "contact_energy" in total_rows[0]
    assert "expected_contact_energy" in total_rows[0]
    assert "contact_active_area" in total_rows[0]
    assert "nodal_cpress_deferred" in total_rows[0]
    assert "contact_constraint_open_closed_source" in total_rows[0]
    assert "contact_constraint_force_distribution" in total_rows[0]
    assert "flat_punch_rf_penetration_gate_passed" in gate_csv.read_text(encoding="utf-8")


def test_flat_punch_gate_rejects_sample_penalty_or_nodal_cloud_priority() -> None:
    rows = [
        {
            "expected_normal_force": 1.0,
            "normal_force_rel_error": 0.0,
            "expected_contact_energy": 1.0,
            "contact_energy_rel_error": 0.0,
            "nodal_cpress_deferred": 1,
            "active_contact_region_count": 1,
            "contact_active_area": 1.0,
            "normal_force": 1.0,
            "constraint_regions": 1,
            "contact_constraint_law_source": "slave_node_region_constraint",
            "contact_constraint_open_closed_source": "signed_area_average_region_gap",
            "contact_constraint_normal_source": "area_average_region_normal",
            "contact_constraint_force_distribution": "region_area_slave_shape_master_payload",
            "contact_constraint_independent_quadrature_penalty_disabled": 1,
        }
    ]

    gate = flat_punch_rf_penetration_gate_metrics(rows)

    assert int(gate["flat_punch_rf_penetration_gate_passed"]) == 1

    sample_penalty = [dict(rows[0])]
    sample_penalty[0]["contact_constraint_law_source"] = "none"
    sample_penalty[0]["contact_constraint_independent_quadrature_penalty_disabled"] = 0
    failed_law = flat_punch_rf_penetration_gate_metrics(sample_penalty)

    assert int(failed_law["flat_punch_rf_penetration_gate_passed"]) == 0
    assert int(failed_law["flat_punch_constraint_region_law_gate_passed"]) == 0

    nodal_cloud_first = [dict(rows[0])]
    nodal_cloud_first[0]["nodal_cpress_deferred"] = 0
    failed_cloud = flat_punch_rf_penetration_gate_metrics(nodal_cloud_first)

    assert int(failed_cloud["flat_punch_rf_penetration_gate_passed"]) == 0
    assert int(failed_cloud["flat_punch_nodal_cpress_deferred"]) == 0
