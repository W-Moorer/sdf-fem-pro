from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_two_block_sliding_region_validation import run_validation


def test_two_block_sliding_region_validation_writes_totals_before_clouds(tmp_path: Path) -> None:
    rows, summary = run_validation(out_dir=tmp_path, resolution=1, steps=5, penetration=0.01, lateral_shift=0.10)

    assert len(rows) == 5
    assert summary["status"] == "completed"
    assert int(summary["constraint_regions_final"]) > 0
    assert float(summary["normal_force_final"]) >= 0.0
    assert int(summary["path_tracking_gate_passed"]) == 1
    assert float(summary["active_region_jaccard_min"]) >= float(summary["path_tracking_min_active_region_jaccard_threshold"])
    assert float(summary["path_cache_hit_fraction_min_after_first"]) >= float(summary["path_tracking_min_cache_hit_threshold"])
    assert float(summary["path_cache_match_fraction_min_after_first"]) >= float(summary["path_tracking_min_cache_match_threshold"])
    assert float(summary["master_face_switch_fraction_max"]) <= float(summary["path_tracking_max_face_switch_threshold"])
    assert float(summary["master_barycentric_drift_max"]) <= float(summary["path_tracking_max_barycentric_drift_threshold"])

    totals = tmp_path / "two_block_sliding_contact_totals.csv"
    continuity = tmp_path / "two_block_sliding_region_continuity.csv"
    tracking = tmp_path / "two_block_sliding_path_tracking.csv"
    report = tmp_path / "two_block_sliding_region_summary.md"
    assert totals.exists()
    assert continuity.exists()
    assert tracking.exists()
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
