from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_flat_punch_region_penetration_validation import run_validation


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

    totals = tmp_path / "flat_punch_rf_penetration.csv"
    summary_csv = tmp_path / "flat_punch_region_summary.csv"
    report = tmp_path / "flat_punch_region_summary.md"
    assert totals.exists()
    assert summary_csv.exists()
    assert report.exists()

    with totals.open(newline="", encoding="utf-8") as handle:
        total_rows = list(csv.DictReader(handle))
    assert "normal_force" in total_rows[0]
    assert "expected_normal_force" in total_rows[0]
    assert "contact_energy" in total_rows[0]
    assert "expected_contact_energy" in total_rows[0]
    assert "contact_active_area" in total_rows[0]
