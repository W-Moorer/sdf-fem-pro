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

    totals = tmp_path / "two_block_sliding_contact_totals.csv"
    continuity = tmp_path / "two_block_sliding_region_continuity.csv"
    report = tmp_path / "two_block_sliding_region_summary.md"
    assert totals.exists()
    assert continuity.exists()
    assert report.exists()

    with totals.open(newline="", encoding="utf-8") as handle:
        total_rows = list(csv.DictReader(handle))
    assert "normal_force" in total_rows[0]
    assert "contact_virtual_work" in total_rows[0]
    assert "contact_region_energy" in total_rows[0]
    assert "contact_active_area" in total_rows[0]
    assert "contact_active_region_jaccard" in total_rows[0]

    with continuity.open(newline="", encoding="utf-8") as handle:
        continuity_rows = list(csv.DictReader(handle))
    assert len(continuity_rows) == 5
    assert "contact_active_region_persistence_fraction" in continuity_rows[0]
    assert "contact_active_region_jaccard" in continuity_rows[0]
