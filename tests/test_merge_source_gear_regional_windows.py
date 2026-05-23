from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.merge_source_gear_regional_windows import merge_regional_csvs


def _write_window(path: Path, rows: list[tuple[float, str, str, float]]) -> None:
    lines = [
        "sfc_time,abaqus_time,region,metric,sfc_p95,abaqus_p95,p95_rel_error,max_rel_error,sample_count,"
        "sfc_has_contact_field,abaqus_has_contact_field,sfc_active_node_count,abaqus_active_node_count"
    ]
    for time_value, region, metric, p95_error in rows:
        lines.append(
            f"{time_value},{time_value},{region},{metric},1.0,1.0,{p95_error},0.0,2,1,1,0,0"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_merge_regional_csvs_deduplicates_by_time_region_metric(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_window(
        first,
        [
            (0.0, "full", "displacement_magnitude", 0.0),
            (2.0e-5, "full", "displacement_magnitude", 0.1),
        ],
    )
    _write_window(
        second,
        [
            (2.0e-5, "full", "displacement_magnitude", 0.2),
            (4.0e-5, "full", "displacement_magnitude", 0.3),
        ],
    )

    rows = merge_regional_csvs([first, second])

    assert [(float(row["sfc_time"]), row["p95_rel_error"]) for row in rows] == [
        (0.0, "0.0"),
        (2.0e-5, "0.2"),
        (4.0e-5, "0.3"),
    ]
    assert [row["pair_index"] for row in rows] == [0, 1, 2]
