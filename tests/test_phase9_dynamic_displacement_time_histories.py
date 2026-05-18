from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.plot_phase9_dynamic_displacement_time_histories import generate_dynamic_time_history_plot


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_dynamic_displacement_time_history_plot_uses_time_axis(tmp_path: Path) -> None:
    phase9 = tmp_path / "phase9"
    _write_csv(
        phase9 / "native_c3d8_linear_dynamic_block_plane.csv",
        [
            {"case": "block_plane_c3d8", "resolution": 1, "time": 0.0, "z_cm": 0.17},
            {"case": "block_plane_c3d8", "resolution": 1, "time": 0.01, "z_cm": 0.16},
        ],
    )
    _write_csv(
        phase9 / "native_c3d8_linear_dynamic_comparison.csv",
        [
            {
                "case_id": "c3d8_linear_dynamic_block_plane_contact",
                "z_cm_l2_rel_error": 0.012,
            }
        ],
    )
    _write_csv(
        phase9 / "c3d8_dynamic_trajectory" / "c3d8_contact_trajectory.csv",
        [
            {"case": "block_plane_c3d8", "resolution": 1, "time": 0.0, "z_cm": 0.17},
            {"case": "block_plane_c3d8", "resolution": 1, "time": 0.01, "z_cm": 0.159},
        ],
    )

    outputs = generate_dynamic_time_history_plot(phase9)

    assert outputs["png"].exists()
    assert outputs["pdf"].exists()
    rows = list(csv.DictReader(outputs["csv"].open(newline="", encoding="utf-8")))
    assert rows[0]["time_seconds"] == "0.0"
    assert "sfc_z_displacement_mm" in rows[0]
    assert "calculix_z_displacement_mm" in rows[0]
