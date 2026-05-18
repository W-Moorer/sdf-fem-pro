from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_long_time_c3d8_dynamic_validation import run_validation  # noqa: E402


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_long_time_c3d8_dynamic_smoke_outputs(tmp_path: Path) -> None:
    outputs = run_validation(
        tmp_path / "long_dynamic",
        case="block_block_c3d8",
        resolution=1,
        total_time=0.04,
        dt=0.02,
        gravity=0.02,
        initial_velocity_z=-0.015,
        damping_alpha=1.5,
    )

    for key in ("history", "summary", "vtk", "history_png", "cloud_png"):
        assert outputs[key].exists(), key
        assert outputs[key].stat().st_size > 0, key

    rows = _rows(outputs["history"])
    assert len(rows) == 3
    assert rows[-1]["linearity"] == "linear_damped"
    assert rows[-1]["element_type"] == "C3D8"
