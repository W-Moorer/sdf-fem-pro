from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_fig6_inspired_frictionless_contact import run_validation  # noqa: E402


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_fig6_inspired_frictionless_contact_quick_outputs(tmp_path: Path) -> None:
    outputs = run_validation(tmp_path / "fig6", quick=True)

    for key in ("metrics", "normal_vtk", "shifted_vtk", "normal_cloud_png", "shifted_cloud_png", "sdf_png"):
        assert outputs[key].exists(), key
        assert outputs[key].stat().st_size > 0, key

    rows = _rows(outputs["metrics"])
    assert {row["stage"] for row in rows} == {"normal_loading", "shifted_frictionless"}
    assert {row["element_type"] for row in rows} == {"C3D8"}
    assert max(int(row["active_samples"]) for row in rows) > 0
