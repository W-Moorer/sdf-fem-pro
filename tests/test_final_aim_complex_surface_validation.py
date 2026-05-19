from __future__ import annotations

import csv
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_final_aim_complex_surface_validation import run_validation

pytestmark = pytest.mark.final_aim


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_final_aim_complex_surface_quick_outputs(tmp_path: Path) -> None:
    outputs = run_validation(tmp_path / "final-aim-complex-surface", quick=True)

    for key in ("accuracy", "queries", "timing", "summary"):
        path = outputs[key]
        assert path.exists(), key
        assert path.stat().st_size > 0, key

    accuracy = {row["method"]: row for row in _rows(outputs["accuracy"])}
    assert {"lagrangian_patch_oracle", "dynamic_field"} <= set(accuracy)
    assert accuracy["lagrangian_patch_oracle"]["status"] == "passed"
    assert float(accuracy["lagrangian_patch_oracle"]["max_gap_abs_error"]) < 1.0e-9
    assert float(accuracy["lagrangian_patch_oracle"]["max_normal_l2_error"]) < 1.0e-7
    assert float(accuracy["lagrangian_patch_oracle"]["max_closest_point_l2_error"]) < 1.0e-8

    timing = {row["method"]: row for row in _rows(outputs["timing"])}
    assert {"brute_force_projection_reference", "lagrangian_patch_oracle", "dynamic_field"} <= set(timing)
    assert float(timing["lagrangian_patch_oracle"]["total_seconds"]) > 0.0
    assert float(timing["dynamic_field"]["update_seconds"]) > 0.0
    assert float(timing["lagrangian_patch_oracle"]["speedup_dynamic_field_total_over_method"]) > 1.0

    queries = _rows(outputs["queries"])
    assert queries
    assert max(float(row["lagrangian_gap_abs_error"]) for row in queries) < 1.0e-9


def test_complex_surface_runner_keeps_projection_as_reference_only() -> None:
    source = (ROOT / "validation" / "run_final_aim_complex_surface_validation.py").read_text(encoding="utf-8")

    assert "LagrangianSDFContactOracle" in source
    assert "MaterialSDF" in source
    assert "surface_projection_distance_kernel" in source
    assert "reference projection is validation evidence" in source
