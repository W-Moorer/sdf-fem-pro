from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_sfc_native_trajectory_equivalence import run_validation


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_sfc_native_trajectory_equivalence_quick_outputs(tmp_path: Path) -> None:
    outputs = run_validation(tmp_path / "native-equivalence", quick=True)

    for key in ("cases", "history", "samples", "summary"):
        path = outputs[key]
        assert path.exists(), key
        assert path.stat().st_size > 0, key

    cases = _rows(outputs["cases"])
    assert {row["case_id"] for row in cases} == {
        "static_linear_tet4_native_sfc_trajectory_equivalence",
        "dynamic_linear_tet4_native_sfc_trajectory_equivalence",
        "static_linear_hex8_native_sfc_trajectory_equivalence",
        "dynamic_linear_hex8_native_sfc_trajectory_equivalence",
    }
    assert {row["candidate_path"] for row in cases} == {"DynamicNarrowBandSDF+field_contact"}
    assert {row["reference_path"] for row in cases} == {"SFC analytic plane same penalty law"}
    assert {row["status"] for row in cases} == {"passed"}
    assert all(float(row["max_state_l2_abs_error"]) < 1.0e-10 for row in cases)
    assert all(float(row["max_gap_abs_error"]) < 1.0e-10 for row in cases)
    assert all(float(row["max_force_l2_rel_error"]) < 1.0e-10 for row in cases)

    history = _rows(outputs["history"])
    assert {row["case_id"] for row in history} == {row["case_id"] for row in cases}
    assert any(int(row["active_count_field"]) > 0 for row in history)
    assert all(row["active_count_field"] == row["active_count_reference"] for row in history)


def test_sfc_native_trajectory_equivalence_is_sfc_only() -> None:
    source = (ROOT / "validation" / "run_sfc_native_trajectory_equivalence.py").read_text(encoding="utf-8")

    assert "DynamicNarrowBandSDF" in source
    assert "field_contact" in source
    assert "dynamic_surface_sdf" not in source
    assert "FuzzyContact" not in source
    assert "CalculiX" not in source
    assert "Abaqus" not in source
    assert "SfePy" not in source
