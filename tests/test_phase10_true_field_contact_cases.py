from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_phase10_true_field_contact_cases import run_validation


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_phase10_true_field_contact_cases_quick_outputs(tmp_path: Path) -> None:
    outputs = run_validation(tmp_path / "phase10-true-field-contact", quick=True)

    for key in ("cases", "samples", "history", "summary"):
        path = outputs[key]
        assert path.exists(), key
        assert path.stat().st_size > 0, key

    cases = _rows(outputs["cases"])
    assert {row["case_id"] for row in cases} == {
        "static_linear_tet4_field_contact",
        "dynamic_linear_tet4_field_contact",
        "static_linear_hex8_field_contact",
        "dynamic_linear_hex8_field_contact",
    }
    assert {row["field_path"] for row in cases} == {"DynamicNarrowBandSDF+field_contact"}
    assert {row["status"] for row in cases} == {"passed"}
    assert {row["element_type"] for row in cases} == {"tet4", "hex8"}
    assert {row["regime"] for row in cases} == {"static", "dynamic"}
    assert all(float(row["max_gap_abs_error"]) < 1.0e-10 for row in cases)
    assert all(float(row["force_l2_rel_error"]) < 1.0e-10 for row in cases)
    assert any(float(row["max_penetration_field"]) > 0.0 for row in cases)

    history = _rows(outputs["history"])
    assert {row["case_id"] for row in history} == {
        "dynamic_linear_tet4_field_contact",
        "dynamic_linear_hex8_field_contact",
    }
    assert any(float(row["reaction_z_field"]) > 0.0 for row in history)


def test_phase10_runner_does_not_import_legacy_projection_query() -> None:
    source = (ROOT / "validation" / "run_phase10_true_field_contact_cases.py").read_text(encoding="utf-8")

    assert "dynamic_surface_sdf" not in source
    assert "DynamicNarrowBandSDF" in source
    assert "compute_field_contact_constraints" in source
