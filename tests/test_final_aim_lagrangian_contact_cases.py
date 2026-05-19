from __future__ import annotations

import csv
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_final_aim_lagrangian_contact_cases import run_validation

pytestmark = pytest.mark.final_aim


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_final_aim_lagrangian_contact_cases_quick_outputs(tmp_path: Path) -> None:
    outputs = run_validation(tmp_path / "final-aim-lagrangian-contact", quick=True)

    for key in ("cases", "samples", "history", "summary"):
        path = outputs[key]
        assert path.exists(), key
        assert path.stat().st_size > 0, key

    cases = _rows(outputs["cases"])
    assert {row["case_id"] for row in cases} == {
        "static_linear_tet4_lagrangian_oracle",
        "dynamic_linear_tet4_lagrangian_oracle",
        "static_linear_hex8_lagrangian_oracle",
        "dynamic_linear_hex8_lagrangian_oracle",
    }
    assert {row["contact_path"] for row in cases} == {"MaterialSDF+LagrangianSDFContactOracle"}
    assert {row["status"] for row in cases} == {"passed"}
    assert {row["element_type"] for row in cases} == {"tet4", "hex8"}
    assert {row["regime"] for row in cases} == {"static", "dynamic"}
    assert all(float(row["max_gap_abs_error"]) < 1.0e-10 for row in cases)
    assert all(float(row["force_l2_rel_error"]) < 1.0e-10 for row in cases)
    assert any(float(row["max_penetration_oracle"]) > 0.0 for row in cases)

    history = _rows(outputs["history"])
    assert {row["case_id"] for row in history} == {
        "dynamic_linear_tet4_lagrangian_oracle",
        "dynamic_linear_hex8_lagrangian_oracle",
    }
    assert any(float(row["reaction_z_oracle"]) > 0.0 for row in history)


def test_final_aim_runner_does_not_import_current_field_or_projection_query() -> None:
    source = (ROOT / "validation" / "run_final_aim_lagrangian_contact_cases.py").read_text(encoding="utf-8")

    assert "DynamicNarrowBandSDF" not in source
    assert "dynamic_surface_sdf" not in source
    assert "compute_field_contact_constraints" not in source
    assert "MaterialSDF" in source
    assert "LagrangianSDFContactOracle" in source
