from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_fuzzycontact_published_benchmarks import run_validation


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_fuzzycontact_published_benchmarks_quick_outputs(tmp_path: Path) -> None:
    outputs = run_validation(tmp_path / "published", quick=True)

    for key in ("cases", "load_steps", "summary"):
        assert outputs[key].exists(), key
        assert outputs[key].stat().st_size > 0, key

    cases = _rows(outputs["cases"])
    steps = _rows(outputs["load_steps"])
    assert [row["problem"] for row in cases] == ["problem_1", "problem_3", "problem_4"]
    assert {row["query_path"] for row in cases} == {"DynamicNarrowBandSDF.build -> field_contact interpolation"}
    assert all(row["status"] == "completed" for row in cases)
    assert all(float(row["field_grid_nodes"]) > 0.0 for row in cases)
    assert all(float(row["field_valid_nodes"]) > 0.0 for row in cases)
    assert all(float(row["final_reaction_rel_error"]) < 1.0e-5 for row in cases)
    assert max(float(row["case_reference_rel_error"]) for row in cases) < 1.0e-2

    assert len([row for row in steps if row["problem"] == "problem_4"]) == 8
    assert all(float(row["reaction_rel_error"]) < 1.0e-5 for row in steps)
    assert any(int(row["active_samples"]) > 0 for row in steps)


def test_fuzzycontact_published_benchmarks_uses_pdf_parameters_not_vtu_replay() -> None:
    source = (ROOT / "validation" / "run_fuzzycontact_published_benchmarks.py").read_text(encoding="utf-8")

    assert "DynamicNarrowBandSDF" in source
    assert "field_contact" in source
    assert "build(" in source
    assert "build_required_points" not in source
    assert "dynamic_surface_sdf" not in source
    assert "run_fuzzycontact_vtu_reference" not in source
    assert "VTUData" not in source
    assert "legacy_abaqus_prototype" not in source
    assert ".odb" not in source
