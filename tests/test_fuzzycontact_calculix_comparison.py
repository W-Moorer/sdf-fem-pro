from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_fuzzycontact_calculix_comparison import run_validation


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_fuzzycontact_calculix_comparison_writes_equivalent_load_inputs(tmp_path: Path) -> None:
    outputs = run_validation(tmp_path / "ccx", quick=True, skip_calculix=True)

    for key in ("comparison", "commands", "summary"):
        assert outputs[key].exists(), key
        assert outputs[key].stat().st_size > 0, key

    rows = _rows(outputs["comparison"])
    assert [row["problem"] for row in rows] == ["problem_1", "problem_3", "problem_4"]
    assert {row["calculix_completed"] for row in rows} == {"false"}
    assert all(float(row["sfc_reaction_N"]) > 0.0 for row in rows)
    assert all(float(row["sfc_max_von_mises_MPa"]) > 0.0 for row in rows)

    commands = _rows(outputs["commands"])
    assert len(commands) == 3
    for row in commands:
        inp = Path(row["input_file"])
        assert inp.exists()
        text = inp.read_text(encoding="utf-8")
        assert "*CLOAD" in text
        assert "*EL PRINT, ELSET=EALL" in text


def test_fuzzycontact_calculix_comparison_is_external_validation_only() -> None:
    source = (ROOT / "validation" / "run_fuzzycontact_calculix_comparison.py").read_text(encoding="utf-8")

    assert "DynamicNarrowBandSDF" in source
    assert "run_fuzzycontact_published_benchmarks" in source
    assert "legacy_abaqus_prototype" not in source
    assert ".odb" not in source
    assert "GLOBALM.mat" not in source
    assert "CalculiX is used only as an external validation reference" in source
