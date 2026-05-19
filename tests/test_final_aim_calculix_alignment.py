from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_final_aim_calculix_alignment import (  # noqa: E402
    _alignment_model,
    _write_calculix_input,
    run_alignment,
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_final_aim_calculix_alignment_skip_outputs(tmp_path: Path) -> None:
    outputs = run_alignment(tmp_path / "alignment", quick=True, skip_calculix=True)

    for key in ("history", "errors", "commands", "plots", "summary"):
        path = outputs[key]
        assert path.exists(), key
        assert path.stat().st_size > 0, key

    history = _rows(outputs["history"])
    assert {row["source"] for row in history} == {"sfc_lagrangian_sdf"}
    assert {row["element_type"] for row in history} == {"tet4", "hex8"}
    assert {row["regime"] for row in history} == {"static", "dynamic"}

    commands = _rows(outputs["commands"])
    assert any(row["solver"] == "sfc_lagrangian_sdf" and row["completed"] == "true" for row in commands)
    assert any(row["solver"] == "calculix" and row["command"] == "skipped" for row in commands)


def test_final_aim_calculix_input_uses_native_contact(tmp_path: Path) -> None:
    model = _alignment_model("hex8", "dynamic", quick=True)
    inp = tmp_path / "case.inp"

    _write_calculix_input(model, inp)
    text = inp.read_text(encoding="utf-8")

    assert "*CONTACT PAIR, INTERACTION=CONTACT, TYPE=SURFACE TO SURFACE" in text
    assert "SSLAVE, SMASTER" in text
    assert "*DYNAMIC, ALPHA=-0.05" in text
    assert "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR" in text
    assert "*ELEMENT, TYPE=C3D8, ELSET=ELALL" in text
    assert "*ELEMENT, TYPE=S4, ELSET=FLOOR" in text


def test_final_aim_alignment_runner_stays_on_lagrangian_sdf_path() -> None:
    source = (ROOT / "validation" / "run_final_aim_calculix_alignment.py").read_text(encoding="utf-8")

    assert "DynamicNarrowBandSDF" not in source
    assert "dynamic_surface_sdf" not in source
    assert "lagrangian_oracle_penalty_response" in source
