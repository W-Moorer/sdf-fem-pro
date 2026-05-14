from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_contact_enforcement_diagnostics import run_diagnostics  # noqa: E402


def test_contact_enforcement_diagnostics_quick_skip_calculix_outputs(tmp_path: Path) -> None:
    outputs = run_diagnostics(tmp_path, quick=True, run_external=False)

    for path in outputs.values():
        assert path.exists()

    with outputs["claims"].open(newline="", encoding="utf-8") as f:
        claims = {row["claim"]: row for row in csv.DictReader(f)}

    assert claims["active_set_diagnostics_available"]["supported"] == "true"
    assert claims["area_weighting_diagnostics_available"]["supported"] == "true"
    assert claims["pressure_overclosure_law_diagnostics_available"]["supported"] == "true"
    assert claims["contact_tangent_finite_difference_check"]["supported"] == "true"
    assert claims["calculix_contact_output_definition_available"]["supported"] == "false"

    with outputs["active_set"].open(newline="", encoding="utf-8") as f:
        active_rows = {row["definition"]: row for row in csv.DictReader(f)}
    assert active_rows["surface_node_gap"]["active_count"] != active_rows["three_point_face_quadrature_gap"]["active_count"]

    with outputs["area"].open(newline="", encoding="utf-8") as f:
        area_rows = {row["definition"]: row for row in csv.DictReader(f)}
    nodal_force = float(area_rows["surface_node_hard_linear_unweighted"]["normal_force"])
    quadrature_force = float(area_rows["three_point_quadrature_hard_linear_area_weighted"]["normal_force"])
    assert nodal_force > quadrature_force

    with outputs["tangent"].open(newline="", encoding="utf-8") as f:
        tangent_rows = list(csv.DictReader(f))
    assert tangent_rows[0]["status"] == "passed"
