from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_lagrangian_sdf_tooth_patch_hard_contact import (  # noqa: E402
    build_case,
    run_sfc,
    tooth_patch_basis,
    write_abaqus_deck,
)


def test_tooth_patch_deck_uses_tilted_surface_to_surface_hard_contact(tmp_path) -> None:
    case = build_case()
    deck = tmp_path / "tooth_patch_hard.inp"

    write_abaqus_deck(case, deck)

    text = deck.read_text(encoding="ascii")
    _t1, _t2, normal = tooth_patch_basis(case)
    assert "*Surface Behavior, pressure-overclosure=HARD" in text
    assert "*Contact Pair, interaction=HARD_FRICTIONLESS, type=SURFACE TO SURFACE" in text
    assert "LOWER_TOP_SURF, UPPER_BOTTOM_SURF" in text
    assert "*Element, type=C3D8" in text
    assert f"UPPER_TOP_ASM, 1, 1, {-case.closure * normal[0]:.12e}" in text
    assert f"UPPER_TOP_ASM, 2, 2, {-case.closure * normal[1]:.12e}" in text


def test_sfc_tooth_patch_hard_contact_enforces_q4_normal_gap(tmp_path) -> None:
    case = build_case()

    diagnostics = tmp_path / "diagnostics.csv"
    path, runtime = run_sfc(case, tmp_path / "sfc.csv", diagnostics_path=diagnostics)

    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(runtime["analysis_wall_seconds"]) >= 0.0
    assert int(row["converged"]) == 1
    assert int(row["quadrature_samples"]) == case.quadrature_order**2
    assert int(row["active_constraints"]) >= 1
    assert float(row["enforcement_gap_max_abs"]) == pytest.approx(0.0, abs=1.0e-8)
    assert float(row["sample_mean_gap"]) < 0.0
    assert float(row["upper_top_un_mean"]) == pytest.approx(-case.closure, abs=1.0e-12)
    assert abs(float(row["normal_x"])) > 1.0e-3
    assert abs(float(row["normal_y"])) > 1.0e-3
    with diagnostics.open(newline="", encoding="utf-8") as handle:
        diagnostic_rows = list(csv.DictReader(handle))
    assert len(diagnostic_rows) == case.quadrature_order**2
    assert diagnostic_rows[0]["master_weights"]
    assert float(diagnostic_rows[0]["enforcement_gap"]) == pytest.approx(0.0, abs=1.0e-8)
