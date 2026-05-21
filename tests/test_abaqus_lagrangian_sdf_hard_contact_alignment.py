from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_lagrangian_sdf_hard_contact_alignment import build_case, run_sfc, write_abaqus_deck


def test_lagrangian_sdf_hard_contact_deck_uses_surface_to_surface_hard(tmp_path) -> None:
    case = build_case()
    deck = tmp_path / "blocks_hard.inp"

    write_abaqus_deck(case, deck)

    text = deck.read_text(encoding="ascii")
    assert "*Surface Behavior, pressure-overclosure=HARD" in text
    assert "*Contact Pair, interaction=HARD_FRICTIONLESS, type=SURFACE TO SURFACE" in text
    assert "LOWER_TOP_SURF, UPPER_BOTTOM_SURF" in text
    assert "*Element, type=C3D8" in text


def test_sfc_lagrangian_sdf_hard_contact_enforces_q4_sample_gaps(tmp_path) -> None:
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
    assert float(row["upper_top_u3_mean"]) == pytest.approx(-case.closure)
    with diagnostics.open(newline="", encoding="utf-8") as handle:
        diagnostic_rows = list(csv.DictReader(handle))
    assert len(diagnostic_rows) == case.quadrature_order**2
    assert diagnostic_rows[0]["master_weights"]
    assert float(diagnostic_rows[0]["enforcement_gap"]) == pytest.approx(0.0, abs=1.0e-8)
