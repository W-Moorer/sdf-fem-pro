from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_hard_contact_active_set_alignment import build_case, run_sfc, write_abaqus_deck


def test_hard_contact_deck_contains_hard_node_to_surface_contact(tmp_path) -> None:
    case = build_case()
    path = tmp_path / "hard_contact.inp"

    write_abaqus_deck(case, path)

    text = path.read_text(encoding="ascii")
    assert "*Surface Behavior, pressure-overclosure=HARD" in text
    assert "*Contact Pair, interaction=HARD_CONTACT, type=NODE TO SURFACE" in text
    assert "*Element, type=SPRING1" in text
    assert "U, CF, RF" in text


def test_sfc_hard_contact_alignment_case_enforces_zero_gap(tmp_path) -> None:
    case = build_case()

    path, runtime = run_sfc(case, tmp_path / "sfc.csv")

    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(runtime["analysis_wall_seconds"]) >= 0.0
    assert float(row["u3"]) == pytest.approx(-case.initial_gap)
    assert float(row["gap"]) == pytest.approx(0.0)
    assert int(row["active"]) == 1
