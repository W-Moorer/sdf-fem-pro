from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_linear_penalty_normal_indentation import (  # noqa: E402
    LinearPenaltyIndentationCase,
    overlap_area,
    run_sfc,
    write_abaqus_deck,
)


def test_linear_penalty_indentation_deck_uses_surface_to_surface_linear_law(tmp_path: Path) -> None:
    case = LinearPenaltyIndentationCase(pressure_stiffness=1234.5)
    path = tmp_path / "linear_penalty.inp"

    write_abaqus_deck(case, path)

    text = path.read_text(encoding="ascii")
    assert "*Surface Behavior, pressure-overclosure=LINEAR" in text
    assert "1.234500000000e+03" in text
    assert "*Contact Pair, interaction=LINEAR_FRICTIONLESS, type=SURFACE TO SURFACE" in text
    assert "LOWER_ALL_ASM, 1, 3, 0." in text
    assert "UPPER_ALL_ASM, 3, 3" in text


def test_sfc_linear_penalty_indentation_matches_pressure_force_law(tmp_path: Path) -> None:
    case = LinearPenaltyIndentationCase(
        size=2.0,
        initial_gap=0.1,
        pressure_stiffness=100.0,
        closures=(0.0, 0.1, 0.15),
    )

    path, runtime = run_sfc(case, tmp_path / "sfc.csv")

    assert path.exists()
    assert float(runtime["analysis_wall_seconds"]) >= 0.0
    rows = np.genfromtxt(path, delimiter=",", names=True)
    by_step = {int(row["step"]): row for row in np.atleast_1d(rows)}
    assert by_step[0]["normal_force_magnitude"] == pytest.approx(0.0)
    assert by_step[1]["normal_force_magnitude"] == pytest.approx(0.0, abs=1.0e-10)
    # penetration = 0.05, area = 4, pressure = 5, force = 20.
    assert by_step[2]["penetration"] == pytest.approx(0.05)
    assert by_step[2]["cpress_mean"] == pytest.approx(5.0)
    assert by_step[2]["normal_force_magnitude"] == pytest.approx(20.0)


def test_partial_overlap_case_reports_geometric_overlap_area() -> None:
    case = LinearPenaltyIndentationCase(size=1.0, upper_size=0.5, upper_offset=(0.25, 0.25))

    assert overlap_area(case) == pytest.approx(0.25)


def test_partial_overlap_deck_offsets_upper_part_nodes(tmp_path: Path) -> None:
    case = LinearPenaltyIndentationCase(size=1.0, upper_size=0.5, upper_offset=(0.25, 0.25))
    path = tmp_path / "partial_overlap.inp"

    write_abaqus_deck(case, path)

    text = path.read_text(encoding="ascii")
    assert "2.500000000000e-01, 2.500000000000e-01" in text
    assert "7.500000000000e-01, 7.500000000000e-01" in text


def test_sfc_partial_overlap_integrates_only_projected_contact_region(tmp_path: Path) -> None:
    case = LinearPenaltyIndentationCase(
        size=1.0,
        upper_size=0.5,
        upper_offset=(0.25, 0.25),
        initial_gap=0.1,
        pressure_stiffness=100.0,
        closures=(0.15,),
    )

    path, _runtime = run_sfc(case, tmp_path / "sfc_partial.csv")

    rows = np.genfromtxt(path, delimiter=",", names=True)
    row = np.atleast_1d(rows)[0]
    # penetration = 0.05, overlap area = 0.25, pressure = 5, force = 1.25.
    assert row["penetration"] == pytest.approx(0.05)
    assert row["overlap_area"] == pytest.approx(0.25)
    assert row["cpress_mean"] == pytest.approx(5.0)
    assert row["normal_force_magnitude"] == pytest.approx(1.25)
