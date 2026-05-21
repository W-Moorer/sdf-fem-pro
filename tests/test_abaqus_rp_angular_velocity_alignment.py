from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_rp_angular_velocity_alignment import build_case, run_sfc, write_abaqus_deck


def test_rp_angular_velocity_deck_contains_velocity_boundary(tmp_path) -> None:
    case = build_case()
    path = tmp_path / "rp_velocity.inp"

    write_abaqus_deck(case, path)

    text = path.read_text(encoding="ascii")
    assert "*MPC" in text
    assert "BEAM, HUB, RP" in text
    assert "*Boundary, type=VELOCITY" in text
    assert "RP, 6, 6," in text
    assert "U, UR, V, VR, A, AR" in text


def test_rp_angular_velocity_sfc_history_has_expected_endpoint(tmp_path) -> None:
    case = build_case()
    path, runtime = run_sfc(case, tmp_path / "sfc.csv")

    rows = path.read_text(encoding="utf-8").strip().splitlines()
    final = rows[-1].split(",")
    assert len(rows) == int(round(case.duration / case.dt)) + 2
    assert float(runtime["analysis_wall_seconds"]) >= 0.0
    assert float(final[1]) == pytest.approx(case.angular_velocity[2] * case.duration)
