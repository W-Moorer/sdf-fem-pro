from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_dual_rp_mpc_dynamics_alignment import build_case, run_sfc_dual_reduced, write_abaqus_deck


def test_dual_rp_mpc_deck_contains_two_beam_mpcs_velocity_and_torque(tmp_path) -> None:
    case = build_case()
    path = tmp_path / "dual_rp.inp"

    write_abaqus_deck(case, path)

    text = path.read_text(encoding="ascii")
    assert "BEAM, HUB1, RP1" in text
    assert "BEAM, HUB2, RP2" in text
    assert "*Boundary, type=VELOCITY" in text
    assert "RP1, 6, 6," in text
    assert "*Cload" in text
    assert "RP2, 6," in text


def test_dual_rp_mpc_sfc_reduced_history_has_two_rp_blocks(tmp_path) -> None:
    case = build_case()

    path, runtime = run_sfc_dual_reduced(case, tmp_path / "sfc.csv")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    final = rows[-1]
    assert len(rows) == int(round(case.duration / case.dt)) + 1
    assert int(runtime["full_dofs"]) == 24
    assert int(runtime["reduced_dofs"]) == 12
    assert float(runtime["rp1_reduced_mass_rz_rz"]) > 0.0
    assert float(runtime["rp2_reduced_mass_rz_rz"]) > 0.0
    assert float(final["rp1_rotation_z"]) == pytest.approx(case.rp1_angular_velocity[2] * case.duration)
    assert float(final["rp2_rotation_z"]) > 0.0
