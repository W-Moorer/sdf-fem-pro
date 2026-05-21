from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_rp_torque_dynamics_alignment import build_case, run_sfc, write_abaqus_deck


def test_rp_torque_dynamics_deck_contains_moment_and_rotation_output(tmp_path) -> None:
    case = build_case()
    path = tmp_path / "rp_torque.inp"

    write_abaqus_deck(case, path)

    text = path.read_text(encoding="ascii")
    assert "*MPC" in text
    assert "BEAM, HUB, RP" in text
    assert "*Cload" in text
    assert "RP, 6," in text
    assert "U, UR, V, VR, A, AR" in text


def test_rp_torque_sfc_reduced_history_has_expected_final_rotation(tmp_path) -> None:
    case = build_case()
    path, runtime = run_sfc(case, tmp_path / "sfc.csv")

    rows = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == int(round(case.duration / case.dt)) + 2
    assert float(runtime["inertia_z"]) > 0.0
    assert "rotation_z" in rows[0]
