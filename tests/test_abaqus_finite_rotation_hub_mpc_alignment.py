from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_finite_rotation_hub_mpc_alignment import build_case, sfc_displacements, write_abaqus_deck


def test_finite_rotation_hub_mpc_deck_contains_beam_mpc_and_nlgeom(tmp_path) -> None:
    case = build_case()
    path = tmp_path / "finite_rotation_hub_mpc.inp"

    write_abaqus_deck(case, path)

    text = path.read_text(encoding="ascii")
    assert "*MPC" in text
    assert "BEAM, HUB, RP" in text
    assert "*Step, name=finite_rotation, nlgeom=YES" in text
    assert "RP, 6, 6" in text


def test_finite_rotation_hub_mpc_sfc_export_has_four_nodes(tmp_path) -> None:
    case = build_case()
    out = sfc_displacements(case, tmp_path / "sfc.csv")

    text = out.read_text(encoding="utf-8")
    assert text.count("\n") == 5
    assert "node_id,ux,uy,uz" in text
    assert not np.allclose(case.rotation, 0.0)
