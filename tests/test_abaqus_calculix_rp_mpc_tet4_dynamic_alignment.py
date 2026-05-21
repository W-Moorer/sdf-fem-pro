from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_abaqus_calculix_rp_mpc_tet4_dynamic_alignment import build_model, write_common_input


def test_rp_mpc_tet4_dynamic_deck_contains_shared_solver_features(tmp_path) -> None:
    model = build_model(resolution=1, duration=0.004, dt=0.002)
    path = tmp_path / "rp_mpc_tet4_dynamic.inp"

    write_common_input(model, path, solver="unit-test")

    text = path.read_text(encoding="ascii")
    assert f"*Rigid Body, REF NODE={model.rp_node_id}, TIE NSET=HUB" in text
    assert "*Dynamic" in text
    assert "*Element, type=C3D4" in text
    assert "*Boundary, amplitude=RAMP" in text
    assert "RP, 6, 6" in text
    assert "*Node Print, nset=NALL, frequency=1" in text
    assert "*El Print, elset=EALL, frequency=1" in text
    assert model.left_nodes.size > 0
    assert model.hub_nodes.size > 0


def test_rp_mpc_tet4_dynamic_calculix_deck_uses_ccx_rigid_body_layout(tmp_path) -> None:
    model = build_model(resolution=1, duration=0.004, dt=0.002)
    path = tmp_path / "rp_mpc_tet4_dynamic_ccx.inp"

    write_common_input(model, path, solver="CalculiX")

    text = path.read_text(encoding="ascii")
    assert f"*Rigid Body, NSET=HUB, REF NODE={model.rp_node_id}, ROT NODE={model.rp_node_id + 1}" in text
    assert "ROT, 3, 3" in text
