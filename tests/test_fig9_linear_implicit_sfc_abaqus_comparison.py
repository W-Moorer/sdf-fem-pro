from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_fig9_linear_implicit_sfc_abaqus_comparison import Fig9Config, build_abaqus_input_text, make_geometry


def test_fig9_implicit_deck_uses_linear_frictionless_undamped_contact() -> None:
    cfg = Fig9Config(nx=4, ny=2, nz=1, driver_nx=2, driver_ny=2, total_time=0.004, dt=0.001, closure_time=0.002)
    text = build_abaqus_input_text(cfg)

    assert "*Dynamic, DIRECT, NOHAF" in text
    assert "*Surface Behavior, pressure-overclosure=LINEAR" in text
    assert "4.500000000000e+03" in text
    assert "*Friction\n0." in text
    assert "*Damping" not in text
    assert "*Contact Damping" not in text
    assert "DRIVER_ALL_ASM, SNEG" in text
    assert text.index("*Contact Pair") < text.index("*Step")


def test_fig9_implicit_geometry_matches_medium_c3d8_layout() -> None:
    cfg = Fig9Config(nx=12, ny=6, nz=3, driver_nx=6, driver_ny=6)
    lower, driver_ref, driver_faces = make_geometry(cfg)

    assert lower.X.shape == ((12 + 1) * (6 + 1) * (3 + 1), 3)
    assert lower.elements.shape == (12 * 6 * 3, 8)
    assert driver_ref.shape == ((6 + 1) * (6 + 1), 3)
    assert driver_faces.shape == (2 * 6 * 6, 3)
