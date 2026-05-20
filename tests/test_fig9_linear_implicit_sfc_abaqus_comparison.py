from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from validation.run_fig9_linear_implicit_sfc_abaqus_comparison import (
    Fig9Config,
    _driver_master_faces,
    _lower_contact_top_element_ids,
    build_abaqus_input_text,
    make_geometry,
)


def test_fig9_implicit_deck_uses_linear_frictionless_undamped_contact() -> None:
    cfg = Fig9Config(nx=4, ny=2, nz=1, driver_nx=2, driver_ny=2, total_time=0.004, dt=0.001, closure_time=0.002)
    text = build_abaqus_input_text(cfg)

    assert "*Dynamic, DIRECT, NOHAF" in text
    assert "*Step, name=FIG9_LINEAR_IMPLICIT, nlgeom=NO" in text
    assert "*Surface Behavior, pressure-overclosure=LINEAR" in text
    assert "4.500000000000e+03" in text
    assert "*Friction\n0." in text
    assert "*Damping" not in text
    assert "*Contact Damping" not in text
    assert "DRIVER_ALL_ASM, SNEG" in text
    assert "S, E, LE" in text
    assert "*Node Output, nset=DRIVER_RP" in text
    assert "U3, RF3, CF3" in text
    assert text.index("*Contact Pair") < text.index("*Step")


def test_fig9_implicit_geometry_matches_medium_c3d8_layout() -> None:
    cfg = Fig9Config(nx=12, ny=6, nz=3, driver_nx=6, driver_ny=6)
    lower, driver_ref, driver_faces = make_geometry(cfg)

    assert lower.X.shape == ((12 + 1) * (6 + 1) * (3 + 1), 3)
    assert lower.elements.shape == (12 * 6 * 3, 8)
    assert driver_ref.shape == ((6 + 1) * (6 + 1), 3)
    assert driver_faces.shape == (2 * 6 * 6, 3)


def test_fig9_sfc_driver_master_faces_point_downward() -> None:
    cfg = Fig9Config(nx=4, ny=2, nz=1, driver_nx=2, driver_ny=2)
    _lower, driver_ref, driver_faces = make_geometry(cfg)
    master_faces = _driver_master_faces(driver_faces)
    tri = driver_ref[master_faces[0]]
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])

    assert normal[2] < 0.0


def test_fig9_contact_patch_tracks_driver_sweep() -> None:
    cfg = Fig9Config(nx=12, ny=6, nz=3, driver_nx=6, driver_ny=6, shift_x=2.0)
    lower, driver_ref, _driver_faces = make_geometry(cfg)
    selected = _lower_contact_top_element_ids(lower, driver_ref, cfg)

    assert 0 < selected.size < lower.nx * lower.ny
