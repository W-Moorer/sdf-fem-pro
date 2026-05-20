from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_flexible_cube_sdf_abaqus_comparison import (  # noqa: E402
    FlexibleCubeConfig,
    _contact_lower_top_element_ids,
    _top_quads_for_element_ids,
    _upper_bottom_triangles,
    build_abaqus_input_text,
    make_geometry,
)


def test_flexible_cube_deck_uses_two_c3d8_flexible_bodies() -> None:
    cfg = FlexibleCubeConfig(lower_nx=4, lower_ny=2, lower_nz=1, upper_nx=2, upper_ny=2, upper_nz=1, total_time=0.004)
    text = build_abaqus_input_text(cfg)

    assert "*Part, name=LOWER" in text
    assert "*Part, name=UPPER" in text
    assert text.count("*Element, type=C3D8") == 2
    assert "*Rigid Body" not in text
    assert "*Element, type=R3D4" not in text
    assert "LOWER_TOP_SURF, UPPER_BOTTOM_SURF" in text
    assert "*Surface Behavior, pressure-overclosure=LINEAR" in text
    assert "*Friction\n0." in text
    assert "*Step, name=FLEXIBLE_CUBE_IMPLICIT, nlgeom=NO" in text
    assert "S, E, LE" in text
    assert "U, V, RF" in text


def test_flexible_cube_contact_surfaces_have_expected_topology() -> None:
    cfg = FlexibleCubeConfig(lower_nx=12, lower_ny=6, lower_nz=3, upper_nx=6, upper_ny=6, upper_nz=2)
    lower, upper = make_geometry(cfg)
    selected = _contact_lower_top_element_ids(lower, upper, cfg)
    slave_quads = _top_quads_for_element_ids(lower, selected)
    master_tris = _upper_bottom_triangles(upper)

    assert 0 < selected.size < lower.nx * lower.ny
    assert slave_quads.shape == (selected.size, 4)
    assert master_tris.shape == (2 * upper.nx * upper.ny, 3)
    tri = upper.X[master_tris[0]]
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    assert normal[2] < 0.0
