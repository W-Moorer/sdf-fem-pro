from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.run_flexible_cube_sdf_abaqus_comparison import (  # noqa: E402
    FlexibleCubeConfig,
    _contact_lower_top_element_ids,
    _nodal_smoothed_pressure,
    _top_quads_for_element_ids,
    _upper_bottom_triangles,
    build_abaqus_input_text,
    make_geometry,
)
from sfc.contact import SurfaceQuadratureCache  # noqa: E402


def test_flexible_cube_deck_uses_two_c3d8_flexible_bodies() -> None:
    cfg = FlexibleCubeConfig(lower_nx=4, lower_ny=2, lower_nz=1, upper_nx=2, upper_ny=2, upper_nz=1, total_time=0.004)
    text = build_abaqus_input_text(cfg)

    assert "*Part, name=LOWER" in text
    assert "*Part, name=UPPER" in text
    assert text.count("*Element, type=C3D8") == 2
    assert "*Rigid Body" not in text
    assert "*Element, type=R3D4" not in text
    assert "LOWER_TOP_SURF, UPPER_BOTTOM_SURF" in text
    assert "*Contact Pair, interaction=LINEAR_FRICTIONLESS, type=SURFACE TO SURFACE" in text
    assert "*Surface Behavior, pressure-overclosure=LINEAR" in text
    assert "*Friction\n0." in text
    assert "*Contact Output" in text
    assert "CSTRESS, CDISP" in text
    assert "*Step, name=FLEXIBLE_CUBE_IMPLICIT, nlgeom=NO" in text
    assert "*Dynamic, ALPHA=-5.000000000000e-02, HAFTOL=1.0e-4" in text
    assert "UPPER_TOP_ASM, 1, 2, 0." in text
    assert "*Cload, amplitude=TANGENTIAL_AMP" not in text
    assert "S, E, LE" in text
    assert "U, V, RF" in text


def test_flexible_cube_deck_can_apply_tangential_force() -> None:
    cfg = FlexibleCubeConfig(
        lower_nx=4,
        lower_ny=2,
        lower_nz=1,
        upper_nx=2,
        upper_ny=2,
        upper_nz=1,
        total_time=0.004,
        closure_time=0.002,
        tangential_force=2000.0,
        tangential_start_time=0.002,
        tangential_ramp_time=0.002,
    )
    text = build_abaqus_input_text(cfg)

    assert "*Amplitude, name=TANGENTIAL_AMP, time=TOTAL TIME" in text
    assert "*Cload, amplitude=TANGENTIAL_AMP" in text
    assert "UPPER_TOP_ASM, 1," in text
    assert "UPPER_TOP_ASM, 2, 2, 0." in text
    assert "UPPER_TOP_ASM, 1, 2, 0." not in text
    assert "2.000000000000e-03" in text


def test_flexible_cube_deck_can_apply_bounded_tangential_displacement() -> None:
    cfg = FlexibleCubeConfig(
        lower_nx=4,
        lower_ny=2,
        lower_nz=1,
        upper_nx=2,
        upper_ny=2,
        upper_nz=1,
        total_time=0.004,
        closure_time=0.002,
        tangential_displacement=0.5,
        tangential_motion_start_time=0.002,
        tangential_motion_ramp_time=0.002,
    )
    text = build_abaqus_input_text(cfg)

    assert "*Amplitude, name=TANGENTIAL_MOTION_AMP, time=TOTAL TIME" in text
    assert "*Boundary, amplitude=TANGENTIAL_MOTION_AMP" in text
    assert "UPPER_TOP_ASM, 1, 1, 5.000000000000e-01" in text
    assert "UPPER_TOP_ASM, 2, 2, 0." in text
    assert "*Cload, amplitude=TANGENTIAL_AMP" not in text


def test_flexible_cube_deck_rejects_two_tangential_controls() -> None:
    cfg = FlexibleCubeConfig(
        lower_nx=4,
        lower_ny=2,
        lower_nz=1,
        upper_nx=2,
        upper_ny=2,
        upper_nz=1,
        tangential_force=1.0,
        tangential_displacement=0.5,
    )

    with pytest.raises(ValueError, match="either tangential_force or tangential_displacement"):
        build_abaqus_input_text(cfg)


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


def test_nodal_smoothed_pressure_projects_quadrature_pressure_by_area() -> None:
    cache = SurfaceQuadratureCache(
        node_ids=np.asarray([[0, 1], [0, 1]], dtype=np.int64),
        weights=np.asarray([[0.75, 0.25], [0.25, 0.75]], dtype=float),
        area_weights=np.asarray([2.0, 6.0], dtype=float),
    )
    pressure, nodal_area = _nodal_smoothed_pressure(
        cache,
        np.asarray([-0.1, -0.3], dtype=float),
        pressure_stiffness=10.0,
        n_slave_nodes=2,
    )

    assert nodal_area.tolist() == pytest.approx([3.0, 5.0])
    assert pressure.tolist() == pytest.approx([(1.0 * 1.5 + 3.0 * 1.5) / 3.0, (1.0 * 0.5 + 3.0 * 4.5) / 5.0])
