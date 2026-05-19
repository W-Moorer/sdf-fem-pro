from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.abaqus_odb_to_vtk import VTK_CELL_TYPES, _object_id_from_instance
from validation.run_abaqus_flexible_body_rigid_plane import (
    CommonConfig,
    CubeConfig,
    SphereConfig,
    _cube_mesh,
    _sphere_mesh,
    build_input_text,
)


def test_exporter_supports_rigid_plane_element_and_object_ids() -> None:
    assert VTK_CELL_TYPES["R3D4"] == 9
    assert _object_id_from_instance("PLANE-1") == 0
    assert _object_id_from_instance("CUBE-1") == 1
    assert _object_id_from_instance("SPHERE-1") == 2


def test_cube_drop_input_uses_explicit_rigid_plane_outputs() -> None:
    text = build_input_text("cube_drop")

    assert "*Dynamic, Explicit" in text
    assert "*Element, type=C3D8R" in text
    assert "*Element, type=R3D4" in text
    assert "PLANE-1.PLANE_RP, ENCASTRE" in text
    assert "*Contact Inclusions, ALL EXTERIOR" in text
    assert "*Output, field, time interval=1.000000000000e-03" in text
    assert "U, V" in text
    assert "S, LE" in text


def test_sphere_drop_input_uses_explicit_rigid_plane_outputs() -> None:
    text = build_input_text("sphere_drop")

    assert "*Dynamic, Explicit" in text
    assert "*Element, type=C3D4" in text
    assert "*Element, type=R3D4" in text
    assert "*Dload" in text
    assert "SPHERE-1.SPHERE_EALL, GRAV" in text


def test_medium_mesh_counts_are_deterministic() -> None:
    common = CommonConfig()
    cube_nodes, cube_elements, cube_all = _cube_mesh(CubeConfig(), common)
    sphere_nodes, sphere_elements, sphere_all = _sphere_mesh(SphereConfig(), common)

    assert len(cube_nodes) == 11 * 11 * 11
    assert len(cube_elements) == 10 * 10 * 10
    assert cube_all == list(range(1, len(cube_nodes) + 1))
    assert len(sphere_nodes) == 3 + (12 - 1) * 24
    assert len(sphere_elements) == 2 * 24 * (12 - 1)
    assert sphere_all == list(range(1, len(sphere_nodes) + 1))


def test_common_case_duration_and_dt() -> None:
    cfg = CommonConfig()

    assert cfg.duration == 3.0
    assert cfg.output_interval == 0.001
