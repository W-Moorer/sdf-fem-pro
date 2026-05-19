from __future__ import annotations

import numpy as np

from sfc.contact.lagrangian_sdf_oracle import LagrangianSDFContactOracle
from sfc.sdf.dynamic_surface_sdf import surface_projection_distance_kernel
from sfc.sdf.material_sdf import MaterialSDF


def _rotation_y(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def _square_surface() -> tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    return nodes, faces


def _strip_surface(nx: int = 9, ny: int = 3) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-1.0, 1.0, nx)
    ys = np.linspace(-0.35, 0.35, ny)
    nodes = np.asarray([[x, y, 0.0] for x in xs for y in ys], dtype=float)
    faces: list[list[int]] = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b = (i + 1) * ny + j
            c = i * ny + j + 1
            d = (i + 1) * ny + j + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    return nodes, np.asarray(faces, dtype=np.int64)


def _bend_strip(reference_nodes: np.ndarray, *, angle: float = 1.15) -> np.ndarray:
    x = reference_nodes[:, 0]
    y = reference_nodes[:, 1]
    radius = 1.0 / angle
    theta = angle * x
    return np.column_stack(
        (
            radius * np.sin(theta),
            y,
            radius * (1.0 - np.cos(theta)),
        )
    )


def test_lagrangian_oracle_rigid_motion_preserves_gap_and_rotates_normal() -> None:
    reference_nodes, faces = _square_surface()
    material = MaterialSDF.from_triangle_surface(
        reference_nodes,
        faces,
        phi=lambda x: float(x[2]),
        gradient=lambda _x: np.asarray([0.0, 0.0, 1.0], dtype=float),
        band_radius=0.25,
    )
    R = _rotation_y(0.72)
    translation = np.asarray([0.35, -0.2, 0.4], dtype=float)
    current = reference_nodes @ R.T + translation
    oracle = LagrangianSDFContactOracle(material, current, search_radius=0.5)

    reference_query = np.asarray([0.15, -0.25, 0.3], dtype=float)
    current_query = reference_query @ R.T + translation
    result = oracle.query(current_query)

    expected_normal = np.asarray([0.0, 0.0, 1.0], dtype=float) @ R.T
    assert np.isclose(result.gap, 0.3, atol=1.0e-12)
    np.testing.assert_allclose(result.normal, expected_normal, atol=1.0e-12)
    np.testing.assert_allclose(result.material_point, np.asarray([0.15, -0.25, 0.0]), atol=1.0e-12)


def test_lagrangian_oracle_large_bend_matches_current_surface_projection() -> None:
    reference_nodes, faces = _strip_surface()
    material = MaterialSDF.from_triangle_surface(reference_nodes, faces, band_radius=0.25)
    current = _bend_strip(reference_nodes, angle=1.2)
    oracle = LagrangianSDFContactOracle(material, current, search_radius=0.3)
    all_faces = np.arange(faces.shape[0], dtype=np.int64)

    for face_id in (3, 8, 13, 20):
        tri = current[faces[face_id]]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        normal = normal / np.linalg.norm(normal)
        point = tri.mean(axis=0) + 0.035 * normal
        oracle_result = oracle.query(point)
        projection = surface_projection_distance_kernel(point, current, faces, all_faces)
        assert oracle_result.face_id == int(projection.face_id)
        assert np.isclose(oracle_result.gap, projection.g, atol=1.0e-11)
        np.testing.assert_allclose(oracle_result.normal, projection.n, atol=1.0e-11)
        np.testing.assert_allclose(oracle_result.barycentric, projection.w, atol=1.0e-11)


def test_lagrangian_oracle_gap_jacobian_matches_finite_difference() -> None:
    reference_nodes, faces = _square_surface()
    material = MaterialSDF.from_triangle_surface(reference_nodes, faces, band_radius=0.2)
    R = _rotation_y(-0.45)
    current = reference_nodes @ R.T + np.asarray([0.1, 0.2, -0.15])
    oracle = LagrangianSDFContactOracle(material, current, search_radius=0.5)
    point = np.asarray([0.05, -0.1, 0.22]) @ R.T + np.asarray([0.1, 0.2, -0.15])
    result = oracle.query(point)
    jacobian = oracle.query_jacobian_wrt_point(point)

    direction = np.asarray([0.31, -0.27, 0.42], dtype=float)
    direction /= np.linalg.norm(direction)
    eps = 1.0e-6
    plus = oracle.query(point + eps * direction).gap
    minus = oracle.query(point - eps * direction).gap
    finite_difference = (plus - minus) / (2.0 * eps)
    assert np.isclose(finite_difference, float(jacobian @ direction), rtol=1.0e-7, atol=1.0e-9)
    np.testing.assert_allclose(jacobian, result.normal, atol=1.0e-12)


def test_lagrangian_oracle_active_patch_cache_does_not_change_result() -> None:
    reference_nodes, faces = _square_surface()
    material = MaterialSDF.from_triangle_surface(reference_nodes, faces, band_radius=0.25)
    current = reference_nodes.copy()
    cached = LagrangianSDFContactOracle(material, current, search_radius=None, cache_enabled=True)
    uncached = LagrangianSDFContactOracle(material, current, search_radius=None, cache_enabled=False)

    first = np.asarray([-0.35, -0.3, 0.08], dtype=float)
    second = np.asarray([0.55, 0.45, 0.12], dtype=float)
    cached.query(first, cache_key="sample")
    cached_result = cached.query(second, cache_key="sample")
    uncached_result = uncached.query(second)

    assert cached_result.candidates_evaluated == uncached_result.candidates_evaluated
    assert cached_result.face_id == uncached_result.face_id
    assert np.isclose(cached_result.gap, uncached_result.gap, atol=1.0e-14)
    np.testing.assert_allclose(cached_result.normal, uncached_result.normal, atol=1.0e-14)
    np.testing.assert_allclose(cached_result.barycentric, uncached_result.barycentric, atol=1.0e-14)
