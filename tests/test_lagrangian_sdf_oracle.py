from __future__ import annotations

import numpy as np
import pytest

from sfc.contact.lagrangian_sdf_oracle import (
    LagrangianSDFContactOracle,
    lagrangian_oracle_constraint_from_sample,
    lagrangian_oracle_jacobian_row,
    lagrangian_oracle_penalty_response,
    lagrangian_oracle_penalty_response_batch,
    lagrangian_patch_pair_query,
)
from sfc.contact.narrow_phase import SurfaceSample
from sfc.fem.deformation_map import FEMDeformationMap
from sfc.mesh.topology import VolumeMesh
from sfc.sdf.dynamic_surface_sdf import surface_projection_distance_kernel
from sfc.sdf.material_sdf import MaterialSDF, MaterialSDFGrid

pytestmark = pytest.mark.final_aim


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


def _tet_plane_volume_model() -> tuple[np.ndarray, np.ndarray, VolumeMesh, MaterialSDF]:
    nodes = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    elements = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    grid = MaterialSDFGrid.from_phi_function(
        origin=np.asarray([-0.2, -0.2, -0.3], dtype=float),
        spacing=0.1,
        shape=(16, 16, 18),
        phi=lambda x: float(x[2]),
        gradient=lambda _x: np.asarray([0.0, 0.0, 1.0], dtype=float),
    )
    material = MaterialSDF.from_grid(nodes, faces, grid, band_radius=0.4)
    return nodes, faces, VolumeMesh(nodes, elements, element_type="tet4"), material


def test_final_aim_kkt_oracle_uses_material_sdf_grid_and_deformation_map() -> None:
    nodes, faces, mesh, material = _tet_plane_volume_model()
    A = np.asarray(
        [
            [1.05, 0.15, 0.2],
            [-0.05, 0.95, 0.1],
            [0.0, 0.05, 1.2],
        ],
        dtype=float,
    )
    translation = np.asarray([0.25, -0.15, 0.35], dtype=float)
    current = nodes @ A.T + translation
    oracle = LagrangianSDFContactOracle(
        material,
        current,
        search_radius=0.6,
        deformation_map=FEMDeformationMap(mesh, current),
        max_newton_iterations=10,
    )
    material_surface_point = np.asarray([0.25, 0.2, 0.0], dtype=float)
    reference_normal = np.asarray([0.0, 0.0, 1.0], dtype=float)
    current_normal = np.linalg.solve(A.T, reference_normal)
    current_normal /= np.linalg.norm(current_normal)
    closest = A @ material_surface_point + translation
    point = closest + 0.075 * current_normal
    result = oracle.query(point)
    projection = surface_projection_distance_kernel(point, current, faces, np.arange(faces.shape[0]))

    assert result.converged
    assert np.isclose(result.gap, 0.075, atol=1.0e-10)
    assert np.isclose(material.query_phi(result.material_point), 0.0, atol=1.0e-10)
    np.testing.assert_allclose(result.normal, projection.n, atol=1.0e-10)
    np.testing.assert_allclose(result.closest_point, projection.p, atol=1.0e-10)
    assert result.master_node_ids.shape == (4,)
    assert np.isclose(float(np.sum(result.master_weights)), 1.0, atol=1.0e-12)


def test_final_aim_oracle_query_does_not_build_current_sdf_grid(monkeypatch) -> None:
    nodes, _faces, mesh, material = _tet_plane_volume_model()
    from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF

    def fail_build(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("current-space SDF grid build was called")

    monkeypatch.setattr(DynamicNarrowBandSDF, "build", fail_build)
    oracle = LagrangianSDFContactOracle(
        material,
        nodes.copy(),
        search_radius=0.4,
        deformation_map=FEMDeformationMap(mesh, nodes.copy()),
    )
    g, n = oracle.query_gap_normal(np.asarray([0.2, 0.25, 0.05], dtype=float))
    assert np.isclose(g, 0.05, atol=1.0e-12)
    np.testing.assert_allclose(n, np.asarray([0.0, 0.0, 1.0]), atol=1.0e-12)


def test_final_aim_pullback_query_matches_exact_affine_plane() -> None:
    nodes, _faces, mesh, material = _tet_plane_volume_model()
    A = np.asarray(
        [
            [1.2, 0.1, 0.0],
            [-0.1, 0.9, 0.05],
            [0.15, 0.0, 1.1],
        ],
        dtype=float,
    )
    translation = np.asarray([0.2, -0.25, 0.4], dtype=float)
    current = nodes @ A.T + translation
    oracle = LagrangianSDFContactOracle(
        material,
        current,
        search_radius=0.6,
        deformation_map=FEMDeformationMap(mesh, current),
    )
    material_point = np.asarray([0.24, 0.18, 0.06], dtype=float)
    point = A @ material_point + translation
    expected_covector = np.linalg.solve(A.T, np.asarray([0.0, 0.0, 1.0], dtype=float))
    expected_scale = np.linalg.norm(expected_covector)

    result = oracle.query_pullback(point)

    np.testing.assert_allclose(result.material_point, material_point, atol=1.0e-12)
    assert np.isclose(result.phi0, material_point[2], atol=1.0e-12)
    assert np.isclose(result.gap, material_point[2] / expected_scale, atol=1.0e-12)
    np.testing.assert_allclose(result.normal, expected_covector / expected_scale, atol=1.0e-12)
    assert result.node_ids.shape == (4,)
    assert np.isclose(float(np.sum(result.shape_values)), 1.0, atol=1.0e-12)


def test_final_aim_oracle_contact_jacobian_matches_finite_difference() -> None:
    nodes, _faces, mesh, material = _tet_plane_volume_model()
    slave = np.asarray([[0.25, 0.25, -0.04]], dtype=float)
    sample = SurfaceSample(
        node_ids=np.asarray([0], dtype=np.int64),
        weights=np.asarray([1.0], dtype=float),
        candidate_face_ids=np.empty(0, dtype=np.int64),
    )

    def constraint_for(q: np.ndarray):
        slave_x = q[:3].reshape((1, 3))
        master_x = q[3:].reshape((4, 3))
        oracle = LagrangianSDFContactOracle(
            material,
            master_x,
            search_radius=0.4,
            deformation_map=FEMDeformationMap(mesh, master_x),
        )
        return lagrangian_oracle_constraint_from_sample(slave_x, sample, oracle)

    q0 = np.concatenate((slave.reshape(-1), nodes.reshape(-1)))
    constraint = constraint_for(q0)
    J = lagrangian_oracle_jacobian_row(
        constraint,
        n_total_dofs=q0.size,
        slave_dof_offset=0,
        master_dof_offset=3,
    )
    direction = np.linspace(-0.31, 0.27, q0.size)
    direction /= np.linalg.norm(direction)
    eps = 1.0e-6
    fd = (constraint_for(q0 + eps * direction).g - constraint_for(q0 - eps * direction).g) / (2.0 * eps)
    assert np.isclose(float((J @ direction)[0]), fd, rtol=1.0e-6, atol=1.0e-8)


def test_final_aim_oracle_penalty_response_action_reaction() -> None:
    nodes, _faces, mesh, material = _tet_plane_volume_model()
    slave = np.asarray([[0.25, 0.25, -0.05]], dtype=float)
    sample = SurfaceSample(
        node_ids=np.asarray([0], dtype=np.int64),
        weights=np.asarray([1.0], dtype=float),
        candidate_face_ids=np.empty(0, dtype=np.int64),
    )
    oracle = LagrangianSDFContactOracle(
        material,
        nodes.copy(),
        search_radius=0.4,
        deformation_map=FEMDeformationMap(mesh, nodes.copy()),
    )
    response = lagrangian_oracle_penalty_response(
        slave,
        [sample],
        oracle,
        pressure_stiffness=100.0,
        n_total_dofs=15,
        slave_dof_offset=0,
        master_dof_offset=3,
    )
    force = response.force.reshape((-1, 3))
    assert response.active_count == 1
    assert np.isclose(response.min_gap, -0.05, atol=1.0e-12)
    np.testing.assert_allclose(force[0] + force[1:].sum(axis=0), np.zeros(3), atol=1.0e-12)
    assert force[0, 2] > 0.0


def test_material_sdf_grid_cubic_interpolation_matches_polynomial_and_derivative() -> None:
    def phi(x: np.ndarray) -> float:
        return float(x[0] ** 3 + 2.0 * x[1] ** 2 - 0.5 * x[2] + 0.25 * x[0] * x[1] * x[2])

    def grad(x: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                3.0 * x[0] ** 2 + 0.25 * x[1] * x[2],
                4.0 * x[1] + 0.25 * x[0] * x[2],
                -0.5 + 0.25 * x[0] * x[1],
            ],
            dtype=float,
        )

    grid = MaterialSDFGrid.from_phi_function(
        origin=np.asarray([-1.0, -1.0, -1.0], dtype=float),
        spacing=0.25,
        shape=(9, 9, 9),
        phi=phi,
        interpolation_order="cubic",
    )
    point = np.asarray([0.17, -0.31, 0.44], dtype=float)
    assert np.isclose(grid.query_phi(point), phi(point), atol=5.0e-14)
    np.testing.assert_allclose(grid.query_raw_gradient(point), grad(point), atol=5.0e-13)
    np.testing.assert_allclose(grid.query_gradient(point), grad(point) / np.linalg.norm(grad(point)), atol=5.0e-13)


def test_reference_patch_spatial_hash_matches_exact_aabb_radius_filter() -> None:
    nodes, faces = _strip_surface(nx=7, ny=4)
    material = MaterialSDF.from_triangle_surface(nodes, faces, band_radius=0.05)
    bvh = material.build_patch_bvh(nodes.copy(), padding=0.05, cell_size=0.25)
    point = np.asarray([0.18, 0.03, 0.025], dtype=float)
    radius = 0.11

    lower_delta = np.maximum(bvh.aabb_min - point, 0.0)
    upper_delta = np.maximum(point - bvh.aabb_max, 0.0)
    dist2 = np.sum((lower_delta + upper_delta) ** 2, axis=1)
    expected = set(int(i) for i in np.flatnonzero(dist2 <= radius * radius))
    actual = set(int(i) for i in bvh.candidates(point, search_radius=radius))

    assert bvh.cells
    assert actual == expected


def test_lagrangian_oracle_batch_query_matches_scalar_loop() -> None:
    reference_nodes, faces = _square_surface()
    material = MaterialSDF.from_triangle_surface(reference_nodes, faces, band_radius=0.35)
    current = reference_nodes @ _rotation_y(0.35).T + np.asarray([0.2, -0.1, 0.05], dtype=float)
    oracle = LagrangianSDFContactOracle(material, current, search_radius=0.5)
    points = np.asarray(
        [
            [0.2, -0.1, 0.22],
            [-0.15, 0.25, 0.18],
            [0.55, -0.32, 0.12],
        ],
        dtype=float,
    )

    gaps, normals = oracle.query_gap_normal_batch(points, cache_keys=["a", "b", "c"])
    scalar = [oracle.query(point, cache_key=f"scalar-{idx}") for idx, point in enumerate(points)]

    np.testing.assert_allclose(gaps, np.asarray([result.gap for result in scalar]), atol=1.0e-14)
    np.testing.assert_allclose(normals, np.vstack([result.normal for result in scalar]), atol=1.0e-14)


def test_lagrangian_oracle_batch_penalty_response_matches_scalar_response() -> None:
    nodes, _faces, mesh, material = _tet_plane_volume_model()
    slave = np.asarray(
        [
            [0.25, 0.25, -0.05],
            [0.35, 0.15, 0.04],
        ],
        dtype=float,
    )
    samples = [
        SurfaceSample(
            node_ids=np.asarray([0], dtype=np.int64),
            weights=np.asarray([1.0], dtype=float),
            candidate_face_ids=np.empty(0, dtype=np.int64),
        ),
        SurfaceSample(
            node_ids=np.asarray([1], dtype=np.int64),
            weights=np.asarray([1.0], dtype=float),
            candidate_face_ids=np.empty(0, dtype=np.int64),
        ),
    ]
    kwargs = dict(
        pressure_stiffness=125.0,
        n_total_dofs=18,
        slave_dof_offset=0,
        master_dof_offset=6,
    )
    scalar = lagrangian_oracle_penalty_response(
        slave,
        samples,
        LagrangianSDFContactOracle(
            material,
            nodes.copy(),
            search_radius=0.4,
            deformation_map=FEMDeformationMap(mesh, nodes.copy()),
        ),
        **kwargs,
    )
    batch = lagrangian_oracle_penalty_response_batch(
        slave,
        samples,
        LagrangianSDFContactOracle(
            material,
            nodes.copy(),
            search_radius=0.4,
            deformation_map=FEMDeformationMap(mesh, nodes.copy()),
        ),
        **kwargs,
    )

    assert batch.active_count == scalar.active_count == 1
    np.testing.assert_allclose(batch.force, scalar.force, atol=1.0e-14)
    np.testing.assert_allclose(batch.stiffness.toarray(), scalar.stiffness.toarray(), atol=1.0e-14)


def test_lagrangian_patch_pair_query_parallel_surfaces() -> None:
    nodes_a, faces = _square_surface()
    nodes_b = nodes_a + np.asarray([0.0, 0.0, 0.2], dtype=float)
    material_a = MaterialSDF.from_triangle_surface(nodes_a, faces, band_radius=0.3)
    material_b = MaterialSDF.from_triangle_surface(nodes_b, faces, band_radius=0.3)
    oracle_a = LagrangianSDFContactOracle(material_a, nodes_a.copy(), search_radius=0.4)
    oracle_b = LagrangianSDFContactOracle(material_b, nodes_b.copy(), search_radius=0.4)

    result = lagrangian_patch_pair_query(oracle_a, oracle_b)

    assert result.candidates_evaluated == 4
    assert np.isclose(result.gap, 0.2, atol=1.0e-12)
    np.testing.assert_allclose(result.normal, np.asarray([0.0, 0.0, 1.0]), atol=1.0e-12)
    np.testing.assert_allclose(result.point_b - result.point_a, np.asarray([0.0, 0.0, 0.2]), atol=1.0e-12)
    assert np.isclose(float(np.sum(result.barycentric_a)), 1.0, atol=1.0e-12)
    assert np.isclose(float(np.sum(result.barycentric_b)), 1.0, atol=1.0e-12)
