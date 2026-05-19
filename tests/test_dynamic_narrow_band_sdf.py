from __future__ import annotations

import importlib

import numpy as np
import pytest

from sfc.contact import (
    SurfaceSample,
    field_contact_constraint_from_sample,
    field_contact_jacobian_row,
    field_penalty_contact_response,
)
from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF, RequiredPointSDFWorkspace
from sfc.sdf.dynamic_surface_sdf import (
    compiled_projection_available,
    surface_projection_distance_kernel,
    surface_projection_distance_kernel_batch_candidates,
    surface_projection_distance_kernel_batch_all_faces,
    surface_projection_distance_kernel_batch_all_faces_compiled,
    surface_projection_distance_kernel_batch_padded_aabb_compiled,
)


def _plane_surface() -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return X, faces


def _large_triangle_surface() -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    return X, np.asarray([[0, 1, 2]], dtype=np.int64)


def _nonplanar_surface(resolution: int = 8) -> tuple[np.ndarray, np.ndarray]:
    nodes = []
    for j in range(resolution + 1):
        y = j / resolution
        for i in range(resolution + 1):
            x = i / resolution
            z = 0.08 * np.sin(np.pi * x) * np.sin(np.pi * y)
            nodes.append([x, y, z])
    faces = []
    stride = resolution + 1
    for j in range(resolution):
        for i in range(resolution):
            n00 = j * stride + i
            n10 = n00 + 1
            n01 = n00 + stride
            n11 = n01 + 1
            faces.append([n00, n10, n11])
            faces.append([n00, n11, n01])
    return np.asarray(nodes, dtype=float), np.asarray(faces, dtype=np.int64)


def _nonplanar_z(x: float, y: float) -> float:
    return float(0.08 * np.sin(np.pi * x) * np.sin(np.pi * y))


def _plane_sdf() -> DynamicNarrowBandSDF:
    X, faces = _plane_surface()
    return DynamicNarrowBandSDF.build(
        X,
        faces,
        spacing=0.25,
        band_radius=0.5,
        origin=np.asarray([-0.25, -0.25, -0.5], dtype=float),
        shape=(7, 7, 5),
    )


def test_plane_sdf_exactness_and_payload() -> None:
    sdf = _plane_sdf()
    x = np.asarray([0.37, 0.41, 0.13], dtype=float)

    phi, normal = sdf.query_gap_normal(x)
    payload = sdf.query_payload(x)

    assert phi == pytest.approx(0.13, abs=1.0e-14)
    assert np.allclose(normal, [0.0, 0.0, 1.0])
    assert np.isclose(np.sum(payload.grid_weights), 1.0)
    assert payload.face_node_ids.shape == (8, 3)
    assert payload.barycentric.shape == (8, 3)


def test_interpolation_consistency() -> None:
    sdf = _plane_sdf()
    x = np.asarray([0.37, 0.41, -0.17], dtype=float)
    stencil = sdf.grid.interpolation(x)
    corner_phi = sdf.grid.phi[
        stencil.corner_indices[:, 0],
        stencil.corner_indices[:, 1],
        stencil.corner_indices[:, 2],
    ]

    assert sdf.query_phi(x) == pytest.approx(float(stencil.weights @ corner_phi))


def test_required_point_workspace_matches_build_required_points() -> None:
    X, faces = _nonplanar_surface(resolution=4)
    points = np.asarray(
        [
            [0.35, 0.25, _nonplanar_z(0.35, 0.25) + 0.035],
            [0.62, 0.58, _nonplanar_z(0.62, 0.58) - 0.025],
            [0.48, 0.72, _nonplanar_z(0.48, 0.72) + 0.015],
        ],
        dtype=float,
    )
    spacing = 0.125
    band = 0.25
    reference = DynamicNarrowBandSDF.build_required_points(
        X,
        faces,
        points,
        spacing=spacing,
        band_radius=band,
        padding=band,
    )
    workspace = RequiredPointSDFWorkspace.from_surface(
        X,
        faces,
        spacing=spacing,
        band_radius=band,
        padding=band,
    )
    cached = workspace.build(X, faces, points)

    for point in points:
        assert cached.query_phi(point) == pytest.approx(reference.query_phi(point), abs=1.0e-14)
        assert np.allclose(cached.query_spatial_derivative_phi(point), reference.query_spatial_derivative_phi(point))
        assert np.array_equal(cached.query_payload(point).face_ids, reference.query_payload(point).face_ids)

    moved = X.copy()
    moved[:, 2] += 0.01 * moved[:, 0]
    moved_cached = workspace.build(moved, faces, points)
    moved_reference = DynamicNarrowBandSDF.build_required_points(
        moved,
        faces,
        points,
        spacing=spacing,
        band_radius=band,
        origin=workspace.origin,
        shape=workspace.shape,
    )
    for point in points:
        assert moved_cached.query_phi(point) == pytest.approx(moved_reference.query_phi(point), abs=1.0e-14)


def test_compiled_all_face_projection_matches_numpy_batch() -> None:
    if not compiled_projection_available():
        pytest.skip("compiled projection backend is not available")
    X, faces = _nonplanar_surface(resolution=4)
    points = np.asarray(
        [
            [0.35, 0.25, _nonplanar_z(0.35, 0.25) + 0.035],
            [0.62, 0.58, _nonplanar_z(0.62, 0.58) - 0.025],
            [0.48, 0.72, _nonplanar_z(0.48, 0.72) + 0.015],
            [0.12, 0.63, _nonplanar_z(0.12, 0.63) + 0.045],
        ],
        dtype=float,
    )
    reference = surface_projection_distance_kernel_batch_all_faces(points, X, faces)
    compiled = surface_projection_distance_kernel_batch_all_faces_compiled(points, X, faces)
    assert np.allclose(compiled.g, reference.g, atol=1.0e-13)
    assert np.allclose(compiled.n, reference.n, atol=1.0e-13)
    assert np.array_equal(compiled.face_id, reference.face_id)
    assert np.allclose(compiled.w, reference.w, atol=1.0e-13)
    assert np.allclose(compiled.p, reference.p, atol=1.0e-13)


def test_compiled_padded_aabb_projection_matches_all_faces_in_band() -> None:
    if not compiled_projection_available():
        pytest.skip("compiled projection backend is not available")
    X, faces = _nonplanar_surface(resolution=4)
    points = np.asarray(
        [
            [0.35, 0.25, _nonplanar_z(0.35, 0.25) + 0.035],
            [0.62, 0.58, _nonplanar_z(0.62, 0.58) - 0.025],
            [0.48, 0.72, _nonplanar_z(0.48, 0.72) + 0.015],
            [0.12, 0.63, _nonplanar_z(0.12, 0.63) + 0.045],
        ],
        dtype=float,
    )
    reference = surface_projection_distance_kernel_batch_all_faces(points, X, faces)
    compiled = surface_projection_distance_kernel_batch_padded_aabb_compiled(
        points,
        X,
        faces,
        delta_safe=0.25,
    )
    assert np.allclose(compiled.g, reference.g, atol=1.0e-13)
    assert np.allclose(compiled.n, reference.n, atol=1.0e-13)
    assert np.array_equal(compiled.face_id, reference.face_id)
    assert np.allclose(compiled.w, reference.w, atol=1.0e-13)
    assert np.allclose(compiled.p, reference.p, atol=1.0e-13)


def test_finite_difference_gradient_consistency() -> None:
    sdf = _plane_sdf()
    x = np.asarray([0.35, 0.45, 0.05], dtype=float)
    eps = 1.0e-6
    fd = np.zeros(3, dtype=float)
    for axis in range(3):
        dx = np.zeros(3, dtype=float)
        dx[axis] = eps
        fd[axis] = (sdf.query_phi(x + dx) - sdf.query_phi(x - dx)) / (2.0 * eps)

    assert np.allclose(sdf.query_gradient(x), fd, atol=1.0e-9)


@pytest.mark.parametrize("spacing", [0.18, 0.12])
def test_nonplanar_scalar_derivative_matches_phi_finite_difference(spacing: float) -> None:
    X, faces = _nonplanar_surface()
    sdf = DynamicNarrowBandSDF.build(
        X,
        faces,
        spacing=spacing,
        band_radius=0.35,
        padding=0.35,
        cell_size=spacing,
    )
    points = np.asarray(
        [
            [0.31, 0.47, _nonplanar_z(0.31, 0.47) + 0.035],
            [0.58, 0.36, _nonplanar_z(0.58, 0.36) - 0.045],
            [0.43, 0.69, _nonplanar_z(0.43, 0.69) + 0.055],
        ],
        dtype=float,
    )
    eps = 1.0e-6
    for point in points:
        grad = sdf.query_spatial_derivative_phi(point)
        fd = np.zeros(3, dtype=float)
        for axis in range(3):
            dx = np.zeros(3, dtype=float)
            dx[axis] = eps
            fd[axis] = (sdf.query_phi(point + dx) - sdf.query_phi(point - dx)) / (2.0 * eps)
        assert np.allclose(sdf.query_gradient(point), grad)
        assert np.allclose(grad, fd, atol=1.0e-8)


@pytest.mark.parametrize("spacing", [0.18, 0.12])
def test_nonplanar_scalar_derivative_normal_has_bounded_projection_error(spacing: float) -> None:
    X, faces = _nonplanar_surface()
    sdf = DynamicNarrowBandSDF.build(
        X,
        faces,
        spacing=spacing,
        band_radius=0.35,
        padding=0.35,
        cell_size=spacing,
    )
    points = np.asarray(
        [
            [0.34, 0.41, _nonplanar_z(0.34, 0.41) + 0.04],
            [0.63, 0.52, _nonplanar_z(0.63, 0.52) - 0.05],
        ],
        dtype=float,
    )
    all_faces = np.arange(faces.shape[0], dtype=np.int64)
    for point in points:
        reference = surface_projection_distance_kernel(point, X, faces, all_faces)
        _phi, normal = sdf.query_gap_normal(point)
        assert np.linalg.norm(normal - reference.n) < 0.35


def test_eikonal_residual_on_plane() -> None:
    sdf = _plane_sdf()
    residual = sdf.eikonal_residuals()

    assert residual.size > 0
    assert float(np.max(np.abs(residual))) < 1.0e-12


def test_field_phi_and_normal_match_current_surface_projection() -> None:
    X, faces = _plane_surface()
    sdf = _plane_sdf()
    points = np.asarray(
        [
            [0.2, 0.2, -0.2],
            [0.6, 0.25, 0.05],
            [0.25, 0.75, 0.21],
        ],
        dtype=float,
    )
    all_faces = np.arange(faces.shape[0], dtype=np.int64)
    for point in points:
        reference = surface_projection_distance_kernel(point, X, faces, all_faces)
        phi, normal = sdf.query_gap_normal(point)
        assert phi == pytest.approx(reference.g, abs=1.0e-12)
        assert np.linalg.norm(normal - reference.n) < 1.0e-12


def test_out_of_band_query_raises() -> None:
    sdf = _plane_sdf()

    with pytest.raises(ValueError, match="outside the narrow-band grid|outside the valid narrow band"):
        sdf.query_phi(np.asarray([0.5, 0.5, 0.8], dtype=float))


def test_field_query_path_does_not_call_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    sdf = _plane_sdf()

    def fail_projection(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("projection was called by field query")

    kernel_module = importlib.import_module("sfc.sdf.dynamic_surface_sdf")
    monkeypatch.setattr(kernel_module, "closest_point_on_triangle", fail_projection)

    x = np.asarray([0.35, 0.45, -0.1], dtype=float)
    assert sdf.query_phi(x) == pytest.approx(-0.1)
    assert np.allclose(sdf.query_gradient(x), [0.0, 0.0, 1.0])
    assert np.allclose(sdf.query_gap_normal(x)[1], [0.0, 0.0, 1.0])
    assert sdf.query_payload(x).face_node_ids.shape == (8, 3)


def test_required_point_sparse_field_matches_full_field_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    X, faces = _nonplanar_surface()
    origin = np.asarray([-0.12, -0.12, -0.30], dtype=float)
    shape = (12, 12, 7)
    points = np.asarray(
        [
            [0.31, 0.47, _nonplanar_z(0.31, 0.47) + 0.035],
            [0.58, 0.36, _nonplanar_z(0.58, 0.36) - 0.045],
            [0.43, 0.69, _nonplanar_z(0.43, 0.69) + 0.055],
        ],
        dtype=float,
    )
    full = DynamicNarrowBandSDF.build(
        X,
        faces,
        spacing=0.12,
        band_radius=0.40,
        origin=origin,
        shape=shape,
        cell_size=0.12,
    )
    sparse = DynamicNarrowBandSDF.build_required_points(
        X,
        faces,
        points,
        spacing=0.12,
        band_radius=0.40,
        origin=origin,
        shape=shape,
        cell_size=0.12,
    )

    assert int(sparse.grid.metadata["required_node_count"]) < int(np.prod(shape))
    assert sparse.stats.valid_node_count == int(sparse.grid.metadata["required_node_count"])
    for point in points:
        assert sparse.query_phi(point) == pytest.approx(full.query_phi(point), abs=1.0e-14)
        assert np.allclose(sparse.query_gradient(point), full.query_gradient(point), atol=1.0e-14)
        assert np.allclose(sparse.query_gap_normal(point)[1], full.query_gap_normal(point)[1], atol=1.0e-14)
        sparse_payload = sparse.query_payload(point)
        full_payload = full.query_payload(point)
        assert np.array_equal(sparse_payload.face_node_ids, full_payload.face_node_ids)
        assert np.allclose(sparse_payload.barycentric, full_payload.barycentric)
        assert np.allclose(sparse_payload.normals, full_payload.normals)

    def fail_projection(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("projection was called by sparse field query")

    kernel_module = importlib.import_module("sfc.sdf.dynamic_surface_sdf")
    monkeypatch.setattr(kernel_module, "closest_point_on_triangle", fail_projection)
    for point in points:
        sparse.query_phi(point)
        sparse.query_gradient(point)
        sparse.query_payload(point)


def test_build_required_points_can_use_candidate_group_batch_projection() -> None:
    X, faces = _nonplanar_surface()
    origin = np.asarray([-0.1, -0.1, -0.35], dtype=float)
    shape = (9, 9, 7)
    points = np.asarray(
        [
            [0.31, 0.47, _nonplanar_z(0.31, 0.47) + 0.035],
            [0.58, 0.36, _nonplanar_z(0.58, 0.36) - 0.045],
            [0.43, 0.69, _nonplanar_z(0.43, 0.69) + 0.055],
        ],
        dtype=float,
    )
    full = DynamicNarrowBandSDF.build(
        X,
        faces,
        spacing=0.12,
        band_radius=0.40,
        origin=origin,
        shape=shape,
        cell_size=0.12,
    )
    sparse = DynamicNarrowBandSDF.build_required_points(
        X,
        faces,
        points,
        spacing=0.12,
        band_radius=0.40,
        origin=origin,
        shape=shape,
        cell_size=0.12,
        batch_projection_threshold=1,
    )

    assert sparse.grid.metadata["projection_mode"] == "candidate_group_batch"
    for point in points:
        assert sparse.query_phi(point) == pytest.approx(full.query_phi(point), abs=1.0e-14)
        assert np.allclose(sparse.query_gradient(point), full.query_gradient(point), atol=1.0e-14)


def test_candidate_batch_projection_matches_scalar_kernel() -> None:
    X, faces = _nonplanar_surface()
    candidates = np.asarray([0, 1, 2, 3], dtype=np.int64)
    points = np.asarray(
        [
            [0.31, 0.47, _nonplanar_z(0.31, 0.47) + 0.035],
            [0.58, 0.36, _nonplanar_z(0.58, 0.36) - 0.045],
            [0.43, 0.69, _nonplanar_z(0.43, 0.69) + 0.055],
        ],
        dtype=float,
    )

    batch = surface_projection_distance_kernel_batch_candidates(points, X, faces, candidates)
    scalar = [surface_projection_distance_kernel(point, X, faces, candidates) for point in points]

    np.testing.assert_allclose(batch.g, [item.g for item in scalar], atol=1.0e-14)
    np.testing.assert_array_equal(batch.face_id, [item.face_id for item in scalar])
    np.testing.assert_allclose(batch.w, np.vstack([item.w for item in scalar]), atol=1.0e-14)
    np.testing.assert_allclose(batch.n, np.vstack([item.n for item in scalar]), atol=1.0e-14)


def test_field_contact_slave_jacobian_matches_finite_difference() -> None:
    X, faces = _large_triangle_surface()
    sdf = DynamicNarrowBandSDF.build(
        X,
        faces,
        spacing=0.25,
        band_radius=0.5,
        origin=np.asarray([0.0, 0.0, -0.5], dtype=float),
        shape=(5, 5, 5),
    )
    slave = np.asarray([[0.42, 0.37, -0.13]], dtype=float)
    sample = SurfaceSample(np.asarray([0], dtype=np.int64), np.asarray([1.0]), np.asarray([0], dtype=np.int64))
    constraint = field_contact_constraint_from_sample(slave, sample, sdf)
    J = field_contact_jacobian_row(
        constraint,
        n_total_dofs=12,
        slave_dof_offset=0,
        master_dof_offset=3,
    ).toarray()[0]

    eps = 1.0e-6
    fd = np.zeros(3, dtype=float)
    for dof in range(3):
        dq = np.zeros_like(slave)
        dq.ravel()[dof] = eps
        g_plus = field_contact_constraint_from_sample(slave + dq, sample, sdf).g
        g_minus = field_contact_constraint_from_sample(slave - dq, sample, sdf).g
        fd[dof] = (g_plus - g_minus) / (2.0 * eps)

    assert np.allclose(fd, J[:3], atol=1.0e-9)


def test_nonplanar_field_contact_slave_jacobian_uses_scalar_phi_derivative() -> None:
    X, faces = _nonplanar_surface()
    sdf = DynamicNarrowBandSDF.build(
        X,
        faces,
        spacing=0.12,
        band_radius=0.35,
        padding=0.35,
        cell_size=0.12,
    )
    point = np.asarray([0.46, 0.54, _nonplanar_z(0.46, 0.54) - 0.04], dtype=float)
    weights = np.asarray([0.2, 0.3, 0.5], dtype=float)
    offsets = np.asarray(
        [
            [0.02, 0.0, 0.0],
            [0.0, 0.02, 0.0],
            [-0.008, -0.012, 0.0],
        ],
        dtype=float,
    )
    slave = point[None, :] + offsets
    assert np.allclose(weights @ slave, point)

    sample = SurfaceSample(np.asarray([0, 1, 2], dtype=np.int64), weights, np.asarray([0], dtype=np.int64))
    constraint = field_contact_constraint_from_sample(slave, sample, sdf)
    J = field_contact_jacobian_row(
        constraint,
        n_total_dofs=slave.size + X.size,
        slave_dof_offset=0,
        master_dof_offset=slave.size,
    ).toarray()[0]

    eps = 1.0e-6
    fd = np.zeros(slave.size, dtype=float)
    for dof in range(slave.size):
        dq = np.zeros_like(slave)
        dq.ravel()[dof] = eps
        g_plus = field_contact_constraint_from_sample(slave + dq, sample, sdf).g
        g_minus = field_contact_constraint_from_sample(slave - dq, sample, sdf).g
        fd[dof] = (g_plus - g_minus) / (2.0 * eps)

    assert np.allclose(fd, J[: slave.size], atol=1.0e-8)


def test_field_contact_master_jacobian_matches_finite_difference() -> None:
    X, faces = _large_triangle_surface()
    origin = np.asarray([0.0, 0.0, -0.5], dtype=float)
    shape = (5, 5, 5)
    sdf = DynamicNarrowBandSDF.build(
        X,
        faces,
        spacing=0.25,
        band_radius=0.5,
        origin=origin,
        shape=shape,
    )
    slave = np.asarray([[0.42, 0.37, -0.13]], dtype=float)
    sample = SurfaceSample(np.asarray([0], dtype=np.int64), np.asarray([1.0]), np.asarray([0], dtype=np.int64))
    constraint = field_contact_constraint_from_sample(slave, sample, sdf)
    J = field_contact_jacobian_row(
        constraint,
        n_total_dofs=12,
        slave_dof_offset=0,
        master_dof_offset=3,
    ).toarray()[0]

    def gap_from_master(master: np.ndarray) -> float:
        rebuilt = DynamicNarrowBandSDF.build(
            master.reshape(3, 3),
            faces,
            spacing=0.25,
            band_radius=0.5,
            origin=origin,
            shape=shape,
        )
        return field_contact_constraint_from_sample(slave, sample, rebuilt).g

    eps = 1.0e-6
    q0 = X.ravel()
    fd = np.zeros_like(q0)
    for dof in range(q0.size):
        dq = np.zeros_like(q0)
        dq[dof] = eps
        fd[dof] = (gap_from_master(q0 + dq) - gap_from_master(q0 - dq)) / (2.0 * eps)

    assert np.allclose(fd, J[3:], atol=1.0e-6)


def test_field_penalty_force_action_reaction() -> None:
    X, faces = _large_triangle_surface()
    sdf = DynamicNarrowBandSDF.build(
        X,
        faces,
        spacing=0.25,
        band_radius=0.5,
        origin=np.asarray([0.0, 0.0, -0.5], dtype=float),
        shape=(5, 5, 5),
    )
    slave = np.asarray([[0.42, 0.37, -0.13]], dtype=float)
    sample = SurfaceSample(np.asarray([0], dtype=np.int64), np.asarray([1.0]), np.asarray([0], dtype=np.int64))
    constraint = field_contact_constraint_from_sample(slave, sample, sdf)

    force, K = field_penalty_contact_response(
        [constraint],
        stiffness=100.0,
        n_total_dofs=12,
        slave_dof_offset=0,
        master_dof_offset=3,
    )

    assert constraint.g < 0.0
    assert force[2] > 0.0
    assert np.allclose(force[:3] + force[3:].reshape(3, 3).sum(axis=0), 0.0, atol=1.0e-12)
    assert K.nnz > 0
