from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import eye

from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF
from sfc.contact import (
    node_to_surface_field_penalty_response,
    surface_to_surface_field_penalty_response,
    surface_to_surface_field_penalty_response_vectorized,
    triangle_surface_quadrature_cache,
    triangle_surface_quadrature_samples,
)
from sfc.contact._cpp_field_contact import is_available as cpp_field_contact_available
from sfc.contact.narrow_phase import SurfaceSample


class _EmptyPlanePayload:
    def master_sensitivity_terms(self) -> tuple[np.ndarray, np.ndarray]:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=float)


class _PlaneSDF:
    def query_payload(self, _x: np.ndarray) -> _EmptyPlanePayload:
        return _EmptyPlanePayload()

    def query_phi(self, x: np.ndarray) -> float:
        return float(np.asarray(x, dtype=float)[2])

    def query_spatial_derivative_phi(self, _x: np.ndarray) -> np.ndarray:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)

    def query_geometric_normal(self, _x: np.ndarray) -> np.ndarray:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)


def test_triangle_surface_quadrature_weights_integrate_face_area() -> None:
    x = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)

    for order, expected_count in [(1, 1), (3, 3), (7, 7)]:
        samples, weights = triangle_surface_quadrature_samples(faces, x, order=order)
        assert len(samples) == expected_count
        assert weights.shape == (expected_count,)
        assert float(np.sum(weights)) == pytest.approx(1.0)
        for sample in samples:
            assert set(sample.node_ids.tolist()) == {0, 1, 2}
            assert float(np.sum(sample.weights)) == pytest.approx(1.0)


def test_surface_to_surface_response_area_integrates_constant_plane_gap() -> None:
    x = np.asarray([[0.0, 0.0, -0.1], [1.0, 0.0, -0.1], [0.0, 1.0, -0.1]], dtype=float)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)

    response = surface_to_surface_field_penalty_response(
        x,
        faces,
        _PlaneSDF(),  # type: ignore[arg-type]
        pressure_stiffness=10.0,
        n_total_dofs=9,
        quadrature_order=7,
    )

    assert response.active_count == 7
    assert response.min_gap == pytest.approx(-0.1)
    assert float(np.sum(response.quadrature_weights)) == pytest.approx(0.5)
    assert float(np.sum(response.force.reshape((-1, 3))[:, 2])) == pytest.approx(0.5)


def test_vectorized_surface_response_matches_reference_force() -> None:
    x = np.asarray(
        [
            [0.0, 0.0, -0.10],
            [1.0, 0.0, -0.08],
            [0.0, 1.0, -0.12],
            [1.0, 1.0, -0.07],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    cache = triangle_surface_quadrature_cache(faces, x, order=7)
    master_x = np.asarray([[-1.0, -1.0, 0.0], [2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [2.0, 2.0, 0.0]])
    master_faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    sdf = DynamicNarrowBandSDF.build_required_points(
        master_x,
        master_faces,
        cache.points(x),
        spacing=0.25,
        band_radius=0.5,
        padding=0.5,
        cell_size=0.25,
    )

    reference = surface_to_surface_field_penalty_response(
        x,
        faces,
        sdf,
        pressure_stiffness=10.0,
        n_total_dofs=24,
        quadrature_order=7,
        master_dof_offset=12,
    )
    vectorized = surface_to_surface_field_penalty_response_vectorized(
        x,
        faces,
        sdf,
        pressure_stiffness=10.0,
        n_total_dofs=24,
        quadrature_order=7,
        quadrature_cache=cache,
        master_dof_offset=12,
    )
    vectorized_with_stiffness = surface_to_surface_field_penalty_response_vectorized(
        x,
        faces,
        sdf,
        pressure_stiffness=10.0,
        n_total_dofs=24,
        quadrature_order=7,
        quadrature_cache=cache,
        master_dof_offset=12,
        assemble_stiffness=True,
    )
    vectorized_with_matrix_free_stiffness = surface_to_surface_field_penalty_response_vectorized(
        x,
        faces,
        sdf,
        pressure_stiffness=10.0,
        n_total_dofs=24,
        quadrature_order=7,
        quadrature_cache=cache,
        master_dof_offset=12,
        matrix_free_stiffness=True,
    )

    np.testing.assert_allclose(vectorized.force, reference.force, atol=1.0e-14)
    np.testing.assert_allclose(vectorized_with_stiffness.force, reference.force, atol=1.0e-14)
    np.testing.assert_allclose(vectorized_with_stiffness.stiffness.toarray(), reference.stiffness.toarray(), atol=1.0e-14)
    assert vectorized_with_matrix_free_stiffness.stiffness_operator is not None
    probe = np.linspace(-0.2, 0.3, 24)
    np.testing.assert_allclose(
        vectorized_with_matrix_free_stiffness.stiffness_operator.matvec(probe),
        reference.stiffness @ probe,
        atol=1.0e-14,
    )
    assert vectorized.active_count == reference.active_count
    assert vectorized.min_gap == pytest.approx(reference.min_gap)


def test_node_to_surface_response_preserves_single_point_scheme() -> None:
    x = np.asarray([[0.0, 0.0, -0.1]], dtype=float)
    samples = [SurfaceSample(np.asarray([0]), np.asarray([1.0]), np.empty(0, dtype=np.int64))]

    response = node_to_surface_field_penalty_response(
        x,
        samples,
        _PlaneSDF(),  # type: ignore[arg-type]
        pressure_stiffness=10.0,
        n_total_dofs=3,
        sample_area_weights=np.asarray([0.5], dtype=float),
    )

    assert response.active_count == 1
    assert response.quadrature_weights.tolist() == [0.5]
    assert response.force[2] == pytest.approx(0.5)


def test_cpp_field_contact_backend_matches_vectorized_force_when_built() -> None:
    if not cpp_field_contact_available():
        pytest.skip("C++ field-contact backend is not built")
    x = np.asarray(
        [
            [0.0, 0.0, -0.10],
            [1.0, 0.0, -0.08],
            [0.0, 1.0, -0.12],
            [1.0, 1.0, -0.07],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    cache = triangle_surface_quadrature_cache(faces, x, order=7)
    master_x = np.asarray([[-1.0, -1.0, 0.0], [2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [2.0, 2.0, 0.0]])
    master_faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    sdf = DynamicNarrowBandSDF.build_required_points(
        master_x,
        master_faces,
        cache.points(x),
        spacing=0.25,
        band_radius=0.5,
        padding=0.5,
        cell_size=0.25,
    )
    from sfc.contact._cpp_field_contact import surface_penalty_response

    force_cpp, gaps_cpp = surface_penalty_response(
        cache.points(x),
        cache.node_ids,
        cache.weights,
        cache.area_weights,
        sdf.grid.origin,
        sdf.grid.spacing,
        np.asarray(sdf.grid.shape, dtype=np.int64),
        sdf.grid.phi,
        sdf.grid.valid_mask,
        sdf.grid.closest_face_id,
        sdf.grid.barycentric,
        sdf.grid.closest_normal,
        sdf.boundary_faces,
        10.0,
        24,
        0,
        12,
    )
    reference = surface_to_surface_field_penalty_response_vectorized(
        x,
        faces,
        sdf,
        pressure_stiffness=10.0,
        n_total_dofs=24,
        quadrature_order=7,
        quadrature_cache=cache,
        master_dof_offset=12,
        matrix_free_stiffness=True,
    )
    np.testing.assert_allclose(force_cpp, reference.force, atol=1.0e-14)
    np.testing.assert_allclose(gaps_cpp, reference.gaps, atol=1.0e-14)


def test_cpp_contact_stiffness_matvec_matches_reference_when_built() -> None:
    if not cpp_field_contact_available():
        pytest.skip("C++ field-contact backend is not built")
    x = np.asarray(
        [
            [0.0, 0.0, -0.10],
            [1.0, 0.0, -0.08],
            [0.0, 1.0, -0.12],
            [1.0, 1.0, -0.07],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    cache = triangle_surface_quadrature_cache(faces, x, order=7)
    master_x = np.asarray([[-1.0, -1.0, 0.0], [2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [2.0, 2.0, 0.0]])
    master_faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    sdf = DynamicNarrowBandSDF.build_required_points(
        master_x,
        master_faces,
        cache.points(x),
        spacing=0.25,
        band_radius=0.5,
        padding=0.5,
        cell_size=0.25,
    )
    explicit = surface_to_surface_field_penalty_response_vectorized(
        x,
        faces,
        sdf,
        pressure_stiffness=10.0,
        n_total_dofs=24,
        quadrature_order=7,
        quadrature_cache=cache,
        master_dof_offset=12,
        assemble_stiffness=True,
    )
    matrix_free = surface_to_surface_field_penalty_response_vectorized(
        x,
        faces,
        sdf,
        pressure_stiffness=10.0,
        n_total_dofs=24,
        quadrature_order=7,
        quadrature_cache=cache,
        master_dof_offset=12,
        matrix_free_stiffness=True,
    )
    assert matrix_free.stiffness_operator is not None
    probe = np.cos(np.arange(24, dtype=float))
    np.testing.assert_allclose(matrix_free.stiffness_operator.matvec(probe), explicit.stiffness @ probe, atol=1.0e-14)


def test_cpp_contact_tangent_pcg_solver_matches_dense_reference_when_built() -> None:
    if not cpp_field_contact_available():
        pytest.skip("C++ field-contact backend is not built")
    x = np.asarray(
        [
            [0.0, 0.0, -0.10],
            [1.0, 0.0, -0.08],
            [0.0, 1.0, -0.12],
            [1.0, 1.0, -0.07],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    cache = triangle_surface_quadrature_cache(faces, x, order=7)
    master_x = np.asarray([[-1.0, -1.0, 0.0], [2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [2.0, 2.0, 0.0]])
    master_faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    sdf = DynamicNarrowBandSDF.build_required_points(
        master_x,
        master_faces,
        cache.points(x),
        spacing=0.25,
        band_radius=0.5,
        padding=0.5,
        cell_size=0.25,
    )
    explicit = surface_to_surface_field_penalty_response_vectorized(
        x,
        faces,
        sdf,
        pressure_stiffness=10.0,
        n_total_dofs=24,
        quadrature_order=7,
        quadrature_cache=cache,
        master_dof_offset=12,
        assemble_stiffness=True,
    )
    matrix_free = surface_to_surface_field_penalty_response_vectorized(
        x,
        faces,
        sdf,
        pressure_stiffness=10.0,
        n_total_dofs=24,
        quadrature_order=7,
        quadrature_cache=cache,
        master_dof_offset=12,
        matrix_free_stiffness=True,
    )
    assert matrix_free.stiffness_operator is not None
    from sfc.contact._cpp_field_contact import solve_contact_tangent_pcg

    effective = (2.0 * eye(24, format="csr")).tocsr()
    rhs = np.sin(np.arange(24, dtype=float))
    op = matrix_free.stiffness_operator
    reference = np.linalg.solve((effective + explicit.stiffness).toarray(), rhs)
    for preconditioner in ("sgs", "block-sgs"):
        solution, info, _iterations, residual = solve_contact_tangent_pcg(
            effective.indptr,
            effective.indices,
            effective.data,
            rhs,
            np.arange(24, dtype=np.int64),
            op.scale,
            op.slave_node_ids,
            op.slave_weights,
            op.gradients,
            op.face_node_ids,
            op.grid_weights,
            op.barycentric,
            op.normals,
            24,
            op.slave_dof_offset,
            op.master_dof_offset,
            1.0e-12,
            1.0e-14,
            200,
            preconditioner=preconditioner,
        )
        assert info == 0
        assert residual < 1.0e-10
        np.testing.assert_allclose(solution, reference, atol=1.0e-10)
