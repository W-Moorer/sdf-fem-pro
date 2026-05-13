import numpy as np
import pytest

import sfc.sdf as sdf
from sfc.mesh import extract_boundary_faces
from sfc.mesh.topology import VolumeMesh
from sfc.sdf.dynamic_surface_sdf import (
    _slow_reference_dynamic_surface_sdf,
    dynamic_surface_sdf,
)


TRIANGLE_NODES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
TRIANGLE_FACE = np.array([[0, 1, 2]], dtype=np.int64)


def _unit_tet_surface() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mesh = VolumeMesh(X=X, elements=np.array([[0, 1, 2, 3]], dtype=np.int64))
    faces, _ = extract_boundary_faces(mesh.elements, mesh.X)
    return X, faces


def test_distance_to_single_triangle() -> None:
    result = dynamic_surface_sdf(
        np.array([0.25, 0.25, 2.0]),
        TRIANGLE_NODES,
        TRIANGLE_FACE,
        candidate_face_ids=np.array([0]),
    )

    assert np.isclose(result.g, 2.0)
    assert np.allclose(result.n, [0.0, 0.0, 1.0])
    assert result.face_id == 0
    assert np.allclose(result.w, [0.5, 0.25, 0.25])
    assert np.allclose(result.p, [0.25, 0.25, 0.0])


def test_distance_to_tetrahedron_boundary() -> None:
    X, faces = _unit_tet_surface()

    result = _slow_reference_dynamic_surface_sdf(np.array([0.2, 0.2, -0.3]), X, faces)

    assert np.isclose(result.g, 0.3)
    assert np.allclose(result.n, [0.0, 0.0, -1.0])
    assert np.allclose(result.p, [0.2, 0.2, 0.0])


def test_closed_tetrahedron_inside_outside_sign_convention() -> None:
    X, faces = _unit_tet_surface()

    inside = _slow_reference_dynamic_surface_sdf(np.array([0.1, 0.1, 0.1]), X, faces)
    outside = _slow_reference_dynamic_surface_sdf(np.array([0.1, 0.1, -0.1]), X, faces)

    assert inside.g < 0.0
    assert outside.g > 0.0


def test_closed_tetrahedron_sign_is_invariant_after_negative_tet_reordering() -> None:
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    positive = VolumeMesh(X=X, elements=np.array([[0, 1, 2, 3]], dtype=np.int64))
    negative = VolumeMesh(X=X, elements=np.array([[0, 2, 1, 3]], dtype=np.int64))
    positive_faces, _ = extract_boundary_faces(positive.elements, positive.X)
    negative_faces, _ = extract_boundary_faces(negative.elements, negative.X)

    for point in (np.array([0.1, 0.1, 0.1]), np.array([0.1, 0.1, -0.1])):
        positive_gap = _slow_reference_dynamic_surface_sdf(point, positive.X, positive_faces).g
        negative_gap = _slow_reference_dynamic_surface_sdf(point, negative.X, negative_faces).g

        assert np.isclose(positive_gap, negative_gap)


def test_closed_tetrahedron_sign_near_face_edge_and_vertex_features() -> None:
    X, faces = _unit_tet_surface()
    inside_points = [
        np.array([0.2, 0.2, 0.02]),
        np.array([0.45, 0.02, 0.02]),
        np.array([0.02, 0.02, 0.02]),
    ]
    outside_points = [
        np.array([0.2, 0.2, -0.02]),
        np.array([0.45, -0.02, 0.02]),
        np.array([-0.02, -0.02, -0.02]),
    ]

    for point in inside_points:
        assert _slow_reference_dynamic_surface_sdf(point, X, faces).g < 0.0
    for point in outside_points:
        assert _slow_reference_dynamic_surface_sdf(point, X, faces).g > 0.0


def test_dynamic_surface_sdf_requires_candidate_face_ids() -> None:
    with pytest.raises(ValueError, match="candidate_face_ids"):
        dynamic_surface_sdf(
            np.array([0.25, 0.25, 1.0]),
            TRIANGLE_NODES,
            TRIANGLE_FACE,
            candidate_face_ids=np.empty(0, dtype=np.int64),
        )


def test_slow_reference_query_is_not_public_sdf_api() -> None:
    assert "slow_reference_dynamic_surface_sdf" not in sdf.__all__
    assert not hasattr(sdf, "slow_reference_dynamic_surface_sdf")


def test_sdf_gradient_with_respect_to_query_point_matches_normal() -> None:
    x = np.array([0.25, 0.25, 0.7])
    eps = 1e-6

    result = dynamic_surface_sdf(x, TRIANGLE_NODES, TRIANGLE_FACE, np.array([0]))
    grad_fd = np.zeros(3)
    for axis in range(3):
        perturb = np.zeros(3)
        perturb[axis] = eps
        g_plus = dynamic_surface_sdf(
            x + perturb,
            TRIANGLE_NODES,
            TRIANGLE_FACE,
            np.array([0]),
        ).g
        g_minus = dynamic_surface_sdf(
            x - perturb,
            TRIANGLE_NODES,
            TRIANGLE_FACE,
            np.array([0]),
        ).g
        grad_fd[axis] = (g_plus - g_minus) / (2.0 * eps)

    assert np.allclose(grad_fd, result.n)
    assert np.isclose(np.linalg.norm(result.n), 1.0)
