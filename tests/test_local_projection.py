import numpy as np

from sfc.sdf.local_projection import (
    closest_point_on_triangle,
    signed_point_triangle_gap,
)


TRIANGLE = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


def test_projection_inside_triangle() -> None:
    x = np.array([0.25, 0.25, 1.0])

    p, w, dist2, region = closest_point_on_triangle(x, *TRIANGLE)

    assert region == "face"
    assert np.allclose(p, [0.25, 0.25, 0.0])
    assert np.allclose(w, [0.5, 0.25, 0.25])
    assert np.isclose(dist2, 1.0)


def test_projection_to_edge() -> None:
    x = np.array([0.5, -0.25, 0.0])

    p, w, dist2, region = closest_point_on_triangle(x, *TRIANGLE)

    assert region == "edge"
    assert np.allclose(p, [0.5, 0.0, 0.0])
    assert np.allclose(w, [0.5, 0.5, 0.0])
    assert np.isclose(dist2, 0.25**2)


def test_projection_to_vertex() -> None:
    x = np.array([-0.5, -0.25, 0.0])

    p, w, _, region = closest_point_on_triangle(x, *TRIANGLE)

    assert region == "vertex"
    assert np.allclose(p, TRIANGLE[0])
    assert np.allclose(w, [1.0, 0.0, 0.0])


def test_barycentric_weights_sum_to_one() -> None:
    points = [
        np.array([0.25, 0.25, 1.0]),
        np.array([0.5, -0.25, 0.0]),
        np.array([-0.5, -0.25, 0.0]),
    ]

    for x in points:
        _, w, _, _ = closest_point_on_triangle(x, *TRIANGLE)
        assert np.isclose(np.sum(w), 1.0)


def test_signed_gap_normal_has_unit_length() -> None:
    x = np.array([0.25, 0.25, 2.0])

    g, n, w = signed_point_triangle_gap(
        x,
        TRIANGLE,
        normal_sign_reference=np.array([0.0, 0.0, 1.0]),
    )

    assert np.isclose(g, 2.0)
    assert np.isclose(np.linalg.norm(n), 1.0)
    assert np.allclose(w, [0.5, 0.25, 0.25])
