import numpy as np

from sfc.contact.broad_phase import UniformTriangleAABBHash, triangle_aabbs


NODES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
        [3.0, 2.0, 0.0],
        [2.0, 3.0, 0.0],
    ]
)
FACES = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)


def test_candidate_is_found_when_point_lies_above_triangle() -> None:
    broad_phase = UniformTriangleAABBHash.from_surface(
        NODES,
        FACES,
        delta_safe=0.2,
        cell_size=0.5,
    )

    candidates = broad_phase.query_point(np.array([0.25, 0.25, 0.1]))

    assert candidates.tolist() == [0]


def test_no_candidate_when_point_is_far_away() -> None:
    broad_phase = UniformTriangleAABBHash.from_surface(
        NODES,
        FACES,
        delta_safe=0.1,
        cell_size=0.5,
    )

    candidates = broad_phase.query_point(np.array([10.0, 10.0, 10.0]))

    assert candidates.size == 0


def test_padding_increases_candidate_set() -> None:
    no_padding = UniformTriangleAABBHash.from_surface(
        NODES,
        FACES,
        delta_safe=0.0,
        cell_size=0.5,
    )
    padded = UniformTriangleAABBHash.from_surface(
        NODES,
        FACES,
        delta_safe=0.2,
        cell_size=0.5,
    )
    point = np.array([1.1, 0.5, 0.0])

    assert no_padding.query_point(point).size == 0
    assert padded.query_point(point).tolist() == [0]


def test_no_false_negative_in_simple_constructed_case() -> None:
    delta = 0.35
    broad_phase = UniformTriangleAABBHash.from_surface(
        NODES,
        FACES,
        delta_safe=delta,
        cell_size=0.25,
    )
    aabb_min, aabb_max = triangle_aabbs(NODES, FACES, delta)

    query_points = np.array(
        [
            (aabb_min[0] + aabb_max[0]) / 2.0,
            aabb_min[0],
            aabb_max[0],
            (aabb_min[1] + aabb_max[1]) / 2.0,
        ]
    )

    candidates = broad_phase.query_points(query_points)

    assert 0 in candidates[0]
    assert 0 in candidates[1]
    assert 0 in candidates[2]
    assert 1 in candidates[3]
