import numpy as np

from sfc.mesh.boundary import extract_boundary_faces, extract_boundary_triangles
from sfc.mesh.topology import VolumeMesh


def _sorted_face_keys(faces: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(sorted(map(int, face))) for face in faces}


def test_one_tetrahedron_has_four_boundary_faces() -> None:
    elements = np.array([[0, 1, 2, 3]])

    faces, adjacent = extract_boundary_faces(elements)

    assert faces.shape == (4, 3)
    assert adjacent.tolist() == [0, 0, 0, 0]
    assert len(_sorted_face_keys(faces)) == 4


def test_two_tetrahedra_sharing_one_face_have_six_boundary_faces() -> None:
    elements = np.array(
        [
            [0, 1, 2, 3],
            [0, 2, 1, 4],
        ]
    )

    faces, adjacent = extract_boundary_faces(elements)

    assert faces.shape == (6, 3)
    assert adjacent.shape == (6,)
    assert _sorted_face_keys(faces) == {
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
        (0, 1, 4),
        (0, 2, 4),
        (1, 2, 4),
    }


def test_boundary_faces_are_not_duplicated() -> None:
    elements = np.array(
        [
            [0, 1, 2, 3],
            [0, 2, 1, 4],
        ]
    )

    faces, _ = extract_boundary_faces(elements)
    keys = [tuple(sorted(map(int, face))) for face in faces]

    assert len(keys) == len(set(keys))
    assert (0, 1, 2) not in keys


def test_boundary_faces_are_oriented_outward_with_coordinates() -> None:
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mesh = VolumeMesh(X=X, elements=np.array([[0, 2, 1, 3]]))

    faces, adjacent = extract_boundary_faces(mesh.elements, mesh.X)
    tet_centroid = mesh.X[mesh.elements[0]].mean(axis=0)

    assert faces.shape == (4, 3)
    assert adjacent.tolist() == [0, 0, 0, 0]
    for face in faces:
        triangle = mesh.X[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        face_centroid = triangle.mean(axis=0)

        assert np.dot(normal, face_centroid - tet_centroid) > 0.0


def _unit_hex_mesh() -> VolumeMesh:
    return VolumeMesh(
        X=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        ),
        elements=np.array([[0, 1, 2, 3, 4, 5, 6, 7]]),
        element_type="hex8",
    )


def test_one_hex8_has_twelve_boundary_triangles() -> None:
    mesh = _unit_hex_mesh()

    triangles, adjacent = extract_boundary_triangles(mesh.elements, mesh.X, element_type=mesh.element_type)

    assert triangles.shape == (12, 3)
    assert adjacent.tolist() == [0] * 12
    assert len(_sorted_face_keys(triangles)) == 12


def test_hex8_boundary_triangles_are_oriented_outward() -> None:
    mesh = _unit_hex_mesh()
    triangles, _ = extract_boundary_triangles(mesh.elements, mesh.X, element_type="hex8")
    element_centroid = mesh.X[mesh.elements[0]].mean(axis=0)

    for triangle_ids in triangles:
        triangle = mesh.X[triangle_ids]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        face_centroid = triangle.mean(axis=0)
        assert np.dot(normal, face_centroid - element_centroid) > 0.0


def test_two_hex8_sharing_one_quad_have_twenty_boundary_triangles() -> None:
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
        ]
    )
    elements = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 8, 9, 2, 5, 10, 11, 6],
        ]
    )

    triangles, adjacent = extract_boundary_triangles(elements, X, element_type="hex8")

    assert triangles.shape == (20, 3)
    assert adjacent.shape == (20,)
