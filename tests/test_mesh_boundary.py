import numpy as np

from sfc.mesh.boundary import extract_boundary_faces
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
