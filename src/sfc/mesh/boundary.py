"""Boundary topology utilities for volume meshes."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


_TET4_LOCAL_FACES = np.array(
    [
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [0, 3, 2],
    ],
    dtype=np.int64,
)


def _as_tet4_connectivity(elements: np.ndarray) -> np.ndarray:
    conn = np.asarray(elements, dtype=np.int64)
    if conn.ndim != 2 or conn.shape[1] != 4:
        raise ValueError("tet4 connectivity must have shape (m, 4)")
    if np.any(conn < 0):
        raise ValueError("tet4 connectivity cannot contain negative node indices")
    if conn.size and np.any(np.diff(np.sort(conn, axis=1), axis=1) == 0):
        raise ValueError("each tet4 element must reference four distinct nodes")
    return conn


def _as_coordinates(X: np.ndarray | None, elements: np.ndarray) -> np.ndarray | None:
    if X is None:
        return None
    coords = np.asarray(X, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("X must have shape (n, 3)")
    if elements.size and int(elements.max()) >= coords.shape[0]:
        raise ValueError("elements reference node indices outside X")
    return coords


def _orient_face_outward(face: np.ndarray, tet: np.ndarray, X: np.ndarray) -> np.ndarray:
    triangle = X[face]
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    face_centroid = triangle.mean(axis=0)
    tet_centroid = X[tet].mean(axis=0)
    if float(np.dot(normal, face_centroid - tet_centroid)) < 0.0:
        return face[[0, 2, 1]]
    return face


def extract_boundary_faces(
    elements: np.ndarray,
    X: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique boundary triangle faces and adjacent element ids.

    Faces shared by two or more tetrahedra are treated as interior/non-boundary
    faces. Duplicate detection is orientation independent; the returned face
    orientation is the local positive-TET4 orientation unless coordinates are
    supplied, in which case each boundary face is oriented outward.
    """

    conn = _as_tet4_connectivity(elements)
    coords = _as_coordinates(X, conn)
    occurrences: dict[tuple[int, int, int], list[tuple[np.ndarray, int]]] = defaultdict(list)

    for elem_id, tet in enumerate(conn):
        for local_face in _TET4_LOCAL_FACES:
            face = tet[local_face].copy()
            key = tuple(sorted(int(node) for node in face))
            occurrences[key].append((face, elem_id))

    boundary_faces: list[np.ndarray] = []
    adjacent_element_ids: list[int] = []
    for face_occurrences in occurrences.values():
        if len(face_occurrences) == 1:
            face, elem_id = face_occurrences[0]
            if coords is not None:
                face = _orient_face_outward(face, conn[elem_id], coords)
            boundary_faces.append(face)
            adjacent_element_ids.append(elem_id)

    if not boundary_faces:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int64)

    return (
        np.asarray(boundary_faces, dtype=np.int64),
        np.asarray(adjacent_element_ids, dtype=np.int64),
    )
