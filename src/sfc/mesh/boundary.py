"""Boundary topology utilities for volume meshes."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .topology import canonical_mesh_element_type


_TET4_LOCAL_FACES = np.array(
    [
        [0, 2, 1],
        [0, 1, 3],
        [1, 2, 3],
        [0, 3, 2],
    ],
    dtype=np.int64,
)

_HEX8_LOCAL_FACES = np.array(
    [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ],
    dtype=np.int64,
)


def _as_connectivity(elements: np.ndarray, *, element_type: str) -> np.ndarray:
    element_type = canonical_mesh_element_type(element_type)
    conn = np.asarray(elements, dtype=np.int64)
    nodes_per_element = 4 if element_type == "tet4" else 8
    if conn.ndim != 2 or conn.shape[1] != nodes_per_element:
        raise ValueError(f"{element_type} connectivity must have shape (m, {nodes_per_element})")
    if np.any(conn < 0):
        raise ValueError(f"{element_type} connectivity cannot contain negative node indices")
    if conn.size and np.any(np.diff(np.sort(conn, axis=1), axis=1) == 0):
        raise ValueError(f"each {element_type} element must reference {nodes_per_element} distinct nodes")
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


def _orient_polygon_outward(face: np.ndarray, element: np.ndarray, X: np.ndarray) -> np.ndarray:
    polygon = X[face]
    normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
    face_centroid = polygon.mean(axis=0)
    element_centroid = X[element].mean(axis=0)
    if float(np.dot(normal, face_centroid - element_centroid)) < 0.0:
        return face[::-1]
    return face


def _triangulate_face(face: np.ndarray) -> list[np.ndarray]:
    if face.size == 3:
        return [face]
    if face.size == 4:
        return [face[[0, 1, 2]], face[[0, 2, 3]]]
    raise ValueError("only triangular and quadrilateral boundary faces can be triangulated")


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

    return extract_boundary_triangles(elements, X, element_type="tet4")


def extract_boundary_triangles(
    elements: np.ndarray,
    X: np.ndarray | None = None,
    *,
    element_type: str = "tet4",
) -> tuple[np.ndarray, np.ndarray]:
    """Return boundary triangles and adjacent element ids for supported elements.

    ``tet4`` boundary faces are returned as one triangle per exposed face.
    ``hex8`` boundary quads are split into two triangles. Duplicate detection is
    orientation independent at the parent face level, so shared HEX8 quads do
    not leak two coincident boundary triangles.
    """

    element_type = canonical_mesh_element_type(element_type)

    conn = _as_connectivity(elements, element_type=element_type)
    coords = _as_coordinates(X, conn)
    local_faces = _TET4_LOCAL_FACES if element_type == "tet4" else _HEX8_LOCAL_FACES
    occurrences: dict[tuple[int, ...], list[tuple[np.ndarray, int]]] = defaultdict(list)
    for elem_id, element in enumerate(conn):
        for local_face in local_faces:
            face = element[local_face].copy()
            key = tuple(sorted(int(node) for node in face))
            occurrences[key].append((face, elem_id))

    boundary_triangles: list[np.ndarray] = []
    adjacent_element_ids: list[int] = []
    for face_occurrences in occurrences.values():
        if len(face_occurrences) == 1:
            face, elem_id = face_occurrences[0]
            if coords is not None:
                face = _orient_polygon_outward(face, conn[elem_id], coords)
            for triangle in _triangulate_face(face):
                boundary_triangles.append(triangle)
                adjacent_element_ids.append(elem_id)

    if not boundary_triangles:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int64)

    return (
        np.asarray(boundary_triangles, dtype=np.int64),
        np.asarray(adjacent_element_ids, dtype=np.int64),
    )
