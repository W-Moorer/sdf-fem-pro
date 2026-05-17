"""Internal current-surface projection kernel for SDF grid construction.

The kernel operates on current boundary triangles. Those triangles may come
from TET4 faces, triangulated HEX8/C3D8 faces, or any other finite-element
boundary that has been converted to oriented triangles by the caller.

This module is retained for grid population, tests, validation references, and
explicit optional refinement. The main dynamic SDF query API is
``DynamicNarrowBandSDF``, whose query path interpolates stored field data.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .local_projection import closest_point_on_triangle


class SurfaceSDFResult(NamedTuple):
    """Result of a current-surface projection-kernel evaluation."""

    g: float
    n: np.ndarray
    face_id: int
    w: np.ndarray
    p: np.ndarray


def _as_point(value: np.ndarray, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return point


def _as_surface(
    x_current: np.ndarray,
    boundary_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(x_current, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("x_current must have shape (n, 3)")

    faces = np.asarray(boundary_faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("boundary_faces must have shape (m, 3)")
    if np.any(faces < 0):
        raise ValueError("boundary_faces cannot contain negative node indices")
    if faces.size and int(faces.max()) >= X.shape[0]:
        raise ValueError("boundary_faces reference nodes outside x_current")
    return X, faces


def _unit_triangle_normal(triangle: np.ndarray) -> np.ndarray:
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    norm = np.linalg.norm(normal)
    if norm <= 0.0:
        raise ValueError("boundary face has zero area")
    return normal / norm


def _signed_triangle_distance(
    x: np.ndarray,
    triangle: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    p, w, dist2, _ = closest_point_on_triangle(x, triangle[0], triangle[1], triangle[2])
    face_normal = _unit_triangle_normal(triangle)

    offset = x - p
    distance = float(np.sqrt(dist2))
    signed_plane_distance = float(np.dot(offset, face_normal))

    if distance <= 1e-15:
        return 0.0, face_normal, w, p, dist2

    sign = 1.0 if signed_plane_distance >= 0.0 else -1.0
    n = sign * offset / distance
    return sign * distance, n, w, p, dist2


def dynamic_surface_sdf(
    x: np.ndarray,
    x_current: np.ndarray,
    boundary_faces: np.ndarray,
    candidate_face_ids: np.ndarray,
) -> SurfaceSDFResult:
    """Evaluate oriented signed distance against supplied candidate triangles.

    This is the internal projection kernel used to populate dynamic SDF grids
    and to support explicitly requested refinement. Candidate face ids must be
    supplied by a broad phase; this function intentionally does not fall back to
    a global all-face search. The sign is controlled by the supplied surface
    orientation: outward-oriented closed surfaces are positive outside and
    negative inside.
    """

    point = _as_point(x, "x")
    X, faces = _as_surface(x_current, boundary_faces)
    candidates = np.asarray(candidate_face_ids, dtype=np.int64).ravel()
    if candidates.size == 0:
        raise ValueError("candidate_face_ids must contain at least one face id")
    if np.any(candidates < 0) or int(candidates.max()) >= faces.shape[0]:
        raise ValueError("candidate_face_ids contains an invalid face id")

    best: SurfaceSDFResult | None = None
    best_dist2 = np.inf
    for face_id in candidates:
        triangle = X[faces[int(face_id)]]
        g, n, w, p, dist2 = _signed_triangle_distance(point, triangle)
        if dist2 < best_dist2:
            best_dist2 = dist2
            best = SurfaceSDFResult(g=float(g), n=n, face_id=int(face_id), w=w, p=p)

    if best is None:
        raise RuntimeError("failed to evaluate any candidate face")
    return best


def query_dynamic_surface_sdf(
    x: np.ndarray,
    x_current: np.ndarray,
    boundary_faces: np.ndarray,
    candidate_face_ids: np.ndarray,
) -> SurfaceSDFResult:
    """Legacy alias for the current-surface projection kernel."""

    return dynamic_surface_sdf(x, x_current, boundary_faces, candidate_face_ids)


def surface_projection_distance_kernel(
    x: np.ndarray,
    x_current: np.ndarray,
    boundary_faces: np.ndarray,
    candidate_face_ids: np.ndarray,
) -> SurfaceSDFResult:
    """Internal projection kernel used during SDF field construction."""

    return dynamic_surface_sdf(x, x_current, boundary_faces, candidate_face_ids)


def _slow_reference_dynamic_surface_sdf(
    x: np.ndarray,
    x_current: np.ndarray,
    boundary_faces: np.ndarray,
) -> SurfaceSDFResult:
    """Slow test/reference query that searches every boundary face."""

    _, faces = _as_surface(x_current, boundary_faces)
    return dynamic_surface_sdf(
        x,
        x_current,
        boundary_faces,
        np.arange(faces.shape[0], dtype=np.int64),
    )
