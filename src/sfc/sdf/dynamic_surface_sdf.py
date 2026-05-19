"""Internal current-surface projection kernel for SDF grid construction.

The kernel operates on current boundary triangles. Those triangles may come
from TET4 faces, triangulated HEX8/C3D8 faces, or any other finite-element
boundary that has been converted to oriented triangles by the caller.

This module is retained for grid population, tests, validation references, and
explicit optional refinement. The main dynamic SDF query API is
``DynamicNarrowBandSDF``, whose query path interpolates stored field data.
"""

from __future__ import annotations

import os
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


class SurfaceSDFBatchResult(NamedTuple):
    """Batch result of current-surface projection-kernel evaluations."""

    g: np.ndarray
    n: np.ndarray
    face_id: np.ndarray
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


def _unit_triangle_normals(triangles: np.ndarray) -> np.ndarray:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    if bool(np.any(norms <= 0.0)):
        raise ValueError("boundary face has zero area")
    return normals / norms[:, None]


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


def surface_projection_distance_kernel_batch_all_faces(
    points: np.ndarray,
    x_current: np.ndarray,
    boundary_faces: np.ndarray,
) -> SurfaceSDFBatchResult:
    """Evaluate the projection kernel for many points against all faces.

    This is used only for SDF field population. It is algebraically the same
    closest-triangle projection as the scalar kernel, but amortizes Python
    overhead over a batch of grid nodes.
    """

    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    X, faces = _as_surface(x_current, boundary_faces)
    if faces.shape[0] == 0:
        raise ValueError("boundary_faces must contain at least one triangle")

    triangles = X[faces]
    closest, bary, dist2 = _closest_points_on_triangles_batch(P, triangles)
    best_face = np.argmin(dist2, axis=1)
    rows = np.arange(P.shape[0], dtype=np.int64)
    best_p = closest[rows, best_face]
    best_w = bary[rows, best_face]
    best_dist2 = dist2[rows, best_face]
    best_triangles = triangles[best_face]
    face_normals = _unit_triangle_normals(best_triangles)
    offset = P - best_p
    distance = np.sqrt(best_dist2)
    signed_plane_distance = np.einsum("ij,ij->i", offset, face_normals)
    sign = np.where(signed_plane_distance >= 0.0, 1.0, -1.0)
    normals = np.empty_like(offset)
    nonzero = distance > 1.0e-15
    normals[nonzero] = sign[nonzero, None] * offset[nonzero] / distance[nonzero, None]
    normals[~nonzero] = face_normals[~nonzero]
    g = sign * distance
    g[~nonzero] = 0.0
    return SurfaceSDFBatchResult(
        g=g.astype(float, copy=False),
        n=normals,
        face_id=best_face.astype(np.int64, copy=False),
        w=best_w,
        p=best_p,
    )


def surface_projection_distance_kernel_batch_all_faces_compiled(
    points: np.ndarray,
    x_current: np.ndarray,
    boundary_faces: np.ndarray,
) -> SurfaceSDFBatchResult:
    """Evaluate all-face projection with the optional compiled backend.

    The compiled path performs the same closest-feature search as
    ``surface_projection_distance_kernel_batch_all_faces`` but avoids the large
    ``(points, faces, 3)`` temporary arrays used by the NumPy vectorized path.
    It is used only during SDF field construction; field queries remain
    interpolation-only.
    """

    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    X, faces = _as_surface(x_current, boundary_faces)
    if faces.shape[0] == 0:
        raise ValueError("boundary_faces must contain at least one triangle")
    _unit_triangle_normals(X[faces])
    use_cpp = os.environ.get("SFC_USE_CPP_PROJECTION", "0").strip().lower() in {"1", "true", "yes"}
    if use_cpp:
        try:
            from ._cpp_projection import closest_points_all_faces, is_available as cpp_projection_available
        except Exception:
            cpp_projection_available = lambda: False  # type: ignore[assignment]
        if not cpp_projection_available():
            from ._numba_projection import closest_points_all_faces
    else:
        from ._numba_projection import closest_points_all_faces

    g, normals, face_id, bary, closest = closest_points_all_faces(P, X, faces)
    return SurfaceSDFBatchResult(
        g=g.astype(float, copy=False),
        n=normals,
        face_id=face_id.astype(np.int64, copy=False),
        w=bary,
        p=closest,
    )


def surface_projection_distance_kernel_batch_padded_aabb_compiled(
    points: np.ndarray,
    x_current: np.ndarray,
    boundary_faces: np.ndarray,
    *,
    delta_safe: float,
    fallback_distance: float | None = None,
) -> SurfaceSDFBatchResult:
    """Evaluate projection with compiled padded-AABB candidate screening."""

    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    X, faces = _as_surface(x_current, boundary_faces)
    if faces.shape[0] == 0:
        raise ValueError("boundary_faces must contain at least one triangle")
    delta = float(delta_safe)
    if delta < 0.0:
        raise ValueError("delta_safe must be non-negative")
    triangles = X[faces]
    _unit_triangle_normals(triangles)
    aabb_min = triangles.min(axis=1) - delta
    aabb_max = triangles.max(axis=1) + delta
    use_cpp = os.environ.get("SFC_USE_CPP_PROJECTION", "0").strip().lower() in {"1", "true", "yes"}
    if use_cpp:
        try:
            from ._cpp_projection import closest_points_padded_aabb, is_available as cpp_projection_available
        except Exception:
            cpp_projection_available = lambda: False  # type: ignore[assignment]
        if not cpp_projection_available():
            from ._numba_projection import closest_points_padded_aabb
    else:
        from ._numba_projection import closest_points_padded_aabb

    g, normals, face_id, bary, closest = closest_points_padded_aabb(
        P,
        X,
        faces,
        aabb_min,
        aabb_max,
        delta if fallback_distance is None else float(fallback_distance),
    )
    return SurfaceSDFBatchResult(
        g=g.astype(float, copy=False),
        n=normals,
        face_id=face_id.astype(np.int64, copy=False),
        w=bary,
        p=closest,
    )


def compiled_projection_available() -> bool:
    """Return whether the optional compiled projection backend is available."""

    try:
        from ._cpp_projection import is_available as cpp_available
    except Exception:
        cpp = False
    else:
        cpp = bool(cpp_available())
    if cpp:
        return True
    try:
        from ._numba_projection import is_available
    except Exception:
        return False
    return bool(is_available())


def surface_projection_distance_kernel_batch_candidates(
    points: np.ndarray,
    x_current: np.ndarray,
    boundary_faces: np.ndarray,
    candidate_face_ids: np.ndarray,
) -> SurfaceSDFBatchResult:
    """Evaluate many projection-kernel points against a shared candidate set.

    This is the batched counterpart of ``surface_projection_distance_kernel``
    for points whose spatial-hash lookup produced the same candidate triangle
    ids. Projection remains an internal field-construction operation.
    """

    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    X, faces = _as_surface(x_current, boundary_faces)
    candidates = np.asarray(candidate_face_ids, dtype=np.int64).ravel()
    if candidates.size == 0:
        raise ValueError("candidate_face_ids must contain at least one face id")
    if np.any(candidates < 0) or int(candidates.max()) >= faces.shape[0]:
        raise ValueError("candidate_face_ids contains an invalid face id")

    triangles = X[faces[candidates]]
    closest, bary, dist2 = _closest_points_on_triangles_batch(P, triangles)
    best_local = np.argmin(dist2, axis=1)
    rows = np.arange(P.shape[0], dtype=np.int64)
    best_face = candidates[best_local]
    best_p = closest[rows, best_local]
    best_w = bary[rows, best_local]
    best_dist2 = dist2[rows, best_local]
    best_triangles = X[faces[best_face]]
    face_normals = _unit_triangle_normals(best_triangles)
    offset = P - best_p
    distance = np.sqrt(best_dist2)
    signed_plane_distance = np.einsum("ij,ij->i", offset, face_normals)
    sign = np.where(signed_plane_distance >= 0.0, 1.0, -1.0)
    normals = np.empty_like(offset)
    nonzero = distance > 1.0e-15
    normals[nonzero] = sign[nonzero, None] * offset[nonzero] / distance[nonzero, None]
    normals[~nonzero] = face_normals[~nonzero]
    g = sign * distance
    g[~nonzero] = 0.0
    return SurfaceSDFBatchResult(
        g=g.astype(float, copy=False),
        n=normals,
        face_id=best_face.astype(np.int64, copy=False),
        w=best_w,
        p=best_p,
    )


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


def _closest_points_on_triangles_batch(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    P = points[:, None, :]
    A = triangles[None, :, 0, :]
    B = triangles[None, :, 1, :]
    C = triangles[None, :, 2, :]
    AB = B - A
    AC = C - A
    AP = P - A

    d1 = np.einsum("nmd,nmd->nm", AB, AP)
    d2 = np.einsum("nmd,nmd->nm", AC, AP)
    BP = P - B
    d3 = np.einsum("nmd,nmd->nm", AB, BP)
    d4 = np.einsum("nmd,nmd->nm", AC, BP)
    CP = P - C
    d5 = np.einsum("nmd,nmd->nm", AB, CP)
    d6 = np.einsum("nmd,nmd->nm", AC, CP)

    shape = d1.shape
    closest = np.empty((*shape, 3), dtype=float)
    bary = np.empty((*shape, 3), dtype=float)
    assigned = np.zeros(shape, dtype=bool)

    def assign(mask: np.ndarray, p: np.ndarray, w: np.ndarray) -> None:
        active = mask & ~assigned
        if not bool(np.any(active)):
            return
        closest[active] = np.broadcast_to(p, closest.shape)[active]
        bary[active] = np.broadcast_to(w, bary.shape)[active]
        assigned[active] = True

    assign((d1 <= 0.0) & (d2 <= 0.0), A, np.asarray([1.0, 0.0, 0.0], dtype=float))
    assign((d3 >= 0.0) & (d4 <= d3), B, np.asarray([0.0, 1.0, 0.0], dtype=float))

    vc = d1 * d4 - d3 * d2
    edge_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    v = _safe_divide(d1, d1 - d3)
    assign(edge_ab, A + v[:, :, None] * AB, np.stack((1.0 - v, v, np.zeros_like(v)), axis=-1))

    assign((d6 >= 0.0) & (d5 <= d6), C, np.asarray([0.0, 0.0, 1.0], dtype=float))

    vb = d5 * d2 - d1 * d6
    edge_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    w_ac = _safe_divide(d2, d2 - d6)
    assign(edge_ac, A + w_ac[:, :, None] * AC, np.stack((1.0 - w_ac, np.zeros_like(w_ac), w_ac), axis=-1))

    va = d3 * d6 - d5 * d4
    edge_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    w_bc = _safe_divide(d4 - d3, (d4 - d3) + (d5 - d6))
    assign(
        edge_bc,
        B + w_bc[:, :, None] * (C - B),
        np.stack((np.zeros_like(w_bc), 1.0 - w_bc, w_bc), axis=-1),
    )

    remaining = ~assigned
    denom = va + vb + vc
    v_face = _safe_divide(vb, denom)
    w_face = _safe_divide(vc, denom)
    u_face = 1.0 - v_face - w_face
    assign(
        remaining,
        u_face[:, :, None] * A + v_face[:, :, None] * B + w_face[:, :, None] * C,
        np.stack((u_face, v_face, w_face), axis=-1),
    )

    diff = P - closest
    dist2 = np.einsum("nmd,nmd->nm", diff, diff)
    return closest, bary, dist2


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=np.abs(denominator) > 1.0e-30,
    )
