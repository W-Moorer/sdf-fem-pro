"""Local closest-point projection primitives."""

from __future__ import annotations

from typing import Literal

import numpy as np

RegionType = Literal["face", "edge", "vertex"]


def _as_point(value: np.ndarray, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return point


def _triangle_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    normal = np.cross(b - a, c - a)
    norm = np.linalg.norm(normal)
    if norm <= 0.0:
        raise ValueError("triangle has zero area")
    return normal / norm


def _result(
    x: np.ndarray,
    p: np.ndarray,
    w: tuple[float, float, float],
    region: RegionType,
) -> tuple[np.ndarray, np.ndarray, float, RegionType]:
    weights = np.asarray(w, dtype=float)
    weights /= np.sum(weights)
    dist2 = float(np.dot(x - p, x - p))
    return p, weights, dist2, region


def closest_point_on_triangle(
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, RegionType]:
    """Project a point onto a triangle.

    Returns the closest point, barycentric coordinates with respect to
    ``(a, b, c)``, squared distance, and the closest feature type.
    """

    x = _as_point(x, "x")
    a = _as_point(a, "a")
    b = _as_point(b, "b")
    c = _as_point(c, "c")
    _triangle_normal(a, b, c)

    ab = b - a
    ac = c - a
    ap = x - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return _result(x, a.copy(), (1.0, 0.0, 0.0), "vertex")

    bp = x - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return _result(x, b.copy(), (0.0, 1.0, 0.0), "vertex")

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        p = a + v * ab
        return _result(x, p, (1.0 - v, v, 0.0), "edge")

    cp = x - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return _result(x, c.copy(), (0.0, 0.0, 1.0), "vertex")

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        p = a + w * ac
        return _result(x, p, (1.0 - w, 0.0, w), "edge")

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        p = b + w * (c - b)
        return _result(x, p, (0.0, 1.0 - w, w), "edge")

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    u = 1.0 - v - w
    p = u * a + v * b + w * c
    return _result(x, p, (u, v, w), "face")


def signed_point_triangle_gap(
    x: np.ndarray,
    triangle: np.ndarray,
    normal_sign_reference: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return signed point-to-triangle gap, unit normal, and barycentric weights.

    The triangle normal is oriented so that it has nonnegative dot product with
    ``normal_sign_reference``. The signed gap is ``dot(x - p, n)``.
    """

    tri = np.asarray(triangle, dtype=float)
    if tri.shape != (3, 3):
        raise ValueError("triangle must have shape (3, 3)")
    ref = _as_point(normal_sign_reference, "normal_sign_reference")
    ref_norm = np.linalg.norm(ref)
    if ref_norm <= 0.0:
        raise ValueError("normal_sign_reference must be nonzero")

    p, w, _, _ = closest_point_on_triangle(x, tri[0], tri[1], tri[2])
    n = _triangle_normal(tri[0], tri[1], tri[2])
    if np.dot(n, ref) < 0.0:
        n = -n

    x = _as_point(x, "x")
    g = float(np.dot(x - p, n))
    return g, n, w
