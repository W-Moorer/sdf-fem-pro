"""Optional C++ projection kernels for SDF field construction."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - availability depends on local extension build.
    from sfc import _sfc_cpp
except Exception as exc:  # pragma: no cover
    _BACKEND = None
    _IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover
    _BACKEND = _sfc_cpp
    _IMPORT_ERROR = None


def is_available() -> bool:
    """Return whether the optional C++ backend is importable."""

    return _BACKEND is not None


def _require_backend() -> None:
    if _BACKEND is None:
        raise RuntimeError("C++ projection backend is not available") from _IMPORT_ERROR


def closest_points_all_faces(
    points: np.ndarray,
    x_current: np.ndarray,
    boundary_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate exact all-face closest-feature projection with C++ loops."""

    _require_backend()
    return _BACKEND.closest_points_all_faces(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(x_current, dtype=np.float64),
        np.ascontiguousarray(boundary_faces, dtype=np.int64),
    )


def closest_points_padded_aabb(
    points: np.ndarray,
    x_current: np.ndarray,
    boundary_faces: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    fallback_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate exact-fallback padded-AABB projection with C++ loops."""

    _require_backend()
    return _BACKEND.closest_points_padded_aabb(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(x_current, dtype=np.float64),
        np.ascontiguousarray(boundary_faces, dtype=np.int64),
        np.ascontiguousarray(aabb_min, dtype=np.float64),
        np.ascontiguousarray(aabb_max, dtype=np.float64),
        float(fallback_distance),
    )
