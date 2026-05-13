"""Uniform spatial hash broad phase for triangle AABBs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


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


def triangle_aabbs(
    x_current: np.ndarray,
    boundary_faces: np.ndarray,
    delta_safe: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return padded triangle AABB lower and upper corners."""

    X, faces = _as_surface(x_current, boundary_faces)
    delta = float(delta_safe)
    if delta < 0.0:
        raise ValueError("delta_safe must be non-negative")
    if faces.shape[0] == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)

    triangles = X[faces]
    mins = triangles.min(axis=1) - delta
    maxs = triangles.max(axis=1) + delta
    return mins, maxs


@dataclass(slots=True)
class UniformTriangleAABBHash:
    """Uniform spatial hash containing padded triangle AABBs."""

    aabb_min: np.ndarray
    aabb_max: np.ndarray
    cell_size: float
    origin: np.ndarray
    cells: dict[tuple[int, int, int], list[int]]

    @classmethod
    def from_surface(
        cls,
        x_current: np.ndarray,
        boundary_faces: np.ndarray,
        *,
        delta_safe: float = 0.0,
        cell_size: float | None = None,
    ) -> "UniformTriangleAABBHash":
        """Build a spatial hash from boundary triangle AABBs."""

        aabb_min, aabb_max = triangle_aabbs(x_current, boundary_faces, delta_safe)
        if cell_size is None:
            if aabb_min.shape[0] == 0:
                h = 1.0
            else:
                extents = np.maximum(aabb_max - aabb_min, 0.0)
                positive = extents[extents > 0.0]
                h = float(np.max(positive)) if positive.size else 1.0
        else:
            h = float(cell_size)
        if h <= 0.0:
            raise ValueError("cell_size must be positive")

        if aabb_min.shape[0] == 0:
            origin = np.zeros(3, dtype=float)
        else:
            origin = aabb_min.min(axis=0)

        cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        index = cls(
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            cell_size=h,
            origin=origin,
            cells=cells,
        )
        for tri_id in range(aabb_min.shape[0]):
            for key in index._cell_keys_for_aabb(aabb_min[tri_id], aabb_max[tri_id]):
                cells[key].append(tri_id)
        index.cells = dict(cells)
        return index

    def _cell_index(self, point: np.ndarray) -> tuple[int, int, int]:
        ijk = np.floor((point - self.origin) / self.cell_size).astype(np.int64)
        return int(ijk[0]), int(ijk[1]), int(ijk[2])

    def _cell_keys_for_aabb(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> Iterable[tuple[int, int, int]]:
        lo = np.floor((lower - self.origin) / self.cell_size).astype(np.int64)
        hi = np.floor((upper - self.origin) / self.cell_size).astype(np.int64)
        for i in range(int(lo[0]), int(hi[0]) + 1):
            for j in range(int(lo[1]), int(hi[1]) + 1):
                for k in range(int(lo[2]), int(hi[2]) + 1):
                    yield i, j, k

    def query_point(self, point: np.ndarray) -> np.ndarray:
        """Return triangle ids whose padded AABB contains ``point``."""

        x = np.asarray(point, dtype=float)
        if x.shape != (3,):
            raise ValueError("point must have shape (3,)")

        ids = self.cells.get(self._cell_index(x), [])
        if not ids:
            return np.empty(0, dtype=np.int64)

        unique = np.array(sorted(set(ids)), dtype=np.int64)
        contains = np.all((self.aabb_min[unique] <= x) & (x <= self.aabb_max[unique]), axis=1)
        return unique[contains]

    def query_points(self, points: np.ndarray) -> list[np.ndarray]:
        """Return candidate triangle ids for each query point."""

        P = np.asarray(points, dtype=float)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError("points must have shape (q, 3)")
        return [self.query_point(point) for point in P]
