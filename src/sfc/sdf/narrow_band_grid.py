"""Narrow-band signed-distance grid storage and interpolation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


_CORNERS = np.asarray(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.int64,
)


@dataclass(frozen=True, slots=True)
class GridInterpolation:
    """Trilinear interpolation stencil for one query point."""

    base_index: np.ndarray
    corner_offsets: np.ndarray
    corner_indices: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True, slots=True)
class GridPayloadInterpolation:
    """Interpolated grid-node payload used by field-contact Jacobians."""

    weights: np.ndarray
    face_ids: np.ndarray
    barycentric: np.ndarray
    normals: np.ndarray


@dataclass(frozen=True, slots=True)
class NarrowBandGrid:
    """Regular current-space SDF grid with narrow-band payload data."""

    origin: np.ndarray
    spacing: np.ndarray
    phi: np.ndarray
    gradient: np.ndarray
    closest_face_id: np.ndarray
    barycentric: np.ndarray
    closest_normal: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        origin = _as_vector3(self.origin, "origin")
        spacing = _as_spacing(self.spacing)
        phi = np.asarray(self.phi, dtype=float)
        if phi.ndim != 3:
            raise ValueError("phi must have shape (nx, ny, nz)")
        if any(size < 2 for size in phi.shape):
            raise ValueError("each grid axis must contain at least two nodes")

        gradient = np.asarray(self.gradient, dtype=float)
        if gradient.shape != (*phi.shape, 3):
            raise ValueError("gradient must have shape (*phi.shape, 3)")
        closest_face_id = np.asarray(self.closest_face_id, dtype=np.int64)
        if closest_face_id.shape != phi.shape:
            raise ValueError("closest_face_id must have the same scalar grid shape as phi")
        barycentric = np.asarray(self.barycentric, dtype=float)
        if barycentric.shape != (*phi.shape, 3):
            raise ValueError("barycentric must have shape (*phi.shape, 3)")
        closest_normal = np.asarray(self.closest_normal, dtype=float)
        if closest_normal.shape != (*phi.shape, 3):
            raise ValueError("closest_normal must have shape (*phi.shape, 3)")
        valid_mask = np.asarray(self.valid_mask, dtype=bool)
        if valid_mask.shape != phi.shape:
            raise ValueError("valid_mask must have the same scalar grid shape as phi")

        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "closest_face_id", closest_face_id)
        object.__setattr__(self, "barycentric", barycentric)
        object.__setattr__(self, "closest_normal", closest_normal)
        object.__setattr__(self, "valid_mask", valid_mask)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the grid node counts along x, y, and z."""

        return tuple(int(v) for v in self.phi.shape)

    @property
    def bounds_min(self) -> np.ndarray:
        """Return the lower current-space grid corner."""

        return self.origin.copy()

    @property
    def bounds_max(self) -> np.ndarray:
        """Return the upper current-space grid corner."""

        return self.origin + self.spacing * (np.asarray(self.shape, dtype=float) - 1.0)

    def coordinates(self) -> np.ndarray:
        """Return all grid-node coordinates with shape ``(*shape, 3)``."""

        axes = [
            self.origin[axis] + self.spacing[axis] * np.arange(self.shape[axis], dtype=float)
            for axis in range(3)
        ]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        return np.stack((X, Y, Z), axis=-1)

    def contains(self, point: np.ndarray, *, tol: float = 1.0e-12) -> bool:
        """Return true when ``point`` lies inside the grid bounds."""

        x = _as_vector3(point, "point")
        return bool(np.all(x >= self.bounds_min - tol) and np.all(x <= self.bounds_max + tol))

    def interpolation(self, point: np.ndarray, *, require_valid: bool = True) -> GridInterpolation:
        """Return the trilinear interpolation stencil for ``point``."""

        x = _as_vector3(point, "point")
        if not self.contains(x):
            raise ValueError("query point is outside the narrow-band grid")

        u = (x - self.origin) / self.spacing
        shape = np.asarray(self.shape, dtype=np.int64)
        base = np.floor(u).astype(np.int64)
        frac = u - base.astype(float)
        for axis in range(3):
            if base[axis] >= shape[axis] - 1:
                base[axis] = shape[axis] - 2
                frac[axis] = 1.0
            elif base[axis] < 0:
                base[axis] = 0
                frac[axis] = 0.0

        corner_indices = base[None, :] + _CORNERS
        if require_valid:
            valid = self.valid_mask[
                corner_indices[:, 0],
                corner_indices[:, 1],
                corner_indices[:, 2],
            ]
            if not bool(np.all(valid)):
                raise ValueError("query point is outside the valid narrow band")

        weights = np.ones(8, dtype=float)
        for axis in range(3):
            weights *= np.where(_CORNERS[:, axis] == 0, 1.0 - frac[axis], frac[axis])
        return GridInterpolation(
            base_index=base,
            corner_offsets=_CORNERS.copy(),
            corner_indices=corner_indices,
            weights=weights,
        )

    def interpolate_scalar(
        self,
        values: np.ndarray,
        point: np.ndarray,
        *,
        require_valid: bool = True,
    ) -> float:
        """Trilinearly interpolate a scalar grid at ``point``."""

        arr = np.asarray(values, dtype=float)
        if arr.shape != self.phi.shape:
            raise ValueError("values must have the same scalar grid shape as phi")
        stencil = self.interpolation(point, require_valid=require_valid)
        vals = arr[
            stencil.corner_indices[:, 0],
            stencil.corner_indices[:, 1],
            stencil.corner_indices[:, 2],
        ]
        return float(stencil.weights @ vals)

    def interpolate_scalar_spatial_derivative(
        self,
        values: np.ndarray,
        point: np.ndarray,
        *,
        require_valid: bool = True,
    ) -> np.ndarray:
        """Return the spatial derivative of trilinear scalar interpolation.

        For ``f(x)=sum_l w_l(x) values_l``, this returns
        ``sum_l values_l grad_x w_l(x)``. This is the variationally consistent
        derivative of the interpolated scalar field, not an interpolation of a
        separately stored gradient grid.
        """

        arr = np.asarray(values, dtype=float)
        if arr.shape != self.phi.shape:
            raise ValueError("values must have the same scalar grid shape as phi")
        stencil = self.interpolation(point, require_valid=require_valid)
        vals = arr[
            stencil.corner_indices[:, 0],
            stencil.corner_indices[:, 1],
            stencil.corner_indices[:, 2],
        ]
        return vals @ self.interpolation_weight_gradients(point, stencil=stencil)

    def interpolation_weight_gradients(
        self,
        point: np.ndarray,
        *,
        stencil: GridInterpolation | None = None,
        require_valid: bool = True,
    ) -> np.ndarray:
        """Return ``grad_x w_l`` for the trilinear interpolation stencil."""

        x = _as_vector3(point, "point")
        interp = self.interpolation(x, require_valid=require_valid) if stencil is None else stencil
        u = (x - self.origin) / self.spacing
        frac = np.clip(u - interp.base_index.astype(float), 0.0, 1.0)

        gradients = np.empty((8, 3), dtype=float)
        for corner_id, corner in enumerate(interp.corner_offsets):
            for axis in range(3):
                value = 1.0 / self.spacing[axis]
                if corner[axis] == 0:
                    value *= -1.0
                for other_axis in range(3):
                    if other_axis == axis:
                        continue
                    value *= frac[other_axis] if corner[other_axis] == 1 else 1.0 - frac[other_axis]
                gradients[corner_id, axis] = value
        return gradients

    def interpolate_vector(
        self,
        values: np.ndarray,
        point: np.ndarray,
        *,
        require_valid: bool = True,
    ) -> np.ndarray:
        """Trilinearly interpolate a vector grid at ``point``."""

        arr = np.asarray(values, dtype=float)
        if arr.shape != (*self.phi.shape, 3):
            raise ValueError("values must have shape (*phi.shape, 3)")
        stencil = self.interpolation(point, require_valid=require_valid)
        vals = arr[
            stencil.corner_indices[:, 0],
            stencil.corner_indices[:, 1],
            stencil.corner_indices[:, 2],
        ]
        return stencil.weights @ vals

    def query_phi(self, point: np.ndarray) -> float:
        """Interpolate the signed-distance value at ``point``."""

        return self.interpolate_scalar(self.phi, point)

    def query_spatial_derivative_phi(self, point: np.ndarray) -> np.ndarray:
        """Return the derivative of the interpolated scalar ``phi`` field."""

        return self.interpolate_scalar_spatial_derivative(self.phi, point)

    def query_gradient(self, point: np.ndarray) -> np.ndarray:
        """Interpolate the stored SDF gradient at ``point``."""

        return self.interpolate_vector(self.gradient, point)

    def payload(self, point: np.ndarray) -> GridPayloadInterpolation:
        """Return the interpolation weights and closest-surface payload."""

        stencil = self.interpolation(point, require_valid=True)
        idx = stencil.corner_indices
        return GridPayloadInterpolation(
            weights=stencil.weights.copy(),
            face_ids=self.closest_face_id[idx[:, 0], idx[:, 1], idx[:, 2]].copy(),
            barycentric=self.barycentric[idx[:, 0], idx[:, 1], idx[:, 2]].copy(),
            normals=self.closest_normal[idx[:, 0], idx[:, 1], idx[:, 2]].copy(),
        )


def compute_finite_difference_gradient(
    phi: np.ndarray,
    spacing: np.ndarray | float,
    *,
    fallback_normals: np.ndarray | None = None,
) -> np.ndarray:
    """Return finite-difference gradients for a scalar regular grid."""

    phi_arr = np.asarray(phi, dtype=float)
    if phi_arr.ndim != 3:
        raise ValueError("phi must have shape (nx, ny, nz)")
    h = _as_spacing(spacing)
    work = phi_arr.copy()
    finite = np.isfinite(work)
    if not bool(np.all(finite)):
        work[~finite] = 0.0
    edge_order = 2 if all(size > 2 for size in phi_arr.shape) else 1
    grads = np.gradient(work, h[0], h[1], h[2], edge_order=edge_order)
    gradient = np.stack(grads, axis=-1)
    if fallback_normals is not None:
        fallback = np.asarray(fallback_normals, dtype=float)
        if fallback.shape != (*phi_arr.shape, 3):
            raise ValueError("fallback_normals must have shape (*phi.shape, 3)")
        bad = ~np.all(np.isfinite(gradient), axis=-1)
        bad |= np.linalg.norm(gradient, axis=-1) <= 1.0e-14
        gradient[bad] = fallback[bad]
    return gradient


def grid_spec_from_surface(
    x_current: np.ndarray,
    boundary_faces: np.ndarray,
    *,
    spacing: np.ndarray | float,
    padding: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return origin and shape for a surface AABB padded in current space."""

    X = np.asarray(x_current, dtype=float)
    faces = np.asarray(boundary_faces, dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("x_current must have shape (n, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("boundary_faces must have shape (m, 3)")
    if faces.size == 0:
        raise ValueError("boundary_faces must contain at least one triangle")
    if np.any(faces < 0) or int(faces.max()) >= X.shape[0]:
        raise ValueError("boundary_faces reference nodes outside x_current")
    h = _as_spacing(spacing)
    pad = float(padding)
    if pad < 0.0:
        raise ValueError("padding must be non-negative")

    surface = X[faces]
    lower = surface.min(axis=(0, 1)) - pad
    upper = surface.max(axis=(0, 1)) + pad
    shape = np.ceil((upper - lower) / h).astype(np.int64) + 1
    shape = np.maximum(shape, 2)
    return lower, tuple(int(v) for v in shape)


def _as_vector3(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return arr


def _as_spacing(value: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        arr = np.full(3, float(arr), dtype=float)
    if arr.shape != (3,):
        raise ValueError("spacing must be a positive scalar or have shape (3,)")
    if np.any(arr <= 0.0):
        raise ValueError("spacing entries must be positive")
    return arr
