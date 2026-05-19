"""Reference-space SDF data for deformation-aware contact queries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .narrow_band_grid import NarrowBandGrid, compute_finite_difference_gradient
from .local_projection import closest_point_on_triangle


PhiFunction = Callable[[np.ndarray], float]
GradPhiFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class MaterialSDF:
    """Reference SDF plus zero-level surface patches.

    The reference SDF ``phi0(X)`` defines the material surface
    ``Gamma0 = {X | phi0(X)=0}``.  The current surface is obtained from the FEM
    deformation map by moving the stored reference patch nodes to their current
    positions.  The fallback ``phi0`` implementation uses the oriented
    triangulated reference surface, which is sufficient for validation and for
    binding patches to FEM surface faces.
    """

    reference_nodes: np.ndarray
    boundary_faces: np.ndarray
    phi: PhiFunction | None = None
    gradient: GradPhiFunction | None = None
    grid: "MaterialSDFGrid | None" = None
    band_radius: float = 0.0

    def __post_init__(self) -> None:
        X = _as_nodes(self.reference_nodes, "reference_nodes")
        faces = _as_faces(self.boundary_faces, X.shape[0])
        band = float(self.band_radius)
        if band < 0.0:
            raise ValueError("band_radius must be non-negative")
        object.__setattr__(self, "reference_nodes", X)
        object.__setattr__(self, "boundary_faces", faces)
        object.__setattr__(self, "band_radius", band)

    @classmethod
    def from_triangle_surface(
        cls,
        reference_nodes: np.ndarray,
        boundary_faces: np.ndarray,
        *,
        phi: PhiFunction | None = None,
        gradient: GradPhiFunction | None = None,
        grid: "MaterialSDFGrid | None" = None,
        band_radius: float = 0.0,
    ) -> "MaterialSDF":
        """Create a material SDF from reference surface triangles."""

        return cls(
            reference_nodes=reference_nodes,
            boundary_faces=boundary_faces,
            phi=phi,
            gradient=gradient,
            grid=grid,
            band_radius=band_radius,
        )

    @classmethod
    def from_grid(
        cls,
        reference_nodes: np.ndarray,
        boundary_faces: np.ndarray,
        grid: "MaterialSDFGrid",
        *,
        band_radius: float = 0.0,
    ) -> "MaterialSDF":
        """Create a material SDF with trilinear reference-grid queries."""

        return cls(
            reference_nodes=reference_nodes,
            boundary_faces=boundary_faces,
            grid=grid,
            band_radius=band_radius,
        )

    @property
    def face_count(self) -> int:
        """Number of reference zero-level patches."""

        return int(self.boundary_faces.shape[0])

    @property
    def reference_triangles(self) -> np.ndarray:
        """Reference patch vertices with shape ``(n_faces, 3, 3)``."""

        return self.reference_nodes[self.boundary_faces]

    @property
    def reference_normals(self) -> np.ndarray:
        """Unit normals induced by reference face orientation."""

        return _unit_normals(self.reference_triangles)

    def query_phi(self, point: np.ndarray) -> float:
        """Evaluate ``phi0(X)``."""

        X = _as_point(point, "point")
        if self.phi is not None:
            return float(self.phi(X))
        if self.grid is not None:
            return self.grid.query_phi(X)
        result = self.closest_reference_patch(X)
        return float(result.gap)

    def query_gradient(self, point: np.ndarray) -> np.ndarray:
        """Evaluate ``grad_X phi0(X)``."""

        grad = self.query_raw_gradient(point)
        norm = float(np.linalg.norm(grad))
        if norm <= 0.0:
            raise ValueError("gradient(point) must be nonzero")
        return grad / norm

    def query_raw_gradient(self, point: np.ndarray) -> np.ndarray:
        """Evaluate the unnormalized material-space SDF derivative."""

        X = _as_point(point, "point")
        if self.gradient is not None:
            grad = np.asarray(self.gradient(X), dtype=float)
            if grad.shape != (3,):
                raise ValueError("gradient(point) must return shape (3,)")
            return grad
        if self.grid is not None:
            return self.grid.query_raw_gradient(X)
        result = self.closest_reference_patch(X)
        return result.normal.copy()

    def closest_reference_patch(self, point: np.ndarray) -> "MaterialPatchProjection":
        """Project a material point to the reference zero-level patches."""

        X = _as_point(point, "point")
        triangles = self.reference_triangles
        normals = self.reference_normals
        best: MaterialPatchProjection | None = None
        best_dist2 = np.inf
        for face_id, tri in enumerate(triangles):
            closest, bary, dist2, _region = closest_point_on_triangle(X, tri[0], tri[1], tri[2])
            if dist2 < best_dist2:
                normal = normals[face_id]
                gap = float(np.dot(X - closest, normal))
                best = MaterialPatchProjection(
                    gap=gap,
                    normal=normal.copy(),
                    face_id=int(face_id),
                    barycentric=bary,
                    closest_point=closest,
                    squared_distance=float(dist2),
                )
                best_dist2 = float(dist2)
        if best is None:
            raise RuntimeError("material SDF has no reference patches")
        return best

    def build_patch_bvh(
        self,
        x_current: np.ndarray,
        *,
        padding: float | None = None,
        cell_size: float | None = None,
    ) -> "ReferencePatchBVH":
        """Build/refit a current-space AABB index over reference patches."""

        return ReferencePatchBVH.from_material(self, x_current, padding=padding, cell_size=cell_size)


@dataclass(frozen=True, slots=True)
class MaterialPatchProjection:
    """Reference patch projection result."""

    gap: float
    normal: np.ndarray
    face_id: int
    barycentric: np.ndarray
    closest_point: np.ndarray
    squared_distance: float


@dataclass(frozen=True, slots=True)
class MaterialSDFGrid:
    """Regular reference-space SDF grid.

    ``interpolation_order`` may be ``"linear"`` for trilinear interpolation or
    ``"cubic"`` for tensor-product cubic Lagrange interpolation on the reference
    grid.  The cubic path is intended for high-order material SDF queries and
    does not create a current-space SDF grid.
    """

    origin: np.ndarray
    spacing: np.ndarray
    phi: np.ndarray
    gradient: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    interpolation_order: str = "linear"

    def __post_init__(self) -> None:
        origin = _as_point(self.origin, "origin")
        spacing = _as_spacing(self.spacing)
        phi = np.asarray(self.phi, dtype=float)
        if phi.ndim != 3:
            raise ValueError("phi must have shape (nx, ny, nz)")
        if any(size < 2 for size in phi.shape):
            raise ValueError("each grid axis must contain at least two nodes")
        valid = np.isfinite(phi) if self.valid_mask is None else np.asarray(self.valid_mask, dtype=bool)
        if valid.shape != phi.shape:
            raise ValueError("valid_mask must match phi shape")
        if self.gradient is None:
            gradient = compute_finite_difference_gradient(phi, spacing)
        else:
            gradient = np.asarray(self.gradient, dtype=float)
        if gradient.shape != (*phi.shape, 3):
            raise ValueError("gradient must have shape (*phi.shape, 3)")
        order = str(self.interpolation_order).lower()
        if order not in {"linear", "cubic"}:
            raise ValueError("interpolation_order must be 'linear' or 'cubic'")
        if order == "cubic" and any(size < 4 for size in phi.shape):
            raise ValueError("cubic material SDF interpolation requires at least four nodes per axis")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "interpolation_order", order)

    @classmethod
    def from_phi_function(
        cls,
        *,
        origin: np.ndarray,
        spacing: np.ndarray | float,
        shape: tuple[int, int, int],
        phi: PhiFunction,
        gradient: GradPhiFunction | None = None,
        interpolation_order: str = "linear",
    ) -> "MaterialSDFGrid":
        """Sample a reference-grid SDF from callables."""

        h = _as_spacing(spacing)
        grid_shape = tuple(int(v) for v in shape)
        values = np.empty(grid_shape, dtype=float)
        gradients = np.empty((*grid_shape, 3), dtype=float) if gradient is not None else None
        base = _as_point(origin, "origin")
        for index in np.ndindex(grid_shape):
            point = base + h * np.asarray(index, dtype=float)
            values[index] = float(phi(point))
            if gradient is not None:
                gradients[index] = np.asarray(gradient(point), dtype=float)
        return cls(origin=base, spacing=h, phi=values, gradient=gradients, interpolation_order=interpolation_order)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return reference-grid dimensions."""

        return tuple(int(v) for v in self.phi.shape)

    def _as_narrow_grid(self) -> NarrowBandGrid:
        zeros_vec = np.zeros((*self.phi.shape, 3), dtype=float)
        zeros_int = np.zeros(self.phi.shape, dtype=np.int64)
        return NarrowBandGrid(
            origin=self.origin,
            spacing=self.spacing,
            phi=self.phi,
            gradient=np.asarray(self.gradient, dtype=float),
            closest_face_id=zeros_int,
            barycentric=zeros_vec,
            closest_normal=np.asarray(self.gradient, dtype=float),
            valid_mask=np.asarray(self.valid_mask, dtype=bool),
        )

    def query_phi(self, point: np.ndarray) -> float:
        """Interpolate ``phi0(X)`` in reference space."""

        if self.interpolation_order == "cubic":
            value, _grad = _tricubic_value_and_derivative(self.phi, self.origin, self.spacing, point)
            return float(value)
        return self._as_narrow_grid().query_phi(point)

    def query_gradient(self, point: np.ndarray) -> np.ndarray:
        """Return the scalar-interpolation derivative of ``phi0``."""

        grad = self.query_raw_gradient(point)
        norm = float(np.linalg.norm(grad))
        if norm <= 0.0:
            raise ValueError("material SDF gradient is zero at query point")
        return grad / norm

    def query_raw_gradient(self, point: np.ndarray) -> np.ndarray:
        """Return the unnormalized scalar-interpolation derivative."""

        if self.interpolation_order == "cubic":
            _value, grad = _tricubic_value_and_derivative(self.phi, self.origin, self.spacing, point)
        else:
            grad = self._as_narrow_grid().query_spatial_derivative_phi(point)
        return np.asarray(grad, dtype=float)


@dataclass(frozen=True, slots=True)
class ReferencePatchBVH:
    """Refit current-space AABB index for material surface patches.

    This class intentionally stores patch topology in reference space and only
    refits current AABBs when the FEM nodal positions change.  The query path
    may use cached patch ids as ordering hints, but candidate enumeration keeps
    an exact fallback so the cache cannot change the returned closest patch.
    """

    material: MaterialSDF
    x_current: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    cell_aabb_min: np.ndarray
    cell_aabb_max: np.ndarray
    padding: float
    cell_size: float
    cells: dict[tuple[int, int, int], tuple[int, ...]]
    origin: np.ndarray

    @classmethod
    def from_material(
        cls,
        material: MaterialSDF,
        x_current: np.ndarray,
        *,
        padding: float | None = None,
        cell_size: float | None = None,
    ) -> "ReferencePatchBVH":
        """Refit current AABBs from the same reference patch topology."""

        X = _as_nodes(x_current, "x_current")
        if X.shape != material.reference_nodes.shape:
            raise ValueError("x_current must match material.reference_nodes shape")
        pad = material.band_radius if padding is None else float(padding)
        if pad < 0.0:
            raise ValueError("padding must be non-negative")
        triangles = X[material.boundary_faces]
        tight_mins = triangles.min(axis=1)
        tight_maxs = triangles.max(axis=1)
        cell_mins = tight_mins - pad
        cell_maxs = tight_maxs + pad
        if tight_mins.shape[0] == 0:
            origin = np.zeros(3, dtype=float)
            h = 1.0
        else:
            origin = cell_mins.min(axis=0)
            if cell_size is None:
                extents = np.maximum(cell_maxs - cell_mins, 0.0)
                positive = extents[extents > 0.0]
                h = float(np.max(positive)) if positive.size else 1.0
            else:
                h = float(cell_size)
        if h <= 0.0:
            raise ValueError("cell_size must be positive")
        cells: dict[tuple[int, int, int], list[int]] = {}
        for patch_id in range(cell_mins.shape[0]):
            for key in _cell_keys_for_aabb(cell_mins[patch_id], cell_maxs[patch_id], origin, h):
                cells.setdefault(key, []).append(int(patch_id))
        return cls(
            material=material,
            x_current=X,
            aabb_min=tight_mins,
            aabb_max=tight_maxs,
            cell_aabb_min=cell_mins,
            cell_aabb_max=cell_maxs,
            padding=pad,
            cell_size=h,
            cells={key: tuple(values) for key, values in cells.items()},
            origin=origin,
        )

    def refit(
        self,
        x_current: np.ndarray,
        *,
        padding: float | None = None,
        cell_size: float | None = None,
    ) -> "ReferencePatchBVH":
        """Return a refit index with unchanged reference patch topology."""

        return type(self).from_material(
            self.material,
            x_current,
            padding=self.padding if padding is None else padding,
            cell_size=self.cell_size if cell_size is None else cell_size,
        )

    def candidates(
        self,
        point: np.ndarray,
        *,
        search_radius: float | None = None,
        cached_face_id: int | None = None,
    ) -> np.ndarray:
        """Return candidate patch ids ordered by AABB distance.

        ``cached_face_id`` is only an ordering hint.  It is prepended when
        valid, but all radius-compatible candidates remain present and an exact
        nearest-AABB fallback is used when the radius query is empty.
        """

        x = _as_point(point, "point")
        dist2 = self.aabb_distance_squared(x)
        if search_radius is None:
            ids = self._hash_candidates_for_point(x)
            if ids.size == 0:
                ids = np.arange(self.material.face_count, dtype=np.int64)
        else:
            radius = float(search_radius)
            if radius < 0.0:
                raise ValueError("search_radius must be non-negative")
            ids = self._hash_candidates_for_ball(x, radius)
            if ids.size:
                ids = ids[dist2[ids] <= radius * radius]
            else:
                ids = np.flatnonzero(dist2 <= radius * radius).astype(np.int64)
            if ids.size == 0 and dist2.size:
                ids = np.asarray([int(np.argmin(dist2))], dtype=np.int64)
        ids = ids[np.argsort(dist2[ids], kind="stable")]
        if cached_face_id is not None and 0 <= int(cached_face_id) < self.material.face_count:
            cached = int(cached_face_id)
            ids = np.asarray([cached, *[int(i) for i in ids if int(i) != cached]], dtype=np.int64)
        return ids

    def _cell_index(self, point: np.ndarray) -> tuple[int, int, int]:
        ijk = np.floor((point - self.origin) / self.cell_size).astype(np.int64)
        return int(ijk[0]), int(ijk[1]), int(ijk[2])

    def aabb_distance_squared(self, point: np.ndarray) -> np.ndarray:
        """Return squared distances from ``point`` to tight patch AABBs."""

        x = _as_point(point, "point")
        lower_delta = np.maximum(self.aabb_min - x, 0.0)
        upper_delta = np.maximum(x - self.aabb_max, 0.0)
        return np.sum((lower_delta + upper_delta) ** 2, axis=1)

    def _hash_candidates_for_point(self, point: np.ndarray) -> np.ndarray:
        ids = self.cells.get(self._cell_index(point), ())
        if not ids:
            return np.empty(0, dtype=np.int64)
        unique = np.asarray(sorted(set(int(v) for v in ids)), dtype=np.int64)
        contains = np.all((self.cell_aabb_min[unique] <= point) & (point <= self.cell_aabb_max[unique]), axis=1)
        return unique[contains]

    def _hash_candidates_for_ball(self, point: np.ndarray, radius: float) -> np.ndarray:
        lower = point - float(radius)
        upper = point + float(radius)
        ids: set[int] = set()
        for key in _cell_keys_for_aabb(lower, upper, self.origin, self.cell_size):
            ids.update(self.cells.get(key, ()))
        if not ids:
            return np.empty(0, dtype=np.int64)
        return np.asarray(sorted(ids), dtype=np.int64)


def _as_nodes(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    return arr.copy()


def _as_faces(value: np.ndarray, n_nodes: int) -> np.ndarray:
    faces = np.asarray(value, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("boundary_faces must have shape (m, 3)")
    if faces.shape[0] == 0:
        raise ValueError("boundary_faces must contain at least one face")
    if np.any(faces < 0) or int(faces.max()) >= int(n_nodes):
        raise ValueError("boundary_faces reference nodes outside reference_nodes")
    return faces.copy()


def _as_point(value: np.ndarray, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return point


def _as_spacing(value: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        arr = np.full(3, float(arr), dtype=float)
    if arr.shape != (3,):
        raise ValueError("spacing must be a positive scalar or have shape (3,)")
    if np.any(arr <= 0.0):
        raise ValueError("spacing entries must be positive")
    return arr


def _cell_keys_for_aabb(
    lower: np.ndarray,
    upper: np.ndarray,
    origin: np.ndarray,
    cell_size: float,
):
    lo = np.floor((lower - origin) / float(cell_size)).astype(np.int64)
    hi = np.floor((upper - origin) / float(cell_size)).astype(np.int64)
    for i in range(int(lo[0]), int(hi[0]) + 1):
        for j in range(int(lo[1]), int(hi[1]) + 1):
            for k in range(int(lo[2]), int(hi[2]) + 1):
                yield i, j, k


def _axis_cubic_indices_and_weights(u: float, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = int(np.floor(float(u))) - 1
    base = max(0, min(base, int(size) - 4))
    nodes = np.arange(base, base + 4, dtype=np.int64)
    t = float(u) - nodes.astype(float)
    weights = np.ones(4, dtype=float)
    derivatives = np.zeros(4, dtype=float)
    for i in range(4):
        wi = 1.0
        for j in range(4):
            if i == j:
                continue
            wi *= t[j] / float(nodes[i] - nodes[j])
        weights[i] = wi
        derivative = 0.0
        for m in range(4):
            if m == i:
                continue
            term = 1.0 / float(nodes[i] - nodes[m])
            for j in range(4):
                if j == i or j == m:
                    continue
                term *= t[j] / float(nodes[i] - nodes[j])
            derivative += term
        derivatives[i] = derivative
    return nodes, weights, derivatives


def _tricubic_value_and_derivative(
    values: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    point: np.ndarray,
) -> tuple[float, np.ndarray]:
    p = _as_point(point, "point")
    arr = np.asarray(values, dtype=float)
    h = _as_spacing(spacing)
    u = (p - _as_point(origin, "origin")) / h
    if np.any(u < -1.0e-12) or np.any(u > np.asarray(arr.shape, dtype=float) - 1.0 + 1.0e-12):
        raise ValueError("query point is outside the material SDF grid")
    idx0, w0, dw0 = _axis_cubic_indices_and_weights(float(u[0]), arr.shape[0])
    idx1, w1, dw1 = _axis_cubic_indices_and_weights(float(u[1]), arr.shape[1])
    idx2, w2, dw2 = _axis_cubic_indices_and_weights(float(u[2]), arr.shape[2])
    value = 0.0
    grad_u = np.zeros(3, dtype=float)
    for a, ia in enumerate(idx0):
        for b, ib in enumerate(idx1):
            for c, ic in enumerate(idx2):
                phi = float(arr[int(ia), int(ib), int(ic)])
                value += w0[a] * w1[b] * w2[c] * phi
                grad_u[0] += dw0[a] * w1[b] * w2[c] * phi
                grad_u[1] += w0[a] * dw1[b] * w2[c] * phi
                grad_u[2] += w0[a] * w1[b] * dw2[c] * phi
    return float(value), grad_u / h


def _unit_normals(triangles: np.ndarray) -> np.ndarray:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    if bool(np.any(norms <= 0.0)):
        raise ValueError("reference patch has zero area")
    return normals / norms[:, None]
