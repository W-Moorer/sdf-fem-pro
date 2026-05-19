"""Reference-space SDF data for deformation-aware contact queries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

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
        band_radius: float = 0.0,
    ) -> "MaterialSDF":
        """Create a material SDF from reference surface triangles."""

        return cls(
            reference_nodes=reference_nodes,
            boundary_faces=boundary_faces,
            phi=phi,
            gradient=gradient,
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
        result = self.closest_reference_patch(X)
        return float(result.gap)

    def query_gradient(self, point: np.ndarray) -> np.ndarray:
        """Evaluate ``grad_X phi0(X)``."""

        X = _as_point(point, "point")
        if self.gradient is not None:
            grad = np.asarray(self.gradient(X), dtype=float)
            if grad.shape != (3,):
                raise ValueError("gradient(point) must return shape (3,)")
            norm = float(np.linalg.norm(grad))
            if norm <= 0.0:
                raise ValueError("gradient(point) must be nonzero")
            return grad / norm
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
    ) -> "ReferencePatchBVH":
        """Build/refit a current-space AABB index over reference patches."""

        return ReferencePatchBVH.from_material(self, x_current, padding=padding)


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
    padding: float

    @classmethod
    def from_material(
        cls,
        material: MaterialSDF,
        x_current: np.ndarray,
        *,
        padding: float | None = None,
    ) -> "ReferencePatchBVH":
        """Refit current AABBs from the same reference patch topology."""

        X = _as_nodes(x_current, "x_current")
        if X.shape != material.reference_nodes.shape:
            raise ValueError("x_current must match material.reference_nodes shape")
        pad = material.band_radius if padding is None else float(padding)
        if pad < 0.0:
            raise ValueError("padding must be non-negative")
        triangles = X[material.boundary_faces]
        mins = triangles.min(axis=1) - pad
        maxs = triangles.max(axis=1) + pad
        return cls(
            material=material,
            x_current=X,
            aabb_min=mins,
            aabb_max=maxs,
            padding=pad,
        )

    def refit(self, x_current: np.ndarray, *, padding: float | None = None) -> "ReferencePatchBVH":
        """Return a refit index with unchanged reference patch topology."""

        return type(self).from_material(
            self.material,
            x_current,
            padding=self.padding if padding is None else padding,
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
        lower_delta = np.maximum(self.aabb_min - x, 0.0)
        upper_delta = np.maximum(x - self.aabb_max, 0.0)
        dist2 = np.sum((lower_delta + upper_delta) ** 2, axis=1)
        if search_radius is None:
            ids = np.arange(self.material.face_count, dtype=np.int64)
        else:
            radius = float(search_radius)
            if radius < 0.0:
                raise ValueError("search_radius must be non-negative")
            ids = np.flatnonzero(dist2 <= radius * radius).astype(np.int64)
            if ids.size == 0 and dist2.size:
                ids = np.asarray([int(np.argmin(dist2))], dtype=np.int64)
        ids = ids[np.argsort(dist2[ids], kind="stable")]
        if cached_face_id is not None and 0 <= int(cached_face_id) < self.material.face_count:
            cached = int(cached_face_id)
            ids = np.asarray([cached, *[int(i) for i in ids if int(i) != cached]], dtype=np.int64)
        return ids


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


def _unit_normals(triangles: np.ndarray) -> np.ndarray:
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    if bool(np.any(norms <= 0.0)):
        raise ValueError("reference patch has zero area")
    return normals / norms[:, None]
