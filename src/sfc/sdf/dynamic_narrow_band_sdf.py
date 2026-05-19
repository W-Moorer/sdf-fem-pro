"""Dynamic narrow-band SDF field induced by the current FEM surface."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .dynamic_surface_sdf import (
    SurfaceSDFResult,
    compiled_projection_available,
    surface_projection_distance_kernel,
    surface_projection_distance_kernel_batch_all_faces,
    surface_projection_distance_kernel_batch_all_faces_compiled,
    surface_projection_distance_kernel_batch_candidates,
    surface_projection_distance_kernel_batch_padded_aabb_compiled,
)
from .narrow_band_grid import (
    GridPayloadInterpolation,
    NarrowBandGrid,
    compute_finite_difference_gradient,
    grid_spec_from_surface,
)

_TRILINEAR_CORNERS = np.asarray(
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
class SDFUpdateStats:
    """Timing and size information for one dynamic SDF rebuild."""

    grid_node_count: int
    valid_node_count: int
    spacing: np.ndarray
    band_radius: float
    update_seconds: float
    projection_seconds: float
    gradient_seconds: float


@dataclass(frozen=True, slots=True)
class FieldQueryPayload:
    """Interpolated payload needed for master-surface gap sensitivity."""

    grid_weights: np.ndarray
    face_ids: np.ndarray
    face_node_ids: np.ndarray
    barycentric: np.ndarray
    normals: np.ndarray

    def master_sensitivity_terms(self) -> tuple[np.ndarray, np.ndarray]:
        """Return master-node ids and ``dg/dx_node`` vectors.

        The field formulation differentiates interpolated grid-node SDF values:
        ``dg/dx_master = -sum_l alpha_l N_l n_l^T``.
        """

        node_ids: list[int] = []
        vectors: list[np.ndarray] = []
        for alpha, nodes, bary, normal in zip(
            self.grid_weights,
            self.face_node_ids,
            self.barycentric,
            self.normals,
            strict=True,
        ):
            if alpha == 0.0:
                continue
            for node, beta in zip(nodes, bary, strict=True):
                weight = float(alpha) * float(beta)
                if weight == 0.0:
                    continue
                node_ids.append(int(node))
                vectors.append(-weight * np.asarray(normal, dtype=float))
        if not node_ids:
            return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=float)
        return _aggregate_node_vectors(np.asarray(node_ids, dtype=np.int64), np.vstack(vectors))


class DynamicNarrowBandSDF:
    """Interpolation-only dynamic narrow-band SDF query object.

    The closest-point projection kernel is used while building the grid. Public
    field queries interpolate stored grid data and do not call projection.
    """

    def __init__(
        self,
        grid: NarrowBandGrid,
        x_current: np.ndarray,
        boundary_faces: np.ndarray,
        stats: SDFUpdateStats,
    ) -> None:
        self.grid = grid
        self.x_current = _as_surface_nodes(x_current)
        self.boundary_faces = _as_boundary_faces(boundary_faces, self.x_current.shape[0])
        self.stats = stats

    @classmethod
    def build(
        cls,
        x_current: np.ndarray,
        boundary_faces: np.ndarray,
        *,
        spacing: float | np.ndarray,
        band_radius: float,
        origin: np.ndarray | None = None,
        shape: tuple[int, int, int] | np.ndarray | None = None,
        padding: float | None = None,
        cell_size: float | None = None,
        gradient_mode: str = "surface_normal",
    ) -> "DynamicNarrowBandSDF":
        """Build a current-space narrow-band SDF from boundary triangles."""

        t0 = perf_counter()
        X = _as_surface_nodes(x_current)
        faces = _as_boundary_faces(boundary_faces, X.shape[0])
        h = _as_spacing(spacing)
        band = float(band_radius)
        if band <= 0.0:
            raise ValueError("band_radius must be positive")
        if origin is None or shape is None:
            grid_origin, grid_shape = grid_spec_from_surface(
                X,
                faces,
                spacing=h,
                padding=band if padding is None else float(padding),
            )
        else:
            grid_origin = _as_vector3(origin, "origin")
            grid_shape = _as_shape3(shape)

        phi = np.full(grid_shape, np.nan, dtype=float)
        closest_face_id = np.full(grid_shape, -1, dtype=np.int64)
        barycentric = np.zeros((*grid_shape, 3), dtype=float)
        closest_normal = np.zeros((*grid_shape, 3), dtype=float)
        valid_mask = np.zeros(grid_shape, dtype=bool)

        from sfc.contact.broad_phase import UniformTriangleAABBHash

        candidate_padding = band + 0.5 * float(np.linalg.norm(h))
        broad_phase = UniformTriangleAABBHash.from_surface(
            X,
            faces,
            delta_safe=candidate_padding,
            cell_size=cell_size,
        )

        projection_start = perf_counter()
        for index in np.ndindex(grid_shape):
            point = grid_origin + h * np.asarray(index, dtype=float)
            candidates = broad_phase.query_point(point)
            if candidates.size == 0:
                continue
            result = surface_projection_distance_kernel(point, X, faces, candidates)
            _store_projection_payload(
                index,
                result,
                phi,
                closest_face_id,
                barycentric,
                closest_normal,
                valid_mask,
                band,
            )
        projection_seconds = perf_counter() - projection_start

        gradient_start = perf_counter()
        if gradient_mode == "surface_normal":
            gradient = closest_normal.copy()
        elif gradient_mode == "finite_difference":
            gradient = compute_finite_difference_gradient(
                phi,
                h,
                fallback_normals=closest_normal,
            )
        else:
            raise ValueError("gradient_mode must be 'surface_normal' or 'finite_difference'")
        invalid = ~np.isfinite(phi)
        if bool(np.any(invalid)):
            gradient[invalid] = closest_normal[invalid]
        gradient_seconds = perf_counter() - gradient_start

        grid = NarrowBandGrid(
            origin=grid_origin,
            spacing=h,
            phi=phi,
            gradient=gradient,
            closest_face_id=closest_face_id,
            barycentric=barycentric,
            closest_normal=closest_normal,
            valid_mask=valid_mask,
            metadata={
                "band_radius": band,
                "candidate_padding": candidate_padding,
            },
        )
        stats = SDFUpdateStats(
            grid_node_count=int(np.prod(grid_shape)),
            valid_node_count=int(np.count_nonzero(valid_mask)),
            spacing=h.copy(),
            band_radius=band,
            update_seconds=perf_counter() - t0,
            projection_seconds=projection_seconds,
            gradient_seconds=gradient_seconds,
        )
        return cls(grid, X, faces, stats)

    @classmethod
    def build_required_points(
        cls,
        x_current: np.ndarray,
        boundary_faces: np.ndarray,
        query_points: np.ndarray,
        *,
        spacing: float | np.ndarray,
        band_radius: float,
        origin: np.ndarray | None = None,
        shape: tuple[int, int, int] | np.ndarray | None = None,
        padding: float | None = None,
        cell_size: float | None = None,
        batch_projection_threshold: int = 5_000_000,
    ) -> "DynamicNarrowBandSDF":
        """Build an exact sparse field for the interpolation cells touched by queries.

        This method computes the same closest-feature payload as ``build`` for
        every grid node needed by trilinear interpolation of ``query_points``.
        Unused grid nodes are left invalid. Query-time APIs remain
        interpolation-only; if a query touches an unpopulated cell, the normal
        narrow-band validity checks raise instead of falling back to projection.
        """

        t0 = perf_counter()
        X = _as_surface_nodes(x_current)
        faces = _as_boundary_faces(boundary_faces, X.shape[0])
        points = _as_query_points(query_points)
        h = _as_spacing(spacing)
        band = float(band_radius)
        if band <= 0.0:
            raise ValueError("band_radius must be positive")
        if origin is None or shape is None:
            grid_origin, grid_shape = grid_spec_from_surface(
                X,
                faces,
                spacing=h,
                padding=band if padding is None else float(padding),
            )
        else:
            grid_origin = _as_vector3(origin, "origin")
            grid_shape = _as_shape3(shape)

        required_indices = _required_trilinear_corner_indices(points, grid_origin, h, grid_shape)
        phi = np.full(grid_shape, np.nan, dtype=float)
        closest_face_id = np.full(grid_shape, -1, dtype=np.int64)
        barycentric = np.zeros((*grid_shape, 3), dtype=float)
        closest_normal = np.zeros((*grid_shape, 3), dtype=float)
        valid_mask = np.zeros(grid_shape, dtype=bool)

        candidate_padding = band + 0.5 * float(np.linalg.norm(h))

        projection_start = perf_counter()
        pair_count = int(required_indices.shape[0]) * int(faces.shape[0])
        points_required = grid_origin + h * required_indices.astype(float)
        threshold = int(batch_projection_threshold)
        used_batch_projection = pair_count <= threshold
        projection_mode = "all_faces_batch" if used_batch_projection else "candidate_group_batch"
        if used_batch_projection:
            batch = surface_projection_distance_kernel_batch_all_faces(points_required, X, faces)
            ii = required_indices[:, 0]
            jj = required_indices[:, 1]
            kk = required_indices[:, 2]
            phi[ii, jj, kk] = batch.g
            closest_face_id[ii, jj, kk] = batch.face_id
            barycentric[ii, jj, kk] = batch.w
            closest_normal[ii, jj, kk] = _unit_rows(batch.n)
            valid_mask[ii, jj, kk] = np.abs(batch.g) <= band + 1.0e-12
        else:
            from sfc.contact.broad_phase import UniformTriangleAABBHash

            broad_phase = UniformTriangleAABBHash.from_surface(
                X,
                faces,
                delta_safe=candidate_padding,
                cell_size=cell_size,
            )
            groups: dict[tuple[int, ...], list[int]] = {}
            for point_id, candidates in enumerate(broad_phase.query_points(points_required)):
                if candidates.size == 0:
                    continue
                key = tuple(int(v) for v in candidates.tolist())
                groups.setdefault(key, []).append(point_id)
            for key, point_ids in groups.items():
                candidates = np.asarray(key, dtype=np.int64)
                chunk_size = max(1, threshold // max(int(candidates.size), 1)) if threshold > 0 else len(point_ids)
                for start in range(0, len(point_ids), chunk_size):
                    ids = np.asarray(point_ids[start : start + chunk_size], dtype=np.int64)
                    batch = surface_projection_distance_kernel_batch_candidates(
                        points_required[ids],
                        X,
                        faces,
                        candidates,
                    )
                    idx = required_indices[ids]
                    ii = idx[:, 0]
                    jj = idx[:, 1]
                    kk = idx[:, 2]
                    phi[ii, jj, kk] = batch.g
                    closest_face_id[ii, jj, kk] = batch.face_id
                    barycentric[ii, jj, kk] = batch.w
                    closest_normal[ii, jj, kk] = _unit_rows(batch.n)
                    valid_mask[ii, jj, kk] = np.abs(batch.g) <= band + 1.0e-12
        projection_seconds = perf_counter() - projection_start

        gradient_start = perf_counter()
        gradient = closest_normal.copy()
        gradient_seconds = perf_counter() - gradient_start
        grid = NarrowBandGrid(
            origin=grid_origin,
            spacing=h,
            phi=phi,
            gradient=gradient,
            closest_face_id=closest_face_id,
            barycentric=barycentric,
            closest_normal=closest_normal,
            valid_mask=valid_mask,
            metadata={
                "band_radius": band,
                "candidate_padding": candidate_padding,
                "population_mode": "required_points",
                "required_node_count": int(required_indices.shape[0]),
                "batch_projection": bool(used_batch_projection),
                "projection_mode": projection_mode,
                "projection_pair_count": int(pair_count),
            },
        )
        stats = SDFUpdateStats(
            grid_node_count=int(np.prod(grid_shape)),
            valid_node_count=int(np.count_nonzero(valid_mask)),
            spacing=h.copy(),
            band_radius=band,
            update_seconds=perf_counter() - t0,
            projection_seconds=projection_seconds,
            gradient_seconds=gradient_seconds,
        )
        return cls(grid, X, faces, stats)


    @property
    def field_update_cost(self) -> float:
        """Return the measured rebuild time in seconds."""

        return float(self.stats.update_seconds)

    def query_phi(self, point: np.ndarray) -> float:
        """Return interpolated signed distance."""

        return self.grid.query_phi(point)

    def query_spatial_derivative_phi(self, point: np.ndarray) -> np.ndarray:
        """Return ``grad_x`` of the trilinearly interpolated scalar ``phi``.

        This is the derivative of ``query_phi(point)`` and is therefore the
        gradient used by variational field-contact Jacobians.
        """

        return self.grid.query_spatial_derivative_phi(point)

    def query_gradient(self, point: np.ndarray) -> np.ndarray:
        """Return the derivative of the interpolated scalar SDF value."""

        return self.query_spatial_derivative_phi(point)

    def query_interpolated_gradient(self, point: np.ndarray) -> np.ndarray:
        """Return the stored gradient-grid interpolation for diagnostics."""

        return self.grid.query_gradient(point)

    def query_geometric_normal(self, point: np.ndarray) -> np.ndarray:
        """Return a normalized interpolated closest-feature normal estimate."""

        payload = self.grid.payload(point)
        normal = payload.weights @ payload.normals
        norm = float(np.linalg.norm(normal))
        if norm <= 1.0e-14:
            stored = self.query_interpolated_gradient(point)
            norm = float(np.linalg.norm(stored))
            if norm <= 1.0e-14:
                raise ValueError("interpolated closest-feature normal is zero")
            normal = stored
        return normal / norm

    def query_gap_normal(self, point: np.ndarray) -> tuple[float, np.ndarray]:
        """Return interpolated gap and normalized scalar-field derivative."""

        phi = self.query_phi(point)
        grad = self.query_spatial_derivative_phi(point)
        norm = float(np.linalg.norm(grad))
        if norm <= 1.0e-14:
            grad = self.query_geometric_normal(point)
            norm = float(np.linalg.norm(grad))
        if norm <= 1.0e-14:
            raise ValueError("interpolated SDF gradient is zero")
        return phi, grad / norm

    def query_payload(self, point: np.ndarray) -> FieldQueryPayload:
        """Return interpolated closest-face payload for master sensitivity."""

        payload: GridPayloadInterpolation = self.grid.payload(point)
        if np.any(payload.face_ids < 0):
            raise ValueError("query point includes invalid grid-node payload")
        return FieldQueryPayload(
            grid_weights=payload.weights,
            face_ids=payload.face_ids,
            face_node_ids=self.boundary_faces[payload.face_ids].copy(),
            barycentric=payload.barycentric,
            normals=payload.normals,
        )

    def refine_query_projection(self, point: np.ndarray) -> SurfaceSDFResult:
        """Optionally refine a query by projecting on payload candidate faces."""

        payload = self.query_payload(point)
        candidates = np.unique(payload.face_ids)
        return surface_projection_distance_kernel(point, self.x_current, self.boundary_faces, candidates)

    def eikonal_residuals(self) -> np.ndarray:
        """Return ``|grad(phi)| - 1`` on valid grid nodes."""

        valid = self.grid.valid_mask
        if not bool(np.any(valid)):
            return np.empty(0, dtype=float)
        norms = np.linalg.norm(self.grid.gradient[valid], axis=1)
        return norms - 1.0


class RequiredPointSDFWorkspace:
    """Reusable sparse narrow-band SDF builder for repeated contact steps.

    The workspace preserves the exact ``build_required_points`` semantics for a
    fixed grid specification, but reuses the dense grid arrays between rebuilds
    and clears only the grid nodes touched by the previous build.  Public query
    APIs still receive a normal ``DynamicNarrowBandSDF`` object and remain
    interpolation-only.
    """

    def __init__(
        self,
        *,
        spacing: float | np.ndarray,
        band_radius: float,
        origin: np.ndarray,
        shape: tuple[int, int, int] | np.ndarray,
        cell_size: float | None = None,
        batch_projection_threshold: int = 5_000_000,
        projection_backend: str = "auto",
        candidate_padding: float | None = None,
    ) -> None:
        h = _as_spacing(spacing)
        band = float(band_radius)
        if band <= 0.0:
            raise ValueError("band_radius must be positive")
        grid_origin = _as_vector3(origin, "origin")
        grid_shape = _as_shape3(shape)
        self.spacing = h
        self.band_radius = band
        self.origin = grid_origin
        self.shape = grid_shape
        self.cell_size = cell_size
        self.batch_projection_threshold = int(batch_projection_threshold)
        if projection_backend not in {"auto", "numpy", "compiled", "compiled_all_faces"}:
            raise ValueError("projection_backend must be 'auto', 'numpy', 'compiled', or 'compiled_all_faces'")
        self.projection_backend = projection_backend
        self.candidate_padding = None if candidate_padding is None else float(candidate_padding)
        self.phi = np.full(grid_shape, np.nan, dtype=float)
        self.closest_face_id = np.full(grid_shape, -1, dtype=np.int64)
        self.barycentric = np.zeros((*grid_shape, 3), dtype=float)
        self.closest_normal = np.zeros((*grid_shape, 3), dtype=float)
        self.gradient = np.zeros((*grid_shape, 3), dtype=float)
        self.valid_mask = np.zeros(grid_shape, dtype=bool)
        self._previous_indices = np.empty((0, 3), dtype=np.int64)

    @classmethod
    def from_surface(
        cls,
        x_reference: np.ndarray,
        boundary_faces: np.ndarray,
        *,
        spacing: float | np.ndarray,
        band_radius: float,
        padding: float | None = None,
        cell_size: float | None = None,
        batch_projection_threshold: int = 5_000_000,
        projection_backend: str = "auto",
        candidate_padding: float | None = None,
    ) -> "RequiredPointSDFWorkspace":
        """Create a reusable workspace from a reference surface grid spec."""

        X = _as_surface_nodes(x_reference)
        faces = _as_boundary_faces(boundary_faces, X.shape[0])
        h = _as_spacing(spacing)
        band = float(band_radius)
        origin, shape = grid_spec_from_surface(
            X,
            faces,
            spacing=h,
            padding=band if padding is None else float(padding),
        )
        return cls(
            spacing=h,
            band_radius=band,
            origin=origin,
            shape=shape,
            cell_size=cell_size,
            batch_projection_threshold=batch_projection_threshold,
            projection_backend=projection_backend,
            candidate_padding=candidate_padding,
        )

    def _clear_previous(self) -> None:
        if self._previous_indices.size == 0:
            return
        ii = self._previous_indices[:, 0]
        jj = self._previous_indices[:, 1]
        kk = self._previous_indices[:, 2]
        self.phi[ii, jj, kk] = np.nan
        self.closest_face_id[ii, jj, kk] = -1
        self.barycentric[ii, jj, kk] = 0.0
        self.closest_normal[ii, jj, kk] = 0.0
        self.gradient[ii, jj, kk] = 0.0
        self.valid_mask[ii, jj, kk] = False

    def build(
        self,
        x_current: np.ndarray,
        boundary_faces: np.ndarray,
        query_points: np.ndarray,
    ) -> DynamicNarrowBandSDF:
        """Populate the reusable workspace for the current surface and queries."""

        t0 = perf_counter()
        X = _as_surface_nodes(x_current)
        faces = _as_boundary_faces(boundary_faces, X.shape[0])
        points = _as_query_points(query_points)
        required_indices = _required_trilinear_corner_indices(points, self.origin, self.spacing, self.shape)
        self._clear_previous()
        self._previous_indices = required_indices.copy()

        exact_fallback_distance = self.band_radius + 0.5 * float(np.linalg.norm(self.spacing))
        candidate_padding = exact_fallback_distance if self.candidate_padding is None else float(self.candidate_padding)
        projection_start = perf_counter()
        pair_count = int(required_indices.shape[0]) * int(faces.shape[0])
        points_required = self.origin + self.spacing * required_indices.astype(float)
        threshold = int(self.batch_projection_threshold)
        use_compiled = self.projection_backend in {"compiled", "compiled_all_faces"} or (
            self.projection_backend == "auto" and compiled_projection_available()
        )
        used_batch_projection = pair_count <= threshold
        if use_compiled:
            if self.projection_backend == "compiled_all_faces":
                batch = surface_projection_distance_kernel_batch_all_faces_compiled(points_required, X, faces)
                projection_mode = "compiled_all_faces"
            else:
                batch = surface_projection_distance_kernel_batch_padded_aabb_compiled(
                    points_required,
                    X,
                    faces,
                    delta_safe=candidate_padding,
                    fallback_distance=candidate_padding,
                )
                projection_mode = "compiled_padded_aabb"
            ids = np.arange(required_indices.shape[0], dtype=np.int64)
            self._store_batch(required_indices, ids, batch)
        elif used_batch_projection:
            batch = surface_projection_distance_kernel_batch_all_faces(points_required, X, faces)
            ids = np.arange(required_indices.shape[0], dtype=np.int64)
            self._store_batch(required_indices, ids, batch)
            projection_mode = "all_faces_batch"
        else:
            from sfc.contact.broad_phase import UniformTriangleAABBHash

            projection_mode = "candidate_group_batch"
            broad_phase = UniformTriangleAABBHash.from_surface(
                X,
                faces,
                delta_safe=candidate_padding,
                cell_size=self.cell_size,
            )
            groups: dict[tuple[int, ...], list[int]] = {}
            for point_id, candidates in enumerate(broad_phase.query_points(points_required)):
                if candidates.size == 0:
                    continue
                key = tuple(int(v) for v in candidates.tolist())
                groups.setdefault(key, []).append(point_id)
            for key, point_ids in groups.items():
                candidates = np.asarray(key, dtype=np.int64)
                chunk_size = max(1, threshold // max(int(candidates.size), 1)) if threshold > 0 else len(point_ids)
                for start in range(0, len(point_ids), chunk_size):
                    ids = np.asarray(point_ids[start : start + chunk_size], dtype=np.int64)
                    batch = surface_projection_distance_kernel_batch_candidates(
                        points_required[ids],
                        X,
                        faces,
                        candidates,
                    )
                    self._store_batch(required_indices, ids, batch)
        projection_seconds = perf_counter() - projection_start

        gradient_start = perf_counter()
        ii = required_indices[:, 0]
        jj = required_indices[:, 1]
        kk = required_indices[:, 2]
        self.gradient[ii, jj, kk] = self.closest_normal[ii, jj, kk]
        gradient_seconds = perf_counter() - gradient_start
        grid = NarrowBandGrid(
            origin=self.origin,
            spacing=self.spacing,
            phi=self.phi,
            gradient=self.gradient,
            closest_face_id=self.closest_face_id,
            barycentric=self.barycentric,
            closest_normal=self.closest_normal,
            valid_mask=self.valid_mask,
            metadata={
                "band_radius": self.band_radius,
                "candidate_padding": candidate_padding,
                "exact_fallback_distance": exact_fallback_distance,
                "population_mode": "required_points_workspace",
                "required_node_count": int(required_indices.shape[0]),
                "batch_projection": bool(used_batch_projection),
                "compiled_projection": bool(use_compiled),
                "projection_mode": projection_mode,
                "projection_pair_count": int(pair_count),
            },
        )
        valid_node_count = int(np.count_nonzero(self.valid_mask[ii, jj, kk]))
        stats = SDFUpdateStats(
            grid_node_count=int(np.prod(self.shape)),
            valid_node_count=valid_node_count,
            spacing=self.spacing.copy(),
            band_radius=self.band_radius,
            update_seconds=perf_counter() - t0,
            projection_seconds=projection_seconds,
            gradient_seconds=gradient_seconds,
        )
        return DynamicNarrowBandSDF(grid, X, faces, stats)

    def _store_batch(self, required_indices: np.ndarray, ids: np.ndarray, batch: SurfaceSDFResult) -> None:
        idx = required_indices[ids]
        ii = idx[:, 0]
        jj = idx[:, 1]
        kk = idx[:, 2]
        self.phi[ii, jj, kk] = batch.g
        self.closest_face_id[ii, jj, kk] = batch.face_id
        self.barycentric[ii, jj, kk] = batch.w
        self.closest_normal[ii, jj, kk] = _unit_rows(batch.n)
        self.valid_mask[ii, jj, kk] = np.abs(batch.g) <= self.band_radius + 1.0e-12


def _store_projection_payload(
    index: tuple[int, int, int],
    result: SurfaceSDFResult,
    phi: np.ndarray,
    closest_face_id: np.ndarray,
    barycentric: np.ndarray,
    closest_normal: np.ndarray,
    valid_mask: np.ndarray,
    band: float,
) -> None:
    phi[index] = float(result.g)
    closest_face_id[index] = int(result.face_id)
    barycentric[index] = np.asarray(result.w, dtype=float)
    closest_normal[index] = _unit(np.asarray(result.n, dtype=float))
    valid_mask[index] = abs(float(result.g)) <= band + 1.0e-12


def _as_query_points(value: np.ndarray) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim == 1:
        if points.shape != (3,):
            raise ValueError("query_points must have shape (n, 3)")
        points = points.reshape((1, 3))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("query_points must have shape (n, 3)")
    if points.shape[0] == 0:
        raise ValueError("query_points must contain at least one point")
    return points


def _required_trilinear_corner_indices(
    points: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    grid_shape = np.asarray(shape, dtype=np.int64)
    u = (points - origin) / spacing
    if bool(np.any(u < -1.0e-12) or np.any(u > (grid_shape.astype(float) - 1.0) + 1.0e-12)):
        raise ValueError("query point is outside the narrow-band grid")
    base = np.floor(u).astype(np.int64)
    for axis in range(3):
        base[:, axis] = np.clip(base[:, axis], 0, int(grid_shape[axis]) - 2)
    corners = (base[:, None, :] + _TRILINEAR_CORNERS[None, :, :]).reshape((-1, 3))
    stride_i = int(grid_shape[1]) * int(grid_shape[2])
    stride_j = int(grid_shape[2])
    linear = corners[:, 0] * stride_i + corners[:, 1] * stride_j + corners[:, 2]
    unique_linear = np.unique(linear)
    ii = unique_linear // stride_i
    rem = unique_linear - ii * stride_i
    jj = rem // stride_j
    kk = rem - jj * stride_j
    return np.column_stack((ii, jj, kk)).astype(np.int64, copy=False)


def _aggregate_node_vectors(node_ids: np.ndarray, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(node_ids, kind="stable")
    sorted_nodes = node_ids[order]
    sorted_vectors = vectors[order]
    unique_nodes: list[int] = []
    summed: list[np.ndarray] = []
    for node, vector in zip(sorted_nodes, sorted_vectors, strict=True):
        if unique_nodes and int(node) == unique_nodes[-1]:
            summed[-1] = summed[-1] + vector
        else:
            unique_nodes.append(int(node))
            summed.append(np.asarray(vector, dtype=float).copy())
    return np.asarray(unique_nodes, dtype=np.int64), np.vstack(summed)


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        return np.zeros(3, dtype=float)
    return value / norm


def _unit_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    norms = np.linalg.norm(arr, axis=1)
    out = np.zeros_like(arr, dtype=float)
    mask = norms > 0.0
    out[mask] = arr[mask] / norms[mask, None]
    return out


def _as_surface_nodes(value: np.ndarray) -> np.ndarray:
    X = np.asarray(value, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("x_current must have shape (n, 3)")
    return X


def _as_boundary_faces(value: np.ndarray, n_nodes: int) -> np.ndarray:
    faces = np.asarray(value, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("boundary_faces must have shape (m, 3)")
    if faces.size == 0:
        raise ValueError("boundary_faces must contain at least one triangle")
    if np.any(faces < 0) or int(faces.max()) >= int(n_nodes):
        raise ValueError("boundary_faces reference nodes outside x_current")
    return faces


def _as_vector3(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return arr


def _as_spacing(value: float | np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        arr = np.full(3, float(arr), dtype=float)
    if arr.shape != (3,):
        raise ValueError("spacing must be a positive scalar or have shape (3,)")
    if np.any(arr <= 0.0):
        raise ValueError("spacing entries must be positive")
    return arr


def _as_shape3(value: tuple[int, int, int] | np.ndarray) -> tuple[int, int, int]:
    arr = np.asarray(value, dtype=np.int64)
    if arr.shape != (3,):
        raise ValueError("shape must have three entries")
    if np.any(arr < 2):
        raise ValueError("each grid axis must contain at least two nodes")
    return int(arr[0]), int(arr[1]), int(arr[2])
