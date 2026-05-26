"""Surface-contact geometry backed by the Lagrangian SDF oracle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from sfc.contact.lagrangian_sdf_oracle import LagrangianSDFContactOracle
from sfc.fem.calculix_aligned import ContactSample
from sfc.sdf.material_sdf import MaterialSDF

try:  # pragma: no cover - optional backend availability is platform-dependent.
    from sfc.sdf._cpp_projection import closest_points_all_faces as _cpp_closest_points_all_faces
    from sfc.sdf._cpp_projection import closest_points_indexed_faces as _cpp_closest_points_indexed_faces
    from sfc.sdf._cpp_projection import (
        closest_points_indexed_faces_normal_compatible as _cpp_closest_points_indexed_faces_normal_compatible,
    )
    from sfc.sdf._cpp_projection import (
        closest_points_indexed_faces_secondary_normal as _cpp_closest_points_indexed_faces_secondary_normal,
    )
    from sfc.sdf._cpp_projection import closest_points_padded_aabb as _cpp_closest_points_padded_aabb
    from sfc.sdf._cpp_projection import indexed_faces_available as _cpp_indexed_faces_available
    from sfc.sdf._cpp_projection import is_available as _cpp_projection_available
    from sfc.sdf._cpp_projection import (
        normal_compatible_indexed_faces_available as _cpp_normal_compatible_indexed_faces_available,
    )
    from sfc.sdf._cpp_projection import secondary_normal_indexed_faces_available as _cpp_secondary_normal_indexed_faces_available
except Exception:  # pragma: no cover
    _cpp_closest_points_all_faces = None
    _cpp_closest_points_indexed_faces = None
    _cpp_closest_points_indexed_faces_normal_compatible = None
    _cpp_closest_points_indexed_faces_secondary_normal = None
    _cpp_closest_points_padded_aabb = None
    _cpp_indexed_faces_available = lambda: False
    _cpp_normal_compatible_indexed_faces_available = lambda: False
    _cpp_secondary_normal_indexed_faces_available = lambda: False
    _cpp_projection_available = lambda: False


_TRI3_BARY = np.asarray(
    [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ],
    dtype=float,
)


@dataclass(frozen=True, slots=True)
class QuadrilateralClosestFeaturePayload:
    """Closest-feature payload on one current Q4 master face."""

    gap: float
    normal: np.ndarray
    closest_point: np.ndarray
    face_id: int
    natural_coordinates: np.ndarray
    master_node_ids: np.ndarray
    master_weights: np.ndarray
    iterations: int
    converged: bool
    candidates_evaluated: int


@dataclass(slots=True)
class LagrangianQ4ClosestFeatureOracle:
    """Current-space closest-feature oracle over Lagrangian Q4 master faces.

    The oracle stores Q4 face topology in reference numbering and refits only
    the current nodal coordinates.  It returns closest-face id, Q4 natural
    coordinates, shape weights, and current normal, so master-side contact
    sensitivity can use the original quadrilateral face instead of a triangle
    split.
    """

    master_reference_nodes: np.ndarray
    master_quads: np.ndarray
    x_current: np.ndarray
    search_radius: float | None = None
    max_iterations: int = 12
    tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        reference = np.asarray(self.master_reference_nodes, dtype=float)
        if reference.ndim != 2 or reference.shape[1] != 3:
            raise ValueError("master_reference_nodes must have shape (n_nodes, 3)")
        quads = np.asarray(self.master_quads, dtype=np.int64)
        if quads.ndim != 2 or quads.shape[1] != 4:
            raise ValueError("master_quads must have shape (n_faces, 4)")
        if np.any(quads < 0) or (quads.size and int(quads.max()) >= reference.shape[0]):
            raise ValueError("master_quads reference nodes outside master_reference_nodes")
        current = np.asarray(self.x_current, dtype=float)
        if current.shape != reference.shape:
            raise ValueError("x_current must match master_reference_nodes")
        if self.search_radius is not None and float(self.search_radius) < 0.0:
            raise ValueError("search_radius must be non-negative")
        if int(self.max_iterations) <= 0:
            raise ValueError("max_iterations must be positive")
        if float(self.tolerance) < 0.0:
            raise ValueError("tolerance must be non-negative")
        self.master_reference_nodes = reference
        self.master_quads = quads
        self.x_current = current

    def refit(self, x_current: np.ndarray) -> None:
        current = np.asarray(x_current, dtype=float)
        if current.shape != self.master_reference_nodes.shape:
            raise ValueError("x_current must match master_reference_nodes")
        self.x_current = current

    def query(
        self,
        point: np.ndarray,
        *,
        preferred_face_ids: np.ndarray | list[int] | tuple[int, ...] | None = None,
    ) -> QuadrilateralClosestFeaturePayload:
        x = _as_point(point, "point")
        candidates = self._candidate_ids(x)
        best: QuadrilateralClosestFeaturePayload | None = None
        best_dist2 = np.inf
        preferred: list[int] = []
        if preferred_face_ids is not None:
            n_faces = int(self.master_quads.shape[0])
            seen: set[int] = set()
            for raw in np.asarray(preferred_face_ids, dtype=np.int64).reshape(-1):
                face_id = int(raw)
                if 0 <= face_id < n_faces and face_id not in seen:
                    preferred.append(face_id)
                    seen.add(face_id)
        candidate_set = {int(face_id) for face_id in np.asarray(candidates, dtype=np.int64).reshape(-1)}
        for face_id in preferred:
            if face_id not in candidate_set:
                candidates = np.concatenate((np.asarray([face_id], dtype=np.int64), candidates))
                candidate_set.add(face_id)
        preferred_array = np.asarray(preferred, dtype=np.int64)
        if preferred_array.size:
            for candidate in preferred_array:
                payload = self._query_one(x, int(candidate), candidate_count=int(candidates.size))
                dist2 = float(np.dot(x - payload.closest_point, x - payload.closest_point))
                if dist2 < best_dist2:
                    best = payload
                    best_dist2 = dist2
        remaining = np.asarray([int(face_id) for face_id in candidates if int(face_id) not in set(preferred)], dtype=np.int64)
        bounds = self._aabb_distance_squared(x, remaining) if remaining.size else np.empty(0, dtype=float)
        order = np.argsort(bounds, kind="stable") if bounds.size else np.empty(0, dtype=np.int64)
        ordered = remaining[order] if order.size else remaining
        ordered_bounds = bounds[order] if order.size else bounds
        for candidate, bound2 in zip(ordered, ordered_bounds, strict=True):
            if best is not None and float(bound2) > best_dist2 + 1.0e-14:
                break
            payload = self._query_one(x, int(candidate), candidate_count=int(candidates.size))
            dist2 = float(np.dot(x - payload.closest_point, x - payload.closest_point))
            if dist2 < best_dist2:
                best = payload
                best_dist2 = dist2
        if best is None:
            raise RuntimeError("no quadrilateral master candidates were available")
        return best

    def _candidate_ids(self, point: np.ndarray) -> np.ndarray:
        all_ids = np.arange(self.master_quads.shape[0], dtype=np.int64)
        if all_ids.size == 0:
            return all_ids
        if self.search_radius is None:
            return all_ids
        radius = float(self.search_radius)
        dist2 = self._aabb_distance_squared(point, all_ids)
        ids = all_ids[dist2 <= radius * radius]
        if ids.size == 0:
            ids = np.asarray([int(all_ids[int(np.argmin(dist2))])], dtype=np.int64)
        return ids

    def _aabb_distance_squared(self, point: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
        quads = self.x_current[self.master_quads[np.asarray(candidate_ids, dtype=np.int64)]]
        lower = quads.min(axis=1)
        upper = quads.max(axis=1)
        lower_delta = np.maximum(lower - point, 0.0)
        upper_delta = np.maximum(point - upper, 0.0)
        return np.sum((lower_delta + upper_delta) ** 2, axis=1)

    def _query_one(self, point: np.ndarray, face_id: int, *, candidate_count: int) -> QuadrilateralClosestFeaturePayload:
        quad_ids = self.master_quads[int(face_id)]
        quad = self.x_current[quad_ids]
        xi_eta = _q4_initial_natural_coordinates(point, quad)
        converged = False
        iterations = 0
        for iterations in range(1, int(self.max_iterations) + 1):
            shape = _q4_shape_functions(float(xi_eta[0]), float(xi_eta[1]))
            dshape = _q4_shape_derivatives(float(xi_eta[0]), float(xi_eta[1]))
            closest = shape @ quad
            residual = closest - point
            tangent_xi = dshape[:, 0] @ quad
            tangent_eta = dshape[:, 1] @ quad
            gradient = np.asarray([float(tangent_xi @ residual), float(tangent_eta @ residual)], dtype=float)
            if float(np.linalg.norm(gradient)) <= float(self.tolerance):
                converged = True
                break
            hessian = np.asarray(
                [
                    [float(tangent_xi @ tangent_xi), float(tangent_xi @ tangent_eta)],
                    [float(tangent_eta @ tangent_xi), float(tangent_eta @ tangent_eta)],
                ],
                dtype=float,
            )
            try:
                step = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError:
                step, *_ = np.linalg.lstsq(hessian, -gradient, rcond=None)
            next_xi_eta = np.clip(xi_eta + step, -1.0, 1.0)
            if float(np.linalg.norm(next_xi_eta - xi_eta)) <= float(self.tolerance):
                xi_eta = next_xi_eta
                converged = True
                break
            xi_eta = next_xi_eta
        shape = _q4_shape_functions(float(xi_eta[0]), float(xi_eta[1]))
        dshape = _q4_shape_derivatives(float(xi_eta[0]), float(xi_eta[1]))
        closest = shape @ quad
        tangent_xi = dshape[:, 0] @ quad
        tangent_eta = dshape[:, 1] @ quad
        normal = np.cross(tangent_xi, tangent_eta)
        norm = float(np.linalg.norm(normal))
        if norm <= 0.0:
            raise ValueError("current Q4 master face has zero area")
        normal = normal / norm
        gap = float((point - closest) @ normal)
        return QuadrilateralClosestFeaturePayload(
            gap=gap,
            normal=normal,
            closest_point=closest,
            face_id=int(face_id),
            natural_coordinates=xi_eta.copy(),
            master_node_ids=quad_ids.copy(),
            master_weights=shape.copy(),
            iterations=int(iterations),
            converged=bool(converged),
            candidates_evaluated=int(candidate_count),
        )


@dataclass(slots=True)
class LagrangianSDFSurfaceContactGeometry:
    """TET4 triangular surface quadrature against a Lagrangian-SDF master.

    The geometry consumes a concatenated two-body current coordinate array.  The
    slave samples are evaluated on triangular boundary faces; master-side
    sensitivity is supplied by the closest-feature payload returned by
    :class:`LagrangianSDFContactOracle`.
    """

    slave_faces: np.ndarray
    master_material: MaterialSDF
    master_reference_nodes: np.ndarray
    pressure_stiffness: float
    slave_node_offset: int = 0
    master_node_offset: int = 0
    quadrature: str = "tri3"
    search_radius: float | None = None
    patch_cell_size: float | None = None
    compiled_batch_projection: bool = False
    clip_to_master_footprint: bool = False
    secondary_tracking_rings: int = 2
    secondary_path_tracking: bool = False
    secondary_line_distance_limit: float | None = None
    secondary_line_hard_distance_limit: float | None = None
    _oracle: LagrangianSDFContactOracle = field(init=False, repr=False)
    _secondary_face_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _secondary_barycentric_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _secondary_region_face_cache: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _secondary_region_barycentric_cache: dict[int, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _master_face_tracking_neighborhoods: tuple[np.ndarray, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        faces = np.asarray(self.slave_faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("slave_faces must have shape (n_faces, 3)")
        if np.any(faces < 0):
            raise ValueError("slave_faces cannot contain negative node ids")
        master_nodes = np.asarray(self.master_reference_nodes, dtype=float)
        if master_nodes.ndim != 2 or master_nodes.shape[1] != 3:
            raise ValueError("master_reference_nodes must have shape (n_nodes, 3)")
        if float(self.pressure_stiffness) <= 0.0:
            raise ValueError("pressure_stiffness must be positive")
        if self.quadrature not in {"centroid", "tri3"}:
            raise ValueError("quadrature must be 'centroid' or 'tri3'")
        if int(self.secondary_tracking_rings) < 0:
            raise ValueError("secondary_tracking_rings must be non-negative")
        if self.secondary_line_distance_limit is not None and float(self.secondary_line_distance_limit) < 0.0:
            raise ValueError("secondary_line_distance_limit must be non-negative")
        if self.secondary_line_hard_distance_limit is not None and float(self.secondary_line_hard_distance_limit) < 0.0:
            raise ValueError("secondary_line_hard_distance_limit must be non-negative")
        self.slave_faces = faces
        self.master_reference_nodes = master_nodes
        self._master_face_tracking_neighborhoods = _triangle_face_neighborhoods(
            self.master_material.boundary_faces,
            rings=int(self.secondary_tracking_rings),
        )
        self._oracle = LagrangianSDFContactOracle(
            self.master_material,
            master_nodes.copy(),
            search_radius=self.search_radius,
            patch_cell_size=self.patch_cell_size,
        )

    def clear_secondary_contact_tracking(self) -> None:
        """Clear secondary-surface contact tracking hints."""

        self._secondary_face_cache = None
        self._secondary_barycentric_cache = None
        self._secondary_region_face_cache.clear()
        self._secondary_region_barycentric_cache.clear()

    def _secondary_cache_size(self) -> int:
        return int(self.slave_faces.shape[0]) * (3 if self.quadrature == "tri3" else 1)

    def _ensure_secondary_face_cache(self) -> np.ndarray:
        size = self._secondary_cache_size()
        if self._secondary_face_cache is None or self._secondary_face_cache.shape != (size,):
            self._secondary_face_cache = np.full(size, -1, dtype=np.int64)
        return self._secondary_face_cache

    def _ensure_secondary_barycentric_cache(self) -> np.ndarray:
        size = self._secondary_cache_size()
        if self._secondary_barycentric_cache is None or self._secondary_barycentric_cache.shape != (size, 3):
            self._secondary_barycentric_cache = np.full((size, 3), np.nan, dtype=float)
        return self._secondary_barycentric_cache

    def _secondary_cache_index_for_key(self, cache_key: object) -> int | None:
        if not isinstance(cache_key, tuple) or len(cache_key) < 3:
            return None
        if cache_key[1] != "secondary":
            return None
        face_id = int(cache_key[0])
        qp = int(cache_key[2])
        n_quadrature = 3 if self.quadrature == "tri3" else 1
        index = face_id * n_quadrature + qp
        if 0 <= index < self._secondary_cache_size():
            return int(index)
        return None

    def _merge_secondary_tracking_candidates(
        self,
        candidate_lists,
        cache_indices: np.ndarray,
        region_hint_faces: Sequence[Sequence[int]] | None = None,
    ) -> list[list[int]]:
        """Merge global/BVH candidates with accepted path-tracking hints.

        Candidate search remains conservative: global candidates and their
        neighborhoods are kept, while accepted-state path hints are tried first
        only when secondary path tracking is enabled.  A cached master face is
        still used when the current search tube returns no raw candidates; this
        is the path-tracking role needed for smooth sliding across accepted
        states.  If the cached face is invalid, the caller still receives the
        raw/global candidate list, possibly empty, and can fall back through its
        existing closest-feature query path.
        """

        cache = self._ensure_secondary_face_cache()
        merged: list[list[int]] = []
        n_faces = len(self._master_face_tracking_neighborhoods)
        region_lists = (
            [() for _ in range(len(candidate_lists))]
            if region_hint_faces is None
            else [tuple(int(face) for face in faces) for faces in region_hint_faces]
        )
        for raw, cache_index, region_faces in zip(
            candidate_lists,
            np.asarray(cache_indices, dtype=np.int64).reshape(-1),
            region_lists,
            strict=True,
        ):
            values: list[int] = []
            seen: set[int] = set()

            def add_face_with_neighbors(face_id: int) -> None:
                fid = int(face_id)
                if 0 <= fid < n_faces and fid not in seen:
                    values.append(fid)
                    seen.add(fid)
                if 0 <= fid < n_faces:
                    for neighbor in self._master_face_tracking_neighborhoods[fid]:
                        nid = int(neighbor)
                        if nid not in seen:
                            values.append(nid)
                            seen.add(nid)

            tracked = (
                int(cache[int(cache_index)])
                if bool(self.secondary_path_tracking) and 0 <= int(cache_index) < cache.shape[0]
                else -1
            )
            if 0 <= tracked < n_faces:
                add_face_with_neighbors(tracked)
            for face_id in region_faces:
                add_face_with_neighbors(int(face_id))
            for face_id in raw:
                add_face_with_neighbors(int(face_id))
            merged.append(values)
        return merged

    def _secondary_region_hint_faces(self, node_ids: np.ndarray, weights: np.ndarray) -> list[int]:
        """Return accepted master-face hints attached to secondary regions."""

        if not bool(self.secondary_path_tracking):
            return []
        nodes = np.asarray(node_ids, dtype=np.int64).reshape(-1)
        weight_values = np.asarray(weights, dtype=float).reshape(-1)
        if nodes.shape != weight_values.shape:
            return []
        order = np.argsort(-np.maximum(weight_values, 0.0))
        hints: list[int] = []
        seen: set[int] = set()
        n_faces = len(self._master_face_tracking_neighborhoods)
        for index in order:
            if float(weight_values[int(index)]) <= 0.0:
                continue
            face_id = self._secondary_region_face_cache.get(int(nodes[int(index)]), -1)
            if 0 <= int(face_id) < n_faces and int(face_id) not in seen:
                hints.append(int(face_id))
                seen.add(int(face_id))
        return hints

    def _secondary_region_hint_face_lists(self, sample_nodes: np.ndarray, sample_weights: np.ndarray) -> list[list[int]]:
        nodes = np.asarray(sample_nodes, dtype=np.int64)
        weights = np.asarray(sample_weights, dtype=float)
        if nodes.ndim != 2 or weights.shape != nodes.shape:
            return [[] for _ in range(nodes.shape[0] if nodes.ndim else 0)]
        return [self._secondary_region_hint_faces(nodes[row], weights[row]) for row in range(nodes.shape[0])]

    def _secondary_tracking_rank(
        self,
        face_id: int,
        *,
        cache_index: int | None,
    ) -> tuple[int, float]:
        if not bool(self.secondary_path_tracking) or cache_index is None:
            return 2, np.inf
        cache = self._ensure_secondary_face_cache()
        if not (0 <= int(cache_index) < cache.shape[0]):
            return 2, np.inf
        tracked = int(cache[int(cache_index)])
        if tracked < 0:
            return 2, np.inf
        if int(face_id) == tracked:
            return 0, 0.0
        if 0 <= tracked < len(self._master_face_tracking_neighborhoods):
            for local, neighbor in enumerate(self._master_face_tracking_neighborhoods[tracked]):
                if int(neighbor) == int(face_id):
                    return 1, float(local + 1)
        return 2, np.inf

    def _secondary_tracking_key(
        self,
        *,
        face_id: int,
        barycentric: np.ndarray,
        abs_gap: float,
        cache_index: int | None,
        region_nodes: np.ndarray | None = None,
        region_weights: np.ndarray | None = None,
    ) -> tuple[float, float, float, float]:
        if not bool(self.secondary_path_tracking):
            return 1.0, float(abs_gap), 0.0, 0.0
        candidates: list[tuple[int, float, float]] = []
        if cache_index is not None:
            rank, neighbor_order = self._secondary_tracking_rank(int(face_id), cache_index=cache_index)
            if rank < 2:
                bary_distance = self._secondary_sample_barycentric_distance(
                    barycentric,
                    cache_index=int(cache_index),
                )
                candidates.append((int(rank), float(neighbor_order), float(bary_distance)))
        if region_nodes is not None and region_weights is not None:
            region_rank = self._secondary_region_tracking_rank(
                int(face_id),
                barycentric=np.asarray(barycentric, dtype=float).reshape(3),
                node_ids=np.asarray(region_nodes, dtype=np.int64).reshape(-1),
                weights=np.asarray(region_weights, dtype=float).reshape(-1),
            )
            if region_rank is not None:
                candidates.append(region_rank)
        if not candidates:
            return 1.0, float(abs_gap), 0.0, 0.0
        rank, neighbor_order, bary_distance = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        return 0.0, float(rank), float(neighbor_order), float(bary_distance)

    def _secondary_sample_barycentric_distance(self, barycentric: np.ndarray, *, cache_index: int) -> float:
        bary_cache = self._ensure_secondary_barycentric_cache()
        if 0 <= int(cache_index) < bary_cache.shape[0]:
            previous = bary_cache[int(cache_index)]
            if np.all(np.isfinite(previous)):
                return float(np.linalg.norm(np.asarray(barycentric, dtype=float).reshape(3) - previous))
        return 0.0

    def _secondary_region_tracking_rank(
        self,
        face_id: int,
        *,
        barycentric: np.ndarray,
        node_ids: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[int, float, float] | None:
        if node_ids.shape != weights.shape:
            return None
        n_faces = len(self._master_face_tracking_neighborhoods)
        best: tuple[int, float, float] | None = None
        order = np.argsort(-np.maximum(weights, 0.0))
        for local_order, index in enumerate(order):
            if float(weights[int(index)]) <= 0.0:
                continue
            region_id = int(node_ids[int(index)])
            tracked = int(self._secondary_region_face_cache.get(region_id, -1))
            if not (0 <= tracked < n_faces):
                continue
            rank = 2
            neighbor_order = np.inf
            if int(face_id) == tracked:
                rank = 0
                neighbor_order = 0.0
            else:
                for neighbor_index, neighbor in enumerate(self._master_face_tracking_neighborhoods[tracked]):
                    if int(neighbor) == int(face_id):
                        rank = 1
                        neighbor_order = float(neighbor_index + 1)
                        break
            if rank >= 2:
                continue
            previous = self._secondary_region_barycentric_cache.get(region_id)
            bary_distance = 0.0
            if previous is not None and np.all(np.isfinite(previous)):
                bary_distance = float(np.linalg.norm(np.asarray(barycentric, dtype=float).reshape(3) - previous))
            candidate = (int(rank), float(neighbor_order + 1.0e-6 * int(local_order)), float(bary_distance))
            if best is None or candidate < best:
                best = candidate
        return best

    def _secondary_region_tracking_flags(
        self,
        face_ids: np.ndarray,
        barycentric: np.ndarray,
        sample_nodes: np.ndarray,
        sample_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        face_values = np.asarray(face_ids, dtype=np.int64).reshape(-1)
        bary_values = np.asarray(barycentric, dtype=float)
        nodes = np.asarray(sample_nodes, dtype=np.int64)
        weights = np.asarray(sample_weights, dtype=float)
        hits = np.zeros(face_values.shape, dtype=bool)
        matches = np.zeros(face_values.shape, dtype=bool)
        distances = np.full(face_values.shape, np.nan, dtype=float)
        if nodes.ndim != 2 or weights.shape != nodes.shape or bary_values.shape[0] != face_values.size:
            return hits, matches, distances
        for row, face_id in enumerate(face_values):
            best_distance = np.inf
            for node, weight in zip(nodes[row], weights[row], strict=True):
                if float(weight) <= 0.0:
                    continue
                previous_face = int(self._secondary_region_face_cache.get(int(node), -1))
                if previous_face < 0:
                    continue
                hits[row] = True
                if previous_face != int(face_id):
                    continue
                matches[row] = True
                previous_bary = self._secondary_region_barycentric_cache.get(int(node))
                if previous_bary is not None and np.all(np.isfinite(previous_bary)):
                    best_distance = min(
                        best_distance,
                        float(np.linalg.norm(np.asarray(bary_values[row], dtype=float).reshape(3) - previous_bary)),
                    )
            if np.isfinite(best_distance):
                distances[row] = float(best_distance)
        return hits, matches, distances

    def _update_secondary_tracking_cache(
        self,
        *,
        cache_index: int | None,
        face_id: int,
        barycentric: np.ndarray,
    ) -> None:
        if cache_index is None or int(face_id) < 0:
            return
        cache = self._ensure_secondary_face_cache()
        if not (0 <= int(cache_index) < cache.shape[0]):
            return
        cache[int(cache_index)] = int(face_id)
        if bool(self.secondary_path_tracking):
            bary_cache = self._ensure_secondary_barycentric_cache()
            bary = np.asarray(barycentric, dtype=float).reshape(3)
            total = float(np.sum(bary))
            if total > 0.0:
                bary = bary / total
            bary_cache[int(cache_index)] = bary

    def commit_secondary_tracking_from_sample_arrays(self, sample_arrays: dict[str, np.ndarray]) -> int:
        """Commit secondary path-tracking payload from an accepted query.

        Trial contact queries often run under a snapshot/restore guard so a
        failed Newton iteration or cutback cannot mutate the accepted tracking
        state.  When such a trial is accepted, this method commits the exact
        closest-feature payload carried by that trial without re-running the
        projection query.
        """

        if not bool(self.secondary_path_tracking):
            return 0
        if "master_face_ids" not in sample_arrays:
            return 0
        committed_regions = self._commit_secondary_region_tracking_from_sample_arrays(sample_arrays)
        if "secondary_cache_indices" not in sample_arrays:
            return int(committed_regions)
        cache_indices = np.asarray(sample_arrays["secondary_cache_indices"], dtype=np.int64).reshape(-1)
        face_ids = np.asarray(sample_arrays["master_face_ids"], dtype=np.int64).reshape(-1)
        bary = sample_arrays.get("master_barycentric", sample_arrays.get("master_weights"))
        if bary is None:
            return int(committed_regions)
        barycentric = np.asarray(bary, dtype=float)
        if barycentric.ndim != 2 or barycentric.shape[1] != 3:
            return int(committed_regions)
        count = min(cache_indices.size, face_ids.size, barycentric.shape[0])
        if count == 0:
            return int(committed_regions)
        if not np.any(cache_indices[:count] >= 0):
            return int(committed_regions)
        face_cache = self._ensure_secondary_face_cache()
        bary_cache = self._ensure_secondary_barycentric_cache()
        committed = 0
        for index in range(count):
            cache_index = int(cache_indices[index])
            face_id = int(face_ids[index])
            if face_id < 0 or not (0 <= cache_index < face_cache.shape[0]):
                continue
            weights = np.asarray(barycentric[index], dtype=float).reshape(3)
            total = float(np.sum(weights))
            if total > 0.0:
                weights = weights / total
            face_cache[cache_index] = face_id
            bary_cache[cache_index] = weights
            committed += 1
        return int(committed if committed else committed_regions)

    def _commit_secondary_region_tracking_from_sample_arrays(self, sample_arrays: dict[str, np.ndarray]) -> int:
        if not bool(self.secondary_path_tracking):
            return 0
        if "master_face_ids" not in sample_arrays:
            return 0
        face_ids = np.asarray(sample_arrays["master_face_ids"], dtype=np.int64).reshape(-1)
        bary = sample_arrays.get("master_barycentric", sample_arrays.get("master_weights"))
        if bary is None:
            return 0
        barycentric = np.asarray(bary, dtype=float)
        if barycentric.ndim != 2 or barycentric.shape[1] != 3:
            return 0
        if "secondary_node_ids" in sample_arrays:
            region_ids = np.asarray(sample_arrays["secondary_node_ids"], dtype=np.int64).reshape(-1)
            weights = np.ones((region_ids.size, 1), dtype=float)
            node_rows = region_ids.reshape((-1, 1))
        else:
            node_rows = np.asarray(sample_arrays.get("sample_node_ids", np.empty((0, 0))), dtype=np.int64)
            weights = np.asarray(sample_arrays.get("sample_weights", np.empty((0, 0))), dtype=float)
            if node_rows.ndim != 2 or weights.shape != node_rows.shape:
                return 0
        count = min(face_ids.size, barycentric.shape[0], node_rows.shape[0])
        committed = 0
        for row in range(count):
            face_id = int(face_ids[row])
            if face_id < 0:
                continue
            weights_row = np.asarray(barycentric[row], dtype=float).reshape(3)
            total = float(np.sum(weights_row))
            if total > 0.0:
                weights_row = weights_row / total
            for node, weight in zip(node_rows[row], weights[row], strict=True):
                if int(node) < 0 or float(weight) <= 0.0:
                    continue
                self._secondary_region_face_cache[int(node)] = int(face_id)
                self._secondary_region_barycentric_cache[int(node)] = weights_row.copy()
                committed += 1
        return int(committed)

    def samples(self, x_current: np.ndarray):
        if bool(self.clip_to_master_footprint):
            yield from self._clipped_samples(x_current)
            return
        arrays = self.sample_arrays(x_current)
        if arrays is not None:
            for idx in range(arrays["gaps"].shape[0]):
                yield ContactSample(
                    node_ids=arrays["sample_node_ids"][idx].copy(),
                    shape_weights=arrays["sample_weights"][idx].copy(),
                    gap=float(arrays["gaps"][idx]),
                    normal=arrays["normals"][idx].copy(),
                    area=float(arrays["areas"][idx]),
                    stiffness=float(self.pressure_stiffness),
                    master_node_ids=arrays["master_node_ids"][idx].copy(),
                    master_shape_weights=arrays["master_weights"][idx].copy(),
                )
            return

        X, _master_x, master_tree, master_max_radius, barycentric, weight_scale = self._prepare_sampling(x_current)
        master_start = int(self.master_node_offset)
        for face_id, face in enumerate(self.slave_faces):
            global_face = face + int(self.slave_node_offset)
            tri = X[global_face]
            area = _triangle_area(tri)
            if area <= 0.0:
                continue
            if master_tree is not None and _slave_face_outside_master_tube(tri, master_tree, master_max_radius, float(self.search_radius)):
                continue
            for qp, (weights, scale) in enumerate(zip(barycentric, weight_scale, strict=True)):
                point = weights @ tri
                query = self._oracle.query(point, cache_key=(int(face_id), int(qp)))
                yield ContactSample(
                    node_ids=global_face.copy(),
                    shape_weights=weights.copy(),
                    gap=float(query.gap),
                    normal=query.normal.copy(),
                    area=float(area * scale),
                    stiffness=float(self.pressure_stiffness),
                    master_node_ids=query.master_node_ids.astype(np.int64) + master_start,
                    master_shape_weights=query.master_weights.copy(),
                )

    def normal_compatible_samples(self, x_current: np.ndarray, *, dot_threshold: float = 0.0):
        """Yield samples whose closest feature is selected from opposing faces.

        This is different from post-filtering samples after a closest-feature
        query.  The query itself searches the candidate set for the nearest
        master face compatible with the current slave face normal, which is the
        surface-to-surface contact semantics needed to avoid dropping valid
        constraints on curved faceted surfaces.
        """

        arrays = self.normal_compatible_sample_arrays(x_current, dot_threshold=dot_threshold)
        if arrays is not None:
            for idx in range(arrays["gaps"].shape[0]):
                yield ContactSample(
                    node_ids=arrays["sample_node_ids"][idx].copy(),
                    shape_weights=arrays["sample_weights"][idx].copy(),
                    gap=float(arrays["gaps"][idx]),
                    normal=arrays["normals"][idx].copy(),
                    area=float(arrays["areas"][idx]),
                    stiffness=float(self.pressure_stiffness),
                    master_node_ids=arrays["master_node_ids"][idx].copy(),
                    master_shape_weights=arrays["master_weights"][idx].copy(),
                )
            return

        X, _master_x, master_tree, master_max_radius, barycentric, weight_scale = self._prepare_sampling(x_current)
        master_start = int(self.master_node_offset)
        for face_id, face in enumerate(self.slave_faces):
            global_face = face + int(self.slave_node_offset)
            tri = X[global_face]
            area = _triangle_area(tri)
            if area <= 0.0:
                continue
            if master_tree is not None and _slave_face_outside_master_tube(tri, master_tree, master_max_radius, float(self.search_radius)):
                continue
            slave_normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            slave_norm = float(np.linalg.norm(slave_normal))
            if slave_norm <= 0.0:
                continue
            slave_normal /= slave_norm
            for qp, (weights, scale) in enumerate(zip(barycentric, weight_scale, strict=True)):
                point = weights @ tri
                query = self._oracle.query_normal_compatible(
                    point,
                    slave_normal,
                    cache_key=(int(face_id), "compatible", int(qp)),
                    dot_threshold=float(dot_threshold),
                )
                yield ContactSample(
                    node_ids=global_face.copy(),
                    shape_weights=weights.copy(),
                    gap=float(query.gap),
                    normal=query.normal.copy(),
                    area=float(area * scale),
                    stiffness=float(self.pressure_stiffness),
                    master_node_ids=query.master_node_ids.astype(np.int64) + master_start,
                    master_shape_weights=query.master_weights.copy(),
                )

    def _clipped_samples(self, x_current: np.ndarray):
        X, master_x, master_tree, master_max_radius, _barycentric, _weight_scale = self._prepare_sampling(x_current)
        master_faces = self.master_material.boundary_faces
        master_start = int(self.master_node_offset)
        for face_id, face in enumerate(self.slave_faces):
            global_face = face + int(self.slave_node_offset)
            tri = X[global_face]
            slave_plane = _triangle_plane(tri)
            if slave_plane is None:
                continue
            slave_origin, slave_normal = slave_plane
            if master_tree is None:
                candidate_ids = np.arange(master_faces.shape[0], dtype=np.int64)
            else:
                slave_centroid = np.mean(tri, axis=0)
                slave_radius = float(np.max(np.linalg.norm(tri - slave_centroid, axis=1)))
                radius = float(self.search_radius) + slave_radius + float(master_max_radius)
                candidate_ids = np.asarray(master_tree.query_ball_point(slave_centroid, radius, return_sorted=False), dtype=np.int64)
            for master_face_id in candidate_ids.reshape(-1):
                master_nodes = master_faces[int(master_face_id)]
                master_tri = master_x[master_nodes]
                basis = _triangle_tangent_basis(master_tri)
                if basis is None:
                    continue
                origin, e1, e2, master_normal = basis
                normal_alignment = abs(float(slave_normal @ master_normal))
                if normal_alignment <= 1.0e-12:
                    continue
                slave_polygon = _ensure_counterclockwise(_project_points_2d(tri, origin, e1, e2))
                master_polygon = _ensure_counterclockwise(_project_points_2d(master_tri, origin, e1, e2))
                overlap = _convex_polygon_clip_2d(slave_polygon, master_polygon)
                if overlap.shape[0] < 3:
                    continue
                overlap = _ensure_counterclockwise(overlap)
                centroid = np.mean(overlap, axis=0)
                for i in range(overlap.shape[0]):
                    a = overlap[i]
                    b = overlap[(i + 1) % overlap.shape[0]]
                    tri_area_projected = abs(_triangle_area_2d(centroid, a, b))
                    if tri_area_projected <= 1.0e-16:
                        continue
                    uv = (centroid + a + b) / 3.0
                    plane_point = origin + uv[0] * e1 + uv[1] * e2
                    line_distance = float(slave_normal @ (slave_origin - plane_point)) / float(slave_normal @ master_normal)
                    point = plane_point + line_distance * master_normal
                    slave_weights = _triangle_barycentric_unclipped(point, tri)
                    if slave_weights is None:
                        continue
                    slave_weights = np.clip(slave_weights, 0.0, 1.0)
                    slave_total = float(np.sum(slave_weights))
                    if slave_total <= 0.0:
                        continue
                    slave_weights = slave_weights / slave_total
                    query = self._oracle.query(point)
                    if int(query.face_id) != int(master_face_id):
                        continue
                    yield ContactSample(
                        node_ids=global_face.copy(),
                        shape_weights=slave_weights.copy(),
                        gap=float(query.gap),
                        normal=query.normal.copy(),
                        area=float(tri_area_projected / normal_alignment),
                        stiffness=float(self.pressure_stiffness),
                        master_node_ids=query.master_node_ids.astype(np.int64) + master_start,
                        master_shape_weights=query.master_weights.copy(),
                    )

    def secondary_normal_projection_samples(self, x_current: np.ndarray, *, dot_threshold: float = 0.0):
        """Yield samples projected along the current secondary-surface normal.

        Abaqus surface-to-surface contact defines the contact direction from an
        average normal of the secondary surface constraint region.  For a
        faceted triangular surface this method projects each slave quadrature
        point along its current face normal onto compatible master triangles.
        If no valid normal projection lands inside a master triangle, it falls
        back to the closest-feature payload so the broad-phase remains robust.
        """

        X, master_x, master_tree, master_max_radius, barycentric, weight_scale = self._prepare_sampling(x_current)
        master_faces = self.master_material.boundary_faces
        master_start = int(self.master_node_offset)
        global_slave_faces = self.slave_faces + int(self.slave_node_offset)
        slave_node_normals = _surface_nodal_normals(X, global_slave_faces)
        for face_id, face in enumerate(self.slave_faces):
            global_face = face + int(self.slave_node_offset)
            tri = X[global_face]
            area = _triangle_area(tri)
            if area <= 0.0:
                continue
            if master_tree is not None and _slave_face_outside_master_tube(tri, master_tree, master_max_radius, float(self.search_radius)):
                continue
            slave_normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            slave_norm = float(np.linalg.norm(slave_normal))
            if slave_norm <= 0.0:
                continue
            slave_normal /= slave_norm
            for qp, (weights, scale) in enumerate(zip(barycentric, weight_scale, strict=True)):
                point = weights @ tri
                averaged_normal = weights @ slave_node_normals[global_face]
                averaged_norm = float(np.linalg.norm(averaged_normal))
                if averaged_norm > 1.0e-30:
                    query_normal = averaged_normal / averaged_norm
                else:
                    query_normal = slave_normal
                payload = self._secondary_projection_query(
                    point,
                    query_normal,
                    master_x,
                    master_faces,
                    master_tree,
                    master_max_radius,
                    dot_threshold=float(dot_threshold),
                    cache_key=(int(face_id), "secondary", int(qp)),
                    region_nodes=global_face,
                    region_weights=weights,
                )
                yield ContactSample(
                    node_ids=global_face.copy(),
                    shape_weights=weights.copy(),
                    gap=float(payload["gap"]),
                    normal=np.asarray(payload["normal"], dtype=float).copy(),
                    area=float(area * scale),
                    stiffness=float(self.pressure_stiffness),
                    master_node_ids=np.asarray(payload["master_node_ids"], dtype=np.int64) + master_start,
                    master_shape_weights=np.asarray(payload["master_weights"], dtype=float).copy(),
                )

    def _secondary_projection_query(
        self,
        point: np.ndarray,
        slave_normal: np.ndarray,
        master_x: np.ndarray,
        master_faces: np.ndarray,
        master_tree: cKDTree | None,
        master_max_radius: float,
        *,
        dot_threshold: float,
        cache_key: object,
        region_nodes: np.ndarray | None = None,
        region_weights: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | float | int]:
        x = np.asarray(point, dtype=float).reshape(3)
        ns = np.asarray(slave_normal, dtype=float).reshape(3)
        ns_norm = float(np.linalg.norm(ns))
        if ns_norm <= 0.0:
            query = self._oracle.query(x, cache_key=cache_key)
            return {
                "gap": float(query.gap),
                "normal": query.normal,
                "master_node_ids": query.master_node_ids,
                "master_weights": query.master_weights,
                "master_face_id": int(query.face_id),
            }
        ns = ns / ns_norm
        if master_tree is not None:
            candidate_radius = float(self.search_radius) + float(master_max_radius) + 1.0e-14
            candidate_ids = np.asarray(master_tree.query_ball_point(x, candidate_radius, return_sorted=False), dtype=np.int64)
        else:
            candidate_ids = np.arange(master_faces.shape[0], dtype=np.int64)
        cache_index = self._secondary_cache_index_for_key(cache_key)
        if cache_index is not None:
            region_hint_faces = None
            if region_nodes is not None and region_weights is not None:
                region_hint_faces = [self._secondary_region_hint_faces(region_nodes, region_weights)]
            merged = self._merge_secondary_tracking_candidates(
                [candidate_ids],
                np.asarray([cache_index], dtype=np.int64),
                region_hint_faces=region_hint_faces,
            )
            candidate_ids = np.asarray(merged[0], dtype=np.int64)
        best: dict[str, np.ndarray | float | int] | None = None
        best_key: tuple[float, float, float, float] = (np.inf, np.inf, np.inf, np.inf)
        best_face_id = -1
        rejected_line: dict[str, np.ndarray | float | int] | None = None
        rejected_abs_gap = np.inf
        for face_id in candidate_ids.reshape(-1):
            nodes = master_faces[int(face_id)]
            master_tri = master_x[nodes]
            master_normal = np.cross(master_tri[1] - master_tri[0], master_tri[2] - master_tri[0])
            master_norm = float(np.linalg.norm(master_normal))
            if master_norm <= 0.0:
                continue
            master_normal /= master_norm
            if float(master_normal @ ns) > float(dot_threshold):
                continue
            contact_normal = -ns
            projection = float(master_normal @ contact_normal)
            if projection <= 1.0e-12:
                continue
            gap = float(master_normal @ (x - master_tri[0]) / projection)
            projected = x - gap * contact_normal
            bary = _triangle_barycentric_unclipped(projected, master_tri)
            if bary is None or np.min(bary) < -1.0e-10 or np.max(bary) > 1.0 + 1.0e-10:
                continue
            abs_gap = abs(gap)
            if (
                self.secondary_line_hard_distance_limit is not None
                and abs_gap > float(self.secondary_line_hard_distance_limit)
            ):
                closest_payload = self._global_closest_feature_payload(x, master_x, master_faces, cache_key=cache_key)
                if gap < -1.0e-14 and float(closest_payload["gap"]) <= 1.0e-14:
                    # Abaqus-style secondary-line tracking should not release
                    # a large overclosure when the closest current feature is
                    # also closed.  The hard tube only rejects remote line hits
                    # over an actually open nearest surface region.
                    pass
                else:
                    if abs_gap < rejected_abs_gap:
                        rejected_abs_gap = abs_gap
                        rejected_line = {
                            "gap": abs_gap,
                            "normal": contact_normal.copy(),
                            "master_node_ids": np.asarray(nodes, dtype=np.int64).copy(),
                            "master_weights": np.clip(bary, 0.0, 1.0),
                            "master_face_id": int(face_id),
                        }
                    continue
            if (
                self.secondary_line_distance_limit is not None
                and abs_gap > float(self.secondary_line_distance_limit)
            ):
                closest_payload = self._global_closest_feature_payload(x, master_x, master_faces, cache_key=cache_key)
                if gap < -1.0e-14 and float(closest_payload["gap"]) <= 1.0e-14:
                    # Large overclosure is physical when the closest feature is
                    # also closed; the distance limit only rejects remote line
                    # intersections over an actually open nearest feature.
                    pass
                else:
                    if abs_gap < rejected_abs_gap:
                        rejected_abs_gap = abs_gap
                        rejected_line = {
                            "gap": abs_gap,
                            "normal": contact_normal.copy(),
                            "master_node_ids": np.asarray(nodes, dtype=np.int64).copy(),
                            "master_weights": np.clip(bary, 0.0, 1.0),
                            "master_face_id": int(face_id),
                        }
                    continue
            key = self._secondary_tracking_key(
                face_id=int(face_id),
                barycentric=np.asarray(bary, dtype=float),
                abs_gap=float(abs_gap),
                cache_index=cache_index,
                region_nodes=region_nodes,
                region_weights=region_weights,
            )
            if key < best_key:
                best_key = key
                best = {
                    "gap": gap,
                    "normal": contact_normal.copy(),
                    "master_node_ids": np.asarray(nodes, dtype=np.int64).copy(),
                    "master_weights": np.clip(bary, 0.0, 1.0),
                    "master_face_id": int(face_id),
                }
                best_face_id = int(face_id)
        if best is not None:
            # A secondary-normal line hit is only a valid contact constraint when
            # the physical closest-feature clearance is also closed.  Otherwise a
            # normal ray can intersect a remote, back-facing facet and create a
            # compressive penalty over an actually open nearest surface region.
            if float(best["gap"]) < -1.0e-14:
                closest_payload = self._global_closest_feature_payload(x, master_x, master_faces, cache_key=cache_key)
                if float(closest_payload["gap"]) > 1.0e-14:
                    return {
                        "gap": float(closest_payload["gap"]),
                        "normal": np.asarray(closest_payload["normal"], dtype=float),
                        "master_node_ids": np.asarray(closest_payload["master_node_ids"], dtype=np.int64),
                        "master_weights": np.asarray(closest_payload["master_weights"], dtype=float),
                        "master_face_id": int(closest_payload.get("master_face_id", -1)),
                    }
            weights = np.asarray(best["master_weights"], dtype=float)
            total = float(np.sum(weights))
            if total > 0.0:
                best["master_weights"] = weights / total
            return best
        if rejected_line is not None:
            weights = np.asarray(rejected_line["master_weights"], dtype=float)
            total = float(np.sum(weights))
            if total > 0.0:
                rejected_line["master_weights"] = weights / total
            return rejected_line
        query = self._oracle.query(x, cache_key=cache_key)
        normal = -ns if float(query.normal @ ns) < 0.0 else ns
        return {
            "gap": float(query.gap),
            "normal": normal,
            "master_node_ids": query.master_node_ids,
            "master_weights": query.master_weights,
            "master_face_id": int(query.face_id),
        }

    def _global_closest_feature_payload(
        self,
        point: np.ndarray,
        master_x: np.ndarray,
        master_faces: np.ndarray,
        *,
        cache_key: object,
    ) -> dict[str, np.ndarray | float | int]:
        x = np.asarray(point, dtype=float).reshape(3)
        if bool(self.compiled_batch_projection) and _cpp_projection_available() and _cpp_closest_points_all_faces is not None:
            gaps, normals, face_ids, bary, _closest = _cpp_closest_points_all_faces(x.reshape(1, 3), master_x, master_faces)
            face_id = int(np.asarray(face_ids, dtype=np.int64)[0])
            if face_id >= 0:
                return {
                    "gap": float(np.asarray(gaps, dtype=float)[0]),
                    "normal": np.asarray(normals, dtype=float)[0],
                    "master_node_ids": np.asarray(master_faces[face_id], dtype=np.int64),
                    "master_weights": np.asarray(bary, dtype=float)[0],
                    "master_face_id": int(face_id),
                }
        query = self._oracle.query(x, cache_key=cache_key)
        return {
            "gap": float(query.gap),
            "normal": query.normal,
            "master_node_ids": query.master_node_ids,
            "master_weights": query.master_weights,
            "master_face_id": int(query.face_id),
        }

    def secondary_normal_projection_sample_arrays(
        self,
        x_current: np.ndarray,
        *,
        dot_threshold: float = 0.0,
    ) -> dict[str, np.ndarray] | None:
        """Return batched secondary-normal projection samples when C++ support exists.

        The scalar Python ``secondary_normal_projection_samples`` method is a
        reference implementation.  This method follows the same Abaqus-style
        secondary-surface normal direction, but performs the per-candidate line
        projection in the compiled indexed projection kernel.  Accepted
        path-tracking caches are read as hints only; callers must explicitly
        commit accepted arrays with
        :meth:`commit_secondary_tracking_from_sample_arrays`.
        """

        if not (
            bool(self.compiled_batch_projection)
            and _cpp_projection_available()
            and _cpp_closest_points_indexed_faces_secondary_normal is not None
            and _cpp_secondary_normal_indexed_faces_available()
            and self.search_radius is not None
        ):
            return None
        X, master_x, master_tree, master_max_radius, barycentric, weight_scale = self._prepare_sampling(
            x_current,
            refit_oracle=False,
        )
        if master_tree is None:
            return None
        global_faces = self.slave_faces + int(self.slave_node_offset)
        triangles = X[global_faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        face_areas = 0.5 * np.linalg.norm(cross, axis=1)
        keep = face_areas > 0.0
        face_normals = np.zeros_like(cross)
        face_normals[keep] = cross[keep] / np.maximum(np.linalg.norm(cross[keep], axis=1)[:, None], 1.0e-30)
        if np.any(keep):
            slave_centroids = np.mean(triangles, axis=1)
            slave_radii = np.max(np.linalg.norm(triangles - slave_centroids[:, None, :], axis=2), axis=1)
            nearest = np.asarray(master_tree.query(slave_centroids, k=1)[0], dtype=float)
            tube_radius = float(self.search_radius) + slave_radii + float(master_max_radius)
            keep &= nearest <= tube_radius + 1.0e-14
        if not np.any(keep):
            return _empty_sample_arrays()
        kept_faces = global_faces[keep]
        kept_triangles = triangles[keep]
        kept_areas = face_areas[keep]
        kept_normals = face_normals[keep]
        slave_node_normals = _surface_nodal_normals(X, global_faces)
        n_quadrature = barycentric.shape[0]
        point_array = np.einsum("qa,fad->fqd", barycentric, kept_triangles).reshape((-1, 3))
        sample_nodes = np.repeat(kept_faces, n_quadrature, axis=0)
        sample_weights = np.tile(barycentric, (kept_faces.shape[0], 1))
        sample_normals = np.einsum("qa,fad->fqd", barycentric, slave_node_normals[kept_faces]).reshape((-1, 3))
        sample_normal_norms = np.linalg.norm(sample_normals, axis=1)
        normal_keep = sample_normal_norms > 1.0e-30
        sample_normals[normal_keep] /= sample_normal_norms[normal_keep, None]
        if np.any(~normal_keep):
            sample_normals[~normal_keep] = np.repeat(kept_normals, n_quadrature, axis=0)[~normal_keep]
        areas = (kept_areas[:, None] * weight_scale[None, :]).reshape(-1)
        kept_face_indices = np.flatnonzero(keep)
        cache_indices = (kept_face_indices[:, None] * n_quadrature + np.arange(n_quadrature, dtype=np.int64)[None, :]).reshape(-1)
        candidate_radius = float(self.search_radius) + float(master_max_radius) + 1.0e-14
        raw_candidates = master_tree.query_ball_point(point_array, candidate_radius, return_sorted=False)
        region_hint_faces = self._secondary_region_hint_face_lists(sample_nodes, sample_weights)
        raw_candidates = self._merge_secondary_tracking_candidates(
            raw_candidates,
            cache_indices,
            region_hint_faces=region_hint_faces,
        )
        point_keep = np.fromiter((len(ids) > 0 for ids in raw_candidates), dtype=bool, count=len(raw_candidates))
        if not np.any(point_keep):
            return _empty_sample_arrays()
        point_array = point_array[point_keep]
        sample_nodes = sample_nodes[point_keep]
        sample_weights = sample_weights[point_keep]
        sample_normals = sample_normals[point_keep]
        areas = areas[point_keep]
        cache_indices = cache_indices[point_keep]
        kept_candidates = [raw_candidates[int(index)] for index in np.flatnonzero(point_keep)]
        candidate_offsets, candidate_face_ids = _flatten_candidate_lists(kept_candidates)
        gaps, normals, face_ids, master_bary, _closest = _cpp_closest_points_indexed_faces_secondary_normal(
            point_array,
            sample_normals,
            master_x,
            self.master_material.boundary_faces,
            candidate_offsets,
            candidate_face_ids,
            float(self.search_radius),
            float(dot_threshold),
        )
        face_ids = np.asarray(face_ids, dtype=np.int64)
        gaps = np.asarray(gaps, dtype=float)
        valid = (face_ids >= 0) & np.isfinite(gaps)
        if not np.any(valid):
            return _empty_sample_arrays()
        valid_cache_indices = cache_indices[valid]
        valid_points = point_array[valid]
        valid_sample_normals = sample_normals[valid]
        sample_nodes = sample_nodes[valid]
        sample_weights = sample_weights[valid]
        areas = areas[valid]
        gaps = gaps[valid]
        normals = np.asarray(normals, dtype=float)[valid]
        master_bary = np.asarray(master_bary, dtype=float)[valid]
        face_ids = face_ids[valid]
        min_projection = max(-float(dot_threshold), 0.0)
        if min_projection > 0.0:
            candidate_faces = self.master_material.boundary_faces[face_ids]
            master_triangles = master_x[candidate_faces]
            master_normals = np.cross(
                master_triangles[:, 1] - master_triangles[:, 0],
                master_triangles[:, 2] - master_triangles[:, 0],
            )
            master_norms = np.linalg.norm(master_normals, axis=1)
            good_normals = master_norms > 1.0e-30
            projections = np.full(face_ids.shape, -np.inf, dtype=float)
            if np.any(good_normals):
                unit_master = master_normals[good_normals] / master_norms[good_normals, None]
                projections[good_normals] = -np.einsum(
                    "ij,ij->i",
                    unit_master,
                    valid_sample_normals[good_normals],
                )
            projection_keep = projections >= min_projection - 1.0e-14
            if not np.any(projection_keep):
                return _empty_sample_arrays()
            valid_cache_indices = valid_cache_indices[projection_keep]
            valid_points = valid_points[projection_keep]
            valid_sample_normals = valid_sample_normals[projection_keep]
            sample_nodes = sample_nodes[projection_keep]
            sample_weights = sample_weights[projection_keep]
            areas = areas[projection_keep]
            gaps = gaps[projection_keep]
            normals = normals[projection_keep]
            master_bary = master_bary[projection_keep]
            face_ids = face_ids[projection_keep]
        if bool(self.secondary_path_tracking):
            cache = self._ensure_secondary_face_cache()
            n_quadrature = barycentric.shape[0]
            for local, cache_index in enumerate(valid_cache_indices):
                ci = int(cache_index)
                if not (0 <= ci < cache.shape[0]) or int(cache[ci]) < 0:
                    continue
                face_index = ci // int(n_quadrature)
                qp_index = ci % int(n_quadrature)
                payload = self._secondary_projection_query(
                    valid_points[int(local)],
                    valid_sample_normals[int(local)],
                    master_x,
                    self.master_material.boundary_faces,
                    master_tree,
                    master_max_radius,
                    dot_threshold=float(dot_threshold),
                    cache_key=(int(face_index), "secondary", int(qp_index)),
                    region_nodes=sample_nodes[int(local)],
                    region_weights=sample_weights[int(local)],
                )
                nodes = np.asarray(payload["master_node_ids"], dtype=np.int64)
                if nodes.shape != (3,):
                    continue
                gaps[int(local)] = float(payload["gap"])
                normals[int(local)] = np.asarray(payload["normal"], dtype=float).reshape(3)
                payload_face_id = int(payload.get("master_face_id", -1))
                if payload_face_id >= 0:
                    face_ids[int(local)] = payload_face_id
                else:
                    matching = np.flatnonzero(
                        np.all(self.master_material.boundary_faces == nodes.reshape(1, 3), axis=1)
                    )
                    if matching.size:
                        face_ids[int(local)] = int(matching[0])
                master_bary[int(local)] = np.asarray(payload["master_weights"], dtype=float).reshape(3)
        active = gaps < -1.0e-14
        if (
            np.any(active)
            and _cpp_projection_available()
            and _cpp_closest_points_all_faces is not None
        ):
            closest_gaps, closest_normals, closest_face_ids, closest_bary, _closest = _cpp_closest_points_all_faces(
                valid_points[active],
                master_x,
                self.master_material.boundary_faces,
            )
            closest_gaps = np.asarray(closest_gaps, dtype=float)
            closest_face_ids = np.asarray(closest_face_ids, dtype=np.int64)
            release = (closest_face_ids >= 0) & (closest_gaps > 1.0e-14)
            if np.any(release):
                active_indices = np.flatnonzero(active)
                release_indices = active_indices[release]
                gaps[release_indices] = closest_gaps[release]
                normals[release_indices] = np.asarray(closest_normals, dtype=float)[release]
                face_ids[release_indices] = closest_face_ids[release]
                master_bary[release_indices] = np.asarray(closest_bary, dtype=float)[release]
            if self.secondary_line_distance_limit is not None:
                outside = active & (np.abs(gaps) > float(self.secondary_line_distance_limit))
                if np.any(outside):
                    outside_active = np.flatnonzero(outside)
                    outside_local = np.flatnonzero(outside[active])
                    closed = (
                        (closest_face_ids[outside_local] >= 0)
                        & (closest_gaps[outside_local] <= 1.0e-14)
                    )
                    release_outside = outside_active[~closed]
                    if release_outside.size:
                        gaps[release_outside] = np.abs(gaps[release_outside])
            if self.secondary_line_hard_distance_limit is not None:
                outside_hard = active & (np.abs(gaps) > float(self.secondary_line_hard_distance_limit))
                if np.any(outside_hard):
                    outside_active = np.flatnonzero(outside_hard)
                    outside_local = np.flatnonzero(outside_hard[active])
                    closed = (
                        (closest_face_ids[outside_local] >= 0)
                        & (closest_gaps[outside_local] <= 1.0e-14)
                    )
                    release_outside = outside_active[~closed]
                    if release_outside.size:
                        gaps[release_outside] = np.abs(gaps[release_outside])
        cache = self._ensure_secondary_face_cache()
        previous_cache_face_ids = cache[valid_cache_indices].copy()
        sample_cache_hits = previous_cache_face_ids >= 0
        sample_cache_matches = sample_cache_hits & (previous_cache_face_ids == face_ids)
        tracking_barycentric_distances = np.full(face_ids.shape, np.nan, dtype=float)
        if bool(self.secondary_path_tracking):
            bary_cache = self._ensure_secondary_barycentric_cache()
            previous_barycentric = bary_cache[valid_cache_indices].copy()
            totals = np.sum(master_bary, axis=1)
            normalized = np.asarray(master_bary, dtype=float).copy()
            good = totals > 1.0e-30
            normalized[good] /= totals[good, None]
            previous_good = np.all(np.isfinite(previous_barycentric), axis=1)
            if np.any(previous_good):
                tracking_barycentric_distances[previous_good] = np.linalg.norm(
                    normalized[previous_good] - previous_barycentric[previous_good],
                    axis=1,
                )
        region_hits, region_matches, region_distances = self._secondary_region_tracking_flags(
            face_ids,
            master_bary,
            sample_nodes,
            sample_weights,
        )
        tracking_cache_hits = sample_cache_hits | region_hits
        tracking_cache_matches = sample_cache_matches | region_matches
        finite_region_distances = np.isfinite(region_distances)
        replace_distances = finite_region_distances & (
            ~np.isfinite(tracking_barycentric_distances) | (region_distances < tracking_barycentric_distances)
        )
        tracking_barycentric_distances[replace_distances] = region_distances[replace_distances]
        master_nodes = self.master_material.boundary_faces[face_ids] + int(self.master_node_offset)
        return {
            "sample_node_ids": np.asarray(sample_nodes, dtype=np.int64),
            "sample_weights": np.asarray(sample_weights, dtype=float),
            "gaps": gaps,
            "normals": normals,
            "areas": np.asarray(areas, dtype=float),
            "master_node_ids": np.asarray(master_nodes, dtype=np.int64),
            "master_weights": master_bary,
            "master_barycentric": master_bary,
            "master_face_ids": np.asarray(face_ids, dtype=np.int64),
            "secondary_cache_indices": np.asarray(valid_cache_indices, dtype=np.int64),
            "tracking_cache_hits": tracking_cache_hits.astype(bool, copy=False),
            "tracking_cache_matches": tracking_cache_matches.astype(bool, copy=False),
            "tracking_barycentric_distances": tracking_barycentric_distances,
        }

    def normal_compatible_sample_arrays(
        self,
        x_current: np.ndarray,
        *,
        dot_threshold: float = 0.0,
    ) -> dict[str, np.ndarray] | None:
        """Return batched normal-compatible sample arrays when C++ support exists."""

        if not (
            bool(self.compiled_batch_projection)
            and _cpp_projection_available()
            and _cpp_closest_points_indexed_faces_normal_compatible is not None
            and _cpp_normal_compatible_indexed_faces_available()
            and self.search_radius is not None
        ):
            return None
        X, master_x, master_tree, master_max_radius, barycentric, weight_scale = self._prepare_sampling(
            x_current,
            refit_oracle=False,
        )
        if master_tree is None:
            return None
        global_faces = self.slave_faces + int(self.slave_node_offset)
        triangles = X[global_faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        face_areas = 0.5 * np.linalg.norm(cross, axis=1)
        keep = face_areas > 0.0
        face_normals = np.zeros_like(cross)
        face_normals[keep] = cross[keep] / np.maximum(np.linalg.norm(cross[keep], axis=1)[:, None], 1.0e-30)
        if np.any(keep):
            slave_centroids = np.mean(triangles, axis=1)
            slave_radii = np.max(np.linalg.norm(triangles - slave_centroids[:, None, :], axis=2), axis=1)
            nearest = np.asarray(master_tree.query(slave_centroids, k=1)[0], dtype=float)
            tube_radius = float(self.search_radius) + slave_radii + float(master_max_radius)
            keep &= nearest <= tube_radius + 1.0e-14
        if not np.any(keep):
            return _empty_sample_arrays()
        kept_faces = global_faces[keep]
        kept_triangles = triangles[keep]
        kept_areas = face_areas[keep]
        kept_normals = face_normals[keep]
        n_quadrature = barycentric.shape[0]
        point_array = np.einsum("qa,fad->fqd", barycentric, kept_triangles).reshape((-1, 3))
        sample_nodes = np.repeat(kept_faces, n_quadrature, axis=0)
        sample_weights = np.tile(barycentric, (kept_faces.shape[0], 1))
        sample_normals = np.repeat(kept_normals, n_quadrature, axis=0)
        areas = (kept_areas[:, None] * weight_scale[None, :]).reshape(-1)
        candidate_radius = float(self.search_radius) + float(master_max_radius) + 1.0e-14
        raw_candidates = master_tree.query_ball_point(point_array, candidate_radius, return_sorted=False)
        point_keep = np.fromiter((len(ids) > 0 for ids in raw_candidates), dtype=bool, count=len(raw_candidates))
        if not np.any(point_keep):
            return _empty_sample_arrays()
        point_array = point_array[point_keep]
        sample_nodes = sample_nodes[point_keep]
        sample_weights = sample_weights[point_keep]
        sample_normals = sample_normals[point_keep]
        areas = areas[point_keep]
        kept_candidates = [raw_candidates[int(index)] for index in np.flatnonzero(point_keep)]
        candidate_offsets, candidate_face_ids = _flatten_candidate_lists(kept_candidates)
        gaps, normals, face_ids, master_bary, _closest = _cpp_closest_points_indexed_faces_normal_compatible(
            point_array,
            sample_normals,
            master_x,
            self.master_material.boundary_faces,
            candidate_offsets,
            candidate_face_ids,
            float(self.search_radius),
            float(dot_threshold),
        )
        face_ids = np.asarray(face_ids, dtype=np.int64)
        valid = face_ids >= 0
        if not np.any(valid):
            return _empty_sample_arrays()
        point_array = point_array[valid]
        sample_nodes = sample_nodes[valid]
        sample_weights = sample_weights[valid]
        areas = areas[valid]
        gaps = np.asarray(gaps, dtype=float)[valid]
        normals = np.asarray(normals, dtype=float)[valid]
        master_bary = np.asarray(master_bary, dtype=float)[valid]
        face_ids = face_ids[valid]
        master_nodes = self.master_material.boundary_faces[face_ids] + int(self.master_node_offset)
        return {
            "sample_node_ids": np.asarray(sample_nodes, dtype=np.int64),
            "sample_weights": np.asarray(sample_weights, dtype=float),
            "gaps": gaps,
            "normals": normals,
            "areas": np.asarray(areas, dtype=float),
            "master_node_ids": np.asarray(master_nodes, dtype=np.int64),
            "master_weights": master_bary,
            "master_face_ids": np.asarray(face_ids, dtype=np.int64),
        }

    def nodal_samples(self, x_current: np.ndarray):
        """Yield slave-corner contact samples with tributary face areas.

        Abaqus surface-to-surface output reports secondary-node overclosure and
        pressure fields.  For linear triangular surface facets, a theory-based
        secondary-node constraint region can be represented by evaluating the
        closest-feature gap at each slave face corner and assigning one third
        of the incident face area to that corner.  Adjacent face-corner samples
        can then be grouped by slave node without reducing the geometric query
        accuracy.
        """

        arrays = self.nodal_sample_arrays(x_current)
        if arrays is not None:
            for idx in range(arrays["gaps"].shape[0]):
                yield ContactSample(
                    node_ids=arrays["sample_node_ids"][idx].copy(),
                    shape_weights=arrays["sample_weights"][idx].copy(),
                    gap=float(arrays["gaps"][idx]),
                    normal=arrays["normals"][idx].copy(),
                    area=float(arrays["areas"][idx]),
                    stiffness=float(self.pressure_stiffness),
                    master_node_ids=arrays["master_node_ids"][idx].copy(),
                    master_shape_weights=arrays["master_weights"][idx].copy(),
                )
            return

        X, _master_x, master_tree, master_max_radius, _barycentric, _weight_scale = self._prepare_sampling(x_current)
        master_start = int(self.master_node_offset)
        for face_id, face in enumerate(self.slave_faces):
            global_face = face + int(self.slave_node_offset)
            tri = X[global_face]
            area = _triangle_area(tri)
            if area <= 0.0:
                continue
            if master_tree is not None and _slave_face_outside_master_tube(tri, master_tree, master_max_radius, float(self.search_radius)):
                continue
            tributary = float(area / 3.0)
            for local, node in enumerate(global_face):
                point = X[int(node)]
                query = self._oracle.query(point, cache_key=(int(face_id), "node", int(local)))
                weights = np.zeros(3, dtype=float)
                weights[int(local)] = 1.0
                yield ContactSample(
                    node_ids=global_face.copy(),
                    shape_weights=weights,
                    gap=float(query.gap),
                    normal=query.normal.copy(),
                    area=tributary,
                    stiffness=float(self.pressure_stiffness),
                    master_node_ids=query.master_node_ids.astype(np.int64) + master_start,
                    master_shape_weights=query.master_weights.copy(),
                )

    def nodal_sample_arrays(self, x_current: np.ndarray) -> dict[str, np.ndarray] | None:
        """Return batched slave-corner sample arrays when the compiled backend is available."""

        if not (bool(self.compiled_batch_projection) and _cpp_projection_available() and _cpp_closest_points_all_faces is not None):
            return None
        use_indexed_projection = (
            _cpp_closest_points_indexed_faces is not None
            and _cpp_indexed_faces_available()
            and self.search_radius is not None
        )
        needs_oracle_bvh = _cpp_closest_points_padded_aabb is not None and not use_indexed_projection
        X, master_x, master_tree, master_max_radius, _barycentric, _weight_scale = self._prepare_sampling(
            x_current,
            refit_oracle=needs_oracle_bvh,
        )
        global_faces = self.slave_faces + int(self.slave_node_offset)
        triangles = X[global_faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        face_areas = 0.5 * np.linalg.norm(cross, axis=1)
        keep = face_areas > 0.0
        if master_tree is not None and np.any(keep):
            slave_centroids = np.mean(triangles, axis=1)
            slave_radii = np.max(np.linalg.norm(triangles - slave_centroids[:, None, :], axis=2), axis=1)
            nearest = np.asarray(master_tree.query(slave_centroids, k=1)[0], dtype=float)
            tube_radius = float(self.search_radius) + slave_radii + float(master_max_radius)
            keep &= nearest <= tube_radius + 1.0e-14
        if not np.any(keep):
            return {
                "sample_node_ids": np.empty((0, 3), dtype=np.int64),
                "sample_weights": np.empty((0, 3), dtype=float),
                "gaps": np.empty(0, dtype=float),
                "normals": np.empty((0, 3), dtype=float),
                "areas": np.empty(0, dtype=float),
                "master_node_ids": np.empty((0, 3), dtype=np.int64),
                "master_weights": np.empty((0, 3), dtype=float),
                "master_face_ids": np.empty(0, dtype=np.int64),
            }
        kept_faces = global_faces[keep]
        kept_triangles = triangles[keep]
        kept_areas = face_areas[keep]
        point_array = kept_triangles.reshape((-1, 3))
        sample_nodes = np.repeat(kept_faces, 3, axis=0)
        sample_weights = np.tile(np.eye(3, dtype=float), (kept_faces.shape[0], 1))
        areas = np.repeat(kept_areas / 3.0, 3)
        candidate_offsets = None
        candidate_face_ids = None
        if master_tree is not None and point_array.size:
            candidate_radius = float(self.search_radius) + float(master_max_radius) + 1.0e-14
            if use_indexed_projection:
                raw_candidates = master_tree.query_ball_point(point_array, candidate_radius, return_sorted=False)
                point_keep = np.fromiter((len(ids) > 0 for ids in raw_candidates), dtype=bool, count=len(raw_candidates))
            else:
                point_nearest = np.asarray(master_tree.query(point_array, k=1)[0], dtype=float)
                point_keep = point_nearest <= candidate_radius
            if not np.any(point_keep):
                return {
                    "sample_node_ids": np.empty((0, 3), dtype=np.int64),
                    "sample_weights": np.empty((0, 3), dtype=float),
                    "gaps": np.empty(0, dtype=float),
                    "normals": np.empty((0, 3), dtype=float),
                    "areas": np.empty(0, dtype=float),
                    "master_node_ids": np.empty((0, 3), dtype=np.int64),
                    "master_weights": np.empty((0, 3), dtype=float),
                    "master_face_ids": np.empty(0, dtype=np.int64),
                }
            point_array = point_array[point_keep]
            sample_nodes = sample_nodes[point_keep]
            sample_weights = sample_weights[point_keep]
            areas = areas[point_keep]
            if use_indexed_projection:
                kept_candidates = [raw_candidates[int(index)] for index in np.flatnonzero(point_keep)]
                candidate_offsets, candidate_face_ids = _flatten_candidate_lists(kept_candidates)
        if candidate_offsets is not None and candidate_face_ids is not None and _cpp_closest_points_indexed_faces is not None:
            gaps, normals, face_ids, master_bary, _closest = _cpp_closest_points_indexed_faces(
                point_array,
                master_x,
                self.master_material.boundary_faces,
                candidate_offsets,
                candidate_face_ids,
                float(self.search_radius) if self.search_radius is not None else 0.0,
            )
        elif _cpp_closest_points_padded_aabb is not None:
            bvh = self._oracle.bvh
            gaps, normals, face_ids, master_bary, _closest = _cpp_closest_points_padded_aabb(
                point_array,
                master_x,
                self.master_material.boundary_faces,
                bvh.cell_aabb_min,
                bvh.cell_aabb_max,
                float(self.search_radius) if self.search_radius is not None else 0.0,
            )
        else:
            gaps, normals, face_ids, master_bary, _closest = _cpp_closest_points_all_faces(
                point_array,
                master_x,
                self.master_material.boundary_faces,
            )
        master_nodes = self.master_material.boundary_faces[np.asarray(face_ids, dtype=np.int64)] + int(self.master_node_offset)
        return {
            "sample_node_ids": np.asarray(sample_nodes, dtype=np.int64),
            "sample_weights": np.asarray(sample_weights, dtype=float),
            "gaps": np.asarray(gaps, dtype=float),
            "normals": np.asarray(normals, dtype=float),
            "areas": np.asarray(areas, dtype=float),
            "master_node_ids": np.asarray(master_nodes, dtype=np.int64),
            "master_weights": np.asarray(master_bary, dtype=float),
            "master_face_ids": np.asarray(face_ids, dtype=np.int64),
        }

    def sample_arrays(self, x_current: np.ndarray) -> dict[str, np.ndarray] | None:
        """Return batched sample arrays for compiled closest-feature queries.

        Accepted path-tracking caches are read as candidate hints and
        diagnostics only.  Trial queries must not mutate those caches; accepted
        arrays are committed explicitly by
        :meth:`commit_secondary_tracking_from_sample_arrays`.
        """

        if not (bool(self.compiled_batch_projection) and _cpp_projection_available() and _cpp_closest_points_all_faces is not None):
            return None
        use_indexed_projection = (
            _cpp_closest_points_indexed_faces is not None
            and _cpp_indexed_faces_available()
            and self.search_radius is not None
        )
        needs_oracle_bvh = _cpp_closest_points_padded_aabb is not None and not use_indexed_projection
        X, master_x, master_tree, master_max_radius, barycentric, weight_scale = self._prepare_sampling(
            x_current,
            refit_oracle=needs_oracle_bvh,
        )
        global_faces = self.slave_faces + int(self.slave_node_offset)
        triangles = X[global_faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        face_areas = 0.5 * np.linalg.norm(cross, axis=1)
        keep = face_areas > 0.0
        if master_tree is not None and np.any(keep):
            slave_centroids = np.mean(triangles, axis=1)
            slave_radii = np.max(np.linalg.norm(triangles - slave_centroids[:, None, :], axis=2), axis=1)
            nearest = np.asarray(master_tree.query(slave_centroids, k=1)[0], dtype=float)
            tube_radius = float(self.search_radius) + slave_radii + float(master_max_radius)
            keep &= nearest <= tube_radius + 1.0e-14
        if not np.any(keep):
            return {
                "sample_node_ids": np.empty((0, 3), dtype=np.int64),
                "sample_weights": np.empty((0, 3), dtype=float),
                "gaps": np.empty(0, dtype=float),
                "normals": np.empty((0, 3), dtype=float),
                "areas": np.empty(0, dtype=float),
                "master_node_ids": np.empty((0, 3), dtype=np.int64),
                "master_weights": np.empty((0, 3), dtype=float),
                "master_face_ids": np.empty(0, dtype=np.int64),
            }
        kept_faces = global_faces[keep]
        kept_triangles = triangles[keep]
        kept_areas = face_areas[keep]
        n_quadrature = barycentric.shape[0]
        point_array = np.einsum("qa,fad->fqd", barycentric, kept_triangles).reshape((-1, 3))
        sample_nodes = np.repeat(kept_faces, n_quadrature, axis=0)
        sample_weights = np.tile(barycentric, (kept_faces.shape[0], 1))
        areas = (kept_areas[:, None] * weight_scale[None, :]).reshape(-1)
        kept_face_indices = np.flatnonzero(keep)
        cache_indices = (kept_face_indices[:, None] * n_quadrature + np.arange(n_quadrature, dtype=np.int64)[None, :]).reshape(-1)
        candidate_offsets = None
        candidate_face_ids = None
        if master_tree is not None and point_array.size:
            candidate_radius = float(self.search_radius) + float(master_max_radius) + 1.0e-14
            use_indexed_candidates = use_indexed_projection
            if use_indexed_candidates:
                raw_candidates = master_tree.query_ball_point(point_array, candidate_radius, return_sorted=False)
                if bool(self.secondary_path_tracking):
                    raw_candidates = self._merge_secondary_tracking_candidates(raw_candidates, cache_indices)
                point_keep = np.fromiter((len(ids) > 0 for ids in raw_candidates), dtype=bool, count=len(raw_candidates))
            else:
                point_nearest = np.asarray(master_tree.query(point_array, k=1)[0], dtype=float)
                point_keep = point_nearest <= candidate_radius
            if not np.any(point_keep):
                return {
                    "sample_node_ids": np.empty((0, 3), dtype=np.int64),
                    "sample_weights": np.empty((0, 3), dtype=float),
                    "gaps": np.empty(0, dtype=float),
                    "normals": np.empty((0, 3), dtype=float),
                    "areas": np.empty(0, dtype=float),
                    "master_node_ids": np.empty((0, 3), dtype=np.int64),
                    "master_weights": np.empty((0, 3), dtype=float),
                    "master_face_ids": np.empty(0, dtype=np.int64),
                }
            point_array = point_array[point_keep]
            sample_nodes = sample_nodes[point_keep]
            sample_weights = sample_weights[point_keep]
            areas = areas[point_keep]
            cache_indices = cache_indices[point_keep]
            if use_indexed_candidates:
                kept_candidates = [raw_candidates[int(index)] for index in np.flatnonzero(point_keep)]
                candidate_offsets, candidate_face_ids = _flatten_candidate_lists(kept_candidates)
        if candidate_offsets is not None and candidate_face_ids is not None and _cpp_closest_points_indexed_faces is not None:
            gaps, normals, face_ids, master_bary, _closest = _cpp_closest_points_indexed_faces(
                point_array,
                master_x,
                self.master_material.boundary_faces,
                candidate_offsets,
                candidate_face_ids,
                float(self.search_radius) if self.search_radius is not None else 0.0,
            )
        elif _cpp_closest_points_padded_aabb is not None:
            bvh = self._oracle.bvh
            gaps, normals, face_ids, master_bary, _closest = _cpp_closest_points_padded_aabb(
                point_array,
                master_x,
                self.master_material.boundary_faces,
                bvh.cell_aabb_min,
                bvh.cell_aabb_max,
                float(self.search_radius) if self.search_radius is not None else 0.0,
            )
        else:
            gaps, normals, face_ids, master_bary, _closest = _cpp_closest_points_all_faces(
                point_array,
                master_x,
                self.master_material.boundary_faces,
            )
        face_ids = np.asarray(face_ids, dtype=np.int64)
        master_bary = np.asarray(master_bary, dtype=float)
        tracking_payload: dict[str, np.ndarray] = {}
        if bool(self.secondary_path_tracking):
            cache = self._ensure_secondary_face_cache()
            previous_cache_face_ids = cache[cache_indices].copy()
            tracking_cache_hits = previous_cache_face_ids >= 0
            tracking_cache_matches = tracking_cache_hits & (previous_cache_face_ids == face_ids)
            tracking_barycentric_distances = np.full(face_ids.shape, np.nan, dtype=float)
            bary_cache = self._ensure_secondary_barycentric_cache()
            previous_barycentric = bary_cache[cache_indices].copy()
            totals = np.sum(master_bary, axis=1)
            normalized = master_bary.copy()
            good = totals > 1.0e-30
            normalized[good] /= totals[good, None]
            previous_good = np.all(np.isfinite(previous_barycentric), axis=1)
            if np.any(previous_good):
                tracking_barycentric_distances[previous_good] = np.linalg.norm(
                    normalized[previous_good] - previous_barycentric[previous_good],
                    axis=1,
                )
            tracking_payload = {
                "secondary_cache_indices": np.asarray(cache_indices, dtype=np.int64),
                "tracking_cache_hits": tracking_cache_hits.astype(bool, copy=False),
                "tracking_cache_matches": tracking_cache_matches.astype(bool, copy=False),
                "tracking_barycentric_distances": tracking_barycentric_distances,
            }
        master_nodes = self.master_material.boundary_faces[face_ids] + int(self.master_node_offset)
        arrays = {
            "sample_node_ids": np.asarray(sample_nodes, dtype=np.int64),
            "sample_weights": np.asarray(sample_weights, dtype=float),
            "gaps": np.asarray(gaps, dtype=float),
            "normals": np.asarray(normals, dtype=float),
            "areas": np.asarray(areas, dtype=float),
            "master_node_ids": np.asarray(master_nodes, dtype=np.int64),
            "master_weights": master_bary,
            "master_barycentric": master_bary,
            "master_face_ids": face_ids,
        }
        arrays.update(tracking_payload)
        return arrays

    def _prepare_sampling(
        self,
        x_current: np.ndarray,
        *,
        refit_oracle: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, cKDTree | None, float, np.ndarray, np.ndarray]:
        X = np.asarray(x_current, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("x_current must have shape (n_nodes, 3)")
        master_start = int(self.master_node_offset)
        master_stop = master_start + self.master_reference_nodes.shape[0]
        if master_stop > X.shape[0]:
            raise ValueError("master_node_offset places master nodes outside x_current")
        master_x = X[master_start:master_stop]
        if bool(refit_oracle):
            self._oracle.refit(master_x)
        master_tree = None
        master_max_radius = 0.0
        if self.search_radius is not None:
            master_centroids, master_radii = _triangle_bounding_spheres(master_x[self.master_material.boundary_faces])
            if master_centroids.size:
                master_tree = cKDTree(master_centroids)
                master_max_radius = float(np.max(master_radii)) if master_radii.size else 0.0
        barycentric = np.asarray([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=float)
        weight_scale = np.asarray([1.0], dtype=float)
        if self.quadrature == "tri3":
            barycentric = _TRI3_BARY
            weight_scale = np.full(3, 1.0 / 3.0, dtype=float)
        return X, master_x, master_tree, master_max_radius, barycentric, weight_scale


@dataclass(slots=True)
class LagrangianSDFQuadrilateralSurfaceContactGeometry:
    """Quadrilateral slave-surface quadrature against a Lagrangian-SDF master.

    This is the C3D8/H8 surface-to-surface counterpart of
    :class:`LagrangianSDFSurfaceContactGeometry`.  Slave points are evaluated
    with tensor-product Gauss quadrature on Q4 faces; each point queries the
    master Lagrangian SDF oracle for gap, normal, and closest-feature payload.
    """

    slave_quads: np.ndarray
    master_material: MaterialSDF
    master_reference_nodes: np.ndarray
    pressure_stiffness: float
    slave_node_offset: int = 0
    master_node_offset: int = 0
    quadrature_order: int = 2
    search_radius: float | None = None
    patch_cell_size: float | None = None
    _oracle: LagrangianSDFContactOracle = field(init=False, repr=False)

    def __post_init__(self) -> None:
        quads = np.asarray(self.slave_quads, dtype=np.int64)
        if quads.ndim != 2 or quads.shape[1] != 4:
            raise ValueError("slave_quads must have shape (n_faces, 4)")
        if np.any(quads < 0):
            raise ValueError("slave_quads cannot contain negative node ids")
        master_nodes = np.asarray(self.master_reference_nodes, dtype=float)
        if master_nodes.ndim != 2 or master_nodes.shape[1] != 3:
            raise ValueError("master_reference_nodes must have shape (n_nodes, 3)")
        if float(self.pressure_stiffness) <= 0.0:
            raise ValueError("pressure_stiffness must be positive")
        if int(self.quadrature_order) not in {1, 2, 3}:
            raise ValueError("quadrature_order must be 1, 2, or 3")
        self.slave_quads = quads
        self.master_reference_nodes = master_nodes
        self._oracle = LagrangianSDFContactOracle(
            self.master_material,
            master_nodes.copy(),
            search_radius=self.search_radius,
            patch_cell_size=self.patch_cell_size,
        )

    def samples(self, x_current: np.ndarray):
        X = np.asarray(x_current, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("x_current must have shape (n_nodes, 3)")
        master_start = int(self.master_node_offset)
        master_stop = master_start + self.master_reference_nodes.shape[0]
        if master_stop > X.shape[0]:
            raise ValueError("master_node_offset places master nodes outside x_current")
        master_x = X[master_start:master_stop]
        self._oracle.refit(master_x)
        points, weights = np.polynomial.legendre.leggauss(int(self.quadrature_order))
        for face_id, quad in enumerate(self.slave_quads):
            global_quad = quad + int(self.slave_node_offset)
            qx = X[global_quad]
            for a, xi in enumerate(points):
                for b, eta in enumerate(points):
                    shape = _q4_shape_functions(float(xi), float(eta))
                    dshape = _q4_shape_derivatives(float(xi), float(eta))
                    tangent_xi = dshape[:, 0] @ qx
                    tangent_eta = dshape[:, 1] @ qx
                    jac = float(np.linalg.norm(np.cross(tangent_xi, tangent_eta)))
                    if jac <= 0.0:
                        continue
                    point = shape @ qx
                    query = self._oracle.query(point, cache_key=(int(face_id), int(a), int(b)))
                    yield ContactSample(
                        node_ids=global_quad.copy(),
                        shape_weights=shape.copy(),
                        gap=float(query.gap),
                        normal=query.normal.copy(),
                        area=float(jac * float(weights[a]) * float(weights[b])),
                        stiffness=float(self.pressure_stiffness),
                        master_node_ids=query.master_node_ids.astype(np.int64) + master_start,
                        master_shape_weights=query.master_weights.copy(),
                    )


@dataclass(slots=True)
class LagrangianSDFQ4MasterSurfaceContactGeometry:
    """Q4 slave quadrature against Q4 master closest-feature payloads."""

    slave_quads: np.ndarray
    master_quads: np.ndarray
    master_reference_nodes: np.ndarray
    pressure_stiffness: float
    slave_node_offset: int = 0
    master_node_offset: int = 0
    quadrature_order: int = 2
    search_radius: float | None = None
    clip_to_master_footprint: bool = False
    secondary_path_tracking: bool = False
    secondary_tracking_rings: int = 2
    _oracle: LagrangianQ4ClosestFeatureOracle = field(init=False, repr=False)
    _secondary_face_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _secondary_master_weight_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _master_face_tracking_neighborhoods: tuple[np.ndarray, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        slave = np.asarray(self.slave_quads, dtype=np.int64)
        if slave.ndim != 2 or slave.shape[1] != 4:
            raise ValueError("slave_quads must have shape (n_faces, 4)")
        if np.any(slave < 0):
            raise ValueError("slave_quads cannot contain negative node ids")
        master_nodes = np.asarray(self.master_reference_nodes, dtype=float)
        if master_nodes.ndim != 2 or master_nodes.shape[1] != 3:
            raise ValueError("master_reference_nodes must have shape (n_nodes, 3)")
        master = np.asarray(self.master_quads, dtype=np.int64)
        if master.ndim != 2 or master.shape[1] != 4:
            raise ValueError("master_quads must have shape (n_faces, 4)")
        if np.any(master < 0) or (master.size and int(master.max()) >= master_nodes.shape[0]):
            raise ValueError("master_quads reference nodes outside master_reference_nodes")
        if float(self.pressure_stiffness) <= 0.0:
            raise ValueError("pressure_stiffness must be positive")
        if int(self.quadrature_order) < 1 or int(self.quadrature_order) > 6:
            raise ValueError("quadrature_order must be between 1 and 6")
        if int(self.secondary_tracking_rings) < 0:
            raise ValueError("secondary_tracking_rings must be non-negative")
        self.slave_quads = slave
        self.master_quads = master
        self.master_reference_nodes = master_nodes
        self._master_face_tracking_neighborhoods = _face_neighborhoods(
            master,
            nodes_per_face=4,
            rings=int(self.secondary_tracking_rings),
        )
        self._oracle = LagrangianQ4ClosestFeatureOracle(
            master_nodes,
            master,
            master_nodes.copy(),
            search_radius=self.search_radius,
        )

    def clear_secondary_contact_tracking(self) -> None:
        """Clear accepted-state secondary tracking hints for Q4 contact."""

        self._secondary_face_cache = None
        self._secondary_master_weight_cache = None

    def _secondary_cache_size(self) -> int:
        q = int(self.quadrature_order)
        return int(self.slave_quads.shape[0]) * q * q

    def _ensure_secondary_face_cache(self) -> np.ndarray:
        size = self._secondary_cache_size()
        if self._secondary_face_cache is None or self._secondary_face_cache.shape != (size,):
            self._secondary_face_cache = np.full(size, -1, dtype=np.int64)
        return self._secondary_face_cache

    def _ensure_secondary_master_weight_cache(self) -> np.ndarray:
        size = self._secondary_cache_size()
        if self._secondary_master_weight_cache is None or self._secondary_master_weight_cache.shape != (size, 4):
            self._secondary_master_weight_cache = np.full((size, 4), np.nan, dtype=float)
        return self._secondary_master_weight_cache

    def commit_secondary_tracking_from_sample_arrays(self, sample_arrays: dict[str, np.ndarray]) -> int:
        """Commit Q4 closest-feature tracking payload from an accepted state."""

        if not bool(self.secondary_path_tracking):
            return 0
        if "secondary_cache_indices" not in sample_arrays or "master_face_ids" not in sample_arrays:
            return 0
        cache_indices = np.asarray(sample_arrays["secondary_cache_indices"], dtype=np.int64).reshape(-1)
        face_ids = np.asarray(sample_arrays["master_face_ids"], dtype=np.int64).reshape(-1)
        weights = np.asarray(sample_arrays.get("master_weights", np.empty((0, 4))), dtype=float)
        if weights.ndim != 2 or weights.shape[0] != face_ids.size:
            return 0
        count = min(cache_indices.size, face_ids.size, weights.shape[0])
        if count == 0:
            return 0
        face_cache = self._ensure_secondary_face_cache()
        weight_cache = self._ensure_secondary_master_weight_cache()
        committed = 0
        for index in range(count):
            cache_index = int(cache_indices[index])
            face_id = int(face_ids[index])
            if face_id < 0 or not (0 <= cache_index < face_cache.shape[0]):
                continue
            row = np.asarray(weights[index], dtype=float).reshape(-1)
            if row.size != 4:
                continue
            total = float(np.sum(row))
            if total > 0.0:
                row = row / total
            face_cache[cache_index] = face_id
            weight_cache[cache_index] = row
            committed += 1
        return int(committed)

    def _preferred_secondary_tracking_faces(self, previous_face: int) -> list[int] | None:
        """Return accepted Q4 face and neighbors as ordered search hints."""

        if not bool(self.secondary_path_tracking):
            return None
        if not (0 <= int(previous_face) < len(self._master_face_tracking_neighborhoods)):
            return None
        values: list[int] = []
        seen: set[int] = set()
        first = int(previous_face)
        values.append(first)
        seen.add(first)
        for raw in self._master_face_tracking_neighborhoods[first]:
            face_id = int(raw)
            if 0 <= face_id < len(self._master_face_tracking_neighborhoods) and face_id not in seen:
                values.append(face_id)
                seen.add(face_id)
        return values

    def sample_arrays(self, x_current: np.ndarray) -> dict[str, np.ndarray]:
        """Return Q4 closest-feature contact payloads as batched arrays.

        The array path exposes the master Q4 face id, Q4 shape weights, and
        accepted-state tracking diagnostics needed by the secondary
        constraint-region contact law.  It reads accepted path-tracking caches
        as search hints but does not mutate them; callers must commit the final
        accepted arrays with :meth:`commit_secondary_tracking_from_sample_arrays`.
        The query still has an exact fallback: cached faces and adjacent master
        patches are only evaluated first, not used as the sole candidate.
        """

        if bool(self.clip_to_master_footprint):
            raise NotImplementedError("sample_arrays is not implemented for clipped Q4 footprint integration")
        X = np.asarray(x_current, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("x_current must have shape (n_nodes, 3)")
        master_start = int(self.master_node_offset)
        master_stop = master_start + self.master_reference_nodes.shape[0]
        if master_stop > X.shape[0]:
            raise ValueError("master_node_offset places master nodes outside x_current")
        self._oracle.refit(X[master_start:master_stop])
        points, weights = np.polynomial.legendre.leggauss(int(self.quadrature_order))
        sample_node_ids: list[np.ndarray] = []
        sample_weights: list[np.ndarray] = []
        gaps: list[float] = []
        normals: list[np.ndarray] = []
        areas: list[float] = []
        master_node_ids: list[np.ndarray] = []
        master_weights: list[np.ndarray] = []
        master_face_ids: list[int] = []
        secondary_cache_indices: list[int] = []
        cache_hits: list[bool] = []
        cache_matches: list[bool] = []
        weight_distances: list[float] = []
        face_cache = self._ensure_secondary_face_cache()
        weight_cache = self._ensure_secondary_master_weight_cache()
        q = int(self.quadrature_order)
        for face_id, quad in enumerate(self.slave_quads):
            global_quad = quad + int(self.slave_node_offset)
            qx = X[global_quad]
            for a, xi in enumerate(points):
                for b, eta in enumerate(points):
                    shape = _q4_shape_functions(float(xi), float(eta))
                    dshape = _q4_shape_derivatives(float(xi), float(eta))
                    tangent_xi = dshape[:, 0] @ qx
                    tangent_eta = dshape[:, 1] @ qx
                    jac = float(np.linalg.norm(np.cross(tangent_xi, tangent_eta)))
                    if jac <= 0.0:
                        continue
                    cache_index = int(face_id) * q * q + int(a) * q + int(b)
                    previous_face = int(face_cache[cache_index]) if 0 <= cache_index < face_cache.shape[0] else -1
                    preferred = self._preferred_secondary_tracking_faces(previous_face)
                    point = shape @ qx
                    payload = self._oracle.query(point, preferred_face_ids=preferred)
                    payload_weights = np.asarray(payload.master_weights, dtype=float).reshape(4)
                    hit = previous_face >= 0
                    match = bool(hit and previous_face == int(payload.face_id))
                    previous_weights = weight_cache[cache_index] if 0 <= cache_index < weight_cache.shape[0] else np.full(4, np.nan)
                    if np.all(np.isfinite(previous_weights)):
                        weight_distance = float(np.linalg.norm(payload_weights - previous_weights))
                    else:
                        weight_distance = np.nan
                    sample_node_ids.append(global_quad.copy())
                    sample_weights.append(shape.copy())
                    gaps.append(float(payload.gap))
                    normals.append(payload.normal.copy())
                    areas.append(float(jac * float(weights[a]) * float(weights[b])))
                    master_node_ids.append(payload.master_node_ids.astype(np.int64) + master_start)
                    master_weights.append(payload_weights.copy())
                    master_face_ids.append(int(payload.face_id))
                    secondary_cache_indices.append(cache_index)
                    cache_hits.append(bool(hit))
                    cache_matches.append(match)
                    weight_distances.append(weight_distance)
        if not gaps:
            return {
                "sample_node_ids": np.empty((0, 4), dtype=np.int64),
                "sample_weights": np.empty((0, 4), dtype=float),
                "gaps": np.empty(0, dtype=float),
                "normals": np.empty((0, 3), dtype=float),
                "areas": np.empty(0, dtype=float),
                "master_node_ids": np.empty((0, 4), dtype=np.int64),
                "master_weights": np.empty((0, 4), dtype=float),
                "master_barycentric": np.empty((0, 4), dtype=float),
                "master_face_ids": np.empty(0, dtype=np.int64),
                "secondary_cache_indices": np.empty(0, dtype=np.int64),
                "tracking_cache_hits": np.empty(0, dtype=bool),
                "tracking_cache_matches": np.empty(0, dtype=bool),
                "tracking_barycentric_distances": np.empty(0, dtype=float),
            }
        return {
            "sample_node_ids": np.vstack(sample_node_ids).astype(np.int64, copy=False),
            "sample_weights": np.vstack(sample_weights).astype(float, copy=False),
            "gaps": np.asarray(gaps, dtype=float),
            "normals": np.vstack(normals).astype(float, copy=False),
            "areas": np.asarray(areas, dtype=float),
            "master_node_ids": np.vstack(master_node_ids).astype(np.int64, copy=False),
            "master_weights": np.vstack(master_weights).astype(float, copy=False),
            "master_barycentric": np.vstack(master_weights).astype(float, copy=False),
            "master_face_ids": np.asarray(master_face_ids, dtype=np.int64),
            "secondary_cache_indices": np.asarray(secondary_cache_indices, dtype=np.int64),
            "tracking_cache_hits": np.asarray(cache_hits, dtype=bool),
            "tracking_cache_matches": np.asarray(cache_matches, dtype=bool),
            "tracking_barycentric_distances": np.asarray(weight_distances, dtype=float),
        }

    def samples(self, x_current: np.ndarray):
        X = np.asarray(x_current, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("x_current must have shape (n_nodes, 3)")
        master_start = int(self.master_node_offset)
        master_stop = master_start + self.master_reference_nodes.shape[0]
        if master_stop > X.shape[0]:
            raise ValueError("master_node_offset places master nodes outside x_current")
        self._oracle.refit(X[master_start:master_stop])
        if bool(self.clip_to_master_footprint):
            yield from self._clipped_samples(X, master_start)
            return
        points, weights = np.polynomial.legendre.leggauss(int(self.quadrature_order))
        for face_id, quad in enumerate(self.slave_quads):
            global_quad = quad + int(self.slave_node_offset)
            qx = X[global_quad]
            for a, xi in enumerate(points):
                for b, eta in enumerate(points):
                    shape = _q4_shape_functions(float(xi), float(eta))
                    dshape = _q4_shape_derivatives(float(xi), float(eta))
                    tangent_xi = dshape[:, 0] @ qx
                    tangent_eta = dshape[:, 1] @ qx
                    jac = float(np.linalg.norm(np.cross(tangent_xi, tangent_eta)))
                    if jac <= 0.0:
                        continue
                    point = shape @ qx
                    payload = self._oracle.query(point)
                    yield ContactSample(
                        node_ids=global_quad.copy(),
                        shape_weights=shape.copy(),
                        gap=float(payload.gap),
                        normal=payload.normal.copy(),
                        area=float(jac * float(weights[a]) * float(weights[b])),
                        stiffness=float(self.pressure_stiffness),
                        master_node_ids=payload.master_node_ids.astype(np.int64) + master_start,
                        master_shape_weights=payload.master_weights.copy(),
                    )

    def _clipped_samples(self, X: np.ndarray, master_start: int):
        """Yield samples integrated over the slave/master projected footprint.

        Abaqus-style surface-to-surface contact should not integrate a whole
        secondary face when only part of it has a valid projection onto a main
        face.  This path clips each Q4 slave footprint against each Q4 master
        footprint in the local master tangent plane and integrates over the
        resulting convex polygon.  It is a geometric support correction, not a
        penalty-stiffness tuning knob.
        """

        master_x = X[master_start : master_start + self.master_reference_nodes.shape[0]]
        for slave_face_id, slave_quad in enumerate(self.slave_quads):
            global_slave_quad = slave_quad + int(self.slave_node_offset)
            slave_qx = X[global_slave_quad]
            slave_plane = _quad_plane(slave_qx)
            if slave_plane is None:
                continue
            slave_origin, slave_normal = slave_plane
            for master_face_id, master_quad in enumerate(self.master_quads):
                master_qx = master_x[master_quad]
                basis = _quad_tangent_basis(master_qx)
                if basis is None:
                    continue
                origin, e1, e2, master_normal = basis
                normal_alignment = abs(float(slave_normal @ master_normal))
                if normal_alignment <= 1.0e-12:
                    continue
                slave_polygon = _ensure_counterclockwise(_project_points_2d(slave_qx, origin, e1, e2))
                master_polygon = _ensure_counterclockwise(_project_points_2d(master_qx, origin, e1, e2))
                overlap = _convex_polygon_clip_2d(slave_polygon, master_polygon)
                if overlap.shape[0] < 3:
                    continue
                overlap = _ensure_counterclockwise(overlap)
                centroid = np.mean(overlap, axis=0)
                for i in range(overlap.shape[0]):
                    a = overlap[i]
                    b = overlap[(i + 1) % overlap.shape[0]]
                    tri_area_projected = abs(_triangle_area_2d(centroid, a, b))
                    if tri_area_projected <= 1.0e-16:
                        continue
                    uv = (centroid + a + b) / 3.0
                    plane_point = origin + uv[0] * e1 + uv[1] * e2
                    line_distance = float(slave_normal @ (slave_origin - plane_point)) / float(slave_normal @ master_normal)
                    point = plane_point + line_distance * master_normal
                    xi_eta = _q4_natural_coordinates(point, slave_qx, clip=True)
                    shape = _q4_shape_functions(float(xi_eta[0]), float(xi_eta[1]))
                    payload = self._oracle.query(point)
                    if int(payload.face_id) != int(master_face_id):
                        continue
                    yield ContactSample(
                        node_ids=global_slave_quad.copy(),
                        shape_weights=shape.copy(),
                        gap=float(payload.gap),
                        normal=payload.normal.copy(),
                        area=float(tri_area_projected / normal_alignment),
                        stiffness=float(self.pressure_stiffness),
                        master_node_ids=payload.master_node_ids.astype(np.int64) + master_start,
                        master_shape_weights=payload.master_weights.copy(),
                    )


def _triangle_area(tri: np.ndarray) -> float:
    T = np.asarray(tri, dtype=float)
    if T.shape != (3, 3):
        raise ValueError("tri must have shape (3, 3)")
    return 0.5 * float(np.linalg.norm(np.cross(T[1] - T[0], T[2] - T[0])))


def _surface_nodal_normals(x_current: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return area-weighted nodal normals for an oriented triangular surface."""

    X = np.asarray(x_current, dtype=float)
    F = np.asarray(faces, dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("x_current must have shape (n_nodes, 3)")
    if F.ndim != 2 or F.shape[1] != 3:
        raise ValueError("faces must have shape (n_faces, 3)")
    normals = np.zeros_like(X, dtype=float)
    if F.size == 0:
        return normals
    tri = X[F]
    area_vectors = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for local in range(3):
        np.add.at(normals, F[:, local], area_vectors)
    lengths = np.linalg.norm(normals, axis=1)
    active = lengths > 1.0e-30
    normals[active] /= lengths[active, None]
    return normals


def _face_neighborhoods(faces: np.ndarray, *, nodes_per_face: int, rings: int) -> tuple[np.ndarray, ...]:
    """Return edge-adjacent face neighborhoods for contact tracking."""

    F = np.asarray(faces, dtype=np.int64)
    if F.ndim != 2 or F.shape[1] != int(nodes_per_face):
        raise ValueError(f"faces must have shape (n_faces, {int(nodes_per_face)})")
    if int(rings) < 0:
        raise ValueError("rings must be non-negative")
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(F):
        nodes = [int(v) for v in face]
        for local in range(int(nodes_per_face)):
            a = nodes[local]
            b = nodes[(local + 1) % int(nodes_per_face)]
            key = (a, b) if a <= b else (b, a)
            edge_to_faces.setdefault(key, []).append(int(face_id))
    adjacency: list[set[int]] = [set() for _ in range(F.shape[0])]
    for face_ids in edge_to_faces.values():
        if len(face_ids) < 2:
            continue
        for face_id in face_ids:
            adjacency[int(face_id)].update(int(other) for other in face_ids if int(other) != int(face_id))
    neighborhoods: list[np.ndarray] = []
    for face_id in range(F.shape[0]):
        seen = {int(face_id)}
        frontier = {int(face_id)}
        for _ring in range(int(rings)):
            next_frontier: set[int] = set()
            for current in frontier:
                next_frontier.update(adjacency[int(current)])
            next_frontier.difference_update(seen)
            if not next_frontier:
                break
            seen.update(next_frontier)
            frontier = next_frontier
        neighborhoods.append(np.asarray(sorted(seen), dtype=np.int64))
    return tuple(neighborhoods)


def _triangle_face_neighborhoods(faces: np.ndarray, *, rings: int) -> tuple[np.ndarray, ...]:
    """Return edge-adjacent triangle face neighborhoods for contact tracking."""

    return _face_neighborhoods(faces, nodes_per_face=3, rings=int(rings))


def _triangle_barycentric_unclipped(point: np.ndarray, tri: np.ndarray) -> np.ndarray | None:
    """Return barycentric coordinates of point in a triangle plane."""

    p = np.asarray(point, dtype=float).reshape(3)
    T = np.asarray(tri, dtype=float)
    if T.shape != (3, 3):
        raise ValueError("tri must have shape (3, 3)")
    v0 = T[1] - T[0]
    v1 = T[2] - T[0]
    v2 = p - T[0]
    d00 = float(v0 @ v0)
    d01 = float(v0 @ v1)
    d11 = float(v1 @ v1)
    d20 = float(v2 @ v0)
    d21 = float(v2 @ v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) <= 1.0e-30:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.asarray([u, v, w], dtype=float)


def _triangle_plane(tri: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    T = np.asarray(tri, dtype=float)
    if T.shape != (3, 3):
        raise ValueError("tri must have shape (3, 3)")
    normal = np.cross(T[1] - T[0], T[2] - T[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-30:
        return None
    return np.mean(T, axis=0), normal / norm


def _triangle_tangent_basis(tri: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    plane = _triangle_plane(tri)
    if plane is None:
        return None
    origin, normal = plane
    T = np.asarray(tri, dtype=float)
    e1 = T[1] - T[0]
    e1_norm = float(np.linalg.norm(e1))
    if e1_norm <= 1.0e-30:
        return None
    e1 = e1 / e1_norm
    e2 = np.cross(normal, e1)
    e2_norm = float(np.linalg.norm(e2))
    if e2_norm <= 1.0e-30:
        return None
    return origin, e1, e2 / e2_norm, normal


def _triangle_bounding_spheres(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return centroids and conservative radii for triangular facets."""

    T = np.asarray(triangles, dtype=float)
    if T.size == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    if T.ndim != 3 or T.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (n, 3, 3)")
    centroids = np.mean(T, axis=1)
    radii = np.max(np.linalg.norm(T - centroids[:, None, :], axis=2), axis=1)
    return centroids, radii


def _empty_sample_arrays() -> dict[str, np.ndarray]:
    """Return an empty batched contact-sample payload."""

    return {
        "sample_node_ids": np.empty((0, 3), dtype=np.int64),
        "sample_weights": np.empty((0, 3), dtype=float),
        "gaps": np.empty(0, dtype=float),
        "normals": np.empty((0, 3), dtype=float),
        "areas": np.empty(0, dtype=float),
        "master_node_ids": np.empty((0, 3), dtype=np.int64),
        "master_weights": np.empty((0, 3), dtype=float),
        "master_barycentric": np.empty((0, 3), dtype=float),
        "master_face_ids": np.empty(0, dtype=np.int64),
        "secondary_cache_indices": np.empty(0, dtype=np.int64),
        "tracking_cache_hits": np.empty(0, dtype=bool),
        "tracking_cache_matches": np.empty(0, dtype=bool),
        "tracking_barycentric_distances": np.empty(0, dtype=float),
    }


def _flatten_candidate_lists(candidate_lists) -> tuple[np.ndarray, np.ndarray]:
    """Flatten per-query candidate face ids for the C++ indexed projection."""

    offsets = np.empty(len(candidate_lists) + 1, dtype=np.int64)
    offsets[0] = 0
    flat: list[int] = []
    for idx, ids in enumerate(candidate_lists):
        flat.extend(int(face_id) for face_id in ids)
        offsets[idx + 1] = len(flat)
    return offsets, np.asarray(flat, dtype=np.int64)


def _slave_face_outside_master_tube(tri: np.ndarray, master_tree: cKDTree, master_max_radius: float, search_radius: float) -> bool:
    """Return whether a slave triangle cannot intersect the master search tube."""

    slave_centroid = np.mean(np.asarray(tri, dtype=float), axis=0)
    slave_radius = float(np.max(np.linalg.norm(np.asarray(tri, dtype=float) - slave_centroid, axis=1)))
    nearest_distance = float(master_tree.query(slave_centroid, k=1)[0])
    tube_radius = float(search_radius) + slave_radius + float(master_max_radius)
    return bool(nearest_distance > tube_radius + 1.0e-14)


def _q4_shape_functions(xi: float, eta: float) -> np.ndarray:
    return 0.25 * np.asarray(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )


def _q4_shape_derivatives(xi: float, eta: float) -> np.ndarray:
    return 0.25 * np.asarray(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [1.0 - eta, -(1.0 + xi)],
            [1.0 + eta, 1.0 + xi],
            [-(1.0 + eta), 1.0 - xi],
        ],
        dtype=float,
    )


def _q4_initial_natural_coordinates(point: np.ndarray, quad: np.ndarray) -> np.ndarray:
    center = _q4_shape_functions(0.0, 0.0) @ quad
    dshape = _q4_shape_derivatives(0.0, 0.0)
    tangent_xi = dshape[:, 0] @ quad
    tangent_eta = dshape[:, 1] @ quad
    A = np.column_stack((tangent_xi, tangent_eta))
    xi_eta, *_ = np.linalg.lstsq(A, np.asarray(point, dtype=float) - center, rcond=None)
    return np.clip(np.asarray(xi_eta, dtype=float), -1.0, 1.0)


def _q4_natural_coordinates(point: np.ndarray, quad: np.ndarray, *, clip: bool) -> np.ndarray:
    xi_eta = _q4_initial_natural_coordinates(point, quad)
    for _ in range(12):
        shape = _q4_shape_functions(float(xi_eta[0]), float(xi_eta[1]))
        dshape = _q4_shape_derivatives(float(xi_eta[0]), float(xi_eta[1]))
        residual = shape @ quad - np.asarray(point, dtype=float)
        tangent_xi = dshape[:, 0] @ quad
        tangent_eta = dshape[:, 1] @ quad
        jac = np.asarray(
            [
                [float(tangent_xi @ tangent_xi), float(tangent_xi @ tangent_eta)],
                [float(tangent_eta @ tangent_xi), float(tangent_eta @ tangent_eta)],
            ],
            dtype=float,
        )
        rhs = -np.asarray([float(tangent_xi @ residual), float(tangent_eta @ residual)], dtype=float)
        try:
            step = np.linalg.solve(jac, rhs)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(jac, rhs, rcond=None)
        xi_eta = xi_eta + step
        if clip:
            xi_eta = np.clip(xi_eta, -1.0, 1.0)
        if float(np.linalg.norm(step)) <= 1.0e-12:
            break
    return np.asarray(xi_eta, dtype=float)


def _quad_plane(quad: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    q = np.asarray(quad, dtype=float)
    center = np.mean(q, axis=0)
    normal = np.cross(q[1] - q[0], q[3] - q[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-30:
        normal = np.cross(q[2] - q[1], q[0] - q[1])
        norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-30:
        return None
    return center, normal / norm


def _quad_tangent_basis(quad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    plane = _quad_plane(quad)
    if plane is None:
        return None
    origin, normal = plane
    q = np.asarray(quad, dtype=float)
    e1 = q[1] - q[0]
    e1_norm = float(np.linalg.norm(e1))
    if e1_norm <= 1.0e-30:
        e1 = q[2] - q[3]
        e1_norm = float(np.linalg.norm(e1))
    if e1_norm <= 1.0e-30:
        return None
    e1 = e1 / e1_norm
    e2 = np.cross(normal, e1)
    e2_norm = float(np.linalg.norm(e2))
    if e2_norm <= 1.0e-30:
        return None
    return origin, e1, e2 / e2_norm, normal


def _project_points_2d(points: np.ndarray, origin: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    return np.column_stack((p @ np.asarray(e1, dtype=float), p @ np.asarray(e2, dtype=float)))


def _polygon_signed_area_2d(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=float)
    if p.shape[0] < 3:
        return 0.0
    x = p[:, 0]
    y = p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ensure_counterclockwise(poly: np.ndarray) -> np.ndarray:
    p = np.asarray(poly, dtype=float)
    if _polygon_signed_area_2d(p) < 0.0:
        return p[::-1].copy()
    return p.copy()


def _triangle_area_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    ac = np.asarray(c, dtype=float) - np.asarray(a, dtype=float)
    return 0.5 * float(ab[0] * ac[1] - ab[1] * ac[0])


def _convex_polygon_clip_2d(subject: np.ndarray, clip: np.ndarray, *, tol: float = 1.0e-12) -> np.ndarray:
    output = [np.asarray(point, dtype=float) for point in np.asarray(subject, dtype=float)]
    clip_poly = [np.asarray(point, dtype=float) for point in np.asarray(clip, dtype=float)]
    if len(output) < 3 or len(clip_poly) < 3:
        return np.empty((0, 2), dtype=float)

    def inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
        edge = edge_end - edge_start
        rel = point - edge_start
        return bool(edge[0] * rel[1] - edge[1] * rel[0] >= -tol)

    def intersection(p0: np.ndarray, p1: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> np.ndarray:
        direction = p1 - p0
        edge = edge_end - edge_start
        denom = direction[0] * edge[1] - direction[1] * edge[0]
        if abs(float(denom)) <= tol:
            return p1.copy()
        delta = edge_start - p0
        t = (delta[0] * edge[1] - delta[1] * edge[0]) / denom
        return p0 + t * direction

    for edge_index, edge_start in enumerate(clip_poly):
        edge_end = clip_poly[(edge_index + 1) % len(clip_poly)]
        input_poly = output
        output = []
        if not input_poly:
            break
        previous = input_poly[-1]
        previous_inside = inside(previous, edge_start, edge_end)
        for current in input_poly:
            current_inside = inside(current, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current, edge_start, edge_end))
            previous = current
            previous_inside = current_inside
    if len(output) < 3:
        return np.empty((0, 2), dtype=float)
    return np.asarray(output, dtype=float)


def _as_point(value: np.ndarray, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return point
