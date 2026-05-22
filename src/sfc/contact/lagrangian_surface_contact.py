"""Surface-contact geometry backed by the Lagrangian SDF oracle."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from sfc.contact.lagrangian_sdf_oracle import LagrangianSDFContactOracle
from sfc.fem.calculix_aligned import ContactSample
from sfc.sdf.material_sdf import MaterialSDF

try:  # pragma: no cover - optional backend availability is platform-dependent.
    from sfc.sdf._cpp_projection import closest_points_all_faces as _cpp_closest_points_all_faces
    from sfc.sdf._cpp_projection import closest_points_padded_aabb as _cpp_closest_points_padded_aabb
    from sfc.sdf._cpp_projection import is_available as _cpp_projection_available
except Exception:  # pragma: no cover
    _cpp_closest_points_all_faces = None
    _cpp_closest_points_padded_aabb = None
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

    def query(self, point: np.ndarray) -> QuadrilateralClosestFeaturePayload:
        x = _as_point(point, "point")
        candidates = self._candidate_ids(x)
        best: QuadrilateralClosestFeaturePayload | None = None
        best_dist2 = np.inf
        bounds = self._aabb_distance_squared(x, candidates) if candidates.size else np.empty(0, dtype=float)
        order = np.argsort(bounds, kind="stable") if bounds.size else np.empty(0, dtype=np.int64)
        candidates = candidates[order] if order.size else candidates
        bounds = bounds[order] if order.size else bounds
        for candidate, bound2 in zip(candidates, bounds, strict=True):
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
    _oracle: LagrangianSDFContactOracle = field(init=False, repr=False)

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
        self.slave_faces = faces
        self.master_reference_nodes = master_nodes
        self._oracle = LagrangianSDFContactOracle(
            self.master_material,
            master_nodes.copy(),
            search_radius=self.search_radius,
            patch_cell_size=self.patch_cell_size,
        )

    def samples(self, x_current: np.ndarray):
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

    def sample_arrays(self, x_current: np.ndarray) -> dict[str, np.ndarray] | None:
        """Return batched sample arrays for compiled closest-feature queries."""

        if not (bool(self.compiled_batch_projection) and _cpp_projection_available() and _cpp_closest_points_all_faces is not None):
            return None
        X, master_x, master_tree, master_max_radius, barycentric, weight_scale = self._prepare_sampling(x_current)
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
            }
        kept_faces = global_faces[keep]
        kept_triangles = triangles[keep]
        kept_areas = face_areas[keep]
        n_quadrature = barycentric.shape[0]
        point_array = np.einsum("qa,fad->fqd", barycentric, kept_triangles).reshape((-1, 3))
        sample_nodes = np.repeat(kept_faces, n_quadrature, axis=0)
        sample_weights = np.tile(barycentric, (kept_faces.shape[0], 1))
        areas = (kept_areas[:, None] * weight_scale[None, :]).reshape(-1)
        if _cpp_closest_points_padded_aabb is not None:
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
        }

    def _prepare_sampling(
        self,
        x_current: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, cKDTree | None, float, np.ndarray, np.ndarray]:
        X = np.asarray(x_current, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("x_current must have shape (n_nodes, 3)")
        master_start = int(self.master_node_offset)
        master_stop = master_start + self.master_reference_nodes.shape[0]
        if master_stop > X.shape[0]:
            raise ValueError("master_node_offset places master nodes outside x_current")
        master_x = X[master_start:master_stop]
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
    _oracle: LagrangianQ4ClosestFeatureOracle = field(init=False, repr=False)

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
        self.slave_quads = slave
        self.master_quads = master
        self.master_reference_nodes = master_nodes
        self._oracle = LagrangianQ4ClosestFeatureOracle(
            master_nodes,
            master,
            master_nodes.copy(),
            search_radius=self.search_radius,
        )

    def samples(self, x_current: np.ndarray):
        X = np.asarray(x_current, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("x_current must have shape (n_nodes, 3)")
        master_start = int(self.master_node_offset)
        master_stop = master_start + self.master_reference_nodes.shape[0]
        if master_stop > X.shape[0]:
            raise ValueError("master_node_offset places master nodes outside x_current")
        self._oracle.refit(X[master_start:master_stop])
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


def _triangle_area(tri: np.ndarray) -> float:
    T = np.asarray(tri, dtype=float)
    if T.shape != (3, 3):
        raise ValueError("tri must have shape (3, 3)")
    return 0.5 * float(np.linalg.norm(np.cross(T[1] - T[0], T[2] - T[0])))


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


def _as_point(value: np.ndarray, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return point
