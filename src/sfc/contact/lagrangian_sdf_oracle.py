"""Lagrangian material-SDF contact oracle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from sfc.fem.deformation_map import FEMDeformationMap
from sfc.sdf.local_projection import closest_point_on_triangle
from sfc.sdf.material_sdf import MaterialSDF, ReferencePatchBVH

from .narrow_phase import SurfaceSample


@dataclass(frozen=True, slots=True)
class LagrangianSDFQueryResult:
    """One deformation-aware material-SDF query."""

    gap: float
    normal: np.ndarray
    closest_point: np.ndarray
    material_point: np.ndarray
    face_id: int
    barycentric: np.ndarray
    iterations: int
    converged: bool
    candidates_evaluated: int
    used_cached_patch: bool
    master_node_ids: np.ndarray
    master_weights: np.ndarray


@dataclass(frozen=True, slots=True)
class LagrangianPatchPairQueryResult:
    """Closest-patch pair result for two Lagrangian SDF oracles."""

    gap: float
    normal: np.ndarray
    point_a: np.ndarray
    point_b: np.ndarray
    material_point_a: np.ndarray
    material_point_b: np.ndarray
    face_id_a: int
    face_id_b: int
    barycentric_a: np.ndarray
    barycentric_b: np.ndarray
    candidates_evaluated: int


@dataclass(frozen=True, slots=True)
class LagrangianPullbackQueryResult:
    """Cheap pull-back material-SDF query before closest-point correction."""

    gap: float
    normal: np.ndarray
    material_point: np.ndarray
    current_point: np.ndarray
    phi0: float
    metric_scale: float
    element_id: int
    node_ids: np.ndarray
    shape_values: np.ndarray


@dataclass(slots=True)
class LagrangianSDFContactOracle:
    """On-demand SDF contact query through a FEM deformation map.

    The oracle stores a reference material SDF and refits current-space AABBs
    for its zero-level patches.  A query first obtains a material-surface patch
    candidate, pulls the point back to the patch parameter domain, and then
    performs a local closest-point Newton correction on the deformed patch.

    The current implementation uses triangular reference patches and affine
    surface interpolation on each patch.  It is a conservative first step
    toward the final material-SDF oracle: no current-space SDF grid is built,
    and cached patch ids are used only as ordering hints, so caching cannot
    change the exact closest-patch result among the enumerated candidates.
    """

    material: MaterialSDF
    x_current: np.ndarray
    search_radius: float | None = None
    max_newton_iterations: int = 8
    newton_tolerance: float = 1.0e-12
    cache_enabled: bool = True
    deformation_map: FEMDeformationMap | None = None
    patch_cell_size: float | None = None
    bvh: ReferencePatchBVH = field(init=False)
    _patch_cache: dict[Hashable, int] = field(default_factory=dict, init=False, repr=False)
    _barycentric_cache: dict[Hashable, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _face_element_cache: dict[tuple[int, ...], int | None] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        X = _as_nodes(self.x_current, "x_current")
        if X.shape != self.material.reference_nodes.shape:
            raise ValueError("x_current must match material.reference_nodes shape")
        if self.search_radius is not None and float(self.search_radius) < 0.0:
            raise ValueError("search_radius must be non-negative")
        if self.deformation_map is not None and self.deformation_map.mesh.X.shape != self.material.reference_nodes.shape:
            raise ValueError("deformation_map mesh must use the same reference nodes as material")
        object.__setattr__(self, "x_current", X)
        object.__setattr__(
            self,
            "bvh",
            self.material.build_patch_bvh(X, padding=self.search_radius, cell_size=self.patch_cell_size),
        )
        if self.deformation_map is not None and not np.allclose(self.deformation_map.x_current, X):
            self.deformation_map = self.deformation_map.with_current(X)

    def refit(self, x_current: np.ndarray) -> None:
        """Refit current patch AABBs without rebuilding a current SDF grid."""

        X = _as_nodes(x_current, "x_current")
        if X.shape != self.material.reference_nodes.shape:
            raise ValueError("x_current must match material.reference_nodes shape")
        self.x_current = X
        self.bvh = self.bvh.refit(X, padding=self.search_radius, cell_size=self.patch_cell_size)
        if self.deformation_map is not None:
            self.deformation_map = self.deformation_map.with_current(X)

    def clear_cache(self) -> None:
        """Clear active patch hints."""

        self._patch_cache.clear()
        self._barycentric_cache.clear()
        self._face_element_cache.clear()

    def query_gap_normal(
        self,
        point: np.ndarray,
        x_current: np.ndarray | None = None,
        *,
        cache_key: Hashable | None = None,
    ) -> tuple[float, np.ndarray]:
        """Return signed gap and current normal for ``point``."""

        result = self.query(point, x_current=x_current, cache_key=cache_key)
        return result.gap, result.normal.copy()

    def query_many(
        self,
        points: np.ndarray,
        x_current: np.ndarray | None = None,
        *,
        cache_keys: list[Hashable] | tuple[Hashable, ...] | None = None,
    ) -> tuple[LagrangianSDFQueryResult, ...]:
        """Evaluate multiple oracle queries with shared refit/cache state."""

        if x_current is not None:
            self.refit(x_current)
        P = np.asarray(points, dtype=float)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        if cache_keys is not None and len(cache_keys) != P.shape[0]:
            raise ValueError("cache_keys must match number of points")
        return tuple(
            self.query(point, cache_key=None if cache_keys is None else cache_keys[idx])
            for idx, point in enumerate(P)
        )

    def query_gap_normal_batch(
        self,
        points: np.ndarray,
        x_current: np.ndarray | None = None,
        *,
        cache_keys: list[Hashable] | tuple[Hashable, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return batch gaps and normals."""

        results = self.query_many(points, x_current=x_current, cache_keys=cache_keys)
        return (
            np.asarray([result.gap for result in results], dtype=float),
            np.vstack([result.normal for result in results]) if results else np.empty((0, 3), dtype=float),
        )

    def query_pullback(
        self,
        point: np.ndarray,
        x_current: np.ndarray | None = None,
        *,
        preferred_element: int | None = None,
        initial_material_point: np.ndarray | None = None,
    ) -> LagrangianPullbackQueryResult:
        """Return the cheap material pull-back SDF query.

        This evaluates ``X ~= chi^{-1}(x)`` and returns
        ``phi0(X) / ||F^{-T} grad_X phi0(X)||``.  It is the coarse stage from
        the final-aim algorithm and does not replace the exact closest-point
        Newton corrector used by :meth:`query`.
        """

        if self.deformation_map is None:
            raise RuntimeError("deformation_map is required for pull-back SDF queries")
        if x_current is not None:
            self.refit(x_current)
        x = _as_point(point, "point")
        evaluation = self.deformation_map.pull_back_current_point(
            x,
            preferred_element=preferred_element,
            initial_material_point=initial_material_point,
            max_iterations=max(1, int(self.max_newton_iterations)),
            tolerance=float(self.newton_tolerance),
        )
        phi0 = float(self.material.query_phi(evaluation.material_point))
        grad = np.asarray(self.material.query_raw_gradient(evaluation.material_point), dtype=float)
        covector = _solve_transpose(evaluation.deformation_gradient, grad)
        metric_scale = float(np.linalg.norm(covector))
        if metric_scale <= 0.0:
            raise ValueError("pull-back metric scale is zero")
        normal = covector / metric_scale
        return LagrangianPullbackQueryResult(
            gap=float(phi0 / metric_scale),
            normal=normal,
            material_point=evaluation.material_point.copy(),
            current_point=evaluation.current_point.copy(),
            phi0=phi0,
            metric_scale=metric_scale,
            element_id=int(evaluation.element_id),
            node_ids=evaluation.node_ids.copy(),
            shape_values=evaluation.shape_values.copy(),
        )

    def query_pullback_gap_normal(
        self,
        point: np.ndarray,
        x_current: np.ndarray | None = None,
        *,
        preferred_element: int | None = None,
        initial_material_point: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        """Return cheap pull-back gap and normal."""

        result = self.query_pullback(
            point,
            x_current=x_current,
            preferred_element=preferred_element,
            initial_material_point=initial_material_point,
        )
        return result.gap, result.normal.copy()

    def query_jacobian_wrt_point(
        self,
        point: np.ndarray,
        x_current: np.ndarray | None = None,
        *,
        cache_key: Hashable | None = None,
    ) -> np.ndarray:
        """Return ``dg/dx`` for the locally corrected gap."""

        return self.query(point, x_current=x_current, cache_key=cache_key).normal.copy()

    def query_master_sensitivity(
        self,
        point: np.ndarray,
        x_current: np.ndarray | None = None,
        *,
        cache_key: Hashable | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return master node ids and ``dg/dx_node`` vectors."""

        result = self.query(point, x_current=x_current, cache_key=cache_key)
        vectors = -result.master_weights[:, None] * result.normal[None, :]
        return result.master_node_ids.copy(), vectors

    def query(
        self,
        point: np.ndarray,
        x_current: np.ndarray | None = None,
        *,
        cache_key: Hashable | None = None,
    ) -> LagrangianSDFQueryResult:
        """Evaluate the deformation-aware material SDF oracle."""

        if x_current is not None:
            self.refit(x_current)
        x = _as_point(point, "point")
        cached_face_id = self._patch_cache.get(cache_key) if self.cache_enabled and cache_key is not None else None
        cached_bary = self._barycentric_cache.get(cache_key) if self.cache_enabled and cache_key is not None else None
        candidates = self.bvh.candidates(
            x,
            search_radius=self.search_radius,
            cached_face_id=cached_face_id,
        )
        if candidates.size:
            candidate_bounds = self.bvh.aabb_distance_squared(x)[candidates]
            order = np.argsort(candidate_bounds, kind="stable")
            candidates = candidates[order]
            candidate_bounds = candidate_bounds[order]
        else:
            candidate_bounds = np.empty(0, dtype=float)
        best: LagrangianSDFQueryResult | None = None
        best_abs_gap = np.inf
        best_dist2 = np.inf
        for candidate_id, bound2 in zip(candidates, candidate_bounds, strict=True):
            if best is not None and float(bound2) > best_dist2 + 1.0e-15:
                break
            initial = cached_bary if cached_face_id == int(candidate_id) and cached_bary is not None else None
            if self.deformation_map is None:
                result = self._query_patch(
                    x,
                    int(candidate_id),
                    initial_barycentric=initial,
                    used_cached_patch=cached_face_id == int(candidate_id),
                    candidate_count=int(candidates.size),
                )
            else:
                result = self._query_kkt(
                    x,
                    int(candidate_id),
                    initial_barycentric=initial,
                    used_cached_patch=cached_face_id == int(candidate_id),
                    candidate_count=int(candidates.size),
                )
            dist2 = float(np.dot(x - result.closest_point, x - result.closest_point))
            if dist2 < best_dist2:
                best = result
                best_dist2 = dist2
                best_abs_gap = abs(float(result.gap))
            elif np.isclose(dist2, best_dist2) and abs(float(result.gap)) < best_abs_gap:
                best = result
                best_abs_gap = abs(float(result.gap))
        if best is None:
            raise RuntimeError("no material-SDF patch candidates were available")
        if self.cache_enabled and cache_key is not None:
            self._patch_cache[cache_key] = int(best.face_id)
            self._barycentric_cache[cache_key] = best.barycentric.copy()
        return best

    def _query_patch(
        self,
        point: np.ndarray,
        face_id: int,
        *,
        initial_barycentric: np.ndarray | None,
        used_cached_patch: bool,
        candidate_count: int,
    ) -> LagrangianSDFQueryResult:
        face = self.material.boundary_faces[int(face_id)]
        reference_triangle = self.material.reference_nodes[face]
        current_triangle = self.x_current[face]
        if initial_barycentric is None:
            bary0 = _pullback_initial(point, current_triangle)
        else:
            bary0 = _project_barycentric_to_simplex(np.asarray(initial_barycentric, dtype=float))
        bary, iterations, converged = _closest_point_newton_on_triangle(
            point,
            current_triangle,
            bary0,
            max_iterations=int(self.max_newton_iterations),
            tolerance=float(self.newton_tolerance),
        )
        closest = bary @ current_triangle
        material_point = bary @ reference_triangle
        normal = _oriented_current_normal(current_triangle)
        gap = float(np.dot(point - closest, normal))
        return LagrangianSDFQueryResult(
            gap=gap,
            normal=normal,
            closest_point=closest,
            material_point=material_point,
            face_id=int(face_id),
            barycentric=bary,
            iterations=int(iterations),
            converged=bool(converged),
            candidates_evaluated=int(candidate_count),
            used_cached_patch=bool(used_cached_patch),
            master_node_ids=face.copy(),
            master_weights=bary.copy(),
        )

    def _query_kkt(
        self,
        point: np.ndarray,
        face_id: int,
        *,
        initial_barycentric: np.ndarray | None,
        used_cached_patch: bool,
        candidate_count: int,
    ) -> LagrangianSDFQueryResult:
        if self.deformation_map is None:
            raise RuntimeError("deformation_map is required for KKT oracle queries")
        face = self.material.boundary_faces[int(face_id)]
        reference_triangle = self.material.reference_nodes[face]
        current_triangle = self.x_current[face]
        if initial_barycentric is None:
            bary0 = _pullback_initial(point, current_triangle)
        else:
            bary0 = _project_barycentric_to_simplex(np.asarray(initial_barycentric, dtype=float))
        X = bary0 @ reference_triangle
        preferred_element = self._element_for_face(face)
        lam = 0.0
        converged = False
        iterations = 0
        evaluation = self.deformation_map.evaluate(X, preferred_element=preferred_element)
        for iterations in range(1, max(1, int(self.max_newton_iterations)) + 1):
            evaluation = self.deformation_map.evaluate(X, preferred_element=preferred_element)
            phi = float(self.material.query_phi(X))
            grad = np.asarray(self.material.query_gradient(X), dtype=float)
            F = np.asarray(evaluation.deformation_gradient, dtype=float)
            residual = evaluation.current_point - point
            stationarity = F.T @ residual + lam * grad
            kkt_residual = np.concatenate((stationarity, np.asarray([phi], dtype=float)))
            if float(np.linalg.norm(kkt_residual)) <= float(self.newton_tolerance):
                converged = True
                break
            tangent = np.zeros((4, 4), dtype=float)
            tangent[:3, :3] = F.T @ F
            tangent[:3, 3] = grad
            tangent[3, :3] = grad
            try:
                step = np.linalg.solve(tangent, -kkt_residual)
            except np.linalg.LinAlgError:
                step, *_ = np.linalg.lstsq(tangent, -kkt_residual, rcond=None)
            X = X + step[:3]
            lam += float(step[3])
            if float(np.linalg.norm(step)) <= float(self.newton_tolerance):
                converged = True
                break
        evaluation = self.deformation_map.evaluate(X, preferred_element=preferred_element)
        grad = np.asarray(self.material.query_gradient(X), dtype=float)
        normal = _push_forward_normal(evaluation.deformation_gradient, grad)
        closest = evaluation.current_point.copy()
        gap = float(np.dot(point - closest, normal))
        return LagrangianSDFQueryResult(
            gap=gap,
            normal=normal,
            closest_point=closest,
            material_point=X.copy(),
            face_id=int(face_id),
            barycentric=bary0,
            iterations=int(iterations),
            converged=bool(converged),
            candidates_evaluated=int(candidate_count),
            used_cached_patch=bool(used_cached_patch),
            master_node_ids=evaluation.node_ids.copy(),
            master_weights=evaluation.shape_values.copy(),
        )

    def _element_for_face(self, face_node_ids: np.ndarray) -> int | None:
        if self.deformation_map is None:
            return None
        key = tuple(sorted(int(v) for v in np.asarray(face_node_ids, dtype=np.int64).ravel()))
        if key not in self._face_element_cache:
            self._face_element_cache[key] = self.deformation_map.element_for_face(np.asarray(face_node_ids, dtype=np.int64))
        return self._face_element_cache[key]


@dataclass(frozen=True, slots=True)
class LagrangianOracleConstraint:
    """One sample-to-material-SDF oracle constraint."""

    g: float
    normal: np.ndarray
    slave_node_ids: np.ndarray
    slave_weights: np.ndarray
    master_node_ids: np.ndarray
    master_weights: np.ndarray
    x_slave: np.ndarray
    query: LagrangianSDFQueryResult

    @property
    def active(self) -> bool:
        """Return true for penetrating constraints."""

        return self.g < 0.0


@dataclass(frozen=True, slots=True)
class LagrangianOracleContactResponse:
    """Penalty response assembled from Lagrangian SDF oracle constraints."""

    force: np.ndarray
    stiffness: csr_matrix
    constraints: tuple[LagrangianOracleConstraint, ...]
    quadrature_weights: np.ndarray

    @property
    def active_count(self) -> int:
        """Number of active constraints."""

        return sum(1 for constraint in self.constraints if constraint.active)

    @property
    def min_gap(self) -> float:
        """Minimum sampled gap."""

        if not self.constraints:
            return 0.0
        return float(min(constraint.g for constraint in self.constraints))


def lagrangian_oracle_constraint_from_sample(
    slave_x_current: np.ndarray,
    sample: SurfaceSample,
    oracle: LagrangianSDFContactOracle,
    *,
    cache_key: Hashable | None = None,
) -> LagrangianOracleConstraint:
    """Evaluate one slave sample against the material-SDF oracle."""

    X = np.asarray(slave_x_current, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("slave_x_current must have shape (n_nodes, 3)")
    point = sample.point(X)
    query = oracle.query(point, cache_key=cache_key)
    return LagrangianOracleConstraint(
        g=float(query.gap),
        normal=query.normal.copy(),
        slave_node_ids=sample.node_ids.copy(),
        slave_weights=sample.weights.copy(),
        master_node_ids=query.master_node_ids.copy(),
        master_weights=query.master_weights.copy(),
        x_slave=point,
        query=query,
    )


def lagrangian_oracle_jacobian_row(
    constraint: LagrangianOracleConstraint,
    *,
    n_total_dofs: int,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
) -> csr_matrix:
    """Assemble one oracle gap Jacobian row."""

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    normal = np.asarray(constraint.normal, dtype=float)
    for node, weight in zip(constraint.slave_node_ids, constraint.slave_weights, strict=True):
        for axis in range(3):
            rows.append(0)
            cols.append(int(slave_dof_offset) + 3 * int(node) + axis)
            vals.append(float(weight) * float(normal[axis]))
    for node, weight in zip(constraint.master_node_ids, constraint.master_weights, strict=True):
        for axis in range(3):
            rows.append(0)
            cols.append(int(master_dof_offset) + 3 * int(node) + axis)
            vals.append(-float(weight) * float(normal[axis]))
    return coo_matrix((vals, (rows, cols)), shape=(1, int(n_total_dofs))).tocsr()


def lagrangian_oracle_penalty_response(
    slave_x_current: np.ndarray,
    samples: list[SurfaceSample] | tuple[SurfaceSample, ...],
    oracle: LagrangianSDFContactOracle,
    *,
    pressure_stiffness: float,
    n_total_dofs: int,
    sample_area_weights: np.ndarray | None = None,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
) -> LagrangianOracleContactResponse:
    """Assemble frictionless penalty contact from oracle gap constraints."""

    k = float(pressure_stiffness)
    if k <= 0.0:
        raise ValueError("pressure_stiffness must be positive")
    constraints = tuple(
        lagrangian_oracle_constraint_from_sample(
            slave_x_current,
            sample,
            oracle,
            cache_key=idx,
        )
        for idx, sample in enumerate(samples)
    )
    if sample_area_weights is None:
        weights = np.ones(len(constraints), dtype=float)
    else:
        weights = np.asarray(sample_area_weights, dtype=float).reshape(-1)
        if weights.shape != (len(constraints),):
            raise ValueError("sample_area_weights must match samples")
    force = np.zeros(int(n_total_dofs), dtype=float)
    active = np.asarray([constraint.active for constraint in constraints], dtype=bool)
    if np.any(active):
        active_constraints = [constraint for constraint, is_active in zip(constraints, active, strict=True) if is_active]
        active_weights = weights[active]
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        for row, constraint in enumerate(active_constraints):
            normal = np.asarray(constraint.normal, dtype=float)
            for node, weight in zip(constraint.slave_node_ids, constraint.slave_weights, strict=True):
                base = int(slave_dof_offset) + 3 * int(node)
                for axis in range(3):
                    rows.append(row)
                    cols.append(base + axis)
                    vals.append(float(weight) * float(normal[axis]))
            for node, weight in zip(constraint.master_node_ids, constraint.master_weights, strict=True):
                base = int(master_dof_offset) + 3 * int(node)
                for axis in range(3):
                    rows.append(row)
                    cols.append(base + axis)
                    vals.append(-float(weight) * float(normal[axis]))
        J = coo_matrix((vals, (rows, cols)), shape=(len(active_constraints), int(n_total_dofs))).tocsr()
        scale = k * active_weights
        penetration = np.asarray([-float(constraint.g) for constraint in active_constraints], dtype=float)
        force = np.asarray(J.T @ (scale * penetration), dtype=float).ravel()
        stiffness = (J.T @ diags(scale, format="csr") @ J).tocsr()
    else:
        stiffness = csr_matrix((int(n_total_dofs), int(n_total_dofs)), dtype=float)
    return LagrangianOracleContactResponse(
        force=force,
        stiffness=stiffness.tocsr(),
        constraints=constraints,
        quadrature_weights=weights,
    )


def lagrangian_oracle_penalty_response_batch(
    slave_x_current: np.ndarray,
    samples: list[SurfaceSample] | tuple[SurfaceSample, ...],
    oracle: LagrangianSDFContactOracle,
    *,
    pressure_stiffness: float,
    n_total_dofs: int,
    sample_area_weights: np.ndarray | None = None,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
) -> LagrangianOracleContactResponse:
    """Batch-oriented alias for oracle penalty response.

    The current implementation reuses the same exact scalar oracle per sample
    while sharing cache/refit state.  It is the Python batch entry point that a
    future C++ backend can replace without changing call sites.
    """

    return lagrangian_oracle_penalty_response(
        slave_x_current,
        samples,
        oracle,
        pressure_stiffness=pressure_stiffness,
        n_total_dofs=n_total_dofs,
        sample_area_weights=sample_area_weights,
        slave_dof_offset=slave_dof_offset,
        master_dof_offset=master_dof_offset,
    )


def lagrangian_patch_pair_query(
    oracle_a: LagrangianSDFContactOracle,
    oracle_b: LagrangianSDFContactOracle,
    *,
    face_ids_a: np.ndarray | None = None,
    face_ids_b: np.ndarray | None = None,
) -> LagrangianPatchPairQueryResult:
    """Return the closest current patch pair between two material SDF oracles."""

    ids_a = np.arange(oracle_a.material.face_count, dtype=np.int64) if face_ids_a is None else np.asarray(face_ids_a, dtype=np.int64).ravel()
    ids_b = np.arange(oracle_b.material.face_count, dtype=np.int64) if face_ids_b is None else np.asarray(face_ids_b, dtype=np.int64).ravel()
    if ids_a.size == 0 or ids_b.size == 0:
        raise ValueError("face id sets must be non-empty")
    best: LagrangianPatchPairQueryResult | None = None
    best_dist2 = np.inf
    evaluated = 0
    total_pairs = int(ids_a.size * ids_b.size)
    for face_a in ids_a:
        tri_a = oracle_a.x_current[oracle_a.material.boundary_faces[int(face_a)]]
        ref_a = oracle_a.material.reference_nodes[oracle_a.material.boundary_faces[int(face_a)]]
        for face_b in ids_b:
            tri_b = oracle_b.x_current[oracle_b.material.boundary_faces[int(face_b)]]
            ref_b = oracle_b.material.reference_nodes[oracle_b.material.boundary_faces[int(face_b)]]
            point_a, point_b, bary_a, bary_b, dist2 = _closest_points_between_triangles(tri_a, tri_b)
            evaluated += 1
            if dist2 < best_dist2:
                delta = point_b - point_a
                distance = float(np.sqrt(max(dist2, 0.0)))
                if distance > 1.0e-14:
                    normal = delta / distance
                else:
                    normal = _oriented_current_normal(tri_a)
                best = LagrangianPatchPairQueryResult(
                    gap=distance,
                    normal=normal,
                    point_a=point_a,
                    point_b=point_b,
                    material_point_a=bary_a @ ref_a,
                    material_point_b=bary_b @ ref_b,
                    face_id_a=int(face_a),
                    face_id_b=int(face_b),
                    barycentric_a=bary_a,
                    barycentric_b=bary_b,
                    candidates_evaluated=total_pairs,
                )
                best_dist2 = float(dist2)
    if best is None:
        raise RuntimeError("failed to evaluate patch pair candidates")
    return best


def _closest_points_between_triangles(
    tri_a: np.ndarray,
    tri_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return closest points and barycentric coordinates on two triangles."""

    A = np.asarray(tri_a, dtype=float)
    B = np.asarray(tri_b, dtype=float)
    if A.shape != (3, 3) or B.shape != (3, 3):
        raise ValueError("triangles must have shape (3, 3)")

    best_point_a: np.ndarray | None = None
    best_point_b: np.ndarray | None = None
    best_bary_a: np.ndarray | None = None
    best_bary_b: np.ndarray | None = None
    best_dist2 = np.inf

    def keep(
        point_a: np.ndarray,
        point_b: np.ndarray,
        bary_a: np.ndarray,
        bary_b: np.ndarray,
    ) -> None:
        nonlocal best_point_a, best_point_b, best_bary_a, best_bary_b, best_dist2
        delta = point_b - point_a
        dist2 = float(delta @ delta)
        if dist2 < best_dist2:
            best_point_a = point_a.copy()
            best_point_b = point_b.copy()
            best_bary_a = bary_a.copy()
            best_bary_b = bary_b.copy()
            best_dist2 = dist2

    for idx in range(3):
        closest_b, bary_b, _dist2, _region = closest_point_on_triangle(A[idx], B[0], B[1], B[2])
        keep(A[idx], closest_b, _unit_barycentric(idx), bary_b)
    for idx in range(3):
        closest_a, bary_a, _dist2, _region = closest_point_on_triangle(B[idx], A[0], A[1], A[2])
        keep(closest_a, B[idx], bary_a, _unit_barycentric(idx))

    for edge_a in _TRIANGLE_EDGES:
        for edge_b in _TRIANGLE_EDGES:
            point_a, point_b, ta, tb, _dist2 = _closest_points_on_segments(
                A[edge_a[0]],
                A[edge_a[1]],
                B[edge_b[0]],
                B[edge_b[1]],
            )
            keep(point_a, point_b, _edge_barycentric(edge_a, ta), _edge_barycentric(edge_b, tb))

    if best_point_a is None or best_point_b is None or best_bary_a is None or best_bary_b is None:
        raise RuntimeError("failed to compute triangle pair distance")
    return best_point_a, best_point_b, best_bary_a, best_bary_b, float(best_dist2)


_TRIANGLE_EDGES: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 0))


def _unit_barycentric(index: int) -> np.ndarray:
    bary = np.zeros(3, dtype=float)
    bary[int(index)] = 1.0
    return bary


def _edge_barycentric(edge: tuple[int, int], t: float) -> np.ndarray:
    bary = np.zeros(3, dtype=float)
    bary[int(edge[0])] = 1.0 - float(t)
    bary[int(edge[1])] = float(t)
    return bary


def _closest_points_on_segments(
    p1: np.ndarray,
    q1: np.ndarray,
    p2: np.ndarray,
    q2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Return closest points between two segments.

    This is the standard clamped two-segment least-squares solve.  It handles
    degenerate segments by falling back to endpoint projection.
    """

    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)
    eps = 1.0e-15

    if a <= eps and e <= eps:
        c1 = p1.copy()
        c2 = p2.copy()
        return c1, c2, 0.0, 0.0, float(np.dot(c1 - c2, c1 - c2))
    if a <= eps:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(d1 @ r)
        if e <= eps:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(d1 @ d2)
            denom = a * e - b * b
            if denom != 0.0:
                s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
            else:
                s = 0.0
            tnom = b * s + f
            if tnom < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif tnom > e:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
            else:
                t = float(tnom / e)
    c1 = p1 + s * d1
    c2 = p2 + t * d2
    return c1, c2, float(s), float(t), float(np.dot(c1 - c2, c1 - c2))


def _pullback_initial(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Least-squares inverse from current space to patch barycentric weights."""

    a = triangle[0]
    e1 = triangle[1] - a
    e2 = triangle[2] - a
    A = np.column_stack((e1, e2))
    uv, *_ = np.linalg.lstsq(A, point - a, rcond=None)
    return _project_barycentric_to_simplex(np.asarray([1.0 - uv[0] - uv[1], uv[0], uv[1]], dtype=float))


def _closest_point_newton_on_triangle(
    point: np.ndarray,
    triangle: np.ndarray,
    barycentric0: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, int, bool]:
    """Newton corrector for ``min 0.5 ||x - chi(u,v)||^2`` on one patch."""

    bary = _project_barycentric_to_simplex(barycentric0)
    uv = np.asarray([bary[1], bary[2]], dtype=float)
    a = triangle[0]
    e1 = triangle[1] - a
    e2 = triangle[2] - a
    H = np.asarray(
        [
            [float(e1 @ e1), float(e1 @ e2)],
            [float(e2 @ e1), float(e2 @ e2)],
        ],
        dtype=float,
    )
    converged = False
    iterations = 0
    for iterations in range(1, max(1, int(max_iterations)) + 1):
        closest = a + uv[0] * e1 + uv[1] * e2
        residual = closest - point
        grad = np.asarray([float(e1 @ residual), float(e2 @ residual)], dtype=float)
        if float(np.linalg.norm(grad)) <= float(tolerance):
            converged = True
            break
        try:
            step = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(H, -grad, rcond=None)
        uv = uv + step
        projected = _project_barycentric_to_simplex(np.asarray([1.0 - uv[0] - uv[1], uv[0], uv[1]], dtype=float))
        uv = np.asarray([projected[1], projected[2]], dtype=float)
        if float(np.linalg.norm(step)) <= float(tolerance):
            converged = True
            break
    return _project_barycentric_to_simplex(np.asarray([1.0 - uv[0] - uv[1], uv[0], uv[1]], dtype=float)), iterations, converged


def _project_barycentric_to_simplex(weights: np.ndarray) -> np.ndarray:
    """Project three weights to the probability simplex."""

    w = np.asarray(weights, dtype=float).reshape(3)
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, 4, dtype=float)
    cond = u - cssv / ind > 0.0
    if not bool(np.any(cond)):
        return np.full(3, 1.0 / 3.0, dtype=float)
    rho = int(np.nonzero(cond)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    projected = np.maximum(w - theta, 0.0)
    total = float(np.sum(projected))
    if total <= 0.0:
        return np.full(3, 1.0 / 3.0, dtype=float)
    return projected / total


def _oriented_current_normal(triangle: np.ndarray) -> np.ndarray:
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("current patch has zero area")
    return normal / norm


def _push_forward_normal(deformation_gradient: np.ndarray, material_gradient: np.ndarray) -> np.ndarray:
    F = np.asarray(deformation_gradient, dtype=float)
    grad = np.asarray(material_gradient, dtype=float)
    if F.shape != (3, 3):
        raise ValueError("deformation_gradient must have shape (3, 3)")
    if grad.shape != (3,):
        raise ValueError("material_gradient must have shape (3,)")
    normal = _solve_transpose(F, grad)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("pushed-forward normal is zero")
    return normal / norm


def _solve_transpose(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    A = np.asarray(matrix, dtype=float)
    b = np.asarray(rhs, dtype=float)
    if A.shape != (3, 3) or b.shape != (3,):
        raise ValueError("matrix must have shape (3, 3) and rhs must have shape (3,)")
    try:
        return np.linalg.solve(A.T, b)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(A.T, b, rcond=None)
        return solution


def _as_nodes(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    return arr.copy()


def _as_point(value: np.ndarray, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return point
