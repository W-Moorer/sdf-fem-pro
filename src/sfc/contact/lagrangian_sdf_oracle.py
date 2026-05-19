"""Lagrangian material-SDF contact oracle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from sfc.fem.deformation_map import FEMDeformationMap
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
    bvh: ReferencePatchBVH = field(init=False)
    _patch_cache: dict[Hashable, int] = field(default_factory=dict, init=False, repr=False)
    _barycentric_cache: dict[Hashable, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        X = _as_nodes(self.x_current, "x_current")
        if X.shape != self.material.reference_nodes.shape:
            raise ValueError("x_current must match material.reference_nodes shape")
        if self.search_radius is not None and float(self.search_radius) < 0.0:
            raise ValueError("search_radius must be non-negative")
        if self.deformation_map is not None and self.deformation_map.mesh.X.shape != self.material.reference_nodes.shape:
            raise ValueError("deformation_map mesh must use the same reference nodes as material")
        object.__setattr__(self, "x_current", X)
        object.__setattr__(self, "bvh", self.material.build_patch_bvh(X, padding=self.search_radius))
        if self.deformation_map is not None and not np.allclose(self.deformation_map.x_current, X):
            self.deformation_map = self.deformation_map.with_current(X)

    def refit(self, x_current: np.ndarray) -> None:
        """Refit current patch AABBs without rebuilding a current SDF grid."""

        X = _as_nodes(x_current, "x_current")
        if X.shape != self.material.reference_nodes.shape:
            raise ValueError("x_current must match material.reference_nodes shape")
        self.x_current = X
        self.bvh = self.bvh.refit(X, padding=self.search_radius)
        if self.deformation_map is not None:
            self.deformation_map = self.deformation_map.with_current(X)

    def clear_cache(self) -> None:
        """Clear active patch hints."""

        self._patch_cache.clear()
        self._barycentric_cache.clear()

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
        best: LagrangianSDFQueryResult | None = None
        best_abs_gap = np.inf
        best_dist2 = np.inf
        for candidate_id in candidates:
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
        preferred_element = self.deformation_map.element_for_face(face)
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
    stiffness = csr_matrix((int(n_total_dofs), int(n_total_dofs)), dtype=float)
    for constraint, area_weight in zip(constraints, weights, strict=True):
        penetration = max(-float(constraint.g), 0.0)
        if penetration <= 0.0:
            continue
        J = lagrangian_oracle_jacobian_row(
            constraint,
            n_total_dofs=int(n_total_dofs),
            slave_dof_offset=int(slave_dof_offset),
            master_dof_offset=int(master_dof_offset),
        )
        lam = k * float(area_weight) * penetration
        force += np.asarray(J.T @ np.asarray([lam], dtype=float)).ravel()
        stiffness = stiffness + (k * float(area_weight)) * (J.T @ J)
    return LagrangianOracleContactResponse(
        force=force,
        stiffness=stiffness.tocsr(),
        constraints=constraints,
        quadrature_weights=weights,
    )


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
    try:
        normal = np.linalg.solve(F.T, grad)
    except np.linalg.LinAlgError:
        normal, *_ = np.linalg.lstsq(F.T, grad, rcond=None)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("pushed-forward normal is zero")
    return normal / norm


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
