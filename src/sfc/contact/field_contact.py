"""Field-based contact constraints from a dynamic narrow-band SDF."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator

from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF, FieldQueryPayload

from .narrow_phase import SurfaceSample

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
class FieldContactConstraint:
    """One contact gap constraint queried from an interpolated SDF field."""

    g: float
    normal: np.ndarray
    gap_gradient: np.ndarray
    slave_node_ids: np.ndarray
    slave_weights: np.ndarray
    master_node_ids: np.ndarray
    master_sensitivity: np.ndarray
    x_slave: np.ndarray
    payload: FieldQueryPayload

    @property
    def active(self) -> bool:
        """Return true when the field gap is penetrating."""

        return self.g < 0.0


@dataclass(frozen=True, slots=True)
class FieldSurfaceContactResponse:
    """Area-integrated field-contact response for a slave surface."""

    force: np.ndarray
    stiffness: csr_matrix
    constraints: tuple[FieldContactConstraint, ...]
    quadrature_weights: np.ndarray
    gaps: np.ndarray | None = None
    stiffness_operator: "FieldContactMatrixFreeStiffness | None" = None

    @property
    def active_count(self) -> int:
        """Number of active quadrature constraints."""

        if self.gaps is not None:
            return int(np.count_nonzero(np.asarray(self.gaps, dtype=float) < 0.0))
        return sum(1 for constraint in self.constraints if constraint.active)

    @property
    def min_gap(self) -> float:
        """Minimum sampled field gap."""

        if self.gaps is not None:
            gap_values = np.asarray(self.gaps, dtype=float)
            return float(np.min(gap_values)) if gap_values.size else 0.0
        if not self.constraints:
            return 0.0
        return float(min(constraint.g for constraint in self.constraints))

    @property
    def max_penetration(self) -> float:
        """Maximum sampled penetration depth."""

        return max(-self.min_gap, 0.0)


@dataclass(frozen=True, slots=True)
class FieldContactMatrixFreeStiffness:
    """Matrix-free contact stiffness action ``Kc v = J^T W J v``.

    The operator stores the same active quadrature rows used by the explicit
    Gauss-Newton contact tangent.  It avoids global ``Kc`` assembly and keeps
    the result algebraically identical to ``_assemble_batch_contact_stiffness``.
    """

    n_total_dofs: int
    scale: np.ndarray
    slave_node_ids: np.ndarray
    slave_weights: np.ndarray
    gradients: np.ndarray
    face_node_ids: np.ndarray
    grid_weights: np.ndarray
    barycentric: np.ndarray
    normals: np.ndarray
    slave_dof_offset: int
    master_dof_offset: int

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        """Apply the contact tangent to a full global vector."""

        x = np.asarray(vector, dtype=float).reshape(-1)
        if x.shape != (int(self.n_total_dofs),):
            raise ValueError("vector size does not match n_total_dofs")
        try:
            from ._cpp_field_contact import contact_stiffness_matvec, is_available as cpp_available
        except Exception:
            cpp = False
        else:
            cpp = bool(cpp_available())
        if cpp:
            return contact_stiffness_matvec(
                x,
                self.scale,
                self.slave_node_ids,
                self.slave_weights,
                self.gradients,
                self.face_node_ids,
                self.grid_weights,
                self.barycentric,
                self.normals,
                int(self.n_total_dofs),
                int(self.slave_dof_offset),
                int(self.master_dof_offset),
            )
        y = np.zeros(int(self.n_total_dofs), dtype=float)
        if self.scale.size == 0:
            return y

        slave_cols = int(self.slave_dof_offset) + self.slave_node_ids[:, :, None] * 3 + np.arange(3, dtype=np.int64)
        slave_values = x[slave_cols]
        slave_jx = np.einsum("sa,sac,sc->s", self.slave_weights, slave_values, self.gradients)

        master_cols = int(self.master_dof_offset) + self.face_node_ids[:, :, :, None] * 3 + np.arange(3, dtype=np.int64)
        master_values = x[master_cols]
        master_dot = np.einsum("slc,slac->sla", self.normals, master_values)
        master_jx = -np.einsum("sl,sla,sla->s", self.grid_weights, self.barycentric, master_dot)

        row_values = self.scale * (slave_jx + master_jx)
        _accumulate_slave_force(
            y,
            self.slave_node_ids,
            self.slave_weights,
            self.gradients,
            row_values,
            int(self.slave_dof_offset),
        )
        _accumulate_master_force(
            y,
            self.face_node_ids,
            self.grid_weights,
            self.barycentric,
            self.normals,
            row_values,
            int(self.master_dof_offset),
        )
        return y

    def as_linear_operator(self, dof_indices: np.ndarray | None = None) -> LinearOperator:
        """Return a SciPy ``LinearOperator`` for all DOFs or a free-DOF subset."""

        if dof_indices is None:
            n = int(self.n_total_dofs)
            return LinearOperator((n, n), matvec=self.matvec, dtype=float)
        indices = np.asarray(dof_indices, dtype=np.int64).ravel()
        n = int(indices.size)

        def restricted_matvec(vector: np.ndarray) -> np.ndarray:
            full = np.zeros(int(self.n_total_dofs), dtype=float)
            full[indices] = np.asarray(vector, dtype=float)
            return self.matvec(full)[indices]

        return LinearOperator((n, n), matvec=restricted_matvec, dtype=float)


@dataclass(frozen=True, slots=True)
class SurfaceQuadratureCache:
    """Cached slave-triangle quadrature topology and area weights."""

    node_ids: np.ndarray
    weights: np.ndarray
    area_weights: np.ndarray

    def points(self, slave_x_current: np.ndarray) -> np.ndarray:
        """Evaluate quadrature points for the current slave coordinates."""

        X = np.asarray(slave_x_current, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError("slave_x_current must have shape (n_nodes, 3)")
        if self.node_ids.size and int(self.node_ids.max()) >= X.shape[0]:
            raise ValueError("quadrature cache references nodes outside slave_x_current")
        return np.einsum("qk,qkd->qd", self.weights, X[self.node_ids])

    def samples(self) -> list[SurfaceSample]:
        """Return ``SurfaceSample`` objects for reference implementations."""

        return [
            SurfaceSample(nodes.copy(), weights.copy(), np.empty(0, dtype=np.int64))
            for nodes, weights in zip(self.node_ids, self.weights, strict=True)
        ]

def node_to_surface_field_penalty_response(
    slave_x_current: np.ndarray,
    samples: Iterable[SurfaceSample],
    master_sdf: DynamicNarrowBandSDF,
    *,
    pressure_stiffness: float,
    n_total_dofs: int,
    sample_area_weights: np.ndarray | Sequence[float] | None = None,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
    refine: bool = False,
) -> FieldSurfaceContactResponse:
    """Assemble the legacy node/point-to-surface field-contact response.

    This function preserves the previous SFC integration path explicitly: each
    supplied ``SurfaceSample`` is an independent point constraint queried
    against the master SDF field. Optional ``sample_area_weights`` reproduce the
    former centroid-plus-area pressure integration used by validation runners.
    """

    k = float(pressure_stiffness)
    if k <= 0.0:
        raise ValueError("pressure_stiffness must be positive")
    constraints = tuple(
        compute_field_contact_constraints(
            slave_x_current,
            samples,
            master_sdf,
            refine=refine,
        )
    )
    if sample_area_weights is None:
        area_weights = np.ones(len(constraints), dtype=float)
    else:
        area_weights = np.asarray(sample_area_weights, dtype=float).ravel()
        if area_weights.shape != (len(constraints),):
            raise ValueError("sample_area_weights must match the number of samples")
    n_dofs = int(n_total_dofs)
    force = np.zeros(n_dofs, dtype=float)
    stiffness = csr_matrix((n_dofs, n_dofs), dtype=float)
    for constraint, area_weight in zip(constraints, area_weights, strict=True):
        penetration = max(-float(constraint.g), 0.0)
        if penetration <= 0.0:
            continue
        J = field_contact_jacobian_row(
            constraint,
            n_total_dofs=n_dofs,
            slave_dof_offset=slave_dof_offset,
            master_dof_offset=master_dof_offset,
        )
        lam = k * float(area_weight) * penetration
        force += np.asarray(J.T @ np.array([lam])).ravel()
        stiffness = stiffness + (k * float(area_weight)) * (J.T @ J)
    return FieldSurfaceContactResponse(
        force=force,
        stiffness=stiffness.tocsr(),
        constraints=constraints,
        quadrature_weights=area_weights,
    )


def triangle_surface_quadrature_samples(
    slave_faces: np.ndarray,
    slave_x_reference: np.ndarray,
    *,
    order: int = 3,
) -> tuple[list[SurfaceSample], np.ndarray]:
    """Build triangle surface quadrature samples and area weights.

    ``order=1`` uses one centroid point per face. ``order=3`` uses the common
    symmetric three-point rule. ``order=7`` uses a degree-five Dunavant rule and
    is the preferred surface-to-surface option for curved or concentrated
    contact because it reduces active-set aliasing at contact patch boundaries.
    """

    cache = triangle_surface_quadrature_cache(slave_faces, slave_x_reference, order=order)
    return cache.samples(), cache.area_weights.copy()


def triangle_surface_quadrature_cache(
    slave_faces: np.ndarray,
    slave_x_reference: np.ndarray,
    *,
    order: int = 3,
) -> SurfaceQuadratureCache:
    """Build reusable triangle surface quadrature topology and weights."""

    faces = np.asarray(slave_faces, dtype=np.int64)
    X = np.asarray(slave_x_reference, dtype=float)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("slave_faces must have shape (n_faces, 3)")
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("slave_x_reference must have shape (n_nodes, 3)")
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= X.shape[0]):
        raise ValueError("slave_faces reference nodes outside slave_x_reference")

    bary, unit_weights = _triangle_quadrature_rule(order)
    node_ids: list[np.ndarray] = []
    sample_weights: list[np.ndarray] = []
    weights: list[float] = []
    for face in faces:
        tri = X[face]
        area = 0.5 * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
        if area <= 0.0:
            continue
        for local_bary, local_weight in zip(bary, unit_weights, strict=True):
            node_ids.append(face.copy())
            sample_weights.append(local_bary.copy())
            weights.append(area * float(local_weight))
    if not node_ids:
        return SurfaceQuadratureCache(
            node_ids=np.empty((0, 3), dtype=np.int64),
            weights=np.empty((0, 3), dtype=float),
            area_weights=np.empty(0, dtype=float),
        )
    return SurfaceQuadratureCache(
        node_ids=np.vstack(node_ids).astype(np.int64, copy=False),
        weights=np.vstack(sample_weights).astype(float, copy=False),
        area_weights=np.asarray(weights, dtype=float),
    )


def surface_to_surface_field_penalty_response(
    slave_x_current: np.ndarray,
    slave_faces: np.ndarray,
    master_sdf: DynamicNarrowBandSDF,
    *,
    pressure_stiffness: float,
    n_total_dofs: int,
    slave_x_reference: np.ndarray | None = None,
    quadrature_order: int = 3,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
    refine: bool = False,
) -> FieldSurfaceContactResponse:
    """Assemble area-integrated surface-to-surface field-contact response.

    The gap at each slave surface quadrature point is queried from the dynamic
    SDF field. The linear pressure-overclosure law is integrated over slave
    faces as ``p = pressure_stiffness * <-g>_+``.
    """

    k = float(pressure_stiffness)
    if k <= 0.0:
        raise ValueError("pressure_stiffness must be positive")
    X_ref = np.asarray(slave_x_current if slave_x_reference is None else slave_x_reference, dtype=float)
    samples, area_weights = triangle_surface_quadrature_samples(
        slave_faces,
        X_ref,
        order=quadrature_order,
    )
    constraints = tuple(
        compute_field_contact_constraints(
            slave_x_current,
            samples,
            master_sdf,
            refine=refine,
        )
    )
    n_dofs = int(n_total_dofs)
    force = np.zeros(n_dofs, dtype=float)
    stiffness = csr_matrix((n_dofs, n_dofs), dtype=float)
    for constraint, area_weight in zip(constraints, area_weights, strict=True):
        penetration = max(-float(constraint.g), 0.0)
        if penetration <= 0.0:
            continue
        J = field_contact_jacobian_row(
            constraint,
            n_total_dofs=n_dofs,
            slave_dof_offset=slave_dof_offset,
            master_dof_offset=master_dof_offset,
        )
        lam = k * float(area_weight) * penetration
        force += np.asarray(J.T @ np.array([lam])).ravel()
        stiffness = stiffness + (k * float(area_weight)) * (J.T @ J)
    return FieldSurfaceContactResponse(
        force=force,
        stiffness=stiffness.tocsr(),
        constraints=constraints,
        quadrature_weights=area_weights,
    )


def surface_to_surface_field_penalty_response_vectorized(
    slave_x_current: np.ndarray,
    slave_faces: np.ndarray,
    master_sdf: DynamicNarrowBandSDF,
    *,
    pressure_stiffness: float,
    n_total_dofs: int,
    slave_x_reference: np.ndarray | None = None,
    quadrature_order: int = 3,
    quadrature_cache: SurfaceQuadratureCache | None = None,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
    assemble_stiffness: bool = False,
    matrix_free_stiffness: bool = False,
) -> FieldSurfaceContactResponse:
    """Vectorized force assembly for area-integrated field contact.

    This is algebraically equivalent to
    ``surface_to_surface_field_penalty_response`` for the force vector, but it
    avoids per-point ``FieldContactConstraint`` objects and performs SDF field
    interpolation and nodal force accumulation in batches.  The optional
    tangent can be returned either as an explicit sparse matrix or as a
    matrix-free ``J^T W J`` operator.
    """

    k = float(pressure_stiffness)
    if k <= 0.0:
        raise ValueError("pressure_stiffness must be positive")
    X = np.asarray(slave_x_current, dtype=float)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("slave_x_current must have shape (n_nodes, 3)")
    if quadrature_cache is None:
        X_ref = np.asarray(X if slave_x_reference is None else slave_x_reference, dtype=float)
        cache = triangle_surface_quadrature_cache(slave_faces, X_ref, order=quadrature_order)
    else:
        cache = quadrature_cache
    points = cache.points(X)
    if bool(assemble_stiffness) and bool(matrix_free_stiffness):
        raise ValueError("assemble_stiffness and matrix_free_stiffness are mutually exclusive")
    if compiled_field_contact_available() and not bool(assemble_stiffness) and not bool(matrix_free_stiffness):
        try:
            from ._cpp_field_contact import is_available as cpp_contact_available
            from ._cpp_field_contact import surface_penalty_response
        except Exception:
            cpp_contact_available = lambda: False  # type: ignore[assignment]
        if not cpp_contact_available():
            from ._numba_field_contact import surface_penalty_response

        force, gaps = surface_penalty_response(
            points,
            cache.node_ids,
            cache.weights,
            cache.area_weights,
            master_sdf.grid.origin,
            master_sdf.grid.spacing,
            np.asarray(master_sdf.grid.shape, dtype=np.int64),
            master_sdf.grid.phi,
            master_sdf.grid.valid_mask,
            master_sdf.grid.closest_face_id,
            master_sdf.grid.barycentric,
            master_sdf.grid.closest_normal,
            master_sdf.boundary_faces,
            k,
            n_total_dofs,
            slave_dof_offset,
            master_dof_offset,
        )
        return FieldSurfaceContactResponse(
            force=force,
            stiffness=csr_matrix((int(n_total_dofs), int(n_total_dofs)), dtype=float),
            constraints=(),
            quadrature_weights=cache.area_weights.copy(),
            gaps=gaps.copy(),
        )

    batch = _field_query_batch(master_sdf, points)
    gaps = batch["gaps"]
    penetration = np.maximum(-gaps, 0.0)
    lambdas = k * cache.area_weights * penetration

    n_dofs = int(n_total_dofs)
    force = np.zeros(n_dofs, dtype=float)
    active = lambdas > 0.0
    if bool(np.any(active)):
        active_lam = lambdas[active]
        _accumulate_slave_force(
            force,
            cache.node_ids[active],
            cache.weights[active],
            batch["gradients"][active],
            active_lam,
            int(slave_dof_offset),
        )
        _accumulate_master_force(
            force,
            batch["face_node_ids"][active],
            batch["weights"][active],
            batch["barycentric"][active],
            batch["normals"][active],
            active_lam,
            int(master_dof_offset),
        )
    stiffness = (
        _assemble_batch_contact_stiffness(
            cache=cache,
            batch=batch,
            active=active,
            pressure_stiffness=k,
            n_total_dofs=n_dofs,
            slave_dof_offset=int(slave_dof_offset),
            master_dof_offset=int(master_dof_offset),
        )
        if bool(assemble_stiffness)
        else csr_matrix((n_dofs, n_dofs), dtype=float)
    )
    stiffness_operator = (
        _make_batch_contact_matrix_free_stiffness(
            cache=cache,
            batch=batch,
            active=active,
            pressure_stiffness=k,
            n_total_dofs=n_dofs,
            slave_dof_offset=int(slave_dof_offset),
            master_dof_offset=int(master_dof_offset),
        )
        if bool(matrix_free_stiffness)
        else None
    )

    return FieldSurfaceContactResponse(
        force=force,
        stiffness=stiffness,
        constraints=(),
        quadrature_weights=cache.area_weights.copy(),
        gaps=gaps.copy(),
        stiffness_operator=stiffness_operator,
    )


def compiled_field_contact_available() -> bool:
    """Return whether the optional compiled field-contact backend is available."""

    try:
        from ._cpp_field_contact import is_available as cpp_available
    except Exception:
        cpp = False
    else:
        cpp = bool(cpp_available())
    if cpp:
        return True
    try:
        from ._numba_field_contact import is_available
    except Exception:
        return False
    return bool(is_available())


def field_contact_constraint_from_sample(
    slave_x_current: np.ndarray,
    sample: SurfaceSample,
    master_sdf: DynamicNarrowBandSDF,
    *,
    refine: bool = False,
) -> FieldContactConstraint:
    """Compute one field-contact constraint from a slave sample.

    By default the query is interpolation-only. Setting ``refine=True`` uses an
    explicit local projection refinement for the returned gap and normal while
    keeping the field payload sensitivity for the master block.
    """

    x_slave = sample.point(slave_x_current)
    payload = master_sdf.query_payload(x_slave)
    if refine:
        refined = master_sdf.refine_query_projection(x_slave)
        g = float(refined.g)
        normal = _unit(np.asarray(refined.n, dtype=float))
        gap_gradient = normal.copy()
    else:
        g = master_sdf.query_phi(x_slave)
        gap_gradient = master_sdf.query_spatial_derivative_phi(x_slave)
        try:
            normal = _unit(gap_gradient)
        except ValueError:
            normal = master_sdf.query_geometric_normal(x_slave)
            gap_gradient = normal.copy()

    master_node_ids, master_sensitivity = payload.master_sensitivity_terms()
    return FieldContactConstraint(
        g=float(g),
        normal=np.asarray(normal, dtype=float),
        gap_gradient=np.asarray(gap_gradient, dtype=float),
        slave_node_ids=sample.node_ids.copy(),
        slave_weights=sample.weights.copy(),
        master_node_ids=master_node_ids,
        master_sensitivity=master_sensitivity,
        x_slave=x_slave,
        payload=payload,
    )


def compute_field_contact_constraints(
    slave_x_current: np.ndarray,
    samples: Iterable[SurfaceSample],
    master_sdf: DynamicNarrowBandSDF,
    *,
    refine: bool = False,
) -> list[FieldContactConstraint]:
    """Compute field-contact constraints for all supplied samples."""

    return [
        field_contact_constraint_from_sample(
            slave_x_current,
            sample,
            master_sdf,
            refine=refine,
        )
        for sample in samples
    ]


def field_contact_jacobian_entries(
    constraint: FieldContactConstraint,
    *,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sparse row entries for one field-contact gap Jacobian."""

    grad = np.asarray(constraint.gap_gradient, dtype=float)
    if grad.shape != (3,):
        raise ValueError("constraint gap_gradient must have shape (3,)")
    cols: list[int] = []
    vals: list[float] = []

    for node, weight in zip(constraint.slave_node_ids, constraint.slave_weights, strict=True):
        base = int(slave_dof_offset) + int(node) * 3
        for component in range(3):
            cols.append(base + component)
            vals.append(float(weight) * grad[component])

    for node, vector in zip(constraint.master_node_ids, constraint.master_sensitivity, strict=True):
        base = int(master_dof_offset) + int(node) * 3
        for component in range(3):
            cols.append(base + component)
            vals.append(float(vector[component]))

    return np.asarray(cols, dtype=np.int64), np.asarray(vals, dtype=float)


def field_contact_jacobian_row(
    constraint: FieldContactConstraint,
    *,
    n_total_dofs: int,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
) -> csr_matrix:
    """Return one sparse row for a field-contact gap constraint."""

    cols, vals = field_contact_jacobian_entries(
        constraint,
        slave_dof_offset=slave_dof_offset,
        master_dof_offset=master_dof_offset,
    )
    rows = np.zeros(cols.size, dtype=np.int64)
    return coo_matrix((vals, (rows, cols)), shape=(1, int(n_total_dofs))).tocsr()


def assemble_field_contact_jacobian(
    constraints: Sequence[FieldContactConstraint],
    *,
    n_total_dofs: int,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
) -> csr_matrix:
    """Assemble field-contact gap Jacobian rows."""

    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_vals: list[np.ndarray] = []
    for row_id, constraint in enumerate(constraints):
        cols, vals = field_contact_jacobian_entries(
            constraint,
            slave_dof_offset=slave_dof_offset,
            master_dof_offset=master_dof_offset,
        )
        all_rows.append(np.full(cols.size, row_id, dtype=np.int64))
        all_cols.append(cols)
        all_vals.append(vals)
    if not all_vals:
        return csr_matrix((0, int(n_total_dofs)), dtype=float)
    return coo_matrix(
        (
            np.concatenate(all_vals),
            (np.concatenate(all_rows), np.concatenate(all_cols)),
        ),
        shape=(len(constraints), int(n_total_dofs)),
    ).tocsr()


def field_penalty_contact_response(
    constraints: Sequence[FieldContactConstraint],
    *,
    stiffness: float,
    n_total_dofs: int,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
) -> tuple[np.ndarray, csr_matrix]:
    """Assemble penalty force and Gauss-Newton stiffness from field contacts."""

    k = float(stiffness)
    if k <= 0.0:
        raise ValueError("stiffness must be positive")
    n_dofs = int(n_total_dofs)
    force = np.zeros(n_dofs, dtype=float)
    K = csr_matrix((n_dofs, n_dofs), dtype=float)
    for constraint in constraints:
        penetration = max(-float(constraint.g), 0.0)
        if penetration <= 0.0:
            continue
        J = field_contact_jacobian_row(
            constraint,
            n_total_dofs=n_dofs,
            slave_dof_offset=slave_dof_offset,
            master_dof_offset=master_dof_offset,
        )
        lam = k * penetration
        force += np.asarray(J.T @ np.array([lam])).ravel()
        K = K + k * (J.T @ J)
    return force, K.tocsr()


def _field_query_batch(master_sdf: DynamicNarrowBandSDF, points: np.ndarray) -> dict[str, np.ndarray]:
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    grid = master_sdf.grid
    if P.shape[0] == 0:
        return {
            "gaps": np.empty(0, dtype=float),
            "gradients": np.empty((0, 3), dtype=float),
            "weights": np.empty((0, 8), dtype=float),
            "face_node_ids": np.empty((0, 8, 3), dtype=np.int64),
            "barycentric": np.empty((0, 8, 3), dtype=float),
            "normals": np.empty((0, 8, 3), dtype=float),
        }

    lower = grid.bounds_min
    upper = grid.bounds_max
    if bool(np.any(P < lower - 1.0e-12) or np.any(P > upper + 1.0e-12)):
        raise ValueError("query point is outside the narrow-band grid")

    u = (P - grid.origin) / grid.spacing
    shape = np.asarray(grid.shape, dtype=np.int64)
    base = np.floor(u).astype(np.int64)
    frac = u - base.astype(float)
    for axis in range(3):
        high = base[:, axis] >= shape[axis] - 1
        low = base[:, axis] < 0
        base[high, axis] = shape[axis] - 2
        frac[high, axis] = 1.0
        base[low, axis] = 0
        frac[low, axis] = 0.0

    idx = base[:, None, :] + _TRILINEAR_CORNERS[None, :, :]
    valid = grid.valid_mask[idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]]
    if not bool(np.all(valid)):
        raise ValueError("query point is outside the valid narrow band")

    weights = np.ones((P.shape[0], 8), dtype=float)
    for axis in range(3):
        weights *= np.where(
            _TRILINEAR_CORNERS[None, :, axis] == 0,
            1.0 - frac[:, None, axis],
            frac[:, None, axis],
        )
    weight_gradients = _batch_weight_gradients(frac, grid.spacing)
    phi = grid.phi[idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]]
    face_ids = grid.closest_face_id[idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]]
    if bool(np.any(face_ids < 0)):
        raise ValueError("query point includes invalid grid-node payload")
    barycentric = grid.barycentric[idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]]
    normals = grid.closest_normal[idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]]
    return {
        "gaps": np.sum(weights * phi, axis=1),
        "gradients": np.sum(phi[:, :, None] * weight_gradients, axis=1),
        "weights": weights,
        "face_node_ids": master_sdf.boundary_faces[face_ids],
        "barycentric": barycentric,
        "normals": normals,
    }


def _batch_weight_gradients(frac: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    gradients = np.empty((frac.shape[0], 8, 3), dtype=float)
    for corner_id, corner in enumerate(_TRILINEAR_CORNERS):
        for axis in range(3):
            value = np.full(frac.shape[0], 1.0 / float(spacing[axis]), dtype=float)
            if corner[axis] == 0:
                value *= -1.0
            for other_axis in range(3):
                if other_axis == axis:
                    continue
                value *= frac[:, other_axis] if corner[other_axis] == 1 else 1.0 - frac[:, other_axis]
            gradients[:, corner_id, axis] = value
    return gradients


def _accumulate_slave_force(
    force: np.ndarray,
    node_ids: np.ndarray,
    sample_weights: np.ndarray,
    gradients: np.ndarray,
    lambdas: np.ndarray,
    dof_offset: int,
) -> None:
    contrib = lambdas[:, None, None] * sample_weights[:, :, None] * gradients[:, None, :]
    nodes = np.asarray(node_ids, dtype=np.int64).reshape(-1)
    values = contrib.reshape((-1, 3))
    for component in range(3):
        np.add.at(force, int(dof_offset) + nodes * 3 + component, values[:, component])


def _accumulate_master_force(
    force: np.ndarray,
    face_node_ids: np.ndarray,
    grid_weights: np.ndarray,
    barycentric: np.ndarray,
    normals: np.ndarray,
    lambdas: np.ndarray,
    dof_offset: int,
) -> None:
    contrib = (
        -lambdas[:, None, None, None]
        * grid_weights[:, :, None, None]
        * barycentric[:, :, :, None]
        * normals[:, :, None, :]
    )
    nodes = np.asarray(face_node_ids, dtype=np.int64).reshape(-1)
    values = contrib.reshape((-1, 3))
    for component in range(3):
        np.add.at(force, int(dof_offset) + nodes * 3 + component, values[:, component])


def _assemble_batch_contact_stiffness(
    *,
    cache: SurfaceQuadratureCache,
    batch: dict[str, np.ndarray],
    active: np.ndarray,
    pressure_stiffness: float,
    n_total_dofs: int,
    slave_dof_offset: int,
    master_dof_offset: int,
) -> csr_matrix:
    active = np.asarray(active, dtype=bool)
    if not bool(np.any(active)):
        return csr_matrix((int(n_total_dofs), int(n_total_dofs)), dtype=float)
    active_rows = np.arange(int(np.count_nonzero(active)), dtype=np.int64)
    row_scale = np.sqrt(float(pressure_stiffness) * cache.area_weights[active])

    slave_nodes = cache.node_ids[active]
    slave_weights = cache.weights[active]
    gradients = batch["gradients"][active]
    slave_cols = int(slave_dof_offset) + slave_nodes[:, :, None] * 3 + np.arange(3, dtype=np.int64)
    slave_vals = slave_weights[:, :, None] * gradients[:, None, :]
    slave_rows = np.repeat(active_rows, slave_cols.shape[1] * slave_cols.shape[2])
    slave_vals_flat = slave_vals.reshape(-1) * np.repeat(row_scale, slave_cols.shape[1] * slave_cols.shape[2])

    face_node_ids = batch["face_node_ids"][active]
    grid_weights = batch["weights"][active]
    barycentric = batch["barycentric"][active]
    normals = batch["normals"][active]
    master_cols = int(master_dof_offset) + face_node_ids[:, :, :, None] * 3 + np.arange(3, dtype=np.int64)
    master_vals = -grid_weights[:, :, None, None] * barycentric[:, :, :, None] * normals[:, :, None, :]
    master_rows = np.repeat(active_rows, master_cols.shape[1] * master_cols.shape[2] * master_cols.shape[3])
    master_vals_flat = master_vals.reshape(-1) * np.repeat(
        row_scale,
        master_cols.shape[1] * master_cols.shape[2] * master_cols.shape[3],
    )

    rows = np.concatenate((slave_rows, master_rows))
    cols = np.concatenate((slave_cols.reshape(-1), master_cols.reshape(-1)))
    vals = np.concatenate((slave_vals_flat, master_vals_flat))
    J = coo_matrix((vals, (rows, cols)), shape=(active_rows.size, int(n_total_dofs))).tocsr()
    return (J.T @ J).tocsr()


def _make_batch_contact_matrix_free_stiffness(
    *,
    cache: SurfaceQuadratureCache,
    batch: dict[str, np.ndarray],
    active: np.ndarray,
    pressure_stiffness: float,
    n_total_dofs: int,
    slave_dof_offset: int,
    master_dof_offset: int,
) -> FieldContactMatrixFreeStiffness:
    active = np.asarray(active, dtype=bool)
    if not bool(np.any(active)):
        return FieldContactMatrixFreeStiffness(
            n_total_dofs=int(n_total_dofs),
            scale=np.empty(0, dtype=float),
            slave_node_ids=np.empty((0, 3), dtype=np.int64),
            slave_weights=np.empty((0, 3), dtype=float),
            gradients=np.empty((0, 3), dtype=float),
            face_node_ids=np.empty((0, 8, 3), dtype=np.int64),
            grid_weights=np.empty((0, 8), dtype=float),
            barycentric=np.empty((0, 8, 3), dtype=float),
            normals=np.empty((0, 8, 3), dtype=float),
            slave_dof_offset=int(slave_dof_offset),
            master_dof_offset=int(master_dof_offset),
        )
    return FieldContactMatrixFreeStiffness(
        n_total_dofs=int(n_total_dofs),
        scale=float(pressure_stiffness) * np.asarray(cache.area_weights[active], dtype=float).copy(),
        slave_node_ids=np.asarray(cache.node_ids[active], dtype=np.int64).copy(),
        slave_weights=np.asarray(cache.weights[active], dtype=float).copy(),
        gradients=np.asarray(batch["gradients"][active], dtype=float).copy(),
        face_node_ids=np.asarray(batch["face_node_ids"][active], dtype=np.int64).copy(),
        grid_weights=np.asarray(batch["weights"][active], dtype=float).copy(),
        barycentric=np.asarray(batch["barycentric"][active], dtype=float).copy(),
        normals=np.asarray(batch["normals"][active], dtype=float).copy(),
        slave_dof_offset=int(slave_dof_offset),
        master_dof_offset=int(master_dof_offset),
    )


def _triangle_quadrature_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    if int(order) == 1:
        return (
            np.asarray([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=float),
            np.asarray([1.0], dtype=float),
        )
    if int(order) == 3:
        return (
            np.asarray(
                [
                    [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
                ],
                dtype=float,
            ),
            np.full(3, 1.0 / 3.0, dtype=float),
        )
    if int(order) == 7:
        a = 0.059715871789770
        b = 0.470142064105115
        c = 0.797426985353087
        d = 0.101286507323456
        return (
            np.asarray(
                [
                    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    [a, b, b],
                    [b, a, b],
                    [b, b, a],
                    [c, d, d],
                    [d, c, d],
                    [d, d, c],
                ],
                dtype=float,
            ),
            np.asarray(
                [
                    0.225000000000000,
                    0.132394152788506,
                    0.132394152788506,
                    0.132394152788506,
                    0.125939180544827,
                    0.125939180544827,
                    0.125939180544827,
                ],
                dtype=float,
            ),
        )
    raise ValueError("order must be one of 1, 3, or 7")


def _unit(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError("normal must have shape (3,)")
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError("normal must be nonzero")
    return arr / norm
