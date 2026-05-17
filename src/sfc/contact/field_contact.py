"""Field-based contact constraints from a dynamic narrow-band SDF."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from sfc.sdf.dynamic_narrow_band_sdf import DynamicNarrowBandSDF, FieldQueryPayload

from .narrow_phase import SurfaceSample


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


def _unit(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError("normal must have shape (3,)")
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError("normal must be nonzero")
    return arr / norm
