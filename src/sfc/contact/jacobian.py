"""Contact gap Jacobian assembly."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .narrow_phase import ContactConstraint


def contact_jacobian_entries(
    constraint: ContactConstraint,
    *,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return column indices and values for one contact gap Jacobian row."""

    n = np.asarray(constraint.n, dtype=float)
    if n.shape != (3,):
        raise ValueError("constraint normal must have shape (3,)")

    cols: list[int] = []
    vals: list[float] = []

    for node, weight in zip(constraint.slave_node_ids, constraint.slave_weights):
        base = int(slave_dof_offset) + int(node) * 3
        for component in range(3):
            cols.append(base + component)
            vals.append(float(weight) * n[component])

    for node, weight in zip(constraint.master_node_ids, constraint.master_weights):
        base = int(master_dof_offset) + int(node) * 3
        for component in range(3):
            cols.append(base + component)
            vals.append(-float(weight) * n[component])

    return np.asarray(cols, dtype=np.int64), np.asarray(vals, dtype=float)


def contact_jacobian_row(
    constraint: ContactConstraint,
    *,
    n_total_dofs: int,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
) -> csr_matrix:
    """Return one sparse contact gap Jacobian row."""

    cols, vals = contact_jacobian_entries(
        constraint,
        slave_dof_offset=slave_dof_offset,
        master_dof_offset=master_dof_offset,
    )
    rows = np.zeros(cols.size, dtype=np.int64)
    return coo_matrix((vals, (rows, cols)), shape=(1, int(n_total_dofs))).tocsr()


def assemble_contact_jacobian(
    constraints: Sequence[ContactConstraint],
    *,
    n_total_dofs: int,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
) -> csr_matrix:
    """Assemble sparse rows for multiple contact gap constraints."""

    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    all_vals: list[np.ndarray] = []

    for row_id, constraint in enumerate(constraints):
        cols, vals = contact_jacobian_entries(
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
