"""Penalty contact force and Gauss-Newton stiffness assembly."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .jacobian import contact_jacobian_row
from .narrow_phase import ContactConstraint


def penalty_contact_response(
    constraints: Sequence[ContactConstraint],
    *,
    stiffness: float,
    n_total_dofs: int,
    slave_dof_offset: int = 0,
    master_dof_offset: int = 0,
) -> tuple[np.ndarray, csr_matrix]:
    """Assemble penalty contact force and Gauss-Newton contact stiffness."""

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

        J = contact_jacobian_row(
            constraint,
            n_total_dofs=n_dofs,
            slave_dof_offset=slave_dof_offset,
            master_dof_offset=master_dof_offset,
        )
        lam = k * penetration
        force += np.asarray(J.T @ np.array([lam])).ravel()
        K = K + k * (J.T @ J)

    return force, K.tocsr()
