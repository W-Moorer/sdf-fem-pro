"""Linear Newmark-beta time integration."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import spsolve

from .constraints import (
    eliminate_fixed_dofs,
    expand_reduced_vector,
    project_fixed_dofs,
)


def _as_system_matrix(matrix, n_dofs: int, name: str) -> csr_matrix:
    A = matrix.tocsr() if issparse(matrix) else csr_matrix(np.asarray(matrix, dtype=float))
    if A.shape != (n_dofs, n_dofs):
        raise ValueError(f"{name} must have shape ({n_dofs}, {n_dofs})")
    return A


def _as_vector(value: np.ndarray | Sequence[float], n_dofs: int, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=float).ravel()
    if vec.shape != (n_dofs,):
        raise ValueError(f"{name} must have shape ({n_dofs},)")
    return vec


def newmark_beta_step(
    M,
    C,
    K,
    u: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    f_ext: np.ndarray,
    *,
    dt: float,
    f_contact: np.ndarray | None = None,
    fixed_dofs: np.ndarray | Sequence[int] | None = None,
    fixed_values: float | np.ndarray | Sequence[float] | None = None,
    beta: float = 0.25,
    gamma: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance one linear Newmark-beta step.

    The solved equation is ``M a + C v + K u = f_ext + f_contact`` at the new
    time level. Fixed DOFs are eliminated from the effective linear system and
    projected onto the returned displacement, velocity, and acceleration.
    """

    dt = float(dt)
    beta = float(beta)
    gamma = float(gamma)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")

    n_dofs = np.asarray(u).size
    u0 = _as_vector(u, n_dofs, "u")
    v0 = _as_vector(v, n_dofs, "v")
    a0 = _as_vector(a, n_dofs, "a")
    f = _as_vector(f_ext, n_dofs, "f_ext")
    if f_contact is not None:
        f = f + _as_vector(f_contact, n_dofs, "f_contact")

    M = _as_system_matrix(M, n_dofs, "M")
    K = _as_system_matrix(K, n_dofs, "K")
    C = csr_matrix((n_dofs, n_dofs), dtype=float) if C is None else _as_system_matrix(C, n_dofs, "C")

    fixed = None if fixed_dofs is None else np.asarray(fixed_dofs, dtype=np.int64).ravel()
    if fixed_values is None and fixed is not None and fixed.size:
        displacement_values = u0[np.unique(fixed)]
    else:
        displacement_values = fixed_values

    u0 = project_fixed_dofs(u0, fixed, displacement_values)
    v0 = project_fixed_dofs(v0, fixed, 0.0)
    a0 = project_fixed_dofs(a0, fixed, 0.0)

    u_pred = u0 + dt * v0 + dt * dt * (0.5 - beta) * a0
    v_pred = v0 + dt * (1.0 - gamma) * a0

    c0 = 1.0 / (beta * dt * dt)
    c1 = gamma / (beta * dt)

    K_eff = K + c0 * M + c1 * C
    rhs = f + M @ (c0 * u_pred) + C @ (c1 * u_pred - v_pred)

    K_reduced, rhs_reduced, free = eliminate_fixed_dofs(
        K_eff,
        rhs,
        fixed,
        displacement_values,
    )
    u_free = spsolve(K_reduced, rhs_reduced) if free.size else np.empty(0, dtype=float)
    u_new = expand_reduced_vector(
        np.asarray(u_free, dtype=float),
        free,
        n_dofs,
        fixed,
        displacement_values,
    )

    a_new = c0 * (u_new - u_pred)
    v_new = v_pred + gamma * dt * a_new

    u_new = project_fixed_dofs(u_new, fixed, displacement_values)
    v_new = project_fixed_dofs(v_new, fixed, 0.0)
    a_new = project_fixed_dofs(a_new, fixed, 0.0)
    return u_new, v_new, a_new
