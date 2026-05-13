"""Dirichlet constraint utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix, issparse


_COMPONENT_INDEX = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}


def _normalize_components(components: str | Sequence[str | int]) -> np.ndarray:
    if isinstance(components, str):
        labels: Sequence[str | int]
        if components.lower() in {"all", "xyz"}:
            labels = ("x", "y", "z")
        else:
            labels = tuple(components.lower())
    else:
        labels = components

    try:
        comp = np.array([_COMPONENT_INDEX[c] for c in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("components must contain only x/y/z or 0/1/2") from exc

    if comp.size == 0:
        raise ValueError("at least one component is required")
    return np.unique(comp)


def _normalize_fixed_dofs(fixed_dofs: np.ndarray | Sequence[int] | None) -> np.ndarray:
    if fixed_dofs is None:
        return np.empty(0, dtype=np.int64)
    fixed = np.asarray(fixed_dofs, dtype=np.int64).ravel()
    if np.any(fixed < 0):
        raise ValueError("fixed DOF indices cannot be negative")
    return np.unique(fixed)


def _normalize_fixed_values(
    fixed_values: float | np.ndarray | Sequence[float] | None,
    n_fixed: int,
) -> np.ndarray:
    if fixed_values is None:
        return np.zeros(n_fixed, dtype=float)

    values = np.asarray(fixed_values, dtype=float)
    if values.ndim == 0:
        return np.full(n_fixed, float(values), dtype=float)
    values = values.ravel()
    if values.shape != (n_fixed,):
        raise ValueError(f"fixed_values must have shape ({n_fixed},)")
    return values.copy()


def fixed_dofs_from_node_set(
    nodes: np.ndarray | Sequence[int],
    components: str | Sequence[str | int] = "xyz",
) -> np.ndarray:
    """Return sorted global DOF indices for node indices and components."""

    node_ids = np.asarray(nodes, dtype=np.int64).ravel()
    if np.any(node_ids < 0):
        raise ValueError("node indices cannot be negative")

    comp = _normalize_components(components)
    dofs = node_ids[:, None] * 3 + comp[None, :]
    return np.unique(dofs.ravel())


def free_dofs(n_dofs: int, fixed_dofs: np.ndarray | Sequence[int] | None) -> np.ndarray:
    """Return sorted DOF indices not included in ``fixed_dofs``."""

    n_dofs = int(n_dofs)
    if n_dofs < 0:
        raise ValueError("n_dofs must be non-negative")

    fixed = _normalize_fixed_dofs(fixed_dofs)
    if fixed.size and int(fixed.max()) >= n_dofs:
        raise ValueError("fixed DOF index is outside the system")

    is_free = np.ones(n_dofs, dtype=bool)
    is_free[fixed] = False
    return np.nonzero(is_free)[0]


def eliminate_fixed_dofs(
    matrix,
    rhs: np.ndarray,
    fixed_dofs: np.ndarray | Sequence[int] | None,
    fixed_values: float | np.ndarray | Sequence[float] | None = None,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Eliminate fixed DOFs from a linear system.

    Returns ``(matrix_ff, rhs_f, free)``. The returned right-hand side includes
    the contribution from nonzero fixed values.
    """

    A = matrix.tocsr() if issparse(matrix) else csr_matrix(np.asarray(matrix, dtype=float))
    b = np.asarray(rhs, dtype=float).ravel()
    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square")
    if b.shape != (A.shape[0],):
        raise ValueError("rhs length must match matrix size")

    fixed = _normalize_fixed_dofs(fixed_dofs)
    if fixed.size and int(fixed.max()) >= A.shape[0]:
        raise ValueError("fixed DOF index is outside the system")

    free = free_dofs(A.shape[0], fixed)
    if fixed.size == 0:
        return A, b.copy(), free

    values = _normalize_fixed_values(fixed_values, fixed.size)
    reduced_rhs = b[free] - A[free][:, fixed] @ values
    return A[free][:, free].tocsr(), np.asarray(reduced_rhs).ravel(), free


def expand_reduced_vector(
    reduced: np.ndarray,
    free: np.ndarray,
    n_dofs: int,
    fixed_dofs: np.ndarray | Sequence[int] | None = None,
    fixed_values: float | np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Expand a reduced free-DOF vector back to the full DOF space."""

    out = np.zeros(int(n_dofs), dtype=float)
    free = np.asarray(free, dtype=np.int64).ravel()
    reduced = np.asarray(reduced, dtype=float).ravel()
    if reduced.shape != free.shape:
        raise ValueError("reduced vector length must match free DOFs")
    out[free] = reduced
    return project_fixed_dofs(out, fixed_dofs, fixed_values)


def project_fixed_dofs(
    vector: np.ndarray,
    fixed_dofs: np.ndarray | Sequence[int] | None,
    fixed_values: float | np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Return a copy of ``vector`` with fixed DOFs set to fixed values."""

    out = np.asarray(vector, dtype=float).copy()
    fixed = _normalize_fixed_dofs(fixed_dofs)
    if fixed.size == 0:
        return out
    if int(fixed.max()) >= out.size:
        raise ValueError("fixed DOF index is outside the vector")

    out[fixed] = _normalize_fixed_values(fixed_values, fixed.size)
    return out
