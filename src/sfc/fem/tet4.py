"""Linear tetrahedral element routines."""

from __future__ import annotations

import numpy as np

from .material import isotropic_linear_elasticity_matrix


def _as_tet4_coordinates(Xe: np.ndarray) -> np.ndarray:
    X = np.asarray(Xe, dtype=float)
    if X.shape != (4, 3):
        raise ValueError("tet4 coordinates must have shape (4, 3)")
    return X


def tet4_volume(Xe: np.ndarray) -> float:
    """Return the positive volume of a linear tetrahedron."""

    X = _as_tet4_coordinates(Xe)
    jac = np.column_stack((X[1] - X[0], X[2] - X[0], X[3] - X[0]))
    volume = abs(float(np.linalg.det(jac))) / 6.0
    if volume <= 0.0:
        raise ValueError("tet4 element has non-positive volume")
    return volume


def tet4_shape_function_gradients(Xe: np.ndarray) -> np.ndarray:
    """Return constant shape-function gradients with shape ``(4, 3)``."""

    X = _as_tet4_coordinates(Xe)
    vandermonde = np.column_stack((np.ones(4), X))
    try:
        coeff = np.linalg.inv(vandermonde)
    except np.linalg.LinAlgError as exc:
        raise ValueError("tet4 element is degenerate") from exc

    grads = coeff[1:4, :].T.copy()
    tet4_volume(X)
    return grads


def tet4_strain_displacement_matrix(Xe: np.ndarray) -> np.ndarray:
    """Return the TET4 strain-displacement matrix with shape ``(6, 12)``."""

    grads = tet4_shape_function_gradients(Xe)
    B = np.zeros((6, 12), dtype=float)

    for i, (dN_dx, dN_dy, dN_dz) in enumerate(grads):
        col = 3 * i
        B[0, col] = dN_dx
        B[1, col + 1] = dN_dy
        B[2, col + 2] = dN_dz
        B[3, col] = dN_dy
        B[3, col + 1] = dN_dx
        B[4, col + 1] = dN_dz
        B[4, col + 2] = dN_dy
        B[5, col] = dN_dz
        B[5, col + 2] = dN_dx

    return B


def tet4_stiffness(Xe: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Return the 12-by-12 linear elastic TET4 stiffness matrix."""

    C = isotropic_linear_elasticity_matrix(E, nu)
    B = tet4_strain_displacement_matrix(Xe)
    V = tet4_volume(Xe)
    return V * (B.T @ C @ B)


def tet4_consistent_mass(Xe: np.ndarray, rho: float) -> np.ndarray:
    """Return the 12-by-12 consistent TET4 mass matrix."""

    rho = float(rho)
    if rho <= 0.0:
        raise ValueError("density rho must be positive")

    V = tet4_volume(Xe)
    scalar = np.full((4, 4), 1.0, dtype=float)
    np.fill_diagonal(scalar, 2.0)
    scalar *= rho * V / 20.0
    return np.kron(scalar, np.eye(3))


def tet4_lumped_mass(Xe: np.ndarray, rho: float) -> np.ndarray:
    """Return the 12-by-12 row-sum lumped TET4 mass matrix."""

    rho = float(rho)
    if rho <= 0.0:
        raise ValueError("density rho must be positive")

    nodal_mass = rho * tet4_volume(Xe) / 4.0
    return np.eye(12, dtype=float) * nodal_mass


def tet4_mass(Xe: np.ndarray, rho: float, *, kind: str = "consistent") -> np.ndarray:
    """Return a TET4 mass matrix.

    ``kind`` may be ``"consistent"`` or ``"lumped"``.
    """

    normalized = kind.lower()
    if normalized == "consistent":
        return tet4_consistent_mass(Xe, rho)
    if normalized == "lumped":
        return tet4_lumped_mass(Xe, rho)
    raise ValueError("kind must be 'consistent' or 'lumped'")
