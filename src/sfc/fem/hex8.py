"""Linear HEX8/C3D8 element routines."""

from __future__ import annotations

import numpy as np

from .material import isotropic_linear_elasticity_matrix

C3D8_NATURAL_NODE_COORDS = np.asarray(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=float,
)
HEX8_CENTER_NATURAL_GRADIENTS = C3D8_NATURAL_NODE_COORDS / 8.0
_GAUSS_POINTS = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))


def _as_hex8_coordinates(Xe: np.ndarray) -> np.ndarray:
    X = np.asarray(Xe, dtype=float)
    if X.shape != (8, 3):
        raise ValueError("hex8 coordinates must have shape (8, 3)")
    return X


def hex8_shape_functions(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Return trilinear HEX8 shape functions at natural coordinates."""

    natural = C3D8_NATURAL_NODE_COORDS
    return 0.125 * (
        (1.0 + natural[:, 0] * float(xi))
        * (1.0 + natural[:, 1] * float(eta))
        * (1.0 + natural[:, 2] * float(zeta))
    )


def hex8_natural_gradients(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Return shape-function gradients with respect to natural coordinates."""

    gradients = np.empty((8, 3), dtype=float)
    for idx, (r, s, t) in enumerate(C3D8_NATURAL_NODE_COORDS):
        gradients[idx, 0] = 0.125 * r * (1.0 + s * eta) * (1.0 + t * zeta)
        gradients[idx, 1] = 0.125 * s * (1.0 + r * xi) * (1.0 + t * zeta)
        gradients[idx, 2] = 0.125 * t * (1.0 + r * xi) * (1.0 + s * eta)
    return gradients


def hex8_strain_displacement_matrix(gradients: np.ndarray) -> np.ndarray:
    """Return the engineering-strain displacement matrix for HEX8."""

    grads = np.asarray(gradients, dtype=float)
    if grads.shape != (8, 3):
        raise ValueError("gradients must have shape (8, 3)")

    B = np.zeros((6, 24), dtype=float)
    for local_node, (dndx, dndy, dndz) in enumerate(grads):
        col = 3 * local_node
        B[0, col] = dndx
        B[1, col + 1] = dndy
        B[2, col + 2] = dndz
        B[3, col] = dndy
        B[3, col + 1] = dndx
        B[4, col + 1] = dndz
        B[4, col + 2] = dndy
        B[5, col] = dndz
        B[5, col + 2] = dndx
    return B


def _physical_gradients(Xe: np.ndarray, xi: float, eta: float, zeta: float) -> tuple[np.ndarray, float]:
    X = _as_hex8_coordinates(Xe)
    natural_gradients = hex8_natural_gradients(xi, eta, zeta)
    jacobian = X.T @ natural_gradients
    det_j = float(np.linalg.det(jacobian))
    if det_j <= 0.0:
        raise ValueError("hex8 element has non-positive Jacobian determinant")
    return natural_gradients @ np.linalg.inv(jacobian), det_j


def hex8_volume(Xe: np.ndarray) -> float:
    """Return the HEX8 volume from 2x2x2 Gauss integration."""

    volume = 0.0
    for xi in _GAUSS_POINTS:
        for eta in _GAUSS_POINTS:
            for zeta in _GAUSS_POINTS:
                _gradients, det_j = _physical_gradients(Xe, xi, eta, zeta)
                volume += det_j
    return float(volume)


def hex8_stiffness(Xe: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Return the linear elastic HEX8 element stiffness matrix."""

    C = isotropic_linear_elasticity_matrix(E, nu)
    Ke = np.zeros((24, 24), dtype=float)
    for xi in _GAUSS_POINTS:
        for eta in _GAUSS_POINTS:
            for zeta in _GAUSS_POINTS:
                gradients, det_j = _physical_gradients(Xe, xi, eta, zeta)
                B = hex8_strain_displacement_matrix(gradients)
                Ke += B.T @ C @ B * det_j
    return Ke


def hex8_consistent_mass(Xe: np.ndarray, rho: float) -> np.ndarray:
    """Return the consistent translational HEX8 mass matrix."""

    density = float(rho)
    if density <= 0.0:
        raise ValueError("rho must be positive")

    M = np.zeros((24, 24), dtype=float)
    for xi in _GAUSS_POINTS:
        for eta in _GAUSS_POINTS:
            for zeta in _GAUSS_POINTS:
                shape = hex8_shape_functions(xi, eta, zeta)
                _gradients, det_j = _physical_gradients(Xe, xi, eta, zeta)
                scalar_mass = density * det_j * np.outer(shape, shape)
                M += np.kron(scalar_mass, np.eye(3))
    return M


def hex8_lumped_mass(Xe: np.ndarray, rho: float) -> np.ndarray:
    """Return a diagonal row-sum HEX8 mass matrix."""

    density = float(rho)
    if density <= 0.0:
        raise ValueError("rho must be positive")

    nodal_mass = density * hex8_volume(Xe) / 8.0
    return np.eye(24, dtype=float) * nodal_mass


def hex8_mass(Xe: np.ndarray, rho: float, kind: str = "consistent") -> np.ndarray:
    """Return a HEX8 mass matrix.

    ``kind`` may be ``"consistent"`` or ``"lumped"``.
    """

    normalized = str(kind).lower()
    if normalized == "consistent":
        return hex8_consistent_mass(Xe, rho)
    if normalized == "lumped":
        return hex8_lumped_mass(Xe, rho)
    raise ValueError("kind must be 'consistent' or 'lumped'")


def hex8_center_strain_stress(
    Xe: np.ndarray,
    Ue: np.ndarray,
    E: float,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover one-point engineering strain and stress at the element center."""

    X = _as_hex8_coordinates(Xe)
    U = np.asarray(Ue, dtype=float)
    if U.shape != (8, 3):
        raise ValueError("Ue must have shape (8, 3)")

    jacobian = X.T @ HEX8_CENTER_NATURAL_GRADIENTS
    det_j = float(np.linalg.det(jacobian))
    if det_j <= 0.0:
        raise ValueError("hex8 element has non-positive Jacobian determinant")
    gradients = HEX8_CENTER_NATURAL_GRADIENTS @ np.linalg.inv(jacobian)
    displacement_gradient = U.T @ gradients
    strain = np.asarray(
        [
            displacement_gradient[0, 0],
            displacement_gradient[1, 1],
            displacement_gradient[2, 2],
            displacement_gradient[0, 1] + displacement_gradient[1, 0],
            displacement_gradient[1, 2] + displacement_gradient[2, 1],
            displacement_gradient[0, 2] + displacement_gradient[2, 0],
        ],
        dtype=float,
    )
    return strain, isotropic_linear_elasticity_matrix(E, nu) @ strain
