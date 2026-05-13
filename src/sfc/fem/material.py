"""Material models and constitutive matrices."""

from __future__ import annotations

import numpy as np


def isotropic_linear_elasticity_matrix(E: float, nu: float) -> np.ndarray:
    """Return the 3D isotropic linear elastic constitutive matrix.

    The strain vector ordering is
    ``[eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_xz]`` with engineering
    shear strains. The returned matrix maps this strain vector to Cauchy stress.
    """

    E = float(E)
    nu = float(nu)
    if E <= 0.0:
        raise ValueError("Young's modulus E must be positive")
    if not (-1.0 < nu < 0.5):
        raise ValueError("Poisson ratio nu must satisfy -1 < nu < 0.5")

    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    C = np.array(
        [
            [lam + 2.0 * mu, lam, lam, 0.0, 0.0, 0.0],
            [lam, lam + 2.0 * mu, lam, 0.0, 0.0, 0.0],
            [lam, lam, lam + 2.0 * mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, mu, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, mu, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, mu],
        ],
        dtype=float,
    )
    return C
