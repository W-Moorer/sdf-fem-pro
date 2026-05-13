"""Deformable body state for FEM assembly and integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sfc.mesh import VolumeMesh


def _state_vector(value: np.ndarray | None, n_dofs: int, name: str) -> np.ndarray:
    if value is None:
        return np.zeros(n_dofs, dtype=float)

    arr = np.asarray(value, dtype=float)
    if arr.shape != (n_dofs,):
        raise ValueError(f"{name} must have shape ({n_dofs},)")
    return arr.copy()


@dataclass(slots=True)
class DeformableBody:
    """A deformable finite-element body and its current state."""

    mesh: VolumeMesh
    material: Any
    density: float
    u: np.ndarray | None = None
    v: np.ndarray | None = None
    a: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.density = float(self.density)
        if self.density <= 0.0:
            raise ValueError("density must be positive")

        n_dofs = self.mesh.X.shape[0] * 3
        self.u = _state_vector(self.u, n_dofs, "u")
        self.v = _state_vector(self.v, n_dofs, "v")
        self.a = _state_vector(self.a, n_dofs, "a")

    @property
    def n_nodes(self) -> int:
        return int(self.mesh.X.shape[0])

    @property
    def n_dofs(self) -> int:
        return self.n_nodes * 3
