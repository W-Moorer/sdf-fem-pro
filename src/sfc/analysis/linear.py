"""Linear static and Newmark dynamic analysis backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import spsolve

from sfc.fem import DeformableBody
from sfc.fem.assembler import assemble_gravity_force, assemble_mass_matrix, assemble_stiffness_matrix
from sfc.fem.constraints import eliminate_fixed_dofs, expand_reduced_vector
from sfc.fem.integrator import newmark_beta_step

from .base import DynamicStepResult, StaticSolveResult, register_analysis_backend


def _as_vector(value: np.ndarray | Sequence[float] | None, n_dofs: int, name: str) -> np.ndarray:
    if value is None:
        return np.zeros(n_dofs, dtype=float)
    vec = np.asarray(value, dtype=float).ravel()
    if vec.shape != (n_dofs,):
        raise ValueError(f"{name} must have shape ({n_dofs},)")
    return vec


def _as_damping_matrix(value, n_dofs: int) -> csr_matrix:
    if value is None:
        return csr_matrix((n_dofs, n_dofs), dtype=float)
    matrix = value.tocsr() if issparse(value) else csr_matrix(np.asarray(value, dtype=float))
    if matrix.shape != (n_dofs, n_dofs):
        raise ValueError(f"damping must have shape ({n_dofs}, {n_dofs})")
    return matrix


def combined_external_force(
    body: DeformableBody,
    *,
    force: np.ndarray | Sequence[float] | None = None,
    gravity: np.ndarray | Sequence[float] | None = None,
    contact_force: np.ndarray | Sequence[float] | None = None,
) -> np.ndarray:
    """Combine mechanical external force, optional gravity, and contact force."""

    total = _as_vector(force, body.n_dofs, "force")
    if gravity is not None:
        total = total + assemble_gravity_force(body, gravity)
    if contact_force is not None:
        total = total + _as_vector(contact_force, body.n_dofs, "contact_force")
    return total


@dataclass(frozen=True, slots=True)
class LinearStaticAnalysis:
    """Linear static analysis backend solving ``K u = f``."""

    name: str = "linear_static"

    def solve(
        self,
        body: DeformableBody,
        *,
        force: np.ndarray | Sequence[float] | None = None,
        gravity: np.ndarray | Sequence[float] | None = None,
        contact_force: np.ndarray | Sequence[float] | None = None,
        fixed_dofs: np.ndarray | Sequence[int] | None = None,
        fixed_values: float | np.ndarray | Sequence[float] | None = None,
    ) -> StaticSolveResult:
        """Solve one linear static equilibrium problem."""

        K = assemble_stiffness_matrix(body)
        rhs = combined_external_force(body, force=force, gravity=gravity, contact_force=contact_force)
        K_reduced, rhs_reduced, free = eliminate_fixed_dofs(K, rhs, fixed_dofs, fixed_values)
        u_free = spsolve(K_reduced, rhs_reduced) if free.size else np.empty(0, dtype=float)
        u = expand_reduced_vector(np.asarray(u_free, dtype=float), free, body.n_dofs, fixed_dofs, fixed_values)
        reactions = np.asarray(K @ u - rhs, dtype=float).ravel()
        return StaticSolveResult(u=u, reactions=reactions, stiffness=K, force=rhs, free_dofs=free)


@dataclass(frozen=True, slots=True)
class LinearNewmarkAnalysis:
    """Linear dynamic analysis backend using Newmark-beta integration."""

    beta: float = 0.25
    gamma: float = 0.5
    mass_kind: str = "consistent"
    name: str = "linear_newmark"

    def step(
        self,
        body: DeformableBody,
        *,
        dt: float,
        force: np.ndarray | Sequence[float] | None = None,
        gravity: np.ndarray | Sequence[float] | None = None,
        contact_force: np.ndarray | Sequence[float] | None = None,
        damping=None,
        fixed_dofs: np.ndarray | Sequence[int] | None = None,
        fixed_values: float | np.ndarray | Sequence[float] | None = None,
        update_body: bool = False,
    ) -> DynamicStepResult:
        """Advance one linear dynamic step without changing physics formulas."""

        K = assemble_stiffness_matrix(body)
        M = assemble_mass_matrix(body, kind=self.mass_kind)
        C = _as_damping_matrix(damping, body.n_dofs)
        rhs = combined_external_force(body, force=force, gravity=gravity, contact_force=contact_force)
        u, v, a = newmark_beta_step(
            M,
            C,
            K,
            body.u,
            body.v,
            body.a,
            rhs,
            dt=dt,
            fixed_dofs=fixed_dofs,
            fixed_values=fixed_values,
            beta=self.beta,
            gamma=self.gamma,
        )
        if update_body:
            body.u = u.copy()
            body.v = v.copy()
            body.a = a.copy()
        return DynamicStepResult(u=u, v=v, a=a, stiffness=K, mass=M, damping=C, force=rhs)


LINEAR_STATIC_BACKEND = LinearStaticAnalysis()
LINEAR_NEWMARK_BACKEND = LinearNewmarkAnalysis()

register_analysis_backend(LINEAR_STATIC_BACKEND)
register_analysis_backend(LINEAR_NEWMARK_BACKEND)
