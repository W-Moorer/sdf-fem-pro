"""Nonlinear static and HHT analysis backends."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import spsolve

from sfc.fem.calculix_aligned import (
    ContactGeometry,
    ContactSample,
    MechanicsModel,
    MechanicsState,
    StaticForceState,
    StepDiagnostics,
    evaluate_state,
    hht_step,
    initial_state,
    static_force_state,
)
from sfc.fem.constraints import eliminate_fixed_dofs, expand_reduced_vector, free_dofs, project_fixed_dofs

from .base import register_analysis_backend


@dataclass(frozen=True, slots=True)
class NoContactGeometry:
    """Contact provider that yields no contact samples."""

    def samples(self, x_current: np.ndarray) -> Iterable[ContactSample]:
        _ = x_current
        return ()


@dataclass(frozen=True, slots=True)
class NonlinearStaticSolveResult:
    """Result of a nonlinear static Newton solve."""

    x: np.ndarray
    u: np.ndarray
    residual: np.ndarray
    force_state: StaticForceState
    diagnostics: StepDiagnostics
    iterations: int
    residual_norm: float
    correction_norm: float
    converged: bool
    free_dofs: np.ndarray


@dataclass(frozen=True, slots=True)
class NonlinearDynamicStepResult:
    """Result of one nonlinear HHT/Newmark step."""

    state: MechanicsState
    previous_static_residual: np.ndarray
    diagnostics: StepDiagnostics


def _as_fixed_values(
    fixed_dofs: np.ndarray | Sequence[int] | None,
    fixed_values: float | np.ndarray | Sequence[float] | None,
) -> tuple[np.ndarray | None, float | np.ndarray | Sequence[float] | None]:
    if fixed_dofs is None:
        return None, fixed_values
    fixed = np.asarray(fixed_dofs, dtype=np.int64).ravel()
    if fixed.size == 0:
        return fixed, fixed_values
    return fixed, fixed_values


@dataclass(frozen=True, slots=True)
class NonlinearStaticNewtonAnalysis:
    """Total-Lagrangian nonlinear static Newton backend."""

    max_iterations: int = 20
    tolerance: float = 1.0e-10
    name: str = "nonlinear_static_newton"

    def solve(
        self,
        model: MechanicsModel,
        contact_geometry: ContactGeometry | None = None,
        *,
        gravity: float = 0.0,
        x0: np.ndarray | None = None,
        fixed_dofs: np.ndarray | Sequence[int] | None = None,
        fixed_values: float | np.ndarray | Sequence[float] | None = None,
    ) -> NonlinearStaticSolveResult:
        """Solve ``f_int(x) - f_ext - f_contact = 0`` by Newton iteration.

        ``fixed_values`` are prescribed displacement values, not absolute
        coordinates. Newton corrections on fixed DOFs are always zero.
        """

        geometry = NoContactGeometry() if contact_geometry is None else contact_geometry
        x = model.X.copy() if x0 is None else np.asarray(x0, dtype=float).copy()
        if x.shape != model.X.shape:
            raise ValueError("x0 must have the same shape as model.X")

        fixed, prescribed = _as_fixed_values(fixed_dofs, fixed_values)
        displacement = project_fixed_dofs((x - model.X).reshape(-1), fixed, prescribed)
        x = model.X + displacement.reshape((-1, 3))

        residual_norm = np.inf
        correction_norm = np.inf
        converged = False
        iterations = 0
        force_state = static_force_state(model, x, geometry, gravity=gravity)
        free = free_dofs(model.n_dofs, fixed)

        for iteration in range(max(1, int(self.max_iterations))):
            force_state = static_force_state(model, x, geometry, gravity=gravity)
            residual_norm = float(np.linalg.norm(force_state.residual[free]))
            if residual_norm <= float(self.tolerance):
                converged = True
                correction_norm = 0.0
                iterations = iteration
                break

            reduced_tangent, reduced_rhs, free = eliminate_fixed_dofs(
                force_state.tangent,
                -force_state.residual,
                fixed,
                0.0,
            )
            if free.size == 0:
                correction = np.zeros(model.n_dofs, dtype=float)
            else:
                reduced_correction = np.asarray(spsolve(reduced_tangent, reduced_rhs), dtype=float)
                correction = expand_reduced_vector(reduced_correction, free, model.n_dofs, fixed, 0.0)
            correction_norm = float(np.linalg.norm(correction))
            displacement = project_fixed_dofs((x - model.X).reshape(-1) + correction, fixed, prescribed)
            x = model.X + displacement.reshape((-1, 3))
            iterations = iteration + 1

            if correction_norm <= float(self.tolerance) * max(1.0, float(np.linalg.norm(displacement))):
                force_state = static_force_state(model, x, geometry, gravity=gravity)
                residual_norm = float(np.linalg.norm(force_state.residual[free]))
                converged = True
                break

        state = MechanicsState(x=x, v=np.zeros_like(x), a=np.zeros_like(x), time=0.0)
        diagnostics = evaluate_state(model, state, geometry, gravity=gravity, assemble_tangent=True)
        diagnostics.newton_iterations = iterations
        diagnostics.newton_residual_norm = residual_norm
        diagnostics.newton_acceptance_policy = "static_newton"
        diagnostics.newton_acceptance_reason = "converged" if converged else "iteration_limit"
        return NonlinearStaticSolveResult(
            x=x,
            u=(x - model.X).reshape(-1),
            residual=force_state.residual,
            force_state=force_state,
            diagnostics=diagnostics,
            iterations=iterations,
            residual_norm=residual_norm,
            correction_norm=correction_norm,
            converged=converged,
            free_dofs=free,
        )


@dataclass(frozen=True, slots=True)
class NonlinearHHTAnalysis:
    """Nonlinear implicit HHT/Newmark dynamic backend."""

    alpha: float = -0.05
    max_iterations: int = 12
    tolerance: float = 1.0e-10
    acceptance_policy: str = "relative_correction"
    name: str = "nonlinear_hht"

    def initial_state(
        self,
        model: MechanicsModel,
        contact_geometry: ContactGeometry | None = None,
        *,
        gravity: float = 0.0,
        initial_velocity: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
        dt: float | None = None,
    ) -> tuple[MechanicsState, np.ndarray]:
        """Create the initial state and accepted static-history vector."""

        geometry = NoContactGeometry() if contact_geometry is None else contact_geometry
        return initial_state(
            model,
            geometry,
            gravity=float(gravity),
            initial_velocity=initial_velocity,
            dt=dt,
            alpha=float(self.alpha),
        )

    def step(
        self,
        model: MechanicsModel,
        state: MechanicsState,
        previous_static_residual: np.ndarray,
        contact_geometry: ContactGeometry | None = None,
        *,
        dt: float,
        gravity: float = 0.0,
    ) -> NonlinearDynamicStepResult:
        """Advance one nonlinear implicit HHT/Newmark step."""

        geometry = NoContactGeometry() if contact_geometry is None else contact_geometry
        next_state, next_previous, diagnostics = hht_step(
            model,
            state,
            previous_static_residual,
            geometry,
            dt=float(dt),
            gravity=float(gravity),
            alpha=float(self.alpha),
            max_iterations=int(self.max_iterations),
            tolerance=float(self.tolerance),
            acceptance_policy=self.acceptance_policy,
        )
        return NonlinearDynamicStepResult(
            state=next_state,
            previous_static_residual=next_previous,
            diagnostics=diagnostics,
        )


NONLINEAR_STATIC_BACKEND = NonlinearStaticNewtonAnalysis()
NONLINEAR_HHT_BACKEND = NonlinearHHTAnalysis()

register_analysis_backend(NONLINEAR_STATIC_BACKEND)
register_analysis_backend(NONLINEAR_HHT_BACKEND)
