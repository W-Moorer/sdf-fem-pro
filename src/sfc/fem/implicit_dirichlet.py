"""Implicit HHT/Newmark stepping with eliminated Dirichlet constraints."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve

from sfc.fem.calculix_aligned import (
    ContactGeometry,
    MechanicsModel,
    MechanicsState,
    StepDiagnostics,
    _begin_contact_increment,
    _begin_contact_newton_iteration,
    _newton_acceptance_metrics,
    _nodal_gravity_loads,
    _restore_contact_state,
    _snapshot_contact_state,
    _static_force_state_from_responses,
    calculix_dynamic_predictor,
    calculix_hht_effective_residual,
    calculix_hht_effective_tangent,
    evaluate_state,
    hht_newmark_parameters,
    static_force_state,
)
from sfc.fem.constraints import free_dofs, project_fixed_dofs


def hht_step_dirichlet(
    model: MechanicsModel,
    state: MechanicsState,
    previous_static_residual: np.ndarray,
    contact_geometry: ContactGeometry,
    *,
    dt: float,
    gravity: float,
    fixed_dofs: np.ndarray,
    fixed_values: np.ndarray,
    alpha: float = -0.05,
    max_iterations: int = 12,
    tolerance: float = 1.0e-10,
    acceptance_policy: str = "relative_correction",
) -> tuple[MechanicsState, np.ndarray, StepDiagnostics]:
    """Advance one HHT step while enforcing prescribed displacement DOFs.

    ``fixed_values`` are total displacements relative to ``model.X`` at the end
    of the step.  They are eliminated from the Newton solve, while all internal
    and contact forces are still evaluated on the full constrained state.
    """

    if acceptance_policy not in {"relative_correction", "calculix_multicriteria"}:
        raise ValueError("acceptance_policy must be 'relative_correction' or 'calculix_multicriteria'")
    fixed = np.asarray(fixed_dofs, dtype=np.int64).ravel()
    values = np.asarray(fixed_values, dtype=float).ravel()
    if fixed.shape != values.shape:
        raise ValueError("fixed_dofs and fixed_values must have the same shape")
    if fixed.size and (np.any(fixed < 0) or int(fixed.max()) >= model.n_dofs):
        raise ValueError("fixed_dofs contain indices outside the model")
    fixed_order = np.argsort(fixed, kind="stable")
    fixed = fixed[fixed_order]
    values = values[fixed_order]
    free = free_dofs(model.n_dofs, fixed)

    beta, gamma = hht_newmark_parameters(alpha)
    c0 = 1.0 / (beta * dt * dt)
    u = (state.x - model.X).reshape(-1)
    v = state.v.reshape(-1)
    a = state.a.reshape(-1)
    u_pred, v_pred, _accold_after_prediction = calculix_dynamic_predictor(u, v, a, dt=dt, beta=beta, gamma=gamma)
    u_guess = project_fixed_dofs(u_pred, fixed, values)
    previous_rhs_balance = np.asarray(previous_static_residual, dtype=float).ravel()
    if previous_rhs_balance.shape != (model.n_dofs,):
        raise ValueError("previous_static_residual length must match model.n_dofs")

    residual_norm = np.inf
    iteration_count = 0
    acceptance_reason = "iteration_limit"
    acceptance_metrics = []
    _begin_contact_increment(contact_geometry, state.x)
    previous_contact_snapshot = _snapshot_contact_state(contact_geometry)
    previous_energy = evaluate_state(model, state, contact_geometry, gravity=gravity, assemble_tangent=False).total_energy
    _restore_contact_state(contact_geometry, previous_contact_snapshot)
    previous_ram: float | None = None
    previous_active_count: int | None = None

    for iteration in range(max(1, int(max_iterations))):
        _begin_contact_newton_iteration(contact_geometry, iteration + 1)
        u_guess = project_fixed_dofs(u_guess, fixed, values)
        x_guess = model.X + u_guess.reshape((-1, 3))
        static_state = static_force_state(model, x_guess, contact_geometry, gravity=gravity)
        a_guess = c0 * (u_guess - u_pred)
        dynamic_residual = calculix_hht_effective_residual(
            model.mass_matrix @ a_guess,
            static_state.calculix_rhs_balance,
            previous_rhs_balance,
            alpha=alpha,
        )
        residual_norm = float(np.linalg.norm(dynamic_residual[free]))
        tangent = calculix_hht_effective_tangent(model.mass_matrix, static_state.tangent, dt=dt, beta=beta, alpha=alpha).tocsc()
        if free.size:
            correction_free = np.asarray(spsolve(tangent[free[:, None], free], -dynamic_residual[free]), dtype=float)
        else:
            correction_free = np.empty(0, dtype=float)
        correction = np.zeros(model.n_dofs, dtype=float)
        correction[free] = correction_free
        u_guess[free] += correction_free
        u_guess = project_fixed_dofs(u_guess, fixed, values)
        iteration_count = iteration + 1

        a_iter = c0 * (u_guess - u_pred)
        v_iter = v_pred + gamma * dt * a_iter
        trial_state = MechanicsState(
            model.X + u_guess.reshape((-1, 3)),
            v_iter.reshape((-1, 3)),
            a_iter.reshape((-1, 3)),
            time=state.time + float(dt),
        )
        trial_diagnostics = evaluate_state(model, trial_state, contact_geometry, gravity=gravity, assemble_tangent=False)
        metric = _newton_acceptance_metrics(
            model,
            dynamic_residual,
            correction,
            u_guess - u,
            trial_diagnostics,
            previous_energy=previous_energy,
            previous_ram=previous_ram,
            previous_active_count=previous_active_count,
            iteration=iteration_count,
            gravity=gravity,
        )
        acceptance_metrics.append(metric)
        previous_ram = metric.ram
        previous_active_count = metric.active_contact_count
        relative_correction_accept = np.linalg.norm(correction_free) <= tolerance * max(1.0, float(np.linalg.norm(u_guess[free])))
        if acceptance_policy == "relative_correction" and relative_correction_accept:
            acceptance_reason = "relative_correction"
            break
        if acceptance_policy == "calculix_multicriteria" and metric.accepted_by_calculix_style:
            acceptance_reason = metric.reason
            break

    a_new = c0 * (u_guess - u_pred)
    v_new = v_pred + gamma * dt * a_new
    next_state = MechanicsState(
        model.X + u_guess.reshape((-1, 3)),
        v_new.reshape((-1, 3)),
        a_new.reshape((-1, 3)),
        time=state.time + float(dt),
    )
    diagnostics = evaluate_state(model, next_state, contact_geometry, gravity=gravity, assemble_tangent=True)
    diagnostics.newton_iterations = iteration_count
    diagnostics.newton_residual_norm = residual_norm
    diagnostics.newton_acceptance_policy = acceptance_policy
    diagnostics.newton_acceptance_reason = acceptance_reason
    diagnostics.newton_acceptance_metrics = acceptance_metrics
    accepted_static_state = _static_force_state_from_responses(
        model,
        diagnostics.internal,
        diagnostics.contact,
        gravity=gravity,
    )
    return next_state, accepted_static_state.calculix_rhs_balance, diagnostics


def initial_state_dirichlet(
    model: MechanicsModel,
    contact_geometry: ContactGeometry,
    *,
    gravity: float,
    fixed_dofs: np.ndarray,
    fixed_values: np.ndarray,
    initial_velocity: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[MechanicsState, np.ndarray]:
    """Create a constrained initial state with fixed displacement DOFs."""

    fixed = np.asarray(fixed_dofs, dtype=np.int64).ravel()
    values = np.asarray(fixed_values, dtype=float).ravel()
    if fixed.shape != values.shape:
        raise ValueError("fixed_dofs and fixed_values must have the same shape")
    velocity = np.zeros_like(model.X)
    velocity[:] = np.asarray(initial_velocity, dtype=float)
    u0 = project_fixed_dofs(np.zeros(model.n_dofs, dtype=float), fixed, values)
    state = MechanicsState(model.X + u0.reshape((-1, 3)), velocity, np.zeros_like(model.X), time=0.0)
    static_state = static_force_state(model, state.x, contact_geometry, gravity=gravity)
    free = free_dofs(model.n_dofs, fixed)
    acceleration = np.zeros(model.n_dofs, dtype=float)
    if free.size and float(np.linalg.norm(static_state.calculix_rhs_balance[free])) > 0.0:
        acceleration[free] = np.asarray(
            spsolve(model.mass_matrix[free[:, None], free].tocsc(), static_state.calculix_rhs_balance[free]),
            dtype=float,
        )
    state.a = acceleration.reshape((-1, 3))
    return state, static_state.calculix_rhs_balance
