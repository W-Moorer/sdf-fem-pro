"""Abaqus-style nonlinear increment acceptance and cutback utilities.

These helpers are intentionally independent from a specific benchmark runner.
They encode the common acceptance gates used by the current implicit contact
paths: residual convergence, displacement correction convergence, contact-force
increment convergence, and active-set stability.  A failed trial at the
iteration limit requests a cutback instead of being accepted as a solved state.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class IncrementConvergenceDecision:
    """One nonlinear increment acceptance decision."""

    converged: bool
    reason: str
    iteration_limited: bool
    cutback_required: bool
    accepted: bool


@dataclass(frozen=True, slots=True)
class AutomaticIncrementEvent:
    """One automatic-increment trial record."""

    trial_index: int
    start_time: float
    end_time: float
    dt: float
    accepted: bool
    retry_required: bool
    cutback_required: bool
    cutback_candidate_dt: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class AutomaticIncrementResult:
    """Automatic-increment controller diagnostics."""

    events: tuple[AutomaticIncrementEvent, ...]
    accepted_times: tuple[float, ...]
    cutback_count: int
    final_time: float


def contact_active_set_is_stable(
    active_signature: tuple[tuple[int, ...], ...],
    previous_active_signature: tuple[tuple[int, ...], ...] | None,
    *,
    require_stability: bool,
) -> bool:
    """Return true when contact status is stable enough for acceptance."""

    if not bool(require_stability):
        return True
    return previous_active_signature is not None and active_signature == previous_active_signature


def active_set_line_search_choice(
    current_signature: tuple[tuple[int, ...], ...] | None,
    candidates: Iterable[tuple[float, tuple[tuple[int, ...], ...]]],
    *,
    require_stability: bool,
) -> tuple[float, bool, int]:
    """Choose a correction fraction that preserves the current active set."""

    first_alpha: float | None = None
    trial_count = 0
    for alpha, signature in candidates:
        alpha_value = float(alpha)
        if first_alpha is None:
            first_alpha = alpha_value
        trial_count += 1
        if (not bool(require_stability)) or current_signature is None or signature == current_signature:
            return alpha_value, True, int(trial_count)
    return (1.0 if first_alpha is None else float(first_alpha)), False, int(trial_count)


def active_set_stability_after_line_search(
    active_set_stable: bool,
    *,
    line_search_attempted: bool,
    line_search_stable: bool,
) -> bool:
    """Combine Newton active-set stability with the line-search result."""

    return bool(active_set_stable) and (not bool(line_search_attempted) or bool(line_search_stable))


def increment_convergence_decision(
    *,
    residual_converged: bool,
    correction_converged: bool,
    contact_force_increment_converged: bool,
    active_set_stable: bool,
    iteration_count: int,
    max_iterations: int,
    accept_unconverged: bool,
) -> IncrementConvergenceDecision:
    """Evaluate residual/correction/contact/status gates for one increment."""

    converged = (
        bool(residual_converged)
        and bool(correction_converged)
        and bool(contact_force_increment_converged)
        and bool(active_set_stable)
    )
    if converged:
        return IncrementConvergenceDecision(
            converged=True,
            reason="residual_correction_contact_force_active_set",
            iteration_limited=False,
            cutback_required=False,
            accepted=True,
        )
    if not bool(residual_converged):
        reason = "residual"
    elif not bool(correction_converged):
        reason = "correction"
    elif not bool(contact_force_increment_converged):
        reason = "contact_force_increment"
    else:
        reason = "contact_active_set"
    limited = int(iteration_count) >= max(1, int(max_iterations))
    return IncrementConvergenceDecision(
        converged=False,
        reason=reason,
        iteration_limited=limited,
        cutback_required=bool(limited and not bool(accept_unconverged)),
        accepted=bool(accept_unconverged),
    )


def increment_cutback_candidate_dt(
    current_dt: float,
    *,
    min_dt: float,
    cutback_factor: float,
) -> float | None:
    """Return the next trial increment or ``None`` when already at the floor."""

    h = float(current_dt)
    h_min = float(min_dt)
    factor = float(cutback_factor)
    if h <= 0.0:
        raise ValueError("current_dt must be positive")
    if h_min <= 0.0:
        raise ValueError("min_dt must be positive")
    if not (0.0 < factor < 1.0):
        raise ValueError("cutback_factor must lie in (0, 1)")
    if h <= h_min * (1.0 + 1.0e-12):
        return None
    return max(h_min, h * factor)


def run_automatic_increment_controller(
    *,
    duration: float,
    initial_dt: float,
    min_dt: float,
    cutback_factor: float,
    trial: Callable[[float, float, int], IncrementConvergenceDecision],
) -> AutomaticIncrementResult:
    """Run an accept/cutback scheduler for nonlinear trial increments."""

    total = float(duration)
    h0 = float(initial_dt)
    h_min = float(min_dt)
    if total < 0.0:
        raise ValueError("duration must be non-negative")
    if h0 <= 0.0:
        raise ValueError("initial_dt must be positive")
    if h_min <= 0.0:
        raise ValueError("min_dt must be positive")
    if not (0.0 < float(cutback_factor) < 1.0):
        raise ValueError("cutback_factor must lie in (0, 1)")
    accepted_time = 0.0
    trial_dt = min(h0, max(total, h_min)) if total > 0.0 else h0
    trial_index = 0
    cutback_count = 0
    events: list[AutomaticIncrementEvent] = []
    accepted_times: list[float] = []
    while accepted_time < total - 1.0e-15:
        h = min(float(trial_dt), total - accepted_time)
        trial_index += 1
        decision = trial(float(accepted_time), float(h), int(trial_index))
        candidate = (
            increment_cutback_candidate_dt(
                h,
                min_dt=h_min,
                cutback_factor=float(cutback_factor),
            )
            if decision.cutback_required
            else None
        )
        retry_required = bool(decision.cutback_required and candidate is not None)
        accepted = bool(decision.accepted and not decision.cutback_required)
        events.append(
            AutomaticIncrementEvent(
                trial_index=int(trial_index),
                start_time=float(accepted_time),
                end_time=float(accepted_time + h),
                dt=float(h),
                accepted=accepted,
                retry_required=retry_required,
                cutback_required=bool(decision.cutback_required),
                cutback_candidate_dt=candidate,
                reason=str(decision.reason),
            )
        )
        if decision.cutback_required:
            if candidate is None:
                raise RuntimeError("automatic increment reached minimum dt without satisfying convergence gates")
            trial_dt = float(candidate)
            cutback_count += 1
            continue
        if not accepted:
            raise RuntimeError("automatic increment trial was neither accepted nor eligible for cutback")
        accepted_time += h
        accepted_times.append(float(accepted_time))
        trial_dt = min(h0, max(h_min, h * (1.25 if decision.converged else 1.0)))
    return AutomaticIncrementResult(
        events=tuple(events),
        accepted_times=tuple(accepted_times),
        cutback_count=int(cutback_count),
        final_time=float(accepted_time),
    )


def increment_gate_row(
    *,
    residual_converged: bool,
    correction_converged: bool,
    contact_force_increment_converged: bool,
    active_set_stable: bool,
    normalized_residual: float,
    normalized_correction: float,
    normalized_contact_force_increment: float,
    contact_force_increment_norm: float,
    decision: IncrementConvergenceDecision,
    cutback_candidate_dt: float | None,
    prefix: str = "source",
) -> dict[str, Any]:
    """Return one history/manifest row for increment acceptance gates."""

    label = str(prefix).strip("_")
    return {
        f"{label}_residual_converged": int(bool(residual_converged)),
        f"{label}_correction_converged": int(bool(correction_converged)),
        f"{label}_contact_force_increment_converged": int(bool(contact_force_increment_converged)),
        f"{label}_active_set_stable": int(bool(active_set_stable)),
        f"{label}_increment_converged": int(bool(decision.converged)),
        f"{label}_increment_accepted": int(bool(decision.accepted)),
        f"{label}_increment_cutback_required": int(bool(decision.cutback_required)),
        f"{label}_increment_cutback_candidate_dt": (
            "" if cutback_candidate_dt is None else float(cutback_candidate_dt)
        ),
        f"{label}_iteration_limit_reached": int(bool(decision.iteration_limited)),
        f"{label}_step_convergence_reason": str(decision.reason),
        f"{label}_normalized_residual": float(normalized_residual),
        f"{label}_normalized_correction": float(normalized_correction),
        f"{label}_normalized_contact_force_increment": float(normalized_contact_force_increment),
        f"{label}_contact_force_increment_norm": float(contact_force_increment_norm),
    }
